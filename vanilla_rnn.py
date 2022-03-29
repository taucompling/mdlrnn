import logging
from typing import List, Optional, Text, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import rnn

import corpora
import utils

utils.setup_logging()


def _ensure_torch(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    return x


def _get_masked_sequence_lengths(x) -> List[int]:
    # Ugly and slow but works.
    sequence_lengths = []
    for b in range(x.shape[0]):
        seq = x[b]
        any_masked = False
        for i in range(seq.shape[0]):
            if np.all(corpora.is_masked(seq[i])):
                any_masked = True
                sequence_lengths.append(i)
                break
        if not any_masked:
            sequence_lengths.append(seq.shape[0])
    return sequence_lengths


def _get_packed_sample_weights(
    sample_weights, max_sequence_length, masked_sequence_lengths
):
    if sample_weights is None:
        return None
    w = np.array(sample_weights).reshape((-1, 1))
    weights_repeated = np.matmul(w, torch.ones((1, max_sequence_length)))
    return _pack_masked(weights_repeated, masked_sequence_lengths).data


def _pack_masked(sequences: np.ndarray, sequence_lengths):
    return rnn.pack_padded_sequence(
        _ensure_torch(sequences),
        lengths=sequence_lengths,
        batch_first=True,
        enforce_sorted=True,
    )


def _unpack(
    packed_tensor,
    packed_sequence_batch_sizes,
    batch_size,
    max_sequence_length,
    num_classes,
):
    # Assumes sorted indices.
    unpacked = np.empty((batch_size, max_sequence_length, num_classes))
    unpacked.fill(corpora.MASK_VALUE)
    i = 0
    for t, batch_size in enumerate(packed_sequence_batch_sizes):
        unpacked[:batch_size, t] = packed_tensor[i : i + batch_size]
        i += batch_size
    return unpacked


class VanillaRNN(nn.Module):
    def __init__(
        self,
        num_hidden_units: int,
        corpus: corpora.Corpus,
        network_type: Text,
        num_epochs: int,
        regularization: Optional[Text],
        regularization_lambda: Optional[float],
    ):
        super(VanillaRNN, self).__init__()

        self._num_hidden_units = num_hidden_units
        self._num_epochs = num_epochs
        self._train_corpus = corpus
        self._regularization = regularization
        self._regularization_lambda = regularization_lambda

        self._input_size = self._train_corpus.input_sequence.shape[-1]
        self._output_size = self._train_corpus.target_sequence.shape[-1]

        rnn_kwargs = {
            "input_size": self._input_size,
            "hidden_size": self._num_hidden_units,
            "batch_first": True,
        }
        rnn_type_to_layer = {"elman": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        rnn_layer = rnn_type_to_layer[network_type]
        self._rnn_layer = rnn_layer(**rnn_kwargs)

        layers = [
            nn.Linear(
                in_features=self._num_hidden_units, out_features=self._output_size,
            ),
        ]
        if self._output_size == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.LogSoftmax(dim=-1))
        self._layers = nn.Sequential(*layers)

    def _forward(self, x):
        rnn_packed_outputs, _ = self._rnn_layer(x)
        rnn_outputs = rnn_packed_outputs.data
        return self._layers(rnn_outputs)

    def fit(self):
        train_max_sequence_length = self._train_corpus.input_sequence.shape[1]
        train_sequence_lengths = _get_masked_sequence_lengths(
            self._train_corpus.input_sequence
        )
        train_inputs_packed = _pack_masked(
            self._train_corpus.input_sequence, sequence_lengths=train_sequence_lengths
        )
        train_targets_packed = _pack_masked(
            self._train_corpus.target_sequence, sequence_lengths=train_sequence_lengths
        ).data

        train_sample_weights_packed = _get_packed_sample_weights(
            self._train_corpus.sample_weights,
            max_sequence_length=train_max_sequence_length,
            masked_sequence_lengths=train_sequence_lengths,
        )

        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(self._num_epochs):
            optimizer.zero_grad()
            output = self._forward(train_inputs_packed)
            cross_entropy_loss, _ = _calculate_loss(
                net=self,
                outputs_packed=output,
                targets_packed=train_targets_packed,
                sample_weights=train_sample_weights_packed,
                regularization=self._regularization,
                regularization_lambda=self._regularization_lambda,
            )
            cross_entropy_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logging.info(
                    f"Epoch {epoch} training loss: {cross_entropy_loss.item():.3e}"
                )

    def feed_sequence(self, input_sequence):
        with torch.no_grad():
            return self._forward(input_sequence)


def _calculate_loss(
    net: VanillaRNN,
    outputs_packed,
    targets_packed,
    sample_weights,
    regularization,
    regularization_lambda,
):
    if targets_packed.shape[-1] == 1:
        loss_func = nn.BCELoss
        target_classes = targets_packed
    else:
        # Not using `CrossEntropyLoss` because network outputs are already log-softmaxed.
        loss_func = nn.NLLLoss
        target_classes = targets_packed.argmax(axis=-1)

    non_reduced_loss = loss_func(reduction="none")(outputs_packed, target_classes)

    if sample_weights is not None:
        weighted_losses = torch.mul(non_reduced_loss, sample_weights)
        weighted_losses_sum = weighted_losses.sum()
        total_chars_in_input = sample_weights.sum()
        average_loss = weighted_losses_sum / total_chars_in_input
    else:
        average_loss = non_reduced_loss.mean()
        weighted_losses_sum = non_reduced_loss.sum()

    regularized_loss = 0
    if regularization == "L1":
        for p in net._rnn_layer.parameters():
            regularized_loss += torch.sum(torch.abs(p))
        for p in net._layers.parameters():
            regularized_loss += torch.sum(torch.abs(p))

    elif regularization == "L2":
        for p in net._rnn_layer.parameters():
            regularized_loss += torch.sum(torch.square(p))
        for p in net._layers.parameters():
            regularized_loss += torch.sum(torch.square(p))

    average_loss = average_loss + (regularization_lambda * regularized_loss)

    return average_loss, weighted_losses_sum


def calculate_symbolic_accuracy(
    found_net: VanillaRNN,
    inputs: np.ndarray,
    target_probabs: np.ndarray,
    sample_weights: Tuple[int, ...],
    input_mask: np.ndarray,
    epsilon: float,
    plots: bool = False,
):
    sequence_lengths = _get_masked_sequence_lengths(inputs)

    inputs_packed = _pack_masked(inputs, sequence_lengths)
    predicted_probabs_packed = found_net.feed_sequence(inputs_packed)
    predicted_probabs = _unpack(
        packed_tensor=predicted_probabs_packed,
        packed_sequence_batch_sizes=inputs_packed.batch_sizes,
        batch_size=inputs.shape[0],
        max_sequence_length=inputs.shape[1],
        num_classes=target_probabs.shape[-1],
    )
    predicted_probabs = np.exp(predicted_probabs)

    return utils.calculate_symbolic_accuracy(
        predicted_probabs=predicted_probabs,
        target_probabs=target_probabs,
        input_mask=input_mask,
        plots=plots,
        sample_weights=sample_weights,
        epsilon=epsilon,
    )


def evaluate(
    net,
    inputs,
    targets,
    sample_weights,
    deterministic_steps_mask,
    regularization,
    regularization_lambda,
):
    sequence_lengths = _get_masked_sequence_lengths(inputs)

    inputs_packed = _pack_masked(inputs, sequence_lengths)
    targets_packed = _pack_masked(targets, sequence_lengths).data

    sample_weights_packed = _get_packed_sample_weights(
        sample_weights,
        max_sequence_length=inputs.shape[1],
        masked_sequence_lengths=sequence_lengths,
    )

    y_pred = net.feed_sequence(inputs_packed)

    cross_entropy_loss, cross_entropy_sum = _calculate_loss(
        net,
        y_pred,
        targets_packed,
        sample_weights_packed,
        regularization,
        regularization_lambda,
    )

    if targets.shape[-1] == 1:
        target_classes = targets_packed.flatten()
        predicted_classes = (y_pred > 0.5).flatten().long()
    else:
        target_classes = targets_packed.argmax(dim=-1).flatten()
        predicted_classes = y_pred.argmax(dim=-1).flatten()

    correct = torch.sum(torch.eq(predicted_classes, target_classes)).item()
    accuracy = correct / len(target_classes)

    if deterministic_steps_mask is not None:
        deterministic_mask_packed = _pack_masked(
            deterministic_steps_mask, sequence_lengths
        ).data.bool()
        det_target_classes = target_classes[deterministic_mask_packed]
        det_correct = torch.eq(
            predicted_classes[deterministic_mask_packed], det_target_classes
        )
        det_flat_sample_weights = sample_weights_packed[deterministic_mask_packed]
        det_correct_weighted = torch.mul(det_correct.int(), det_flat_sample_weights)
        det_accuracy = (
            f"{det_correct_weighted.sum() / det_flat_sample_weights.sum().item():.5f}"
        )
    else:
        det_accuracy = None

    logging.info(
        f"Accuracy: {accuracy:.5f} (Correct: {correct} / {len(target_classes)})\n"
        f"Deterministic accuracy: {det_accuracy}\n"
        f"Cross-entropy loss: {cross_entropy_loss:.2f}\n"
        f"Cross-entropy sum: {cross_entropy_sum:.2f}"
    )
    return (
        accuracy,
        cross_entropy_loss.item(),
        cross_entropy_sum.item(),
        det_accuracy,
    )
