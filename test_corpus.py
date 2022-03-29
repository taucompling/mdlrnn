import unittest

import numpy as np

import configuration
import corpora
import manual_nets
import network
import simulations
import utils

utils.setup_logging()


def _test_xor_correctness(input_sequence, target_sequence):
    memory = {"step_t_minus_1": 0, "step_t_minus_2": 0}

    def predictor(x):
        memory["step_t_minus_2"] = memory["step_t_minus_1"]
        memory["step_t_minus_1"] = x
        return memory["step_t_minus_1"] ^ memory["step_t_minus_2"]

    correct = 0
    for in_, target in zip(input_sequence[0], target_sequence[0]):
        prediction = predictor(in_)
        if np.all(prediction == target):
            correct += 1

    accuracy = correct / input_sequence.shape[1]
    print(f"Accuracy: {accuracy:.2f}")

    # A perfect network predicts 33% of the values deterministically, and another 66% x 0.5 by chance.
    assert accuracy >= 0.66


class TestCorpus(unittest.TestCase):
    def test_train_test_shuffle(self):
        train_ratio = 0.7
        batch_size = 1000
        train_corpus = corpora.make_binary_palindrome_fixed_length(
            batch_size=batch_size, sequence_length=10, train_set_ratio=train_ratio
        )
        test_corpus = train_corpus.test_corpus
        assert (
            sum(train_corpus.sample_weights) + sum(test_corpus.sample_weights)
            == batch_size
        )

        assert train_corpus.input_sequence.shape[0] == 22
        assert test_corpus.input_sequence.shape[0] == 10

    def test_mini_batches(self):
        mini_batch_size = 100

        net = manual_nets.make_emmanuel_triplet_xor_network()
        full_corpus = corpora.optimize_for_feeding(
            corpora.make_elman_xor_binary(sequence_length=300, batch_size=1000)
        )

        mini_batches = corpora.split_to_mini_batches(
            full_corpus, mini_batch_size=mini_batch_size
        )
        assert len(mini_batches) == 10

        config = configuration.SimulationConfig(
            **{**simulations.DEFAULT_CONFIG, "mini_batch_size": mini_batch_size},
            simulation_id="",
            num_islands=1,
            seed=1,
        )
        net = network.calculate_fitness(net, full_corpus, config)
        print(net)
        assert net.fitness.data_encoding_length == 200000

    def test_0_1_pattern_binary(self):
        seq_length = 10
        batch_size = 10
        corpus = corpora.make_0_1_pattern_binary(
            sequence_length=seq_length, batch_size=batch_size
        )
        assert corpus.sample_weights == (batch_size,)
        assert corpus.test_corpus.input_sequence.shape == (1, seq_length * 50_000, 1)

        seq_length = 10
        batch_size = 1
        corpus = corpora.make_0_1_pattern_binary(
            sequence_length=seq_length, batch_size=batch_size
        )
        assert corpus.sample_weights is None

    def test_xor_corpus_correctness(self):
        sequence_length = 9999
        xor_corpus_binary = corpora.make_elman_xor_binary(sequence_length, batch_size=1)
        xor_corpus_one_hot = corpora.make_elman_xor_one_hot(
            sequence_length, batch_size=1
        )
        _test_xor_correctness(
            xor_corpus_binary.input_sequence.astype(np.int),
            xor_corpus_binary.target_sequence.astype(np.int),
        )
        _test_xor_correctness(
            xor_corpus_one_hot.input_sequence.argmax(axis=-1),
            xor_corpus_one_hot.target_sequence.argmax(axis=-1),
        )

    def test_binary_addition_corpus_correctness(self):
        addition_corpus = corpora.make_binary_addition(min_n=0, max_n=100)

        for input_sequence, target_sequence in zip(
            addition_corpus.input_sequence, addition_corpus.target_sequence
        ):
            n1 = 0
            n2 = 0
            sum_ = 0
            for i, bin_digit in enumerate(input_sequence):
                n1_binary_digit = bin_digit[0]
                n2_binary_digit = bin_digit[1]
                current_exp = 2 ** i
                if n1_binary_digit == 1:
                    n1 += current_exp
                if n2_binary_digit == 1:
                    n2 += current_exp
                target_binary_digit = target_sequence[i]
                if target_binary_digit == 1:
                    sum_ += current_exp

            assert n1 + n2 == sum_, (n1, n2, sum_)

    def test_an_bn_corpus(self):
        n_values = tuple(range(50))
        an_bn_corpus = corpora.optimize_for_feeding(
            corpora._make_ain_bjn_ckn_dtn_corpus(
                n_values, multipliers=(1, 1, 0, 0), prior=0.1, sort_by_length=True
            )
        )

        for n in n_values:
            row = 49 - n  # Sequences are sorted by decreasing length.
            input_seq = an_bn_corpus.input_sequence[row]
            target_seq = an_bn_corpus.target_sequence[row]
            seq_len = 1 + (2 * n)

            zeros_start = 1
            ones_start = n + 1

            assert not np.all(corpora.is_masked(input_seq[:seq_len]))
            assert np.all(corpora.is_masked(input_seq[seq_len:]))

            input_classes = np.argmax(input_seq, axis=-1)[:seq_len]
            target_classes = np.argmax(target_seq, axis=-1)[:seq_len]

            assert np.sum(input_classes == 1) == n
            assert np.sum(input_classes == 2) == n

            assert input_classes[0] == 0  # Start of sequence.
            assert np.all(input_classes[zeros_start:ones_start] == 1)
            assert np.all(input_classes[ones_start:seq_len] == 2)

            assert target_classes[seq_len - 1] == 0  # End of sequence.
            assert np.all(target_classes[zeros_start - 1 : ones_start - 1] == 1)
            assert np.all(target_classes[ones_start - 1 : seq_len - 1] == 2)

    def test_dfa_baseline_d_g(self):
        dfa_ = corpora.make_between_dfa(start=4, end=4)

        corpus = corpora.make_exactly_n_quantifier(4, sequence_length=50, batch_size=10)

        optimal_d_g = dfa_.get_optimal_data_given_grammar_for_dfa(corpus.input_sequence)

        num_non_masked_steps = np.sum(np.all(~np.isnan(corpus.input_sequence), axis=-1))
        assert optimal_d_g == num_non_masked_steps
