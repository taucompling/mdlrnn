import collections
import csv
import dataclasses
import itertools
import logging
import math
import random
from typing import Dict, FrozenSet, List, Optional, Text, Tuple, Union

import configuration
import dfa
import numpy as np
import utils

_DEFAULT_CONFIG = configuration.SimulationConfig(
    simulation_id="test",
    num_islands=1,
    migration_ratio=0.1,
    migration_interval_seconds=20,
    migration_interval_generations=1000,
    num_generations=1000,
    population_size=20,
    elite_ratio=0.05,
    allowed_activations=(0, 1, 2, 3, 4, 5, 6,),
    start_smooth=False,
    allowed_unit_types=(0, 1,),
    tournament_size=4,
    mutation_probab=1.0,
    grammar_multiplier=1,
    data_given_grammar_multiplier=1,
    compress_grammar_encoding=False,
    max_network_units=1024,
    softmax_outputs=False,
    truncate_large_values=True,
    bias_connections=True,
    recurrent_connections=True,
    seed=1,
    corpus_seed=100,
    generation_dump_interval=1,
    parallelize=False,
    migration_channel="file",
    mini_batch_size=None,
)


MASK_VALUE = np.nan

_Vocabulary = Dict[int, Text]


def is_masked(x: Union[np.ndarray, float]) -> Union[np.ndarray, bool]:
    return np.isnan(x)


@dataclasses.dataclass(frozen=True)
class Corpus:
    name: Text
    input_sequence: np.ndarray
    target_sequence: np.ndarray

    optimal_d_given_g: Optional[float] = None
    vocabulary: Optional[_Vocabulary] = None
    deterministic_steps_mask: Optional[np.ndarray] = None

    # Precomputed values for feeding efficiency.
    input_mask: Optional[np.ndarray] = None
    targets_mask: Optional[np.ndarray] = None
    input_values_per_time_step: Optional[Dict[int, List[np.ndarray]]] = None
    sample_weights: Optional[Tuple[int, ...]] = None

    test_corpus: Optional["Corpus"] = None


def precompute_mask_idxs(corpus: Corpus) -> Corpus:
    masked = is_masked(corpus.input_sequence)
    input_mask = np.array(
        [
            ~np.all(masked[i, j])
            for (i, j) in np.ndindex(corpus.input_sequence.shape[:2])
        ],
        dtype=np.bool,
    ).reshape(corpus.input_sequence.shape[:2])
    return dataclasses.replace(corpus, input_mask=input_mask)


def _precompute_input_unit_values(corpus: Corpus) -> Corpus:
    unit_to_timestep_val = {}
    for unit in range(corpus.input_sequence.shape[-1]):
        unit_to_timestep_val[unit] = []
        for time_step in range(corpus.input_sequence.shape[1]):
            time_step_input = np.ascontiguousarray(
                corpus.input_sequence[:, time_step, unit]
            )
            time_step_input.flags.writeable = False
            unit_to_timestep_val[unit].append(time_step_input)
        unit_to_timestep_val[unit] = tuple(unit_to_timestep_val[unit])
    return dataclasses.replace(corpus, input_values_per_time_step=unit_to_timestep_val)


def _precompute_targets_mask(corpus: Corpus) -> Corpus:
    if corpus.target_sequence.shape[-1] == 1:
        targets_mask = corpus.target_sequence == 1
    else:
        targets_mask = np.zeros_like(corpus.target_sequence, dtype=np.bool)
        target_classes = corpus.target_sequence.argmax(axis=-1).flatten()
        batch_idxs, time_idxs = tuple(
            zip(*np.ndindex(corpus.target_sequence.shape[:2]))
        )
        targets_mask[batch_idxs, time_idxs, target_classes] = True
    return dataclasses.replace(corpus, targets_mask=targets_mask)


def _make_inputs_read_only(corpus: Corpus) -> Corpus:
    corpus.input_sequence.flags.writeable = False
    return corpus


def optimize_for_feeding(corpus: Corpus) -> Corpus:
    logging.info(f"Optimizing corpus for feeding...")
    corpus = _make_inputs_read_only(corpus)
    corpus = _precompute_targets_mask(corpus)
    corpus = precompute_mask_idxs(corpus)
    corpus = _precompute_input_unit_values(corpus)
    return corpus


def make_random_binary(sequence_length: int = 100, batch_size: int = 1,) -> Corpus:
    return Corpus(
        "random_binary",
        input_sequence=np.random.randint(0, 2, size=(batch_size, sequence_length, 1)),
        target_sequence=np.random.randint(0, 2, size=(batch_size, sequence_length, 1)),
    )


def make_random_one_hot(
    num_input_classes: int,
    num_target_classes: int,
    sequence_length: int = 100,
    batch_size: int = 1,
) -> Corpus:
    input_classes = np.random.randint(
        0, num_input_classes, size=(batch_size, sequence_length)
    )
    target_classes = np.random.randint(
        0, num_target_classes, size=(batch_size, sequence_length)
    )
    return make_one_hot_corpus(
        "random_one_hot",
        input_classes=input_classes,
        target_classes=target_classes,
        num_input_classes=num_input_classes,
        num_target_classes=num_target_classes,
    )


def make_one_hot_corpus(
    name: Text,
    input_classes: Union[List, np.ndarray],
    target_classes: Union[List, np.ndarray],
    num_input_classes: int,
    num_target_classes: int,
    weights: Optional[Tuple[int, ...]] = None,
    vocabulary: Optional[_Vocabulary] = None,
) -> Corpus:
    return Corpus(
        name,
        input_sequence=_make_one_hot_sequence(
            np.array(input_classes), num_input_classes
        ),
        target_sequence=_make_one_hot_sequence(
            np.array(target_classes), num_target_classes
        ),
        sample_weights=weights,
        vocabulary=vocabulary,
    )


def _force_batch_dimension(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.expand_dims(arr, axis=0)
    return arr


def _make_one_hot_sequence(classes: np.ndarray, num_classes: int) -> np.ndarray:
    classes = _force_batch_dimension(classes)
    batch_size = classes.shape[0]
    sequence_length = classes.shape[1]

    one_hot = np.zeros(
        (batch_size, sequence_length, num_classes), dtype=utils.FLOAT_DTYPE, order="C"
    )

    for b in range(batch_size):
        for s in range(sequence_length):
            c = classes[b, s]
            if is_masked(c):
                one_hot[b, s] = MASK_VALUE
            else:
                one_hot[b, s, int(c)] = 1.0
    return one_hot


def make_between_dfa(start: int, end: int) -> dfa.DFA:
    final_state = end + 1
    transitions = {}
    for i in range(start):
        transitions[i] = {"0": i, "1": i + 1}
    for i in range(start, end):
        transitions[i] = {"0": i, "1": i + 1, dfa.END_OF_SEQUENCE: final_state}
    transitions[end] = {"0": end, dfa.END_OF_SEQUENCE: final_state}
    return dfa.DFA(transitions=transitions, accepting_states={final_state})


def make_at_least_dfa(n: int) -> dfa.DFA:
    transitions = {}
    for i in range(n):
        transitions[i] = {"0": i, "1": i + 1}
    transitions[n] = {"0": n, "1": n, dfa.END_OF_SEQUENCE: n + 1}
    return dfa.DFA(transitions=transitions, accepting_states={n + 1})


def _make_at_most_dfa(n: int) -> dfa.DFA:
    transitions = {}
    accepting_state = n + 1
    for i in range(n):
        transitions[i] = {"0": i, "1": i + 1, dfa.END_OF_SEQUENCE: accepting_state}
    transitions[n] = {"0": n, dfa.END_OF_SEQUENCE: accepting_state}
    return dfa.DFA(transitions=transitions, accepting_states={accepting_state})


def _dfa_to_inputs(
    dfa_: dfa.DFA, batch_size: int, end_of_sequence_char: int, max_sequence_length: int,
) -> np.ndarray:
    batch = np.empty((batch_size, max_sequence_length))
    batch.fill(MASK_VALUE)

    for b in range(batch_size):
        input_idx = 0
        while True:
            string = dfa_.generate_string()
            if len(string) > max_sequence_length:
                continue
            if input_idx + len(string) > max_sequence_length:
                break
            for i, char in enumerate(string):
                if char == dfa.END_OF_SEQUENCE:
                    char_int = end_of_sequence_char
                else:
                    char_int = int(char)
                batch[b, input_idx + i] = char_int
            input_idx += len(string)

    return batch


def make_identity(
    sequence_length: int = 1000, batch_size: int = 10, num_classes: int = 2
) -> Corpus:
    sequence = np.random.randint(
        num_classes, size=(batch_size, sequence_length)
    ).astype(utils.FLOAT_DTYPE)
    return make_one_hot_corpus("identity", sequence, sequence, num_classes, num_classes)


def make_identity_binary(sequence_length: int, batch_size: int) -> Corpus:
    sequence = np.random.randint(2, size=(batch_size, sequence_length, 1)).astype(
        utils.FLOAT_DTYPE
    )
    input_sequence = np.copy(sequence)
    input_sequence[sequence == 0] = -1
    return Corpus("identity", input_sequence, sequence)


def _make_repetition_sequence(
    input_sequence: np.ndarray, offset: int, padding: float = 0.0
) -> np.ndarray:
    """[a,b,c,d, ..., y,z] -> [<padding>,a,b,c, ..., y] """
    assert input_sequence.ndim == 2
    batch_size = input_sequence.shape[0]
    padded_arr = np.empty((batch_size, offset), dtype=utils.FLOAT_DTYPE)
    padded_arr.fill(padding)
    return np.concatenate((padded_arr, input_sequence[:, :-offset],), axis=-1,)


def _make_prediction_sequence(
    input_sequence: np.ndarray, lookahead: int = 1, padding: float = 0.0
) -> np.ndarray:
    """[a,b,c,d, ..., y,z] -> [b,c,d,e, ..., z,<padding>] """
    input_sequence = _force_batch_dimension(input_sequence)
    assert input_sequence.ndim == 2
    batch_size = input_sequence.shape[0]
    padded_arr = np.empty((batch_size, lookahead), dtype=utils.FLOAT_DTYPE)
    padded_arr.fill(padding)
    return np.concatenate((input_sequence[:, lookahead:], padded_arr), axis=-1)


def make_prev_char_repetition(
    sequence_length: int = 1000,
    batch_size: int = 10,
    repetition_offset: int = 1,
    num_classes: int = 2,
) -> Corpus:
    input_sequence = np.random.randint(num_classes, size=(batch_size, sequence_length))
    target_sequence = _make_repetition_sequence(input_sequence, repetition_offset)

    return make_one_hot_corpus(
        f"repeat_prev_{repetition_offset}_char",
        input_sequence,
        target_sequence,
        num_classes,
        num_classes,
    )


def make_prev_char_repetition_binary(
    sequence_length: int, batch_size: int, repetition_offset: int,
) -> Corpus:
    input_sequence = np.random.randint(2, size=(batch_size, sequence_length)).astype(
        utils.FLOAT_DTYPE
    )
    target_sequence = _make_repetition_sequence(input_sequence, repetition_offset)
    return Corpus(
        f"repeat_prev_{repetition_offset}_char_binary",
        np.expand_dims(input_sequence, -1),
        np.expand_dims(target_sequence, -1),
    )


def make_elman_xor_binary(sequence_length: int = 3000, batch_size: int = 1) -> Corpus:
    assert sequence_length % 3 == 0

    input_batch = []
    target_batch = []
    for b in range(batch_size):
        sequence = []
        for pair_idx in range(sequence_length // 3):
            a, b = random.choice([0, 1]), random.choice([0, 1])
            sequence += [a, b, a ^ b]
        input_batch.append(sequence)
        # Target output is the next character of input
        target_sequence = sequence[1:] + [0]
        target_batch.append(target_sequence)
    input_batch = np.expand_dims(
        np.array(input_batch, dtype=utils.FLOAT_DTYPE), axis=-1
    )
    target_batch = np.expand_dims(
        np.array(target_batch, dtype=utils.FLOAT_DTYPE), axis=-1
    )
    return Corpus("elman_xor_binary", input_batch, target_batch)


def make_elman_xor_one_hot(sequence_length: int = 3000, batch_size: int = 1) -> Corpus:
    binary_corpus = make_elman_xor_binary(sequence_length, batch_size)

    return make_one_hot_corpus(
        "elman_xor_one_hot",
        binary_corpus.input_sequence,
        binary_corpus.target_sequence,
        num_input_classes=2,
        num_target_classes=2,
    )


def make_semi_random_corpus(sequence_length: int = 100, batch_size: int = 10) -> Corpus:
    """ One random bit, one identical bit, e.g.: [0,0,0,0,1,1,0,0,1,1,0,0, ...] """
    assert sequence_length % 2 == 0
    input_batch = []
    target_batch = []
    for _ in range(batch_size):
        sequence = []
        for s in range(sequence_length // 2):
            sequence += [random.randrange(2)] * 2
        input_batch.append(sequence)
        target_sequence = sequence[1:] + [0]
        target_batch.append(target_sequence)

    input_batch = np.expand_dims(np.array(input_batch), axis=-1)
    target_batch = np.expand_dims(np.array(target_batch), axis=-1)
    return Corpus("semi_random_pairs", input_batch, target_batch)


def make_elman_badigu(num_consonants: int = 1000) -> Corpus:
    # ba, dii, guuu
    feature_table = {
        # cons, vowel, int, high, back, voiced
        "b": [1, 0, 1, 0, 0, 1],  # b
        "d": [1, 0, 1, 1, 0, 1],  # d
        "g": [1, 0, 1, 0, 1, 1],  # g
        "a": [0, 1, 0, 0, 1, 1],  # a
        "i": [0, 1, 0, 1, 0, 1],  # i
        "u": [0, 1, 0, 1, 1, 1],  # u
    }
    segments = list("bdgaiu")
    num_classes = len(segments)
    segment_to_idx = {x: i for i, x in enumerate(segments)}

    consonant_to_sequence = {"b": list("ba"), "d": list("dii"), "g": list("guuu")}
    consonant_sequence = np.random.choice(["b", "d", "g"], size=num_consonants)

    letters_sequence = list(
        itertools.chain(*(consonant_to_sequence[c] for c in consonant_sequence))
    )
    input_sequence = [segment_to_idx[x] for x in letters_sequence]
    target_sequence = input_sequence[1:] + [0]

    logging.info(f"Elman badigu sequence: {letters_sequence}")
    consonants = tuple("bdg")

    consonant_percentage = len([x for x in letters_sequence if x in consonants]) / len(
        letters_sequence
    )

    logging.info(f"Max accuracy for task: {1-consonant_percentage:.2f}")

    return make_one_hot_corpus(
        "elman_badigu", input_sequence, target_sequence, num_classes, num_classes
    )


def _make_0_1_pattern_binary(sequence_length: int, batch_size: int) -> Corpus:
    assert sequence_length % 2 == 0

    input_seq = np.array([[0, 1] * (sequence_length // 2)], dtype=utils.FLOAT_DTYPE)
    target_seq = _make_prediction_sequence(input_seq, lookahead=1, padding=0.0)

    input_seq = np.expand_dims(input_seq, axis=2)
    target_seq = np.expand_dims(target_seq, axis=2)

    return Corpus(
        name=f"0_1_pattern_binary_length_{sequence_length}_batch_{batch_size}",
        input_sequence=input_seq,
        target_sequence=target_seq,
        optimal_d_given_g=0.0,
        vocabulary={0: "1", 1: "1"},
        sample_weights=(batch_size,) if batch_size > 1 else None,
    )


def make_0_1_pattern_binary(sequence_length: int, batch_size: int) -> Corpus:
    train_corpus = _make_0_1_pattern_binary(sequence_length, batch_size)
    test_corpus = _make_0_1_pattern_binary(
        sequence_length=sequence_length * 50_000, batch_size=1
    )
    return dataclasses.replace(train_corpus, test_corpus=test_corpus)


def _make_0_1_pattern_one_hot(
    sequence_length: int, add_end_of_sequence: bool, batch_size: int
) -> Corpus:
    assert sequence_length % 2 == 0

    input_classes = [0, 1] * (sequence_length // 2)
    num_classes = 2
    vocabulary = {0: "0", 1: "1"}

    if add_end_of_sequence:
        num_classes = 3
        input_classes += [2]
        vocabulary[2] = "#"

    vocabulary.update({x + len(vocabulary): vocabulary[x] for x in vocabulary})

    input_classes_arr = np.array([input_classes])
    target_classes_arr = _make_prediction_sequence(
        input_classes_arr, lookahead=1, padding=0.0
    )

    corpus = make_one_hot_corpus(
        name=f"0_1_pattern_one_hot_length_{sequence_length}_batch_{batch_size}{'_eos' if add_end_of_sequence else ''}",
        input_classes=input_classes_arr,
        target_classes=target_classes_arr,
        num_input_classes=num_classes,
        num_target_classes=num_classes,
        weights=(batch_size,) if batch_size > 1 else None,
        vocabulary=vocabulary,
    )
    return dataclasses.replace(
        # TODO: calculate optimal D
        corpus,
        optimal_d_given_g=0.0,
    )


def make_0_1_pattern_one_hot(
    sequence_length: int, add_end_of_sequence: bool, batch_size: int
) -> Corpus:
    train_corpus = _make_0_1_pattern_one_hot(
        sequence_length, add_end_of_sequence, batch_size
    )
    test_corpus = _make_0_1_pattern_one_hot(
        sequence_length=sequence_length * 50_000,
        add_end_of_sequence=add_end_of_sequence,
        batch_size=1,
    )
    return dataclasses.replace(train_corpus, test_corpus=test_corpus)


def make_123_n_pattern_corpus(
    base_sequence_length: int = 3, sequence_length: int = 100
):
    # [0,1,2, ..., n-1] repeated
    assert sequence_length % base_sequence_length == 0
    input_sequence = np.array(
        list(range(base_sequence_length)) * (sequence_length // base_sequence_length)
    )
    target_sequence = _make_prediction_sequence(input_sequence, lookahead=1)
    return make_one_hot_corpus(
        f"1_to_{base_sequence_length}_pattern",
        input_sequence,
        target_sequence,
        num_input_classes=base_sequence_length,
        num_target_classes=base_sequence_length,
    )


def make_between_quantifier(
    start: int, end: int, sequence_length: int = 100, batch_size: int = 1
) -> Corpus:
    assert end <= sequence_length
    between_dfa = make_between_dfa(start, end)
    between_dfa.visualize(f"between_{start}_{end}_dfa")
    input_batch = _dfa_to_inputs(
        between_dfa,
        batch_size=batch_size,
        end_of_sequence_char=2,
        max_sequence_length=sequence_length,
    )
    target_batch = _make_prediction_sequence(input_batch, lookahead=1)
    if start == end:
        name = f"exactly_{start}"
    else:
        name = f"between_{start}_{end}"
    num_classes = 3
    return make_one_hot_corpus(
        name, input_batch, target_batch, num_classes, num_classes
    )


def make_exactly_n_quantifier(
    n: int = 1, sequence_length: int = 100, batch_size: int = 1
) -> Corpus:
    return make_between_quantifier(
        start=n, end=n, sequence_length=sequence_length, batch_size=batch_size
    )


def make_at_least_quantifier(
    n: int = 1, sequence_length: int = 100, batch_size: int = 1
) -> Corpus:
    name = f"at_least_{n}"
    at_least_dfa = make_at_least_dfa(n)
    at_least_dfa.visualize(name)
    input_batch = _dfa_to_inputs(
        at_least_dfa,
        batch_size=batch_size,
        end_of_sequence_char=2,
        max_sequence_length=sequence_length,
    )
    target_batch = _make_prediction_sequence(input_batch, lookahead=1)
    num_classes = 3
    return make_one_hot_corpus(
        name, input_batch, target_batch, num_classes, num_classes
    )


def make_at_most_quantifier(
    n: int = 1, sequence_length: int = 100, batch_size: int = 1
) -> Corpus:
    name = f"at_most_{n}"
    at_most_dfa = _make_at_most_dfa(n)
    at_most_dfa.visualize(name)
    input_batch = _dfa_to_inputs(
        at_most_dfa,
        batch_size=batch_size,
        end_of_sequence_char=2,
        max_sequence_length=sequence_length,
    )
    target_batch = _make_prediction_sequence(input_batch, lookahead=1)
    num_classes = 3
    return make_one_hot_corpus(
        name, input_batch, target_batch, num_classes, num_classes
    )


def make_every_quantifier(sequence_length: int = 100, batch_size: int = 1) -> Corpus:
    input_batch = np.ones((batch_size, sequence_length))
    return make_one_hot_corpus(f"every_quantifier", input_batch, input_batch, 2, 2)


def _int_to_classes(n: int) -> List[int]:
    return list(reversed(list(map(int, list(str(n))))))


def make_count_corpus(max_int: 100, batch_size: int = 100) -> Corpus:
    # Predict n+1 from n in a language-model setting.
    sequence_length = int(np.floor(np.log10(max_int))) + 1
    input_classes = np.zeros((batch_size, sequence_length))
    target_classes = np.zeros((batch_size, sequence_length))

    for b in range(batch_size):
        n = random.randrange(max_int)
        input_ = _int_to_classes(n)
        target = _int_to_classes(n + 1)
        input_classes[b, : len(input_)] = input_
        target_classes[b, : len(target)] = target

    return make_one_hot_corpus(
        f"count_to_{max_int}",
        input_classes,
        target_classes,
        num_input_classes=10,
        num_target_classes=10,
    )


def base10_to_binary_vector(n: int, sequence_length=None) -> np.ndarray:
    """8 -> [0,0,0,1], 7 -> [1,1,1] """
    if n == 0:
        return np.zeros(sequence_length)

    powers = []
    while n:
        power = int(np.floor(np.log2(n)))
        powers.append(power)
        n -= 2 ** power

    rightmost_one_position = int(max(powers))
    if sequence_length is None:
        sequence_length = rightmost_one_position + 1
    binary = np.zeros(sequence_length)
    binary[powers] = 1.0
    # TODO: mask redundant positions?
    return binary


def _make_binary_addition_corpus(min_n: int, max_n: int):
    all_summands = tuple(itertools.product(range(min_n, max_n), repeat=2))
    summands = []
    for b, (n1, n2) in enumerate(all_summands):
        summands.append([n1, n2])
    summands = np.array(summands)
    sums = np.sum(summands, axis=1)
    sequence_length = int(np.ceil(np.log2(np.max(sums)))) + 1

    summand_binaries = []
    sum_binaries = []
    for (n1, n2), sum_ in zip(summands, sums):
        summand_binaries.append(
            [
                base10_to_binary_vector(n1, sequence_length),
                base10_to_binary_vector(n2, sequence_length),
            ]
        )
        sum_binaries.append(base10_to_binary_vector(sum_, sequence_length))

    summand_inputs = np.array(
        [np.stack(summands, axis=1,) for summands in summand_binaries]
    )

    sum_outputs = np.expand_dims(np.stack(sum_binaries), axis=-1)

    return dataclasses.replace(
        Corpus(
            name=f"binary_addition_{min_n}_to_{max_n}",
            input_sequence=summand_inputs,
            target_sequence=sum_outputs,
        ),
        optimal_d_given_g=0.0,
    )


def make_binary_addition(min_n: int, max_n: int) -> Corpus:
    training_corpus = _make_binary_addition_corpus(min_n=min_n, max_n=max_n)
    test_corpus = _make_binary_addition_corpus(min_n=max_n + 1, max_n=max_n + 251)
    training_corpus = dataclasses.replace(training_corpus, test_corpus=test_corpus)
    return training_corpus


def an_bn_handmade_net(input_seq: np.ndarray, prior: float):
    # Optimal network according to Schimdhuber (2001).
    outputs = np.zeros_like(input_seq)
    for b in range(input_seq.shape[0]):
        num_seen_a = 0
        for t in range(input_seq.shape[1]):
            input_vec = input_seq[b, t]
            input_class = input_vec.argmax()
            if input_class == 0:
                # Start of sequence symbol, always predict "a" (no empty string in current corpus).
                outputs[b, t] = [0.0, 1.0, 0.0]
            elif input_class == 1:
                # "a".
                num_seen_a += 1
                outputs[b, t] = [0.0, 1 - prior, prior]
            elif input_class == 2:
                # "b".
                num_seen_a -= 1
                if num_seen_a > 0:
                    outputs[b, t] = [0.0, 0.0, 1.0]
                else:
                    outputs[b, t] = [1.0, 0.0, 0.0]

    return outputs


def make_english_onset_phonotactics(split_ratio: Optional[float] = None):
    PHONOTACTIC_COUNTS = {
        "k": 2764,
        "r": 2752,
        "d": 2526,
        "s": 2215,
        "m": 1965,
        "p": 1881,
        "b": 1544,
        "l": 1225,
        "f": 1222,
        "h": 1153,
        "t": 1146,
        "pr": 1046,
        "w": 780,
        "n": 716,
        "v": 615,
        "g": 537,
        "dÇ": 524,
        "st": 521,
        "tr": 515,
        "kr": 387,
        "+": 379,
        "gr": 331,
        "t+": 329,
        "br": 319,
        "sp": 313,
        "fl": 290,
        "kl": 285,
        "sk": 278,
        "j": 268,
        "fr": 254,
        "pl": 238,
        "bl": 213,
        "sl": 213,
        "dr": 211,
        "kw": 201,
        "str": 183,
        "‡": 173,
        "sw": 153,
        "gl": 131,
        "hw": 111,
        "sn": 109,
        "skr": 93,
        "z": 83,
        "sm": 82,
        "‡r": 73,
        "skw": 69,
        "tw": 55,
        "spr": 51,
        "+r": 40,
        "spl": 27,
        "L": 19,
        "dw": 17,
        "gw": 11,
        "‡w": 4,
        "skl": 1,
    }
    name = "english_onset_phonotactics"
    inputs, targets, num_classes, weights = _make_phonotactic_corpus(
        phonotactic_counts=PHONOTACTIC_COUNTS
    )
    if not split_ratio:
        return make_one_hot_corpus(
            name, inputs, targets, num_classes, num_classes, weights
        )
    train_inputs, test_inputs = split_train_and_test(inputs, split_ratio)
    train_targets, test_targets = split_train_and_test(targets, split_ratio)
    return Corpus(
        name=f"{name}_{split_ratio}_split",
        input_sequence=_make_one_hot_sequence(train_inputs, num_classes),
        target_sequence=_make_one_hot_sequence(train_targets, num_classes),
        test_input_sequence=_make_one_hot_sequence(test_inputs, num_classes),
        test_target_sequence=_make_one_hot_sequence(test_targets, num_classes),
        sample_weights=weights,
    )


def _load_futrell_phonotactic_data(filename: Text) -> Dict[Text, int]:
    word_to_count = {}
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word_to_count[row["Phonology"]] = int(row["LemmaFrequency"])
    return word_to_count


def make_futrell_german_phonotactic_corpus(split_ratio: float):
    inputs, targets, num_classes, weights = _make_phonotactic_corpus(
        _load_futrell_phonotactic_data("German.Dict.CSV")
    )
    train_inputs, test_inputs = split_train_and_test(inputs, split_ratio)
    train_targets, test_targets = split_train_and_test(targets, split_ratio)
    return Corpus(
        name=f"german_phonotactics_{split_ratio}_split",
        input_sequence=_make_one_hot_sequence(train_inputs, num_classes),
        target_sequence=_make_one_hot_sequence(train_targets, num_classes),
        test_input_sequence=_make_one_hot_sequence(test_inputs, num_classes),
        test_target_sequence=_make_one_hot_sequence(test_targets, num_classes),
        sample_weights=weights,
    )


def _make_phonotactic_corpus(phonotactic_counts: Dict[Text, int]):
    segments = sorted(list(set("".join(phonotactic_counts))))
    segment_to_index = {segment: i for i, segment in enumerate(segments)}
    sequences = list(phonotactic_counts.keys())
    max_sequence_len = max([len(x) for x in sequences])
    segments = list(segment_to_index)
    weights = tuple(phonotactic_counts[seq] for seq in sequences)
    inputs = np.empty((len(sequences), max_sequence_len + 1))
    targets = np.empty_like(inputs)
    inputs.fill(MASK_VALUE)
    targets.fill(MASK_VALUE)
    for i in range(len(sequences)):
        # Start-of-sequence symbol
        inputs[i, 0] = len(segments)
        for j in range(len(sequences[i])):
            inputs[i, j + 1] = segment_to_index[sequences[i][j]]
            targets[i, j] = segment_to_index[sequences[i][j]]
        # End-of-sequence symbol
        targets[i, len(sequences[i])] = len(segments)
    return inputs, targets, len(segments) + 1, weights


class _DerivationTooLong(Exception):
    pass


def _generate_string_from_pcfg(
    pcfg: Dict, max_length: Optional[int] = None
) -> Tuple[Text, ...]:
    """Stops when all generated characters are terminals.
    To stop without adding an epsilon terminal, use the empty string '', i.e. add a rule `S->''`. """
    stack = ["S"]
    terminals = []
    while stack:
        node = stack[0]
        stack = stack[1:]

        if node not in pcfg:
            terminals.append(node)
            if max_length is not None and len(terminals) > max_length:
                raise _DerivationTooLong
            continue

        rules, probabs = list(zip(*pcfg[node]))
        rule_idx = np.random.choice(len(rules), p=probabs)
        rule = rules[rule_idx]

        stack = list(rule) + stack

    return tuple(terminals)


def _make_corpus_from_pcfg(
    name: Text,
    pcfg: Dict,
    batch_size: int,
    max_derivation_length: Optional[int] = None,
    sort_by_length: bool = False,
) -> Corpus:
    sequences = []
    while len(sequences) < batch_size:
        try:
            sequence = _generate_string_from_pcfg(
                pcfg, max_length=max_derivation_length
            )
        except _DerivationTooLong:
            continue
        sequences.append(sequence)

    if sort_by_length:
        sequences = sorted(sequences, key=len, reverse=True)

    lengths = list(map(len, sequences))

    sequence_counts = collections.Counter(sequences)
    unique_sequences, weights = tuple(zip(*sequence_counts.items()))

    logging.info(f"PCFG sum of sequence lengths: {sum(lengths)}")
    logging.info(f"PCFG max sequence length: {max(lengths)}")
    logging.info(f"PCFG mean sequence length: {np.mean(lengths)}")
    logging.info(
        f"PCFG unique sequences: {len(unique_sequences)}/{len(sequences)} ({len(unique_sequences)/len(sequences):.2f})"
    )

    alphabet = set()
    for rules in pcfg.values():
        alphabet |= set(itertools.chain(*set(map(lambda x: x[0], rules))))
    alphabet -= set(pcfg.keys())
    alphabet = ("#",) + tuple(sorted(alphabet))

    symbol_to_idx = {x: i for i, x in enumerate(alphabet)}

    max_seq_length = max(map(len, unique_sequences))
    input_classes = np.empty((len(unique_sequences), max_seq_length + 1))
    target_classes = np.empty_like(input_classes)
    input_classes.fill(MASK_VALUE)
    target_classes.fill(MASK_VALUE)

    for i, sequence in enumerate(unique_sequences):
        sequence_classes = [symbol_to_idx[symbol] for symbol in sequence]
        input_row = [symbol_to_idx["#"]] + sequence_classes
        target_row = sequence_classes + [symbol_to_idx["#"]]
        input_classes[i, : len(sequence_classes) + 1] = input_row
        target_classes[i, : len(sequence_classes) + 1] = target_row

    inputs = _make_one_hot_sequence(input_classes, num_classes=len(alphabet))
    targets = _make_one_hot_sequence(target_classes, num_classes=len(alphabet))

    vocabulary = _make_identical_input_output_vocabulary(alphabet)

    return Corpus(
        name=name,
        input_sequence=inputs,
        target_sequence=targets,
        sample_weights=weights,
        vocabulary=vocabulary,
    )


def make_center_embedding(
    batch_size: int, embedding_depth_probab: float, dependency_distance_probab: float
) -> Corpus:
    pcfg = {
        "S": (
            (("NP_s", "VP_s"), (1 - embedding_depth_probab) / 2),
            (("NP_p", "VP_p"), (1 - embedding_depth_probab) / 2),
            (("NP_s", "S", "VP_s"), embedding_depth_probab / 2),
            (("NP_p", "S", "VP_p"), embedding_depth_probab / 2),
        ),
        "NP_s": (
            (("N_s",), 1 - dependency_distance_probab),
            # (("A", "NP_s"), dependency_distance_probab),
        ),
        "NP_p": (
            (("N_p",), 1 - dependency_distance_probab),
            # (("A", "NP_p"), dependency_distance_probab),
        ),
        "VP_s": (
            (("V_s",), 1 - dependency_distance_probab),
            # (("A", "VP_s"), dependency_distance_probab),
        ),
        "VP_p": (
            (("V_p",), 1 - dependency_distance_probab),
            # (("A", "VP_p"), dependency_distance_probab),
        ),
        "N_s": (
            (("cat",), 1.0),
            # (("dog",), 1.0),
            # (("horse",), 0.2),
            # (("rat",), 0.2),
            # (("flower",), 0.2),
        ),
        "N_p": (
            (("cats",), 1.0),
            # (("dogs",), 1.0),
            # (("horses",), 0.2),
            # (("rats",), 0.2),
            # (("flowers",), 0.2),
        ),
        "V_s": (
            (("runs",), 1.0),
            # (("talks",), 1.0),
            # (("dances",), 0.2),
            # (("eats",), 0.2),
            # (("drinks",), 0.2),
        ),
        "V_p": (
            (("run",), 1.0),
            # (("talk",), 1.0),
            # (("dance",), 0.2),
            # (("eat",), 0.2),
            # (("drink",), 0.2),
        ),
        # "A": (
        #     (("good",), 0.5),
        # (("bad",), 0.5),
        # (("nice",), 0.2),
        # (("smart",), 0.2),
        # (("funny",), 0.2),
        # ),
    }
    corpus = _make_corpus_from_pcfg(
        f"center_embedding_pcfg_embedding_{embedding_depth_probab}_distance_{dependency_distance_probab}",
        pcfg=pcfg,
        batch_size=batch_size,
    )
    input_classes = np.argmax(corpus.input_sequence, axis=-1)
    deterministic_steps_mask = (~np.all(is_masked(corpus.input_sequence), axis=-1)) & (
        # "cat/s"
        (input_classes == 3)
        | (input_classes == 4)
    )
    return dataclasses.replace(
        corpus, deterministic_steps_mask=deterministic_steps_mask
    )


def make_palindrome_with_middle_marker_distinct(batch_size: int, nesting_probab: float):
    pcfg = {
        "S": (
            (("0", "S", "0"), nesting_probab / 2),
            (("1", "S", "1"), nesting_probab / 2),
            (("@",), 1 - nesting_probab),
        )
    }
    return _make_corpus_from_pcfg(
        name=f"palindrome_middle_marker__batch_{batch_size}__p_{nesting_probab}",
        pcfg=pcfg,
        batch_size=batch_size,
    )


def _optimal_d_g_for_fixed_palindrome(corpus) -> float:
    sequence_length = corpus.input_sequence.shape[1]
    batch_size = sum(corpus.sample_weights)
    deterministic_length = sequence_length // 2
    return batch_size * deterministic_length


def make_binary_palindrome_fixed_length(
    batch_size: int, sequence_length: int, train_set_ratio: float
) -> Corpus:
    assert sequence_length % 2 == 0
    prefixes_non_unique = np.random.randint(
        2, size=(batch_size, sequence_length // 2)
    ).astype(utils.FLOAT_DTYPE)

    sequence_counts = collections.Counter(list(map(tuple, prefixes_non_unique)))
    unique_prefixes, weights = list(zip(*sequence_counts.items()))

    prefixes = np.array(unique_prefixes)
    suffixes = np.flip(prefixes, axis=1)
    sequences = np.concatenate([prefixes, suffixes], axis=1)
    targets = _make_prediction_sequence(input_sequence=sequences, lookahead=1)

    input_sequences = np.expand_dims(sequences, axis=2)
    target_sequences = np.expand_dims(targets, axis=2)

    logging.info(
        f"Fixed palindrome: {len(unique_prefixes)}/{len(prefixes_non_unique)} unique sequences"
    )

    full_corpus = optimize_for_feeding(
        Corpus(
            name=f"palindrome_binary_fixed_length_batch_{batch_size}_length_{sequence_length}",
            input_sequence=input_sequences,
            target_sequence=target_sequences,
            sample_weights=weights,
        )
    )

    train, test = split_train_test(full_corpus, train_ratio=train_set_ratio)
    logging.info(
        f"Train size: {train.input_sequence.shape[0]}, test size: {test.input_sequence.shape[0]}"
    )
    test = dataclasses.replace(
        test, optimal_d_given_g=_optimal_d_g_for_fixed_palindrome(test)
    )
    return dataclasses.replace(
        train,
        test_corpus=test,
        optimal_d_given_g=_optimal_d_g_for_fixed_palindrome(train),
    )


def _make_an_bn_square_corpus(n_values: Tuple[int, ...], prior: float):
    start_end_of_sequence_symbol = 0
    max_n = max(n_values)
    max_sequence_length = max_n + (max_n ** 2) + 1

    n_values_counts = collections.Counter(n_values)
    unique_n_values, n_values_weights = tuple(zip(*n_values_counts.items()))

    inputs = np.empty((len(unique_n_values), max_sequence_length))
    targets = np.empty_like(inputs)
    inputs.fill(MASK_VALUE)
    targets.fill(MASK_VALUE)

    for b, n in enumerate(unique_n_values):
        input_seq = [start_end_of_sequence_symbol] + ([1] * n) + ([2] * n ** 2)
        target_seq = input_seq[1:] + [start_end_of_sequence_symbol]
        inputs[b, : len(input_seq)] = input_seq
        targets[b, : len(input_seq)] = target_seq

    corpus = make_one_hot_corpus(
        f"an_bn_square_batch_{len(n_values)}_p_{prior}",
        inputs,
        targets,
        num_input_classes=3,
        num_target_classes=3,
        vocabulary=_make_identical_input_output_vocabulary(alphabet=("#", "a", "b")),
        weights=n_values_weights,
    )
    return dataclasses.replace(
        corpus,
        optimal_d_given_g=_get_ain_bjn_ckn_dtn_optimal_d_given_g(prior, n_values),
    )


def make_an_bn_square(batch_size: int, prior: float) -> Corpus:
    training_n_values = tuple(np.random.geometric(p=prior, size=batch_size))
    training_corpus = _make_an_bn_square_corpus(training_n_values, prior)

    max_training_n = max(training_n_values)
    test_n_values = tuple(range(max_training_n + 1, max_training_n + 11))
    test_corpus = _make_an_bn_square_corpus(test_n_values, prior)
    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def _make_identical_input_output_vocabulary(alphabet: Tuple[Text, ...]) -> _Vocabulary:
    # Create class index to symbol mapping, assuming inputs and outputs are identical and ordered identically.
    class_to_symbol = {idx: alphabet[idx] for idx in range(len(alphabet))}
    class_to_symbol.update(
        {idx + len(alphabet): symbol for idx, symbol in class_to_symbol.items()}
    )
    return class_to_symbol


def _get_ain_bjn_ckn_dtn_optimal_d_given_g(prior, n_values) -> float:
    return -np.sum(
        [(n - 1) * (np.log2(1 - prior)) + np.log2(prior) for n in n_values]
    ).item()


def get_num_chars_in_corpus(corpus: Corpus) -> int:
    non_masked = ~np.all(is_masked(corpus.input_sequence), axis=-1)
    num_chars_per_row = np.sum(non_masked, axis=1)
    if corpus.sample_weights:
        total_chars = np.dot(num_chars_per_row, corpus.sample_weights)
    else:
        total_chars = np.sum(num_chars_per_row)
    return total_chars.item()


def make_inputs_counter(num_inputs: int, num_ones: int, batch_size: int):
    # From Schmidhuber (1997) -- network's goal is to output the number of ones in the input.
    # The optimal solution is to set all weights to 1.
    inputs = np.zeros((batch_size, 1, num_inputs), dtype=utils.FLOAT_DTYPE)
    for b in range(batch_size):
        idxs = random.choices(range(num_inputs), k=num_ones)
        inputs[b, idxs] = 1.0

    targets = np.ones((batch_size, 1, 1)) * num_ones

    return Corpus(
        name=f"inputs_counter", input_sequence=inputs, target_sequence=targets,
    )

    pass


def _make_ain_bjn_ckn_dtn_corpus(
    n_values: Tuple[int, ...],
    multipliers: Tuple[int, ...],
    prior: float,
    sort_by_length: bool,
) -> Corpus:
    # Create a corpus of a^in, b^jn, c^kn, d^tn, multipliers = [i,j,k,n].
    max_n = max(n_values)
    max_sequence_length = (max_n * sum(multipliers)) + 1

    start_end_of_sequence_symbol = 0  # Using same symbol for start/end of sequence, as in Schmidhuber et al. (2001).

    n_values_counts = collections.Counter(n_values)
    n_value_counts_items = tuple(n_values_counts.items())
    if sort_by_length:
        n_value_counts_items = sorted(n_value_counts_items, reverse=True)

    unique_n_values, n_values_weights = tuple(zip(*n_value_counts_items))

    inputs = np.empty((len(unique_n_values), max_sequence_length))
    targets = np.empty_like(inputs)
    inputs.fill(MASK_VALUE)
    targets.fill(MASK_VALUE)

    for b, n in enumerate(unique_n_values):
        input_seq = (
            [start_end_of_sequence_symbol]
            + ([1] * n * multipliers[0])
            + ([2] * n * multipliers[1])
            + ([3] * n * multipliers[2])
            + ([4] * n * multipliers[3])
        )
        target_seq = input_seq[1:] + [start_end_of_sequence_symbol]
        inputs[b, : len(input_seq)] = input_seq
        targets[b, : len(input_seq)] = target_seq

    name = f"a{multipliers[0]}n_b{multipliers[1]}n_c{multipliers[2]}n_d{multipliers[3]}n__p_{prior}__batch_{len(n_values)}"
    num_input_classes = sum([1 for x in multipliers if x != 0]) + 1

    alphabet = ("#", "a", "b", "c", "d")[:num_input_classes]
    vocabulary = _make_identical_input_output_vocabulary(alphabet)

    corpus = make_one_hot_corpus(
        name,
        inputs,
        targets,
        num_input_classes=num_input_classes,
        num_target_classes=num_input_classes,
        weights=n_values_weights,
        vocabulary=vocabulary,
    )
    return dataclasses.replace(
        corpus,
        optimal_d_given_g=_get_ain_bjn_ckn_dtn_optimal_d_given_g(prior, n_values),
        # TODO: this assumes no empty sequences in corpus.
        deterministic_steps_mask=(~is_masked(inputs)) & (inputs != 1),
    )


def make_ain_bjn_ckn_dtn(
    batch_size: int,
    prior: float,
    multipliers: Tuple[int, ...],
    sort_by_length: bool = False,
) -> Corpus:
    training_n_values = tuple(np.random.geometric(p=prior, size=batch_size))
    training_corpus = _make_ain_bjn_ckn_dtn_corpus(
        training_n_values, multipliers, prior, sort_by_length
    )

    max_training_n = max(training_n_values)
    test_n_values = tuple(range(max_training_n + 1, max_training_n + 1001))
    test_corpus = _make_ain_bjn_ckn_dtn_corpus(
        test_n_values, multipliers, prior, sort_by_length
    )
    test_corpus = dataclasses.replace(test_corpus, name=f"{test_corpus.name}_test")

    logging.info(f"Created corpus {training_corpus.name}")
    logging.info(f"Max n in training set: {max_training_n}")
    logging.info(f"Optimal training set D:G: {training_corpus.optimal_d_given_g:,.2f}")
    logging.info(f"Optimal test set D:G: {test_corpus.optimal_d_given_g:,.2f}")

    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def _get_an_bm_cn_plus_m_corpus_optimal_d_given_g(
    n_plus_m_values: Tuple[int, ...], prior: float
) -> float:
    return -np.sum(
        [
            ((n_plus_m - 2) * np.log2(1 - prior)) + (2 * np.log2(prior))
            for n_plus_m in n_plus_m_values
        ]
    ).item()


def _make_an_bm_cn_plus_m_corpus(
    n_values: Tuple[int, ...],
    m_values: Tuple[int, ...],
    prior: float,
    sort_by_length: bool,
) -> Corpus:
    sum_values = tuple(np.add(n_values, m_values))
    start_end_of_sequence_symbol = 0
    max_sequence_length = 2 * max(sum_values) + 1

    n_m_values_counts = collections.Counter(zip(n_values, m_values))
    n_m_values_counts_items = tuple(n_m_values_counts.items())
    if sort_by_length:
        n_m_values_counts_items = sorted(
            n_m_values_counts_items, key=lambda x: sum(x[0]), reverse=True
        )

    unique_n_m_values, n_m_values_weights = tuple(zip(*n_m_values_counts_items))

    inputs = np.empty((len(unique_n_m_values), max_sequence_length))
    targets = np.empty_like(inputs)
    inputs.fill(MASK_VALUE)
    targets.fill(MASK_VALUE)

    for b in range(len(unique_n_m_values)):
        n, m = unique_n_m_values[b]
        input_seq = (
            [start_end_of_sequence_symbol] + ([1] * n) + ([2] * m) + ([3] * (n + m))
        )
        target_seq = input_seq[1:] + [start_end_of_sequence_symbol]
        inputs[b, : len(input_seq)] = input_seq
        targets[b, : len(input_seq)] = target_seq

    vocabulary = _make_identical_input_output_vocabulary(alphabet=("#", "a", "b", "c"))

    corpus = make_one_hot_corpus(
        f"an_bm_cn_plus_m__batch_{len(n_values)}_p_{prior}",
        inputs,
        targets,
        num_input_classes=4,
        num_target_classes=4,
        weights=n_m_values_weights,
        vocabulary=vocabulary,
    )

    return dataclasses.replace(
        corpus,
        optimal_d_given_g=_get_an_bm_cn_plus_m_corpus_optimal_d_given_g(
            sum_values, prior
        ),
        # TODO: this assumes no empty sequences in corpus.
        deterministic_steps_mask=(~is_masked(inputs)) & (inputs != 1) & (inputs != 2),
    )


def make_an_bm_cn_plus_m(
    batch_size: int, prior: float, sort_by_length: bool = False,
) -> Corpus:
    training_n_values = tuple(np.random.geometric(p=prior, size=batch_size))
    training_m_values = tuple(np.random.geometric(p=prior, size=batch_size))

    training_corpus = _make_an_bm_cn_plus_m_corpus(
        training_n_values, training_m_values, prior, sort_by_length
    )
    max_n = max(training_n_values)
    max_m = max(training_m_values)
    max_training_n_or_m = max(max_n, max_m)

    test_n_values, test_m_values = zip(
        *itertools.product(
            range(max_training_n_or_m + 1, max_training_n_or_m + 50), repeat=2
        )
    )

    test_corpus = _make_an_bm_cn_plus_m_corpus(
        test_n_values, test_m_values, prior, sort_by_length
    )
    test_corpus = dataclasses.replace(test_corpus, name=f"{test_corpus.name}_test")

    logging.info(f"Created corpus {training_corpus.name}")
    logging.info(f"Max n in training: {max_n}")
    logging.info(f"Max m in training: {max_m}")
    logging.info(f"Optimal training set D:G: {training_corpus.optimal_d_given_g:,.2f}")
    logging.info(f"Optimal test set D:G: {test_corpus.optimal_d_given_g:,.2f}")
    logging.info(f"Training set dimensions: {training_corpus.input_sequence.shape}")
    logging.info(f"Test set dimensions: {test_corpus.input_sequence.shape}")

    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def _calculate_nesting_depths(corpus):
    input_classes = corpus.input_sequence.argmax(axis=-1)
    opening_classes = {1, 3}
    closing_classes = {2, 4}
    depths = []
    for b in range(input_classes.shape[0]):
        depth = 0
        max_depth = 0
        for i in range(input_classes[b].shape[0]):
            if np.all(is_masked(corpus.input_sequence[b, i])):
                break
            if input_classes[b, i] in opening_classes:
                depth += 1
                max_depth = max(max_depth, depth)
            elif input_classes[b, i] in closing_classes:
                depths.append(depth)
                depth -= 1

    depth_counts = dict(collections.Counter(depths).most_common())
    max_depth = max(depths)
    logging.info(f"Max depth in corpus: {max_depth}")
    logging.info(f"Depth counts: {depth_counts}")


def _get_dyck_1_symbol_counts(corpus) -> Tuple[int, int, int]:
    # Masks (nans) become 0 here after argmax, but we ignore it since we count only 1's and 2's.
    input_classes = corpus.input_sequence.argmax(axis=-1)

    num_ends_of_sequence = np.sum(corpus.sample_weights).item()
    num_opening_brackets = np.dot(
        corpus.sample_weights, np.sum(input_classes == 1, axis=1)
    ).item()
    num_closing_brackets = np.dot(
        corpus.sample_weights, np.sum(input_classes == 2, axis=1)
    ).item()

    return num_ends_of_sequence, num_opening_brackets, num_closing_brackets


def get_dyck_1_target_probabs(corpus, prior) -> np.ndarray:
    input_classes = np.argmax(corpus.input_sequence, axis=-1).astype(np.float64)
    input_classes[~corpus.input_mask] = np.nan

    target_probabs = np.zeros_like(corpus.target_sequence)
    target_probabs[~corpus.input_mask] = np.nan

    for b in range(input_classes.shape[0]):
        open_brackets = 0
        for i in range(input_classes.shape[1]):
            if np.isnan(input_classes[b, i]):
                break
            if input_classes[b, i] == 0:
                target_probabs[b, i, 0] = 1 - prior
                target_probabs[b, i, 1] = prior
            elif input_classes[b, i] == 1:
                open_brackets += 1
                target_probabs[b, i, 2] = 1 - prior
                target_probabs[b, i, 1] = prior
            elif input_classes[b, i] == 2:
                open_brackets -= 1
                if open_brackets == 0:
                    target_probabs[b, i, 0] = 1 - prior
                    target_probabs[b, i, 1] = prior
                else:
                    target_probabs[b, i, 1] = prior
                    target_probabs[b, i, 2] = 1 - prior

    return target_probabs


def _make_dyck_n_corpus(
    batch_size: int,
    nesting_probab: float,
    n: int,
    max_sequence_length: Optional[int] = None,
    sort_by_length: bool = False,
):
    bracket_pairs = (
        ("[", "]"),
        ("(", ")"),
        ("{", "}"),
        ("<", ">"),
        ("⟦", "⟧"),
        ("〔", " 〕"),
    )
    single_nesting_probab = nesting_probab / n

    bracket_derivations = []
    for i in range(n):
        bracket_derivations.append(
            # e.g. `S -> ("[", S, "]", S)`.
            (
                (bracket_pairs[i][0], "S", bracket_pairs[i][1], "S",),
                single_nesting_probab,
            )
        )

    pcfg = {"S": tuple(bracket_derivations) + (("", 1 - nesting_probab),)}
    corpus = _make_corpus_from_pcfg(
        name=f"dyck_{n}__batch_{batch_size}__p_{nesting_probab}",
        pcfg=pcfg,
        batch_size=batch_size,
        max_derivation_length=max_sequence_length,
        sort_by_length=sort_by_length,
    )

    if n == 1:
        (
            num_ends_of_sequence,
            num_opening_brackets,
            num_closing_brackets,
        ) = _get_dyck_1_symbol_counts(corpus)

        optimal_d_given_g = (
            -1
            * (
                (num_ends_of_sequence * np.log2(1 - nesting_probab))
                + (num_opening_brackets * np.log2(nesting_probab))
                + (num_closing_brackets * np.log2(1 - nesting_probab))
            ).item()
        )
        corpus = dataclasses.replace(corpus, optimal_d_given_g=optimal_d_given_g)

    if n == 2:
        import manual_nets
        import network

        stack_net = manual_nets.make_emmanuel_dyck_2_network(
            nesting_probab=nesting_probab
        )
        stack_net = network.calculate_fitness(
            stack_net, optimize_for_feeding(corpus), config=_DEFAULT_CONFIG,
        )
        corpus = dataclasses.replace(
            corpus, optimal_d_given_g=stack_net.fitness.data_encoding_length
        )
        logging.info(f"Optimal |D:G|: {corpus.optimal_d_given_g:,.2f}")

    _calculate_nesting_depths(corpus)
    return corpus


def _get_sequence_strings(corpus) -> FrozenSet[Text]:
    unique_sequences = set()
    for b in range(corpus.input_sequence.shape[0]):
        seq = corpus.input_sequence[b]
        seq_str = str(np.argmax(seq, axis=-1).tolist())
        unique_sequences.add(seq_str)
    return frozenset(unique_sequences)


def make_dyck_n(
    batch_size: int,
    nesting_probab: float,
    n: int,
    max_sequence_length: Optional[int] = None,
    sort_by_length: bool = False,
) -> Corpus:
    training_corpus = _make_dyck_n_corpus(
        batch_size=batch_size,
        nesting_probab=nesting_probab,
        n=n,
        max_sequence_length=max_sequence_length,
        sort_by_length=sort_by_length,
    )
    test_corpus = _make_dyck_n_corpus(
        batch_size=50_000,
        nesting_probab=nesting_probab,
        n=n,
        max_sequence_length=max_sequence_length,
        sort_by_length=sort_by_length,
    )

    training_sequences = _get_sequence_strings(training_corpus)
    test_sequences = _get_sequence_strings(test_corpus)
    shared = training_sequences & test_sequences

    logging.info(
        f"Dyck-{n} Sequences shared between train and test: {len(shared)} ({len(shared)/len(test_sequences):.2f} of test)"
    )

    return dataclasses.replace(training_corpus, test_corpus=test_corpus)


def split_train_test(corpus: Corpus, train_ratio: float) -> Tuple[Corpus, Corpus]:
    batch_size = corpus.input_sequence.shape[0]
    train_size = math.floor(train_ratio * batch_size)
    shuffled_idxs = np.random.permutation(batch_size)
    train_idxs = sorted(shuffled_idxs[:train_size])
    test_idxs = sorted(shuffled_idxs[train_size:])
    return _split_corpus(corpus, batch_idxs_per_corpus=(train_idxs, test_idxs))


def _split_corpus(
    corpus: Corpus, batch_idxs_per_corpus: Tuple[np.ndarray, ...]
) -> Tuple[Corpus, ...]:
    new_corpora = []
    for batch_idxs in batch_idxs_per_corpus:
        max_sample_length = max(np.where(corpus.input_mask[batch_idxs])[-1]) + 1

        inputs = corpus.input_sequence[batch_idxs, :, :max_sample_length]
        targets = corpus.target_sequence[batch_idxs, :, :max_sample_length]
        target_mask = corpus.targets_mask[batch_idxs, :, :max_sample_length]
        input_mask = corpus.input_mask[batch_idxs, :max_sample_length]

        # TODO: recalculating indices for this is hard, need to change the data format.
        inputs_per_time_step = None

        if corpus.sample_weights:
            sample_weights = tuple(corpus.sample_weights[i] for i in batch_idxs)
        else:
            sample_weights = None

        new_corpora.append(
            Corpus(
                name=corpus.name,
                input_sequence=inputs,
                target_sequence=targets,
                input_values_per_time_step=inputs_per_time_step,
                input_mask=input_mask,
                targets_mask=target_mask,
                sample_weights=sample_weights,
            )
        )

    return tuple(new_corpora)


def split_to_mini_batches(
    corpus: Corpus, mini_batch_size: Optional[int]
) -> Tuple[Corpus, ...]:
    if mini_batch_size is None:
        return (corpus,)

    num_samples = corpus.input_sequence.shape[0]
    mini_batch_idxs = tuple(
        np.split(np.arange(num_samples), np.arange(0, num_samples, mini_batch_size)[1:])
    )

    return _split_corpus(corpus, mini_batch_idxs)
