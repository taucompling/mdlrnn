import collections
import dataclasses
import fractions
import functools
import itertools
import logging
import math
import pathlib
import pickle
import random
import zlib
from typing import Dict, FrozenSet, List, Optional, Text, Tuple, Union

import numba
import numpy as np
from numba import types
from scipy import special

import configuration
import corpora
import utils

_NODE_VALUE_LIMIT = np.int32(1_000_000)

LINEAR = 0
RELU = 1
SIGMOID = 2
_TANH = 3
SQUARE = 4
FLOOR = 5
UNSIGNED_STEP = 6
_GAUSSIAN = 7
_SINUS = 8
_INVERSE = 9
_ABS = 10
_COSINE = 11
MODULO_2 = 12
SHIFT_RIGHT = 13
MODULO_3 = 14

FORWARD_CONNECTION = 0
RECURRENT_CONNECTION = 1
_BIAS_CONNECTION = 2

SUMMATION_UNIT = 0
MULTIPLICATION_UNIT = 1

_CLASSES_DTYPE = np.uint16

_INF = float("inf")


def _linear(x):
    return x


@numba.jit(
    (types.Array(utils.NUMBA_FLOAT_DTYPE, ndim=1, layout="C", readonly=True),),
    nopython=True,
)
def _relu(x):
    return x * (x > 0)


@numba.jit(
    (types.Array(utils.NUMBA_FLOAT_DTYPE, ndim=1, layout="C", readonly=True),),
    nopython=True,
)
def _square(x):
    return x ** 2


_sigmoid = special.expit


def _tanh(x):
    return np.tanh(x)


def _floor(x):
    return np.floor(x)


def _gaussian(x):
    return np.exp(-np.multiply(x, x) / 2.0)


@numba.jit(
    (types.Array(utils.NUMBA_FLOAT_DTYPE, ndim=1, layout="C", readonly=True),),
    nopython=True,
)
def _unsigned_step(x):
    # Taken from Gaier & Ha.
    return 1.0 * (x > 0.0)


def _sinus(x):
    # Taken from Gaier & Ha.
    return np.sin(np.pi * x)


def _inverse(x):
    return -x


def _cosine(x):
    # Taken from Giaer & Ha.
    return np.cos(np.pi * x)


@numba.jit(
    (types.Array(utils.NUMBA_FLOAT_DTYPE, ndim=1, layout="C", readonly=True),),
    nopython=True,
)
def _modulo_2_numba(x):
    return np.mod(x, 2)


@numba.jit(
    (types.Array(utils.NUMBA_FLOAT_DTYPE, ndim=1, layout="C", readonly=True),),
    nopython=True,
)
def _modulo_3_numba(x):
    return np.mod(x, 3)


@numba.jit(
    (types.Array(utils.NUMBA_FLOAT_DTYPE, ndim=1, layout="C", readonly=True),),
    nopython=True,
)
def _shift_right(x):
    return np.floor_divide(x, 2)


_ACTIVATIONS = {
    LINEAR: _linear,
    RELU: _relu,
    SIGMOID: _sigmoid,
    _TANH: _tanh,
    SQUARE: _square,
    FLOOR: _floor,
    UNSIGNED_STEP: _unsigned_step,
    _GAUSSIAN: _gaussian,
    _SINUS: _sinus,
    _INVERSE: _inverse,
    _COSINE: _cosine,
    _ABS: np.abs,
    MODULO_2: _modulo_2_numba,
    MODULO_3: _modulo_3_numba,
    SHIFT_RIGHT: _shift_right,
}

_ACTIVATION_NAMES = {
    LINEAR: "linear",
    RELU: "relu",
    SIGMOID: "sigmoid",
    _TANH: "tanh",
    SQUARE: "square",
    FLOOR: "floor",
    _GAUSSIAN: "gaussian",
    UNSIGNED_STEP: "step",
    _SINUS: "sin",
    _INVERSE: "inverse",
    _ABS: "abs",
    _COSINE: "cosine",
    MODULO_2: "mod2",
    MODULO_3: "mod3",
    SHIFT_RIGHT: ">>",
}

_ACTIVATION_COSTS = {
    # Ordered by performance cost.
    LINEAR: 0,
    _INVERSE: 1,
    _ABS: 1,
    RELU: 2,
    SQUARE: 2,
    SIGMOID: 4,
    FLOOR: 4,
    _TANH: 8,
    MODULO_2: 8,
    MODULO_3: 8,
    UNSIGNED_STEP: 8,
    _GAUSSIAN: 16,
    _SINUS: 16,
    _COSINE: 16,
    SHIFT_RIGHT: 16,
}


def _eq_zero(x):
    # Using `isclose` for zero comparison because of precision-point float instabilities.
    # See `https://stackoverflow.com/questions/69256825/numpy-matrix-multiplication-instability-across-rows`.
    return np.isclose(x, 0.0)


def _binary_output_probabs(x: np.ndarray) -> np.ndarray:
    return np.clip(x, a_min=0, a_max=1.0)


def normalize_multiclass(x: np.ndarray) -> np.ndarray:
    x[x < 0] = 0
    sums = np.sum(x, axis=-1)
    # If output is all zeros, force probabs to be uniform.
    sums_eq_zero = _eq_zero(sums)
    zero_output_idxs = np.where(sums_eq_zero)
    x[zero_output_idxs] = 1
    sums[sums_eq_zero] = x.shape[-1]
    return x / np.expand_dims(sums, -1)


@dataclasses.dataclass(frozen=True)
class _Weight:
    sign: int
    numerator: int
    denominator: int


_Node = int
_Edge = Tuple[_Node, _Node]
_Connections = Dict[_Node, FrozenSet[_Node]]
_Weights = Dict[_Edge, _Weight]
_Biases = Dict[_Node, _Weight]
_Activations = Dict[_Node, int]


@dataclasses.dataclass(frozen=True)
class Fitness:
    mdl: float
    grammar_encoding_length: float
    data_encoding_length: float
    accuracy: float


@dataclasses.dataclass(frozen=True)
class Network:
    input_units_range: List[int]
    output_units_range: List[int]

    forward_weights: _Weights
    recurrent_weights: _Weights

    forward_connections: _Connections
    reverse_forward_connections: _Connections
    recurrent_connections: _Connections
    reverse_recurrent_connections: _Connections

    biases: _Biases
    activations: _Activations
    unit_types: _Activations

    fitness: Optional[Fitness] = None

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        tup = tuple(
            tuple(sorted(x.items()))
            for x in [
                self.forward_weights,
                self.recurrent_weights,
                self.forward_connections,
                self.recurrent_connections,
                self.biases,
                self.activations,
                self.unit_types,
            ]
        )
        return hash(tup)

    def __repr__(self):
        return to_string(self)


_IDENTITY_WEIGHT = _Weight(1, 1, 1)


class _InvalidNet(Exception):
    pass


def _make_random_weight() -> _Weight:
    numerator, denominator = _geometric_sample(shape=2, minimum=1)
    return _Weight(
        sign=np.random.choice([-1, 1]), numerator=numerator, denominator=denominator
    )


def _get_weight_real_value(w: _Weight) -> float:
    return w.sign * w.numerator / w.denominator


def _get_next_available_unit(active_units: FrozenSet[_Node]) -> int:
    """
    Units = {0,1,2,3} -> 4
    Units = {0,1,2,5} -> 3
    """
    try:
        return next(x for x in range(len(active_units)) if x not in active_units)
    except StopIteration:
        return len(active_units)


def topological_sort_through_time(
    net: Network,
) -> Tuple[FrozenSet[_Edge], Tuple[_Node, ...]]:
    units = get_units(net)
    loop_edges_for_forward_paths, topological_sort = run_dfs(
        units=units,
        reverse_connections=net.reverse_forward_connections,
        input_units=net.input_units_range,
        output_units=net.output_units_range,
    )

    units_with_outgoing_recurrent_edges = frozenset(
        node
        for node in net.recurrent_connections
        if len(net.recurrent_connections[node])
    )
    units_with_outgoing_recurrent_not_in_topo_sort = list(
        units_with_outgoing_recurrent_edges - frozenset(topological_sort)
    )
    loop_edges_including_recurrent_paths, recurrent_topological_sort = run_dfs(
        units=units,
        reverse_connections=net.reverse_forward_connections,
        input_units=net.input_units_range,
        output_units=units_with_outgoing_recurrent_not_in_topo_sort,
    )

    topological_sort_including_recurrent = topological_sort + tuple(
        x for x in recurrent_topological_sort if x not in frozenset(topological_sort)
    )

    loop_edges = loop_edges_for_forward_paths | loop_edges_including_recurrent_paths

    return loop_edges, topological_sort_including_recurrent


def run_dfs(
    units: FrozenSet[_Node],
    reverse_connections: _Connections,
    input_units: List[_Node],
    output_units: List[_Node],
) -> Tuple[FrozenSet[_Edge], Tuple[_Node, ...]]:
    """ Runs DFS (iterative) on a directed graph which may contain cycles.
    Returns a set of back edges and a topological sort of nodes in the graph as if it didn't contain cycles.
    """
    reverse_connections = _dict_copy(reverse_connections)

    active_units = units
    entry_node = _get_next_available_unit(active_units)
    sink_node = _get_next_available_unit(active_units | {entry_node})

    reverse_connections[entry_node] = frozenset(output_units)
    for input_node in input_units:
        if input_node not in reverse_connections:
            reverse_connections[input_node] = frozenset()
        reverse_connections[input_node] |= {sink_node}

    loop_edges = set()
    topological_sort = []

    black = set()
    gray = {entry_node}
    stack = [entry_node]

    while stack:
        node = stack[0]
        neighbors = reverse_connections.get(node, set())
        recurse = False
        for neighbor in neighbors:
            if neighbor in gray:
                loop_edges.add((neighbor, node))
            elif neighbor not in black:
                stack.insert(0, neighbor)
                gray.add(neighbor)
                recurse = True
                break
        if recurse:
            continue

        black.add(node)
        gray.remove(node)
        stack.pop(0)
        if node not in (entry_node, sink_node):
            topological_sort.append(node)

    return frozenset(loop_edges), tuple(topological_sort)


def _geometric_sample(shape=None, minimum: int = 0, dtype=np.int):
    return np.floor(-np.log2(np.random.random(shape))).astype(dtype) + minimum


def _ceil_log2(x):
    return np.ceil(np.log2(x))


@functools.lru_cache(maxsize=100_000)
def _ceil_log2_python(x):
    return math.ceil(math.log2(x))


def _reverse_set_dict(d: Dict) -> Dict:
    new_dict = collections.defaultdict(frozenset)
    for x, vals in d.items():
        for y in vals:
            new_dict[y] |= {x}
    return dict(new_dict)


@functools.lru_cache(maxsize=100_000)
def _int_to_binary_string(n, sequence_length=None) -> Text:
    binary = f"{n:b}"
    if sequence_length is not None:
        binary = ("0" * (sequence_length - len(binary))) + binary
    return binary


@functools.lru_cache(maxsize=100_000)
def _integer_encoding(n) -> Text:
    # Based on Vitanyi & Li's self-delimiting encoding: `unary_encoding(log2_N) + '0' + binary_encoding(N)`.
    if n == 0:
        return "0"
    binary = _int_to_binary_string(n)
    return ("1" * len(binary)) + "0" + binary


@functools.lru_cache(maxsize=100_000)
def _get_weight_encoding(weight: _Weight) -> Text:
    encoding = ""
    encoding += "0" if weight.sign == -1 else "1"
    encoding += _integer_encoding(weight.numerator)
    encoding += _integer_encoding(weight.denominator)
    return encoding


def _get_bitsring_encoding(net: Network, allowed_activations: Tuple[int, ...]) -> Text:
    num_units = get_num_units(net)

    single_activation_encoding_length = _ceil_log2_python(len(allowed_activations))

    # Using fixed-length encoding to prevent a bias for lower-id units.
    unit_id_encoding_length = _ceil_log2_python(num_units)

    # Specify how many units we need to parse.
    encoding = _integer_encoding(num_units)

    for unit in range(num_units):
        unit_forward_targets = net.forward_connections.get(unit, ())
        unit_recurrent_targets = net.recurrent_connections.get(unit, ())

        # Specify how many connections we need to parse for this unit.
        encoding += _integer_encoding(
            len(unit_forward_targets) + len(unit_recurrent_targets)
        )

        for forward_target_unit in unit_forward_targets:
            encoding += "0"  # "Forward" bit.
            encoding += _int_to_binary_string(
                forward_target_unit, sequence_length=unit_id_encoding_length
            )
            encoding += _get_weight_encoding(
                net.forward_weights[(unit, forward_target_unit)]
            )

        for recurrent_target_unit in unit_recurrent_targets:
            encoding += "1"  # "Recurrent" bit.
            encoding += _int_to_binary_string(
                recurrent_target_unit, sequence_length=unit_id_encoding_length
            )
            encoding += _get_weight_encoding(
                net.recurrent_weights[(unit, recurrent_target_unit)]
            )

        # Unit type.
        encoding += {SUMMATION_UNIT: "1", MULTIPLICATION_UNIT: "0"}[
            net.unit_types[unit]
        ]

        # Unit activation and delimiter.
        encoding += _int_to_binary_string(
            net.activations[unit], sequence_length=single_activation_encoding_length
        )
        encoding += ("1" * _ACTIVATION_COSTS[net.activations[unit]]) + "0"

        # Unit bias.
        if unit in net.biases:
            encoding += _get_weight_encoding(net.biases[unit])
        else:
            # "Skip".
            encoding += "00"

    return encoding


def get_encoding_length(
    net: Network, allowed_activations: Tuple[int, ...], compress_encoding: bool
) -> float:
    encoding_bitstring = _get_bitsring_encoding(
        net=net, allowed_activations=allowed_activations
    )
    if compress_encoding:
        bytes_ = zlib.compress(encoding_bitstring.encode("utf-8"))
        return len(bytes_)

    return len(encoding_bitstring)


def _try_early_stop(
    sequence_length: int,
    targets_mask: Optional[np.ndarray],
    input_mask: Optional[np.ndarray],
    time_step: int,
    time_step_outputs: np.ndarray,
):
    """Stop sequence feeding in case a target is given a categorical 0.0 probab, causing D:G => infinity. """
    # Using log to scale slowly with sequence length.
    early_stop_limit = _ceil_log2_python(sequence_length)

    if targets_mask is None or input_mask is None or time_step >= early_stop_limit:
        return

    time_step_targets_mask = targets_mask[:, time_step]
    time_step_input_mask = input_mask[:, time_step]

    early_stop = False

    if time_step_targets_mask.shape[-1] == 1:
        # Binary outputs. Mask means target == 1.
        targets_mask = time_step_targets_mask.squeeze(axis=-1)
        target_outputs = np.copy(time_step_outputs.squeeze(axis=-1))
        target_outputs[~targets_mask] = 1 - target_outputs[~targets_mask]
        target_outputs = target_outputs[time_step_input_mask]
        if not all(target_outputs):
            early_stop = True

    else:
        target_outputs = time_step_outputs[time_step_targets_mask]
        target_outputs_leq_zero = target_outputs[time_step_input_mask] <= 0
        if any(target_outputs_leq_zero):
            # Rule out cases of all-zeros outputs, which return a uniform probability.
            positive_outputs = time_step_outputs[time_step_input_mask] > 0
            early_stop = np.any(positive_outputs[target_outputs_leq_zero])

    if early_stop:
        raise _InvalidNet(f"Early stop on step {time_step}")


def _calculate_inputs_product(weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    # Weighted inputs.
    multiplication = np.multiply(weights, inputs.T)
    # Product of weighted inputs.
    return np.prod(multiplication, axis=1)


@numba.jit(
    (types.Array(utils.NUMBA_FLOAT_DTYPE, ndim=1, layout="C"), types.int32),
    nopython=True,
)
def _clip(arr, limit):
    arr[arr < -limit] = -limit
    arr[arr > limit] = limit


@numba.jit(
    (
        types.Array(utils.NUMBA_FLOAT_DTYPE, ndim=1, layout="C"),
        types.Array(utils.NUMBA_FLOAT_DTYPE, ndim=2, layout="C", readonly=True),
    ),
    nopython=True,
)
def _scalar_matmul_numba(weights, inputs) -> np.ndarray:
    return (weights[0] * inputs).reshape((-1,))


@numba.jit(
    (types.Array(utils.NUMBA_FLOAT_DTYPE, ndim=2, layout="C", readonly=True),),
    nopython=True,
)
def _scalar_negative_numba(inputs: np.ndarray) -> np.ndarray:
    return np.negative(inputs).reshape((-1,))


def _calculate_incoming_value_by_unit_type(
    weights: np.ndarray, inputs: np.ndarray, unit_type: int
) -> np.ndarray:
    if unit_type == SUMMATION_UNIT or inputs.shape[0] == 1:
        if weights.shape[0] == 1:
            if weights[0] == 1.0:
                # TODO: Not using the faster reshape((-1,) since it returns a view which currently creates problems downstream.
                return inputs.flatten()
            if weights[0] == -1.0:
                return _scalar_negative_numba(inputs)
            return _scalar_matmul_numba(weights, inputs)
        return np.matmul(weights, inputs)
    elif unit_type == MULTIPLICATION_UNIT:
        return _calculate_inputs_product(weights, inputs)
    raise ValueError(f"Invalid unit type {unit_type}")


def _calculate_forward_inputs(
    node: int,
    node_values: Dict,
    unit_type: int,
    incoming_units: List[int],
    incoming_weights: np.ndarray,
    is_input_unit: bool,
):
    if not incoming_units:
        return

    if len(incoming_units) == 1:
        incoming_unit_values = node_values[incoming_units[0]].reshape((1, -1))
    else:
        incoming_unit_values = np.array([node_values[x] for x in incoming_units])

    incoming_value = _calculate_incoming_value_by_unit_type(
        incoming_weights, incoming_unit_values, unit_type
    )

    if is_input_unit:
        # Only input units can have non-zero values here, intermediate nodes are guaranteed to be 0 because of topological order.
        # Creating new value here (instead of +=) since input values are immutable.
        if unit_type == SUMMATION_UNIT:
            node_values[node] = node_values[node] + incoming_value
        elif unit_type == MULTIPLICATION_UNIT:
            node_values[node] = node_values[node] * incoming_value
    else:
        node_values[node] = incoming_value


def _calculate_recurrent_inputs(
    node: int,
    current_step_node_values: Dict,
    previous_step_node_values: Dict,
    incoming_recurrent_units: List[int],
    incoming_recurrent_weights: np.ndarray,
    unit_type: int,
    batch_size: int,
):
    if len(incoming_recurrent_units) == 1:
        recurrent_slice = previous_step_node_values[
            incoming_recurrent_units[0]
        ].reshape((1, -1))
    else:
        recurrent_vals = []
        for x in incoming_recurrent_units:
            try:
                recurrent_vals.append(previous_step_node_values[x])
            except KeyError:
                recurrent_vals.append(np.zeros(batch_size, dtype=utils.FLOAT_DTYPE))
        recurrent_slice = np.array(recurrent_vals)

    incoming_value = _calculate_incoming_value_by_unit_type(
        incoming_recurrent_weights, recurrent_slice, unit_type
    )

    # `ValueError` is raised if unit is an input unit, whose values are immutable, or for a unit
    # with recurrent inputs but no forward inputs, in which case values are the default immutable zeros.
    if unit_type == SUMMATION_UNIT:
        try:
            current_step_node_values[node] += incoming_value
        except ValueError:
            current_step_node_values[node] = (
                current_step_node_values[node] + incoming_value
            )
    elif unit_type == MULTIPLICATION_UNIT:
        try:
            current_step_node_values[node] *= incoming_value
        except ValueError:
            current_step_node_values[node] = (
                current_step_node_values[node] * incoming_value
            )
    else:
        raise ValueError("Invalid unit type")


def _calculate_bias(node: int, node_bias: _Weight, node_values: Dict, unit_type: int):
    node_values[node] = node_values[node] + _get_weight_real_value(node_bias)


def _apply_activation(node: int, activation: int, node_values: Dict):
    node_values[node] = _ACTIVATIONS[activation](node_values[node])


def _get_incoming_units_and_weights(
    net: Network,
    topological_sort: Tuple[_Node, ...],
    loop_edges: FrozenSet[_Edge],
    recurrent_connections: bool,
) -> Tuple[Dict, Dict, Dict, Dict]:
    incoming_forward_units_per_node = {}
    incoming_forward_weights_per_node = {}
    incoming_recurrent_units_per_node = {}
    incoming_recurrent_weights_per_node = {}

    for node in topological_sort:
        incoming_forward_units_per_node[node] = list(
            x
            for x in net.reverse_forward_connections.get(node, [])
            if (x, node) not in loop_edges
        )
        forward_weights = np.array(
            [
                _get_weight_real_value(net.forward_weights[(from_node, node)])
                for from_node in incoming_forward_units_per_node[node]
            ],
            dtype=utils.FLOAT_DTYPE,
        )
        incoming_forward_weights_per_node[node] = forward_weights

        if not (
            recurrent_connections
            and node in net.reverse_recurrent_connections
            and net.reverse_recurrent_connections[node]
        ):
            continue

        incoming_recurrent_units = net.reverse_recurrent_connections[node]
        if incoming_recurrent_units:
            incoming_recurrent_units_per_node[node] = list(
                net.reverse_recurrent_connections[node]
            )

            recurrent_weights = np.array(
                [
                    _get_weight_real_value(net.recurrent_weights[(from_node, node)])
                    for from_node in incoming_recurrent_units_per_node[node]
                ],
                dtype=utils.FLOAT_DTYPE,
            )
            incoming_recurrent_weights_per_node[node] = recurrent_weights

    return (
        incoming_forward_units_per_node,
        incoming_forward_weights_per_node,
        incoming_recurrent_units_per_node,
        incoming_recurrent_weights_per_node,
    )


def predict_probabs(
    net: Network,
    input_sequence: np.ndarray,
    recurrent_connections: bool,
    truncate_large_values: bool,
    softmax_outputs: bool,
    inputs_per_time_step: Optional[Dict[int, np.ndarray]] = None,
    input_mask: Optional[np.ndarray] = None,
    targets_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    sequence_outputs = feed_sequence(
        net,
        input_sequence,
        recurrent_connections=recurrent_connections,
        truncate_large_values=truncate_large_values,
        inputs_per_time_step=inputs_per_time_step,
        input_mask=input_mask,
        targets_mask=targets_mask,
    )

    output_size = len(net.output_units_range)
    if output_size == 1:
        sequence_output_probabs = _binary_output_probabs(sequence_outputs)
    else:
        if softmax_outputs:
            sequence_output_probabs = special.softmax(sequence_outputs, axis=-1)
        else:
            sequence_output_probabs = normalize_multiclass(sequence_outputs)

    return sequence_output_probabs


def _truncate_large_values(
    truncate_large_values: bool, node_values: Dict[int, np.ndarray], current_node: int
):
    if (
        truncate_large_values
        # No point in clipping input values (array immutability indicates input values).
        and node_values[current_node].flags.writeable
    ):
        _clip(node_values[current_node], _NODE_VALUE_LIMIT)


def feed_sequence(
    net: Network,
    input_sequence: np.ndarray,
    recurrent_connections: bool,
    truncate_large_values: bool,
    inputs_per_time_step: Optional[Dict[int, np.ndarray]] = None,
    input_mask: Optional[np.ndarray] = None,
    targets_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    batch_size = input_sequence.shape[0]
    sequence_length = input_sequence.shape[1]

    loop_edges, topological_sort = topological_sort_through_time(net)

    (
        incoming_forward_units_per_node,
        incoming_forward_weights_per_node,
        incoming_recurrent_units_per_node,
        incoming_recurrent_weights_per_node,
    ) = _get_incoming_units_and_weights(
        net, topological_sort, loop_edges, recurrent_connections
    )

    sequence_outputs = []
    previous_step_node_values = {}

    initial_zeros = np.zeros(batch_size, dtype=utils.FLOAT_DTYPE)
    initial_zeros.flags.writeable = False

    for time_step in range(sequence_length):
        node_values = {}
        for input_unit in net.input_units_range:
            if inputs_per_time_step:
                input_val = inputs_per_time_step[input_unit][time_step]
            else:
                input_val = np.ascontiguousarray(
                    input_sequence[:, time_step, input_unit]
                )
            node_values[input_unit] = input_val

        # Fill node values using dynamic programming based on topological sort.

        for current_node in topological_sort:
            if current_node not in node_values:
                node_values[current_node] = initial_zeros

            _calculate_forward_inputs(
                node=current_node,
                node_values=node_values,
                unit_type=net.unit_types[current_node],
                incoming_units=incoming_forward_units_per_node[current_node],
                incoming_weights=incoming_forward_weights_per_node[current_node],
                is_input_unit=current_node in net.input_units_range,
            )

            if (
                recurrent_connections
                and time_step > 0
                and current_node in incoming_recurrent_units_per_node
            ):
                _calculate_recurrent_inputs(
                    node=current_node,
                    current_step_node_values=node_values,
                    previous_step_node_values=previous_step_node_values,
                    incoming_recurrent_units=incoming_recurrent_units_per_node[
                        current_node
                    ],
                    incoming_recurrent_weights=incoming_recurrent_weights_per_node[
                        current_node
                    ],
                    unit_type=net.unit_types[current_node],
                    batch_size=batch_size,
                )

            if current_node in net.biases:
                _calculate_bias(
                    node=current_node,
                    node_bias=net.biases[current_node],
                    node_values=node_values,
                    unit_type=net.unit_types[current_node],
                )

            _truncate_large_values(truncate_large_values, node_values, current_node)
            _apply_activation(current_node, net.activations[current_node], node_values)

        current_time_step_outputs = np.array(
            [node_values[x] for x in net.output_units_range]
        ).T
        sequence_outputs.append(current_time_step_outputs)

        _try_early_stop(
            sequence_length=sequence_length,
            targets_mask=targets_mask,
            input_mask=input_mask,
            time_step=time_step,
            time_step_outputs=current_time_step_outputs,
        )

        previous_step_node_values = node_values

    return np.array(sequence_outputs).swapaxes(0, 1)


def feed_language_model(
    net: Network,
    symbol_to_class: Dict[Text, int],
    sample: bool,
    prefix: Tuple[Text, ...],
    end_of_sequence_class: int,
    softmax_outputs: bool,
    truncate_large_values: bool,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], np.ndarray]:
    prefix_inputs = np.zeros((1, len(prefix), len(symbol_to_class)))
    for i, symbol in enumerate(prefix):
        prefix_inputs[0, i, symbol_to_class[symbol]] = 1.0

    inputs = np.copy(prefix_inputs)
    output_classes = [symbol_to_class[x] for x in prefix]

    while True:
        outputs = predict_probabs(
            net,
            inputs,
            recurrent_connections=True,
            truncate_large_values=truncate_large_values,
            softmax_outputs=softmax_outputs,
        )

        if sample:
            next_class = np.random.choice(len(symbol_to_class), p=outputs[0, -1])
        else:
            next_class = np.argmax(outputs[0, -1])

        output_classes.append(next_class)

        if next_class == end_of_sequence_class:
            break

        next_step_input = np.zeros((1, 1, outputs.shape[-1]))
        next_step_input[0, 0, next_class] = 1.0

        inputs = np.concatenate([inputs, next_step_input], axis=1)

    input_classes = tuple(np.argmax(inputs, axis=-1).squeeze(axis=0))

    return (
        tuple(output_classes),
        input_classes,
        np.concatenate([prefix_inputs[:, 1:], outputs[:, len(prefix) - 1 :]], axis=1),
    )


def _calculate_accuracy(
    predicted_classes: np.ndarray,
    target_classes: np.ndarray,
    input_mask: Optional[np.ndarray],
) -> float:
    if input_mask is not None:
        predicted_classes = predicted_classes[input_mask]
        target_classes = target_classes[input_mask]
    correct_predictions = predicted_classes == target_classes
    if logging.getLogger().level == logging.DEBUG:
        logging.debug(
            f"Sequences with wrong predictions: {list(set(np.where(~correct_predictions)[0]))}"
        )
    return (np.sum(correct_predictions) / predicted_classes.size).item()


def calculate_deterministic_accuracy(
    net: Network, corpus: corpora.Corpus, config: configuration.SimulationConfig
) -> float:
    output_probabs = predict_probabs(
        net,
        input_sequence=corpus.input_sequence,
        recurrent_connections=config.recurrent_connections,
        truncate_large_values=config.truncate_large_values,
        softmax_outputs=config.softmax_outputs,
        inputs_per_time_step=corpus.input_values_per_time_step,
        input_mask=corpus.input_mask,
        targets_mask=corpus.targets_mask,
    )

    target_sequence = corpus.target_sequence
    deterministic_steps_mask = corpus.deterministic_steps_mask

    rows_repeated_by_weights = None
    if corpus.sample_weights:
        rows_repeated_by_weights = list(
            itertools.chain(
                *[[row] * weight for row, weight in enumerate(corpus.sample_weights)]
            )
        )
        output_probabs = output_probabs[rows_repeated_by_weights]
        deterministic_steps_mask = deterministic_steps_mask[rows_repeated_by_weights]
        target_sequence = target_sequence[rows_repeated_by_weights]

    deterministic_probabs = output_probabs[deterministic_steps_mask]
    predicted_classes = deterministic_probabs.argmax(axis=-1)

    target_classes = target_sequence[deterministic_steps_mask].argmax(axis=-1)

    original_rows = np.where(deterministic_steps_mask)[0]
    rows_with_errors = np.where(predicted_classes != target_classes)[0]
    original_rows_with_errors = frozenset([original_rows[i] for i in rows_with_errors])

    if rows_repeated_by_weights:
        original_rows_with_errors = frozenset(
            rows_repeated_by_weights[i] for i in original_rows_with_errors
        )

    if original_rows_with_errors:
        logging.debug(
            f"Deterministic accuracy: rows with errors: {original_rows_with_errors}"
        )

    accuracy = np.sum(predicted_classes == target_classes) / target_classes.size
    return accuracy.item()


def _get_sequence_encoding_length(
    net: Network,
    input_sequence: np.ndarray,
    target_sequence: np.ndarray,
    recurrent_connections: bool,
    truncate_large_values: bool,
    softmax_outputs: bool,
    inputs_per_time_step: Optional[Dict],
    inputs_mask: Optional[np.ndarray],
    targets_mask: Optional[np.ndarray],
    sample_weights: Optional[Tuple[int, ...]],
) -> Tuple[float, float]:
    output_probabs = predict_probabs(
        net=net,
        input_sequence=input_sequence,
        recurrent_connections=recurrent_connections,
        truncate_large_values=truncate_large_values,
        softmax_outputs=softmax_outputs,
        inputs_per_time_step=inputs_per_time_step,
        input_mask=inputs_mask,
        targets_mask=targets_mask,
    )

    output_size = target_sequence.shape[-1]

    if output_size == 1:
        # Binary outputs.
        target_classes = target_sequence
        predicted_classes = (output_probabs > 0.5).astype(_CLASSES_DTYPE)
        output_probabs[target_classes == 0] = 1 - output_probabs[target_classes == 0]
        target_classes_probabs = output_probabs
    else:
        # One-hot outputs.
        target_classes = np.argmax(target_sequence, axis=-1)

        if targets_mask is not None:
            target_classes_probabs = output_probabs[targets_mask].reshape(
                output_probabs.shape[:2]
            )
        else:
            # TODO: Legacy. We should always compute target masks.
            outputs_squeezed = output_probabs.reshape(-1, output_size)
            target_classes_probabs = outputs_squeezed[
                range(outputs_squeezed.shape[0]), target_classes.flatten()
            ]
            target_classes_probabs = target_classes_probabs.reshape(
                output_probabs.shape[:2]
            )

    if inputs_mask is not None:
        # Set to 1 so log2(1) = 0 won't count in encoding length.
        target_classes_probabs[~inputs_mask] = 1

    if np.any(target_classes_probabs == 0.0) or np.any(
        np.isnan(target_classes_probabs)
    ):
        raise _InvalidNet("Probability 0 or nan assigned to target class")

    log_probabilities = -np.log2(target_classes_probabs)

    if sample_weights:
        encoding_length_per_sample = np.sum(log_probabilities, axis=1)
        encoding_length = float(
            np.dot(encoding_length_per_sample.squeeze(), sample_weights)
        )
    else:
        encoding_length = float(np.sum(log_probabilities))

    if output_size > 1:
        predicted_classes = output_probabs.argmax(axis=-1)

    accuracy = _calculate_accuracy(
        predicted_classes=predicted_classes,
        target_classes=target_classes,
        input_mask=inputs_mask,
    )
    return encoding_length, accuracy


def _get_data_given_g_encoding_length(
    net: Network, corpus: corpora.Corpus, config: configuration.SimulationConfig
) -> Tuple[float, float]:
    # TODO: split once.
    mini_batches = corpora.split_to_mini_batches(
        corpus, mini_batch_size=config.mini_batch_size
    )

    batch_encoding_lengths = []
    batch_accuracies = []
    for mini_batch in mini_batches:
        try:
            batch_encoding_length, batch_accuracy = _get_sequence_encoding_length(
                net=net,
                input_sequence=mini_batch.input_sequence,
                target_sequence=mini_batch.target_sequence,
                recurrent_connections=config.recurrent_connections,
                truncate_large_values=config.truncate_large_values,
                softmax_outputs=config.softmax_outputs,
                inputs_per_time_step=mini_batch.input_values_per_time_step,
                inputs_mask=mini_batch.input_mask,
                targets_mask=mini_batch.targets_mask,
                sample_weights=mini_batch.sample_weights,
            )

            batch_encoding_lengths.append(batch_encoding_length)
            batch_accuracies.append(batch_accuracy)

        except _InvalidNet as e:
            logging.debug(f"Invalid network: {str(e)}")
            return _INF, 0.0

    encoding_length = np.sum(batch_encoding_lengths).item()
    batch_sizes = tuple(x.input_sequence.shape[0] for x in mini_batches)
    accuracy = np.average(batch_accuracies, weights=batch_sizes).item()

    return encoding_length, accuracy


def calculate_fitness(
    net: Network, corpus: corpora.Corpus, config: configuration.SimulationConfig
) -> Network:
    if net.fitness is not None:
        return net

    grammar_encoding_length = config.grammar_multiplier * get_encoding_length(
        net=net,
        allowed_activations=config.allowed_activations,
        compress_encoding=config.compress_grammar_encoding,
    )
    data_encoding_length, accuracy = _get_data_given_g_encoding_length(
        net, corpus, config
    )

    data_encoding_length *= config.data_given_grammar_multiplier

    return dataclasses.replace(
        net,
        fitness=Fitness(
            mdl=grammar_encoding_length + data_encoding_length,
            grammar_encoding_length=grammar_encoding_length,
            data_encoding_length=data_encoding_length,
            accuracy=accuracy,
        ),
    )


def get_num_units(net) -> int:
    return len(net.activations)


def get_total_connections(net: Network, include_biases: bool) -> int:
    num_connections = len(net.forward_weights) + len(net.recurrent_weights)
    if include_biases:
        num_connections += len(net.biases)
    return num_connections


def get_units(net: Network) -> FrozenSet[_Node]:
    return frozenset(net.activations)


def get_loop_edges(net: Network) -> FrozenSet[_Edge]:
    forward_loop_edges, _ = run_dfs(
        units=get_units(net),
        reverse_connections=net.reverse_forward_connections,
        input_units=net.input_units_range,
        output_units=net.output_units_range,
    )
    return forward_loop_edges


def fix_loops(net: Network) -> Network:
    loop_edges = get_loop_edges(net)
    if not loop_edges:
        return net
    new_forward_connections = _dict_copy(net.forward_connections)
    new_reverse_forward_connections = _dict_copy(net.reverse_forward_connections)
    new_forward_weights = _dict_copy(net.forward_weights)

    for from_node, to_node in loop_edges:
        try:
            new_forward_connections[from_node] -= {to_node}
            new_reverse_forward_connections[to_node] -= {from_node}
            del new_forward_weights[(from_node, to_node)]
        except KeyError:
            pass

    return dataclasses.replace(
        net,
        forward_connections=new_forward_connections,
        forward_weights=new_forward_weights,
        reverse_forward_connections=new_reverse_forward_connections,
    )


def make_random_net(
    input_size: int,
    output_size: int,
    allowed_activations: Tuple[int, ...],
    start_smooth: bool,
) -> Network:
    input_units_range = list(range(input_size))
    output_units_range = list(range(input_size, input_size + output_size))

    forward_connections = {}
    forward_weights = {}

    # Ensure each output unit has an incoming connection.
    for output_unit in output_units_range:
        incoming_unit = random.choice(input_units_range)
        if incoming_unit not in forward_connections:
            forward_connections[incoming_unit] = frozenset()
        forward_connections[incoming_unit] |= {output_unit}
        forward_weights[(incoming_unit, output_unit)] = _make_random_weight()

    num_units = input_size + output_size

    activations = {}
    for i in range(num_units):
        if i in set(input_units_range):
            activation = LINEAR
        elif start_smooth:
            activation = SIGMOID
        else:
            activation = random.choice(allowed_activations)
        activations[i] = activation

    unit_types = {i: SUMMATION_UNIT for i in range(num_units)}

    return Network(
        input_units_range=input_units_range,
        output_units_range=output_units_range,
        forward_weights=forward_weights,
        recurrent_weights={},
        forward_connections=forward_connections,
        reverse_forward_connections=_reverse_set_dict(forward_connections),
        recurrent_connections={},
        reverse_recurrent_connections={},
        biases={},
        activations=activations,
        unit_types=unit_types,
    )


def invalidate_fitness(net: Network) -> Network:
    return dataclasses.replace(net, fitness=None)


def get_forward_weights(net: Network) -> np.ndarray:
    return np.array([_get_weight_real_value(w) for w in net.forward_weights.values()])


def get_recurrent_weights(net: Network) -> np.ndarray:
    return np.array([_get_weight_real_value(w) for w in net.recurrent_weights.values()])


def get_connections_and_weights_by_edge_type(
    net: Network, edge_type: int
) -> Tuple[_Connections, _Connections, _Weights]:
    if edge_type == FORWARD_CONNECTION:
        return (
            net.forward_connections,
            net.reverse_forward_connections,
            net.forward_weights,
        )
    elif edge_type == RECURRENT_CONNECTION:
        return (
            net.recurrent_connections,
            net.reverse_recurrent_connections,
            net.recurrent_weights,
        )
    else:
        raise ValueError


def _get_weight(net: Network, edge_type: int, from_node: int, to_node: int) -> float:
    _, _, weights = get_connections_and_weights_by_edge_type(net, edge_type)
    weight = weights[(from_node, to_node)]
    return weight.sign * weight.numerator / weight.denominator


def _get_num_hidden_units(net: Network) -> int:
    return get_num_units(net) - len(net.input_units_range) - len(net.output_units_range)


def _get_hidden_units(net: Network) -> FrozenSet[_Node]:
    return frozenset(range(net.output_units_range[-1] + 1, get_num_units(net)))


def _get_number_of_possible_connections(net: Network, edge_type: int) -> int:
    # Number of legal connections, excluding into inputs and out from outputs, and forward self-loops.
    num_hidden_units = _get_num_hidden_units(net)
    num_inputs = len(net.input_units_range)
    num_outputs = len(net.output_units_range)
    possible_outgoing_from_inputs = num_inputs * (num_hidden_units + num_outputs)
    if edge_type == FORWARD_CONNECTION:
        possible_outgoing_from_hidden = num_hidden_units * (
            (num_hidden_units - 1) + num_outputs
        )
    else:
        # Recurrent self-loops are allowed.
        possible_outgoing_from_hidden = num_hidden_units * (
            num_hidden_units + num_outputs
        )
    return possible_outgoing_from_inputs + possible_outgoing_from_hidden


def mutate(net: Network, config: configuration.SimulationConfig) -> Network:
    num_units = get_num_units(net)
    total_connections_including_biases = get_total_connections(net, include_biases=True)
    total_connections = get_total_connections(net, include_biases=False)
    num_biases = len(net.biases)
    num_hidden_units = _get_num_hidden_units(net)
    num_outputs = len(net.output_units_range)

    max_total_connections = _get_number_of_possible_connections(net, FORWARD_CONNECTION)
    if config.recurrent_connections:
        max_total_connections += _get_number_of_possible_connections(
            net, RECURRENT_CONNECTION
        )

    mutation_conditions = (
        (
            _add_random_unit,
            num_units < config.max_network_units and total_connections > 0,
        ),
        (_remove_random_unit, num_hidden_units > 0),
        (
            _mutate_activation,
            (num_hidden_units + num_outputs) > 0
            and len(config.allowed_activations) > 1,
        ),
        (
            _mutate_unit_type,
            len(config.allowed_unit_types) > 1 and (num_hidden_units + num_outputs) > 0,
        ),
        (_add_random_connection, total_connections < max_total_connections),
        (_remove_random_connection, total_connections > 0),
        (_add_random_bias, num_biases < (num_hidden_units + num_outputs)),
        (_remove_random_bias, num_biases > 0),
        (_mutate_weight, total_connections_including_biases > 0),
    )
    valid_mutations = tuple(filter(lambda x: x[1], mutation_conditions))
    mutation_func = random.choice(valid_mutations)[0]

    logging.debug(f"Applying mutation {mutation_func.__name__}")

    mutated_net = mutation_func(net, config)
    return invalidate_fitness(mutated_net)


def _add_unit(
    net: Network,
    edge: _Edge,
    edge_type: int,
    activation: int,
    unit_type: int,
    source_weight: _Weight,
    target_weight: _Weight,
) -> Network:
    # Split an existing connection, add a unit in the middle.
    # If the connection is recurrent, feed the new unit with the same recurrent edge,
    # and connect to target via a forward connection. This prevents creating a recurrent connection that goes 2 time steps back.

    new_activations = _dict_copy(net.activations)
    new_unit_types = _dict_copy(net.unit_types)
    new_forward_connections = _dict_copy(net.forward_connections)
    new_reverse_forward_connections = _dict_copy(net.reverse_forward_connections)
    new_forward_weights = _dict_copy(net.forward_weights)

    new_recurrent_connections = _dict_copy(net.recurrent_connections)
    new_reverse_recurrent_connections = _dict_copy(net.reverse_recurrent_connections)
    new_recurrent_weights = _dict_copy(net.recurrent_weights)

    new_unit = _get_next_available_unit(get_units(net))

    source_unit, target_unit = edge

    new_reverse_forward_connections[target_unit] = new_reverse_forward_connections.get(
        target_unit, frozenset()
    ) | {new_unit}

    new_forward_connections[new_unit] = frozenset([target_unit])
    new_forward_weights[(new_unit, target_unit)] = target_weight

    if edge_type == FORWARD_CONNECTION:
        new_connections = new_forward_connections
        new_reverse_connections = new_reverse_forward_connections
        new_weights = new_forward_weights
    else:
        new_connections = new_recurrent_connections
        new_reverse_connections = new_reverse_recurrent_connections
        new_weights = new_recurrent_weights

    new_connections[source_unit] -= {target_unit}
    new_connections[source_unit] |= {new_unit}

    new_reverse_connections[target_unit] -= {source_unit}
    new_reverse_connections[new_unit] = frozenset([source_unit])
    del new_weights[(source_unit, target_unit)]
    new_weights[(source_unit, new_unit)] = source_weight

    new_activations[new_unit] = activation
    new_unit_types[new_unit] = unit_type

    logging.debug(
        f"Added unit {new_unit} connected from {source_unit} to {target_unit}, type {unit_type}, activation {activation}, source weight {source_weight}, target weight {target_weight}"
    )

    return dataclasses.replace(
        net,
        activations=new_activations,
        unit_types=new_unit_types,
        forward_connections=new_forward_connections,
        reverse_forward_connections=new_reverse_forward_connections,
        forward_weights=new_forward_weights,
        recurrent_connections=new_recurrent_connections,
        reverse_recurrent_connections=new_reverse_recurrent_connections,
        recurrent_weights=new_recurrent_weights,
    )


def _add_random_unit(net: Network, config: configuration.SimulationConfig) -> Network:
    unit_type = random.choice(config.allowed_unit_types)
    activation = random.choice(config.allowed_activations)

    edge, edge_type = _get_random_edge_proportional_to_type_counts(net)
    _, _, weights = get_connections_and_weights_by_edge_type(net, edge_type)
    incoming_weight = weights[edge]
    outgoing_weight = _Weight(sign=1, numerator=1, denominator=1)

    return _add_unit(
        net,
        edge=edge,
        edge_type=edge_type,
        activation=activation,
        unit_type=unit_type,
        source_weight=incoming_weight,
        target_weight=outgoing_weight,
    )


def _replace_bias(net: Network, unit: _Node, weight: _Weight) -> Network:
    new_biases = _dict_copy(net.biases)
    new_biases[unit] = weight
    return dataclasses.replace(net, biases=new_biases)


def _add_random_bias(net: Network, config: configuration.SimulationConfig) -> Network:
    del config
    units_without_bias = (_get_hidden_units(net) | set(net.output_units_range)) - set(
        net.biases
    )
    target_unit = random.choice(tuple(units_without_bias))
    weight = _make_random_weight()
    logging.debug(f"Added bias {weight} to unit {target_unit}")
    return _replace_bias(net, unit=target_unit, weight=weight)


def _remove_random_bias(
    net: Network, config: configuration.SimulationConfig
) -> Network:
    del config
    target_unit = random.choice(tuple(net.biases))
    new_biases = _dict_copy(net.biases)
    del new_biases[target_unit]
    logging.debug(f"Remove bias from unit {target_unit}")
    return dataclasses.replace(net, biases=new_biases)


def remove_unit(net: Network, unit_to_remove: int) -> Network:
    new_forward_connections = _dict_copy(net.forward_connections)
    new_reverse_forward_connections = _dict_copy(net.reverse_forward_connections)
    new_forward_weights = _dict_copy(net.forward_weights)

    new_recurrent_connections = _dict_copy(net.recurrent_connections)
    new_reverse_recurrent_connections = _dict_copy(net.reverse_recurrent_connections)
    new_recurrent_weights = _dict_copy(net.recurrent_weights)

    new_activations = _dict_copy(net.activations)
    new_unit_types = _dict_copy(net.unit_types)
    new_biases = _dict_copy(net.biases)

    incoming_units = net.reverse_forward_connections.get(
        unit_to_remove, frozenset()
    ) | net.reverse_recurrent_connections.get(unit_to_remove, frozenset())
    outgoing_units = net.forward_connections.get(
        unit_to_remove, frozenset()
    ) | net.recurrent_connections.get(unit_to_remove, frozenset())

    for unit_from in incoming_units:
        for unit_to in outgoing_units:
            if unit_from == unit_to_remove or unit_to == unit_to_remove:
                # Self loop.
                continue
            edge_from = (unit_from, unit_to_remove)
            edge_to = (unit_to_remove, unit_to)

            add_recurrent = (
                edge_from in net.recurrent_weights or edge_to in net.recurrent_weights
            )
            add_forward = (
                edge_from in net.forward_weights and edge_to in net.forward_weights
            ) or (
                edge_from in net.forward_weights and edge_from in net.recurrent_weights
            )

            bridge_edge = (unit_from, unit_to)

            if add_recurrent and bridge_edge not in net.recurrent_weights:
                try:
                    recurrent_weight = net.recurrent_weights[edge_from]
                except KeyError:
                    recurrent_weight = net.forward_weights[edge_from]

                new_recurrent_weights[bridge_edge] = recurrent_weight

                if unit_from not in new_recurrent_connections:
                    new_recurrent_connections[unit_from] = frozenset()
                new_recurrent_connections[unit_from] |= {unit_to}

                if unit_to not in new_reverse_recurrent_connections:
                    new_reverse_recurrent_connections[unit_to] = frozenset()
                new_reverse_recurrent_connections[unit_to] |= {unit_from}

            if (
                add_forward
                # Prevent forward self-loop creation.
            ):
                if bridge_edge in net.forward_weights:
                    # If a bypassing edge already exists, simply its weight.
                    forward_weight = net.forward_weights[bridge_edge]
                else:
                    forward_weight = net.forward_weights[edge_from]

                if unit_from != unit_to:
                    new_forward_weights[bridge_edge] = forward_weight

                    if unit_from not in new_forward_connections:
                        new_forward_connections[unit_from] = frozenset()
                    new_forward_connections[unit_from] |= {unit_to}

                    if unit_to not in new_reverse_forward_connections:
                        new_reverse_forward_connections[unit_to] = frozenset()
                    new_reverse_forward_connections[unit_to] |= {unit_from}

    # Cleanup outside loop in case of removed-units without inputs or without outputs.
    for unit_to in net.forward_connections.get(unit_to_remove, frozenset()):
        new_reverse_forward_connections[unit_to] -= {unit_to_remove}
        if not new_reverse_forward_connections[unit_to]:
            del new_reverse_forward_connections[unit_to]
        del new_forward_weights[unit_to_remove, unit_to]

    for unit_from in net.reverse_forward_connections.get(unit_to_remove, frozenset()):
        new_forward_connections[unit_from] -= {unit_to_remove}
        if not new_forward_connections[unit_from]:
            del new_forward_connections[unit_from]
        del new_forward_weights[unit_from, unit_to_remove]

    for unit_from in net.reverse_recurrent_connections.get(unit_to_remove, frozenset()):
        new_recurrent_connections[unit_from] -= {unit_to_remove}
        if not new_recurrent_connections[unit_from]:
            del new_recurrent_connections[unit_from]
        del new_recurrent_weights[(unit_from, unit_to_remove)]

    for unit_to in net.recurrent_connections.get(unit_to_remove, frozenset()):
        if unit_to == unit_to_remove:
            # Self recurrent edge handled by previous loop.
            continue
        new_reverse_recurrent_connections[unit_to] -= {unit_to_remove}
        if not new_reverse_recurrent_connections[unit_to]:
            del new_reverse_recurrent_connections[unit_to]
        del new_recurrent_weights[(unit_to_remove, unit_to)]

    try:
        del new_forward_connections[unit_to_remove]
    except KeyError:
        pass
    try:
        del new_reverse_forward_connections[unit_to_remove]
    except KeyError:
        pass
    try:
        del new_recurrent_connections[unit_to_remove]
    except KeyError:
        pass
    try:
        del new_reverse_recurrent_connections[unit_to_remove]
    except KeyError:
        pass

    try:
        del new_forward_weights[(unit_to_remove, unit_to_remove)]
    except KeyError:
        pass
    try:
        del new_recurrent_weights[(unit_to_remove, unit_to_remove)]
    except KeyError:
        pass

    del new_activations[unit_to_remove]
    del new_unit_types[unit_to_remove]
    try:
        del new_biases[unit_to_remove]
    except KeyError:
        pass

    new_net = dataclasses.replace(
        net,
        forward_connections=new_forward_connections,
        reverse_forward_connections=new_reverse_forward_connections,
        forward_weights=new_forward_weights,
        recurrent_connections=new_recurrent_connections,
        reverse_recurrent_connections=new_reverse_recurrent_connections,
        recurrent_weights=new_recurrent_weights,
        activations=new_activations,
        unit_types=new_unit_types,
        biases=new_biases,
    )

    return _defragment_unit_numbers(new_net, unit_to_remove)


def remove_unit_single_input_output(net: Network, unit_to_remove: int) -> Network:
    # Assumes unit has only one outgoing connection and one incoming connection, and no recurrent connections.
    connected_to = tuple(net.forward_connections.get(unit_to_remove))[0]
    connected_from = tuple(net.reverse_forward_connections.get(unit_to_remove))[0]

    new_forward_connections = _dict_copy(net.forward_connections)
    new_reverse_forward_connections = _dict_copy(net.reverse_forward_connections)
    new_forward_weights = _dict_copy(net.forward_weights)
    new_activations = _dict_copy(net.activations)
    new_unit_types = _dict_copy(net.unit_types)

    del new_forward_connections[unit_to_remove]
    del new_reverse_forward_connections[unit_to_remove]

    new_forward_connections[connected_from] -= {unit_to_remove}
    if len(new_forward_connections[connected_from]) == 0:
        del new_forward_connections[connected_from]

    new_reverse_forward_connections[connected_to] -= {unit_to_remove}
    if len(new_reverse_forward_connections[connected_to]) == 0:
        del new_reverse_forward_connections[connected_to]

    del new_forward_weights[(connected_from, unit_to_remove)]
    del new_forward_weights[(unit_to_remove, connected_to)]
    del new_activations[unit_to_remove]
    del new_unit_types[unit_to_remove]

    new_net = dataclasses.replace(
        net,
        forward_connections=new_forward_connections,
        reverse_forward_connections=new_reverse_forward_connections,
        forward_weights=new_forward_weights,
        activations=new_activations,
        unit_types=new_unit_types,
    )
    logging.debug(f"Removed unit {unit_to_remove}")
    return _defragment_unit_numbers(new_net, unit_to_remove)


def _get_removable_units(net: Network) -> Tuple[_Node]:
    # To keep mutations symmetric, only remove units in the same status they have when they are created,
    # i.e., with only one incoming and one outgoing forward connection, and no recurrent connections.
    return tuple(
        x
        for x in net.forward_connections
        if len(net.forward_connections[x]) == 1
        and x in net.reverse_forward_connections
        and len(net.reverse_forward_connections[x]) == 1
        and x not in frozenset(net.input_units_range)
        and x not in frozenset(net.output_units_range)
        and x not in net.recurrent_connections
        and x not in net.reverse_recurrent_connections
        and x not in net.biases
    )


def _remove_random_unit(
    net: Network, config: configuration.SimulationConfig
) -> Network:
    del config
    target = random.choice(tuple(_get_hidden_units(net)))
    return remove_unit(net, target)


def _defragment_unit_numbers(net: Network, removed_unit: int) -> Network:
    # Move the highest numbered unit to fill the place of `removed_unit`.
    max_unit = max(net.activations)

    if max_unit == removed_unit - 1:
        return net

    for edge_type, connections, reverse_connections, weights in (
        (
            FORWARD_CONNECTION,
            net.forward_connections,
            net.reverse_forward_connections,
            net.forward_weights,
        ),
        (
            RECURRENT_CONNECTION,
            net.recurrent_connections,
            net.reverse_recurrent_connections,
            net.recurrent_weights,
        ),
    ):
        new_connections = _dict_copy(connections)
        new_reverse_connections = _dict_copy(reverse_connections)
        new_weights = _dict_copy(weights)

        incoming_units = new_reverse_connections.get(max_unit, frozenset())
        outgoing_units = new_connections.get(max_unit, frozenset())

        weights_to = [(x, max_unit) for x in incoming_units]
        weights_from = [(max_unit, x) for x in outgoing_units]

        for incoming_unit in incoming_units:
            new_connections[incoming_unit] -= {max_unit}
            new_connections[incoming_unit] |= {removed_unit}

        for outgoing_unit in outgoing_units:
            new_reverse_connections[outgoing_unit] -= {max_unit}
            new_reverse_connections[outgoing_unit] |= {removed_unit}

        for weight_to in weights_to:
            if weight_to[0] == weight_to[1]:
                # Self-loop.
                source = removed_unit
            else:
                source = weight_to[0]
            new_weights[(source, removed_unit)] = new_weights[weight_to]
            del new_weights[weight_to]

        for weight_from in weights_from:
            if weight_from[0] == weight_from[1]:
                # Loop. Already deleted in previous loop.
                continue
            new_weights[(removed_unit, weight_from[1])] = new_weights[weight_from]
            del new_weights[weight_from]

        try:
            new_connections[removed_unit] = new_connections[max_unit]
            del new_connections[max_unit]
        except KeyError:
            pass
        try:
            new_reverse_connections[removed_unit] = new_reverse_connections[max_unit]
            del new_reverse_connections[max_unit]
        except KeyError:
            pass

        if edge_type == FORWARD_CONNECTION:
            net = dataclasses.replace(
                net,
                forward_connections=new_connections,
                reverse_forward_connections=new_reverse_connections,
                forward_weights=new_weights,
            )
        else:
            net = dataclasses.replace(
                net,
                recurrent_connections=new_connections,
                reverse_recurrent_connections=new_reverse_connections,
                recurrent_weights=new_weights,
            )

    new_biases = _dict_copy(net.biases)
    new_activations = _dict_copy(net.activations)
    new_unit_types = _dict_copy(net.unit_types)

    try:
        new_biases[removed_unit] = new_biases[max_unit]
        del new_biases[max_unit]
    except KeyError:
        pass

    new_activations[removed_unit] = new_activations[max_unit]
    del new_activations[max_unit]

    new_unit_types[removed_unit] = new_unit_types[max_unit]
    del new_unit_types[max_unit]

    return dataclasses.replace(
        net, biases=new_biases, activations=new_activations, unit_types=new_unit_types
    )


def _replace_weight(
    net: Network, new_weight: _Weight, edge_or_bias: Union[_Edge, _Node], edge_type: int
) -> Network:
    if edge_type == _BIAS_CONNECTION:
        old_weights = net.biases
    else:
        *_, old_weights = get_connections_and_weights_by_edge_type(net, edge_type)

    edge_type_keyword = {
        FORWARD_CONNECTION: "forward_weights",
        RECURRENT_CONNECTION: "recurrent_weights",
        _BIAS_CONNECTION: "biases",
    }[edge_type]

    new_weights = _dict_copy(old_weights)
    new_weights[edge_or_bias] = new_weight
    return dataclasses.replace(net, **{edge_type_keyword: new_weights})


def _perturbate_weight(weight: _Weight) -> _Weight:
    # Flip sign with higher probability the closer we are to 0.0.
    flip_sign_weight = max(0.0, 1 - (weight.numerator / weight.denominator))
    mutation_target = random.choices(
        ["sign", "numerator", "denominator"], [flip_sign_weight, 1, 1]
    )[0]

    if mutation_target == "sign":
        new_val = weight.sign * -1
    else:
        curr_val = getattr(weight, mutation_target)
        change = random.choice([1, -1]) if curr_val > 1 else 1
        new_val = curr_val + change

    return dataclasses.replace(weight, **{mutation_target: new_val})


def _mutate_weight(net: Network, config: configuration.SimulationConfig) -> Network:
    available_edge_types_and_weights = [(FORWARD_CONNECTION, len(net.forward_weights))]
    if config.recurrent_connections:
        available_edge_types_and_weights.append(
            (RECURRENT_CONNECTION, len(net.recurrent_weights))
        )
    if config.bias_connections:
        available_edge_types_and_weights.append((_BIAS_CONNECTION, len(net.biases)))

    available_edge_types, edge_type_weights = list(
        zip(*available_edge_types_and_weights)
    )

    edge_type = random.choices(available_edge_types, weights=edge_type_weights)[0]

    if edge_type == _BIAS_CONNECTION:
        old_weights = net.biases
    else:
        *_, old_weights = get_connections_and_weights_by_edge_type(net, edge_type)

    target_edge = random.choice(list(old_weights.keys()))
    current_weight = old_weights[target_edge]
    new_weight = _perturbate_weight(current_weight)

    return _replace_weight(
        net, new_weight=new_weight, edge_or_bias=target_edge, edge_type=edge_type
    )


def _get_edges(net: Network, edge_type: int) -> FrozenSet[_Edge]:
    _, _, weights = get_connections_and_weights_by_edge_type(net, edge_type)
    return frozenset(weights.keys())


def _replace_connections_and_weights(
    net: Network,
    edge_type: int,
    new_connections: _Connections,
    new_reverse_connections: _Connections,
    new_weights: _Weights,
) -> Network:
    if edge_type == FORWARD_CONNECTION:
        replace_args = (
            "forward_connections",
            "reverse_forward_connections",
            "forward_weights",
        )
    else:
        replace_args = (
            "recurrent_connections",
            "reverse_recurrent_connections",
            "recurrent_weights",
        )

    replace_kwargs = dict(
        zip(replace_args, (new_connections, new_reverse_connections, new_weights),)
    )
    return dataclasses.replace(net, **replace_kwargs)


def _add_connection(
    net: Network, edge: _Edge, edge_type: int, weight: _Weight
) -> Network:
    (
        old_connections,
        old_reverse_connections,
        old_weights,
    ) = get_connections_and_weights_by_edge_type(net, edge_type)

    new_connections = _dict_copy(old_connections)
    new_reverse_connections = _dict_copy(old_reverse_connections)
    new_weights = _dict_copy(old_weights)

    source_unit, target_unit = edge

    if source_unit not in old_connections:
        new_connections[source_unit] = frozenset()
    new_connections[source_unit] |= {target_unit}

    if target_unit not in new_reverse_connections:
        new_reverse_connections[target_unit] = frozenset()
    new_reverse_connections[target_unit] |= {source_unit}

    new_weights[edge] = weight

    return _replace_connections_and_weights(
        net,
        edge_type=edge_type,
        new_connections=new_connections,
        new_reverse_connections=new_reverse_connections,
        new_weights=new_weights,
    )


def _get_num_connections(net: Network, edge_type: int) -> int:
    return len(
        {
            FORWARD_CONNECTION: net.forward_weights,
            RECURRENT_CONNECTION: net.recurrent_weights,
        }[edge_type]
    )


def _add_random_connection(
    net: Network, config: configuration.SimulationConfig
) -> Network:
    available_edge_types = []

    if _get_num_connections(
        net, FORWARD_CONNECTION
    ) < _get_number_of_possible_connections(net, FORWARD_CONNECTION):
        available_edge_types.append(FORWARD_CONNECTION)

    if config.recurrent_connections and _get_num_connections(
        net, RECURRENT_CONNECTION
    ) < _get_number_of_possible_connections(net, RECURRENT_CONNECTION):
        available_edge_types.append(RECURRENT_CONNECTION)

    edge_type = random.choice(available_edge_types)
    existing_connections, _, _ = get_connections_and_weights_by_edge_type(
        net, edge_type=edge_type
    )

    hidden_units = _get_hidden_units(net)
    valid_source_units = frozenset(net.input_units_range) | hidden_units
    valid_target_units = hidden_units | frozenset(net.output_units_range)
    valid_sources_to_valid_targets = {}

    for unit in valid_source_units:
        unit_existing_targets = existing_connections.get(unit, frozenset())
        valid_targets = valid_target_units - unit_existing_targets

        # No point in creating a forward self-loop.
        if edge_type == FORWARD_CONNECTION:
            valid_targets -= {unit}

        valid_sources_to_valid_targets[unit] = tuple(valid_targets)

    valid_sources = tuple(
        x for x in valid_sources_to_valid_targets if valid_sources_to_valid_targets[x]
    )

    source_unit = random.choice(valid_sources)
    target_unit = random.choice(valid_sources_to_valid_targets[source_unit])

    return _add_connection(
        net,
        edge=(source_unit, target_unit),
        edge_type=edge_type,
        weight=_make_random_weight(),
    )


def _remove_connection(net: Network, edge: _Edge, edge_type: int) -> Network:
    (
        old_connections,
        old_reverse_connections,
        old_weights,
    ) = get_connections_and_weights_by_edge_type(net, edge_type)

    new_connections = _dict_copy(old_connections)
    new_reverse_connections = _dict_copy(old_reverse_connections)
    new_weights = _dict_copy(old_weights)

    source_unit, target_unit = edge

    new_connections[source_unit] -= {target_unit}
    if len(new_connections[source_unit]) == 0:
        del new_connections[source_unit]
    new_reverse_connections[target_unit] -= {source_unit}
    if len(new_reverse_connections[target_unit]) == 0:
        del new_reverse_connections[target_unit]
    del new_weights[edge]

    return _replace_connections_and_weights(
        net,
        edge_type=edge_type,
        new_connections=new_connections,
        new_reverse_connections=new_reverse_connections,
        new_weights=new_weights,
    )


def _get_random_edge_proportional_to_type_counts(net: Network) -> Tuple[_Edge, int]:
    edge_type = random.choices(
        (FORWARD_CONNECTION, RECURRENT_CONNECTION),
        weights=(
            _get_num_connections(net, FORWARD_CONNECTION),
            _get_num_connections(net, RECURRENT_CONNECTION),
        ),
    )[0]

    existing_edges = _get_edges(net, edge_type=edge_type)
    edge = random.choice(tuple(existing_edges))
    return edge, edge_type


def _remove_random_connection(
    net: Network, config: configuration.SimulationConfig
) -> Network:
    del config
    edge, edge_type = _get_random_edge_proportional_to_type_counts(net)
    return _remove_connection(net, edge=edge, edge_type=edge_type)


def _replace_activation(net: Network, unit: int, new_activation: int) -> Network:
    new_activations = _dict_copy(net.activations)
    new_activations[unit] = new_activation
    return dataclasses.replace(net, activations=new_activations)


def _mutate_activation(net: Network, config: configuration.SimulationConfig) -> Network:
    target = random.choice(tuple(_get_hidden_units(net) | set(net.output_units_range)))
    old_activation = net.activations[target]
    new_activation = random.choice(
        tuple(set(config.allowed_activations) - {old_activation})
    )
    logging.debug(
        f"Mutated unit {target} activation: {old_activation}->{new_activation}"
    )
    return _replace_activation(net, target, new_activation)


def _mutate_unit_type(net: Network, config: configuration.SimulationConfig) -> Network:
    target = random.choice(tuple(_get_hidden_units(net) | set(net.output_units_range)))
    old_type = net.unit_types[target]
    new_type = random.choice(tuple(set(config.allowed_unit_types) - {old_type}))
    new_types = _dict_copy(net.unit_types)
    new_types[target] = new_type
    logging.debug(f"Mutated unit {target} type: {old_type}->{new_type}")
    return dataclasses.replace(net, unit_types=new_types)


def get_dot_string(
    net: Network,
    hide_non_connected_units: bool = False,
    class_to_label: Optional[Dict[int, Text]] = None,
) -> Text:
    activation_to_color = {
        LINEAR: "#f2e7f7",
        SIGMOID: "#ffeba9",
        SQUARE: "#a9ddff",
        _TANH: "burlywood3",
        RELU: "#b394d8",
        FLOOR: "deeppink3",
    }

    border_color = "#333333"
    input_fill_color = "#FBD35A"
    output_fill_color = "#4CA0E8"

    default_color = "#CECECE"

    dot = "digraph G {\n" "colorscheme=X11\n"
    dot += 'rankdir="LR"; \n'
    # dot += "graph [ dpi = 100 ] \n"

    weight_font_size = 9
    node_font_size = 9
    arrow_size = 0.5
    unit_size = 0.5
    margin = 0.0

    visible_units = set()

    for unit in get_units(net):
        if (
            hide_non_connected_units
            and (unit not in net.output_units_range)
            and (
                unit not in net.forward_connections or not net.forward_connections[unit]
            )
            and (
                unit not in net.reverse_forward_connections
                or not net.reverse_forward_connections[unit]
            )
            and (
                unit not in net.recurrent_connections
                or not net.recurrent_connections[unit]
            )
            and (unit not in net.biases)
        ):
            continue

        visible_units.add(unit)

        activation = net.activations[unit]

        if unit in net.input_units_range:
            shape = "doublecircle"
            color = border_color
            style = "filled"
            fillcolor = input_fill_color
            # dot += f'to_input_{unit} [shape="plaintext" label=<<&#119909><FONT POINT-SIZE="10">t</FONT>>]; \n'

            dot += f'to_input_{unit} [shape="plaintext" label="{class_to_label.get(unit, unit) if class_to_label else unit}" fontsize={node_font_size} margin=0 width=0 height=0]; \n'
            dot += f"to_input_{unit} -> {unit} [ arrowsize={arrow_size} fontsize={weight_font_size}];\n"

        elif unit in net.output_units_range:
            shape = "doublecircle"
            color = border_color
            style = "filled"
            fillcolor = output_fill_color

            dot += f'out_str_{unit} [label="P({class_to_label.get(unit, unit) if class_to_label else unit})" shape=plaintext margin=0.05 width=0 height=0 fontsize={node_font_size}]\n'
            dot += f"{unit} -> out_str_{unit} [ arrowsize={arrow_size} ];\n"

        else:
            shape = "circle"
            color = border_color
            style = "filled"
            fillcolor = activation_to_color.get(activation, default_color)

        dot += f'{unit} [label="{unit}\n{_ACTIVATION_NAMES[activation].title()}\n{"+" if net.unit_types[unit] == SUMMATION_UNIT else "x"}" shape={shape} style={style} fillcolor="{fillcolor}" margin={margin} width={unit_size} height={unit_size} fontsize={node_font_size} color="{color}"] \n'

    visible_inputs = set(net.input_units_range) & visible_units
    dot += "{rank = same; %s}\n" % "; ".join(map("to_input_{}".format, visible_inputs))
    dot += "{rank = same; %s}\n" % "; ".join(map(str, visible_inputs))
    dot += "{rank = same; %s}\n" % "; ".join(map(str, net.output_units_range))

    # Connections.
    for edge_type in (FORWARD_CONNECTION, RECURRENT_CONNECTION):
        (
            connections,
            reversed_connections,
            weights,
        ) = get_connections_and_weights_by_edge_type(net, edge_type)
        for node in get_units(net):
            if edge_type == RECURRENT_CONNECTION:
                style = "style=dashed"
            else:
                style = ""

            if node not in connections:
                continue
            for neighbor in connections[node]:
                weight = _get_weight(net, edge_type, node, neighbor)
                dot += f'{node} -> {neighbor} [ {style} label="{weight:.2f}" arrowsize={arrow_size} fontsize={weight_font_size}];\n'

    # Bias connections.
    for bias_node in get_units(net):
        if bias_node not in net.biases:
            continue
        bias_weight = net.biases[bias_node]
        bias = bias_weight.sign * bias_weight.numerator / bias_weight.denominator
        dot += f'bias_{bias_node} [style="invis"]; \n'
        dot += f'bias_{bias_node} -> {bias_node} [ label="{bias:.2f}" arrowsize=0.5 fontsize={weight_font_size} minlen=0.5];\n'

    dot += "}"
    return dot


def visualize(
    net: Network,
    filename: Text,
    to_png: bool = False,
    hide_non_connected_units: bool = False,
    class_to_label: Optional[Dict[int, Text]] = None,
):
    dot = get_dot_string(
        net,
        hide_non_connected_units=hide_non_connected_units,
        class_to_label=class_to_label,
    )
    path = pathlib.Path(f"networks/{filename}.dot")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(dot)
    if to_png:
        import graphviz

        graphviz.render("dot", "png", str(path))


def save(net: Network, name: Text):
    path = pathlib.Path(f"./networks/{name}.pickle")
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(net, f)


def to_string(net: Network) -> Text:
    if net.fitness is not None:
        fitness = f"{net.fitness.mdl:,.2f}"
        g = f"{net.fitness.grammar_encoding_length:,.2f}"
        d_g = f"{net.fitness.data_encoding_length:,.2f}"
        accuracy = f"{net.fitness.accuracy:.2f}"
    else:
        fitness = g = d_g = accuracy = "?"

    return (
        f"MDL = {fitness}\t"
        f"G={g}"
        f"\tD:G = {d_g}"
        f"\tNum units: {get_num_units(net)}"
        f"\tNum connections: {get_total_connections(net, include_biases=False)}"
        f"\tAccuracy: {accuracy}"
    )


def _offset_all_node_ids_by(connections: Dict, offset: int) -> Dict:
    new_dict = {}
    for key, val in connections.items():
        if isinstance(key, tuple):
            # Edge.
            new_key = tuple(map(lambda x: x + offset, key))
        elif isinstance(key, _Node):
            new_key = key + offset
        else:
            raise ValueError

        if isinstance(val, frozenset):
            new_val = frozenset(map(lambda x: x + offset, val))
        else:
            new_val = val
        new_dict[new_key] = new_val
    return new_dict


def make_custom_network(
    input_size: int,
    output_size: int,
    num_units: int,
    forward_weights: Dict[int, Tuple[Tuple[int, int, int, int], ...]],
    recurrent_weights: Dict[int, Tuple[Tuple[int, int, int, int], ...]],
    activations: Optional[Dict[int, int]] = None,
    unit_types: Optional[Dict[int, int]] = None,
    biases: Optional[Dict[int, float]] = None,
) -> Network:
    """
    forward/recurrent weights structure: {from_unit: ((to_unit, sign, numerator, denominator), ...), ...}
    """
    forward_connections = collections.defaultdict(frozenset)
    forward_reverse_connections = collections.defaultdict(frozenset)
    net_forward_weights = {}

    recurrent_connections = collections.defaultdict(frozenset)
    recurrent_reverse_connections = collections.defaultdict(frozenset)
    net_recurrent_weights = {}

    for edge_type, current_weights in zip(
        [FORWARD_CONNECTION, RECURRENT_CONNECTION],
        [forward_weights, recurrent_weights],
    ):
        if edge_type == FORWARD_CONNECTION:
            connections = forward_connections
            reverse_connections = forward_reverse_connections
            weights = net_forward_weights
        else:
            connections = recurrent_connections
            reverse_connections = recurrent_reverse_connections
            weights = net_recurrent_weights

        for from_unit, unit_connections in current_weights.items():
            for (to_unit, sign, numerator, denominator) in unit_connections:
                connections[from_unit] |= {to_unit}
                reverse_connections[to_unit] |= {from_unit}
                weights[(from_unit, to_unit)] = _Weight(sign, numerator, denominator)

    # TODO: this is a shortcut to avoid defining biases using the numerator/denominator method.
    net_biases = {}
    if biases:
        for unit, bias in biases.items():
            sign = np.sign(bias).item()
            fraction = fractions.Fraction(bias).limit_denominator(10)
            net_biases[unit] = _Weight(
                sign, abs(fraction.numerator), abs(fraction.denominator)
            )

    if activations is None:
        activations = {}
    net_activations = {unit: activations.get(unit, LINEAR) for unit in range(num_units)}

    if unit_types is None:
        unit_types = {}
    net_unit_types = {
        unit: unit_types.get(unit, SUMMATION_UNIT) for unit in range(num_units)
    }

    return Network(
        input_units_range=list(range(input_size)),
        output_units_range=list(range(input_size, input_size + output_size)),
        forward_weights=net_forward_weights,
        forward_connections=dict(forward_connections),
        reverse_forward_connections=dict(forward_reverse_connections),
        recurrent_connections=dict(recurrent_connections),
        reverse_recurrent_connections=dict(recurrent_reverse_connections),
        recurrent_weights=net_recurrent_weights,
        biases=net_biases,
        unit_types=net_unit_types,
        activations=net_activations,
    )


def _dict_copy(d: Dict) -> Dict:
    # Fast copy assuming all values are immutable (int/frozenset/Weight).
    return {**d}


def calculate_symbolic_accuracy(
    target_probabs: np.ndarray,
    found_net: Network,
    inputs: np.ndarray,
    input_mask: Optional[np.ndarray],
    sample_weights: Tuple[int, ...],
    config: configuration.SimulationConfig,
    plots: bool,
    epsilon: float = 0.0,
) -> Tuple[float, Tuple[int, ...]]:
    predicted_probabs = predict_probabs(
        net=found_net,
        input_sequence=inputs,
        recurrent_connections=config.recurrent_connections,
        truncate_large_values=config.truncate_large_values,
        softmax_outputs=config.softmax_outputs,
    )
    return utils.calculate_symbolic_accuracy(
        predicted_probabs=predicted_probabs,
        target_probabs=target_probabs,
        input_mask=input_mask,
        plots=plots,
        epsilon=epsilon,
        sample_weights=sample_weights,
    )
