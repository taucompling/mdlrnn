import dataclasses
import itertools
import logging
import pickle
import unittest

import numpy as np
import toolz

import configuration
import corpora
import manual_nets
import network
import utils

utils.setup_logging()

_EPSILON = 0.01

_TEST_CONFIG = configuration.SimulationConfig(
    simulation_id="test",
    num_islands=1,
    migration_ratio=0.1,
    migration_interval_seconds=20,
    migration_interval_generations=1000,
    num_generations=1000,
    population_size=20,
    elite_ratio=0.05,
    allowed_activations=(
        network.SIGMOID,
        network.LINEAR,
        network.RELU,
        network.SQUARE,
    ),
    start_smooth=False,
    allowed_unit_types=(network.SUMMATION_UNIT, network.MULTIPLICATION_UNIT),
    tournament_size=4,
    mutation_probab=0.9,
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


def _evolve_weight(
    target_weight: network._Weight, max_retries: int = 1_000_000
) -> network._Weight:
    i = 0
    w = network._make_random_weight()
    initial_w = network._Weight(**w.__dict__)
    while i < max_retries:
        if network._get_weight_real_value(w) == network._get_weight_real_value(
            target_weight
        ):
            logging.info(
                f"Weight {network._get_weight_real_value(w)} generation took {i} steps, started from {network._get_weight_real_value(initial_w):.2f}"
            )
            return w
        w = network._perturbate_weight(w)
        i += 1
    raise ValueError(
        f"Couldn't generate weight {network._get_weight_real_value(target_weight)}, started from {network._get_weight_real_value(initial_w)}"
    )


utils.setup_logging()


class TestNetwork(unittest.TestCase):
    def test_topological_sort(self):
        net = manual_nets.make_emmanuel_triplet_xor_network()
        network.visualize(net, "dfs_emmanuel_triplet_xor_network")
        loop_edges, topological_sort = network.run_dfs(
            network.get_units(net),
            reverse_connections=net.reverse_forward_connections,
            input_units=net.input_units_range,
            output_units=net.output_units_range,
        )
        print(topological_sort)
        assert len(topological_sort) == network.get_num_units(net)
        assert len(loop_edges) == 0

        node_to_topological_idx = {
            node: idx for idx, node in enumerate(topological_sort)
        }

        net_layers = [(0, 2, 4), (3,), (5, 6), (1,)]

        for layer_1, layer_2 in toolz.sliding_window(2, net_layers):
            for node_1, node_2 in itertools.product(layer_1, layer_2):
                assert (
                    node_to_topological_idx[node_1] < node_to_topological_idx[node_2]
                ), (node_1, node_2)

    def test_topological_sort_through_time(self):
        num_units = 5
        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=num_units,
            forward_weights={
                0: ((1, 1, 1, 1), (2, 1, 1, 1)),
                1: ((3, 1, 1, 1),),
                2: ((3, 1, 1, 1),),
                3: ((4, 1, 1, 1),),
            },
            recurrent_weights={
                2: ((2, 1, 1, 1),),
                3: ((1, 1, 1, 1),),
                4: ((3, 1, 1, 1),),
            },
        )
        network.visualize(net, "test_topological_sort_through_time")
        (
            loop_edges,
            topological_sort_through_time,
        ) = network.topological_sort_through_time(net)

        print(topological_sort_through_time)
        assert topological_sort_through_time in {(0, 1, 2, 3, 4), (0, 2, 1, 3, 4)}

    def test_get_edges(self):
        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=5,
            forward_weights={
                0: ((4, 1, 1, 1),),
                1: ((4, 1, 1, 1),),
                4: ((2, 1, 1, 1), (3, 1, 1, 1),),
            },
            recurrent_weights={3: ((3, 1, 1, 1),)},
        )
        assert network._get_edges(net, edge_type=network.FORWARD_CONNECTION) == {
            (0, 4),
            (1, 4),
            (4, 2),
            (4, 3),
        }
        assert network._get_edges(net, edge_type=network.RECURRENT_CONNECTION) == {
            (3, 3)
        }

    def test_add_edge(self):
        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=5,
            forward_weights={
                0: ((4, 1, 1, 1),),
                1: ((4, 1, 1, 1),),
                4: ((2, 1, 1, 1), (3, 1, 1, 1),),
            },
            recurrent_weights={3: ((3, 1, 1, 1),)},
        )

        network.visualize(net, "test_add_edge_before")
        net = network._add_connection(
            net,
            edge=(1, 3),
            edge_type=network.FORWARD_CONNECTION,
            weight=network._Weight(1, 1, 1),
        )
        net = network._add_connection(
            net,
            edge=(2, 3),
            edge_type=network.RECURRENT_CONNECTION,
            weight=network._Weight(1, 1, 1),
        )
        network.visualize(net, "test_add_edge_after")

        assert (1, 3) in network._get_edges(net, network.FORWARD_CONNECTION)
        assert (2, 3) in network._get_edges(net, network.RECURRENT_CONNECTION)
        assert 3 in net.forward_connections[1]
        assert 1 in net.reverse_forward_connections[3]
        assert 3 in net.recurrent_connections[2]
        assert 2 in net.reverse_recurrent_connections[3]

    def test_remove_edge(self):
        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=5,
            forward_weights={
                0: ((4, 1, 1, 1),),
                1: ((4, 1, 1, 1),),
                4: ((2, 1, 1, 1), (3, 1, 1, 1),),
            },
            recurrent_weights={3: ((3, 1, 1, 1),), 4: ((1, 1, 1, 1),)},
        )

        network.visualize(net, "test_remove_edge_before")
        net = network._remove_connection(
            net, edge=(4, 2), edge_type=network.FORWARD_CONNECTION
        )
        net = network._remove_connection(
            net, edge=(4, 1), edge_type=network.RECURRENT_CONNECTION
        )
        network.visualize(net, "test_remove_edge_after")

        assert (4, 2) not in network._get_edges(net, network.FORWARD_CONNECTION)
        assert (4, 1) not in network._get_edges(net, network.RECURRENT_CONNECTION)

        assert 4 not in net.recurrent_connections
        assert 1 not in net.reverse_recurrent_connections

        assert 4 in net.forward_connections and 2 not in net.forward_connections[4]
        assert 2 not in net.reverse_forward_connections

    def test_add_remove_edge_symmetry(self):
        original_net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=4,
            forward_weights={0: ((3, 1, 1, 1),), 1: ((2, 1, 1, 1),),},
            recurrent_weights={},
        )

        for edge_type in (network.FORWARD_CONNECTION, network.RECURRENT_CONNECTION):
            network.visualize(original_net, "test_add_remove_edge_symmetry_1")
            net_2 = network._add_connection(
                original_net,
                edge=(0, 2),
                edge_type=edge_type,
                weight=network._Weight(1, 1, 1),
            )
            network.visualize(net_2, "test_add_remove_edge_symmetry_2")
            net_3 = network._remove_connection(net_2, edge=(0, 2), edge_type=edge_type)
            network.visualize(net_3, "test_add_remove_edge_symmetry_3")
            assert net_3 == original_net

    def test_equality(self):
        net1 = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=4,
            forward_weights={0: ((3, 1, 1, 1),), 1: ((2, 1, 3, 4),),},
            recurrent_weights={0: ((2, -1, 2, 3),)},
        )
        network.visualize(net1, "test_equality_1")
        net2 = pickle.loads(pickle.dumps(net1))
        net3 = dataclasses.replace(net1, fitness=network.Fitness(1, 2, 3, 4))
        assert net1 == net2
        assert net1 == net3

        assert len(frozenset([net1, net2, net3])) == 1

    def test_add_unit_on_recurrent_edge(self):
        original_net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=4,
            forward_weights={0: ((3, 1, 1, 1),), 1: ((2, 1, 3, 4),),},
            recurrent_weights={0: ((2, -1, 2, 3),)},
        )
        edge = (0, 2)
        net2 = network._add_unit(
            original_net,
            edge=edge,
            edge_type=network.RECURRENT_CONNECTION,
            activation=network.SIGMOID,
            unit_type=network.SUMMATION_UNIT,
            source_weight=original_net.recurrent_weights[edge],
            target_weight=network._Weight(1, 1, 1),
        )
        network.visualize(original_net, "test_add_unit_on_recurrent_edge_1")
        network.visualize(net2, "test_add_unit_on_recurrent_edge_2")

        assert net2.recurrent_weights[(0, 4)] == original_net.recurrent_weights[(0, 2)]
        assert net2.recurrent_connections[0] == {4}
        assert net2.forward_weights[(4, 2)] == network._Weight(1, 1, 1)
        assert net2.reverse_forward_connections[2] == {1, 4}
        assert net2.reverse_recurrent_connections[2] == frozenset()
        assert net2.reverse_recurrent_connections[4] == {0}

        net3 = network.remove_unit(net2, unit_to_remove=4)
        network.visualize(net3, "test_add_unit_on_recurrent_edge_3")

    def test_add_remove_unit_symmetry(self):
        original_net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=4,
            forward_weights={0: ((3, 1, 1, 1), (2, -1, 2, 3)), 1: ((2, 1, 3, 4),),},
            recurrent_weights={},
        )

        network.visualize(original_net, "test_add_remove_unit_symmetry_1")
        net_2 = network._add_unit(
            original_net,
            edge=(1, 2),
            edge_type=network.FORWARD_CONNECTION,
            activation=network.SIGMOID,
            unit_type=network.SUMMATION_UNIT,
            source_weight=original_net.forward_weights[(1, 2)],
            target_weight=network._Weight(sign=1, numerator=1, denominator=1),
        )
        network.visualize(net_2, "test_add_remove_unit_symmetry_2")

        assert (1, 2) not in net_2.forward_weights
        assert (1, 4) in net_2.forward_weights
        assert (4, 2) in net_2.forward_weights
        assert net_2.forward_connections[1] == {4}
        assert net_2.forward_connections[4] == {2}
        assert net_2.reverse_forward_connections[4] == {1}
        assert net_2.reverse_forward_connections[2] == {0, 4}

        net_3 = network._add_unit(
            net_2,
            edge=(4, 2),
            edge_type=network.FORWARD_CONNECTION,
            activation=network.SQUARE,
            unit_type=network.MULTIPLICATION_UNIT,
            source_weight=net_2.forward_weights[(4, 2)],
            target_weight=network._Weight(sign=1, numerator=1, denominator=1),
        )
        network.visualize(net_3, "test_add_remove_unit_symmetry_3")
        net_4 = network.remove_unit(net_3, unit_to_remove=5)
        network.visualize(net_4, "test_add_remove_unit_symmetry_4")
        assert net_4 == net_2
        net_5 = network.remove_unit(net_4, unit_to_remove=4)
        network.visualize(net_5, "test_add_remove_unit_symmetry_5")
        assert net_5 == original_net

    def test_add_bias(self):
        net1 = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=5,
            forward_weights={
                0: ((4, 1, 1, 1),),
                1: ((4, 1, 1, 1),),
                4: ((2, 1, 1, 1), (3, 1, 1, 1),),
            },
            recurrent_weights={3: ((3, 1, 1, 1),), 4: ((1, 1, 1, 1),)},
        )
        net2 = network._add_random_bias(net1, _TEST_CONFIG)
        network.visualize(net1, "test_add_bias_before")
        network.visualize(net2, "test_add_bias_after")

        assert len(net2.biases) == len(net1.biases) + 1
        # Only allow biases on hidden/output units.
        assert tuple(net2.biases)[0] in {
            2,
            3,
            4,
        }

    def test_remove_bias(self):
        net1 = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=5,
            forward_weights={
                0: ((4, 1, 1, 1),),
                1: ((4, 1, 1, 1),),
                4: ((2, 1, 1, 1), (3, 1, 1, 1),),
            },
            recurrent_weights={},
            biases={4: 5},
        )
        net2 = network._remove_random_bias(net1, _TEST_CONFIG)
        assert len(net2.biases) == len(net1.biases) - 1
        assert 4 not in net2.biases

        network.visualize(net1, "test_remove_bias_before")
        network.visualize(net2, "test_remove_bias_after")

    def test_add_unit(self):
        num_units = 5
        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=num_units,
            forward_weights={
                0: ((4, 1, 1, 1),),
                1: ((4, 1, 1, 1),),
                4: ((2, 1, 1, 1), (3, 1, 1, 1),),
            },
            recurrent_weights={},
        )

        network.visualize(net, "test_add_node_before")
        g_before = network.get_encoding_length(
            net,
            allowed_activations=_TEST_CONFIG.allowed_activations,
            compress_encoding=_TEST_CONFIG.compress_grammar_encoding,
        )
        net = network._add_unit(
            net,
            edge=(4, 2),
            edge_type=network.FORWARD_CONNECTION,
            activation=network.LINEAR,
            unit_type=network.SUMMATION_UNIT,
            source_weight=network._Weight(sign=1, numerator=1, denominator=1),
            target_weight=network._Weight(sign=1, numerator=3, denominator=4),
        )
        network.visualize(net, "test_add_node_after")
        g_after = network.get_encoding_length(
            net,
            allowed_activations=_TEST_CONFIG.allowed_activations,
            compress_encoding=_TEST_CONFIG.compress_grammar_encoding,
        )
        print(f"G before={g_before:.2f}\t G after={g_after:.2f}")
        assert network.get_num_units(net) == num_units + 1
        assert network.get_total_connections(net, include_biases=False) == 5

        assert 5 in net.forward_connections[4]
        assert 5 in net.forward_connections
        assert 2 in net.forward_connections[5]
        assert 5 in net.reverse_forward_connections
        assert 4 in net.reverse_forward_connections[5]
        assert 5 in net.reverse_forward_connections[2]

    def test_remove_unit_which_creates_loop(self):
        num_units = 3
        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=num_units,
            forward_weights={0: ((2, 1, 1, 1),), 2: ((1, 1, 1, 1), (0, 1, 1, 1)),},
            recurrent_weights={},
        )
        network.visualize(net, "test_remove_unit_which_creates_loop_before")
        net_2 = network.remove_unit(net, 2)
        network.visualize(net_2, "test_remove_unit_which_creates_loop_after")

        print(net_2.forward_connections)
        print(net_2.reverse_forward_connections)

        assert (0, 0) not in net_2.forward_weights
        assert 0 not in net_2.forward_connections[0]
        assert 0 not in net_2.reverse_forward_connections

    def test_remove_unit_with_recurrent_loop(self):
        num_units = 3
        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=num_units,
            forward_weights={0: ((2, 1, 1, 1),), 2: ((1, 1, 1, 1),),},
            recurrent_weights={2: ((2, 1, 1, 1),),},
        )
        network.visualize(net, "test_remove_unit_with_recurrent_loop")
        net2 = network.remove_unit(net, 2)
        network.visualize(net2, "test_remove_unit_with_recurrent_loop_after")
        assert 2 not in net2.forward_connections
        assert 2 not in net2.reverse_forward_connections
        assert 2 not in net2.recurrent_connections
        assert 2 not in net2.reverse_recurrent_connections
        assert (2, 2) not in net2.recurrent_weights

    def test_remove_chain_of_units(self):
        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=5,
            forward_weights={
                0: ((2, 1, 1, 1), (1, 1, 5, 1)),
                2: ((3, 1, 2, 1),),
                3: ((4, 1, 3, 1), (1, 1, 6, 1)),
                4: ((1, 1, 4, 1),),
            },
            recurrent_weights={},
            activations={2: network.LINEAR, 3: network.SIGMOID, 4: network.RELU},
        )
        network.visualize(net, "test_remove_chain_of_units_before")

        net_after = network.remove_unit(net, 4)
        network.visualize(net_after, "test_remove_chain_of_units_step_1")
        net_after = network.remove_unit(net_after, 3)
        network.visualize(net_after, "test_remove_chain_of_units_step_2")
        net_after = network.remove_unit(net_after, 2)
        network.visualize(net_after, "test_remove_chain_of_units_after")

        target_net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=2,
            forward_weights={0: ((1, 1, 5, 1),),},
            recurrent_weights={},
        )

        assert net_after == target_net

    def test_add_random_connection(self):
        num_units = 5
        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=num_units,
            forward_weights={
                0: ((4, 1, 1, 1),),
                1: ((4, 1, 1, 1),),
                4: ((2, 1, 1, 1), (3, 1, 1, 1),),
            },
            recurrent_weights={},
        )

        network.visualize(net, "test_add_random_connection_before")
        net_2 = network._add_random_connection(net, _TEST_CONFIG)
        network.visualize(net_2, "test_add_random_connection_after")

        assert (
            network.get_total_connections(net_2, include_biases=False)
            == network.get_total_connections(net, include_biases=False) + 1
        )

    def test_get_number_of_possible_connections(self):
        net = network.make_custom_network(
            input_size=2,
            output_size=3,
            num_units=9,
            forward_weights={},
            recurrent_weights={},
        )
        network.visualize(net, "a")
        assert (
            network._get_number_of_possible_connections(
                net, edge_type=network.FORWARD_CONNECTION
            )
            == 38
        )
        assert (
            network._get_number_of_possible_connections(
                net, edge_type=network.RECURRENT_CONNECTION
            )
            == 42
        )

    def test_mutate_weight(self):
        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=5,
            forward_weights={
                0: ((4, 1, 1, 1),),
                1: ((4, 1, 2, 1),),
                4: ((2, 1, 3, 1), (3, 1, 4, 1),),
            },
            recurrent_weights={},
        )

        network.visualize(net, "test_mutate_weight_before")
        net_after = network._mutate_weight(net, _TEST_CONFIG)
        network.visualize(net_after, "test_mutate_weight_after")

        assert net.forward_weights != net_after.forward_weights

    def test_add_random_unit(self):
        num_units = 5
        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=num_units,
            forward_weights={
                0: ((4, 1, 7, 1),),
                1: ((4, 1, 2, 1),),
                4: ((2, 1, 3, 1), (3, 1, 4, 1),),
            },
            recurrent_weights={},
        )

        network.visualize(net, "test_add_random_unit_before")
        g_before = network.get_encoding_length(
            net,
            allowed_activations=_TEST_CONFIG.allowed_activations,
            compress_encoding=_TEST_CONFIG.compress_grammar_encoding,
        )
        net_after = network._add_random_unit(net, _TEST_CONFIG)
        network.visualize(net_after, "test_add_random_unit_after")
        g_after = network.get_encoding_length(
            net_after,
            allowed_activations=_TEST_CONFIG.allowed_activations,
            compress_encoding=_TEST_CONFIG.compress_grammar_encoding,
        )
        print(f"G before={g_before:.2f}\t G after={g_after:.2f}")

        assert network.get_num_units(net_after) == 6
        assert network.get_total_connections(net_after, include_biases=False) == 5

        net_removed = network.remove_unit(net_after, unit_to_remove=5)
        network.visualize(net_removed, "test_add_random_unit_removed")

        assert net_removed == net

    def test_remove_node_incoming_recurrent(self):
        num_units = 5
        net = network.make_custom_network(
            input_size=3,
            output_size=1,
            num_units=num_units,
            forward_weights={
                0: ((4, 1, 3, 1),),
                4: ((3, 1, 17, 1),),
                2: ((4, -1, 2, 1),),
            },
            recurrent_weights={
                1: ((4, 1, 5, 1),),
                2: ((4, -1, 2, 1),),
                4: ((3, -1, 17, 1),),
            },
        )
        assert network.get_num_units(net) == 5

        network.visualize(net, "test_remove_node_recurrent_before")
        net_removed = network.remove_unit(net, 4)
        network.visualize(net_removed, "test_remove_node_recurrent_after")

        assert (1, 3) in net_removed.recurrent_weights
        assert net_removed.recurrent_weights[(1, 3)] == network._Weight(1, 5, 1)
        assert (1, 3) not in net_removed.forward_weights

        assert 3 in net_removed.recurrent_connections[1]
        assert 1 in net_removed.reverse_recurrent_connections[3]

        assert (2, 3) in net_removed.forward_weights
        assert (2, 3) in net_removed.recurrent_weights

        assert net_removed.forward_weights[(2, 3)] == network._Weight(-1, 2, 1)
        assert net_removed.recurrent_weights[(2, 3)] == network._Weight(-1, 2, 1)

        assert 4 not in net_removed.forward_connections
        assert 4 not in net_removed.reverse_forward_connections[3]
        assert 4 not in net_removed.recurrent_connections

    def test_remove_isolated_node(self):
        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=5,
            forward_weights={4: ((2, 1, 2, 1), (3, 1, 3, 1)),},
            recurrent_weights={},
        )
        network.visualize(net, "test_remove_isolated_node_before")
        net_removed = network.remove_unit(net, 4)
        network.visualize(net_removed, "test_remove_isolated_node_after")

        assert (4, 2) not in net_removed.forward_weights
        assert (4, 3) not in net_removed.forward_weights
        assert 4 not in net_removed.forward_connections
        assert 2 not in net_removed.reverse_forward_connections
        assert 3 not in net_removed.reverse_forward_connections

    def test_remove_node(self):
        num_units = 6
        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=num_units,
            forward_weights={
                0: ((4, 1, 1, 1),),
                4: ((2, 1, 2, 1),),
                5: ((3, 1, 3, 1), (5, 1, 9, 1)),
            },
            recurrent_weights={5: ((1, 1, 7, 1), (5, 1, 19, 1))},
            activations={5: network.SIGMOID},
        )
        assert network.get_num_units(net) == 6

        network.visualize(net, "test_remove_node_before")
        net = network.remove_unit(net, 4)
        network.visualize(net, "test_remove_node_after")

        assert network.get_num_units(net) == 5
        assert (4, 2) not in network.get_forward_weights(net)
        assert net.forward_weights[(0, 2)] == network._Weight(1, 1, 1)
        assert net.activations[4] == network.SIGMOID
        assert net.reverse_forward_connections[2] == frozenset([0])

    def test_remove_node_2(self):
        num_units = 5
        net = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=num_units,
            forward_weights={
                0: ((3, 1, 1, 1),),
                1: ((4, 1, 1, 1), (3, 1, 6, 1)),
                4: ((3, 1, 5, 1),),
                3: ((2, 1, 1, 1),),
            },
            recurrent_weights={1: ((2, 1, 1, 1),)},
            activations={3: network.SIGMOID, 4: network.SQUARE},
        )

        network.visualize(net, "test_remove_node_2_before")
        net = network.remove_unit(net, 4)
        network.visualize(net, "test_remove_node_2_after")

        assert net.forward_connections == {0: {3}, 3: {2}, 1: {3}}
        assert net.reverse_forward_connections == {3: {0, 1}, 2: {3}}
        assert net.recurrent_connections == {1: {2}}
        assert net.reverse_recurrent_connections == {2: {1}}

        forward_edges = network._get_edges(net, network.FORWARD_CONNECTION)
        assert forward_edges == {(0, 3), (1, 3), (3, 2)}

        recurrent_edges = network._get_edges(net, network.RECURRENT_CONNECTION)
        assert recurrent_edges == {(1, 2)}

        assert frozenset(net.forward_weights.keys()) == forward_edges

        # Verify existing bypassing edge is kept.
        assert net.forward_weights[(1, 3)] == network._Weight(
            sign=1, numerator=6, denominator=1
        )

    def test_remove_node_3(self):
        num_units = 5
        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=num_units,
            forward_weights={
                0: ((3, 1, 3, 1), (4, 1, 4, 1)),
                1: ((2, 1, 2, 1),),
                4: ((3, 1, 3, 1),),
            },
            recurrent_weights={},
        )

        assert network._get_removable_units(net) == (4,)

        network.visualize(net, "test_remove_node_3_before")
        net_after = network.remove_unit(net, 4)
        network.visualize(net_after, "test_remove_node_3_after")

        assert net_after.forward_connections == {0: {3}, 1: {2}}
        assert net_after.reverse_forward_connections == {3: {0}, 2: {1}}
        # Verify immutability.
        assert net.forward_connections == {
            0: {3, 4},
            1: {2},
            4: {3},
        }
        assert (0, 4) not in net_after.forward_weights
        assert (4, 3) not in net_after.forward_weights

    def test_remove_node_with_loop(self):
        num_units = 5
        net = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=num_units,
            forward_weights={
                0: ((4, 1, 4, 1,),),
                4: ((3, 1, 3, 1,),),
                3: ((2, 1, 2, 1,),),
                1: ((2, 1, 2, 1),),
            },
            recurrent_weights={3: ((3, 1, 3, 1,),),},
            activations={3: network.SQUARE, 4: network.RELU},
        )
        network.visualize(net, "test_remove_node_with_loop_before")
        net_after = network.remove_unit(net, 3)
        network.visualize(net_after, "test_remove_node_with_loop_after")

        assert net_after.forward_connections == {3: {2}, 1: {2}, 0: {3}}
        assert net_after.reverse_forward_connections == {2: {1, 3}, 3: {0}}
        edges = network._get_edges(net_after, network.FORWARD_CONNECTION)
        print(edges)
        assert edges == {
            (0, 3),
            (3, 2),
            (1, 2),
        }

    def test_remove_node_4(self):
        net = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=6,
            forward_weights={
                0: ((3, 1, 1, 2), (5, 1, 1, 1)),
                3: ((4, 1, 1, 3),),
                4: ((1, 1, 1, 4),),
                5: ((1, 1, 1, 1),),
            },
            recurrent_weights={},
        )
        network.visualize(net, "test_remove_node_4_before")
        net_2 = network.remove_unit(net, unit_to_remove=4)
        network.visualize(net_2, "test_remove_node_4_after")

    def test_loop(self):
        binary_corpus = corpora.optimize_for_feeding(
            corpora.make_binary_addition(min_n=0, max_n=20)
        )
        net = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=3,
            forward_weights={
                0: ((0, 1, 1, 5), (1, -1, 1, 1), (2, -1, 2, 1),),
                1: ((0, -1, 1, 1), (2, 1, 6, 1,)),
            },
            recurrent_weights={0: ((0, 1, 1, 1),)},
            biases={2: -4},
            activations={0: network.RELU, 1: network.SQUARE, 2: network.SIGMOID},
        )
        network.visualize(net, "test_loop")
        net = network.calculate_fitness(net, binary_corpus, _TEST_CONFIG)
        print(net)

        loop_edges, topological_order = network.run_dfs(
            units=network.get_units(net),
            reverse_connections=net.reverse_forward_connections,
            input_units=net.input_units_range,
            output_units=net.output_units_range,
        )
        assert loop_edges == {(0, 0), (0, 1)}

        assert network.get_loop_edges(net)

        net_without_loops = network.fix_loops(net)

        assert not network.get_loop_edges(net_without_loops)
        assert network.get_loop_edges(net)  # Test immutability.

        net_without_loops = network.invalidate_fitness(net_without_loops)
        net_without_loops = network.calculate_fitness(
            net_without_loops, binary_corpus, _TEST_CONFIG
        )

        network.visualize(net_without_loops, "net_without_loops")

        assert (
            net_without_loops.fitness.data_encoding_length
            == net.fitness.data_encoding_length
        )

    def test_loop_detection_for_recurrent_path(self):
        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=4,
            forward_weights={2: ((3, 1, 1, 1),), 3: ((2, 1, 1, 1),)},
            recurrent_weights={2: ((1, 1, 1, 1),)},
        )
        network.visualize(net, "test_loop_detection_for_recurrent_path")

        loop_edges, topological_sort = network.topological_sort_through_time(net)
        print(loop_edges)
        assert loop_edges == {(2, 3)} or loop_edges == {(3, 2)}

    def test_feed_forward(self):
        num_units = 5
        net = network.make_custom_network(
            input_size=3,
            output_size=2,
            num_units=num_units,
            forward_weights={
                0: ((3, 1, 1, 1),),
                1: ((3, 1, 1, 1), (4, 1, 1, 1),),
                2: ((4, 1, 1, 1),),
            },
            recurrent_weights={},
        )
        network.visualize(net, "test_feed_forward")

        input_sequence = np.array([[[2.0, 1.0, 3.0]]])
        outputs = network.predict_probabs(
            net,
            input_sequence,
            softmax_outputs=False,
            recurrent_connections=True,
            truncate_large_values=False,
        )
        print(outputs)

        assert np.all(outputs - np.array([[[0.42, 0.57]]]) < _EPSILON)

    def test_feed_forward_recurrent(self):
        input_sequence = np.array([[[2.0], [4.0]]])
        target_sequence = np.array([[[0, 1], [0, 1]]])

        num_units = 5
        test_corpus = corpora.Corpus("test", input_sequence, target_sequence)

        net = network.make_custom_network(
            input_size=1,
            output_size=2,
            num_units=num_units,
            forward_weights={
                0: ((1, 1, 1, 1), (3, 1, 1, 1),),
                3: ((4, 1, 1, 1),),
                4: ((2, 1, 1, 1),),
            },
            recurrent_weights={4: ((3, 1, 1, 1),)},
        )

        network.visualize(net, "test_feed_forward_recurrent")

        outputs = network.predict_probabs(
            net,
            input_sequence,
            softmax_outputs=False,
            recurrent_connections=True,
            truncate_large_values=True,
        )
        print(outputs)

        expected_outputs = np.array([[[0.5, 0.5], [0.4, 0.6]]])

        assert np.all(np.abs(outputs - expected_outputs) < _EPSILON)

        d_g, _ = network._get_data_given_g_encoding_length(
            net, test_corpus, _TEST_CONFIG
        )
        print(d_g)
        assert np.abs(d_g - 1.73) < _EPSILON

    def test_echo_network(self):
        echo_corpus = corpora.make_identity(
            sequence_length=100, batch_size=100, num_classes=2
        )

        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=4,
            forward_weights={0: ((2, 1, 1, 1),), 1: ((3, 1, 1, 1),),},
            recurrent_weights={},
        )

        network.visualize(net, "test_echo_network")
        net = network.calculate_fitness(net, echo_corpus, _TEST_CONFIG)
        print(net)

    def test_found_echo_network(self):
        echo_corpus = corpora.optimize_for_feeding(
            corpora.make_identity(sequence_length=100, batch_size=100, num_classes=2)
        )

        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=4,
            forward_weights={
                0: ((1, -1, 1, 1),),
                1: ((3, 1, 4, 1),),
                3: ((2, -1, 4, 1),),
            },
            recurrent_weights={},
        )

        network.visualize(net, "test_weird_echo_network")
        net = network.calculate_fitness(net, echo_corpus, _TEST_CONFIG)
        print(net)

    def test_identity_network_accuracy(self):
        corpus = corpora.make_identity(num_classes=2, batch_size=1)

        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=4,
            forward_weights={0: ((2, 1, 1, 1),), 1: ((3, 1, 1, 1),),},
            recurrent_weights={},
        )

        network.visualize(net, "identity_net")

        net = network.calculate_fitness(net, corpus, _TEST_CONFIG)
        print(net)
        assert net.fitness.data_encoding_length == 0.0
        assert net.fitness.accuracy == 1.0

    def test_masked_input(self):
        input_classes = np.array([0, 1] * 5 + [corpora.MASK_VALUE] * 50)
        target_classes = np.array([0, 1] * 5 + [corpora.MASK_VALUE] * 50)
        masked_corpus = corpora.make_one_hot_corpus(
            "masked_corpus",
            input_classes=input_classes,
            target_classes=target_classes,
            num_input_classes=2,
            num_target_classes=2,
        )
        masked_corpus = corpora.precompute_mask_idxs(masked_corpus)

        net = network.make_custom_network(
            input_size=2,
            output_size=2,
            num_units=4,
            forward_weights={0: ((2, 1, 1, 1),), 1: ((3, 1, 1, 1),),},
            recurrent_weights={},
        )

        network.visualize(net, "identity_net_masked")
        net = network.calculate_fitness(net, masked_corpus, _TEST_CONFIG)
        print(net)
        # D:G shouldn't change when masked length changes.
        assert net.fitness.data_encoding_length == 0.0
        assert net.fitness.accuracy == 1.0

    def test_binary_output(self):
        input_sequence = np.array([[[1], [1]]], dtype=utils.FLOAT_DTYPE)
        target_sequence = np.array([[[0], [1]]], dtype=utils.FLOAT_DTYPE)

        binary_corpus = corpora.Corpus(
            "binary_corpus",
            input_sequence=input_sequence,
            target_sequence=target_sequence,
        )

        weight_val = 5

        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=2,
            forward_weights={0: ((1, 1, weight_val, 1),)},
            recurrent_weights={},
            activations={1: network.SIGMOID},
        )

        network.visualize(net, "binary_output_net")
        net = network.calculate_fitness(net, binary_corpus, _TEST_CONFIG)
        print(net)
        target_encoding_length = -np.sum(
            np.log2([network._sigmoid(weight_val), 1 - network._sigmoid(weight_val),])
        )
        print(target_encoding_length)
        assert abs(net.fitness.data_encoding_length - target_encoding_length) < _EPSILON
        assert net.fitness.accuracy == 0.5

    def test_conll_addition_network(self):
        # Network reported in CoNLL 2020 paper.
        binary_corpus = corpora.make_binary_addition(min_n=0, max_n=20)

        net = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=4,
            forward_weights={
                0: ((1, 1, 1, 1),),
                1: ((2, 1, 1, 1), (3, 7, 1, 1),),
                3: ((2, -1, 4, 1),),
            },
            recurrent_weights={3: ((1, 1, 1, 1),),},
            activations={1: network.SQUARE, 3: network.SIGMOID},
            biases={3: -16},
        )
        net_name = "conll_addition_net"
        network.visualize(net, net_name)
        net = network.calculate_fitness(net, binary_corpus, _TEST_CONFIG,)
        network.save(net, net_name)
        print(net)
        assert np.round(net.fitness.data_encoding_length, 2) == 0.67
        assert net.fitness.accuracy == 1.0
        assert net.fitness.grammar_encoding_length == 119

    def test_emmanual_addition_network_1(self):
        # Network uses the floor function.
        binary_corpus = corpora.make_binary_addition(min_n=0, max_n=100)
        net = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=5,
            forward_weights={
                0: ((3, 1, 1, 1),),
                1: ((3, 1, 1, 1),),
                3: ((2, 1, 1, 1), (4, 1, 1, 2),),
                4: ((2, -1, 2, 1),),
            },
            recurrent_weights={4: ((3, 1, 1, 1),),},
            activations={4: network.FLOOR},
        )
        net_name = "emmanuel_binary_output_net_1"
        network.visualize(net, net_name)
        net = network.calculate_fitness(net, binary_corpus, _TEST_CONFIG)
        network.save(net, net_name)
        print(net)
        assert net.fitness.data_encoding_length == 0.0
        assert net.fitness.accuracy == 1.0

    def test_emmanual_addition_network_2(self):
        # A non-perfect attempt to avoid floor function, not 100% accurate.
        binary_corpus = corpora.make_binary_addition(min_n=0, max_n=100)
        net = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=5,
            forward_weights={
                0: ((4, -1, 1, 1),),
                1: ((4, 1, 1, 1),),
                3: ((2, 1, 1, 2),),
                4: ((2, -1, 1, 1),),
            },
            recurrent_weights={
                0: ((3, 1, 1, 1),),
                1: ((3, 1, 1, 1),),
                2: ((3, -1, 1, 1),),
                3: ((3, 1, 1, 1),),
            },
            activations={2: network.SQUARE, 4: network.SQUARE},
        )
        net_name = "emmanuel_binary_output_net_2"
        network.visualize(net, net_name)
        net = network.calculate_fitness(net, binary_corpus, _TEST_CONFIG)
        network.save(net, net_name)
        print(net)

    def test_emmanuel_an_bn_network(self):
        anbn_corpus = corpora.optimize_for_feeding(
            corpora._make_ain_bjn_ckn_dtn_corpus(
                n_values=tuple(range(50)),
                multipliers=(1, 1, 0, 0),
                prior=0.1,
                sort_by_length=False,
            )
        )
        net = network.make_custom_network(
            input_size=3,
            output_size=3,
            num_units=6,
            forward_weights={
                1: ((0, -1, 1, 1),),
                2: ((0, 1, 1, 1), (4, -1, 1, 2)),
                0: ((3, 1, 1, 1),),
                4: ((5, -1, 1, 1), (3, -1, 1, 1)),
                3: ((5, -1, 1, 1),),
            },
            recurrent_weights={0: ((0, 1, 1, 1),)},
            biases={4: 0.5, 5: 1},
            activations={3: network.RELU},
        )
        network.visualize(
            net, "emmanuel_an_bn_net", class_to_label=anbn_corpus.vocabulary
        )
        net = network.calculate_fitness(net, anbn_corpus, _TEST_CONFIG)
        print(net)
        assert net.fitness.data_encoding_length == 1275

    def test_naive_123456_network(self):
        corpus_123456 = corpora.make_123_n_pattern_corpus(
            base_sequence_length=6, sequence_length=1200
        )
        edge_weight = 1
        forward_weights = {
            i: ((((i + 1) % 6) + 6, edge_weight, 1, 1),) for i in range(6)
        }
        net = network.make_custom_network(
            input_size=6,
            output_size=6,
            num_units=12,
            forward_weights=forward_weights,
            recurrent_weights={},
            activations={i: network.SQUARE for i in range(12)},
        )
        net_name = "123456_net"
        network.visualize(net, net_name)
        net = network.calculate_fitness(net, corpus_123456, _TEST_CONFIG)
        network.save(net, net_name)
        print(net)
        assert net.fitness.accuracy == 1.0

    def test_emmanuel_xor_triplet_network(self):
        xor_corpus = corpora.make_elman_xor_binary(3000, 1)
        net = manual_nets.make_emmanuel_triplet_xor_network()
        net_name = "emmanuel_triplet_xor_net"
        network.visualize(net, net_name)
        net = network.calculate_fitness(net, xor_corpus, _TEST_CONFIG)
        network.save(net, net_name)
        print(net)
        assert net.fitness.data_encoding_length == 2000
        print(hash(net))

    def test_stable_hash(self):
        net1 = network.make_custom_network(
            input_size=3,
            output_size=3,
            num_units=7,
            forward_weights={0: ((6, 1, 1, 1), (5, 1, 1, 1)), 2: ((3, 1, 1, 1),)},
            recurrent_weights={1: ((3, 1, 1, 1),), 3: ((4, 1, 1, 1),)},
            activations={3: network.SIGMOID, 6: network.SQUARE},
            biases={3: 4, 6: 3},
            unit_types={5: network.MULTIPLICATION_UNIT, 3: network.MULTIPLICATION_UNIT},
        )
        net2 = network.make_custom_network(
            input_size=3,
            output_size=3,
            num_units=7,
            forward_weights={2: ((3, 1, 1, 1),), 0: ((5, 1, 1, 1), (6, 1, 1, 1)),},
            recurrent_weights={3: ((4, 1, 1, 1),), 1: ((3, 1, 1, 1),),},
            activations={6: network.SQUARE, 3: network.SIGMOID},
            biases={6: 3, 3: 4},
            unit_types={3: network.MULTIPLICATION_UNIT, 5: network.MULTIPLICATION_UNIT},
        )
        network.visualize(net1, "net1_hash")
        network.visualize(net2, "net2_hash")

        assert net1 == net2
        assert hash(net1) == hash(net2)

    def test_an_bn_handmade_network(self):
        prior = 0.1

        utils.seed(1)
        anbn_corpus = corpora.make_ain_bjn_ckn_dtn(
            batch_size=100, prior=prior, multipliers=(1, 1, 0, 0)
        )
        anbn_corpus = corpora.precompute_mask_idxs(anbn_corpus)

        outputs = corpora.an_bn_handmade_net(anbn_corpus.input_sequence, prior=prior,)
        outputs = network.normalize_multiclass(outputs)

        input_mask_flat = anbn_corpus.input_mask.flatten()

        predicted_classes_flat = outputs.argmax(axis=-1).flatten()
        predicted_classes_flat = np.delete(predicted_classes_flat, input_mask_flat)

        target_classes_flat = anbn_corpus.target_sequence.argmax(axis=-1).flatten()
        target_classes_flat = target_classes_flat[input_mask_flat]

        output_probabs = outputs.reshape(-1, outputs.shape[-1])
        output_probabs = output_probabs[input_mask_flat]
        target_classes_probabs = output_probabs[
            range(output_probabs.shape[0]), target_classes_flat
        ]

        accuracy = np.sum(predicted_classes_flat == target_classes_flat) / len(
            target_classes_flat
        )
        print("Accuracy:", accuracy)

        data_encoding_length = float(np.sum(-np.log2(target_classes_probabs)))
        print("D:G", data_encoding_length)

        assert data_encoding_length - 462.76 <= 0.01

    def test_early_stop_binary(self):
        inputs = np.ones((1, 3, 1))
        targets = np.copy(inputs)
        corpus = corpora.Corpus(
            name="early_stop_binary", input_sequence=inputs, target_sequence=targets
        )
        corpus = corpora.optimize_for_feeding(corpus)
        # A network that outputs 0 after the first step.
        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=2,
            forward_weights={0: ((1, 1, 1, 1),),},
            recurrent_weights={0: ((0, -1, 1, 1),),},
            activations={1: network.RELU},
        )
        net = network.calculate_fitness(net, corpus, _TEST_CONFIG)
        assert np.isinf(net.fitness.data_encoding_length)
        assert net.fitness.accuracy == 0.0

        early_stop = False
        try:
            network.predict_probabs(
                net=net,
                input_sequence=inputs,
                recurrent_connections=True,
                truncate_large_values=False,
                softmax_outputs=False,
                inputs_per_time_step=corpus.input_values_per_time_step,
                targets_mask=corpus.targets_mask,
                input_mask=corpus.input_mask,
            )
        except network._InvalidNet:
            early_stop = True
        assert early_stop

    def test_dont_early_stop_on_all_zeros(self):
        # Verify no early stop on all-zero outputs which are converted to a uniform probability.
        inputs = np.zeros((1, 3, 3))
        inputs[[0, 0, 0], [0, 1, 2], [1, 1, 1]] = 1.0
        targets = np.copy(inputs)
        corpus = corpora.Corpus(
            name="early_stop_multiclass", input_sequence=inputs, target_sequence=targets
        )
        corpus = corpora.optimize_for_feeding(corpus)
        net = network.make_custom_network(
            input_size=3,
            output_size=3,
            num_units=6,
            forward_weights={},
            recurrent_weights={},
        )
        network.visualize(net, "early_stop_net")
        early_stop = False
        try:
            outputs = network.predict_probabs(
                net=net,
                input_sequence=inputs,
                recurrent_connections=True,
                truncate_large_values=False,
                softmax_outputs=False,
                inputs_per_time_step=corpus.input_values_per_time_step,
                targets_mask=corpus.targets_mask,
                input_mask=corpus.input_mask,
            )
            print(outputs)
        except network._InvalidNet:
            early_stop = True
        assert not early_stop

        net = network.calculate_fitness(net, corpus, _TEST_CONFIG)
        assert not np.isinf(net.fitness.data_encoding_length)
        assert net.fitness.accuracy == 0.0

    def test_early_stop_multiclass(self):
        inputs = np.zeros((1, 3, 3))
        inputs[[0, 0, 0], [0, 1, 2], [1, 1, 1]] = 1.0
        targets = np.copy(inputs)
        corpus = corpora.Corpus(
            name="early_stop_multiclass", input_sequence=inputs, target_sequence=targets
        )
        corpus = corpora.optimize_for_feeding(corpus)
        # A network that outputs the correct class for the first step, then wrong on second step.
        net = network.make_custom_network(
            input_size=3,
            output_size=3,
            num_units=6,
            forward_weights={
                0: ((3, 1, 1, 1),),
                1: ((4, 1, 1, 1),),
                2: ((5, 1, 1, 1),),
            },
            recurrent_weights={1: ((1, -1, 1, 1), (2, 1, 1, 1))},
        )
        early_stop = False
        try:
            network.predict_probabs(
                net=net,
                input_sequence=inputs,
                recurrent_connections=True,
                truncate_large_values=False,
                softmax_outputs=False,
                inputs_per_time_step=corpus.input_values_per_time_step,
                targets_mask=corpus.targets_mask,
                input_mask=corpus.input_mask,
            )
        except network._InvalidNet:
            early_stop = True
        assert early_stop

        net = network.calculate_fitness(net, corpus, _TEST_CONFIG)
        assert np.isinf(net.fitness.data_encoding_length)
        assert net.fitness.accuracy == 0.0

    def test_incremental_network_buildup(self):
        # Verify series of mutations can lead to found net.
        net = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=3,
            forward_weights={},
            recurrent_weights={},
            activations={},
            biases={},
        )
        network.visualize(net, "net_evolution_step_1")
        net = network._add_connection(
            net,
            edge=(1, 2),
            edge_type=network.FORWARD_CONNECTION,
            weight=_evolve_weight(network._Weight(1, 1, 1)),
        )
        network.visualize(net, "net_evolution_step_2")
        net = network._add_unit(
            net,
            edge=(1, 2),
            edge_type=network.FORWARD_CONNECTION,
            activation=network.LINEAR,
            unit_type=network.SUMMATION_UNIT,
            source_weight=net.forward_weights[(1, 2)],
            target_weight=network._Weight(1, 1, 1),
        )
        network.visualize(net, "net_evolution_step_3")
        net = network._replace_activation(net, unit=3, new_activation=network.SQUARE)
        network.visualize(net, "net_evolution_step_4")
        net = network._add_connection(
            net,
            edge=(3, 2),
            edge_type=network.FORWARD_CONNECTION,
            weight=_evolve_weight(network._Weight(1, 7, 1)),
        )
        network.visualize(net, "net_evolution_step_5")

        net = network._add_unit(
            net,
            edge=(3, 2),
            edge_type=network.FORWARD_CONNECTION,
            activation=network.LINEAR,
            unit_type=network.SUMMATION_UNIT,
            source_weight=net.forward_weights[(3, 2)],
            target_weight=network._Weight(1, 1, 1),
        )
        network.visualize(net, "net_evolution_step_6")
        net = network._replace_activation(net, unit=4, new_activation=network.SIGMOID)
        network.visualize(net, "net_evolution_step_7")
        net = network._add_connection(
            net,
            edge=(3, 2),
            edge_type=network.FORWARD_CONNECTION,
            weight=_evolve_weight(network._Weight(1, 1, 1)),
        )
        network.visualize(net, "net_evolution_step_8")
        net = network._add_connection(
            net,
            edge=(4, 3),
            edge_type=network.RECURRENT_CONNECTION,
            weight=_evolve_weight(network._Weight(1, 1, 1)),
        )
        network.visualize(net, "net_evolution_step_9")
        net = network._replace_weight(
            net,
            new_weight=_evolve_weight(network._Weight(-1, 4, 1)),
            edge_or_bias=(4, 2),
            edge_type=network.FORWARD_CONNECTION,
        )
        network.visualize(net, "net_evolution_step_10")
        net = network._add_connection(
            net,
            edge=(4, 3),
            edge_type=network.RECURRENT_CONNECTION,
            weight=_evolve_weight(network._Weight(1, 1, 1)),
        )
        network.visualize(net, "net_evolution_step_11")
        net = network._replace_bias(
            net, unit=4, weight=_evolve_weight(network._Weight(-1, 16, 1))
        )
        network.visualize(net, "net_evolution_step_12")
        net = network._add_connection(
            net,
            edge=(0, 3),
            edge_type=network.FORWARD_CONNECTION,
            weight=_evolve_weight(network._Weight(1, 1, 1)),
        )
        network.visualize(net, "net_evolution_step_14")
        net = network._add_unit(
            net,
            edge=(0, 3),
            edge_type=network.FORWARD_CONNECTION,
            activation=network.LINEAR,
            unit_type=network.SUMMATION_UNIT,
            source_weight=net.forward_weights[(0, 3)],
            target_weight=_evolve_weight(network._Weight(1, 1, 1)),
        )
        network.visualize(net, "net_evolution_step_15")

        corpus = corpora.optimize_for_feeding(
            corpora.make_binary_addition(min_n=0, max_n=20)
        )
        network.visualize(net, "net_evolution_step_16")

        net = network.calculate_fitness(net, corpus, _TEST_CONFIG)
        print(net)
        assert net.fitness.accuracy == 1.0

    def test_iclr_addition_network(self):
        # Network reported in ICLR 2020 submission.
        corp = corpora.make_binary_addition(min_n=0, max_n=20)

        net = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=4,
            forward_weights={
                0: ((1, 1, 1, 1),),
                1: ((2, 1, 1, 1), (3, 1, 7, 1),),
                3: ((2, -1, 4, 1),),
            },
            recurrent_weights={3: ((1, 1, 1, 1),)},
            activations={1: network.SQUARE, 3: network.SIGMOID},
            biases={3: -16},
        )

        net_io_protected = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=6,
            forward_weights={
                0: ((4, 1, 1, 1),),
                1: ((5, 1, 1, 1),),
                4: ((5, 1, 1, 1),),
                5: ((2, 1, 1, 1), (3, 1, 7, 1),),
                3: ((2, -1, 4, 1),),
            },
            recurrent_weights={3: ((5, 1, 1, 1),)},
            activations={5: network.SQUARE, 3: network.SIGMOID},
            biases={3: -16},
        )

        net = network.calculate_fitness(net, corp, _TEST_CONFIG)
        net_io_protected = network.calculate_fitness(
            net_io_protected, corp, _TEST_CONFIG
        )
        print(net)
        print(net_io_protected)
        network.visualize(net, "iclr_addition")
        network.visualize(net_io_protected, "iclr_addition_io_protected")
        assert net.fitness.accuracy == 1.0

    def test_emmanual_xor_sliding_window_network(self):
        xor_corpus = corpora.make_elman_xor_binary(3000, 1)

        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=2,
            forward_weights={0: ((1, 1, 1, 1),),},
            recurrent_weights={0: ((1, -1, 1, 1),),},
            activations={1: network.SQUARE},
        )
        net_name = "emmanuel_xor_net"
        network.visualize(net, net_name)
        net = network.calculate_fitness(net, xor_corpus, _TEST_CONFIG)
        print(net)

        xor_corpus_optimized = corpora.optimize_for_feeding(xor_corpus)
        net_early_stop = network.calculate_fitness(
            network.invalidate_fitness(net), xor_corpus_optimized, _TEST_CONFIG
        )
        print(net_early_stop)
        assert np.isinf(net_early_stop.fitness.data_encoding_length)
        assert net_early_stop.fitness.accuracy == 0.0

    def test_bias(self):
        num_units = 5
        corpus = corpora.make_random_one_hot(3, 2)
        net = network.make_custom_network(
            input_size=3,
            output_size=2,
            num_units=num_units,
            forward_weights={
                0: ((3, 1, 1, 1),),
                1: ((3, 1, 1, 1), (4, 1, 1, 1),),
                2: ((4, 1, 1, 1),),
            },
            biases={3: 4, 4: 5},
            recurrent_weights={},
        )

        input_sequence = np.array([[[2.0, 1.0, 3.0]]])
        net = network.calculate_fitness(net, corpus, _TEST_CONFIG)
        softmax = network.predict_probabs(
            net,
            input_sequence,
            recurrent_connections=True,
            softmax_outputs=False,
            truncate_large_values=False,
        )
        print(softmax)
        network.visualize(net, "test_bias")

        assert np.all(softmax - np.array([[[0.437, 0.562]]]) < _EPSILON)

    def test_binary_echo_network(self):
        binary_corpus = corpora.optimize_for_feeding(
            corpora.make_identity_binary(100, 100)
        )
        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=2,
            forward_weights={0: ((1, 1, 100, 1),),},
            recurrent_weights={},
        )
        network.visualize(net, "binary_echo")
        net = network.calculate_fitness(net, binary_corpus, _TEST_CONFIG)
        print(net)
        assert net.fitness.accuracy == 1.0

    def test_recurrent_loop(self):
        binary_corpus = corpora.make_identity_binary(10, 1)
        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=2,
            forward_weights={0: ((1, 1, 1, 1),),},
            recurrent_weights={1: ((1, 1, 1, 1),),},
        )
        print(
            network.predict_probabs(
                net=net,
                input_sequence=np.ones((1, 10, 1)),
                softmax_outputs=False,
                recurrent_connections=True,
                truncate_large_values=False,
            )
        )
        network.visualize(net, "recurrent_loop")
        net = network.calculate_fitness(net, binary_corpus, _TEST_CONFIG)
        print(net)
        assert np.isinf(net.fitness.data_encoding_length)
        assert net.fitness.grammar_encoding_length == 41

    def test_calculate_d_g_with_weights(self):
        input_sequence = np.array(
            [
                [[1.0, 0.0, 0.0], [corpora.MASK_VALUE] * 3],
                [[0.0, 1.0, 0.0], [corpora.MASK_VALUE] * 3],
            ]
        )
        target_sequence = np.copy(input_sequence)

        non_weighted_corpus = corpora.Corpus("test", input_sequence, target_sequence)
        non_weighted_corpus = corpora.optimize_for_feeding(non_weighted_corpus)
        weighted_corpus = dataclasses.replace(
            non_weighted_corpus, sample_weights=(2, 10)
        )

        # Imperfect prediction so D:G != 0 and can be weighted.
        num_units = 6
        net = network.make_custom_network(
            input_size=3,
            output_size=3,
            num_units=num_units,
            forward_weights={
                0: ((3, 1, 1, 2),),
                1: ((4, 1, 1, 2),),
                2: ((5, 1, 1, 2),),
            },
            biases={3: 0.1, 4: 0.1, 5: 0.1},
            recurrent_weights={},
        )
        network.visualize(net, "weighted_d_g_identity_net")

        weighted_eval = network.calculate_fitness(net, weighted_corpus, _TEST_CONFIG)
        non_weighted_eval = network.calculate_fitness(
            net, non_weighted_corpus, _TEST_CONFIG
        )

        print(weighted_eval)
        print(non_weighted_eval)

        single_output_val = non_weighted_eval.fitness.data_encoding_length / 2
        assert (
            weighted_eval.fitness.data_encoding_length
            == 2 * single_output_val + 10 * single_output_val
        )

    def test_multiplication_unit(self):
        num_units = 4

        input_sequence = np.array([[[1, 2, 3], [4, 5, 6]]])

        net = network.make_custom_network(
            input_size=3,
            output_size=1,
            num_units=num_units,
            forward_weights={
                0: ((3, 1, 3, 1),),
                1: ((3, 1, 4, 1),),
                2: ((3, 1, 5, 1),),
            },
            recurrent_weights={},
            unit_types={
                0: network.SUMMATION_UNIT,
                1: network.SUMMATION_UNIT,
                2: network.SUMMATION_UNIT,
                3: network.MULTIPLICATION_UNIT,
            },
        )

        network.visualize(net, "multiplication_net")
        outputs = network.feed_sequence(
            net,
            input_sequence,
            recurrent_connections=True,
            truncate_large_values=False,
        )

        print(outputs)
        assert np.all(outputs.flatten() == [360, 7200])

    def test_multiplication_unit_recurrent(self):
        num_units = 3

        input_sequence = np.array([[[1], [2], [2], [2]]], dtype=utils.FLOAT_DTYPE)

        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=num_units,
            forward_weights={0: ((2, 1, 1, 1),), 2: ((1, 1, 1, 1),),},
            recurrent_weights={2: ((2, 1, 2, 1),),},
            unit_types={
                0: network.SUMMATION_UNIT,
                1: network.SUMMATION_UNIT,
                2: network.MULTIPLICATION_UNIT,
            },
        )

        network.visualize(net, "multiplication_net_recurrent")

        outputs = network.feed_sequence(
            net,
            input_sequence,
            recurrent_connections=True,
            truncate_large_values=False,
        )

        print(outputs)
        assert np.all(outputs.flatten() == [1, 4, 16, 64])

    def test_multiplication_unit_with_bias(self):
        num_units = 3

        input_sequence = np.array([[[1], [1], [1], [1]]], dtype=utils.FLOAT_DTYPE)

        utils.seed(100)

        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=num_units,
            forward_weights={0: ((2, 1, 1, 1),), 2: ((1, 1, 1, 1),),},
            recurrent_weights={2: ((2, 1, 2, 1),),},
            unit_types={
                0: network.SUMMATION_UNIT,
                1: network.SUMMATION_UNIT,
                2: network.MULTIPLICATION_UNIT,
            },
            biases={2: 2},
        )

        network.visualize(net, "multiplication_net_with_bias")

        outputs = network.feed_sequence(
            net,
            input_sequence,
            recurrent_connections=True,
            truncate_large_values=False,
        )

        print(outputs)
        assert np.all(outputs.flatten() == [2, 8, 32, 128])

    def test_emmanuel_dyck_2_net(self):
        # Make sure value truncation leaves enough depth in a stack.
        config = dataclasses.replace(_TEST_CONFIG, truncate_large_values=True)

        nesting_probab = 0.3
        utils.seed(100)
        corpus = corpora.make_dyck_n(
            batch_size=10_000,
            nesting_probab=nesting_probab,
            n=2,
            max_sequence_length=200,
        )
        corpus = corpora.optimize_for_feeding(corpus)

        original_emmanuel_net = manual_nets.make_emmanuel_dyck_2_network(
            nesting_probab=nesting_probab
        )
        net_with_io_units_protected = manual_nets.make_emmanuel_dyck_2_network_io_protection(
            nesting_probab=nesting_probab
        )

        network.visualize(
            original_emmanuel_net,
            "emmanuel_dyck_2_net",
            class_to_label=corpus.vocabulary,
        )
        network.visualize(
            net_with_io_units_protected,
            "emmanuel_dyck_2_net_io_protected",
            class_to_label=corpus.vocabulary,
        )

        original_emmanuel_net = network.calculate_fitness(
            original_emmanuel_net, corpus, config
        )
        print(original_emmanuel_net)
        assert (
            np.round(original_emmanuel_net.fitness.data_encoding_length, 2) == 29585.77
        )

        net_with_io_units_protected = network.calculate_fitness(
            net_with_io_units_protected, corpus, config
        )
        print(net_with_io_units_protected)
        assert (
            np.round(net_with_io_units_protected.fitness.data_encoding_length, 2)
            == 29585.77
        )

    def test_emmanuel_dyck_2_net_deep_nesting(self):
        # Disables large-value truncation to support a deeper stack.
        config = dataclasses.replace(_TEST_CONFIG, truncate_large_values=False)

        nesting_probab = 0.5
        utils.seed(100)
        corpus = corpora.make_dyck_n(
            batch_size=1000, nesting_probab=nesting_probab, n=2, max_sequence_length=200
        )
        corpus = corpora.optimize_for_feeding(corpus)

        net = manual_nets.make_emmanuel_dyck_2_network(nesting_probab=nesting_probab)

        net = network.calculate_fitness(net, corpus, config)
        assert net.fitness.data_encoding_length == 18220.0

    def test_stack_circuit(self):
        # Push '1' four times, pop four times.
        input_sequence = np.array(
            [[[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]]],
            dtype=utils.FLOAT_DTYPE,
        )
        print(input_sequence.shape)
        target_sequence = np.zeros((1, 8, 1), dtype=utils.FLOAT_DTYPE)
        corpus = corpora.Corpus(
            "test_stack_circuit",
            input_sequence=input_sequence,
            target_sequence=target_sequence,
        )

        num_units = 9

        """
        Units:
        0 - input to push gate and push value
        1 - input to pop gate

        2 - output (memory value)
        3 - memory unit
        4 - push input
        5 - push gate
        6 - shift-right memory
        7 - pop gate
        8 - mod2 (pop LSB)
        """
        net = network.make_custom_network(
            input_size=2,
            output_size=1,
            num_units=num_units,
            forward_weights={
                0: ((5, 1, 1, 1,), (4, 1, 1, 1),),
                1: ((7, 1, 1, 1),),
                3: ((2, 1, 1, 1), (8, 1, 1, 1),),
                4: ((5, 1, 1, 1),),
                5: ((3, 1, 1, 1),),
                6: ((7, 1, 1, 1),),
                7: ((3, 1, 1, 1),),
                # 8: ((2, 1, 1, 1),),
            },
            recurrent_weights={3: ((4, 1, 2, 1), (6, 1, 1, 2),),},
            unit_types={
                5: network.MULTIPLICATION_UNIT,  # Push gate.
                7: network.MULTIPLICATION_UNIT,  # Pop gate.
            },
            activations={6: network.FLOOR, 8: network.MODULO_2},
        )
        net = network.calculate_fitness(net, corpus, config=_TEST_CONFIG)
        print(net)

        network.visualize(net, "memory_circuit")

        outputs = network.feed_sequence(
            net,
            corpus.input_sequence,
            recurrent_connections=True,
            truncate_large_values=False,
        )

        print(outputs)

        assert tuple(outputs.flatten()) == (1, 3, 7, 15, 7, 3, 1, 0)

    def test_multiplication_loop(self):
        net = network.make_custom_network(
            input_size=1,
            output_size=1,
            num_units=3,
            forward_weights={0: ((2, 1, 1, 1),), 2: ((1, 1, 1, 1),)},
            recurrent_weights={2: ((2, 1, 1, 2),)},
            unit_types={2: network.MULTIPLICATION_UNIT},
        )
        network.visualize(net, "test_multiplication_loop")
        input_seq = np.ones((1, 10, 1))
        target_seq = np.copy(input_seq)
        corpus = corpora.Corpus("test", input_seq, target_seq)
        corpus = corpora.optimize_for_feeding(corpus)

        output = network.feed_sequence(
            net,
            input_seq,
            recurrent_connections=True,
            truncate_large_values=True,
            inputs_per_time_step=corpus.input_values_per_time_step,
            input_mask=corpus.input_mask,
            targets_mask=corpus.targets_mask,
        )
        print(output)
        net = network.calculate_fitness(net, corpus, _TEST_CONFIG)
        print(net)
        assert output.flatten()[-1] == 2 ** -9

    def test_random_net(self):
        for i in range(5):
            net = network.make_random_net(
                input_size=5,
                output_size=5,
                allowed_activations=_TEST_CONFIG.allowed_activations,
                start_smooth=False,
            )

            network.visualize(net, f"test_random_net_{i}")

            for output_unit in net.output_units_range:
                assert (
                    output_unit in net.reverse_forward_connections
                    and len(net.reverse_forward_connections[output_unit]) > 0
                ) or (
                    output_unit in net.reverse_recurrent_connections
                    and len(net.reverse_recurrent_connections[output_unit]) > 0
                )

                assert len(network.get_loop_edges(net)) == 0

    def test_hidden_unit_cost(self):
        fully_connected_weights = {x: ((x + 5, 1, 1, 1),) for x in range(5)}

        config = dataclasses.replace(_TEST_CONFIG)

        net1 = network.make_custom_network(
            input_size=5,
            output_size=5,
            num_units=10,
            forward_weights=fully_connected_weights,
            recurrent_weights={},
        )
        net1_g_explicit = network._get_bitsring_encoding(
            net1, config.allowed_activations
        )
        print("net1 explicit |G|", len(net1_g_explicit))

        net2 = network.make_custom_network(
            input_size=5,
            output_size=5,
            num_units=11,
            forward_weights={
                **fully_connected_weights,
                0: ((10, 1, 1, 1),),
                # 1: ((11, 1, 1, 1),),
                # 2: ((12, 1, 1, 1),),
                # 3: ((13, 1, 1, 1),),
                10: ((5, 1, 1, 1),),
                # 11: ((5, 1, 1, 1),),
                # 12: ((7, 1, 1, 1),),
                # 13: ((8, 1, 1, 1),),
            },
            recurrent_weights={},
        )
        net2_g_explicit = network._get_bitsring_encoding(
            net2, config.allowed_activations
        )
        print("net2 explicit |G|", len(net2_g_explicit))

        net3 = network.make_custom_network(
            input_size=5,
            output_size=5,
            num_units=10,
            forward_weights={**fully_connected_weights, 0: ()},
            recurrent_weights={},
        )
        net3_g_explicit = network._get_bitsring_encoding(
            net3, config.allowed_activations
        )
        print("net3 explicit |G|", len(net3_g_explicit))

        network.visualize(net1, "net1")
        network.visualize(net2, "net2")
        network.visualize(net3, "net3")

        assert len(net2_g_explicit) - len(net1_g_explicit) <= 21
