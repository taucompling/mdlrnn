import network


def make_emmanuel_dyck_2_network(nesting_probab: float):
    opening_bracket_output_bias = nesting_probab / (2 * (1 - nesting_probab))
    return network.make_custom_network(
        input_size=5,
        output_size=5,
        num_units=17,
        forward_weights={
            1: ((3, 1, 2, 1),),
            2: ((12, -1, 1, 1), (13, 1, 1, 1)),
            3: ((10, 1, 1, 1),),
            4: ((2, 1, 1, 1),),
            5: ((9, -1, 1, 1),),
            6: ((8, 1, 1, 1),),
            7: ((9, -1, 1, 1),),
            10: ((11, 1, 1, 1),),
            11: ((15, 1, 1, 1),),
            12: ((11, 1, 1, 1),),
            13: ((15, 1, 1, 1),),
            14: ((13, 1, 1, 1),),
            15: ((16, 1, 1, 1),),
            16: ((5, -1, 1, 1), (7, 1, 1, 1)),
        },
        recurrent_weights={15: ((10, 1, 3, 1), (14, 1, 1, 3))},
        unit_types={11: network.MULTIPLICATION_UNIT, 13: network.MULTIPLICATION_UNIT,},
        biases={5: 1, 6: opening_bracket_output_bias, 7: -1, 9: 1, 12: 1},
        activations={
            5: network.UNSIGNED_STEP,
            7: network.UNSIGNED_STEP,
            14: network.FLOOR,
            16: network.MODULO_3,
        },
    )


def make_emmanuel_dyck_2_network_io_protection(nesting_probab: float):
    opening_bracket_output_bias = nesting_probab / (2 * (1 - nesting_probab))
    return network.make_custom_network(
        input_size=5,
        output_size=5,
        num_units=23,
        forward_weights={
            1: ((18, 1, 2, 1),),
            2: ((17, 1, 1, 1),),
            3: ((18, 1, 1, 1),),
            4: ((17, 1, 1, 1),),
            10: ((11, 1, 1, 1),),
            11: ((15, 1, 1, 1),),
            12: ((11, 1, 1, 1),),
            13: ((15, 1, 1, 1),),
            14: ((13, 1, 1, 1),),
            15: ((16, 1, 1, 1),),
            16: ((19, -1, 1, 1), (20, 1, 1, 1)),
            17: ((12, -1, 1, 1), (13, 1, 1, 1)),
            18: ((10, 1, 1, 1),),
            19: ((21, -1, 1, 1), (20, 1, 1, 1), (5, 1, 1, 1)),
            20: ((7, 1, 1, 1), (21, -1, 1, 1)),
            21: ((9, 1, 1, 1),),
            22: ((6, 1, 1, 1), (8, 1, 1, 1)),
        },
        recurrent_weights={15: ((10, 1, 3, 1), (14, 1, 1, 3))},
        unit_types={11: network.MULTIPLICATION_UNIT, 13: network.MULTIPLICATION_UNIT,},
        biases={12: 1, 19: 1, 20: -1, 21: 1, 22: opening_bracket_output_bias},
        activations={
            14: network.FLOOR,
            16: network.MODULO_3,
            19: network.UNSIGNED_STEP,
            20: network.UNSIGNED_STEP,
        },
    )


def make_emmanuel_triplet_xor_network():
    net = network.make_custom_network(
        input_size=1,
        output_size=1,
        num_units=7,
        forward_weights={
            0: ((2, 1, 1, 1),),
            2: ((5, -1, 1, 1), (6, 1, 1, 1),),
            3: ((6, 1, 1, 1), (5, 1, 1, 1)),
            4: ((3, 1, 1, 1),),
            5: ((1, -1, 1, 2),),
            6: ((1, 1, 1, 2),),
        },
        recurrent_weights={
            0: ((2, -1, 1, 1),),
            3: ((4, -1, 3, 1),),
            4: ((4, 1, 1, 1),),
        },
        biases={1: 0.5, 3: -1, 4: 1, 6: -1},
        activations={
            2: network.SQUARE,
            3: network.RELU,
            5: network.RELU,
            6: network.RELU,
        },
    )
    return net
