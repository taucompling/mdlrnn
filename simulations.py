import corpora
import network

DEFAULT_ACTIVATIONS = (
    network.SIGMOID,
    network.LINEAR,
    network.RELU,
    network.SQUARE,
    network.UNSIGNED_STEP,
    network.FLOOR,
)

EXTENDED_ACTIVATIONS = DEFAULT_ACTIVATIONS + (network.MODULO_2, network.MODULO_3,)

DEFAULT_UNIT_TYPES = (network.SUMMATION_UNIT,)

DEFAULT_CONFIG = {
    "migration_ratio": 0.01,
    "migration_interval_seconds": 1800,
    "migration_interval_generations": 1000,
    "num_generations": 25_000,
    "population_size": 500,
    "elite_ratio": 0.001,
    "allowed_activations": DEFAULT_ACTIVATIONS,
    "allowed_unit_types": DEFAULT_UNIT_TYPES,
    "start_smooth": False,
    "compress_grammar_encoding": False,
    "tournament_size": 2,
    "mutation_probab": 1.0,
    "mini_batch_size": None,
    "grammar_multiplier": 1,
    "data_given_grammar_multiplier": 1,
    "max_network_units": 1024,
    "softmax_outputs": False,
    "truncate_large_values": True,
    "bias_connections": True,
    "recurrent_connections": True,
    "corpus_seed": 100,
    "parallelize": True,
    "migration_channel": "file",
    "generation_dump_interval": 250,
}

SIMULATIONS = {
    "identity": {
        "corpus": {
            "factory": corpora.make_identity_binary,
            "args": {"sequence_length": 100, "batch_size": 10},
        }
    },
    "repeat_last_char": {
        "corpus": {
            "factory": corpora.make_prev_char_repetition_binary,
            "args": {"sequence_length": 100, "batch_size": 10, "repetition_offset": 1,},
        }
    },
    "binary_addition": {
        "corpus": {
            "factory": corpora.make_binary_addition,
            "args": {"min_n": 0, "max_n": 20},
        },
    },
    "dyck_1": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "n": 1,
                "batch_size": 100,
                "nesting_probab": 0.3,
                "max_sequence_length": 200,
            },
        },
    },
    "dyck_2": {
        "corpus": {
            "factory": corpora.make_dyck_n,
            "args": {
                "batch_size": 20_000,
                "nesting_probab": 0.3,
                "n": 2,
                "max_sequence_length": 200,
            },
        },
        "config": {
            "allowed_activations": EXTENDED_ACTIVATIONS,
            "allowed_unit_types": (network.SUMMATION_UNIT, network.MULTIPLICATION_UNIT),
        },
    },
    "an_bn": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn,
            "args": {"batch_size": 100, "prior": 0.3, "multipliers": (1, 1, 0, 0)},
        }
    },
    "an_bn_cn": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn,
            "args": {"batch_size": 100, "prior": 0.3, "multipliers": (1, 1, 1, 0)},
        }
    },
    "an_bn_cn_dn": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn,
            "args": {"batch_size": 100, "prior": 0.3, "multipliers": (1, 1, 1, 1)},
        }
    },
    "an_b2n": {
        "corpus": {
            "factory": corpora.make_ain_bjn_ckn_dtn,
            "args": {"batch_size": 100, "prior": 0.3, "multipliers": (1, 2, 0, 0)},
        }
    },
    "an_bn_square": {
        "corpus": {
            "factory": corpora.make_an_bn_square,
            "args": {"batch_size": 1000, "prior": 0.5},
        }
    },
    "palindrome_fixed_length": {
        "corpus": {
            "factory": corpora.make_binary_palindrome_fixed_length,
            "args": {
                "batch_size": 1000,
                "sequence_length": 50,
                "train_set_ratio": 0.7,
            },
        }
    },
    "an_bm_cn_plus_m": {
        "corpus": {
            "factory": corpora.make_an_bm_cn_plus_m,
            "args": {"batch_size": 100, "prior": 0.3},
        }
    },
    "center_embedding": {
        "corpus": {
            "factory": corpora.make_center_embedding,
            "args": {
                "batch_size": 20_000,
                "embedding_depth_probab": 0.3,
                "dependency_distance_probab": 0.0,
            },
        },
        "config": {
            "allowed_activations": DEFAULT_ACTIVATIONS + (network.MODULO_2,),
            "allowed_unit_types": (network.SUMMATION_UNIT, network.MULTIPLICATION_UNIT),
        },
    },
    "0_1_pattern_binary": {
        "corpus": {
            "factory": corpora.make_0_1_pattern_binary,
            "args": {"sequence_length": 20, "batch_size": 1},
        }
    },
    "0_1_pattern_one_hot_no_eos": {
        "corpus": {
            "factory": corpora.make_0_1_pattern_one_hot,
            "args": {
                "add_end_of_sequence": False,
                "sequence_length": 50,
                "batch_size": 1,
            },
        }
    },
    "0_1_pattern_one_hot_with_eos": {
        "corpus": {
            "factory": corpora.make_0_1_pattern_one_hot,
            "args": {
                "add_end_of_sequence": True,
                "sequence_length": 50,
                "batch_size": 1,
            },
        }
    },
}
