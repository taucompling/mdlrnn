import dataclasses
from typing import Optional, Text, Tuple


@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    simulation_id: Text
    num_islands: int
    migration_interval_seconds: int
    migration_interval_generations: int
    migration_ratio: float

    num_generations: int
    population_size: int
    elite_ratio: float
    mutation_probab: float
    allowed_activations: Tuple[int, ...]
    allowed_unit_types: Tuple[int, ...]
    start_smooth: bool

    max_network_units: int
    tournament_size: int

    grammar_multiplier: int
    data_given_grammar_multiplier: int

    compress_grammar_encoding: bool
    softmax_outputs: bool
    truncate_large_values: bool
    bias_connections: bool
    recurrent_connections: bool
    generation_dump_interval: int

    seed: int
    corpus_seed: int

    mini_batch_size: Optional[int] = None
    resumed_from_simulation_id: Optional[Text] = None
    comment: Optional[Text] = None
    parallelize: bool = True
    migration_channel: Text = "file"  # {'file', 'mpi'}
