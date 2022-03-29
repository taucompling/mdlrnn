import dataclasses
import json

import configuration
import genetic_algorithm
import island
import simulations
import utils

utils.setup_logging()


_NUM_REPRODUCTIONS = 1


def _make_corpus(factory, args, corpus_seed):
    utils.seed(corpus_seed)
    return factory(**args)


def run():
    arg_parser = utils.make_cli_arguments()
    arguments = arg_parser.parse_args()

    simulation = simulations.SIMULATIONS[arguments.simulation_name]

    for i in range(_NUM_REPRODUCTIONS):
        simulation_seed = arguments.base_seed + i

        base_config = configuration.SimulationConfig(
            seed=simulation_seed,
            simulation_id=arguments.simulation_name,
            num_islands=arguments.total_islands,
            **{**simulations.DEFAULT_CONFIG, **simulation.get("config", {})},
        )

        corpus_args = simulation["corpus"]["args"]
        if arguments.corpus_args is not None:
            corpus_args.update(json.loads(arguments.corpus_args))

        corpus = _make_corpus(
            factory=simulation["corpus"]["factory"],
            args=corpus_args,
            corpus_seed=base_config.corpus_seed,
        )

        simulation_config = dataclasses.replace(
            base_config,
            comment=f"Corpus params: {json.dumps(simulation['corpus']['args'])}, input shape: {corpus.input_sequence.shape}. Output shape: {corpus.target_sequence.shape}",
            simulation_id=corpus.name,
            resumed_from_simulation_id=arguments.resumed_simulation_id,
        )
        simulation_config = utils.add_hash_to_simulation_id(simulation_config, corpus)

        if arguments.override_existing:
            genetic_algorithm.remove_simulation_directory(
                simulation_id=simulation_config.simulation_id,
            )

        utils.seed(simulation_seed)

        island.run(
            corpus=corpus,
            config=simulation_config,
            first_island=arguments.first_island
            if arguments.first_island is not None
            else 0,
            last_island=(
                arguments.last_island
                if arguments.last_island is not None
                else simulation_config.num_islands - 1
            ),
        )


if __name__ == "__main__":
    run()
