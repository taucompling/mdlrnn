import dataclasses
import json
import logging
import multiprocessing
import pickle
import queue
from datetime import datetime
from typing import Dict, Optional, Text, Tuple

import numpy as np
from mpi4py import MPI

import configuration
import corpora
import genetic_algorithm
import network
import utils


_MPI_COMMUNICATOR = MPI.COMM_WORLD
_MPI_RANK = _MPI_COMMUNICATOR.Get_rank()

_MPI_LOGGING_BUFFER_SIZE = 5_000_000


def _island_process(
    island_num: int,
    config: configuration.SimulationConfig,
    corpus: corpora.Corpus,
    result_queue: multiprocessing.Queue,
    logging_queue: multiprocessing.Queue,
    process_locks: Tuple[multiprocessing.Lock, ...],
):
    try:
        result_network = genetic_algorithm.run(
            island_num=island_num,
            corpus=corpus,
            config=config,
            logging_queue=logging_queue,
            process_locks=process_locks,
        )
        result_queue.put((island_num, result_network))
    except:
        logging.exception(f"Exception in island {island_num}:")


class _DummyProcess:
    # Used for profiling without multiprocessing.
    def __init__(self, target, kwargs):
        self._target = target
        self._kwargs = kwargs
        self.__setattr__ = lambda x: x

    def start(self):
        self._target(**self._kwargs)


def _log_net_stats(
    stats: Dict,
    config: configuration.SimulationConfig,
    train_corpus: corpora.Corpus,
    test_corpus: Optional[corpora.Corpus],
    best_net: network.Network,
    status: Text,
):
    stats.update(
        {
            "simulation": config.simulation_id,
            "simulation id": config.simulation_id,
            "configuration": json.dumps(config.__dict__),
            "params": config.comment,
            "status": status,
            "best island": stats["island"],
            "last update": datetime.now().isoformat(),
        }
    )

    log_text = (
        f"Current best net, island {stats['island']}, generation {stats['generation']}: "
        f"MDL = {stats['mdl']:,.2f}, |G| = {stats['|g|']}, |D:G| = {stats['|d:g|']:,.2f}, units: {stats['units']}, connections: {stats['connections']}."
    )

    train_num_chars = corpora.get_num_chars_in_corpus(train_corpus)
    average_train_dg_per_char = best_net.fitness.data_encoding_length / train_num_chars
    stats.update(
        {
            "average train d:g per character": f"{average_train_dg_per_char:.2f}",
            "training set num. chars": f"{train_num_chars}",
        }
    )

    if train_corpus.deterministic_steps_mask is not None:
        train_deterministic_accuracy = network.calculate_deterministic_accuracy(
            best_net, train_corpus, config
        )
        stats.update({"deterministic accuracy": train_deterministic_accuracy})
        log_text += (
            f" Training set deterministic accuracy: {train_deterministic_accuracy:,.2f}"
        )

    if test_corpus is not None:
        test_net = network.invalidate_fitness(best_net)
        test_net = network.calculate_fitness(
            test_net, corpus=test_corpus, config=config
        )

        test_num_chars = corpora.get_num_chars_in_corpus(test_corpus)
        average_test_dg_per_char = (
            test_net.fitness.data_encoding_length / test_num_chars
        )

        stats.update(
            {
                "test set d:g": f"{test_net.fitness.data_encoding_length:.2f}",
                "test set accuracy": f"{test_net.fitness.accuracy}",
                "average test d:g per character": f"{average_test_dg_per_char:.2f}",
            }
        )
        log_text += f" Test set |D:G|: {test_net.fitness.data_encoding_length:,.2f}."

        if test_corpus.deterministic_steps_mask is not None:
            test_deterministic_accuracy = network.calculate_deterministic_accuracy(
                test_net, test_corpus, config
            )
            stats.update(
                {"test set deterministic accuracy": test_deterministic_accuracy}
            )
            log_text += (
                f" Test set deterministic accuracy: {test_deterministic_accuracy:,.2f}"
            )

    logging.info(log_text)

    current_best_net_filename = f"{config.simulation_id}__current_best"
    network.save(best_net, current_best_net_filename)
    network.visualize(
        best_net,
        current_best_net_filename,
        class_to_label=test_corpus.vocabulary if test_corpus else None,
    )


def _mpi_logging_worker(
    config: configuration.SimulationConfig,
    train_corpus: corpora.Corpus,
    test_corpus: Optional[corpora.Corpus],
):
    best_mdl = float("inf")
    buffer = np.empty(7)
    logging.info(f"Started MPI logging worker, rank {_MPI_RANK}")
    while True:
        _MPI_COMMUNICATOR.Recv(buffer, tag=utils.MPI_LOGGING_TAG)
        (
            island_num,
            generation,
            mdl,
            grammar_encoding_length,
            data_encoding_length,
            num_units,
            connections,
        ) = buffer
        island_num = int(island_num)
        if mdl < best_mdl:
            best_mdl = mdl

            with open(
                f"./networks/{config.simulation_id}__island_{island_num}__best_network.pickle",
                "rb",
            ) as f:
                best_net = pickle.load(f)

            best_island_stats = {
                "island": island_num,
                "generation": int(generation),
                "mdl": mdl,
                "|g|": grammar_encoding_length,
                "|d:g|": data_encoding_length,
                "units": int(num_units),
                "connections": int(connections),
                "accuracy": best_net.fitness.accuracy,
            }

            _log_net_stats(
                stats=best_island_stats,
                config=config,
                train_corpus=train_corpus,
                test_corpus=test_corpus,
                best_net=best_net,
                status="Running",
            )


def _queue_logging_worker(
    logging_queue: multiprocessing.Queue,
    config: configuration.SimulationConfig,
    train_corpus: corpora.Corpus,
    test_corpus: Optional[corpora.Corpus],
):
    best_net_fitness = float("inf")

    while True:
        island_num_to_data = {}
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < 60:
            try:
                data = logging_queue.get(timeout=5)
                island_num = data["stats"]["island"]
                island_num_to_data[island_num] = data
            except queue.Empty:
                pass

        best_island_data = None
        for data in island_num_to_data.values():
            stats = data["stats"]
            if stats["mdl"] <= best_net_fitness:
                best_net_fitness = stats["mdl"]
                best_island_data = data

        if best_island_data is None:
            continue

        _log_net_stats(
            stats=best_island_data["stats"],
            config=config,
            train_corpus=train_corpus,
            test_corpus=test_corpus,
            best_net=best_island_data["best_net"],
            status="Running",
        )


def _create_csv_and_spreadsheet_entry(
    config: configuration.SimulationConfig, train_corpus, test_corpus
):
    data = {
        "simulation": config.simulation_id,
        "simulation id": config.simulation_id,
        "params": config.comment,
        "configuration": json.dumps(config.__dict__),
        "status": "Running",
        "started": datetime.now().isoformat(),
        "seed": config.seed,
    }

    train_num_chars = corpora.get_num_chars_in_corpus(train_corpus)
    data.update(
        {"training set num. chars": f"{train_num_chars}",}
    )
    if train_corpus.optimal_d_given_g is not None:
        data.update(
            {
                "training set optimal d:g": f"{train_corpus.optimal_d_given_g:.2f}",
                "optimal average train d:g per character": f"{train_corpus.optimal_d_given_g / train_num_chars:.2f}",
            }
        )

    if test_corpus is not None:
        test_num_chars = corpora.get_num_chars_in_corpus(test_corpus)
        data.update(
            {
                "test set params": f"Input shape: {test_corpus.input_sequence.shape}. Output shape: {test_corpus.target_sequence.shape}",
                "test set num. chars": f"{test_num_chars}",
            }
        )
        if test_corpus.optimal_d_given_g is not None:
            data.update(
                {
                    "test set optimal d:g": f"{test_corpus.optimal_d_given_g:.2f}",
                    "optimal average test d:g per character": f"{test_corpus.optimal_d_given_g / test_num_chars:.2f}",
                }
            )


def _init_islands(
    first_island: int,
    last_island: int,
    config: configuration.SimulationConfig,
    train_corpus,
    result_queue: multiprocessing.Queue,
    logging_queue: multiprocessing.Queue,
    process_locks: Tuple[multiprocessing.Lock, ...],
) -> Tuple[multiprocessing.Process, ...]:
    processes = []
    for i in range(first_island, last_island + 1):
        if config.parallelize:
            process_class = multiprocessing.Process
        else:
            process_class = _DummyProcess

        p = process_class(
            target=_island_process,
            kwargs={
                "island_num": i,
                "config": config,
                "corpus": train_corpus,
                "result_queue": result_queue,
                "logging_queue": logging_queue,
                "process_locks": process_locks,
            },
        )
        p.daemon = True
        processes.append(p)

    return tuple(processes)


def _get_island_results_from_queue_or_mpi(result_queue: multiprocessing.Queue):
    if result_queue is not None:
        return result_queue.get()
    return _MPI_COMMUNICATOR.recv(tag=utils.MPI_RESULTS_TAG)


def _collect_results(
    num_islands_to_collect: int,
    result_queue: multiprocessing.Queue,
    config: configuration.SimulationConfig,
    train_corpus: corpora.Corpus,
    test_corpus: Optional[corpora.Corpus],
):
    result_network_per_island = {}
    while len(result_network_per_island) < num_islands_to_collect:
        island_num, result_net = _get_island_results_from_queue_or_mpi(
            result_queue=result_queue
        )
        result_network_per_island[island_num] = result_net
        logging.info(f"Island {island_num} best network:\n{str(result_net)}")
        logging.info(
            f"{len(result_network_per_island)}/{num_islands_to_collect} (global {config.num_islands}) islands done."
        )

    best_island, best_net = min(
        result_network_per_island.items(), key=lambda x: x[1].fitness.mdl
    )
    network_filename = f"{config.simulation_id}__best_network"
    network.visualize(
        best_net, network_filename, class_to_label=train_corpus.vocabulary
    )
    network.save(best_net, network_filename)

    logging.info(f"Best network of all islands:\n{str(best_net)}")

    csv_data = {
        "simulation id": config.simulation_id,
        "island": best_island,
        "mdl": best_net.fitness.mdl,
        "|g|": best_net.fitness.grammar_encoding_length,
        "|d:g|": best_net.fitness.data_encoding_length,
        "units": network.get_num_units(best_net),
        "connections": network.get_total_connections(best_net, include_biases=True),
        "accuracy": best_net.fitness.accuracy,
        "generation": config.num_generations,
        "finished": datetime.now().isoformat(),
    }
    _log_net_stats(
        stats=csv_data,
        config=config,
        train_corpus=train_corpus,
        test_corpus=test_corpus,
        best_net=best_net,
        status="Done",
    )


def _run_mpi_island(island_num, corpus, config):
    result_network = genetic_algorithm.run(
        island_num=island_num,
        corpus=corpus,
        config=config,
        logging_queue=None,
        process_locks=None,
    )
    _MPI_COMMUNICATOR.send(
        (island_num, result_network),
        dest=config.num_islands + 1,
        tag=utils.MPI_RESULTS_TAG,
    )
    while True:
        # TODO: need to keep running this so buffer won't fill. need to find a better solution.
        genetic_algorithm.get_migrants_through_mpi()


def run(
    corpus: corpora.Corpus,
    config: configuration.SimulationConfig,
    first_island: int,
    last_island: int,
):
    train_corpus = corpora.optimize_for_feeding(
        dataclasses.replace(corpus, test_corpus=None)
    )
    test_corpus = (
        corpora.optimize_for_feeding(corpus.test_corpus) if corpus.test_corpus else None
    )

    genetic_algorithm.verify_existing_simulation_override(
        simulation_id=config.simulation_id,
    )

    logging.info(f"Starting simulation {config.simulation_id}\n")
    logging.info(f"Config: {config}\n")
    logging.info(
        f"Running islands {first_island}-{last_island} ({last_island-first_island+1}/{config.num_islands})"
    )

    if config.migration_channel == "file":
        genetic_algorithm.get_migration_path(config.simulation_id).mkdir(
            parents=True, exist_ok=True
        )

        if first_island == 0:
            _create_csv_and_spreadsheet_entry(config, train_corpus, test_corpus)

        result_queue = multiprocessing.Queue()
        logging_queue = multiprocessing.Queue()
        process_locks = tuple(multiprocessing.Lock() for _ in range(config.num_islands))

        island_processes = _init_islands(
            first_island=first_island,
            last_island=last_island,
            config=config,
            train_corpus=train_corpus,
            result_queue=result_queue,
            logging_queue=logging_queue,
            process_locks=process_locks,
        )

        for p in island_processes:
            p.start()

        logging_process = multiprocessing.Process(
            target=_queue_logging_worker,
            args=(logging_queue, config, train_corpus, test_corpus),
        )
        logging_process.daemon = True
        logging_process.start()

        _collect_results(
            num_islands_to_collect=len(island_processes),
            result_queue=result_queue,
            config=config,
            train_corpus=train_corpus,
            test_corpus=test_corpus,
        )

    elif config.migration_channel == "mpi":
        if _MPI_RANK < config.num_islands:
            # Ranks [0, num_islands - 1] = islands
            _run_mpi_island(
                island_num=_MPI_RANK, corpus=train_corpus, config=config,
            )
        elif _MPI_RANK == config.num_islands:
            # Rank num_islands = logger
            _create_csv_and_spreadsheet_entry(config, train_corpus, test_corpus)
            _mpi_logging_worker(config, train_corpus, test_corpus)
        elif _MPI_RANK == config.num_islands + 1:
            # Rank num_island + 1 = results collector
            logging.info(f"Starting results collection, rank {_MPI_RANK}")
            _collect_results(
                num_islands_to_collect=config.num_islands,
                result_queue=None,
                config=config,
                train_corpus=train_corpus,
                test_corpus=test_corpus,
            )
