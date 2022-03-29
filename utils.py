import argparse
import dataclasses
import hashlib
import itertools
import logging
import os
import pathlib
import pickle
import random
from typing import Any, Dict, Iterable, Optional, Text, Tuple

import numpy as np
from numba import types

import corpora

BASE_SEED = 100

MPI_LOGGING_TAG = 1
MPI_RESULTS_TAG = 2
MPI_MIGRANTS_TAG = 3

FLOAT_DTYPE = np.float64
NUMBA_FLOAT_DTYPE = types.float64


def kwargs_from_param_grid(param_grid: Dict[Text, Iterable[Any]]) -> Iterable[Dict]:
    arg_names = list(param_grid.keys())
    arg_products = list(itertools.product(*param_grid.values()))
    for arg_product in arg_products:
        yield {arg: val for arg, val in zip(arg_names, arg_product)}


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%d-%m-%Y:%H:%M:%S",
        level=logging.INFO,
    )


def load_network_from_zoo(name, subdir=None):
    print(name)
    path = pathlib.Path(f"./network_zoo/")
    if subdir:
        path = path.joinpath(subdir.strip("/"))
    path = path.joinpath(f"{name}.pickle")
    print(path)
    with path.open("rb") as f:
        return pickle.load(f)


def seed(n):
    random.seed(n)
    np.random.seed(n)


def dict_and_corpus_hash(dict_, corpus) -> Text:
    s = corpus.name
    for key in sorted(dict_.keys()):
        s += f"{key} {dict_[key]}"
    # TODO: ugly but works.
    s += corpus.name + " "
    s += str(corpus.input_sequence) + " "
    s += str(corpus.target_sequence)
    hash = hashlib.sha1()
    hash.update(s.encode())
    return hash.hexdigest()


def add_hash_to_simulation_id(simulation_config, corpus):
    config_hash = dict_and_corpus_hash(simulation_config.__dict__, corpus)
    simulation_id = f"{corpus.name}_{config_hash}"
    return dataclasses.replace(simulation_config, simulation_id=simulation_id)


def make_cli_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-s",
        "--simulation",
        dest="simulation_name",
        required=True,
        help=f"Simulation name.",
    )

    arg_parser.add_argument(
        "-n",
        "--total-islands",
        type=int,
        dest="total_islands",
        default=os.cpu_count(),
        help=f"Total number of islands in entire simulation (including other machines). Default: number of local cores ({os.cpu_count()}).",
    )

    arg_parser.add_argument(
        "--first-island",
        type=int,
        default=None,
        dest="first_island",
        help="First island index on this machine. Default: 0.",
    )

    arg_parser.add_argument(
        "--last-island",
        type=int,
        default=None,
        dest="last_island",
        help="Last island index on this machine. Default: number of islands minus 1.",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        dest="base_seed",
        help=f"Base seed value. Default: {BASE_SEED}. For the i-th reproduction (0-based), the seed will be {BASE_SEED} + i.",
    )

    arg_parser.add_argument(
        "--override",
        action="store_true",
        dest="override_existing",
        help="Override an existing simulation that has the same hash.",
    )

    arg_parser.add_argument(
        "--resume",
        dest="resumed_simulation_id",
        default=None,
        help="Resume simulation <simulation id> from latest generations.",
    )

    arg_parser.add_argument(
        "--corpus-args",
        default=None,
        dest="corpus_args",
        help="json to override default corpus arguments.",
    )

    return arg_parser


def calculate_symbolic_accuracy(
    predicted_probabs: np.ndarray,
    target_probabs: np.ndarray,
    input_mask: Optional[np.ndarray],
    sample_weights: Tuple[int],
    plots: bool,
    epsilon: float = 0.0,
) -> Tuple[float, Tuple[int, ...]]:
    zero_target_probabs = target_probabs == 0.0

    zero_predicted_probabs = predicted_probabs <= epsilon

    prediction_matches = np.all(
        np.equal(zero_predicted_probabs, zero_target_probabs), axis=-1
    )

    prediction_matches[~input_mask] = True

    sequence_idxs_with_errors = tuple(np.where(np.any(~prediction_matches, axis=1))[0])
    logging.info(f"Sequence idxs with mismatches: {sequence_idxs_with_errors}")

    incorrect_predictions_per_time_step = np.sum(~prediction_matches, axis=0)

    if plots:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        ax.set_title("Num prediction mismatches by time step")
        ax.bar(
            np.arange(len(incorrect_predictions_per_time_step)),
            incorrect_predictions_per_time_step,
        )
        plt.show()

    prediction_matches_without_masked = prediction_matches[input_mask]

    w = np.array(sample_weights).reshape((-1, 1))
    weights_repeated = np.matmul(w, np.ones((1, predicted_probabs.shape[1])))
    weights_masked = weights_repeated[input_mask]

    prediction_matches_weighted = np.multiply(
        prediction_matches_without_masked, weights_masked
    )

    symbolic_accuracy = np.sum(prediction_matches_weighted) / np.sum(weights_masked)
    return symbolic_accuracy, sequence_idxs_with_errors


def plot_probabs(probabs: np.ndarray, input_classes, class_to_label=None):
    from matplotlib import _color_data as matploit_color_data
    from matplotlib import pyplot as plt

    if probabs.shape[-1] == 1:
        # Binary outputs, output is P(1).
        probabs_ = np.zeros((probabs.shape[0], 2))
        probabs_[:, 0] = (1 - probabs).squeeze()
        probabs_[:, 1] = probabs.squeeze()
        probabs = probabs_

    masked_timesteps = np.where(corpora.is_masked(probabs))[0]
    if len(masked_timesteps):
        first_mask_step = masked_timesteps[0]
        probabs = probabs[:first_mask_step]

    plt.rc("grid", color="w", linestyle="solid")
    if class_to_label is None:
        class_to_label = {i: str(i) for i in range(len(input_classes))}
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, facecolor="white")
    x = np.arange(probabs.shape[0])
    num_classes = probabs.shape[1]
    width = 0.8
    colors = (
        list(matploit_color_data.TABLEAU_COLORS) + list(matploit_color_data.XKCD_COLORS)
    )[:num_classes]
    for c in range(num_classes):
        ax.bar(
            x,
            probabs[:, c],
            label=f"P({class_to_label[c]})" if num_classes > 1 else "P(1)",
            color=colors[c],
            width=width,
            bottom=np.sum(probabs[:, :c], axis=-1),
        )
    ax.set_facecolor("white")
    ax.set_xticks(x)
    ax.set_xticklabels([class_to_label[x] for x in input_classes], fontsize=13)

    ax.set_xlabel("Input characters", fontsize=15)
    ax.set_ylabel("Next character probability", fontsize=15)

    ax.grid(b=True, color="#bcbcbc")
    # plt.title("Next step prediction probabilities", fontsize=22)
    plt.legend(loc="upper left", fontsize=15)

    # fig.savefig("test.png")
    fig.subplots_adjust(bottom=0.1)

    plt.show()
    fig.savefig(
        f"./figures/net_probabs_{random.randint(0,10_000)}.pdf",
        dpi=300,
        facecolor="white",
    )
