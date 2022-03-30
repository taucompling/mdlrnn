import json
import operator
import re
from typing import Dict

import numpy as np


def _params_key(params_dict):
    return str(tuple(sorted(params_dict.items())))


def _parse_mdlnn_params(params_string) -> Dict:
    return json.loads(re.findall(r"\{.+\}", params_string)[0])


def _parse_rnn_params(params_string) -> Dict:
    return json.loads(params_string)


def _dict_subset(parent_dict, child_dict):
    t = all(child_key in parent_dict for child_key in child_dict.keys())
    return t


def _float_or_none(val):
    return float(val) if val is not None else -1


def _get_value_from_sim_stats(sims_list, key):
    return np.array([_float_or_none(x[key]) for x in sims_list])


def _percent_format(x):
    return f"{x*100:.1f}%"


def _ln_to_log2(ln_loss):
    return ln_loss / np.log(2)


def _compare_mdl_and_rnn_winners_by_stat(mdl_stats, rnn_stats, key):
    mdl_best_val = _float_or_none(mdl_stats[key])
    rnn_best_val = _float_or_none(rnn_stats[key])

    if "accuracy" in key.lower():
        # Accuracy, higher is better.
        op = operator.gt
        mdl_best_val_str = _percent_format(mdl_best_val)
        rnn_best_val_str = _percent_format(rnn_best_val)
        optimal_str = ""
    else:
        # Cross-entropy, lower is better.
        op = operator.lt
        rnn_best_val = _ln_to_log2(rnn_best_val)
        mdl_best_val_str = f"{mdl_best_val*100:.1f}"
        rnn_best_val_str = f"{rnn_best_val*100:.1f}"
        optimal_str = f"& ({_float_or_none(rnn_stats['Optimal average test CE per character'])*100:.1f})"

    print(f"\t{key}")
    print(f"\tMDLRNN & RNN")
    print(f"\t{mdl_best_val_str} & {rnn_best_val_str} {optimal_str}")

    if mdl_best_val == rnn_best_val:
        print("\t\tTie")
    elif op(mdl_best_val, rnn_best_val):
        print("\t\tMDL wins")
    else:
        print(f"\t\tRNN wins")


def compare_mdlnn_rnn_stats(select_by: str):
    print(f"Comparing based on {select_by}\n")

    with open("./tacl_stats.json", "r") as f:
        data = json.load(f)

    mdlnn_rnn_comparisons = (
        "Average test CE per character",
        "Test accuracy",
        "Test deterministic accuracy",
    )

    test_best_rnn_ids = []
    test_best_mdlnn_ids = []

    for task_dict in data.values():
        corpus_name = task_dict["corpus_name"]
        try:
            mdl_sims = task_dict["mdl"]
        except KeyError:
            print(f"No MDL model for {corpus_name}\n\n")
            continue
        rnn_sims = task_dict["rnn"]

        print(f"* {corpus_name}")

        train_winning_mdl_sim_idx = np.argmin(
            _get_value_from_sim_stats(mdl_sims, "MDL")
        )
        train_winning_mdl_sim_stats = mdl_sims[train_winning_mdl_sim_idx]

        train_mdl_g_scores = _get_value_from_sim_stats(mdl_sims, "|G|")
        test_mdl_scores = train_mdl_g_scores + _get_value_from_sim_stats(
            mdl_sims, "Test set D:G"
        )

        test_winning_mdl_sim_idx = np.argmin(test_mdl_scores)
        test_winning_mdl_sim_stats = mdl_sims[test_winning_mdl_sim_idx]

        train_winning_rnn_sim_idx = np.argmin(
            _get_value_from_sim_stats(rnn_sims, "Average train CE per character")
        )
        train_winning_rnn_sim_stats = rnn_sims[train_winning_rnn_sim_idx]

        print(f"\tTrain-best MDL:\t{train_winning_mdl_sim_stats['Simulation id']}")
        print(f"\tTrain-best RNN:\t{train_winning_rnn_sim_stats['Simulation id']}\n")

        test_winning_rnn_sim_idx = np.argmin(
            # Take winner based on actual CE performance, without regularization ('loss' stat includes regularization term).
            _get_value_from_sim_stats(rnn_sims, "Average test CE per character")
        )
        test_winning_rnn_sim_stats = rnn_sims[test_winning_rnn_sim_idx]
        test_rnn_args = json.loads(test_winning_rnn_sim_stats["Params"])

        print(
            f"\tTest-best MDL:\t{test_winning_mdl_sim_stats['Simulation id']}.\n{test_winning_mdl_sim_stats['Units']} units, {test_winning_mdl_sim_stats['Connections']} connections"
        )
        print(
            f"\tTest-best RNN:\t{test_winning_rnn_sim_stats['Simulation id']}\n{test_rnn_args['network_type']}, {test_rnn_args['num_hidden_units']} units, {test_rnn_args.get('regularization', 'no')} regularization, {test_rnn_args.get('regularization_lambda', '-')} lambda.\n"
        )

        test_best_mdlnn_ids.append(test_winning_mdl_sim_stats["Simulation id"])
        test_best_rnn_ids.append(test_winning_rnn_sim_stats["Simulation id"])

        if select_by == "train":
            for comparison_key in mdlnn_rnn_comparisons:
                _compare_mdl_and_rnn_winners_by_stat(
                    mdl_stats=train_winning_mdl_sim_stats,
                    rnn_stats=train_winning_rnn_sim_stats,
                    key=comparison_key,
                )
        else:
            for comparison_key in mdlnn_rnn_comparisons:
                _compare_mdl_and_rnn_winners_by_stat(
                    mdl_stats=test_winning_mdl_sim_stats,
                    rnn_stats=test_winning_rnn_sim_stats,
                    key=comparison_key,
                )


if __name__ == "__main__":
    compare_mdlnn_rnn_stats(select_by="train")
    compare_mdlnn_rnn_stats(select_by="test")
