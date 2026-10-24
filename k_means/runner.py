from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

import pandas as pd
import numpy as np
import argparse
import os
import time

from data_utils import *
from utils import *

from assignmentLP import assignment_LP
from profileILP import additive_apx_ILP


def generate_instances(dataset, name):
    prng = np.random.default_rng(42)

    GRID = ParameterGrid({
        "dataset": [name],
        "n": [dataset["n"]],
        "d": [dataset["d"]],
        "t": [dataset["t"]],
        "k": [dataset["t"]*2*i for i in range(1, 6)],
        "n_c": [dataset["n"]],
        "C_idx": [np.arange(dataset["n"])],
        "G": [dataset["G"]],
        "U": [dataset["U"]],
        "seed": prng.integers(0, 10**9, size=10).tolist(),
    })

    yield from GRID


def run_baseline(instance, baseline):
    start = time.time()
    centers, center_idx, assignments, cost = baseline(
        instance, instance["seed"]
    )
    runtime = time.time() - start

    counts = prepare_group_cluster_counts(instance, assignments)
    metrics = compute_metrics(counts, instance["G"][center_idx])

    to_store = ["dataset", "n", "d", "t", "k", "n_c", "seed"]
    output = {key: instance[key] for key in to_store}
    output["algorithm"] = baseline.__name__
    output["status"] = "optimal"
    output["l"] = 1
    output["centers"] = centers.tolist()
    output["pre_cost"] = cost
    output["post_cost"] = cost
    output["cost"] = cost
    output["runtime"] = runtime
    output.update(metrics)
    return [output]


def run_approximations(instance, objective):
    rows = []
    to_store = ["dataset", "n", "d", "t", "k", "n_c", "seed"]
    output = {key: instance[key] for key in to_store}

    start = time.time()
    centers, _, assignments, pre_cost = kmeans_plusplus(
        instance, instance["seed"], centroids=True
    )
    base_runtime = time.time() - start

    for alpha in [.2, .4, .6, .8, 1]:
        instance["alpha"] = alpha
        start = time.time()
        U, radii_sq = create_utility_matrix(
            instance, centers, assignments, alpha, objective
        )
        prep_runtime = base_runtime + time.time() - start
        prepared = (U, centers, assignments, pre_cost, radii_sq, prep_runtime)

        for metrics in assignment_LP(instance, objective, prepared):
            row = output.copy()
            row["algorithm"] = "assignmentLP"
            row["alpha"] = alpha
            row.update(metrics)
            rows.append(row)

        for delta in [.1, .05, .01]:
            instance["delta"] = delta
            instance["epsilon"] = delta / instance["k"]
            row = output.copy()
            row["algorithm"] = "profileILP"
            row["alpha"] = alpha
            row["delta"] = delta
            row["epsilon"] = instance["epsilon"]
            row["l"] = 1
            row.update(additive_apx_ILP(instance, objective, prepared)[0])
            rows.append(row)

    return rows


if __name__ == "__main__":
    DATASETS = {
        "heart": load_heart,
        "student": load_student,
        "census_adult": load_census_adult,
        "bank": load_bank,
        "credit_card": load_credit_card,
        "movie_lens": load_movielens_1m
    }

    MAX_WORKERS = 16

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--objective", choices={"local", "global"}, required=True)
    parser.add_argument("--minprop", default=0, type=float)
    args = parser.parse_args()

    name = args.dataset
    load_dataset = DATASETS[name]
    min_prop = args.minprop
    objective = args.objective
    BASELINES = [kmeans_plusplus]

    dataset = load_dataset(min_prop)
    ROWS = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for instance in generate_instances(dataset, name):
            futures += [
                ex.submit(run_baseline, instance.copy(), baseline)
                for baseline in BASELINES
            ]
            futures += [
                ex.submit(
                    run_approximations,
                    instance.copy(),
                    objective
                ),
            ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            ROWS.extend(future.result())

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(ROWS).to_csv(
        f"results/{name}_{objective}_minprop_{min_prop:g}_results.csv",
        index=False,
    )
