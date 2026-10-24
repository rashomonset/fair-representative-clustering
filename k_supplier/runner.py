from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

import pandas as pd
import numpy as np
import argparse
import random
import time

from data_utils import *
from utils import *

from fair_k_supplier_3apx import fair_k_supplier_3apx
from k_supplier_3apx import k_supplier_3apx
from assignmentLP import assignment_LP
from profileILP import additive_apx_ILP


def generate_instances(dataset, name):

    prng = np.random.default_rng(0)

    GRID = ParameterGrid({
        "dataset":[name],
        "n":[dataset["n"]],
        "d":[dataset["d"]],
        "t":[dataset["t"]],
        "k":[dataset["t"]*2*i for i in range(1,6,1)],
        "n_c":np.array([dataset["n"]]),
        "n_f":np.array([int(val*dataset["n"]) for val in np.linspace(0,1,6)][1:]),
        "C_idx":[np.array(range(dataset["n"]))],
        "split":["proportional","equitable"],
        "G":[dataset["G"]],
        "U":[dataset["U"]],
        "seed":prng.integers(0, 10**9, size=10).tolist(),
    })

    for instance in GRID:

        run_prng = np.random.default_rng(instance["seed"])

        instance["rvec"] = create_rvec(
            G = instance["G"],
            t = instance["t"],
            k = instance["k"],
            split = instance["split"],
            prng = run_prng
        )

        instance["F_idx"], instance["GG_idx"], instance["GG_sizes"] = create_facilities(
            G = instance["G"],
            t = instance["t"],
            n_f = instance["n_f"],
            rvec = instance["rvec"],
            prng = run_prng
        )

        yield instance


def run_baseline(instance, baseline):

    start = time.time()
    _, center_idx, cost, apx_ratio = baseline(instance, instance["seed"])
    runtime = time.time() - start

    if baseline == k_supplier_3apx:
        center_idx = instance["F_idx"][np.asarray(center_idx, dtype=int)]
    
    metrics = compute_metrics_baseline(I=instance,center_idx=center_idx)
    to_store = ["dataset","split","n", "d", "t", "k", "n_c", "n_f", "GG_sizes", "rvec", "seed"]

    output = {key:instance[key] for key in to_store}
    output["cost"] = cost
    output["post_cost"] = cost
    output["runtime"] = runtime
    output["algorithm"] = baseline.__name__
    output["apx_ratio"] = apx_ratio
    output["l"] = 1
    output.update(metrics)
    return [output]


def run_approximations(instance, objective):

    rows = []
    to_store = [
        "dataset", "split", "n", "d", "t", "k", "n_c",
        "n_f", "GG_sizes", "rvec", "seed"
    ]
    output = {key: instance[key] for key in to_store}

    start = time.time()
    prepared = create_utility_matrix(
        instance, k_supplier, objective
    )
    prepared = (*prepared, time.time() - start)

    for metrics in assignment_LP(
        instance, objective, prepared
    ):
        row = output.copy()
        row["algorithm"] = "assignmentLP"
        row.update(metrics)
        rows.append(row)

    for delta in [.05, .01]:
        instance["delta"] = delta
        instance["epsilon"] = delta / instance["k"]

        row = output.copy()
        row["algorithm"] = f"profileILP"
        row["delta"] = delta
        row["epsilon"] = instance["epsilon"]
        row["l"] = 1
        row.update(
            additive_apx_ILP(
                I=instance,
                objective=objective,
                prepared=prepared
            )[0]
        )
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--objective", choices={"local","global"}, required=True)
    parser.add_argument("--minprop", default=0, type=float)
    args = parser.parse_args()

    name = args.dataset
    load_dataset = DATASETS[name]
    min_prop = args.minprop
    objective = args.objective
    MAX_WORKERS = 16
    BASELINES = [
        k_supplier_3apx,
        fair_k_supplier_3apx
    ]

    dataset = load_dataset(min_prop)
    ROWS = []

    """Parallel Processing"""
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for instance in generate_instances(dataset,name):
            futures += [
                ex.submit(run_baseline, instance.copy(), baseline)
                for baseline in BASELINES
            ]
            futures.append(
                ex.submit(run_approximations, instance.copy(), objective)
            )

        for future in tqdm(as_completed(futures), total=len(futures)):
            ROWS.extend(future.result())

    pd.DataFrame(ROWS).to_csv(
        f"results/{name}_{objective}_minprop_{min_prop:g}_results.csv",
        index=False,
    )
