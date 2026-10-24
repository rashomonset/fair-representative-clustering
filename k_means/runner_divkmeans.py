"""
Diversity-aware k-means baseline: the LP + LS1 heuristic of Thejaswi et al. (KDD 2022),
Code: github.com/suhastheju/diversity-aware-clustering, expected in the subfolder divkmeans/
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

import pandas as pd
import numpy as np 
import argparse
import os
import time

from scipy.spatial.distance import cdist
from data_utils import *
from utils import *

from pathlib import Path
import random
import sys

DIVKMEANS_DIR = Path(__file__).resolve().parent / "divkmeans"
sys.path.extend(
    map(str, [DIVKMEANS_DIR, DIVKMEANS_DIR / "bicriteria"])
)

from feasibility import feasibility
from divkmeans.utils import kmedkpm

# L2 Distance 
kmedkpm.cdist = lambda XA, XB: cdist(XA, XB, "sqeuclidean")

OFFSET = -2

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


def lp_ls1_kmeans(instance, rvec, logfile):
    U = instance["U"]
    G = np.asarray(instance["G"], dtype=int)
    t, k, seed = instance["t"], instance["k"], instance["seed"]
    n, d = U.shape

    # colour matrix (t x n)
    color_mat = np.zeros((t, n), dtype=int)
    color_mat[t - 1 - G, np.arange(n)] = 1

    random.seed(seed)
    np.random.seed(seed)
    stats = feasibility.calculate(
        k, np.array(rvec), color_mat, "linear-program", False, logfile, True
    )
    set_mappings = stats["subset_map"]
    set_to_indices = {
        idx: key for idx, key in enumerate(sorted(set_mappings.keys()))
    }
    solution = stats["solution"]
    patterns = [] if len(solution) == 0 else np.unique(np.stack(solution, axis=0), axis=0)

    # LS1 on every pattern; keep the cheapest solution
    X = U + OFFSET
    best = None
    for pattern in patterns:
        E, E_idx = {}, {}
        i = 0
        for idx, count in enumerate(pattern):
            members = set_mappings[set_to_indices[idx]]
            for _ in range(count):
                E[i] = X[members, :]
                E_idx[i] = members
                i += 1
        if k > i:
            continue

        ls = kmedkpm.k_median_k_partitions_LS(
            E, X, None, n, d, k, seed, 0.0, is_coreset=False
        )
        if best is None or ls["cost"] < best["cost"]:
            center_idx = np.array(
                [E_idx[j][ls["facilities_indices"][j]] for j in range(k)]
            )
            best = {
                "cost": ls["cost"],
                "center_idx": center_idx,
                "iterations": ls["iterations"],
            }

    return best, len(patterns)


def run_divkmeans(instance, split):
    start = time.time()
    rvec = create_rvec(
        G=instance["G"],
        t=instance["t"],
        k=instance["k"],
        split=split,
        prng=np.random.default_rng(instance["seed"])
    )
    with open(os.devnull, "w") as logfile:
        best, n_patterns = lp_ls1_kmeans(instance, rvec, logfile)
    runtime = time.time() - start

    to_store = ["dataset", "n", "d", "t", "k", "n_c", "seed"]
    output = {key: instance[key] for key in to_store}
    output["algorithm"] = "divkmeans_LP_LS1"
    output["split"] = split
    output["rvec"] = rvec.tolist()
    output["n_patterns"] = n_patterns
    output["l"] = 1
    output["runtime"] = runtime
    output["postprocess_runtime"] = 0
    output["total_runtime"] = runtime

    if best is None:
        output["status"] = "infeasible"
        return [output]

    center_idx = best["center_idx"]
    _, first = np.unique(instance["U"][center_idx], axis=0, return_index=True)
    center_idx = center_idx[np.sort(first)]

    centers = instance["U"][center_idx]
    assignments = assign_points(instance["U"], centers)
    cost = compute_kmeans_cost(instance["U"], centers, assignments)
    counts = prepare_group_cluster_counts(instance, assignments)
    metrics = compute_metrics(
        counts, np.asarray(instance["G"])[center_idx]
    )

    output["status"] = "completed"
    output["centers"] = centers.tolist()
    output["post_centers"] = centers.tolist()
    output["center_idx"] = center_idx.tolist()
    output["n_centers"] = len(center_idx)
    output["pre_cost"] = cost
    output["post_cost"] = cost
    output["cost"] = cost
    output["ls1_cost"] = best["cost"] * instance["n"]  
    output["ls1_iterations"] = best["iterations"]
    output.update(metrics)
    return [output]


if __name__ == "__main__":
    DATASETS = {
        "heart": load_heart,
        "student": load_student,
        "census_adult": load_census_adult,
        "bank": load_bank,
        "credit_card": load_credit_card,
        "movie_lens": load_movielens_1m
    }

    MAX_WORKERS = 32

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices={"equitable", "proportional"}, default="equitable")
    parser.add_argument("--minprop", default=0, type=float)
    args = parser.parse_args()

    name = args.dataset
    load_dataset = DATASETS[name]
    min_prop = args.minprop
    split = args.split

    dataset = load_dataset(min_prop)
    ROWS = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for instance in generate_instances(dataset, name):
            futures += [
                ex.submit(run_divkmeans, instance.copy(), split)
            ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            ROWS.extend(future.result())

    pd.DataFrame(ROWS).to_csv(
        f"results/{name}_divkmeans_{split}_{min_prop:g}.csv",
        index=False,
    )
