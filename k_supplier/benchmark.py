from sklearn.model_selection import ParameterGrid
from pathlib import Path
import argparse
import signal
import json
import time

import numpy as np
import pandas as pd

from data_utils import create_facilities, create_rvec, load_synthetic
from utils import *
from fair_k_supplier_3apx import fair_k_supplier_3apx
from profileILP import additive_apx_ILP
from assignmentLP import assignment_LP

ROUNDING_TRIALS = 64
DEFAULT_MASTER_SEED = 42
DEFAULT_TIMEOUT_SECONDS = 1800

RESULT_COLUMNS = [
    "seed", "dataset", "split", "objective", "algorithm",
    "n", "d", "t", "k", "n_c", "n_f", "facility_fraction",
    "group_sizes", "GG_sizes", "rvec", "center_groups",
    "epsilon","delta","l","apx_ratio",
    "LP_obj", "ILP_obj","gap","absolute_gap","ILP_bound","status",
    "construction_runtime","allocation_runtime","total_runtime",
    "cost", "pre_cost", "post_cost", 
    "global_metric_by_group","global_metric_min", "global_metric_max", "global_metric_std",
    "local_metric_by_group","local_metric_min", "local_metric_max", "local_metric_std"
]

def raise_timeout(signum, frame):
    raise TimeoutError("Fair k-supplier timed out")

def generate_instances(master_seed=DEFAULT_MASTER_SEED):

    master_rng = np.random.default_rng(master_seed)

    base_grid = ParameterGrid({
        "seed": master_rng.integers(0, 10**9, size=10),
         "n": 10 ** np.arange(3, 8),
         "t": 2 ** np.arange(1, 6),
    })

    configuration_grid = ParameterGrid({
        "facility_fraction": (0.2, 1.0),
        "k_multiplier": (2, 4, 6, 8, 10),
    })

    for params in base_grid:

        seed = params["seed"]
        n = params["n"]
        t = params["t"]

        dataset = load_synthetic(
            n=n,
            t=t,
            seed=int(seed),
            min_group_size=10,
        )

        for config in configuration_grid:

            n_f = int(np.ceil(config["facility_fraction"]*n))
            k = config["k_multiplier"]*t

            if n_f < k:
                continue

            run_rng = np.random.default_rng(seed)
            rvec = create_rvec(
                G=dataset["G"],
                t=t,
                k=k,
                split="equitable",
                prng=run_rng,
            )
            F_idx, GG_idx, GG_sizes = create_facilities(
                G=dataset["G"],
                t=t,
                n_f=n_f,
                rvec=rvec,
                prng=run_rng,
            )

            yield {
                "dataset": "synthetic",
                "split": "equitable",
                "n": n,
                "d": dataset["d"],
                "t": t,
                "k": k,
                "n_c": n,
                "n_f": n_f,
                "facility_fraction": config["facility_fraction"],
                "C_idx": np.arange(n, dtype=np.int32),
                "F_idx": F_idx,
                "GG_idx": GG_idx,
                "GG_sizes": GG_sizes,
                "rvec": rvec,
                "G": dataset["G"],
                "U": dataset["U"],
                "group_sizes": dataset["group_sizes"],
                "seed": int(seed),
            }


def prepare_instance(instance, objective):
    start = time.perf_counter()
    utilities, centers, pre_cost, assignments, apx_ratio = (
        create_utility_matrix(
            instance,
            k_supplier,
            objective,
        )
    )
    prep_runtime = time.perf_counter() - start
    return (
        utilities,
        centers,
        pre_cost,
        assignments,
        apx_ratio,
        prep_runtime,
    )

def run_algorithm(instance,prepared,objective,method,timelimit=1800):

    algorithm = method["algorithm"]
    run_instance = instance.copy()
    metadata_keys = (
        "dataset", "split", "n", "d", 
        "t", "k","n_c", "n_f", "facility_fraction",
        "group_sizes", "GG_sizes", "rvec", "seed",
    )
    metadata = {
        key: run_instance[key]
        for key in metadata_keys
    }
    metadata.update({
        "algorithm": algorithm,
        "objective": objective,
    })

    start = time.perf_counter()

    if algorithm == "profileILP":
        delta = float(method["delta"])
        epsilon = delta/run_instance["k"]
        run_instance["epsilon"] = float(epsilon)
        metadata["delta"] = delta
        metadata["epsilon"] = run_instance["epsilon"]
        outputs = additive_apx_ILP(
            I=run_instance,
            objective=objective,
            prepared=prepared,
            timelimit=timelimit
        )
    elif algorithm == "assignmentLP":
        metadata["l"] = ROUNDING_TRIALS
        outputs = assignment_LP(
            I=run_instance,
            objective=objective,
            prepared=prepared,
            l_values={ROUNDING_TRIALS},
            timelimit=timelimit
        )
    elif algorithm == "fairksupplier":
        previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, timelimit)
        try:
            _, center_idx, cost, apx_ratio = fair_k_supplier_3apx(
                run_instance, run_instance["seed"]
            )
            metrics = compute_metrics_baseline(run_instance, center_idx)
            outputs = [{
                "status": "completed",
                "cost": cost,
                "post_cost": cost,
                "apx_ratio": apx_ratio,
                **metrics,
            }]
        except TimeoutError:
            outputs = [{"status": "timeout"}]
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    is_baseline = algorithm == "fairksupplier"

    if is_baseline:
        construction_runtime = 0.
        allocation_runtime = 0.
        total_runtime = time.perf_counter() - start
    else:
        construction_runtime = prepared[-1]
        allocation_runtime = outputs[0]["runtime"] - construction_runtime
        total_runtime = construction_runtime + allocation_runtime
    return [
        {
            **metadata,
            **output,
            "construction_runtime": construction_runtime,
            "allocation_runtime": allocation_runtime,
            "total_runtime": total_runtime,
        }
        for output in outputs
    ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--objective",choices=("global", "local"),default="global")
    parser.add_argument("--output",type=Path,default=Path("results/benchmark.csv"))
    parser.add_argument(
            "--method",
            type=str,
            default=[
                {"algorithm": "fairksupplier"},
                {'algorithm': 'assignmentLP'}, 
                {'algorithm': 'profileILP', 'delta': 0.01}
            ]
        )
    args = parser.parse_args()

    if isinstance(args.method,str):
        methods = [{'algorithm': args.method}]
        if args.method == "profileILP":
            methods[0]['delta'] = .01 
    else:
        methods = args.method

    for instance_number, instance in enumerate(generate_instances(DEFAULT_MASTER_SEED),start=1):

        prepared = None
        rows = []
        elapsed = 0

        for method in methods:

            if prepared is None:
                prepared = prepare_instance(instance, args.objective)
                elapsed = prepared[-1]

            if method["algorithm"] == "fairksupplier":
                effective_timelimit = elapsed + DEFAULT_TIMEOUT_SECONDS
            else:
                effective_timelimit = DEFAULT_TIMEOUT_SECONDS

            result = run_algorithm(
                instance=instance,
                prepared=prepared,
                objective=args.objective,
                method=method,
                timelimit=effective_timelimit
            )
            rows.extend(result)

        frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        frame.to_csv(
            args.output,
            mode="a",
            header=not args.output.exists(),
            index=False,
        )
        print(
            f"[{instance_number}] "
            f"n={instance['n']} "
            f"t={instance['t']} "
            f"k={instance['k']} "
            f"n_f={instance['n_f']} ",
            flush=True,
        )
