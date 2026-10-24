from gurobipy import GRB
import gurobipy as gp
import numpy as np

import time

from utils import *


def randomized_rounding(X,l,prng):
    m, n = X.shape
    allocations = []
    for _ in range(l):
        allocation = []
        for j in range(n):
            p = np.maximum(X[:, j], 0)
            p /= p.sum()
            allocation.append(prng.choice(m, p=p))
        allocations.append(allocation)
    return np.asarray(allocations)


def max_min_allocation(utilities,seed,logfile,timelimit=None):
    """
    m agents
    n goods
    utilities[i,j] = utility of good j for agent i 
    """
    m, n = utilities.shape
    model = gp.Model("maxmin")
    model.setParam("Threads", 1)
    model.setParam("Seed", seed)
    model.setParam("LogToConsole", 0)
    model.setParam("Logfile",logfile)
    if timelimit is not None:
        model.setParam("TimeLimit",timelimit)

    x = model.addVars(m, n, lb=0, name="x", vtype=GRB.CONTINUOUS)
    T = model.addVar(lb=0, name="T", vtype=GRB.CONTINUOUS)
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in range(m)) == 1 for j in range(n))
    )
    model.addConstrs(
        (
            gp.quicksum(utilities[i, j] * x[i, j] for j in range(n)) >= T
            for i in range(m)
        )
    )
    model.addConstrs(
        (x[i, j] == 0
        for i in range(m)
        for j in range(n)
        if utilities[i, j] == 0)
    )
    model.setObjective(T, GRB.MAXIMIZE)
    model.optimize()
    assignments = np.array([[x[i, j].X for j in range(n)] for i in range(m)])
    return T.X, assignments


def assignment_LP(I, objective, prepared, l_values=None, timelimit=None):

    prng = np.random.default_rng(I["seed"])
    U, centers, assignments, pre_cost, radii_sq, prep_runtime = prepared

    if l_values is None:
        l_values = {2**i for i in range(11)}
    else:
        l_values = set(l_values)

    max_l = max(l_values)

    if not np.all(U.any(axis=0)):
        return [{
            "l": l, "status": "infeasible", "runtime": prep_runtime,
            "pre_cost": pre_cost, "post_cost": np.nan, "LP_obj": np.nan,
        } for l in sorted(l_values)]

    start = time.time()
    LP_obj, x = max_min_allocation(
        utilities=U,
        seed=I["seed"],
        logfile=f"logs/{I['dataset']}_assignment_LP_{I['seed']}.log",
        timelimit=timelimit
    )
    counts = prepare_group_cluster_counts(I, assignments)
    lp_runtime = time.time() - start

    rounding_runtime = 0
    pre_runtime = 0
    best_metric = -np.inf
    best_allocation = None

    outputs = []

    for r in range(1, max_l + 1):

        # Generate a rounded allocation
        start = time.time()
        allocation = randomized_rounding(x, 1, prng)[0]
        rounding_runtime += time.time() - start

        # Evaluate before center reassignment
        start = time.time()
        metric = compute_metrics(counts, allocation)[f"{objective}_metric_min"]
        if metric > best_metric:
            best_metric = metric
            best_allocation = allocation.copy()
        pre_runtime += time.time() - start

        if r in l_values:
            outputs.append({
                "runtime": (
                    prep_runtime
                    + lp_runtime
                    + rounding_runtime
                    + pre_runtime
                ),
                "status": "completed",
                "l":r,
                "pre_cost":pre_cost,
                "post_cost":compute_clustering_cost(
                    I, 
                    centers, 
                    assignments, 
                    best_allocation, radii_sq
                ),
                "LP_obj":LP_obj,
                **compute_metrics(
                    counts, best_allocation
                ),
            })
    return outputs
