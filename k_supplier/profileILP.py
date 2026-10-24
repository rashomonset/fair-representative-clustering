from pathlib import Path
import os 

from gurobipy import GRB
import gurobipy as gp
import numpy as np
import time

from helper_routines import compute_k_supplier_cost
from utils import *

def ILP(utilities,epsilon,seed,logfile,threads,timelimit=None):
    """
    m agents
    n goods
    utilities[i,j] = utility type of good j for agent i 
    """
    m, n = utilities.shape
    U = np.zeros((m,n))
    # Round Utilities
    for i in range(m):
        for j in range(n):
            if utilities[i,j]>=epsilon:
                k = int ( np.floor( np.log(utilities[i,j]) / np.log1p(epsilon) ) )
                U[i,j] = np.power(1+epsilon,k)

    active = np.any(U > 0, axis=0)
    profiles, cluster_to_profile, counts = np.unique(
        U[:, active].T,
        axis=0,
        return_inverse=True,
        return_counts=True
    )
    num_profiles = len(profiles)
    model = gp.Model("additive_apx")
    model.setParam("Seed", seed)
    model.setParam("Threads", 1)
    model.setParam("LogToConsole", 0)
    model.setParam("Logfile",logfile)
    model.setParam("MIPGap",1e-3)
    if timelimit is not None:
        model.setParam("TimeLimit",timelimit)

    x = model.addVars(num_profiles, m, vtype=GRB.INTEGER, lb=0, name="x")
    C = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="C")

    for t in range(num_profiles):
        model.addConstr(gp.quicksum(x[t, i] for i in range(m)) == counts[t])

    for i in range(m):
        model.addConstr(gp.quicksum(profiles[t, i] * x[t, i] for t in range(num_profiles)) >= C)
    
    model.addConstrs(
        (x[t, i] == 0
        for t in range(num_profiles)
        for i in range(m)
        if profiles[t, i] == 0)
    )

    model.setObjective(C, GRB.MAXIMIZE)
    model.optimize()
    status = int(model.Status)
    gap = (
        float(model.MIPGap) 
        if model.IsMIP and model.SolCount > 0 
        else None
    )
    ILP_obj = float(model.ObjVal)
    ILP_bound = float(model.ObjBound)
    absolute_gap = abs(ILP_bound - ILP_obj)
    X = np.rint([[x[p, i].X for i in range(m)] for p in range(num_profiles)]).astype(int).reshape(num_profiles, m)
    return ILP_obj, X, gap, status, cluster_to_profile, active, ILP_bound, absolute_gap


def recover_allocations(X, inverse, prng):
    num_profiles, m = X.shape
    n = len(inverse)
    allocations = np.zeros(n, dtype=int)
    for t in range(num_profiles):
        clusters = np.where(inverse==t)[0]
        prng.shuffle(clusters)
        idx = 0
        for i in range(m):
            c = X[t,i]
            allocations[clusters[idx:idx+c]] = i
            idx += c
    return allocations


def additive_apx_ILP(I, objective, prepared, timelimit=None):

    prng = np.random.default_rng(I["seed"])
    U, centers, pre_cost, assignments, apx_ratio, prep_runtime = prepared

    start = time.time()

    counts = prepare_group_cluster_counts(I, assignments)
    epsilon = I["epsilon"]
    ILP_obj, X, gap, status, cluster_to_profile, active, ILP_bound, absolute_gap = ILP(
        utilities=U,
        epsilon=epsilon,
        seed=I["seed"],
        logfile=f"logs/{I['dataset']}_additive_apx_ILP_{I['seed']}.log",
        threads=1,
        timelimit=timelimit
    )

    active_allocations = recover_allocations(
        X,
        inverse=cluster_to_profile,
        prng=prng
    )

    allocations = np.full(U.shape[1], -1, dtype=int)
    allocations[active] = active_allocations

    inactive = np.flatnonzero(~active)
    if inactive.size:
        allocations[inactive] = np.argmax(
            U[:,inactive],
            axis=0,
        )

    runtime = prep_runtime + time.time() - start
    metrics = compute_metrics(counts,allocations)
    metrics["centers"]=centers.tolist()

    return [{
        "runtime":runtime,
        "apx_ratio":apx_ratio,
        "pre_cost":pre_cost,
        "post_cost": compute_clustering_cost(
            I, centers, assignments, allocations
        ),
        "ILP_obj":ILP_obj,
        "gap":gap,
        "status":status,
        "ILP_bound":ILP_bound,
        "absolute_gap":absolute_gap,
        **metrics,
    }]
