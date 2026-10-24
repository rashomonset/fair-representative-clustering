from k_supplier_3apx import k_supplier_3apx
from helper_routines import *
import numpy as np


def assign_points_l1(points, centers):
    assignments = np.zeros(len(points), dtype=int)
    min_distances = l1_distances(points, centers[0])
    for center_index in range(1, len(centers)):
        distances = l1_distances(
            points,
            centers[center_index],
        )
        closer = distances < min_distances
        min_distances[closer] = distances[closer]
        assignments[closer] = center_index
    return assignments


def k_supplier(I,seed):
    S, S_idx, cost, apx_ratio = k_supplier_3apx(I,seed)
    _, positions = np.unique(S_idx, return_index=True)
    positions = np.sort(positions)
    S = np.asarray(S)[positions]
    S_idx = np.asarray(S_idx)[positions]
    return S, S_idx, cost, apx_ratio


def create_utility_matrix(I,k_supplier,objective):

    _, center_idx, cost, apx_ratio = k_supplier(I, I["seed"])
    center_idx = I["F_idx"][np.asarray(center_idx, dtype=int)]
    centers = I["U"][center_idx]

    assignments = assign_points_l1(
        I["U"],
        centers,
    )

    t = I["t"]
    k = len(center_idx)
    G = np.asarray(I["G"], dtype=int)
    group_sizes = np.bincount(G, minlength=t)
    cluster_sizes = np.bincount(assignments, minlength=k)

    U = np.zeros((t, k), dtype=float)
    for j in range(k):
        members = assignments == j
        cluster_idx = np.flatnonzero(members)
        hits = np.bincount(G[members], minlength=t)
        for i in range(t):
            if np.intersect1d(cluster_idx, I["GG_idx"][i]).size:
                if objective=="global":
                    U[i, j] = hits[i] / group_sizes[i]
                else:
                    U[i, j] = hits[i] / cluster_sizes[j]

    return U, centers, cost, assignments, apx_ratio


def compute_objective(allocations,utilities):
    m, n = utilities.shape
    T = np.zeros(m)
    for i in range(n):
        T[allocations[i]] += utilities[allocations[i],i]
    return min(T)


def compute_group_representation_metric(I, assignments, allocations, objective):
    G = np.asarray(I["G"], dtype=int)
    cluster_sizes = np.bincount(assignments, minlength=I["k"])
    group_sizes = np.bincount(G, minlength=I["t"])
    metric = np.zeros(I["t"])
    for j, group in enumerate(allocations):
        members = assignments == j
        hits = np.count_nonzero(G[members] == group)
        if objective=="global":
            metric[group] += hits / group_sizes[group]
        else:
            metric[group] += hits / cluster_sizes[j]
    return metric.min()


def prepare_group_cluster_counts(I,assignments):
    G = np.asarray(I["G"], dtype=int)
    t, k = I["t"], I["k"]
    return np.bincount(
        G * k + assignments,
        minlength=t * k,
    ).reshape(t, k)


def compute_metrics(counts, allocations):

    group_sizes = counts.sum(axis=1)
    cluster_sizes = counts.sum(axis=0)

    t = group_sizes.shape[0]
    global_metric = np.zeros(t)
    local_metric = np.zeros(t)

    for j, group in enumerate(allocations):
        hits = counts[group, j]
        global_metric[group] += hits / group_sizes[group]
        local_metric[group] += hits / cluster_sizes[j]

    return {
        "center_groups": allocations.tolist(),

        "global_metric_by_group": global_metric.tolist(),
        "global_metric_min": float(global_metric.min()),
        "global_metric_max": float(global_metric.max()),
        "global_metric_std": float(global_metric.std()),

        "local_metric_by_group": local_metric.tolist(),
        "local_metric_min": float(local_metric.min()),
        "local_metric_max": float(local_metric.max()),
        "local_metric_std": float(local_metric.std()),
    }

def compute_clustering_cost(I, centers, assignments, allocations):
    clustering_cost = 0.0
    for j, group in enumerate(allocations):
        members = np.flatnonzero(assignments == j)
        if members.size == 0:
            continue
        candidates = np.intersect1d(members, I["GG_idx"][group])
        distances = l1_distances(I["U"][candidates], centers[j])
        center_idx = candidates[np.argmin(distances)]
        cluster_cost = l1_distances(
            I["U"][members], I["U"][center_idx]
        ).max()
        clustering_cost = max(clustering_cost, float(cluster_cost))
    return clustering_cost

def compute_metrics_baseline(I, center_idx):

    G = np.asarray(I["G"], dtype=int)
    t = I["t"]

    centers = I["U"][center_idx]
    assignments = assign_points_l1(
        I["U"],
        centers,
    )
    center_groups = G[center_idx]
    group_sizes = np.bincount(G, minlength=t)
    cluster_sizes = np.bincount(assignments, minlength=I["k"])

    local_metric = np.zeros(t, dtype=float)
    global_metric = np.zeros(t, dtype=float)

    for j, allocated_group in enumerate(center_groups):
        members = assignments == j
        cluster_size = int(members.sum())
        if cluster_size == 0:
            continue
        hits = np.bincount(G[members], minlength=t)
        global_metric[allocated_group] += hits[allocated_group] / group_sizes[allocated_group]
        local_metric[allocated_group] += hits[allocated_group] / cluster_sizes[j]

    return {
        "centers": I["U"][center_idx].tolist(),
        "center_groups": center_groups.tolist(),

        "global_metric_by_group": global_metric.tolist(),
        "global_metric_min": float(global_metric.min()),
        "global_metric_max": float(global_metric.max()),
        "global_metric_std": float(global_metric.std()),

        "local_metric_by_group": local_metric.tolist(),
        "local_metric_min": float(local_metric.min()),
        "local_metric_max": float(local_metric.max()),
        "local_metric_std": float(local_metric.std()),
    }
