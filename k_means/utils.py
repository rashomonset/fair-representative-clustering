from sklearn.cluster import kmeans_plusplus as initialize_kmeans_plusplus
import numpy as np

def kmeans_plusplus(I, seed, centroids=False):
    centers, center_idx = initialize_kmeans_plusplus(
        I["U"],
        n_clusters=I["k"],
        random_state=seed,
        n_local_trials=1
    )
    assignments = assign_points(I["U"], centers)
    used, assignments = np.unique(assignments, return_inverse=True)
    centers = centers[used]
    center_idx = center_idx[used]
    if centroids:
        centers = np.vstack([
            I["U"][assignments == j].mean(axis=0)
            for j in range(len(centers))
        ])
    cost = compute_kmeans_cost(I["U"], centers, assignments)
    return centers, center_idx, assignments, cost


def assign_points(points, centers):
    distances = (
        np.square(points).sum(axis=1)[:, None]
        + np.square(centers).sum(axis=1)
        - 2 * points @ centers.T
    )
    return np.argmin(distances, axis=1)


def compute_kmeans_cost(points, centers, assignments=None):
    if assignments is None:
        assignments = assign_points(points, centers)
    return np.square(points - centers[assignments]).sum()


def create_utility_matrix(I, centers, assignments, alpha, objective):

    t = I["t"]
    k = len(centers)
    G = np.asarray(I["G"], dtype=int)
    group_sizes = np.bincount(G, minlength=t)
    cluster_sizes = np.bincount(assignments, minlength=k)

    point_costs = np.square(I["U"] - centers[assignments]).sum(axis=1)
    cluster_costs = np.bincount(
        assignments,
        weights=point_costs,
        minlength=k
    )
    radii_sq = alpha * cluster_costs / cluster_sizes
    eligible = point_costs <= radii_sq[assignments]

    U = np.zeros((t, k), dtype=float)
    for j in range(k):
        members = assignments == j
        hits = np.bincount(G[members], minlength=t)
        valid = np.bincount(G[members & eligible], minlength=t) > 0
        if objective == "global":
            U[valid, j] = hits[valid] / group_sizes[valid]
        else:
            U[valid, j] = hits[valid] / cluster_sizes[j]

    return U, radii_sq


def recover_centers(I, centers, assignments, allocations, radii_sq):
    G = np.asarray(I["G"], dtype=int)
    point_costs = np.square(I["U"] - centers[assignments]).sum(axis=1)
    center_idx = []
    for j, group in enumerate(allocations):
        candidates = np.flatnonzero(
            (assignments == j)
            & (G == group)
            & (point_costs <= radii_sq[j])
        )
        center_idx.append(candidates[np.argmin(point_costs[candidates])])
    return np.asarray(center_idx, dtype=int)


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


def compute_clustering_cost(I, centers, assignments, allocations, radii_sq):
    center_idx = recover_centers(I, centers, assignments, allocations, radii_sq)
    return compute_kmeans_cost(I["U"], I["U"][center_idx], assignments)
