"""
Code taken from https://github.com/suhastheju/fair-k-supplier-source
"""
import numpy as np
import random
import sys
import math

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

import psutil
import time

####################################################################################################
def l1_distances(points, center):
    distances = np.abs(points[:, 0] - center[0])
    for dimension in range(1, points.shape[1]):
        distances += np.abs(
            points[:, dimension] - center[dimension]
        )
    return distances

def contains_facility(f, G_i):
    return np.any(np.all(G_i == f, axis=1))

def get_facility_type_with_S(S, GG):
    facility_type = np.full(len(S), -1)
    for i in range(len(S)):
        s_i = S[i]
        for j in range(len(GG)):
            G_j = GG[j]
            if contains_facility(s_i, G_j):
                facility_type[i] = j
                break
            #end if
        # end for
    # end for
    return facility_type
#end get_facility_type()

def get_facility_type(S_idx, GG_idx):
    facility_type = np.full(len(S_idx), -1)
    for i in range(len(S_idx)):
        s_idx = S_idx[i]
        for j in range(len(GG_idx)):
            G_idx_j = GG_idx[j]
            if s_idx in G_idx_j:
                facility_type[i] = j
                break
            #end if
        # end for
    # end for
    return facility_type
#end get_facility_type()

def choose_farthest_client(C, Cp):
    distances = np.sum(np.abs(C[:, np.newaxis] - Cp), axis=2)
    min_distances = np.min(distances, axis=1)
    farthest_client_index = np.argmax(min_distances)
    return C[farthest_client_index]
# end choose_farthest_client()

def get_sorted_radii(F, Cp):
    distances = np.sum(np.abs(F[:, np.newaxis] - Cp), axis=2)
    unique_distances = np.unique(distances)
    return np.sort(unique_distances)
#end get_sorted_radii()

# def compute_k_supplier_cost(C, S):
#     distances = np.sum(np.abs(C[:, np.newaxis] - S), axis=2)
#     min_distances = np.min(distances, axis=1)
#     cost = np.max(min_distances)
#     return cost
# # end compute_k_supplier_cost()

def compute_k_supplier_cost(C, S):
    min_distances = l1_distances(C, S[0])
    for center in S[1:]:
        distances = l1_distances(C, center)
        np.minimum(
            min_distances,
            distances,
            out=min_distances,
        )
    return min_distances.max()

# def is_dist_ci_and_Gj_lmbda(c, G_j, lmbda):
#     distances = np.sum(np.abs(G_j - c), axis=1)
#     return np.any(distances <= lmbda)
# end is_dist_ci_and_Gj_lmbda()

def is_dist_ci_and_Gj_lmbda(c, G_j, lmbda):
    return np.any(l1_distances(G_j, c) <= lmbda)

def min_distance(c, G_j):
    distances = np.sum(np.abs(G_j - c), axis=1)
    return np.min(distances)
# end min_distance()

# def find_facility_within_lmbda(c, G_j, lmbda):
#     distances = np.sum(np.abs(G_j - c), axis=1)
#     return G_j[np.argmin(distances)]
# # end find_facility_within_lambda()

# def find_closest_facility(c, F):
#     distances = np.sum(np.abs(F - c), axis=1)
#     s_idx = np.argmin(distances)
#     return F[s_idx], s_idx
# # end find_closest_facility()

def find_closest_facility(c, F):
    distances = l1_distances(F, c)
    facility_index = np.argmin(distances)
    return F[facility_index], facility_index

def extend_graph_with_capacities_mk1(adj_matrix, capacities):
    # Number of nodes in Set 1 (rows) and Set 2 (columns)
    n_set1, n_set2 = adj_matrix.shape

    # Create a new adjacency matrix based on capacities
    new_columns = sum(capacities)  # Total number of new nodes in extended Set 2
    extended_matrix = np.zeros((n_set1, new_columns))

    # Map to track which original Set 2 node corresponds to each extended column
    original_to_extended_map = []

    # Populate the extended matrix
    col_idx = 0
    for i in range(n_set2):  # Iterate over the original Set 2 nodes
        for _ in range(capacities[i]):  # Duplicate columns based on the capacity
            extended_matrix[:, col_idx] = adj_matrix[:, i]
            original_to_extended_map.append(i)  # Track which original column this extended column maps to
            col_idx += 1

    return extended_matrix, original_to_extended_map
# end extend_graph_with_capacities_mk1()

def extend_graph_with_capacities(adj_matrix, capacities):
    # Number of nodes in Set 1 (rows) and Set 2 (columns)
    n_set1, n_set2 = adj_matrix.shape

    # Create a new adjacency matrix based on capacities
    new_columns = sum(cap for cap in capacities if cap > 0)  # Only count non-zero capacities
    extended_matrix = np.zeros((n_set1, new_columns))

    # Map to track which original Set 2 node corresponds to each extended column
    original_to_extended_map = []

    # Populate the extended matrix
    col_idx = 0
    for i in range(n_set2):  # Iterate over the original Set 2 nodes
        if capacities[i] == 0:
            continue  # Skip nodes with zero capacity

        for _ in range(capacities[i]):  # Duplicate columns based on the capacity
            extended_matrix[:, col_idx] = adj_matrix[:, i]
            original_to_extended_map.append(i)  # Track which original column this extended column maps to
            col_idx += 1

    return extended_matrix, original_to_extended_map
# end extend_graph_with_capacities()

####################################################################################################
# debugging information
def compare_instances(I1, I2):
    # Check if both have the same keys
    if I1.keys() != I2.keys():
        return False
    
    for key in I1:
        value1 = I1[key]
        value2 = I2[key]
        
        # If values are numpy arrays, use np.array_equal to compare
        if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            if not np.array_equal(value1, value2):
                return False
        
        # If values are lists, compare each element
        elif isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                return False
            for v1, v2 in zip(value1, value2):
                if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                    if not np.array_equal(v1, v2):
                        return False
                elif v1 != v2:
                    return False

        # For other types (int, str, etc.), use direct comparison
        elif value1 != value2:
            return False
    
    return True
# end compare_instances()

def check_valid_solution(I, S_idx):
    GG_idx = I["GG_idx"]
    rvec  = I["rvec"]
    t     = I["t"]

    rvec_S = np.zeros(t, dtype=int)
    for s_idx in S_idx:
        for j in range(t):
            G_idx_j = GG_idx[j]
            if s_idx in G_idx_j:
                rvec_S[j] += 1
                break
            # end if
        # end for
    # end for
    return np.array_equal(rvec, rvec_S)
# end check_valid_solution()

def get_rvec_from_S_idx(S_idx, GG_idx):
    t = len(GG_idx)
    rvec_S = np.zeros(t, dtype=int)
    for s_idx in S_idx:
        for j in range(t):
            G_idx_j = GG_idx[j]
            if s_idx in G_idx_j:
                rvec_S[j] += 1
            # end if
        # end for
    # end for
    return rvec_S
# end get_rvec_from_S_idx()

def debug_stats(process, tstart, logfile=sys.stdout):
    logfile.write("=======================================================================\n")
    logfile.write("STATISTICS: \n")
    logfile.write("--------------\n")
    logfile.write("CPU:           [CORES-TOTAL: %s, THREADS-TOTAL: %s, THREADS-USED: %s]\n"%\
                     (psutil.cpu_count(logical=False), psutil.cpu_count(), process.num_threads()))
    logfile.write("MEMORY:        [RAM-MEM: %.2fMB, VIR-MEM: %.2fMB, TOTAL: %.2fMB]\n"%\
                     (process.memory_info().rss/(1024*1024),\
                      process.memory_info().vms/(1024*1024),\
                      (process.memory_info().rss+process.memory_info().vms)/(1024*1024)))
    logfile.write("TOTAL-TIME:    %.2fs\n"%(time.time()-tstart))
    logfile.write("======================================================================\n")
    logfile.flush()
#end debug_details()
