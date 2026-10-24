"""
Code taken from https://github.com/suhastheju/fair-k-supplier-source
"""
import numpy as np
import random
import sys
import math
import time
import psutil

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from helper_routines import find_closest_facility, check_valid_solution, debug_stats
from helper_routines import l1_distances, get_sorted_radii, is_dist_ci_and_Gj_lmbda
from helper_routines import extend_graph_with_capacities, compute_k_supplier_cost


def get_sorted_radii(F, Cp):
    distances = np.empty((len(F), len(Cp)))
    for center_index, center in enumerate(Cp):
        distances[:, center_index] = l1_distances(F, center)
    return np.unique(distances)

###############################################################################
def fair_k_supplier_3apx(I, seed):
    prng = np.random.default_rng(seed)

    n_c      = I["n_c"]
    k        = I["k"]
    C_idx    = I["C_idx"]
    F_idx    = I["F_idx"]
    GG_idx   = I["GG_idx"]
    GG_sizes = I["GG_sizes"]
    rvec     = I["rvec"]
    t        = I["t"]  # number of groups
    U        = I["U"]
    d        = U.shape[1]  # dimension of clients and facilities

    # Extract clients, facilities and groups
    C = U if n_c == len(U) else U[C_idx, :]
    F = U if I["n_f"] == len(U) else U[F_idx, :]
    GG       = [U[GG_idx[i],:] for i in range(t)]

    # Initialize and choose k farthest clients
    Cp = np.zeros((k, d))
    Cp[0] = C[prng.integers(0, n_c)]

    min_distances = l1_distances(C, Cp[0])

    for i in range(1, k):
        Cp[i] = C[np.argmax(min_distances)]
        if i < k - 1:
            distances = l1_distances(C, Cp[i])
            np.minimum(
                min_distances,
                distances,
                out=min_distances,
            )
    # end for

    S = []
    S_idx = []
    radius = math.inf
    cost = math.inf

    # Iterate over l = 1 to k for each group of clients
    left, right = 1, k+1
    # for l in range(1, k + 1):
    while left <= right:
        l = (left + right) // 2
        Sl = []
        Sl_idx = []
        Gammal = get_sorted_radii(F, Cp[:l])
        lmbdal = math.inf
        cost_Sl = math.inf

        # Process each λ in sorted radii
        low, high = 0, len(Gammal)-1
        valid_lambda = False

        # for lmbda in Gammal:
        while low <= high:
            mid          = (low + high) // 2
            lmbda        = Gammal[mid]
            Sl_lmbda     = []
            Sl_lmbda_idx = []

            # Construct H_k_λ graph using broadcasting instead of loops
            Hl_lmbda = np.zeros((l, t), dtype=int)
            for i in range(l):
                c_i = Cp[i]
                for j in range(t):
                    G_j = GG[j]
                    if is_dist_ci_and_Gj_lmbda(c_i, G_j, lmbda):
                        Hl_lmbda[i, j] = 1
                    # end if
                # end for
            # end for

            # Extend the graph based on number of requirements
            Hl_lmbda_ext, ext_map = extend_graph_with_capacities(Hl_lmbda, rvec)
            sparse_Hl_lmbda = csr_matrix(Hl_lmbda_ext)

            # Find maximum matching
            Ml_lmbda_ext = maximum_bipartite_matching(sparse_Hl_lmbda, perm_type='column')
            Ml_lmbda = [ext_map[col] if col != -1 else -1 for col in Ml_lmbda_ext]

            if -1 in Ml_lmbda:
                low = mid + 1 # continue searching in the right half
                continue  # No valid matching, skip this λ
            else: 
                valid_lambda = True # Found a valid matching

                # Build solution S^l_λ
                rvec_diff = rvec.copy()
                for i in range(l):
                    j = Ml_lmbda[i]
                    c_i = Cp[i]
                    G_j = GG[j]
                    # s = find_facility_within_lmbda(c_i, G_j, lmbda)
                    s, s_idx = find_closest_facility(c_i, G_j)
                    Sl_lmbda.append(s)
                    Sl_lmbda_idx.append(GG_idx[j][s_idx])
                    rvec_diff[j] -= 1  # reduce remaining requirements
                # end for

                # Add remaining facilities to meet requirements
                for j in range(t):
                    G_j = GG[j]
                    r_diff = rvec_diff[j]
                    Sl_lmbda.extend(G_j[:r_diff])  # Directly extend Sl_lmbda
                    Sl_lmbda_idx.extend(GG_idx[j][:r_diff])
                # end for

                # Compute cost and update if improved
                Sl_lmbda_vstack = np.vstack(Sl_lmbda)
                cost_Sl_lmbda = compute_k_supplier_cost(C, Sl_lmbda_vstack)
                if cost_Sl_lmbda < cost_Sl:
                    lmbdal = lmbda
                    cost_Sl = cost_Sl_lmbda
                    Sl = Sl_lmbda
                    Sl_idx = Sl_lmbda_idx
                # end if
            # end if

            high = mid - 1  # continue searching in the left half
            if low >= high:
                break # Exit after finding the best λ
        # end while

        # Update the global solution if this facility subset is better
        if not valid_lambda:
            left = l + 1
        else:
            if cost_Sl < cost:
                radius = lmbdal
                cost = cost_Sl
                S = Sl
                S_idx = Sl_idx
            # end if
            right = l - 1
        # end if
        if left >= right:
            break # Exit after finding the best l
    # end while

    return S, S_idx, cost, 3
#end fair_k_supplier_3apx()
