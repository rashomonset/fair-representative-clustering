import numpy as np
import random

from fair_k_supplier_3apx import find_closest_facility
from fair_k_supplier_3apx import compute_k_supplier_cost
from helper_routines import *

########################################################################################
def k_supplier_3apx(I, seed):
    prng = np.random.default_rng(seed)

    n_c    = I["n_c"]
    k      = I["k"]
    C_idx  = I["C_idx"]
    F_idx  = I["F_idx"]
    U      = I["U"]
    d      = U.shape[1]
    C = U if n_c == len(U) else U[C_idx, :]
    F = U if I["n_f"] == len(U) else U[F_idx, :]

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

    S = []
    S_idx = []
    for i in range(k):
        c_i = Cp[i]
        s, s_idx = find_closest_facility(c_i, F)
        S.append(s)
        S_idx.append(s_idx)
    # end for
    cost = compute_k_supplier_cost(C, np.vstack(S))

    return S, S_idx, cost, 3
# end of k_supplier_three_apx()
