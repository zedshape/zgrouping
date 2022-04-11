"""
Matrix generation modules of Z-Grouping
============================
submitted to ECML-PKDD 2022
- numba supported version for fast computation
"""

from numba import njit
import numpy as np
from typing import Tuple, List, Any
from time import process_time

@njit
def cumsum_custom(x: np.ndarray) -> Any:
    mat = np.zeros(x.shape[0], dtype=np.int32)
    cum = np.zeros(x.shape[0], dtype=np.int32)
    for i in range(x.shape[1]):
        cum = np.add(cum, x[:, i])
        mat = np.append(mat, cum)
    return mat.reshape(x.shape[1] + 1, x.shape[0])

@njit
def csX(x: np.ndarray) -> Any:
    return cumsum_custom(x).T

@njit
def get_rows(cs0: np.ndarray, alpha: float, a: int, b: int) -> Any:
    x = cs0[:, b] - cs0[:, a]
    p = (-x).argsort()
    return (x[p].cumsum() >= (alpha * (b - a) * np.arange(1, len(x) + 1)))[p.argsort()]

@njit
def generate_final_return(cs0: np.ndarray, alpha: float, row: np.ndarray) -> Tuple[int, int, np.ndarray]:
    return (row[0], row[1], get_rows(cs0, alpha, row[0], row[1]))

def sort_queue(tt: np.ndarray) -> Any:
    s = np.array(list(zip(tt[:, 1] - tt[:, 0], -tt[:, 2])), dtype=[("range", "i4"), ("score", "i4")])
    indices = np.argsort(s, order=["score", "range"])
    return tt[indices]

@njit
def generate_initial_queue(cs0: np.ndarray, X: np.ndarray, alpha: float) -> Tuple[float, np.ndarray]:
    cs00 = cs0.sum(axis=0)
    i = 0
    best = 0.0
    n = X.shape[0]
    m = X.shape[1]

    tt = np.zeros((int((m * (m + 1) / 2)), 3), dtype=np.int32)

    for ab in range(m - 1, -1, -1):
        for a in range(0, m - ab):
            b = a + ab + 1 
            if best <= cs00[b] - cs00[a]:
                tt[
                    i,
                ] = [a, b, np.sum((cs0[:, b] - cs0[:, a])[get_rows(cs0, alpha, a, b)])]
                if best < tt[i, 2]:
                    best = tt[i, 2]
            else:
                tt[
                    i,
                ] = [a, b, cs00[b] - cs00[a]]
            i += 1

    return best, tt

@njit
def update_tt(cs0: np.ndarray, alpha: float, X: np.ndarray, tt: np.ndarray) -> Tuple[float, np.ndarray]:
    cs = csX(X)
    tt[0, 2] = 0
    best = 0
    j = 1
    while j < tt.shape[0] and best <= tt[j, 2]:
        tt[j, 2] = np.sum((cs[:, tt[j, 1]] - cs[:, tt[j, 0]])[get_rows(cs0, alpha, tt[j, 0], tt[j, 1])])
        if tt[j, 2] > best:
            best = tt[j, 2]
        j = j + 1
    return best, tt

def run_algorithm(X: np.ndarray, alpha: float, K: int) -> List[Any]:
    xsum = X.sum()
    cs0 = csX(X)

    best, tt = generate_initial_queue(cs0, X, alpha)

    res = np.zeros([K, 2]).astype("int")
    cur_r = 0

    for i in range(K):
        tt = sort_queue(tt)
        res[
            i,
        ] = tt[0, :2]
        cur_r = cur_r + np.sum(X[get_rows(cs0, alpha, res[i, 0], res[i, 1]), (res[i, 0]) : (res[i, 1])]) / xsum

        if cur_r >= 1:
            break
        if i < K:
            X[get_rows(cs0, alpha, tt[0, 0], tt[0, 1]), tt[0, 0] : (tt[0, 1])] = 0

            best, tt = update_tt(cs0, alpha, X, tt)

    if i < K:
        res = res[: i + 1, :]
    results = []
    for r in res:
        results.append(generate_final_return(cs0, alpha, r))
    return results

def createMatrixCandidate(X: np.ndarray, alpha: float, K: int = 30, debug: bool = False) -> Any:
    t1_start = process_time()
    out = run_algorithm(X, alpha, K)
    t1_end = process_time()
    if debug == True:
        print("[DEBUG] Generating local grouping candidates from one event label channel - time taken:", t1_end - t1_start)
    return out

def calculateTilePossession(tiles):
    tiles_elements_table = []
    for tile in tiles:
        tiles_elements_table.append(tile[2])
    tiles_elements_table = np.array(tiles_elements_table).astype(np.int32)
    return tiles_elements_table.T
