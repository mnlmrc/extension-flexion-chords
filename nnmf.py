import os

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as mse

import globals as gl


def calc_nnmf(X, k):
    """
    Perform Non-Negative Matrix Factorization (NNMF) on the given matrix.

    Parameters:
    matrix (numpy.ndarray): The input matrix to be decomposed.
    n_components (int): The number of components to use for NNMF.

    Returns:
    tuple: A tuple containing the matrices W and H from NNMF decomposition, and the reconstructed matrix.
    """
    model = NMF(n_components=k, init='random', random_state=0, max_iter=1000, tol=0.001)
    W = model.fit_transform(X)
    H = model.components_
    return W, H


def calc_r2(X, Xhat):
    """
    Calculate the R² (fraction of variance accounted for) of the reconstructed matrix.

    Parameters:
    matrix (numpy.ndarray): The original matrix.
    reconstructed_matrix (numpy.ndarray): The reconstructed matrix obtained from NNMF.

    Returns:
    float: The R² value.
    """
    ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2)
    ss_res = np.sum((X - Xhat) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.astype(float)


def iterative_nnmf(X, thresh=0.05):
    """

    Args:
        X:
        k:
        thresh:
        max_iterations:
        repetitions:

    Returns:

    """

    err, W, H, r2, k = None, None, None, None, None
    for k in range(X.shape[1]):
        W, H = calc_nnmf(X, k + 1)
        Xhat = np.dot(W, H)
        r2 = calc_r2(X, Xhat)
        err = mse(X, Xhat)

        print(f"k:{k + 1}, R²: {r2:.4f}")

        if 1 - r2 < thresh:
            break

    return W, H, r2, err, k + 1


def calc_reconerr(W, Hp, M):
    return np.linalg.norm(M - np.dot(W, Hp))



# def fit_k_chords(df, k, W, M):
#
#     df_sel = df.sample(n=k)
#
#     H_chord = df_sel[[f'emg_hold_avg_e{e+1}' for e in range(5)] + [f'emg_hold_avg_f{f+1}' for f in range(5)]].to_numpy()
#     chords = list(df_sel['chordID'])
#
#     Mhat = np.dot(W, H_chord)
#
#     err = mse(Mhat, M)
#
#     return chords, err
#
#
# def iterative_fit(M, df, max_iterations=1000):
#
#     W, _, r2, err0, k = iterative_nnmf(M, thresh=0.1)
#
#     chords = list()
#     reconerr = list()
#     for i in range(max_iterations):
#         ch, err = fit_k_chords(df, k, W, M)
#
#         chords.append(ch)
#         reconerr.append(err)
#
#     return r2, err0, chords, reconerr
