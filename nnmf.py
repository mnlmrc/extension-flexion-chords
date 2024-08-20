import os

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as mse
# from deap import base, creator, tools, algorithms

import globals as gl


def get_emg_chords(experiment, participant_id):
    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    M = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.chordDir, 'natChord_chord.tsv'), sep='\t')
    M = M[M['sn'] == sn]
    chords = list(M['chordID'])
    M = M[[f'emg_hold_avg_e{e + 1}' for e in range(5)] +
          [f'emg_hold_avg_f{f + 1}' for f in range(5)]].to_numpy()

    return M, chords


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


def iterative_nnmf(X, thresh=0.1):
    """

    Args:
        X:
        k:
        thresh:
        max_iterations:
        repetitions:

    Returns:

    """

    W, H, r2 = None, None, None
    for k in range(X.shape[1]):
        W, H = calc_nnmf(X, k + 1)
        Xhat = np.dot(W, H)
        r2 = calc_r2(X, Xhat)
        err = mse(X, Xhat)

        # print(f"k:{k + 1}, R²: {r2:.4f}")

        if 1 - r2 < thresh:
            break

    return W, H, r2,


def calc_reconerr(W, Hp, M):
    return np.linalg.norm(M - np.dot(W, Hp))


# def optimize_H(W, M, M_chord, method='genetic'):
#     N = M_chord.shape[0]
#     k = W.shape[1]
#
#     Hp = None
#
#     if method == 'greedy':
#
#         selected_indices = np.random.choice(N, k, replace=False)
#         Hp = M_chord[selected_indices, :]
#         min_error = calc_reconerr(W, Hp, M)
#
#         improved = True
#         n = 0
#         while improved:
#             n = n + 1
#             print(f'greedy - iteration {n}')
#
#             improved = False
#             for i in range(k):
#                 for j in range(N):
#                     if j not in selected_indices:
#                         temp_indices = selected_indices.copy()
#                         temp_indices[i] = j
#                         temp_Hp = M_chord[temp_indices, :]
#                         error = calc_reconerr(W, temp_Hp, M)
#                         if error < min_error:
#                             min_error = error
#                             selected_indices = temp_indices
#                             Hp = temp_Hp
#                             improved = True
#
#     elif method == 'genetic':
#
#         creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#         creator.create("Individual", list, fitness=creator.FitnessMin)
#
#         toolbox = base.Toolbox()
#         toolbox.register("indices", np.random.choice, N, k, replace=False)
#         toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
#         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
#         def evaluate(individual):
#             indices = np.array(individual)
#             Hp = M_chord[indices, :]
#             return (calc_reconerr(W, Hp, M),)
#
#         toolbox.register("mate", tools.cxTwoPoint)
#         toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
#         toolbox.register("select", tools.selTournament, tournsize=3)
#         toolbox.register("evaluate", evaluate)
#
#         # Genetic Algorithm parameters
#         population = toolbox.population(n=5000)  # Population size
#         NGEN = 1000  # Number of generations
#         CXPB, MUTPB = 0.5, 0.2  # Crossover and mutation probabilities
#
#         # Run the genetic algorithm
#         for gen in range(NGEN):
#             print(f'genetic - generation {gen}')
#             # Apply crossover and mutation
#             offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
#
#             # Evaluate the fitness of the offspring
#             fits = toolbox.map(toolbox.evaluate, offspring)
#
#             for fit, ind in zip(fits, offspring):
#                 ind.fitness.values = fit
#
#             # Select the next generation population
#             population = toolbox.select(offspring, k=len(population))
#
#         # Get the best individual solution
#         best_ind = tools.selBest(population, 1)[0]
#         Hp = M_chord[np.array(best_ind), :]
#
#     return Hp


def assert_selected_rows_belong(M_chord, Hp):
    # N, num_columns = M_chord.shape
    # k, selected_columns = Hp.shape
    #
    # # assert selected_columns == num_columns, f"Selected rows must have the same number of columns as M_chord ({num_columns})."

    for row in Hp:
        if not any(np.array_equal(row, original_row) for original_row in M_chord):
            raise AssertionError("One of the selected rows does not belong to the original matrix.")

    print("Assertion passed: All selected rows belong to the original matrix.")

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
