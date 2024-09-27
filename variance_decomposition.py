import warnings

import numpy as np


def reliability_var(Y, subj_vec, part_vec, cond_vec=None, centered=False):
    """
    Description:
        Variance decomposition on dataset. The multi-channel data Y_ij follow the following format:
                Y_ij = g + s_i + e_ij
        where i is the subject number and j is the partition/session number.

        Assuming a) g, s_i, e_ij are mutually independent b) e_ij and s_i are i.i.d, we can estimate the term variances as follows:

        Across subjects:
        v_g = E[Y_ij, Y_kl]
        Within subject, Across run:
        v_g + v_s = E[Y_ij, Y_ik]
        Within observation/partition:
        v_g + v_s + v_e = E[Y_ij, Y_ij]

        To develop estimators for these quantities we replace the
        Expectation with the mean over all possible pairings.

    Args:
        Y: Data vector/matrix vertically concatenated for subjects and partitions.
        subj_vec: Column vector of subject numbers.
        part_vec: Column vector of partition numbers.
        cond_vec: Optional column vector of condition numbers. If provided, the reliability is calculated separately for different conditions.
        centered: If True, centers the data within observation Y_ij before variance decomposition.

    Returns:
        v_g: Estimated across subject variance (global effect).
        v_gs: Estimated (Within subject, Across run) variance.
        v_gse: Estimated (Within observation) variance.
    """

    if np.isnan(Y).any():
        warnings.warn('Input data contains nan elements')

    if cond_vec is None:
        cond_vec = np.ones_like(subj_vec)

    subjects = np.unique(subj_vec)
    partitions = np.unique(part_vec)
    conds = np.unique(cond_vec)

    v_gs = 0
    v_gse = 0
    if len(conds) > 1:
        v_gs = [0] * len(conds)
        v_gse = [0] * len(conds)

    subj_data = {}

    for k in range(len(conds)):
        for i in range(len(subjects)):
            A = []
            for j in range(len(partitions)):
                part_data = Y[(subj_vec == subjects[i]) &
                              (part_vec == partitions[j]) &
                              (cond_vec == conds[k]), :]

                if centered:
                    part_data = part_data - np.nanmean(part_data, axis=0)

                A.append(part_data.flatten())

            A = np.column_stack(A)
            # A = A[~np.isnan(A).any(axis=1)]
            subj_data[(i, k)] = A

            # B = A.T @ A
            B = np.nansum(A.T[:, :, None] * A[None, :, :], axis=1)
            N = A.shape[1]

            tmp_v_gse = np.trace(B) / (N * len(subjects))

            if N > 1:
                mean_cov = B * (1 - np.eye(N))
                mean_cov = np.nansum(mean_cov) / (N * (N - 1))
                tmp_v_gs = mean_cov / len(subjects)
            else:
                tmp_v_gs = 0

            if len(conds) > 1:
                v_gse[k] += tmp_v_gse
                v_gs[k] += tmp_v_gs
            else:
                v_gse += tmp_v_gse
                v_gs += tmp_v_gs

    v_g = 0
    if len(conds) > 1:
        v_g = [0] * len(conds)

    N = len(subjects)
    for k in range(len(conds)):
        for i in range(N - 1):
            for j in range(i + 1, N):
                B = subj_data[(i, k)].T @ subj_data[(j, k)]
                tmp_v_g = np.nansum(B) / (B.shape[0] ** 2 * (N * (N - 1) / 2))

                if len(conds) > 1:
                    v_g[k] += tmp_v_g
                else:
                    v_g += tmp_v_g

    return v_g, v_gs, v_gse, conds