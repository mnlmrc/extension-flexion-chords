import argparse
import os
import warnings
from itertools import combinations

import pandas as pd

import globals as gl

import numpy as np


def within_subj_var(Y, partition_vec, cond_vec, subtract_mean=True):
    '''
        Estimate the within subject and noise variance for each subject.

        Args:
            data: 2D numpy array of shape (N-regressors by P-voxels)
            partition_vec: 1D numpy array of shape (N-regressors) with partition index
            cond_vec: 1D numpy array of shape (N-regressors) with condition index
            subtract_mean: Subtract the mean of voxels across conditions within a run.

        Returns:
            v_s:  1D array containing the subject variance.
            v_se: 1D array containing subject + noise variance.
    '''

    # In case partition_vec was not contiguous, e.g.,: [1, 1, 2, 2, 1, 1, 2, 2] instead of [1,1,1,1,2,2,2,2].
    # First, make partition indices contiguous by sorting the rows:
    sorted_indices = np.argsort(partition_vec)
    Y = Y[sorted_indices]
    partition_vec = partition_vec[sorted_indices]
    cond_vec = cond_vec[sorted_indices]

    cond = np.unique(cond_vec)
    partition = np.unique(partition_vec)

    v_s, v_se = None, None

    # subtract mean of voxels across conditions within each run:
    if subtract_mean:
        N, P = Y.shape

        # Reshape Y to separate each partition
        Y_reshaped = Y.reshape(partition.shape[0], cond.shape[0], P)

        # mean of voxels across conditions for each partition:
        partition_means = Y_reshaped.mean(axis=1, keepdims=True)

        # subtract the partition means from the original reshaped Y and rehsape back to original:
        Y = (Y_reshaped - partition_means).reshape(N, P)

        cov_Y = Y @ Y.T / Y.shape[1]

        # avg of the main diagonal:
        avg_main_diag = np.sum(np.diag(cov_Y)) / (len(cond) * len(partition))

        # avg of the main off-diagonal (i.e., within-run regressor covariance):
        mask = np.kron(np.eye(len(partition)), np.ones((len(cond), len(cond))))
        mask = mask - np.eye(mask.shape[0])
        avg_main_off_diag = np.sum(cov_Y * mask) / (np.sum(mask))

        # within partition variance:
        v_se = avg_main_diag

        # avg across session diagonals (i.e., covariance between regressors across runs):
        mask = np.kron(np.ones((len(partition), len(partition))), np.eye(len(cond)))
        mask = mask - np.eye(mask.shape[0])
        avg_across_diag = np.sum(cov_Y * mask) / (np.sum(mask))

        # avg across session off-diagonals:
        mask = np.kron(1 - np.eye(len(partition)), np.ones((len(cond), len(cond))))
        mask = mask - np.kron(np.ones((len(partition), len(partition))), np.eye(len(cond))) + np.eye(mask.shape[0])
        avg_across_off_diag = np.sum(cov_Y * mask) / (np.sum(mask))

        # across partition variance:
        v_s = avg_across_diag

    else:
        pass

    return v_s, v_se


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--day', type=int, default=None)
    parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--glm', type=int, default=None)

    args = parser.parse_args()

    if args.what == 'within_subj_var':

        for Hem in ['L', 'R']:
            for roi in gl.rois['ROI']:
                Y = np.load(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}', f'subj{args.sn}',
                                     f'ROI.{Hem}.{roi}.beta.npy'))
                res = np.load(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}', f'subj{args.sn}',
                                     f'ROI.{Hem}.{roi}.res.npy'))
                Y_prewhitened = Y / np.sqrt(res)
                reginfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}', f'subj{args.sn}',
                                     'reginfo.tsv'), sep='\t')
                partition_vec = reginfo.run
                cond_vec = reginfo.name
                v_s, v_se = within_subj_var(Y_prewhitened, partition_vec, cond_vec, subtract_mean=True)

                snr = v_s / v_se

                print(f'subj{args.sn}, Hem: {Hem}, roi: {roi}, glm: {args.glm}, snr: {snr:.2f}')

        pass

if __name__ == '__main__':
    main()
