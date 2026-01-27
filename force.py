import argparse
import pickle
import warnings

import numpy as np
# from matplotlib import pyplot as plt
# from scipy.signal import correlate
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

import globals as gl
import re
import os
import pandas as pd

import warnings

from util import lowpass_butter, lowpass_fir


def calc_md(X):
    N, m = X.shape
    F1 = X[0]
    FN = X[-1] - F1  # Shift the end point

    shifted_matrix = X - F1  # Shift all points

    d = list()

    for t in range(1, N - 1):
        Ft = shifted_matrix[t]

        # Project Ft onto the ideal straight line
        proj = np.dot(Ft, FN) / np.dot(FN, FN) * FN

        # Calculate the Euclidean distance
        d.append(np.linalg.norm(Ft - proj))

    d = np.array(d)
    MD = d.mean()

    return MD, d


def load_mov(filename):
    """
    load .mov file of one block

    :return:
    """

    try:
        with open(filename, 'rt') as fid:
            trial = 0
            A = []
            for line in fid:
                if line.startswith('Trial'):
                    print(f'load_mov...reading trial: {trial}')
                    trial_number = int(line.split(' ')[1])
                    trial += 1
                    if trial_number != trial:
                        warnings.warn('Trials out of sequence')
                        trial = trial_number
                    A.append([])
                else:
                    # Convert line to a numpy array of floats and append to the last trial's list
                    data = np.fromstring(line, sep=' ')
                    if A:
                        A[-1].append(data)
                    else:
                        # This handles the case where a data line appears before any 'Trial' line
                        warnings.warn('Data without trial heading detected')
                        A.append([data])

            # Convert all sublists to numpy arrays
            mov = [np.array(trial_data) for trial_data in A]
            # # vizForce = [np.array(trial_data)[:, 9:] for trial_data in A]
            # state = [np.array(trial_data) for trial_data in A]

    except IOError as e:
        raise IOError(f"Could not open {filename}") from e

    return mov


import numpy as np

def find_sustained_threshold_crossing(X, channels, threshold, fsample, duration_ms=600):
    """
    Find the first timepoint at which all selected channels exceed a threshold
    and remain above it for a given duration.

    Parameters:
    - X: np.ndarray of shape (timepoints, channels)
    - channels: list of int, indices of selected channels
    - threshold: float, the threshold to exceed
    - fsample: float, the sampling frequency in Hz
    - duration_ms: float, duration in milliseconds the signal must stay above threshold

    Returns:
    - int: index of the first timepoint meeting the condition, or -1 if not found
    """
    n_samples = int((duration_ms / 1000) * fsample)
    above_thresh = X[:, channels] > threshold

    # Create a combined signal: True only if all selected channels are above threshold
    all_channels_above = np.all(above_thresh, axis=1).astype(int)

    # Convolve with a flat window of size n_samples
    sustained = np.convolve(all_channels_above, np.ones(n_samples, dtype=int), mode='valid')

    # Find first timepoint where all values in the window are True (i.e. sum == n_samples)
    idx = np.where(sustained == n_samples)[0]

    if len(idx) > 0:
        return int(idx[0])
    else:
        return -1


def calc_single_trial_metrics(experiment=None, sn=None, session=None, day=None,experiment_code=None):
    experiment_code = experiment_code if experiment_code is not None else experiment_code
    ch_idx = np.array(gl.diffCols[experiment])

    dat = pd.read_csv(os.path.join(gl.baseDir, experiment, session, f'day{day}', f'{experiment_code}_{sn}.dat'),
                      sep='\t')

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
    trained = pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.')

    single_trial_metrics = {
        'subNum': [],
        'BN': [],
        'Repetition': [],
        'TN': [],
        'thumb': [],
        'index': [],
        'middle': [],
        'ring': [],
        'pinkie': [],
        'thumb_abs': [],
        'index_abs': [],
        'middle_abs': [],
        'ring_abs': [],
        'pinkie_abs': [],
        'thumb_der': [],
        'index_der': [],
        'middle_der': [],
        'ring_der': [],
        'pinkie_der': [],
        'trialPoint': [],
        'RT': [],
        'ET': [],
        'MD': [],
        # 'MD_c++': [],
        'chordID': [],
        'chord': [],
        'day': [],
        'session': [],
        'week': []

    }

    prefix = f'{experiment_code}_{sn}'
    ext = '.mov'
    path = os.path.join(gl.baseDir, experiment, session, f'day{day}')
    block_files = [f for f in os.listdir(path)
         if f.startswith(prefix) and f.endswith(ext)]

    blocks = [
        int(f.split("_")[-1].split('.')[0]) for f in block_files
    ]

    for bl in blocks:

        dat_tmp = dat[dat['BN'] == int(bl)]

        filename = os.path.join(gl.baseDir, experiment, session, f'day{day}',f'{prefix}_{int(bl):02d}.mov')

        mov = load_mov(filename)
        mov = np.concatenate(mov, axis=0)

        mov = mov[mov[:, 1] == gl.wait_exec[experiment]]

        #ntrials = int(mov[:, 0].max())
        TN = np.unique(mov[:, 0])

        print(f'Processing... subj{sn}, day{day}, block {bl}, {TN.size} trials found...')

        assert (TN.size == len(dat_tmp))

        for ntrial in TN: #range(ntrials):

            force_tmp = mov[mov[:, 0] == ntrial][:, ch_idx] * gl.fGain
            dat_row = dat_tmp[dat_tmp.TN == ntrial].reset_index()
            force_filt = lowpass_fir(force_tmp, n_ord=4, cutoff=10, fsample=gl.fsample['force'], axis=0)
            force_der1 = np.gradient(force_filt, 1 / gl.fsample['force'], axis=0)

            force_der1_avg = np.abs(force_der1.mean(axis=0))

            chordID = dat_row['chordID'][0]
            channels = [i for i, c in enumerate(str(chordID)) if c in ('1', '2')]

            et_samples = find_sustained_threshold_crossing(np.abs(force_filt), channels, gl.ftarget, fsample=gl.fsample['force'])

            if et_samples > 0: #dat_tmp.iloc[ntrial].trialPoint == 1:
                rt = dat_row['RT'][0] / 1000
                # et = dat_tmp.ilooc[ntrial]['ET'] / 1000
                rt_samples = int(rt * gl.fsample['force'])
                trialPoint = 1

                # et_samples = int( et * gl.fsample['force'])
            else:
                rt_samples = 0
                trialPoint = 0
                et_samples = len(force_tmp) - int(gl.hold_time * gl.fsample['force'])

            print(f'subj{sn}, day{day}, block {bl}, ntrial {ntrial + 1}, et {et_samples}, rt {rt_samples}, trialPoint {trialPoint}')

            assert et_samples > rt_samples

            MD, _ = calc_md(force_tmp[rt_samples:et_samples])
            force_avg = force_tmp[-et_samples:].mean(axis=0)

            if dat_row['chordID'][0].astype(str) in trained:
                chord = 'trained'
            else:
                chord = 'untrained'

            if ntrial>TN.min():
                prevChordID = dat_tmp[dat_tmp.TN == ntrial - 1].reset_index()['chordID'][0]
                Rep = 2 if prevChordID == chordID else 1
            else:
                Rep = 1

            single_trial_metrics['subNum'].append(dat_row['subNum'][0])
            single_trial_metrics['chordID'].append(chordID)
            single_trial_metrics['chord'].append(chord)
            single_trial_metrics['thumb'].append(force_avg[0])
            single_trial_metrics['index'].append(force_avg[1])
            single_trial_metrics['middle'].append(force_avg[2])
            single_trial_metrics['ring'].append(force_avg[3])
            single_trial_metrics['pinkie'].append(force_avg[4])
            single_trial_metrics['thumb_abs'].append(np.abs(force_avg[0]))
            single_trial_metrics['index_abs'].append(np.abs(force_avg[1]))
            single_trial_metrics['middle_abs'].append(np.abs(force_avg[2]))
            single_trial_metrics['ring_abs'].append(np.abs(force_avg[3]))
            single_trial_metrics['pinkie_abs'].append(np.abs(force_avg[4]))
            single_trial_metrics['thumb_der'].append(force_der1_avg[0])
            single_trial_metrics['index_der'].append(force_der1_avg[1])
            single_trial_metrics['middle_der'].append(force_der1_avg[2])
            single_trial_metrics['ring_der'].append(force_der1_avg[3])
            single_trial_metrics['pinkie_der'].append(force_der1_avg[4])
            single_trial_metrics['day'].append(dat_row['day'][0])
            single_trial_metrics['week'].append(dat_row['week'][0])
            single_trial_metrics['session'].append(dat_row['session'][0])
            single_trial_metrics['RT'].append(dat_row['RT'][0])
            single_trial_metrics['ET'].append(et_samples / gl.fsample['force'])
            single_trial_metrics['MD'].append(MD)
            single_trial_metrics['Repetition'].append(Rep)
            single_trial_metrics['BN'].append(dat_row['BN'][0])
            single_trial_metrics['TN'].append(dat_row['TN'][0])
            single_trial_metrics['trialPoint'].append(trialPoint)

    single_trial_metrics = pd.DataFrame(single_trial_metrics)

    return single_trial_metrics


def main(args):
    if args.what == 'single_trial':
        single_trial_metrics = calc_single_trial_metrics(args.experiment, args.sn, args.session, args.day,
                                                         args.experiment_code)
        single_trial_metrics.to_csv(os.path.join(gl.baseDir, args.experiment, args.session, f'day{args.day}',
                                        f'{args.experiment_code}_{args.sn}_single_trial.tsv'), sep='\t', index=False)

    if args.what == 'single_trial_days':
        for day in args.days:
            args = argparse.Namespace(
                            what='single_trial',
                            experiment=args.experiment,
                            experiment_code=args.experiment_code,
                             session=args.session,
                             day=day,
                             blocks=args.blocks,
                             sn=args.sn,)
            main(args)
    if args.what == 'single_trial_days_all':
        for sn in args.sns:
            args = argparse.Namespace(
                what='single_trial_days',
                experiment=args.experiment,
                experiment_code=args.experiment_code,
                sn=sn,
                session=args.session,
                days=args.days,
                blocks = args.blocks,
            )
            main(args)

    if args.what == 'pool_behaviour':
        data_pooled = pd.DataFrame()
        for sn in args.sns:
            for day in range(24):
                data = pd.read_csv(os.path.join(gl.baseDir, args.experiment, args.session, f'day{day+1}',
                             f'{args.experiment_code}_{sn}_single_trial.tsv'), sep = '\t',)
                data_pooled = pd.concat([data_pooled, data])
        data_pooled.to_csv(os.path.join(gl.baseDir, args.experiment, args.session, f'single_trial_behaviour.tsv',),
                           sep = '\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='EFC_learningfMRI')
    parser.add_argument('--experiment_code', type=str, default='efc4')
    parser.add_argument('--session', type=str, default='behavioural')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[101, 102, 103, 104, 105, 106, 107])
    parser.add_argument('--day', type=int, default=None)
    parser.add_argument('--days', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    parser.add_argument('--blocks', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    args = parser.parse_args()

    main(args)
