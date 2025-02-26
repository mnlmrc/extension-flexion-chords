import argparse
import pickle
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import globals as gl

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


def calc_segment(x, hold_time=gl.hold_time, chordID=None):
    c = np.any(np.abs(x) > gl.fthresh, axis=1)

    start_samp_exec = np.argmax(c)

    if hold_time is None:
        # get reaction time in seconds
        starttime = start_samp_exec / gl.fsample['force']

        # get first sample all fingers are in target
        if type(starttime) != str:
            chordID = str(int(chordID))
        idxs = [i for i, char in enumerate(chordID) if char != '9']
        d = np.all(np.abs(x[:, idxs]) > gl.ftarget, axis=1)
        end_samp_exec = np.argmax(d) if np.any(d) else None

        # get execution time in seconds
        endtime = end_samp_exec / gl.fsample['force'] if end_samp_exec is not None else None

        # get segment for MD calculation
        x_s = x[start_samp_exec:end_samp_exec]
    else:
        x_s = x[start_samp_exec:-int(hold_time * gl.fsample['force'])]
        starttime = start_samp_exec / gl.fsample['force']
        endtime = ((len(x) - int(hold_time * gl.fsample['force'])) / gl.fsample['force']) - starttime

    return x_s, starttime, endtime


def calc_single_trial_metrics(experiment=None, sn=None, session=None, day=None, blocks=None):
    ch_idx = np.array(gl.diffCols[experiment])

    dat = pd.read_csv(os.path.join(gl.baseDir, experiment, session, f'day{day}', f'subj{sn}',f'{experiment}_{sn}.dat'),
                      sep='\t')

    single_trial_metrics = {
        'subNum': [],
        'BN': [],
        'TN': [],
        'thumb': [],
        'index': [],
        'middle': [],
        'ring': [],
        'pinkie': [],
        'thumb_der': [],
        'index_der': [],
        'middle_der': [],
        'ring_der': [],
        'pinkie_der': [],
        'trialPoint': [],
        'RT': [],
        'ET': [],
        'MD': [],
        'MD_c++': [],
        'chordID': []

    }
    for bl in blocks:

        dat_tmp = dat[dat['BN'] == int(bl)]

        filename = os.path.join(gl.baseDir, experiment, session, f'day{day}', f'subj{sn}', f'{experiment}_{sn}_{int(bl):02d}.mov')

        mov = load_mov(filename)
        mov = np.concatenate(mov, axis=0)

        mov = mov[mov[:, 1] == gl.wait_exec[experiment]]

        ntrials = int(mov[:, 0].max())

        print(f'Processing... subj{sn}, block {bl}, {ntrials} trials found...')

        assert (mov[:, 0].max() == len(dat_tmp))

        for ntrial in range(ntrials):

            force_tmp = mov[mov[:, 0] == ntrial + 1][:, ch_idx] * gl.fGain

            force_filt = lowpass_fir(force_tmp, n_ord=4, cutoff=10, fsample=gl.fsample['force'], axis=0)
            force_der1 = np.gradient(force_filt, 1 / gl.fsample['force'], axis=0)

            force_der1_avg = np.abs(force_der1.mean(axis=0))

            if dat_tmp.iloc[ntrial].trialPoint == 1:
                rt = dat_tmp.iloc[ntrial]['RT'] / 1000
                et = dat_tmp.iloc[ntrial]['ET'] / 1000
                rt_samples = int(rt * gl.fsample['force'])
                et_samples = rt_samples + int( et * gl.fsample['force'])
            else:
                rt_samples = 0
                et_samples = len(force_tmp) - int(gl.hold_time * gl.fsample['force'])
            
            MD, _ = calc_md(force_tmp[rt_samples:et_samples])
            force_avg = force_tmp[-et_samples:].mean(axis=0)

            single_trial_metrics['subNum'].append(dat_tmp.iloc[ntrial]['subNum'])
            single_trial_metrics['chordID'].append(dat_tmp.iloc[ntrial]['chordID'])
            single_trial_metrics['thumb'].append(force_avg[0])
            single_trial_metrics['index'].append(force_avg[1])
            single_trial_metrics['middle'].append(force_avg[2])
            single_trial_metrics['ring'].append(force_avg[3])
            single_trial_metrics['pinkie'].append(force_avg[4])
            single_trial_metrics['thumb_der'].append(force_der1_avg[0])
            single_trial_metrics['index_der'].append(force_der1_avg[1])
            single_trial_metrics['middle_der'].append(force_der1_avg[2])
            single_trial_metrics['ring_der'].append(force_der1_avg[3])
            single_trial_metrics['pinkie_der'].append(force_der1_avg[4])
            single_trial_metrics['RT'].append(dat_tmp.iloc[ntrial]['RT'])
            single_trial_metrics['ET'].append(dat_tmp.iloc[ntrial]['ET'])
            single_trial_metrics['MD'].append(MD)
            single_trial_metrics['MD_c++'].append(dat_tmp.iloc[ntrial]['MD'])
            single_trial_metrics['BN'].append(dat_tmp.iloc[ntrial]['BN'])
            single_trial_metrics['TN'].append(dat_tmp.iloc[ntrial]['TN'])
            single_trial_metrics['trialPoint'].append(dat_tmp.iloc[ntrial]['trialPoint'])

    single_trial_metrics = pd.DataFrame(single_trial_metrics)

    return single_trial_metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--session', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--day', type=int, default=None)
    parser.add_argument('--blocks', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    args = parser.parse_args()

    if args.what == 'single_trial':

        single_trial_metrics = calc_single_trial_metrics(args.experiment, args.sn, args.session, args.day, args.blocks)
        single_trial_metrics.to_csv(os.path.join(gl.baseDir, args.experiment, args.session, f'day{args.day}', f'subj{args.sn}',
                                        'single_trial.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
