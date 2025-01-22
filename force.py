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

from util import lowpass_butter


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--day', type=int, default=None)

    args = parser.parse_args()

    if args.what == 'single_trial':

        behavioural_dict = {
            'BN': [],
            'TN': [],
            'participant_id': [],
            'subNum': [],
            'chordID': [],
            'chord': [],
            'trialPoint': [],
            'repetition': [],
            'day': [],
            'MD': [],
            'RT': [],
            'ET': [],
            'thumb_force': [],
            'index_force': [],
            'middle_force': [],
            'ring_force': [],
            'pinkie_force': [],
        }

        path = os.path.join(gl.baseDir, args.experiment, gl.behavDir, f"day{args.day}", f'subj{args.sn}')

        pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')
        blocks = pinfo[pinfo.sn == args.sn].FuncRuns[0].split('.')

        dat = pd.read_csv(os.path.join(path, f"{args.experiment}_{args.sn}.dat"), sep="\t")

        for block in blocks:

            block = int(block)

            print(f"experiment:{args.experiment}, "
                  f"participant_id:subj{args.sn}, "
                  f"day:{args.day}, "
                  f"block:{block}")

            filename = os.path.join(path, f'{args.experiment}_{args.sn}_{block:02d}.mov')

            mov = load_mov(filename)

            dat_tmp = dat[dat.BN == block].reset_index()  # .dat file for block

            for tr in range(len(mov)):

                if tr == 0 or dat_tmp.iloc[tr].chordID != dat_tmp.iloc[tr - 1].chordID:
                    rep = 1
                else:
                    rep += 1

                chordID = dat_tmp.iloc[tr].chordID.astype(int).astype(str)
                chord = 'trained' if chordID in pinfo[pinfo['sn'] == args.sn]['trained'][0].split('.') else 'untrained'

                # add trial info to dictionary
                behavioural_dict['BN'].append(dat_tmp.iloc[tr].BN)
                behavioural_dict['TN'].append(dat_tmp.iloc[tr].TN)
                behavioural_dict['subNum'].append(args.sn)
                behavioural_dict['participant_id'].append(f'subj{args.sn}')
                behavioural_dict['chordID'].append(chordID)
                behavioural_dict['trialPoint'].append(dat_tmp.iloc[tr].trialPoint)
                behavioural_dict['chord'].append(chord)
                behavioural_dict['day'].append(args.day)
                behavioural_dict['repetition'].append(rep)

                if dat_tmp.iloc[tr].trialPoint == 1:

                    forceRaw = mov[tr][:, gl.diffCols][
                                   mov[tr][:, 1] == 4] * gl.fGain  # take only states 3 (i.e., WAIT_EXEC)

                    # calc single trial metrics
                    force, rt, et = calc_segment(forceRaw, hold_time=None, chordID=dat_tmp.iloc[tr].chordID)

                    force_avg = force.mean(axis=0)

                    assert rt > 0, "negative reaction time"
                    # if et is not None:
                    assert et > 0, "negative execution time"
                    # else:
                    #     pass

                    md, _ = calc_md(force)

                    # add measures to dictionary
                    behavioural_dict['RT'].append(rt)
                    behavioural_dict['ET'].append(et)
                    behavioural_dict['MD'].append(md)
                    behavioural_dict['thumb_force'].append(force_avg[0])
                    behavioural_dict['index_force'].append(force_avg[1])
                    behavioural_dict['middle_force'].append(force_avg[2])
                    behavioural_dict['ring_force'].append(force_avg[3])
                    behavioural_dict['pinkie_force'].append(force_avg[4])

                else:

                    # add to dictionary
                    behavioural_dict['RT'].append(None)
                    behavioural_dict['ET'].append(None)
                    behavioural_dict['MD'].append(None)
                    behavioural_dict['thumb_force'].append(None)
                    behavioural_dict['index_force'].append(None)
                    behavioural_dict['middle_force'].append(None)
                    behavioural_dict['ring_force'].append(None)
                    behavioural_dict['pinkie_force'].append(None)

        behavioural = pd.DataFrame(behavioural_dict)
        behavioural.to_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'day{args.day}', f'subj{args.sn}',
                                        'single_trial.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
