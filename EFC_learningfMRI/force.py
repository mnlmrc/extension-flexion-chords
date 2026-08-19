import argparse
import EFC_learningfMRI.globals as gl
import os
import pandas as pd
import numpy as np
import warnings

from EFC_learningfMRI.util import lowpass_fir, get_trained_and_untrained


FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinkie']


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
                    #print(f'load_mov...reading trial: {trial}')
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



def _find_blocks(path, prefix, ext='.mov'):
    """Block numbers available on disk for a subject, sorted."""
    files = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith(ext)]
    return sorted(int(f.split('_')[-1].split('.')[0]) for f in files)


def _load_block_mov(filename):
    """Load a .mov block, keep only the execution phase samples."""
    mov = np.concatenate(load_mov(filename), axis=0)
    return mov[mov[:, 1] == gl.wait_exec]


def _trial_row(force_tmp=None, dat_row=None, prev_chordID=None, trained=None):
    """Compute all single-trial metrics for one trial, returned as one row dict."""
    fsample = gl.fsample['force']
    chordID = dat_row['chordID'][0]

    # force derivative features
    force_filt = lowpass_fir(force_tmp, n_ord=4, cutoff=10, fsample=fsample, axis=0)
    der_abs    = np.abs(np.gradient(force_filt, 1 / fsample, axis=0))
    der_peak   = der_abs.max(axis=0)
    der_t2peak = np.argmax(der_abs, axis=0) / fsample
    der_avg    = der_abs.mean(axis=0)

    # execution-time / reaction-time boundaries
    channels   = [i for i, c in enumerate(str(chordID)) if c in ('1', '2')]
    et_samples = find_sustained_threshold_crossing(np.abs(force_filt), channels, gl.ftarget, fsample=fsample)
    if et_samples > 0:
        rt_samples = int((dat_row['RT'][0] / 1000) * fsample)
        trialPoint = 1
    else:
        rt_samples = 0
        trialPoint = 0
        et_samples = len(force_tmp) - int(gl.hold_time * fsample)
    assert et_samples > rt_samples

    MD, _     = calc_md(force_tmp[rt_samples:et_samples])
    force_avg = force_tmp[-et_samples:].mean(axis=0)

    chord = 'trained' if str(chordID) in trained else 'untrained'
    Rep   = 2 if (prev_chordID is not None and prev_chordID == chordID) else 1

    row = {
        'subNum'      : dat_row['subNum'][0],
        'BN'          : dat_row['BN'][0],
        'Repetition'  : Rep,
        'TN'          : dat_row['TN'][0],
        'trialPoint'  : trialPoint,
        'RT'          : dat_row['RT'][0],
        'ET'          : et_samples / fsample,
        'MD'          : MD,
        'chordID'     : chordID,
        'chord'       : chord,
        'session'     : dat_row['day'][0],
        'session_type': dat_row['session'][0],
        'week'        : dat_row['week'][0],
    }
    for i, f in enumerate(FINGERS):
        row[f] = force_avg[i]                          # average signed force
        row[f'{f}_abs'] = np.abs(force_avg[i])         # average absolute force
        row[f'{f}_der'] = der_avg[i]                   # average force derivative
        row[f'{f}_der_peak'] = der_peak[i]             # peak force derivative
        row[f'{f}_der_t2peak'] = der_t2peak[i]         # time to peak force derivative
    return row


def single_trial_behaviour(sn=None, session=None):
    ch_idx = np.array(gl.diffCols)
    path   = os.path.join(gl.baseDir, 'behavioural', f'day{session}')

    dat     = pd.read_csv(os.path.join(path, f'efc4_{sn}.dat'), sep='\t')
    pinfo   = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    trained = get_trained_and_untrained(sn)

    assert dat.subNum.unique() == sn

    rows = []
    for bl in _find_blocks(path, prefix=f'efc4_{sn}'):
        dat_tmp = dat[dat['BN'] == bl]
        mov     = _load_block_mov(os.path.join(path, f'efc4_{sn}_{int(bl):02d}.mov'))
        TN      = np.unique(mov[:, 0])

        print(f'Processing... subj{sn}, session{session}, block {bl}, {TN.size} trials found...')
        assert TN.size == len(dat_tmp)

        prev_chordID = None
        for ntrial in TN:
            force_tmp = mov[mov[:, 0] == ntrial][:, ch_idx] * gl.fGain
            dat_row   = dat_tmp[dat_tmp.TN == ntrial].reset_index()

            row = _trial_row(force_tmp, dat_row, prev_chordID, trained)
            print(f"subj{sn}, session {session}, block {bl}, ntrial {ntrial + 1}, "
                  f"ET {row['ET']:.2f}s, trialPoint {row['trialPoint']}")
            rows.append(row)
            prev_chordID = row['chordID']

    single_trial_metrics = pd.DataFrame(rows)
    single_trial_metrics.to_csv(os.path.join(path, f'efc4_{sn}_single_trial.tsv'), sep='\t', index=False)


