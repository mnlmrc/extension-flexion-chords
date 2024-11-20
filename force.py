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


def calc_pca(X):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca = pca.fit(X_scaled)

    loadings = pca.components_
    explained = pca.explained_variance_ratio_
    principal_components = pca.fit_transform(X_scaled)

    return explained, loadings, principal_components


def calc_jerk(X, fsample=gl.fsample['force']):
    # Ensure the positions and times are numpy arrays
    pos = np.array(X)

    vel = np.gradient(pos, 1 / fsample, axis=0)
    acc = np.gradient(vel, 1 / fsample, axis=0)
    jerk = np.gradient(acc, 1 / fsample, axis=0)

    av_squared_jerk = np.linalg.norm(jerk).mean()

    return av_squared_jerk


def calc_dist(X, latency=.01, fsample=gl.fsample):
    index = int(latency * fsample)
    x = X[index]

    L = x[-1] - x[0]
    L_norm = L / np.linalg.norm(L)
    x_norm = x / np.linalg.norm(x)

    d = x_norm - L_norm

    return d


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


def calc_xcorr(X):
    tau = np.zeros((X.shape[1], X.shape[1]))
    xcorr = np.zeros((X.shape[1], X.shape[1], X.shape[0] * 2 - 1))
    lags = None
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            a = X[:, i]
            v = X[:, j]

            norm = np.std(a) * np.std(v) * len(a)

            xcorr_tmp = np.correlate(a - a.mean(), v - v.mean(),
                                     mode='full') / norm  # this is to map xcorr between -1 and 1
            xcorr[i, j] = xcorr_tmp

            lags = np.arange(-len(a) + 1, len(a)) / gl.fsample['force']
            tau[i, j] = lags[np.argmax(np.abs(xcorr_tmp))]

    return xcorr, tau, lags


def calc_finger_latency(X, chordID, fthresh=None):
    if fthresh is None:
        fthresh = gl.fthresh

    X = np.abs(X)
    latency = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        a = X[:, i]
        da = np.gradient(a, 1 / gl.fsample['force'])

        latency[i] = np.argmax((a > fthresh) & (da > 0)) / gl.fsample['force']

    for i, char in enumerate(str(chordID)):
        if char == '9':
            latency[i] = np.nan

    order = np.argsort(latency[~np.isnan(latency)])
    fingers = [f for f in gl.channels['force'] if not np.isnan(latency[gl.channels['force'].index(f)])]
    finger_pos = np.full(len(gl.channels['force']), np.nan)
    for i, idx in enumerate(order):
        finger_pos[gl.channels['force'].index(fingers[idx])] = i

    return latency, finger_pos


# def calc_sim_chord(X, chordID, chord_vec):
#
#     sim_chord = {str(ch): [] for ch in chordID}
#
#     for ch in chordID:
#         ch = str(ch)
#
#         sel_finger = np.zeros_like(chord_vec, dtype='bool')
#
#         for f_idx, f in enumerate(ch):
#             sel_finger += np.array([str(chv)[f_idx] == f
#                                     if f is not '9' else False
#                                     for chv in chord_vec])
#
#         sim_chord[ch] = (X[sel_finger] - X[sel_finger, :, 0][:, :, np.newaxis]).mean(axis=0)
#
#     return sim_chord

def calc_entry_times(X, chordID):
    X = np.abs(X)
    for k, char in enumerate(str(chordID)):
        if char == '9':
            X[:, k] = np.nan
    entry_time = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        a = X[:, i]

        inTarget = (a > 2).astype(int)
        inTarget_diff = np.diff(inTarget)
        entry_time_tmp = np.where(inTarget_diff == 1)[0]
        entry_time[i] = entry_time_tmp[-1] / gl.fsample['force'] if len(entry_time_tmp) > 0 else np.nan

    order = np.argsort(entry_time[~np.isnan(entry_time)])
    fingers = [f for f in gl.channels['force'] if not np.isnan(entry_time[gl.channels['force'].index(f)])]
    finger_pos = np.full(len(gl.channels['force']), np.nan)
    for i, idx in enumerate(order):
        finger_pos[gl.channels['force'].index(fingers[idx])] = i

    return entry_time, finger_pos


def get_segment(x, hold_time=gl.hold_time):
    c = np.any(np.abs(x) > gl.fthresh, axis=1)

    start_samp_exec = np.argmax(c)

    if hold_time is None:
        x_s = x[start_samp_exec:]
        starttime = start_samp_exec / gl.fsample['force']
        endtime = (len(x) / gl.fsample['force']) - starttime
    else:
        x_s = x[start_samp_exec:-int(hold_time * gl.fsample['force'])]
        starttime = start_samp_exec / gl.fsample['force']
        endtime = ((len(x) - int(hold_time * gl.fsample['force'])) / gl.fsample['force']) - starttime

    return x_s, starttime, endtime


def calc_metrics(force):
    md, _ = calc_md(force)
    exp, loadings, _ = calc_pca(force)

    L = force[-1] - force[0]
    L_norm = L / np.linalg.norm(L)
    pc1 = loadings[0]
    pc1_norm = pc1 / np.linalg.norm(pc1)
    proj = np.dot(pc1_norm, L_norm)
    angle = np.arccos(proj)
    sine = np.sin(angle)
    jerk = calc_jerk(force)
    # dist = calc_dist(force)

    metrics = {
        'MD': md,
        'PC': exp,
        'angle': angle,
        'sine': sine,
        'jerk': jerk,
        # 'dist': dist
    }

    return metrics


class Force:
    def __init__(self, experiment=None, participant_id=None, session=None, day=None):

        self.experiment = experiment
        self.participant_id = participant_id if isinstance(participant_id, str) else participant_id[0]
        self.sn = int(''.join([c for c in self.participant_id if c.isdigit()]))
        self.session = session
        self.day = day
        self.path = os.path.join(gl.baseDir, experiment, session, f"day{day}")
        self.dat = pd.read_csv(os.path.join(self.path, f"{experiment}_{self.sn}.dat"), sep="\t")

        self.pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
        self.trained = [int(x) for x in
                        self.pinfo[self.pinfo['participant_id'] == self.participant_id]['trained'].iloc[0].split('.')]
        self.untrained = [int(x) for x in
                          self.pinfo[self.pinfo['participant_id'] == self.participant_id]['untrained'].iloc[0].split(
                              '.')]

    def load_pkl(self):

        experiment = self.experiment
        session = self.session
        day = self.day
        sn = self.sn

        with open(os.path.join(gl.baseDir, experiment, session,
                               f'day{day}', f'{experiment}_{sn}_force.pkl'), "rb") as file:
            force_dict = pickle.load(file)

        return force_dict

    def crosscorrelation(self):

        force_dict = self.load_pkl()

        xcorr, tau, lags = [], [], []
        for force_trial in force_dict['force']:
            if force_trial is not None:
                xcorr_tmp, tau_tmp, lags_tmp = calc_xcorr(force_trial)
                xcorr.append(xcorr_tmp)
                lags.append(lags_tmp)
                tau.append(tau_tmp)
            else:
                xcorr.append(None)
                lags.append(None)
                tau.append(None)

        return xcorr, tau, lags

    # def preprocessing(self):
    #
    #     experiment = self.experiment
    #     participant_id = self.participant_id
    #     session = self.session
    #     day = self.day
    #     dat = self.dat
    #     path = self.path
    #     sn = self.sn
    #
    #     scaler = MinMaxScaler()
    #
    #     # init dict metrics
    #     metrics_dict = {
    #         'MD': [],
    #         'RT': [],
    #         'ET': [],
    #         'PC': [],
    #         'angle': [],
    #         'jerk': [],
    #         'sine': [],
    #         # 'dist': [],
    #         'repetition': [],
    #         'finger_asynch': []
    #     }
    #
    #     # init dict force
    #     force_dict = {
    #         'force': [],
    #         'force_filt10Hz': [],
    #         'force_filt20Hz': [],
    #         'experiment': [],
    #         'participant_id': [],
    #         'session': [],
    #         'day': [],
    #         'block': [],
    #         'chordID': [],
    #         'success': [],
    #     }
    #
    #     dforce_times = []
    #     dforce_order = []
    #     onset_times = []
    #     onset_order = []
    #     exit_times = []
    #     exit_order = []
    #     entry_times = []
    #     entry_order = []
    #
    #     nblocks = len(self.pinfo[self.pinfo['participant_id'] == participant_id]
    #                   [f'blocks Chords day{day}'].iloc[0].split('.'))
    #
    #     for bl in range(nblocks):
    #         block = f'{bl + 1:02d}'
    #         print(f"experiment:{experiment}, "
    #               f"participant_id:{participant_id}, "
    #               f"session:{session}, "
    #               f"day:{day}, "
    #               f"block:{block}")
    #         filename = os.path.join(path, f'{experiment}_{sn}_{block}.mov')
    #         mov = load_mov(filename)
    #
    #         dat_tmp = dat[dat.BN == bl + 1].reset_index()
    #         rep = 1
    #
    #         ntrial = len(mov)
    #
    #         for tr in range(ntrial):
    #             if tr == 0 or dat_tmp.iloc[tr].chordID != dat_tmp.iloc[tr - 1].chordID:
    #                 rep = 1
    #             else:
    #                 rep += 1
    #             metrics_dict['repetition'].append(rep)
    #
    #             if dat_tmp.iloc[tr].trialPoint == 1:
    #
    #                 forceRaw = mov[tr][:, gl.diffCols][
    #                                mov[tr][:, 0] == 3] * gl.fGain  # take only states 3 (i.e., WAIT_EXEC)
    #                 force, rt, et = get_segment(forceRaw)
    #
    #                 assert rt > 0, "negative reaction time"
    #                 assert et > 0, "negative execution time"
    #
    #                 forceRaw_filt10Hz = lowpass_butter(forceRaw.T, 10, gl.fsample['force']).T
    #                 forceRaw_filt20Hz = lowpass_butter(forceRaw.T, 20, gl.fsample['force']).T
    #
    #                 dforceRaw = np.gradient(forceRaw_filt10Hz, 1 / gl.fsample['force'], axis=0)
    #                 dforceRaw = scaler.fit_transform(np.abs(dforceRaw))
    #
    #                 metrics_tmp = calc_metrics(force)
    #
    #                 # time to derivative peak
    #                 dforce_times_tmp = np.argmax(dforceRaw, axis=0) / gl.fsample['force']
    #                 for i, char in enumerate(str(dat_tmp.iloc[tr].chordID)):
    #                     if char == '9':
    #                         dforce_times_tmp[i] = np.nan
    #
    #                 # order at derivative peak
    #                 order_tmp = np.argsort(dforce_times_tmp[~np.isnan(dforce_times_tmp)])
    #                 fingers = [f for f in gl.channels['force'] if
    #                            not np.isnan(dforce_times_tmp[gl.channels['force'].index(f)])]
    #                 dforce_order_tmp = np.full(len(gl.channels['force']), np.nan)
    #                 for i, idx in enumerate(order_tmp):
    #                     dforce_order_tmp[gl.channels['force'].index(fingers[idx])] = i
    #
    #                 # order at force onset
    #                 # onset_times_tmp, onset_order_tmp = calc_exit_times(dforceRaw, dat_tmp.iloc[tr].chordID, fthresh=.2)
    #
    #                 finger_asynch = np.nanmax(onset_times_tmp) - np.nanmin(onset_times_tmp)
    #
    #                 # exit_times_tmp, exit_order_tmp = calc_exit_times(forceRaw, dat_tmp.iloc[tr].chordID)
    #                 entry_times_tmp, entry_order_tmp = calc_entry_times(forceRaw, dat_tmp.iloc[tr].chordID)
    #
    #                 dforce_times.append(dforce_times_tmp)
    #                 dforce_order.append(dforce_order_tmp)
    #                 # onset_times.append(onset_times_tmp)
    #                 # onset_order.append(onset_order_tmp)
    #                 # exit_times.append(exit_times_tmp)
    #                 # exit_order.append(exit_order_tmp)
    #                 entry_times.append(entry_times_tmp)
    #                 entry_order.append(entry_order_tmp)
    #
    #                 metrics_dict['RT'].append(rt)
    #                 metrics_dict['ET'].append(et)
    #                 metrics_dict['MD'].append(metrics_tmp['MD'])
    #                 metrics_dict['PC'].append(metrics_tmp['PC'])
    #                 metrics_dict['angle'].append(metrics_tmp['angle'])
    #                 metrics_dict['sine'].append(metrics_tmp['sine'])
    #                 metrics_dict['jerk'].append(metrics_tmp['jerk'])
    #                 metrics_dict['finger_asynch'].append(finger_asynch)
    #
    #                 force_dict['force'].append(forceRaw)
    #                 force_dict['force_filt10Hz'].append(forceRaw_filt10Hz)
    #                 force_dict['force_filt20Hz'].append(forceRaw_filt20Hz)
    #                 force_dict['experiment'].append(experiment)
    #                 force_dict['participant_id'].append(participant_id)
    #                 force_dict['session'].append(session)
    #                 force_dict['day'].append(day)
    #                 force_dict['block'].append(bl + 1)
    #                 force_dict['chordID'].append(dat_tmp.iloc[tr].chordID)
    #                 force_dict['success'].append('successful')
    #
    #             else:
    #
    #                 metrics_dict['RT'].append(None)
    #                 metrics_dict['ET'].append(None)
    #                 metrics_dict['MD'].append(None)
    #                 metrics_dict['PC'].append(None)
    #                 metrics_dict['angle'].append(None)
    #                 metrics_dict['sine'].append(None)
    #                 metrics_dict['jerk'].append(None)
    #                 metrics_dict['finger_asynch'].append(None)
    #
    #                 dforce_order.append([None] * 5)
    #                 dforce_times.append([None] * 5)
    #                 onset_times.append([None] * 5)
    #                 onset_order.append([None] * 5)
    #                 exit_times.append([None] * 5)
    #                 exit_order.append([None] * 5)
    #                 entry_times.append([None] * 5)
    #                 entry_order.append([None] * 5)
    #
    #                 force_dict['force'].append(None)
    #                 force_dict['force_filt10Hz'].append(None)
    #                 force_dict['force_filt20Hz'].append(None)
    #                 force_dict['experiment'].append(experiment)
    #                 force_dict['participant_id'].append(participant_id)
    #                 force_dict['session'].append(session)
    #                 force_dict['day'].append(day)
    #                 force_dict['block'].append(bl + 1)
    #                 force_dict['chordID'].append(dat_tmp.iloc[tr].chordID)
    #                 force_dict['success'].append('unsuccessful')
    #
    #     metrics = dat[['BN', 'TN', 'subNum', 'chordID', 'trialPoint']].copy()
    #     for key, values in metrics_dict.items():
    #         metrics[key] = values
    #
    #     metrics['day'] = day
    #     metrics['session'] = session
    #     metrics['participant_id'] = participant_id
    #     metrics.loc[metrics['chordID'].isin(self.trained), 'chord'] = 'trained'
    #     metrics.loc[metrics['chordID'].isin(self.untrained), 'chord'] = 'untrained'
    #
    #     dforce_times = np.array(dforce_times)
    #     dforce_order = np.array(dforce_order)
    #     onset_times_tmp = np.array(onset_times)
    #     onset_order_tmp = np.array(onset_order)
    #     exit_times = np.array(exit_times)
    #     exit_order = np.array(exit_order)
    #     entry_times = np.array(entry_times)
    #     entry_order = np.array(entry_order)
    #
    #     metrics = pd.concat([metrics, pd.DataFrame(onset_times_tmp,
    #                                                columns=['thumb_onset',
    #                                                         'index_onset',
    #                                                         'middle_onset',
    #                                                         'ring_onset',
    #                                                         'pinkie_onset'])], axis=1)
    #     metrics = pd.concat([metrics, pd.DataFrame(onset_order_tmp, columns=['thumb_onset_order',
    #                                                                          'index_onset_order',
    #                                                                          'middle_onset_order',
    #                                                                          'ring_onset_order',
    #                                                                          'pinkie_onset_order'])], axis=1)
    #     metrics = pd.concat([metrics, pd.DataFrame(dforce_times,
    #                                                columns=['thumb_dforce',
    #                                                         'index_dforce',
    #                                                         'middle_dforce',
    #                                                         'ring_dforce',
    #                                                         'pinkie_dforce'])], axis=1)
    #     metrics = pd.concat([metrics, pd.DataFrame(dforce_order, columns=['thumb_dforce_order',
    #                                                                       'index_dforce_order',
    #                                                                       'middle_dforce_order',
    #                                                                       'ring_dforce_order',
    #                                                                       'pinkie_dforce_order'])], axis=1)
    #     metrics = pd.concat([metrics, pd.DataFrame(exit_times,
    #                                                columns=['thumb_exit',
    #                                                         'index_exit',
    #                                                         'middle_exit',
    #                                                         'ring_exit',
    #                                                         'pinkie_exit'])], axis=1)
    #     metrics = pd.concat([metrics, pd.DataFrame(exit_order, columns=['thumb_exit_order',
    #                                                                     'index_exit_order',
    #                                                                     'middle_exit_order',
    #                                                                     'ring_exit_order',
    #                                                                     'pinkie_exit_order'])], axis=1)
    #     metrics = pd.concat([metrics, pd.DataFrame(entry_times, columns=['thumb_entry',
    #                                                                      'index_entry',
    #                                                                      'middle_entry',
    #                                                                      'ring_entry',
    #                                                                      'pinkie_entry'])], axis=1)
    #     metrics = pd.concat([metrics, pd.DataFrame(entry_order, columns=['thumb_entry_order',
    #                                                                      'index_entry_order',
    #                                                                      'middle_entry_order',
    #                                                                      'ring_entry_order',
    #                                                                      'pinkie_entry_order'])], axis=1)
    #
    #     return metrics, force_dict
