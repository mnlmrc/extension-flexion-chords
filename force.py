import pickle
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import globals as gl

import os
import pandas as pd

import warnings


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

    return MD


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
    md = calc_md(force)
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

        pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
        self.trained = [int(x) for x in
                        pinfo[pinfo['participant_id'] == self.participant_id]['trained'].iloc[0].split('.')]
        self.untrained = [int(x) for x in
                          pinfo[pinfo['participant_id'] == self.participant_id]['untrained'].iloc[0].split('.')]

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

    def preprocessing(self):

        experiment = self.experiment
        participant_id = self.participant_id
        session = self.session
        day = self.day
        dat = self.dat
        path = self.path
        sn = self.sn

        # init dict metrics
        metrics_dict = {
            'MD': [],
            'RT': [],
            'ET': [],
            'PC': [],
            'angle': [],
            'jerk': [],
            'sine': [],
            # 'dist': [],
            'repetition': []
        }

        # init dict force
        force_dict = {
            'force': [],
            'experiment': [],
            'participant_id': [],
            'session': [],
            'day': [],
            'block': [],
            'chordID': [],
            'success': [],
        }

        for bl in range(gl.nblocks):
            block = f'{bl + 1:02d}'
            print(f"experiment:{experiment}, "
                  f"participant_id:{participant_id}, "
                  f"session:{session}, "
                  f"day:{day}, "
                  f"block:{block}")
            filename = os.path.join(path, f'{experiment}_{sn}_{block}.mov')
            mov = load_mov(filename)

            dat_tmp = dat[dat.BN == bl + 1].reset_index()
            rep = 1

            ntrial = len(mov)

            for tr in range(ntrial):
                if tr == 0 or dat_tmp.iloc[tr].chordID != dat_tmp.iloc[tr - 1].chordID:
                    rep = 1
                else:
                    rep += 1
                metrics_dict['repetition'].append(rep)

                if dat_tmp.iloc[tr].trialPoint == 1:

                    force = mov[tr][:, gl.diffCols][mov[tr][:, 0] == 3]  # take only states 3 (i.e., WAIT_EXEC)
                    force, rt, endtime = get_segment(force)

                    et = endtime - rt

                    metrics_tmp = calc_metrics(force)

                    metrics_dict['RT'].append(rt)
                    metrics_dict['ET'].append(et)
                    metrics_dict['MD'].append(metrics_tmp['MD'])
                    metrics_dict['PC'].append(metrics_tmp['PC'])
                    metrics_dict['angle'].append(metrics_tmp['angle'])
                    metrics_dict['sine'].append(metrics_tmp['sine'])
                    metrics_dict['jerk'].append(metrics_tmp['jerk'])

                    force_dict['force'].append(force)
                    force_dict['experiment'].append(experiment)
                    force_dict['participant_id'].append(participant_id)
                    force_dict['session'].append(session)
                    force_dict['day'].append(day)
                    force_dict['block'].append(bl + 1)
                    force_dict['chordID'].append(dat_tmp.iloc[tr].chordID)
                    force_dict['success'].append('successful')

                else:

                    metrics_dict['RT'].append(None)
                    metrics_dict['ET'].append(None)
                    metrics_dict['MD'].append(None)
                    metrics_dict['PC'].append(None)
                    metrics_dict['angle'].append(None)
                    metrics_dict['sine'].append(None)
                    metrics_dict['jerk'].append(None)

                    force_dict['force'].append(None)
                    force_dict['experiment'].append(experiment)
                    force_dict['participant_id'].append(participant_id)
                    force_dict['session'].append(session)
                    force_dict['day'].append(day)
                    force_dict['block'].append(bl + 1)
                    force_dict['chordID'].append(dat_tmp.iloc[tr].chordID)
                    force_dict['success'].append('unsuccessful')

        metrics = dat[['BN', 'TN', 'subNum', 'chordID', 'trialPoint']].copy()
        for key, values in metrics_dict.items():
            metrics[key] = values

        metrics['day'] = day
        metrics['session'] = session
        metrics['participant_id'] = participant_id
        metrics.loc[metrics['chordID'].isin(self.trained), 'chord'] = 'trained'
        metrics.loc[metrics['chordID'].isin(self.untrained), 'chord'] = 'untrained'

        return metrics, force_dict
