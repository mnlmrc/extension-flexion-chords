import warnings

import numpy as np
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


def calc_jerk(X, fsample=gl.fsample):
    # Ensure the positions and times are numpy arrays
    pos = np.array(X)

    vel = np.gradient(pos, 1 / fsample, axis=0)
    acc = np.gradient(vel, 1 / fsample, axis=0)
    jerk = np.gradient(acc, 1 / fsample, axis=0)

    av_squared_jerk = np.linalg.norm(jerk).mean()

    return av_squared_jerk


def calc_dist(X, latency=.05, fsample=gl.fsample):
    index = int(latency * fsample)
    x = X[:index].mean(axis=0)

    L = x[-1] - x[0]
    L_norm = L / np.linalg.norm(L)
    x_norm = x / np.linalg.norm(x)

    d = x_norm - L_norm

    return d


# # Example usage
# n = 30
# g = np.random.normal(0, 2, n)
# s1 = np.random.normal(0, 1, n)
# s2 = np.random.normal(0, 1, n)
#
# x1 = np.concatenate([g + s1 + np.random.normal(0, 0.3, n), g + s1 + np.random.normal(0, 0.3, n)])
# x2 = np.concatenate([g + s2 + np.random.normal(0, 0.3, n), g + s2 + np.random.normal(0, 0.3, n)])
#
# Y = np.concatenate([x1, x2])
# subj_vec = np.kron([1, 2], np.ones(2 * n))
# part_vec = np.kron([1, 2, 1, 2], np.ones(n))
#
# v_g, v_gs, v_gse = reliability_var(Y, subj_vec, part_vec)
#
# # Theoretical var decomp
# theoretical_var_g = 4 / (4 + 1 + 0.09)
# theoretical_var_s = 1 / (4 + 1 + 0.09)
# theoretical_var_e = 0.09 / (4 + 1 + 0.09)
# print(
#     f'Theoretical:\nvar_g = {theoretical_var_g:.4f}, var_s = {theoretical_var_s:.4f}, var_e = {theoretical_var_e:.4f}')
#
# # Estimated
# estimated_var_g = v_g / v_gse
# estimated_var_s = (v_gs - v_g) / v_gse
# estimated_var_e = (v_gse - v_gs) / v_gse
# print(f'Estimated:\nvar_g = {estimated_var_g:.4f}, var_s = {estimated_var_s:.4f}, var_e = {estimated_var_e:.4f}')


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


def calc_metrics(mov, tr):
    force = mov[tr][:, gl.diffCols]
    c = np.any(np.abs(force) > gl.fthresh, axis=1)

    # extrapolate force from RT to start of hold time
    force = force[c][:int(-gl.hold_time * gl.fsample)]

    rt = np.argmax(c) / gl.fsample
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
    dist = calc_dist(force)

    metrics = {
        'RT': rt,
        'MD': md,
        'PC': exp,
        'angle': angle,
        'sine': sine,
        'jerk': jerk,
        'dist': dist
    }

    return metrics


class Force:
    def __init__(self, experiment, participant_id, session, day):

        self.experiment = experiment
        self.participant_id = participant_id
        self.sn = int(''.join([c for c in participant_id if c.isdigit()]))
        self.session = session
        self.day = day
        self.path = os.path.join(gl.baseDir, experiment, session, f"day{day}")
        self.dat = pd.read_csv(os.path.join(self.path, f"{experiment}_{self.sn}.dat"), sep="\t")

        pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
        self.trained = [int(x) for x in pinfo[pinfo['participant_id'] == participant_id]['trained'].iloc[0].split('.')]
        self.untrained = [int(x) for x in
                          pinfo[pinfo['participant_id'] == participant_id]['untrained'].iloc[0].split('.')]

    def preprocessing(self):

        experiment = self.experiment
        participant_id = self.participant_id
        session = self.session
        day = self.day
        dat = self.dat
        path = self.path
        sn = self.sn

        # init dict
        results = {
            'MD': [], 'RT': [], 'PC': [], 'angle': [],
            'jerk': [], 'sine': [], 'dist': [], 'repetition': []
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
                results['repetition'].append(rep)

                if dat_tmp.iloc[tr].trialPoint:

                    metrics = calc_metrics(mov, tr)

                    results['RT'].append(metrics['RT'])
                    results['MD'].append(metrics['MD'])
                    results['PC'].append(metrics['PC'])
                    results['angle'].append(metrics['angle'])
                    results['sine'].append(metrics['sine'])
                    results['jerk'].append(metrics['jerk'])
                    results['dist'].append(metrics['dist'])

                else:

                    results['RT'].append(None)
                    results['MD'].append(None)
                    results['PC'].append(None)
                    results['angle'].append(None)
                    results['sine'].append(None)
                    results['jerk'].append(None)
                    results['dist'].append(None)

        df = dat[['BN', 'TN', 'subNum', 'chordID', 'trialPoint']].copy()
        for key, values in results.items():
            if key in ['PC', 'dist']:
                for i in range(5):
                    df[f'{key}{i}'] = [val[i] if val is not None else None for val in values]
            else:
                df[key] = values

        df['session'] = session
        df['participant_id'] = participant_id
        df['day'] = day
        df.loc[df['chordID'].isin(self.trained), 'chord'] = 'trained'
        df.loc[df['chordID'].isin(self.untrained), 'chord'] = 'untrained'
        df = pd.concat([df, df])

        return df
