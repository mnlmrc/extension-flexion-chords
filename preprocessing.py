import argparse
import os

import numpy as np

import globals as gl
import pandas as pd
from fetch import load_mov_block

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


def calc_init_dist(X, latency=.1, fsample=gl.fsample):
    index = int(latency * fsample)
    x = X[index]

    L = x[-1] - x[0]
    L_norm = L / np.linalg.norm(L)
    x_norm = x / np.linalg.norm(x)

    d = x_norm - L_norm

    return d


def preprocessing(experiment="efc2",
                  participant_id="subj100",
                  session="testing",
                  day="1"):

    path = os.path.join(gl.baseDir, experiment, session, f"day{day}")

    sn = int(''.join([c for c in participant_id if c.isdigit()]))

    dat = pd.read_csv(os.path.join(path, f"{experiment}_{sn}.dat"), sep="\t")

    # Init lists
    md, rt, explained, angle, jerk, sine, d, repetition = [], [], [], [], [], [], [], []

    for bl in range(gl.nblocks):

        block = '%02d' % int(bl + 1)
        print(f"experiment:{experiment}, "
              f"participant_id:{participant_id}, "
              f"session:{session}, "
              f"day:{day}, "
              f"block:{block}")

        filename = os.path.join(path, f'{experiment}_{sn}_{block}.mov')

        mov = load_mov_block(filename)
        dat_tmp = dat[dat.BN == bl + 1].reset_index()

        for tr in range(len(mov)):

            if tr == 0:
                rep = 1
            else:
                if dat_tmp.iloc[tr].chordID == dat_tmp.iloc[tr - 1].chordID:
                    rep += 1
                else:
                    rep = 1

            repetition.append(rep)

            if bool(dat_tmp.iloc[tr].trialPoint) is True:
                force = mov[tr][:, gl.diffCols]  # extract only differential forces

                c = np.any(np.abs(force) > gl.fthresh, axis=1)
                force = force[c][:int(-gl.hold_time * gl.fsample)]

                # compute rt and md
                rt.append(np.argmax(c) / gl.fsample)
                md.append(calc_md(force))

                # perform pca
                exp, loadings, _ = calc_pca(force)
                explained.append(exp)

                # compute angle between pc1 and ideal trajectory
                L = force[-1] - force[0]
                L_norm = L / np.linalg.norm(L)
                pc1 = loadings[0]
                pc1_norm = pc1 / np.linalg.norm(pc1)
                proj = np.dot(pc1_norm, L_norm)
                angle.append(np.arccos(proj))
                sine.append(np.sin(np.arccos(proj)))

                # compute squared jerk
                jerk.append(calc_jerk(force))

                # compute diff at start
                d.append(calc_init_dist(force))

            else:
                md.append(None)
                rt.append(None)
                angle.append(None)
                explained.append([None] * 5)
                d.append([None] * 5)
                sine.append(None)
                jerk.append(None)
                # repetition.append(None)

    df = dat[['BN', 'TN', 'subNum', 'chordID', 'trialPoint']].copy()
    df['MD'] = md
    df['RT'] = rt
    df['angle'] = angle
    df['sine'] = sine
    df['jerk'] = jerk
    df['repetition'] = repetition
    for i in range(5):
        df[f'PC{i}'] = np.array(explained)[:, i]
    for i in range(5):
        df[f'd{i}'] = np.array(d)[:, i]

    # df.to_csv(os.path.join(path, f"{experiment}_{sn}.csv"))

    return df


if __name__ == "__main__":

    experiment = 'efc2'
    participants = [
        'subj100',
        'subj101',
        'subj102',
        'subj103',
        'subj104',
        # 'subj105',
        'subj106',
        'subj107'
    ]
    sessions = ['testing',
                'training']
    days = ['1', '2', '3', '4', '5']

    df = pd.DataFrame()

    for participant_id in participants:
        for day in days:
            if day == '1' or day == '5':
                session = 'testing'
            else:
                session = 'training'

            df_tmp = preprocessing(experiment=experiment,
                                   participant_id=participant_id,
                                   session=session,
                                   day=day)
            df_tmp['session'] = session
            df_tmp['participant_id'] = participant_id
            df_tmp['day'] = day
            df_tmp.loc[df_tmp['chordID'].isin(gl.trained), 'chord'] = 'trained'
            df_tmp.loc[df_tmp['chordID'].isin(gl.untrained), 'chord'] = 'untrained'
            df = pd.concat([df, df_tmp])

    df.to_csv(os.path.join(gl.baseDir, experiment, 'results.csv'))
