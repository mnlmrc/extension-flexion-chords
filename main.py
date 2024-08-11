import argparse
import json
import os

import scipy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import pickle
import json
import globals as gl
import argcomplete
import numpy as np

import seaborn as sns

from force import Force
from nnmf import iterative_nnmf, optimize_H, calc_reconerr, assert_selected_rows_belong, calc_r2
from plot import plot_days
from stats import perm_test_1samp
from util import load_nat_emg, calc_avg, calc_success
from variance_decomposition import reliability_var


def main(what, experiment=None, participant_id=None, session=None, day=None,
         chordID=None, chord=None, ntrial=None, metric=None, fig=None, axs=None):
    if participant_id is None:
        participant_id = gl.participants[experiment]

    if len(participant_id) == 1:
        participant_id = participant_id[0]

    match what:
        # region FORCE:preprocessing
        case 'FORCE:preprocessing':

            metrics = pd.DataFrame()
            for p in participant_id:
                for day in gl.days:
                    if day == '1' or day == '5':
                        session = 'testing'
                    else:
                        session = 'training'

                    force = Force(experiment,
                                  p,
                                  session,
                                  day)
                    metrics_tmp, force_dict = force.preprocessing()
                    metrics = pd.concat([metrics, metrics_tmp])

                    # save force to binary
                    with open(os.path.join(gl.baseDir, experiment, session,
                                           f'day{day}', f'{experiment}_{force.sn}_force.pkl'), "wb") as file:
                        pickle.dump(force_dict, file)

            print('saving...')
            metrics.to_csv(os.path.join(gl.baseDir, experiment, 'metrics.csv'))

            return metrics
        # endregion

        # region FORCE:xcorr
        case 'FORCE:xcorr':

            tau_dict = {
                'tau': [],
                'experiment': [],
                'participant_id': [],
                'session': [],
                'day': [],
                'chordID': [],
                'chord': []
            }

            for p in participant_id:
                for day in gl.days:

                    print(f'xcorr - participant_id: {p}, day: {day}')

                    if day == '1' or day == '5':
                        session = 'testing'
                    else:
                        session = 'training'

                    force = Force(experiment, p, session, day)
                    force_dict = force.load_pkl()
                    _, tau, _ = force.crosscorrelation()
                    for T, t in enumerate(tau):
                        tau_dict['tau'].append(t)
                        tau_dict['experiment'].append(experiment)
                        tau_dict['participant_id'].append(p)
                        tau_dict['session'].append(session)
                        tau_dict['day'].append(day)
                        tau_dict['chordID'].append(force_dict['chordID'][T])
                        tau_dict['chord'].append('trained' if force_dict['chordID'][T]
                                                              in force.trained else 'untrained')

                    pass

            with open(os.path.join(gl.baseDir, experiment, f'tau.pkl'), "wb") as file:
                pickle.dump(tau_dict, file)

            return tau_dict

        #endregion

        # region EMG:nnmf
        case 'EMG:nnmf':

            M = pd.read_csv(os.path.join(gl.baseDir, experiment, dataset, 'M.tsv'), sep='\t')[
                gl.channels['emg'] + ['sn']]

            W, H, r2, k = [], [], [], []
            for p in participant_id:
                sn = int(''.join([c for c in p if c.isdigit()]))
                M_tmp = M[M['sn'] == sn].to_numpy()
                W_tmp, H_tmp, r2_tmp = iterative_nnmf(M_tmp, thresh=0.1)

                k_tmp = H_tmp.shape[0]

                W.append(W_tmp)
                H.append(H_tmp)
                r2.append(r2_tmp)
                k.append(k_tmp)

            return W, H, np.array(r2), np.array(k)
        # endregion

        # region RECONSTRUCT:emg
        case 'RECONSTRUCT:emg':

            M = pd.read_csv(os.path.join(gl.baseDir, experiment, 'chords', 'M.tsv'), sep='\t')[gl.channels['emg'] +
                                                                                               ['sn', 'chordID']]

            recon_dict = {
                'participant_id': [],
                'partition': [],
                # 'matches': [],
                'r2_chord2nat': [],
                'r2_nat2chord': [],
                'nc_natural': [],
                'nc_chords': [],
                'r2_chord2nat_shuffle': [],
                'r2_nat2chord_shuffle': [],
                'nat_recon': [],
                'chord_recon': []
            }

            scaler = MinMaxScaler()

            for p in participant_id:
                sn = int(''.join([c for c in p if c.isdigit()]))

                M_tmp = M[M['sn'] == sn].groupby(['chordID']).mean().reset_index()[gl.channels['emg']].to_numpy()
                M_tmp = scaler.fit_transform(M_tmp)
                W_tmp, H_tmp, r2_tmp = iterative_nnmf(M_tmp, thresh=0.1)

                M_nat = load_nat_emg(os.path.join(gl.baseDir, experiment, 'natural',
                                                  f'natChord_{p}_emg_natural_whole_sampled.mat'))

                # noise ceiling chords
                nc_chords = list()
                for j in range(M_tmp.shape[0]):
                    M_tmp_j = M_tmp[j, :]
                    M_tmp_not_j = np.delete(M_tmp, j, axis=0)

                    W_j, H_j, _ = iterative_nnmf(M_tmp_not_j, thresh=0.1)
                    W_c, _ = nnls(H_j.T, M_tmp_j)
                    M_rec = np.dot(W_c, H_j)

                    nc_chords.append(calc_r2(M_tmp_j, M_rec))
                recon_dict['nc_chords'].append(np.array(nc_chords).mean())

                matches = [[] for _ in range(len(M_nat))]
                for m, M_nat_tmp in enumerate(M_nat):
                    M_nat_tmp = scaler.fit_transform(M_nat_tmp)
                    norm = np.linalg.norm(M_nat_tmp, axis=1)
                    M_nat_tmp = M_nat_tmp[norm > 2 * norm.mean()]

                    W_nat_tmp, H_nat_tmp, r2_nat_tmp = iterative_nnmf(M_nat_tmp, thresh=0.1)

                    # find matching synergies
                    dprod = np.dot(H_tmp, H_nat_tmp.T)
                    idx_matched = np.argmax(dprod, axis=0)

                    for i in range(H_nat_tmp.shape[0]):
                        matches[m].append((H_nat_tmp[i], H_tmp[idx_matched[i]]))

                    # reconstruct chords from natural
                    W_c = np.zeros((M_tmp.shape[0], H_nat_tmp.shape[0]))
                    res = np.zeros(M_tmp.shape[0])
                    for i in range(M_tmp.shape[0]):
                        W_c[i], res[i] = nnls(H_nat_tmp.T, M_tmp[i])

                    M_rec = np.dot(W_c, H_nat_tmp)

                    recon_dict['r2_nat2chord'].append(calc_r2(M_tmp, M_rec))

                    # shuffle synergies
                    nrep = 10
                    r2_shuffle = list()
                    for j in range(nrep):
                        H_nat_tmp_shuffle = H_nat_tmp.copy().T
                        np.random.shuffle(H_nat_tmp_shuffle)
                        W_c = np.zeros((M_tmp.shape[0], H_nat_tmp.shape[0]))
                        for i in range(M_tmp.shape[0]):
                            W_c[i], res_shuffle_tmp = nnls(H_nat_tmp_shuffle, M_tmp[i])
                        r2_shuffle.append(calc_r2(M_tmp, np.dot(W_c, H_nat_tmp_shuffle.T)))

                    recon_dict['r2_nat2chord_shuffle'].append(np.array(r2_shuffle).mean())

                    # noise ceiling natural
                    nc_natural = []
                    for j, M_nat_j in enumerate(M_nat):
                        if j != m:
                            norm = np.linalg.norm(M_nat_j, axis=1)
                            M_nat_j = M_nat_j[norm > norm.mean()]
                            M_nat_j = scaler.fit_transform(M_nat_j)

                            if M_nat_tmp.shape[0] > M_nat_j.shape[0]:
                                M_nat_tmp = M_nat_tmp[:M_nat_j.shape[0]]
                            else:
                                M_nat_j = M_nat_j[:M_nat_tmp.shape[0]]

                            _, H_nat_j, _ = iterative_nnmf(M_nat_j, thresh=0.1)
                            W_c = np.zeros((M_nat_tmp.shape[0], H_nat_j.shape[0]))
                            for k in range(M_nat_tmp.shape[0]):
                                W_c[k], _ = nnls(H_nat_j.T, M_nat_tmp[k])
                            nc_natural.append(calc_r2(M_nat_tmp, np.dot(W_c, H_nat_j)))

                    recon_dict['nc_natural'].append(np.array(nc_natural).mean())

                    # reconstruct natural from chords
                    W_c = np.zeros((M_nat_tmp.shape[0], M_tmp.shape[0]))
                    res = np.zeros(M_nat_tmp.shape[0])
                    for i in range(M_nat_tmp.shape[0]):
                        W_c[i], res[i] = nnls(M_tmp.T, M_nat_tmp[i])

                    M_rec = np.dot(W_c, M_tmp)
                    recon_dict['r2_chord2nat'].append(calc_r2(M_nat_tmp, M_rec))
                    recon_dict['nat_recon'].append(M_rec)

                    # shuffle chords
                    nrep = 10
                    r2_shuffle = list()
                    for j in range(nrep):
                        M_tmp_shuffle = M_tmp.copy().T
                        np.random.shuffle(M_tmp_shuffle)
                        W_c = np.zeros((M_nat_tmp.shape[0], M_tmp.shape[0]))
                        for i in range(M_nat_tmp.shape[0]):
                            W_c[i], _ = nnls(M_tmp_shuffle, M_nat_tmp[i])
                        r2_shuffle.append(calc_r2(M_nat_tmp, np.dot(W_c, M_tmp_shuffle.T)))

                    recon_dict['r2_chord2nat_shuffle'].append(np.array(r2_shuffle).mean())

                    recon_dict['participant_id'].append(p)
                    recon_dict['partition'].append(m)

                    pass

            with open(os.path.join(gl.baseDir, experiment, f'recon_emg.pkl'), "wb") as file:
                pickle.dump(recon_dict, file)

            return recon_dict

        # endregion

        # region NOISE_CEILING:force
        case 'NOISE_CEILING:force':

            F = pd.read_csv(os.path.join(gl.baseDir, experiment, 'force', 'M.tsv'), sep='\t')[gl.channels['force'] +
                                                                                              ['sn', 'chordID']]
            model = LinearRegression()

            nc_up, nc_low = [], []

            # noise ceiling upper
            F_avg = F.groupby(['chordID'])[gl.channels['force']].mean().reset_index()[gl.channels['force']].to_numpy()
            for p in participant_id:
                sn = int(''.join([c for c in p if c.isdigit()]))
                F_tmp = F[F['sn'] == sn].groupby(['chordID']).mean().reset_index()[gl.channels['force']].to_numpy()
                model.fit(F_avg, F_tmp)
                F_hat = model.predict(F_tmp)
                nc_up.append(calc_r2(F_avg, F_hat))

            # noise ceiling lower
            for p in participant_id:
                sn = int(''.join([c for c in p if c.isdigit()]))
                F_tmp = F[F['sn'] == sn].groupby(['chordID']).mean().reset_index()[gl.channels['force']].to_numpy()
                F_avg_tmp = F[F['sn'] != sn].groupby(['chordID']).mean().reset_index()[gl.channels['force']].to_numpy()
                model.fit(F_avg_tmp, F_tmp)
                F_hat = model.predict(F_tmp)
                nc_low.append(calc_r2(F_avg_tmp, F_hat))

            return np.array(nc_up).mean(), np.array(nc_low).mean()
        # endregion

        # region NOISE_CEILING:tau
        case 'NOISE_CEILING:tau':

            with open(os.path.join(gl.baseDir, experiment, f'tau.pkl'), "rb") as file:
                tau_dict = pickle.load(file)

            df_tau = pd.DataFrame(tau_dict)

            tau_vec = np.zeros((len(df_tau), 10))
            for i, df in df_tau[['tau', 'chordID']].iterrows():
                if df['tau'] is not None:
                    chordID_str = f"{int(df['chordID']):d}"
                    fingers = np.array([f != '9' for f in chordID_str])

                    tau_tmp = df['tau'].copy()
                    np.fill_diagonal(tau_tmp, np.nan)
                    tau_tmp[fingers == False, :] = np.nan
                    tau_tmp[:, fingers == False] = np.nan
                    triu = np.triu(np.ones(tau_tmp.shape), k=1)
                    tau_vec[i] = np.extract(triu, tau_tmp)
                else:
                    tau_vec[i] = np.nan

            tau = np.array([x.flatten() for x in df_tau['tau'] if x is not None])
            df_tau_tmp = pd.DataFrame(tau_vec)
            # df_tau_tmp = df_tau_tmp.loc[:, (df_tau_tmp != 0).any(axis=0)]
            # df_tau_tmp.columns = [i for i in range(20)]
            df_tau = pd.concat([df_tau, df_tau_tmp], axis=1).drop(columns=['tau'])
            # df_tau = df_tau.loc[df_tau['chord'] == 'trained']
            # df_tau['partition'] = np.random.randint(low=0, high=2, size=len(df_tau))

            df_tau_avg = df_tau.groupby(['chordID'])[[i for i in range(10)]].mean().reset_index()[
                [i for i in range(10)] + ['chordID']]

            model = LinearRegression()

            nc_up, nc_low = [], []

            # noise ceiling upper
            corr = list()
            for p in participant_id:
                for ch in df_tau['chordID'].unique():
                    df_tau_tmp = df_tau[(df_tau['chordID'] == ch) & (df_tau['participant_id'] == p)]
                    tau_tmp = df_tau_tmp[[i for i in range(10)]].to_numpy()
                    tau_tmp = tau_tmp[~np.isnan(tau_tmp).all(axis=1)]
                    tau_avg_tmp = tau_tmp.mean(axis=0)
                    for t in tau_tmp:
                        corr_tmp = np.corrcoef(t[~np.isnan(t)], tau_avg_tmp[~np.isnan(tau_avg_tmp)])[0, 1]
                        # if np.isnan(corr_tmp):
                        #     pass
                        corr.append(corr_tmp)
            corr = np.array(corr)
            nc = np.nanmean(corr)
            # model.fit(tau_avg, tau_tmp)
            # tau_hat = model.predict(tau_tmp)
            # nc_up.append(calc_r2(tau_avg, tau_hat))

            # # noice ceiling lower
            # for p in participant_id:
            #     df_tau_tmp = df_tau.loc[df_tau['participant_id'] == p]
            #     tau_tmp = df_tau_tmp.groupby(['chordID'])[[i for i in range(10)]].mean().reset_index()[
            #         [i for i in range(10)]].to_numpy()
            #     tau_avg_tmp = df_tau.loc[df_tau['participant_id'] != p].groupby(['chordID'])[
            #         [i for i in range(10)]].mean().reset_index()[[i for i in range(10)]].to_numpy()
            #     corr = list()
            #     for c1, c2 in zip(tau_tmp, tau_avg_tmp):
            #         corr.append(np.corrcoef(c1[~np.isnan(c1)], c2[~np.isnan(c2)])[0, 1])
            #     nc_low.append(np.array(corr).mean())
            #     # model.fit(tau_avg_tmp, tau_tmp)
            #     # tau_hat = model.predict(tau_tmp)
            #     # nc_low.append(calc_r2(tau_avg_tmp, tau_hat))

            return nc

        # endregion

        # region PLOT:success
        case 'PLOT:success':
            if fig is None or axs is None:
                fig, axs = plt.subplots()

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.csv'))
            df_success = calc_success(metrics)

            sns.boxplot(df_success, ax=axs, x='day', y='success', hue='chord', dodge=True, palette=['red', 'blue'])

            custom_handles = [
                Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', label='untrained'),
                Line2D([0], [0], marker='o', color='red', markerfacecolor='red', label='trained')
            ]

            return fig, axs, custom_handles

        # endregion

        # region PLOT:metric_repetition
        case 'PLOT:metric_repetition':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.csv'))
            df = calc_avg(metrics, by=['chord', 'day', 'repetition', 'participant_id'], columns=metric)
            offset = 5
            df['offset'] = df['repetition'] + df['day'] * offset

            sns.lineplot(data=df[df['chord'] == 'trained'], ax=axs, x='offset', y=metric, hue='day', errorbar='se',
                         palette=['red'] * 5, marker='o', markeredgewidth=0, err_kws={'linewidth': 0})
            sns.lineplot(data=df[df['chord'] == 'untrained'], ax=axs, x='offset', y=metric, hue='day', errorbar='se',
                         palette=['blue'] * 2, marker='o', markeredgewidth=0, err_kws={'linewidth': 0})
            axs.set_xticks(np.linspace(8, 28, 5))
            axs.set_xticklabels(np.linspace(1, 5, 5, dtype=int))

            custom_handles = [
                Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', label='untrained'),
                Line2D([0], [0], marker='o', color='red', markerfacecolor='red', label='trained')
            ]

            return fig, axs, custom_handles
        # endregion

        # region PLOT:force_in_trial
        case 'PLOT:force_in_trial':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            force = Force(experiment, participant_id, session=session, day=day)
            force_dict = force.load_pkl()
            force_trial = force_dict['force'][ntrial] * np.array(
                [1, 1, 1, 1.5, 1.5])  # specify force gain for visualization
            chordID = int(force_dict['chordID'][ntrial])
            # session = force_dict['session'][ntrial]
            day = force_dict['day'][ntrial]

            tAx = np.linspace(0, force_trial.shape[0] / gl.fsample, force_trial.shape[0])

            axs.plot(tAx, force_trial)

            axs.axhline(0, color='k', lw=.8, ls='-')
            axs.axhline(-1.2, color='k', lw=.8, ls='--')
            axs.axhline(1.2, color='k', lw=.8, ls='--')
            axs.axhline(2, color='k', lw=.8, ls='-.')
            axs.axhline(-2, color='k', lw=.8, ls='-.')
            axs.axhline(5, color='k', lw=.8, ls='-.')
            axs.axhline(-5, color='k', lw=.8, ls='-.')

            chord = "trained" if chordID in force.trained else "untrained"

            # axs.set_xlabel('time (s)')
            # axs.set_ylabel('force (N)')
            # axs.set_title(f'{chordID}, day{day}, {"trained" if chordID in force.trained else "untrained"}')

            # axs.legend(gl.channels['force'])

            # plt.show()

            return fig, axs, chordID, day, chord
        # endregion

        # region PLOT:xcorr_chord
        case 'PLOT:xcorr_chord':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            with open(os.path.join(gl.baseDir, experiment, f'tau.pkl'), "rb") as file:
                tau_dict = pickle.load(file)

            tau = list()
            df_tau = pd.DataFrame(tau_dict)

            chordID_str = str(chordID)
            fingers = np.array([f != '9' for f in chordID_str])

            for p in participant_id:
                df_tau_tmp = df_tau[(df_tau['participant_id'] == p) &
                                    (df_tau['day'] == day) &
                                    (df_tau['chord'] == chord) &
                                    (df_tau['chordID'] == chordID)]
                df_tau_tmp.dropna(subset=['tau'], inplace=True)

                if len(df_tau_tmp) > 0:
                    tau.append(np.stack(df_tau_tmp['tau'].values).mean(axis=0))

            tau = np.array(tau).mean(axis=0)
            np.fill_diagonal(tau, np.nan)
            tau[fingers == False, :] = np.nan
            tau[:, fingers == False] = np.nan

            cax = axs.imshow(tau, vmin=-.5, vmax=.5, cmap='PiYG')

            axs.set_xticks(np.linspace(0, 4, 5))
            axs.set_yticks(np.linspace(0, 4, 5))
            axs.set_xticklabels(gl.channels['force'], rotation=45)
            axs.set_yticklabels(gl.channels['force'], rotation=45)

            # fig.suptitle(f'{chordID}, {chord}, day{day}')

            return fig, axs, cax
        # endregion

        # region PLOT:xcorr_day
        case 'PLOT:xcorr_day':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            with open(os.path.join(gl.baseDir, experiment, f'tau.pkl'), "rb") as file:
                tau_dict = pickle.load(file)

            df_tau = pd.DataFrame(tau_dict)

            xcorr = {
                'participant_id': [],
                'day': [],
                'chordID': [],
                'chord': [],
                'tau': []
            }

            for p in participant_id:
                for day in gl.days:
                    for chordID in gl.chordID:

                        if day == '1' or day == '5':
                            session = 'testing'
                        else:
                            session = 'training'

                        if chordID in Force(experiment, p, session, day).trained:
                            chord = 'trained'
                        else:
                            chord = 'untrained'

                        df_tau_tmp = df_tau[(df_tau['participant_id'] == p) &
                                            (df_tau['day'] == day) &
                                            (df_tau['chordID'] == chordID)]

                        chordID_str = str(chordID)
                        fingers = np.array([f != '9' for f in chordID_str])

                        tau_avg = []
                        for tau_tmp in df_tau_tmp['tau']:
                            if tau_tmp is not None:
                                np.fill_diagonal(tau_tmp, np.nan)
                                tau_tmp[fingers == False, :] = np.nan
                                tau_tmp[:, fingers == False] = np.nan
                                tau_avg.append(np.nanmean(np.abs(tau_tmp)))

                        xcorr['participant_id'].append(p)
                        xcorr['day'].append(day)
                        xcorr['chordID'].append(chordID)
                        xcorr['chord'].append(chord)
                        xcorr['tau'].append(np.array(tau_avg).mean())

            df = pd.DataFrame(xcorr)

            sns.boxplot(data=df, ax=axs, x='day', y='tau', hue='chord', palette=['red', 'blue'])

            return fig, axs

        # region PLOT:xcorr_corr
        case 'PLOT:xcorr_corr':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            with open(os.path.join(gl.baseDir, experiment, f'tau.pkl'), "rb") as file:
                tau_dict = pickle.load(file)

            df_tau = pd.DataFrame(tau_dict)

            xcorr_corr = {
                'participant_id': [],
                'day': [],
                'chordID': [],
                'chord': [],
                'corr': [],
                'corr_shuff': []
            }

            nperm = 10

            for p in participant_id:
                for day in gl.days:
                    for chordID in gl.chordID:

                        if day == '1' or day == '5':
                            session = 'testing'
                        else:
                            session = 'training'

                        if chordID in Force(experiment, p, session, day).trained:
                            chord = 'trained'
                        else:
                            chord = 'untrained'

                        df_tau_tmp = df_tau[(df_tau['participant_id'] == p) &
                                            (df_tau['day'] == day) &
                                            (df_tau['chordID'] == chordID)]

                        if len(df_tau_tmp) > 0:

                            tau = [t for t in df_tau_tmp['tau'] if t is not None]
                            tau = np.stack(tau)

                            chordID_str = str(chordID)
                            fingers = np.array([f != '9' for f in chordID_str])

                            tau_vec = np.zeros((tau.shape[0], 10))
                            for t in range(tau.shape[0]):
                                tau_tmp = tau[t, ...].copy()
                                np.fill_diagonal(tau_tmp, np.nan)
                                tau_tmp[fingers == False, :] = np.nan
                                tau_tmp[:, fingers == False] = np.nan
                                triu = np.triu(np.ones(tau_tmp.shape), k=1)
                                tau_vec[t] = np.extract(triu, tau_tmp)

                            df_tau_vec = pd.DataFrame(tau_vec.T).dropna(axis=0, how='all')
                            corr = df_tau_vec.corr().to_numpy()
                            np.fill_diagonal(corr, np.nan)

                            mcorr_shuff = np.zeros(nperm)
                            for i in range(nperm):
                                df_tau_vec_shuff = pd.DataFrame(
                                    np.apply_along_axis(np.random.permutation, 1, df_tau_vec.values),
                                    columns=df_tau_vec.columns)
                                corr_shuff = df_tau_vec_shuff.corr().to_numpy()
                                np.fill_diagonal(corr_shuff, np.nan)
                                mcorr_shuff[i] = np.nanmean(corr_shuff)

                            xcorr_corr['participant_id'].append(p)
                            xcorr_corr['day'].append(day)
                            xcorr_corr['chordID'].append(str(chordID))
                            xcorr_corr['chord'].append(chord)
                            xcorr_corr['corr'].append(np.nanmean(corr))
                            xcorr_corr['corr_shuff'].append(mcorr_shuff.mean())

            df = pd.DataFrame(xcorr_corr)

            sns.boxplot(data=df, ax=axs, x='day', y='corr', hue='chord', palette=['red', 'blue'])

            axs.axhline(df['corr_shuff'].mean(), color='k', lw=.8, ls='--')
            axs.axhline(0, color='k', lw=.8, ls='-')
            axs.legend().remove()

            axs.text(axs.get_xlim()[1], df['corr_shuff'].mean(), 'shuffled data', va='center', ha='left')

            return fig, axs
        # endregion

        # region PLOT:recon_emg
        case 'PLOT:recon_emg':
            if fig is None or axs is None:
                fig, axs = plt.subplots(figsize=(3, 4))

            with open(os.path.join(gl.baseDir, experiment, 'recon_emg.pkl'), 'rb') as file:
                recon_dict = pickle.load(file)

            df = pd.DataFrame({key: recon_dict[key] for key in ['participant_id',
                                                                'partition',
                                                                'r2_chord2nat',
                                                                'r2_nat2chord',
                                                                'r2_chord2nat_shuffle',
                                                                'r2_nat2chord_shuffle']})
            df_r2 = df.groupby('participant_id')[['r2_chord2nat',
                                                  'r2_nat2chord']].mean().reset_index()
            df_shuffle = df.groupby('participant_id')[['r2_chord2nat_shuffle',
                                                       'r2_nat2chord_shuffle']].mean().reset_index()

            df_r2_melt = df_r2.melt(id_vars=['participant_id'],
                                    value_vars=['r2_chord2nat', 'r2_nat2chord'],
                                    var_name='reconstruction', value_name='R²')

            pval_chord2nat, pval_perm_chord2nat = perm_test_1samp(df_r2['r2_chord2nat'],
                                                                  df_shuffle['r2_chord2nat_shuffle'], nperm=5000)
            pval_nat2chord, pval_perm_nat2chord = perm_test_1samp(df_r2['r2_nat2chord'],
                                                                  df_shuffle['r2_nat2chord_shuffle'], nperm=5000)

            width = .5
            H = sns.boxplot(ax=axs, x='reconstruction', y='R²', data=df_r2_melt, width=width)

            pos = H.get_xticks()
            shuffle_chord = df_shuffle['r2_chord2nat_shuffle']
            axs.hlines(y=shuffle_chord.mean(), xmin=pos[0] - width / 2, xmax=pos[0] + width / 2, color='k', ls='-',
                       lw=2)
            axs.hlines(y=shuffle_chord, xmin=pos[0] - width / 2, xmax=pos[0] + width / 2, color='k', ls='-',
                       alpha=0.2, lw=.8)

            shuffle_nat = df_shuffle['r2_nat2chord_shuffle']
            axs.hlines(y=shuffle_nat.mean(), xmin=pos[1] - width / 2, xmax=pos[1] + width / 2, color='k', ls='-',
                       lw=2)
            axs.hlines(y=shuffle_nat, xmin=pos[1] - width / 2, xmax=pos[1] + width / 2, color='k', ls='-',
                       alpha=0.2, lw=.8)

            fig.subplots_adjust(left=0.2)

            plt.show()

            pass
        # endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None, choices=[
        'FORCE:preprocessing',  # ok
        # 'FORCE:variance_decomposition',
        'FORCE:xcorr',  # ok
        'NOISE_CEILING:tau',  # ok
        # 'EMG:recon_chord2nat',
        # 'EMG:recon_chord2chord',
        'EMG:nnmf',
        'RECONSTRUCT:force',
        'RECONSTRUCT:emg',
        'PLOT:success',
        # 'PLOT:variance_decomposition',
        'PLOT:metric_repetition',  # ok
        # 'PLOT:model_xcorr_day',
        'PLOT:force_in_trial',  # ok
        'PLOT:xcorr_chord',  # ok
        # 'PLOT:xcorr_day',
        'PLOT:recon_emg',
        'PLOT:xcorr_corr',  # ok
    ])
    parser.add_argument('--experiment', default='efc2', help='')
    parser.add_argument('--participant_id', nargs='+', default=None, help='')
    parser.add_argument('--session', default=None, help='', choices=['training', 'testing'])
    parser.add_argument('--day', default=gl.days, help='')
    parser.add_argument('--ntrial', default=None, help='')
    parser.add_argument('--metric', default=None, choices=['MD', 'RT', 'ET'], help='')
    parser.add_argument('--chordID', default=None, help='')
    parser.add_argument('--chord', default=None, help='', choices=['trained', 'untrained'])
    # parser.add_argument('--dataset', default=None, help='', choices=['natural', 'chords'])

    args = parser.parse_args()

    what = args.what
    experiment = args.experiment
    participant_id = args.participant_id
    session = args.session
    day = args.day
    ntrial = int(args.ntrial) if args.ntrial is not None else None
    metric = args.metric
    chordID = int(args.chordID) if args.chordID is not None else None
    chord = args.chord
    # dataset = args.dataset

    if participant_id is None:
        participant_id = gl.participants[experiment]

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    main(what, experiment=experiment, participant_id=participant_id, session=session, day=day,
         ntrial=ntrial, chordID=chordID, chord=chord, metric=metric)

    plt.show()
