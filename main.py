import argparse
import json
import os

import scipy
from matplotlib import pyplot as plt
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
from stats import perm_test_1samp
from util import load_nat_emg
from variance_decomposition import reliability_var


def main(what, experiment=None, participant_id=None, session=None, day=None, ntrial=None, chordID=None, chord=None,
         dataset=None, fig=None, axs=None):
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
                    tau_dict['tau'].append(np.stack(tau, axis=2))
                    tau_dict['experiment'].append(experiment)
                    tau_dict['participant_id'].append(p)
                    tau_dict['session'].append(session)
                    tau_dict['day'].append(day)
                    tau_dict['chordID'].append(force_dict['chordID'])
                    tau_dict['chord'].append(['trained' if CID in force.trained
                                              else 'untrained' for CID in force_dict['chordID']])

                    pass

            with open(os.path.join(gl.baseDir, experiment, f'tau.pkl'), "wb") as file:
                pickle.dump(tau_dict, file)

            return tau_dict

        #endregion

        # region FORCE:variance_decomposition
        case 'FORCE:variance_decomposition':

            group_cols = ['subNum', 'chord', 'day', 'chordID', 'BN', 'TN', 'repetition', 'trialPoint', 'participant_id']
            num_cols = [f'dist{i}' for i in range(5)]

            df = pd.read_csv(os.path.join(gl.baseDir, experiment, 'results.csv'))
            df = df[df['chord'] == 'trained']
            df = df.groupby(group_cols)[num_cols].mean().reset_index()

            df_tmp = pd.DataFrame()
            part_vec = list()
            for p in df['participant_id'].unique():
                for c in df['chordID'].unique():
                    for day in df['day'].unique():
                        part_vec.extend(np.repeat([1, 2, 3, 4, 5], 5))
                        df_tmp = pd.concat([df_tmp,
                                            df[(df['chordID'] == c) &
                                               (df['day'] == day) &
                                               (df['participant_id'] == p)].iloc[:25]])

            df = df_tmp
            df['part_vec'] = part_vec

            v_g, v_gs, v_gse, conds, days = [], [], [], [], []
            for day in gl.days:
                df_tmp = df[df['day'] == int(day)]
                Y = df_tmp[num_cols].values
                v_g_tmp, v_gs_tmp, v_gse_tmp, conds_tmp = reliability_var(Y,
                                                                          df_tmp['participant_id'],
                                                                          df_tmp['part_vec'],
                                                                          cond_vec=df_tmp['chordID'],
                                                                          centered=True)
                v_g.extend(v_g_tmp)
                v_gs.extend(v_gs_tmp)
                v_gse.extend(v_gse_tmp)
                conds.extend(conds_tmp)
                days.extend(np.repeat(day, len(conds_tmp)))

            df_out = pd.DataFrame({
                'v_g': v_g,
                'v_gs': v_gs,
                'v_gse': v_gse,
                'conds': conds,
                'days': days
            })

            return df_out

        # endregion

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

        # region RECONSTRUCT:emg_chord_2_force_chord
        case 'RECONSTRUCT:emg_chord_2_force_chord':

            M = pd.read_csv(os.path.join(gl.baseDir, experiment, dataset, 'M.tsv'), sep='\t')[gl.channels['emg'] +
                                                                                              ['sn', 'chordID']]
            F = pd.read_csv(os.path.join(gl.baseDir, experiment, dataset, 'M.tsv'), sep='\t')[gl.channels['force'] +
                                                                                              ['sn', 'chordID']]

            model = LinearRegression()
            scaler = MinMaxScaler()

            r2_red, r2_all = [], []
            for p in participant_id:
                sn = int(''.join([c for c in p if c.isdigit()]))
                M_tmp = M[M['sn'] == sn].groupby(['chordID']).mean().reset_index()[gl.channels['emg']].to_numpy()
                M_tmp = scaler.fit_transform(M_tmp)
                W_tmp, H_tmp, r2_tmp = iterative_nnmf(M_tmp, thresh=0.1)

                F_tmp = F[F['sn'] == sn].groupby(['chordID']).mean().reset_index()[gl.channels['force']].to_numpy()

                model.fit(np.dot(W_tmp, H_tmp), F_tmp)
                F_hat = model.predict(np.dot(W_tmp, H_tmp))
                r2_red.append(calc_r2(F_tmp, F_hat))

                model.fit(M_tmp, F_tmp)
                F_hat = model.predict(M_tmp)
                r2_all.append(calc_r2(F_tmp, F_hat))

                pass

            return np.array(r2_red), np.array(r2_all)
        # endregion

        # region RECONSTRUCT:emg_chord_2_emg_nat
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

        # region FORCE:noise_ceiling
        case 'FORCE:noise_ceiling':

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

        # region PLOT:variance_decomposition
        case 'PLOT:variance_decomposition':

            df = main('FORCE:variance_decomposition', experiment, participant_id)
            df['v_s'] = (df['v_gs'] - df['v_g']) / df['v_gse']
            df['v_e'] = (df['v_gse'] - df['v_gs']) / df['v_gse']
            df['v_g'] = df['v_g'] / df['v_gse']
            df = df.melt(id_vars=['conds', 'days'], value_vars=['v_g', 'v_s', 'v_e'], var_name='variance')

            chords = df['conds'].unique()
            fig, axs = plt.subplots(nrows=len(chords), sharex=True, sharey=True, figsize=(4, 8))

            for ch, chord in enumerate(chords):
                sns.barplot(data=df[df['conds'] == chord],
                            ax=axs[ch],
                            x='days',
                            y='value',
                            hue='variance')
                axs[ch].legend_.remove()
                axs[ch].set_ylabel('')
                axs[ch].set_title(chord)
                axs[ch].set_yscale('linear')

            handles, _ = axs[0].get_legend_handles_labels()
            fig.legend(handles, ['chord', 'subject', 'noise'], loc='upper center', ncol=3)
            fig.subplots_adjust(left=0.2, hspace=0.4)
            fig.supylabel('variance')
            # fig.tight_layout()

            plt.show()

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

            axs.set_xlabel('time (s)')
            axs.set_ylabel('force (N)')
            axs.set_title(f'{chordID}, day{day}, {"trained" if chordID in force.trained else "untrained"}')

            axs.legend(gl.channels['force'])

            plt.show()
        # endregion

        # region PLOT:xcorr
        case 'PLOT:xcorr_chord':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            with open(os.path.join(gl.baseDir, experiment, f'tau.pkl'), "rb") as file:
                tau_dict = pickle.load(file)

            tau = list()

            chordID_str = str(chordID)
            fingers = np.array([f != '9' for f in chordID_str])

            for p in participant_id:
                pix = [i for i, (P, d)
                       in enumerate(zip(tau_dict['participant_id'],
                                        tau_dict['day'])) if (P == p) & (d == day)][0]
                tau_tmp = tau_dict['tau'][pix][..., (np.array(tau_dict['chord'][pix]) == chord) &
                                                    (np.array(tau_dict['chordID'][pix]) == chordID)]
                tau.append(tau_tmp.mean(axis=-1))

            tau = np.array(tau).mean(axis=0)
            np.fill_diagonal(tau, np.nan)
            tau[fingers == False, :] = np.nan
            tau[:, fingers == False] = np.nan

            axs.imshow(tau, vmin=-.5, vmax=.5, cmap='seismic')
            fig.suptitle(f'{chordID}, {chord}, day{day}')

            plt.show()

        # endregion

        # region FORCE:xcorr_corr
        case 'PLOT:xcorr_corr':

            if fig is None or axs is None:
                fig, axs = plt.subplots(figsize=(3, 4))

            with open(os.path.join(gl.baseDir, experiment, f'tau.pkl'), "rb") as file:
                tau_dict = pickle.load(file)

            xcorr_corr = {
                'participant_id': [],
                'day': [],
                'chordID': [],
                'chord': [],
                'corr': []
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

                        pix = [i for i, (P, d)
                               in enumerate(zip(tau_dict['participant_id'],
                                                tau_dict['day'])) if (P == p) & (d == day)][0]
                        tau = tau_dict['tau'][pix][..., (np.array(tau_dict['chord'][pix]) == chord) &
                                                        (np.array(tau_dict['chordID'][pix]) == chordID)]

                        chordID_str = str(chordID)
                        fingers = np.array([f != '9' for f in chordID_str])

                        tau_vec = np.zeros((tau.shape[-1], 10))
                        for t in range(tau.shape[-1]):
                            tau_tmp = tau[..., t].copy()
                            np.fill_diagonal(tau_tmp, np.nan)
                            tau_tmp[fingers == False, :] = np.nan
                            tau_tmp[:, fingers == False] = np.nan
                            triu = np.triu(np.ones(tau_tmp.shape), k=1)
                            tau_vec[t] = np.extract(triu, tau_tmp)

                        corr = pd.DataFrame(tau_vec.T).corr().to_numpy()
                        np.fill_diagonal(corr, np.nan)

                        xcorr_corr['participant_id'].append(p)
                        xcorr_corr['day'].append(day)
                        xcorr_corr['chordID'].append(chordID)
                        xcorr_corr['chord'].append(chord)
                        xcorr_corr['corr'].append(np.nanmean(corr))

            df = pd.DataFrame(xcorr_corr)

            sns.boxplot(data=df, ax=axs, x='day', y='corr', hue='chord')

            plt.show()

        # endregion

        # region PLOT:xcorr_day
        case 'PLOT:xcorr_day':

            if fig is None or axs is None:
                fig, axs = plt.subplots(figsize=(3, 4))

            with open(os.path.join(gl.baseDir, experiment, f'tau.pkl'), "rb") as file:
                tau_dict = pickle.load(file)

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

                        pix = [i for i, (P, d)
                               in enumerate(zip(tau_dict['participant_id'],
                                                tau_dict['day'])) if (P == p) & (d == day)][0]
                        tau = tau_dict['tau'][pix][..., (np.array(tau_dict['chord'][pix]) == chord) &
                                                        (np.array(tau_dict['chordID'][pix]) == chordID)]

                        chordID_str = str(chordID)
                        fingers = np.array([f != '9' for f in chordID_str])

                        tau_avg = np.zeros(tau.shape[-1])
                        for t in range(tau.shape[-1]):
                            tau_tmp = tau[..., t].copy()
                            np.fill_diagonal(tau_tmp, np.nan)
                            tau_tmp[fingers == False, :] = np.nan
                            tau_tmp[:, fingers == False] = np.nan
                            tau_avg[t] = np.nanmean(np.abs(tau_tmp))

                        xcorr['participant_id'].append(p)
                        xcorr['day'].append(day)
                        xcorr['chordID'].append(chordID)
                        xcorr['chord'].append(chord)
                        xcorr['tau'].append(tau_avg.mean())

            df = pd.DataFrame(xcorr)

            sns.boxplot(data=df, ax=axs, x='day', y='tau', hue='chord')

            plt.show()

            pass

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
        'FORCE:preprocessing',
        'FORCE:variance_decomposition',
        'FORCE:xcorr',
        'FORCE:noise_ceiling',
        'EMG:recon_chord2nat',
        'EMG:recon_chord2chord',
        'EMG:nnmf',
        'RECONSTRUCT:force',
        'RECONSTRUCT:emg',
        'PLOT:variance_decomposition',
        'PLOT:force_in_trial',
        'PLOT:xcorr_chord',
        'PLOT:xcorr_day',
        'PLOT:recon_emg',
        'PLOT:xcorr_corr',
    ])
    parser.add_argument('--experiment', default='efc2', help='')
    parser.add_argument('--participant_id', nargs='+', default=None, help='')
    parser.add_argument('--session', default=None, help='', choices=['training', 'testing'])
    parser.add_argument('--day', default=None, help='')
    parser.add_argument('--ntrial', default=None, help='')
    parser.add_argument('--chordID', default=None, help='')
    parser.add_argument('--chord', default=None, help='', choices=['trained', 'untrained'])
    parser.add_argument('--dataset', default=None, help='', choices=['natural', 'chords'])

    args = parser.parse_args()

    what = args.what
    experiment = args.experiment
    participant_id = args.participant_id
    session = args.session
    day = args.day
    ntrial = int(args.ntrial) if args.ntrial is not None else None
    chordID = int(args.chordID) if args.chordID is not None else None
    chord = args.chord
    dataset = args.dataset

    if participant_id is None:
        participant_id = gl.participants[experiment]

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    out = main(what, experiment=experiment, participant_id=participant_id, session=session, day=day, ntrial=ntrial,
               chordID=chordID, chord=chord, dataset=dataset)
