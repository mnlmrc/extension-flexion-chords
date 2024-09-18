import argparse
import os
import time

from matplotlib import pyplot as plt
from scipy.optimize import nnls
from scipy.signal import resample, find_peaks
from sklearn.linear_model import LinearRegression

from scipy.stats import ttest_1samp, spearmanr

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import pickle
import globals as gl
import numpy as np

from joblib import Parallel, delayed

import seaborn as sns

from force import Force
from nnmf import iterative_nnmf, calc_reconerr, assert_selected_rows_belong, calc_r2
from stats import perm_test_1samp
from util import load_nat_emg, calc_avg, calc_success, lowpass_butter, time_to_seconds, lowpass_fir, \
    calc_distance_from_distr, fit_sigmoids, sigmoid
from variance_decomposition import reliability_var


def main(what, experiment=None, participant_id=None, session=None, day=None, chordID=None, chord=None, ntrial=None,
         fname=None, muscles=None, trigger=None, n_jobs=None,
         fig=None, axs=None, width=None, linewidth=None, linecolor=None, showfliers=True, color=None, palette=None,
         err_kw=None):
    if participant_id is None:
        participant_id = gl.participants[experiment]

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

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

                    force = Force(experiment, p, session, day)
                    metrics_tmp, force_dict = force.preprocessing()
                    metrics = pd.concat([metrics, metrics_tmp])

                    # save force to binary
                    with open(os.path.join(gl.baseDir, experiment, session,
                                           f'day{day}', f'{experiment}_{force.sn}_force.pkl'), "wb") as file:
                        pickle.dump(force_dict, file)

            print('saving...')
            metrics.to_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')

            return metrics
        # endregion

        # region FORCE:average
        case 'FORCE:average':

            # Calculate average force response on specified day

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')

            if day is None:
                day = '5'

            win = .8

            force_avg = {str(key): [] for key in gl.chordID}

            for P, p in enumerate(participant_id):

                metrics_tmp = metrics[(metrics['participant_id'] == p) &
                                      (metrics['day'] == int(day))]

                if day == '1' or day == '5':
                    session = 'testing'
                    # chords = gl.chordID
                else:
                    session = 'training'

                if chord is None:
                    chords = [str(ch) for ch in gl.chordID]
                else:
                    chords = pinfo[pinfo['participant_id'] == p][chord].iloc[0].split('.')

                force = Force(experiment, p, session, day)
                force_dict = force.load_pkl()

                # color = ['#4169E1', '#DC143C', '#228B22', '#DAA520', '#9932CC']

                F = list()
                chordID = list()
                for f, force_tmp in enumerate(force_dict['force_filt10Hz']):
                    if force_tmp is not None:
                        F.append(force_tmp[:int(win * gl.fsample['force'])].swapaxes(0, 1))
                        chordID.append(metrics_tmp['chordID'].iloc[f].astype(str))

                F = np.stack(F)
                chordID = pd.Series(chordID)

                for ch in chords:
                    ind = chordID == ch
                    F_tmp = F[ind]
                    force_avg[ch].append(F_tmp.mean(axis=0))

            return force_avg
        # endregion

        # region FORCE:derivative
        case 'FORCE:derivative':

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')

            if day is None:
                day = '5'

            win = .8

            scaler = MinMaxScaler()

            dforce_avg = {str(key): [] for key in gl.chordID}
            dforce_avg_norm = {str(key): [] for key in gl.chordID}

            for P, p in enumerate(participant_id):

                metrics_tmp = metrics[(metrics['participant_id'] == p) &
                                      (metrics['day'] == int(day))]

                if day == '1' or day == '5':
                    session = 'testing'
                    # chords = gl.chordID
                else:
                    session = 'training'

                if chord is None:
                    chords = [str(ch) for ch in gl.chordID]
                else:
                    chords = pinfo[pinfo['participant_id'] == p][chord].iloc[0].split('.')

                force = Force(experiment, p, session, day)
                force_dict = force.load_pkl()

                # color = ['#4169E1', '#DC143C', '#228B22', '#DAA520', '#9932CC']

                dF = list()
                dF_norm = list()
                chordID = list()
                cn = 0
                for f, force_tmp in enumerate(force_dict['force_filt10Hz']):
                    if force_tmp is not None:
                        df_tmp = np.gradient(force_tmp, 1 / gl.fsample['force'], axis=0)
                        df_tmp_norm = scaler.fit_transform(np.abs(df_tmp[:int(win * gl.fsample['force'])]))
                        dF.append(df_tmp[:int(win * gl.fsample['force'])].swapaxes(0, 1))
                        dF_norm.append(df_tmp_norm[:int(win * gl.fsample['force'])].swapaxes(0, 1))

                        chordID.append(metrics_tmp['chordID'].iloc[f].astype(str))

                dF = np.stack(dF)
                dF_norm = np.stack(dF_norm)
                chordID = pd.Series(chordID)

                for ch in chords:
                    ind = chordID == ch
                    dF_tmp = dF[ind]
                    dF_tmp_norm = dF_norm[ind]
                    dforce_avg[ch].append(dF_tmp.mean(axis=0))
                    dforce_avg_norm[ch].append(dF_tmp_norm.mean(axis=0))

            return dforce_avg, dforce_avg_norm

        # endregion

        # region EMG:csv2df
        case 'EMG:csv2df':

            if muscles is None:
                muscles = gl.channels['emgTMS']

            if trigger is None:
                trigger = 'Analog 1'

            filepath = os.path.join(gl.baseDir, experiment, session, f'day{day}', participant_id[0], fname)

            # read data from .csv file (Delsys output)
            with open(filepath, 'rt') as fid:
                A = []
                for line in fid:
                    # Strip whitespace and newline characters, then split
                    split_line = [elem.strip() for elem in line.strip().split(',')]
                    A.append(split_line)

            # identify columns with data from each muscle
            muscle_columns = {}
            for muscle in muscles:
                for c, col in enumerate(A[3]):
                    if muscle in col:
                        muscle_columns[muscle] = c + 1  # EMG is on the right of Timeseries data (that's why + 1)
                        break
                for c, col in enumerate(A[5]):
                    if f'{muscle} (mV)' in col:
                        muscle_columns[muscle] = c
                        break

            df_raw = pd.DataFrame(A[8:])  # get rid of header
            df_out = pd.DataFrame()  # init final dataframe

            for muscle in muscle_columns:
                df_out[muscle] = pd.to_numeric(df_raw[muscle_columns[muscle]],
                                               errors='coerce').replace('', np.nan).dropna()  # add EMG to dataframe

            # df_out = df_raw.iloc[:, 1::2].iloc[:, :14]
            # df_out = df_out.apply(pd.to_numeric, errors='coerce')
            # df_out = df_out.replace('', np.nan).dropna()

            # add trigger column
            trigger_column = None
            for c, col in enumerate(A[5]):
                if f'{trigger} (V)' in col:
                    trigger_column = c

            try:
                triggerCol = df_raw[trigger_column].replace('', np.nan).dropna().to_numpy().astype(float)
                triggerCol = resample(triggerCol, len(df_out))
            except IOError as e:
                raise IOError("Trigger not found") from e

            # df_out[trigger] = trigger

            bl = int(''.join([c for c in fname if c.isdigit()]))

            df_out = pd.concat([df_out, pd.DataFrame(triggerCol, columns=['Trigger'])], axis=1)

            df_out.to_csv(os.path.join(gl.baseDir, experiment, session,
                                       f'day{day}', participant_id[0], f'block{bl}.tsv'), sep='\t', index=False)

            # add time column
            # df_out['time'] = df_raw.loc[:, 0]

            return df_out
        # endregion

        # region EMG:merge_blocks
        case 'EMG:merge_blocks':

            blocks = pinfo[pinfo['participant_id'] == participant_id[0]][f'blocks {session[3:]} day{day}'][0].split('.')

            df_out = pd.DataFrame()
            for block in blocks:
                df_tmp = pd.read_csv(os.path.join(gl.baseDir, experiment, session,
                                                  f'day{day}', participant_id[0], f'block{block}.tsv'), sep='\t')
                df_out = pd.concat([df_out, df_tmp], axis=0)

            df_out.to_csv(os.path.join(gl.baseDir, experiment, session, f'day{day}', participant_id[0],
                                       'emg_continuous.tsv'), sep='\t', index=False)

        # endregion

        # region EMG:df2chord
        case 'EMG:df2chord':

            df_out = pd.DataFrame()

            for p in participant_id:
                sn = int(''.join([c for c in p if c.isdigit()]))

                dat = pd.read_csv(os.path.join(gl.baseDir, experiment, 'testing',
                                               f'day{day}', f'{experiment}_{sn}.dat'), sep='\t')

                df = pd.read_csv(os.path.join(gl.baseDir, experiment, session, f'day{day}',
                                              p, f'emg_continuous.tsv'), sep='\t')

                df.drop(gl.removeEMG[experiment][p], axis=1)

                trigger = df['Trigger'].to_numpy()
                df = df.drop('Trigger', axis=1)
                trigOn = trigger > 4
                trigOn_diff = np.diff(trigOn.astype(int))
                trigOn_times = np.where(trigOn_diff == 1)[0][dat['trialPoint'] == 1]  # Trigger turning on (rising edge)
                ET = (dat[dat['trialPoint'] == 1]['RT'] - 600) / 1000
                trigOn_times = (trigOn_times + ET * gl.fsample['emg']).to_numpy().astype(int)

                print(f'{p}: {len(trigOn_times)} trials found...')

                nsamples = int(.6 * gl.fsample['emg'])
                pattern = []
                chordID = []
                chord = []
                for TN in range(len(trigOn_times)):
                    pattern.append(
                        np.abs(df.iloc[trigOn_times[TN]:trigOn_times[TN] + nsamples, :].to_numpy()).mean(axis=0))
                    chordID_tmp = dat.iloc[TN]['chordID'].astype(int).astype(str)
                    chord_tmp = 'trained' if (chordID_tmp in
                                              pinfo[pinfo['participant_id'] == p].trained.iloc[0].split('.')) \
                        else 'untrained'
                    chordID.append(chordID_tmp)
                    chord.append(chord_tmp)

                df_out_tmp = pd.DataFrame(np.stack(pattern, axis=0), columns=df.columns)
                df_out_tmp = pd.concat([df_out_tmp,
                                        pd.DataFrame(chordID, columns=['chordID']),
                                        pd.DataFrame(chord, columns=['chord']),
                                        ], axis=1)
                df_out_tmp['BN'] = dat['BN'][dat['trialPoint'] == 1]
                df_out_tmp['participant_id'] = p

                df_out = pd.concat([df_out, df_out_tmp], axis=0)
                df_out.drop(gl.removeEMG[experiment][p], axis=1, inplace=True)

                df_out.to_csv(os.path.join(gl.baseDir, experiment, session,
                                           f'day{day}', p, f'Chords.tsv'), sep='\t', index=False)

        # endregion

        # region EMG:df2mep
        case 'EMG:df2mep':

            for p in participant_id:
                filepath = os.path.join(gl.baseDir, experiment, 'emgTMS', f'day{day}', p, 'emg_continuous.tsv')
                df = pd.read_csv(filepath, sep='\t')

                # nn_log = pd.read_csv(os.path.join(gl.baseDir, 'efc3', 'Brainsight', 'efc3_100_pretraining.tsv'), sep='\t')
                # time_in_seconds = [time_to_seconds(t) for t in nn_log['Time'].tolist()]
                # time_diff_NN = np.diff(time_in_seconds)

                trigger = df['Trigger'].to_numpy()
                df = df.drop('Trigger', axis=1)
                trigOn = trigger > 4
                trigOn_diff = np.diff(trigOn.astype(int))
                trigOn_times = np.where(trigOn_diff == 1)[0]  # Trigger turning on (rising edge)

                print(f'{p}: {len(trigOn_times)} MEPs found...')

                # time_diff_EMG = np.diff(trigOn_times / gl.fsample['emg'])

                nsamples = int(.05 * gl.fsample['emg'])
                mep = np.zeros((len(trigOn_times), df.shape[1], nsamples))
                for TN in range(len(trigOn_times)):
                    mep[TN] = df.iloc[trigOn_times[TN]:trigOn_times[TN] + nsamples, :].to_numpy().T

                np.save(os.path.join(gl.baseDir, experiment, 'emgTMS', f'day{day}', p, 'mep_segmented.npy'), mep)

            return mep, trigOn_times
        # endregion

        # region EMG:mep2amp
        case 'EMG:mep2amp':

            for p in participant_id:
                mep = np.load(os.path.join(gl.baseDir, experiment, 'emgTMS', f'day{day}', p, 'mep_segmented.npy'))
                mepAmp = np.ptp(mep, axis=-1)
                df_mepAmp = pd.DataFrame(mepAmp, columns=gl.channels['emgTMS'])
                df_mepAmp.drop(gl.removeEMG[experiment][p], axis=1, inplace=True)
                df_mepAmp.to_csv(os.path.join(gl.baseDir, experiment, 'emgTMS', f'day{day}', p, 'mepAmp.tsv'), sep='\t',
                                 index=False)

            return df_mepAmp
        # endregion

        # region EMG:nnmf
        case 'EMG:nnmf':

            W, H, r2 = [], [], []
            for p in participant_id:
                M = pd.read_csv(os.path.join(gl.baseDir, experiment, session, p, fname), sep='\t')
                M.drop(M.columns[0], axis=1, inplace=True)
                M = M.to_numpy()

                W_tmp, H_tmp, r2_tmp = iterative_nnmf(M, thresh=0.1)

                W.append(W_tmp)
                H.append(H_tmp)
                r2.append(r2_tmp)

            return W, H, r2

        # endregion

        # region EMG:distance
        case 'EMG:distance':

            n_thresh = 20

            log_prob = {
                'participant_id': [],
                'day': [],
                'chordID': [],
                'chord': [],
                'distance': [],
                'slope': [],
                'log_slope': []
            }

            for p in participant_id:
                for day in ['1', '5']:
                    mepAmp = pd.read_csv(os.path.join(gl.baseDir, experiment,
                                                      'emgTMS', f'day{day}', p, 'mepAmp.tsv'), sep='\t')
                    Chords = pd.read_csv(os.path.join(gl.baseDir, experiment,
                                                      'emgChords', f'day{day}', p, 'Chords.tsv'), sep='\t')
                    cols = Chords.columns
                    muscles = [col for col in cols if col in gl.channels['emgTMS']]
                    chords_avg = Chords.groupby(['chordID', 'chord']).mean(numeric_only=True).reset_index()

                    for index, row in chords_avg.iterrows():
                        pattern = row[muscles].to_numpy().astype(float)
                        distr = mepAmp.to_numpy()
                        d = calc_distance_from_distr(pattern, distr)
                        x = np.linspace(1, n_thresh, n_thresh)
                        slope = np.dot(x, d[:n_thresh]) / np.dot(x, x)
                        log_slope = np.log(1 / slope)

                        log_prob['chordID'].append(row['chordID'])
                        log_prob['chord'].append(row['chord'])
                        log_prob['participant_id'].append(p)
                        log_prob['day'].append(day)
                        log_prob['distance'].append(d)
                        log_prob['slope'].append(slope)
                        log_prob['log_slope'].append(log_slope)

            df_out = pd.DataFrame(log_prob)
            df_out.to_csv(os.path.join(gl.baseDir, experiment, 'log_prob.tsv'), sep='\t', index=False)

            return df_out
        # endregion

        # region NATURAL:peaks
        case 'NATURAL:extract_patterns_from_peaks':

            scaler = MinMaxScaler()
            for p in participant_id:
                natEMG = pd.read_csv(os.path.join(gl.baseDir, experiment, 'emgNatural', p, 'Natural_raw.tsv'),
                                     sep='\t', index_col=None)
                natEMG = natEMG.drop(natEMG.columns[0], axis=1)

                natEMG.drop(gl.removeEMG[experiment][p], axis=1, inplace=True)

                cols = natEMG.columns

                natEMG = natEMG.to_numpy()
                # natEMG = scaler.fit_transform(natEMG)
                natEMG = np.abs(natEMG).T

                norm = np.linalg.norm(natEMG, axis=0)

                print(f'{p}: lowpass filtering...')
                norm_lp = lowpass_butter(norm, cutoff=10, fsample=gl.fsample['emg'], axis=-1)

                peaks = find_peaks(norm_lp)[0]

                emg_patterns = natEMG[:, peaks]

                df = pd.DataFrame(emg_patterns.T, columns=cols)
                df.to_csv(os.path.join(gl.baseDir, experiment, 'emgNatural', p, 'Natural_patterns.tsv'), sep='\t')

            return df
        # endregion

        # region EMG:mep_amplitude
        case 'MEP:nnmf':

            scaler = MinMaxScaler()

            for p in participant_id:
                mepAmp = pd.read_csv(os.path.join(gl.baseDir, 'efc3', 'emgTMS', p, 'mepAmp.tsv'), sep='\t').to_numpy()[
                         :, 1:]
                mepAmp = scaler.fit_transform(mepAmp)
                W, H, r2 = iterative_nnmf(mepAmp, thresh=0.1)

                df_H = pd.DataFrame(H, columns=gl.channels['emgTMS'])
                df_H.to_csv(os.path.join(gl.baseDir, 'efc3', 'emgTMS', p, 'preTraining_H.tsv'), sep='\t')
                pass

        # endregion

        # region XCORR:tau
        case 'XCORR:tau':

            # Dictionary
            tau_dict = {
                'tau': [],
                'tauAbs': [],
                'tauRel': [],
                'repetition': [],
                'experiment': [],
                'participant_id': [],
                'session': [],
                'day': [],
                'chordID': [],
                'chord': []
            }

            # add inter-finger lag channels
            for I, i in enumerate(gl.channels['force']):
                for J, j in enumerate(gl.channels['force']):
                    tau_dict[f'{i}-{j}'] = []

            for p in participant_id:
                for day in gl.days:

                    sn = int(''.join([c for c in p if c.isdigit()]))

                    print(f'xcorr - participant_id: {p}, day: {day}')

                    if day == '1' or day == '5':
                        session = 'testing'
                    else:
                        session = 'training'

                    dat = pd.read_csv(
                        os.path.join(gl.baseDir, experiment, session, f'day{day}', f"{experiment}_{sn}.dat"),
                        sep="\t")

                    force = Force(experiment, p, session, day)
                    force_dict = force.load_pkl()
                    pass
                    _, tau, _ = force.crosscorrelation()
                    for tr, t in enumerate(tau):

                        if tr == 0 or dat.iloc[tr - 1].TN == 50 or dat.iloc[tr].chordID != dat.iloc[tr - 1].chordID:
                            rep = 1
                        else:
                            rep += 1
                        tau_dict['repetition'].append(rep)

                        tau_dict['tau'].append(t)
                        tau_dict['experiment'].append(experiment)
                        tau_dict['participant_id'].append(p)
                        tau_dict['session'].append(session)
                        tau_dict['day'].append(day)
                        chordID = force_dict['chordID'][tr]
                        tau_dict['chordID'].append(chordID)
                        tau_dict['chord'].append('trained' if force_dict['chordID'][tr] in force.trained
                                                 else 'untrained')

                        chordID_str = str(chordID.astype(int))
                        fingers = np.array([f != '9' for f in chordID_str])

                        if t is not None:
                            np.fill_diagonal(t, np.nan)
                            t[fingers == False, :] = np.nan
                            t[:, fingers == False] = np.nan
                            off_diag = np.triu(t)
                            off_diag[off_diag == 0] = np.nan
                            tau_dict['tauAbs'].append(np.nanmean(np.abs(t)))
                            tau_dict['tauRel'].append(np.nanmean(off_diag))
                            for I, i in enumerate(gl.channels['force']):
                                for J, j in enumerate(gl.channels['force']):
                                    tau_dict[f'{i}-{j}'].append(t[I, J])
                        else:
                            tau_dict['tauAbs'].append(None)
                            tau_dict['tauRel'].append(None)
                            for I, i in enumerate(gl.channels['force']):
                                for J, j in enumerate(gl.channels['force']):
                                    tau_dict[f'{i}-{j}'].append(None)

            with open(os.path.join(gl.baseDir, experiment, f'tau.pkl'), "wb") as file:
                pickle.dump(tau_dict, file)

            df_tau = pd.DataFrame(tau_dict)
            df_tau.to_csv(os.path.join(gl.baseDir, experiment, f'tau.tsv'), index=False, sep='\t')

            return tau_dict, df_tau
        # endregion

        # region ORDER:rank_corr
        case 'ORDER:rank_corr':

            import warnings

            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, f'metrics.tsv'), sep='\t')

            rank_corr = {
                'onset': [],
                'exit': [],
                'dforce': [],
                'entry': [],
                'participant_id': [],
                'repetition': [],
                'day': [],
                'chordID': [],
                'chord': []
            }

            for p in participant_id:
                for day in metrics['day'].unique():
                    for chordID in metrics['chordID'].unique():
                        for repetition in metrics['repetition'].unique():

                            print(f'order_corr - participant_id: {p}, day: {day}, chordID: {chordID}, repetition: {repetition}')

                            metrics_tmp = metrics[(metrics['chordID'] == chordID) &
                                                  (metrics['day'] == day) &
                                                  (metrics['participant_id'] == p) &
                                                  (metrics['repetition'] == repetition) &
                                                  (metrics['trialPoint'] == 1)].reset_index()
                            if len(metrics_tmp) > 0:
                                rank_corr_onset_tmp, _ = spearmanr(metrics_tmp[['thumb_onset_order',
                                                       'index_onset_order',
                                                       'middle_onset_order',
                                                       'ring_onset_order',
                                                       'pinkie_onset_order']].to_numpy(),
                                                                   nan_policy='omit', axis=1)

                                rank_corr_dforce_tmp, _ = spearmanr(metrics_tmp[['thumb_dforce_order',
                                                       'index_dforce_order',
                                                       'middle_dforce_order',
                                                       'ring_dforce_order',
                                                       'pinkie_dforce_order']].to_numpy(),
                                                                   nan_policy='omit', axis=1)

                                rank_corr_exit_tmp, _ = spearmanr(metrics_tmp[['thumb_exit_order',
                                                       'index_exit_order',
                                                       'middle_exit_order',
                                                       'ring_exit_order',
                                                       'pinkie_exit_order']].to_numpy(),
                                                                   nan_policy='omit', axis=1)

                                rank_corr_entry_tmp, _ = spearmanr(metrics_tmp[['thumb_entry_order',
                                                       'index_entry_order',
                                                       'middle_entry_order',
                                                       'ring_entry_order',
                                                       'pinkie_entry_order']].to_numpy(),
                                                                   nan_policy='omit', axis=1)

                            else:
                                rank_corr_onset_tmp = np.nan
                                rank_corr_dforce_tmp = np.nan
                                rank_corr_exit_tmp = np.nan
                                rank_corr_entry_tmp = np.nan

                            rank_corr['onset'].append(np.mean(rank_corr_onset_tmp))
                            rank_corr['dforce'].append(np.mean(rank_corr_dforce_tmp))
                            rank_corr['exit'].append(np.mean(rank_corr_exit_tmp))
                            rank_corr['entry'].append(np.mean(rank_corr_entry_tmp))
                            rank_corr['participant_id'].append(p)
                            rank_corr['day'].append(day)
                            rank_corr['chordID'].append(chordID)
                            rank_corr['repetition'].append(repetition)

                            chord = 'trained' if (
                                    str(chordID) in pinfo[pinfo['participant_id'] == p]['trained'].to_numpy()[0].split(
                                '.')) \
                                else 'untrained'

                            rank_corr['chord'].append(chord)

            rank_corr = pd.DataFrame(rank_corr)
            rank_corr.to_csv(os.path.join(gl.baseDir, experiment, f'rank_corr.tsv'), sep='\t', index=False)

            return rank_corr
         # endregion

        # region ORDER:left2right
        case 'ORDER:left2right':

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')

            x = np.arange(1, 5)

            slope = {
                'exit': [],
                'entry': [],
                'participant_id': [],
                'repetition': [],
                'day': [],
                'chordID': [],
                'chord': []
            }

            for i, rowi in metrics.iterrows():
                if rowi['trialPoint'] == 1:
                    y_exit = rowi[['thumb_exit_order',
                                   'index_exit_order',
                                   'middle_exit_order',
                                   'ring_exit_order',
                                   'pinkie_exit_order', ]].astype(float)
                    y_entry = rowi[['thumb_entry_order',
                                    'index_entry_order',
                                    'middle_entry_order',
                                    'ring_entry_order',
                                    'pinkie_entry_order', ]].astype(float)

                    chordID = rowi['chordID']

                    for k, char in enumerate(str(chordID)):
                        if char == '9':
                            y_exit[k] = np.nan
                            y_entry[k] = np.nan

                    y_exit = y_exit[~np.isnan(y_exit)]
                    y_entry = y_entry[~np.isnan(y_entry)]

                    slope_exit_tmp, _ = np.polyfit(x, y_exit, 1)
                    slope_entry_tmp, _ = np.polyfit(x, y_entry, 1)

                    slope['exit'].append(slope_exit_tmp)
                    slope['entry'].append(slope_entry_tmp)
                    slope['participant_id'].append(rowi['participant_id'])
                    slope['day'].append(rowi['day'])
                    slope['chordID'].append(chordID)
                    slope['chord'].append(rowi['chord'])
                    slope['repetition'].append(rowi['repetition'])


                else:

                    slope['exit'].append(None)
                    slope['entry'].append(None)
                    slope['participant_id'].append(rowi['participant_id'])
                    slope['day'].append(rowi['day'])
                    slope['chordID'].append(chordID)
                    slope['chord'].append(rowi['chord'])
                    slope['repetition'].append(rowi['repetition'])

            slope = pd.DataFrame(slope)
            slope.to_csv(os.path.join(gl.baseDir, experiment, 'slope.tsv'), sep='\t')

            return slope

        # endregion

        # region ORDER:frequency
        case 'ORDER:frequency':

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')

            force_avg = {str(key): [] for key in gl.chordID}

            fingers = gl.channels['force']

            order = {
                'day': [],
                'participant_id': [],
                'chordID': [],
                'chord': [],
                'order': []
            }

            for day in metrics['day'].unique():
                for p in participant_id:

                    if chord is None:
                        chords = [str(ch) for ch in gl.chordID]
                    else:
                        chords = pinfo[pinfo['participant_id'] == p][chord].iloc[0].split('.')

                    for ch in chords:

                        metrics_tmp = metrics[(metrics['participant_id'] == p) &
                                              (metrics['day'] == int(day)) &
                                              (metrics['chordID'] == int(ch)) &
                                              metrics['trialPoint'] > 0].reset_index()

                        rank = metrics_tmp[
                            ['thumb_exit_order',
                             'index_exit_order',
                             'middle_exit_order',
                             'ring_exit_order',
                             'pinkie_exit_order']
                        ]

                        for i, rowi in rank.iterrows():
                            fingers_tmp = [fingers[k] for k, char in enumerate(ch) if char != '9']
                            order_tmp = [fingers_tmp[int(pos)] for pos in rowi if ~np.isnan(pos)]
                            order_tmp = '.'.join(order_tmp)

                            order['order'].append(order_tmp)
                            order['participant_id'].append(p)
                            order['chordID'].append(ch)
                            order['day'].append(day)
                            order['chord'].append('trained' if ch in
                                                               pinfo[pinfo['participant_id'] == p]['trained'].iloc[
                                                                   0].split('.')
                                                  else 'untrained')

            order = pd.DataFrame(order)
            order.to_csv(os.path.join(gl.baseDir, experiment, 'order.tsv'), sep='\t')

            return order
        # endregion

        # region ORDER:sliding_window
        case 'ORDER:sliding_window':

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')

            if day is None:
                day = '5'

            win = .8

            scaler = MinMaxScaler()

            rank_corr = {
                'rho': [],
                'rhod': [],
                'rhod_norm': [],
                'chordID': [],
                'chord': [],
                'day': [],
                'participant_id': [],
            }

            for P, p in enumerate(participant_id):

                metrics_tmp = metrics[(metrics['participant_id'] == p) &
                                      (metrics['day'] == int(day))]

                if day == '1' or day == '5':
                    session = 'testing'
                    # chords = gl.chordID
                else:
                    session = 'training'

                force = Force(experiment, p, session, day)
                force_dict = force.load_pkl()

                F = list()
                dF = list()
                dF_norm = list()
                chordID = list()
                repetition = list()
                for f, force_tmp in enumerate(force_dict['force_filt10Hz']):
                    if force_tmp is not None:

                        # force
                        F.append(force_tmp[:int(win * gl.fsample['force'])].swapaxes(0, 1))

                        # derivative
                        df_tmp = np.gradient(force_tmp, 1 / gl.fsample['force'], axis=0)
                        df_tmp_norm = scaler.fit_transform(np.abs(df_tmp[:int(win * gl.fsample['force'])]))
                        dF.append(df_tmp[:int(win * gl.fsample['force'])].swapaxes(0, 1))
                        dF_norm.append(df_tmp_norm[:int(win * gl.fsample['force'])].swapaxes(0, 1))

                        chordID.append(metrics_tmp['chordID'].iloc[f].astype(str))
                        repetition.append(metrics_tmp['repetition'].iloc[f].astype(str))

                F = np.abs(np.stack(F))
                dF = np.stack(dF)
                dF_norm = np.stack(dF_norm)
                chordID = pd.Series(chordID)

                for ch in chordID.unique():

                    keep = np.ones(5).astype(bool)
                    for i, char in enumerate(str(ch)):
                        if char == '9':
                            keep[i] = False

                    chord = 'trained' if ch in pinfo[pinfo['participant_id'] == p]['trained'].iloc[0].split('.') else 'untrained'

                    print(f'participant_id: {p}, chord: {ch}')

                    rho = np.zeros(F.shape[-1])
                    rhod = np.zeros(F.shape[-1])
                    rhod_norm = np.zeros(F.shape[-1])
                    for t in range(F.shape[-1]):

                        # force
                        F_tmp = F[chordID == ch, :, t][:, keep]
                        order_tmp = np.argsort(F_tmp, axis=1)
                        rho_tmp, _ = spearmanr(order_tmp, axis=1)
                        rho[t] = rho_tmp.mean()

                        # derivative
                        dF_tmp = dF[chordID == ch, :, t][:, keep]
                        order_tmp = np.argsort(dF_tmp, axis=1)
                        rho_tmp, _ = spearmanr(order_tmp, axis=1)
                        rhod[t] = rho_tmp.mean()

                        # derivative norm
                        dF_tmp_norm = dF_norm[chordID == ch, :, t][:, keep]
                        order_tmp = np.argsort(dF_tmp_norm, axis=1)
                        rho_tmp, _ = spearmanr(order_tmp, axis=1)
                        rhod_norm[t] = rho_tmp.mean()

                    rank_corr['rho'].append(rho)
                    rank_corr['rhod'].append(rhod)
                    rank_corr['rhod_norm'].append(rhod_norm)
                    rank_corr['chord'].append(chord)
                    rank_corr['participant_id'].append(p)
                    rank_corr['day'].append(day)
                    rank_corr['chordID'].append(ch)

                    pass

            with open(os.path.join(gl.baseDir, experiment, 'rank_corr_timepoints.pkl'), 'wb') as f:
                pickle.dump(rank_corr, f)

            return rank_corr

        # endregion

        # region ORDER:variance_decomposition
        case 'ORDER:variance_decomposition':

            metrics = pd.read_csv(os.path.join(gl.baseDir, experiment, 'metrics.tsv'), sep='\t')

            # Initialize the variance decomposition dictionary
            var_dec = {
                'v_g_exit': [],
                'v_gs_exit': [],
                'v_gse_exit': [],
                'v_s_exit': [],
                'v_e_exit': [],
                'v_g_entry': [],
                'v_gs_entry': [],
                'v_gse_entry': [],
                'v_s_entry': [],
                'v_e_entry': [],
                'chordID': [],
                'chord': [],
                'day': []
            }

            # Precompute minimum trials per chordID
            trial_counts = metrics.groupby(['participant_id', 'chordID']).size().reset_index(name='trial_count')
            min_trials_per_chordID = trial_counts.groupby('chordID')['trial_count'].min()

            # Group by chord and day
            grouped_metrics = metrics.groupby(['chord', 'day'])

            # Iterate through each chord and day combination
            for (chord, day), metricsD in grouped_metrics:
                chordIDs = metricsD['chordID'].unique()

                for chordID in chordIDs:

                    print(chordID, day)

                    metrics_tmp = metricsD[metricsD['chordID'] == chordID]
                    min_trials = min_trials_per_chordID.loc[chordID]

                    # Ensure that we do not sample more than the available number of trials
                    tau_tmp_counts = metrics_tmp.groupby('participant_id').size()
                    max_available_trials = tau_tmp_counts.min()

                    if min_trials > max_available_trials:
                        min_trials = max_available_trials

                    # Sample the minimum number of trials for each participant
                    sampled_metrics_tmp = metrics_tmp.groupby('participant_id', group_keys=False).sample(n=min_trials,
                                                                                                         random_state=42)

                    part_vec = (sampled_metrics_tmp.groupby('participant_id').cumcount() // 5 + 1).to_numpy()
                    subj_vec = sampled_metrics_tmp['participant_id'].to_numpy()

                    # Exit
                    Y_exit = sampled_metrics_tmp[['thumb_exit_order',
                                                  'index_exit_order',
                                                  'middle_exit_order',
                                                  'ring_exit_order',
                                                  'pinkie_exit_order']]

                    for k, char in enumerate(str(chordID)):
                        if char == '9':
                            Y_exit = np.delete(Y_exit, k, axis=1)

                    # Perform variance decomposition
                    v_g, v_gs, v_gse, _ = reliability_var(Y_exit, subj_vec, part_vec, centered=False)

                    var_dec['v_g_exit'].append(v_g / v_gse)
                    var_dec['v_gs_exit'].append(v_gs)
                    var_dec['v_gse_exit'].append(v_gse)
                    var_dec['v_s_exit'].append((v_gs - v_g) / v_gse)
                    var_dec['v_e_exit'].append((v_gse - v_gs) / v_gse)

                    # Entry
                    Y_entry = sampled_metrics_tmp[['thumb_entry_order',
                                                   'index_entry_order',
                                                   'middle_entry_order',
                                                   'ring_entry_order',
                                                   'pinkie_entry_order']].to_numpy()

                    for k, char in enumerate(str(chordID)):
                        if char == '9':
                            Y_entry = np.delete(Y_entry, k, axis=1)

                    # Perform variance decomposition
                    v_g, v_gs, v_gse, _ = reliability_var(Y_entry, subj_vec, part_vec, centered=False)

                    var_dec['v_g_entry'].append(v_g / v_gse)
                    var_dec['v_gs_entry'].append(v_gs)
                    var_dec['v_gse_entry'].append(v_gse)
                    var_dec['v_s_entry'].append((v_gs - v_g) / v_gse)
                    var_dec['v_e_entry'].append((v_gse - v_gs) / v_gse)
                    var_dec['chordID'].append(chordID)
                    var_dec['chord'].append(chord)
                    var_dec['day'].append(day)

            var_dec = pd.DataFrame(var_dec)
            var_dec.to_csv(os.path.join(gl.baseDir, experiment, 'var_dec_order.tsv'), sep='\t', index=False)

            return var_dec

        # endregion

        # region EMG:nnmf_tmp
        case 'EMG:nnmf_tmp':

            M = pd.read_csv(os.path.join(gl.baseDir, experiment, 'chords', 'M.tsv'), sep='\t')[
                gl.channels['emg'] + ['sn', 'chordID']]

            chords = {
                'W': [],
                'H': [],
                'r2': [],
                'k': []
            }

            natural = {
                'W': [],
                'H': [],
                'r2': [],
                'k': []
            }

            scaler = MinMaxScaler()

            for p in participant_id:
                sn = int(''.join([c for c in p if c.isdigit()]))
                M_tmp = M[M['sn'] == sn].groupby(['chordID']).mean().reset_index()[gl.channels['emg']].to_numpy()
                M_tmp = scaler.fit_transform(M_tmp)
                W_tmp, H_tmp, r2_tmp = iterative_nnmf(M_tmp, thresh=0.1)

                k_tmp = H_tmp.shape[0]

                chords['W'].append(W_tmp)
                chords['H'].append(H_tmp)
                chords['r2'].append(r2_tmp)
                chords['k'].append(k_tmp)

                M_nat = load_nat_emg(os.path.join(gl.baseDir, experiment, 'natural',
                                                  f'natChord_{p}_emg_natural_whole_sampled.mat'))

                W_nat_tmp, H_nat_tmp, r2_nat_tmp, k_nat_tmp = [], [], [], []
                for m, M_nat_tmp in enumerate(M_nat):
                    M_nat_tmp = scaler.fit_transform(M_nat_tmp)
                    norm = np.linalg.norm(M_nat_tmp, axis=1)
                    M_nat_tmp = M_nat_tmp[norm > 2 * norm.mean()]

                    W, H, r2 = iterative_nnmf(M_nat_tmp, thresh=0.1)

                    W_nat_tmp.append(W)
                    H_nat_tmp.append(H)
                    r2_nat_tmp.append(r2)
                    k_nat_tmp.append(H.shape[0])

                natural['W'].append(W_nat_tmp)
                natural['H'].append(H_nat_tmp)
                natural['r2'].append(np.median(np.array(r2_nat_tmp)))
                natural['k'].append(np.median(np.array(k_nat_tmp)))

            return chords, natural
        # endregion

        # region RECONSTRUCT:emg
        case 'RECONSTRUCT:emg':

            M = pd.read_csv(os.path.join(gl.baseDir, experiment, 'chords', 'M.tsv'), sep='\t')[gl.channels['emg'] +
                                                                                               ['sn', 'chordID']]
            M_avg = M.groupby('chordID').mean().reset_index().sort_values(by='chordID')[gl.channels['emg']].to_numpy()

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

                M_tmp = M[M['sn'] == sn].groupby(['chordID']).mean().reset_index().sort_values(by='chordID')[
                    gl.channels['emg']].to_numpy()
                M_tmp = scaler.fit_transform(M_tmp)
                W_tmp, H_tmp, r2_tmp = iterative_nnmf(M_tmp, thresh=0.1)

                M_nat = load_nat_emg(os.path.join(gl.baseDir, experiment, 'natural',
                                                  f'natChord_{p}_emg_natural_whole_sampled.mat'))

                # noise ceiling chords
                B = np.zeros((M_avg.shape[1], M_avg.shape[1]))
                for m in range(M_tmp.shape[1]):
                    B[:, m], _ = nnls(M_tmp, M_avg[:, m])

                recon_dict['nc_chords'].append(calc_r2(M_avg, np.dot(M_tmp, B)))

                matches = [[] for _ in range(len(M_nat))]
                for m, M_nat_tmp in enumerate(M_nat):

                    print(f'RECONSTRUCT:emg - participant_id: {p}, partition: {m}')

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
                            M_nat_j = scaler.fit_transform(M_nat_j)
                            norm = np.linalg.norm(M_nat_j, axis=1)
                            M_nat_j = M_nat_j[norm > norm.mean()]

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
                    for i in range(M_nat_tmp.shape[0]):
                        W_c[i], _ = nnls(M_tmp.T, M_nat_tmp[i])

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

            tau = pd.read_csv(os.path.join(gl.baseDir, experiment, 'tau.tsv'), sep='\t')
            chordIDs = tau['chordID'].unique()

            noise_ceiling = {
                'corr': [],
                'day': [],
                'participant_id': [],
                'chordID': [],
                'chord': []
            }

            for day in range(1, 6):
                for p in participant_id:
                    for c, chordID in enumerate(chordIDs):

                        tau_tmp = tau[(tau['chordID'] == chordID) & (tau['participant_id'] == p) & (
                                tau['day'] == day)].reset_index()

                        if len(tau_tmp) > 0:
                            chord = tau_tmp['chord']
                            tau_tmp = tau_tmp.drop(['index', 'tau', 'tauAbs', 'experiment', 'participant_id', 'session',
                                                    'day', 'chordID', 'chord'], axis=1)

                            finger_combinations = tau_tmp.columns
                            unique_combinations = set(
                                '-'.join(sorted(combo.split('-'))) for combo in finger_combinations
                            )

                            tau_tmp = tau_tmp[list(unique_combinations)]
                            # tau_tmp['chord'] = pd.DataFrame(chord)
                            tau_tmp = tau_tmp.dropna(axis=1, how='all').dropna(axis=0, how='all')

                            tau_avg = tau_tmp.mean(axis=0)

                            noise_ceiling['corr'].append(tau_tmp.corrwith(tau_avg, axis=1).mean())
                            noise_ceiling['day'].append(day)
                            noise_ceiling['participant_id'].append(p)
                            noise_ceiling['chordID'].append(chordID.astype(int))
                            noise_ceiling['chord'].append(chord[0])

            noise_ceiling = pd.DataFrame(noise_ceiling)
            noise_ceiling.to_csv(os.path.join(gl.baseDir, experiment, 'corr_noise_ceiling.tsv'), sep='\t')

            return noise_ceiling
        # endregion

        # region XCORR:corr
        case 'XCORR:corr':

            tau = pd.read_csv(os.path.join(gl.baseDir, experiment, 'tau.tsv'), sep='\t')
            chordIDs = tau['chordID'].unique()

            corr = {
                'participant_id': [],
                'day': [],
                'chordID': [],
                'chord': [],
                'corr': [],
            }
            for p in participant_id:
                for day in range(1, 6):
                    for c, chordID in enumerate(chordIDs):

                        tau_tmp = tau[(tau['chordID'] == chordID) & (tau['participant_id'] == p) & (tau['day'] == day)]

                        if tau_tmp['session'].unique() == 'testing':
                            chord = tau_tmp['chord'].unique()[0]
                        else:
                            chord = 'trained'

                        tau_tmp = tau_tmp.iloc[:, 8:]

                        finger_combinations = tau_tmp.columns
                        unique_combinations = set(
                            '-'.join(sorted(combo.split('-'))) for combo in finger_combinations
                        )

                        tau_tmp = tau_tmp[list(unique_combinations)].dropna(axis=0, how='all').dropna(axis=1, how='all')

                        corr['participant_id'].append(p)
                        corr['chordID'].append(chordID.astype(int))
                        corr['day'].append(day)
                        corr['chord'].append(chord)
                        corr['corr'].append(np.nanmean(tau_tmp.transpose().corr().to_numpy()))

                        pass

            corr_df = pd.DataFrame(corr)
            corr_df.to_csv(os.path.join(gl.baseDir, experiment, 'corr.tsv'), sep='\t', index=False)

            return corr_df
        # endregion

        # region XCORR:variance_decomposition
        case 'XCORR:variance_decomposition':
            # Load the data
            tau = pd.read_csv(os.path.join(gl.baseDir, experiment, 'tau.tsv'), sep='\t')

            # Initialize the variance decomposition dictionary
            var_dec = {
                'v_g': [],
                'v_gs': [],
                'v_gse': [],
                'v_s': [],
                'v_e': [],
                'chordID': [],
                'chord': [],
                'day': []
            }

            # Precompute minimum trials per chordID
            trial_counts = tau.groupby(['participant_id', 'chordID']).size().reset_index(name='trial_count')
            min_trials_per_chordID = trial_counts.groupby('chordID')['trial_count'].min()

            # Group by chord and day
            grouped_tau = tau.groupby(['chord', 'day'])

            # Iterate through each chord and day combination
            for (chord, day), tauD in grouped_tau:
                chordIDs = tauD['chordID'].unique()

                for chordID in chordIDs:
                    tau_tmp = tauD[tauD['chordID'] == chordID]
                    min_trials = min_trials_per_chordID.loc[chordID]

                    # Ensure that we do not sample more than the available number of trials
                    tau_tmp_counts = tau_tmp.groupby('participant_id').size()
                    max_available_trials = tau_tmp_counts.min()

                    if min_trials > max_available_trials:
                        min_trials = max_available_trials

                    # Sample the minimum number of trials for each participant
                    sampled_tau_tmp = tau_tmp.groupby('participant_id', group_keys=False).sample(n=min_trials,
                                                                                                 random_state=42)

                    part_vec = (sampled_tau_tmp.groupby('participant_id').cumcount() // 5 + 1).to_numpy()
                    subj_vec = sampled_tau_tmp['participant_id'].to_numpy()

                    # Keep only relevant columns (finger combinations)
                    sampled_tau_tmp = sampled_tau_tmp.iloc[:, 8:]
                    unique_combinations = sampled_tau_tmp.columns.str.split('-').map(sorted).str.join('-').unique()

                    Y = sampled_tau_tmp[list(unique_combinations)].dropna(axis=1, how='all').to_numpy()

                    # Perform variance decomposition
                    v_g, v_gs, v_gse, _ = reliability_var(Y, subj_vec, part_vec, centered=True)

                    # Store the results
                    var_dec['v_g'].append(v_g / v_gse)
                    var_dec['v_gs'].append(v_gs)
                    var_dec['v_gse'].append(v_gse)
                    var_dec['v_s'].append((v_gs - v_g) / v_gse)
                    var_dec['v_e'].append((v_gse - v_gs) / v_gse)
                    var_dec['chordID'].append(chordID)
                    var_dec['chord'].append(chord)
                    var_dec['day'].append(day)

            var_dec = pd.DataFrame(var_dec)
            var_dec.to_csv(os.path.join(gl.baseDir, experiment, 'var_dec.tsv'), sep='\t')

            return var_dec

        # endregion

        # region PLOT:force_in_trial
        case 'PLOT:force_in_trial':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            force = Force(experiment, participant_id, session=session, day=day)
            force_dict = force.load_pkl()
            force_trial = force_dict['force'][ntrial] * np.array(
                [1, 1, 1, 1.5, 1.5])  # specify force gain for visualization
            force_trial = lowpass_butter(force_trial.T, 30, gl.fsample['force']).T
            chordID = int(force_dict['chordID'][ntrial])
            # session = force_dict['session'][ntrial]
            day = force_dict['day'][ntrial]

            tAx = np.linspace(0, force_trial.shape[0] / gl.fsample['force'], force_trial.shape[0])

            axs.plot(tAx, force_trial, lw=linewidth)

            axs.axhline(0, color='k', lw=.8, ls='-')
            # axs.axhline(-1.2, color='k', lw=.8, ls='--')
            # axs.axhline(1.2, color='k', lw=.8, ls='--')
            axs.axhspan(-1.2, 1.2, color='grey', alpha=0.3, lw=0)
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

        # region PLOT:recon_emg
        case 'PLOT:recon_emg':
            if fig is None or axs is None:
                fig, axs = plt.subplots()

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

            pval = {
                'pval_chord2nat_shuffle': [],
                'pval_nat2chord_shuffle': [],
                'pval_chord2nat_ceiling': [],
                'pval_nat2chord_ceiling': []
            }

            _, pval['pval_chord2nat_shuffle'] = ttest_1samp(df_r2['r2_chord2nat'],
                                                            df_shuffle['r2_chord2nat_shuffle'].mean(),
                                                            alternative='greater')
            _, pval['pval_nat2chord_shuffle'] = ttest_1samp(df_r2['r2_nat2chord'],
                                                            df_shuffle['r2_nat2chord_shuffle'].mean(),
                                                            alternative='greater')
            _, pval['pval_chord2nat_ceiling'] = ttest_1samp(df_r2['r2_chord2nat'],
                                                            np.array(recon_dict['nc_natural']).mean(),
                                                            alternative='greater')
            _, pval['pval_nat2chord_ceiling'] = ttest_1samp(df_r2['r2_nat2chord'],
                                                            np.array(recon_dict['nc_chords']).mean(),
                                                            alternative='greater')

            sns.boxplot(ax=axs, x='reconstruction', y='R²', data=df_r2_melt, width=width, linewidth=linewidth,
                        linecolor=linecolor, showfliers=showfliers, color=color, palette=palette)

            return fig, axs, pval
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


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None, choices=[
        'FORCE:preprocessing',  # ok
        'FORCE:average',
        'FORCE:derivative',
        'FORCE:fit_sinusoid',
        'XCORR:tau',  # ok
        'NOISE_CEILING:tau',  # ok
        'EMG:nnmf',
        'EMG:df2chord',
        'EMG:merge_blocks',
        'EMG:csv2df',
        'EMG:df2mep',
        'EMG:distance',
        'MEP:nnmf',
        'EMG:mep2amp',
        'RECONSTRUCT:force',
        'RECONSTRUCT:emg',
        'NATURAL:extract_patterns_from_peaks',
        'XCORR:corr',
        'XCORR:variance_decomposition',
        'ORDER:rank_corr',
        'ORDER:left2right',
        'ORDER:sliding_window',
        'ORDER:frequency',
        'ORDER:variance_decomposition',
        'PLOT:force_in_trial',  # ok
        'PLOT:xcorr_chord',  # ok
        'PLOT:recon_emg',
    ])
    parser.add_argument('--experiment', default='efc2', help='')
    parser.add_argument('--participant_id', nargs='+', default=None, help='')
    parser.add_argument('--session', default=None, help='',
                        choices=['training', 'testing', 'emgTMS', 'emgNatural', 'emgChords'])
    parser.add_argument('--day', default=gl.days, help='')
    parser.add_argument('--ntrial', default=None, help='')
    parser.add_argument('--metric', default=None, choices=['MD', 'RT', 'ET'], help='')
    parser.add_argument('--chordID', default=None, help='')
    parser.add_argument('--chord', default=None, help='', choices=['trained', 'untrained'])
    parser.add_argument('--dataset', default=None, help='', choices=['natural', 'chords'])
    parser.add_argument('--fname', default=None, help='')
    parser.add_argument('--n_jobs', default=None, help='')

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
    dataset = args.dataset
    fname = args.fname
    n_jobs = args.n_jobs

    if participant_id is None:
        participant_id = gl.participants[experiment]

    main(what, experiment=experiment, participant_id=participant_id, session=session, day=day,
         ntrial=ntrial, chordID=chordID, chord=chord, fname=fname, n_jobs=n_jobs)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    plt.show()
