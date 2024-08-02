import argparse
import json
import os

import scipy
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import pickle
import json
import globals as gl
import argcomplete
import numpy as np

import seaborn as sns

from force import Force
from nnmf import iterative_nnmf, optimize_H, calc_reconerr, assert_selected_rows_belong, get_emg_chords
from variance_decomposition import reliability_var


def main(what, experiment=None, participant_id=None, session=None, day=None, ntrial=None, fig=None, axs=None):
    match what:
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

        case 'FORCE:xcorr':

            tau_dict = {
                'tau': [],
                'experiment': [],
                'participant_id': [],
                'session': [],
                'day': [],
                'chordID': []
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
                    tau_dict['tau'].append(np.concatenate(tau))
                    tau_dict['experiment'].append(experiment)
                    tau_dict['participant_id'].append(p)
                    tau_dict['session'].append(session)
                    tau_dict['day'].append(day)
                    tau_dict['chordID'].append(force_dict['chordID'])

            with open(os.path.join(gl.baseDir, experiment, f'{experiment}_tau.pkl'), "wb") as file:
                pickle.dump(tau_dict, file)

            return tau_dict

        case 'PLOT:xcorr':

            pass

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

        case 'EMG:recon_chord2chord':

            pass

            # path = os.path.join(gl.baseDir, experiment, gl.chordDir)
            #
            # scaler = MinMaxScaler()
            #
            # for p in participant_id:
            #     M, chords = get_emg_chords(experiment, p)
            #     W, H, r2 = iterative_nnmf(M)

        case 'EMG:recon_chord2nat':

            pass

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

        case 'PLOT:force_in_trial':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            force = Force(experiment, participant_id, session=session, day=day)
            force_dict = force.load_pkl()
            force_trial = force_dict['force'][ntrial] * np.array([1, 1, 1, 1.5, 1.5])  # specify force gain for visualization
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

            pass

        case 'PLOT:xcorr_in_trial':

            if fig is None or axs is None:
                fig, axs = plt.subplots(5, 5, sharex=True, sharey=True)

            force = Force(experiment=experiment, participant_id=participant_id, session=session, day=day)
            force_dict = force.load_pkl()
            chordID = int(force_dict['chordID'][ntrial])
            # session = force_dict['session'][ntrial]
            day = force_dict['day'][ntrial]
            xcorr, _, lags = force.crosscorrelation()
            xcorr_trial = xcorr[ntrial]
            lags_trial = lags[ntrial]

            for i in range(xcorr_trial.shape[0]):
                for j in range(xcorr_trial.shape[1]):

                    axs[i, j].plot(lags_trial, np.abs(xcorr_trial[i, j]))
                    axs[i, j].axvline(0, color='k', ls='-', lw=.8)
                    axs[i, j].axvline(lags_trial[np.argmax(np.abs(xcorr_trial[i, j]))], color='k', ls='--', lw=.8)

            fig.supxlabel('lag (s)')
            fig.supylabel('correlation')
            fig.suptitle(f'{chordID}, day{day}, {"trained" if chordID in force.trained else "untrained"}')

            plt.show()








if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None, choices=[
        'FORCE:preprocessing',
        'FORCE:variance_decomposition',
        'FORCE:xcorr',
        'EMG:recon_chord2nat',
        'EMG:recon_chord2chord',
        'PLOT:variance_decomposition',
        'PLOT:force_in_trial',
        'PLOT:xcorr_in_trial'
    ])
    parser.add_argument('--experiment', default='efc2', help='')
    parser.add_argument('--participant_id', nargs='+', default=None, help='')
    parser.add_argument('--session', default=None, help='')
    parser.add_argument('--day', default=None, help='')
    parser.add_argument('--ntrial', default=None, help='')

    args = parser.parse_args()

    what = args.what
    experiment = args.experiment
    participant_id = args.participant_id
    session = args.session
    day = args.day
    ntrial = int(args.ntrial) if args.ntrial is not None else None

    if participant_id is None:
        participant_id = gl.participants[experiment]

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    main(what, experiment=experiment, participant_id=participant_id, session=session, day=day, ntrial=ntrial)
