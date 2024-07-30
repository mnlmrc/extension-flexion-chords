import argparse
import os

import scipy

from sklearn.preprocessing import MinMaxScaler

import pandas as pd

import globals as gl
import argcomplete
import numpy as np

from force import Force
from nnmf import iterative_nnmf, optimize_H, calc_reconerr, assert_selected_rows_belong, get_emg_chords
from variance_decomposition import reliability_var


def main(what, experiment, participant_id, varargin=None):
    match what:
        case 'FORCE:preprocessing':

            df = pd.DataFrame()
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
                    df_tmp = force.preprocessing()
                    df = pd.concat([df, df_tmp])

            df.to_csv(os.path.join(gl.baseDir, experiment, 'results.csv'))

            return df

        case 'FORCE:variance_decomposition':

            group_cols = ['subNum', 'chord', 'day', 'chordID', 'BN', 'TN']
            num_cols = [f'dist{i}' for i in range(5)]

            df = pd.read_csv(os.path.join(gl.baseDir, experiment, 'results.csv'))
            df = df[df['chord'] == 'trained']
            df = df.groupby(group_cols)[num_cols].mean().reset_index()

            v_g, v_gs, v_gse = [], [], []
            for day in gl.days:
                df_tmp = df[df['day'] == int(day)]
                Y = df_tmp[num_cols].values
                v_g_tmp, v_gs_tmp, v_gse_tmp = reliability_var(Y,
                                                               df_tmp['subNum'],
                                                               df_tmp['BN'],
                                                               cond_vec=df_tmp['chordID'],
                                                               centered=True)
                v_g.append(v_g_tmp)
                v_gs.append(v_gs_tmp)
                v_gse.append(v_gse_tmp)

            return np.array(v_g), np.array(v_gs), np.array(v_gse)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None, choices=[
        'FORCE:preprocessing',
        'FORCE:variance_decomposition',
        'EMG:recon_chord2nat',
        'EMG:recon_chord2chord',
    ])
    parser.add_argument('--experiment', default='efc2', help='')
    parser.add_argument('--participant_id', nargs='+', default=None, help='')

    args = parser.parse_args()

    what = args.what
    experiment = args.experiment
    participant_id = args.participant_id

    if participant_id is None:
        participant_id = gl.participants[experiment]

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    main(what, experiment, participant_id)
