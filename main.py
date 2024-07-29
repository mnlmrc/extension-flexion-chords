import argparse
import os

import scipy

from sklearn.preprocessing import MinMaxScaler

import pandas as pd

import globals as gl
import argcomplete
import numpy as np

from nnmf import iterative_nnmf, optimize_H, calc_reconerr, assert_selected_rows_belong, get_emg_chords


def main(what):
    match what:
        case 'FORCE:mov2npz':
            pass

        case 'EMG:recon_chord2chord':

            path = os.path.join(gl.baseDir, experiment, gl.chordDir)

            scaler = MinMaxScaler()

            for p in participant_id:

                M, chords = get_emg_chords(experiment, p)

                W, H, r2, err, k = iterative_nnmf(M)

                pass












        case 'EMG:recon_chord2nat':

            path = os.path.join(gl.baseDir, experiment, gl.natDir)

            scaler = MinMaxScaler()

            Hp_dict = {key: [] for key in participant_id}
            H_nat_dict = {key: [] for key in participant_id}
            reconerr_dict = {key: [] for key in participant_id}

            for p in participant_id:

                sn = int(''.join([c for c in p if c.isdigit()]))
                filename = f'natChord_subj{"{:02}".format(sn)}_emg_natural_whole_sampled.mat'
                mat = scipy.io.loadmat(os.path.join(path, filename))
                emg_nat = [np.array(matrix[0]) for matrix in mat['emg_natural_dist']['dist'][0][0]]

                M_chords = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.chordDir, 'natChord_chord.tsv'), sep='\t')
                M_chords = M_chords[M_chords['sn'] == sn]
                chords = list(M_chords['chordID'])
                M_chords = M_chords[[f'emg_hold_avg_e{e+1}' for e in range(5)] +
                                                      [f'emg_hold_avg_f{f+1}' for f in range(5)]].to_numpy()
                M_chords = scaler.fit_transform(M_chords)

                for M_nat in emg_nat:

                    M_nat = scaler.fit_transform(M_nat)
                    W_nat, H_nat, r2_nat, _, _ = iterative_nnmf(M_nat)

                    Hp = optimize_H(W_nat, M_nat, M_chords)

                    assert_selected_rows_belong(M_chords, Hp)

                    reconerr = calc_reconerr(W_nat, Hp, M_nat)

                    Hp_dict[p].append(Hp)
                    H_nat_dict[p].append(H_nat)
                    reconerr_dict[p].append(reconerr)

            return Hp_dict, H_nat_dict, reconerr_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None, choices=[
        'EMG:recon_chord2nat',
        'EMG:recon_chord2chord',
    ])
    parser.add_argument('--experiment', default='efc1', help='')
    parser.add_argument('--participant_id', nargs='+', default=None, help='')

    args = parser.parse_args()

    what = args.what
    experiment = args.experiment
    participant_id = args.participant_id

    if participant_id is None:
        participant_id = gl.participants[experiment]

    # pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    Hp_dict, H_nat_dict, reconerr_dict = main(what)

