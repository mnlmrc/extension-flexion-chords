import argparse
import os

import scipy

from sklearn.preprocessing import MinMaxScaler

import pandas as pd

import globals as gl
import argcomplete
import numpy as np

from nnmf import iterative_nnmf


def main(what):
    match what:
        case 'FORCE:mov2npz':
            pass

        case 'EMG:recon_chord2nat':

            path = os.path.join(gl.baseDir, experiment, gl.natDir)

            scaler = MinMaxScaler()

            for p in participant_id:

                sn = int(''.join([c for c in p if c.isdigit()]))
                filename = f'natChord_subj{"{:02}".format(sn)}_emg_natural_whole_sampled.mat'
                mat = scipy.io.loadmat(os.path.join(path, filename))
                emg_nat = [np.array(matrix[0]) for matrix in mat['emg_natural_dist']['dist'][0][0]]

                emg_chords = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.chordDir, 'natChord_chord.tsv'), sep='\t')
                emg_chords = emg_chords[emg_chords['sn'] == sn]

                for M_nat in emg_nat:

                    M_nat = scaler.fit_transform(M_nat)
                    Wnat, Hnat, r2nat, _, _ = iterative_nnmf(M_nat)

                    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None, choices=[
        'EMG:recon_chord2nat',
    ])
    parser.add_argument('--experiment', default='efc1', help='')
    parser.add_argument('--participant_id', nargs='+', default=None, help='')

    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    what = args.what
    experiment = args.experiment
    participant_id = args.participant_id

    if participant_id is None:
        participant_id = gl.participants[experiment]

    # pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    df = main(what)

