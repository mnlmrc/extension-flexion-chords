import os

import pandas as pd
import numpy as np

experiment = 'efc3'
sn = 100
day = 1

nruns = 10

file_path = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/ExtFlexChord/efc3/target/template.tgt'
out_path = '/Volumes/diedrichsen_data$/data/Chord_exp/ExtFlexChord/efc3/target'
data = pd.read_csv(file_path, delimiter='\t')

for n in range(nruns):
    # Shuffle the order of chordID groups first
    unique_chord_ids = data['chordID'].unique()
    np.random.shuffle(unique_chord_ids)

    # Shuffling rows within each 'chordID' group and combining them in a shuffled order
    shuffled_groups = [
        data[data['chordID'] == chord_id].sample(frac=1).reset_index(drop=True)
        for chord_id in unique_chord_ids
    ]
    target = pd.concat(shuffled_groups).reset_index(drop=True)

    # Shuffle 'iti' and 'Trig' columns independently
    target['iti'] = np.random.permutation(target['iti'].values)
    target['Trig'] = np.random.permutation(target['Trig'].values)
    target['stimTrig'] = np.random.permutation(target['stimTrig'].values)

    target.to_csv(os.path.join(out_path, f'{experiment}_{sn}_day{day}_run{n+1}.tsv'), index=False, sep='\t')




