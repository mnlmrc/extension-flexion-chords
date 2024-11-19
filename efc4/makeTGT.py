import os

import pandas as pd
import numpy as np

experiment = 'efc4'
sn = 100
day = 1

nruns = 10

file_path = 'target/template.tgt'
out_path = 'target/'
data = pd.read_csv(file_path, delimiter='\t')

for n in range(nruns):

    target = pd.DataFrame()
    for mblock in range(3):
        # preserved_columns = data[['startTime', 'endTime']]

        # Shuffle the order of chordID groups first
        unique_chord_ids = data['chordID'].unique()  # data.drop(columns=['startTime', 'endTime'])['chordID'].unique()
        np.random.shuffle(unique_chord_ids)

        # Shuffling rows within each 'chordID' group and combining them in a shuffled order
        shuffled_groups = [
            data[data['chordID'] == chord_id].sample(frac=1).reset_index(drop=True)
            for chord_id in unique_chord_ids
        ]
        target_tmp = pd.concat(shuffled_groups).reset_index(drop=True)
        # target_tmp[['startTime', 'endTime']] = preserved_columns

        target = pd.concat([target, target_tmp]).reset_index(drop=True)

    start = 6000
    step = 6500
    num = target.shape[0]

    startTime = start + step * np.arange(num)
    startTime[16:] = startTime[16:] + 16000
    startTime[32:] = startTime[32:] + 16000
    endTime = np.zeros_like(startTime)
    endTime[-1] = startTime[-1] + target[['planTime', 'execMaxTime', 'feedbackTime', 'iti']].iloc[-1].sum() + 12500

    # # Shuffle 'iti' and 'Trig' columns independently
    # target['iti'] = np.random.permutation(target['iti'].values)
    # target['Trig'] = np.random.permutation(target['Trig'].values)
    # target['stimTrig'] = np.random.permutation(target['stimTrig'].values)
    target['startTime'] = startTime
    target['endTime'] = endTime

    target.to_csv(os.path.join(out_path, f'{experiment}_{sn}_day{day}_run{n+1}.tgt'), index=False, sep='\t')




