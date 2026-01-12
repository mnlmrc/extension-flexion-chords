import os

import pandas as pd
import numpy as np
import globals as gl

experiment = 'efc4'
sn = 105
# day = 2

pinfo = pd.read_table(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
pinfo_row = pinfo[pinfo.sn == sn].reset_index(drop=True)
trained = np.array(pinfo_row['trained'][0].split('.'), dtype='int')

sess_info = pd.read_csv('target/sess_info.tgt', delimiter='\t')

for i, row in sess_info.iterrows():

    day = row['day']
    week = row['week']
    nruns = row['nruns']
    session = row['session']
    miniblocks = row['miniblocks']

    # nruns = 10

    file_path = 'target/template.tgt'
    out_path = 'target/'
    target_template = pd.read_csv(file_path, delimiter='\t')

    for n in range(nruns):

        target = pd.DataFrame()
        for mblock in range(miniblocks):

            # Shuffle the order of chordID groups first
            unique_chord_ids = target_template['chordID'].unique()
            if session == 'training':
                unique_chord_ids = unique_chord_ids[np.isin(unique_chord_ids, trained)]
            np.random.shuffle(unique_chord_ids)

            # Shuffling rows within each 'chordID' group and combining them in a shuffled order
            shuffled_groups = [
                target_template[target_template['chordID'] == chord_id].sample(frac=1).reset_index(drop=True)
                for chord_id in unique_chord_ids
            ]
            target_tmp = pd.concat(shuffled_groups).reset_index(drop=True)
            # target_tmp[['startTime', 'endTime']] = preserved_columns

            target = pd.concat([target, target_tmp]).reset_index(drop=True)

        start = 6000
        step = 6000
        num = target.shape[0]

        startTime = start + step * np.arange(num)
        if session == 'scanning':  # add breaks for baseline recordings if it is a scanning session
            startTime[16:] = startTime[16:] + 16000
            startTime[32:] = startTime[32:] + 16000
        endTime = np.zeros_like(startTime)
        if session == 'scanning':
            endTime[-1] = startTime[-1] + target[['planTime', 'execMaxTime', 'feedbackTime', 'iti']].iloc[-1].sum() + 10500

        target['startTime'] = startTime
        target['endTime'] = endTime

        target['session'] = session
        target['day'] = day
        target['week'] = week

        target['subNum'] = sn

        target.to_csv(os.path.join(out_path, f'{experiment}_{session}_{sn}_day{day}_run{n+1}.tgt'), index=False, sep='\t')




