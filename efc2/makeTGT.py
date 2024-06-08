import argparse
import random

import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--subNum', default='101',
                        help='Participant ID (e.g., subj100, subj101, ...)')
    parser.add_argument('--session', default='testing', help='Atlas name')

    args = parser.parse_args()

    subNum = args.subNum
    session = args.session

    if session == 'training':
        chordIDs = np.array([29212, 92122, 91211, 22911])
        nChunks = 20  # number of chunks to repeat chords
        nDays = 4
    elif session == 'testing':
        chordIDs = np.array([29212, 92122, 91211, 22911, 21291, 12129, 12291, 19111])
        nChunks = 10  # number of chunks to repeat chords
        nDays = 2
    else:
        raise ValueError('session must be either training or testing')

    nChords = len(chordIDs)  # number of chords
    nRep = 5  # number of repetition of each chord
    nRuns = 8
    planTime = 0  # time for planning
    execMaxTime = 10000  # maximum time for execution
    feedbackTime = 500  # time to present feedback
    # iti =   # inter-trial interval

    totTrial = nRep * nChunks

    prevChord = 00000
    # chords = np.array([], dtype='int')
    for day in range(nDays):

        chords = np.array([], dtype='int')

        for ch in range(nChunks * len(chordIDs)):

            valid = False
            while not valid:
                currentChord = chordIDs[random.randint(0, chordIDs.size - 1)]

                if currentChord != prevChord:
                    valid = True
                    prevChord = currentChord
                    break

            chunk = np.repeat(currentChord, nRep, axis=0)
            chords = np.concatenate((chords, chunk), axis=0)

        runs = np.split(chords, nRuns)

        # column names:
        column_names = ['subNum', 'chordID', 'planTime', 'execMaxTime', 'feedbackTime', 'iti']
        for r, run in enumerate(runs):
            # building the dataframe:
            df = pd.DataFrame(columns=column_names)
            df['chordID'] = run
            df['subNum'] = subNum
            df['planTime'] = planTime
            df['execMaxTime'] = execMaxTime
            df['feedbackTime'] = feedbackTime
            df['iti'] = np.random.randint(500, 1500, len(run))

            fname = f'efc2_{session}_{subNum:02}_day{day + 1}_run{r +1}.tgt'
            df.to_csv('target/' + fname, sep='\t', index=False)

