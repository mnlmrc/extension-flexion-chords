import argparse
import random

import numpy as np
import pandas as pd


def shuffle_chord_seq(chordSeq):
    np.random.shuffle(chordSeq)
    for i in range(1, len(chordSeq)):
        if chordSeq[i] == chordSeq[i - 1]:
            # Find the first subsequent element that is different
            for j in range(i + 1, len(chordSeq)):
                if chordSeq[j] != chordSeq[i - 1]:
                    # Swap the current element with the found element
                    chordSeq[i], chordSeq[j] = chordSeq[j], chordSeq[i]
                    break
    return chordSeq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--subNum', default='100',
                        help='Participant ID (e.g., subj100, subj101, ...)')
    parser.add_argument('--session', default='training')

    args = parser.parse_args()

    subNum = args.subNum
    session = args.session

    nRep = 5  # number of repetition of each chord
    planTime = 0  # time for planning
    execMaxTime = 10000  # maximum time for execution
    feedbackTime = 500  # time to present feedback
    # iti =   # inter-trial interval

    # column names:
    column_names = ['subNum', 'chordID', 'planTime', 'execMaxTime', 'feedbackTime', 'iti']

    if session == 'training':
        chordIDs = np.array([21291, 12129, 12291, 19111])
        nChunks = 3  # number of chunks to repeat chords on each block
        nDays = 4
        nRuns = 8
        nChords = len(chordIDs)
        totChunks = nChunks * nChords
        currentChord = np.repeat(chordIDs, int(totChunks / nChords), axis=0)
        currentChord = shuffle_chord_seq(currentChord)

    elif session == 'testing':
        chordIDs = np.array([29212, 92122, 91211, 22911, 21291, 12129, 12291, 19111])
        nChunks = 2  # number of chunks to repeat chords
        nDays = 2
        nRuns = 6
        totChunks = nChunks * len(chordIDs)
        currentChord = np.repeat(chordIDs, int(totChunks / len(chordIDs)), axis=0)
        currentChord = shuffle_chord_seq(currentChord)

        pass
    else:
        raise ValueError('session must be either training or testing')

    # chords = np.array([], dtype='int')
    for day in range(nDays):
        for bl in range(nRuns):
            chords = np.array([], dtype='int')
            for ch in range(len(currentChord)):
                chunk = np.repeat(currentChord[ch], nRep, axis=0)
                chords = np.concatenate((chords, chunk), axis=0)

            # build the dataframe:
            df = pd.DataFrame(columns=column_names)
            df['chordID'] = chords
            df['subNum'] = subNum
            df['planTime'] = planTime
            df['execMaxTime'] = execMaxTime
            df['feedbackTime'] = feedbackTime
            df['iti'] = np.random.randint(500, 1500, len(chords))

            fname = f'efc3_{session}_{subNum:02}_day{day + 1}_run{bl + 1}.tgt'
            df.to_csv('target/' + fname, sep='\t', index=False)
