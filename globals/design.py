import numpy as np

fthresh = 1.2  # threshold to exit the baseline area
ftarget = 2
fGain = np.array([1, 1, 1, 1.5, 1.5])
hold_time = .6  # in seconds

fsample = {
    'emg': 2148,
    'force': 500,
}

nblocks = 8

chordID = np.sort(np.array([21911, 92122, 91211, 22911, 21291, 12129, 12291, 11911]))
sessions = [3, 9, 23]
nSess = 3
diffCols = [18, 19, 20, 21, 22]
wait_exec = 4

trialPoint_mapping = {
    1: 'success',
    0: 'unsuccess',
}

sess_mapping = {
    'sess03': 0,
    'sess09': 1,
    'sess23': 2
}