import os
from pathlib import Path

fsample = 500
fthresh = 1.2  # threshold to exit the baseline area
hold_time = .6  # in seconds

nblocks = 8

# trained = [29212, 92122, 91211, 22911]
# untrained = [21291, 12129, 12291, 19111]

movCols = ['state', 'timeReal', 'time',
           'eThumb', 'eIndex', 'eMiddle', 'eRing', 'ePinkie',
           'fThumb', 'fIndex', 'fMiddle', 'fRing', 'fPinkie',
           'Thumb', 'Index', 'Middle', 'Ring', 'Pinkie',  # 13, 14, 15, 16, 17
           'vThumb', 'vIndex', 'vMiddle', 'vRing', 'vPinkie']
diffCols = [13, 14, 15, 16, 17]

Dirs = ["/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/ExtFlexChords"]

# natChord_chordDir = "/Users/mnlmrc/Downloads/natChord_chord.tsv"


baseDir = next((Dir for Dir in Dirs if Path(Dir).exists()), None)

if baseDir:
    print(f"Base directory found: {baseDir}")
else:
    print("No valid base directory found.")

participants = {
    'efc1': [
        'subj01',
        'subj02',
        'subj03',
        'subj04',
        'subj05',
        'subj06',
        'subj07',
        'subj08',
        'subj09',
        'subj10'
    ],
    'efc2': [
        'subj100',
        'subj101',
        'subj102',
        'subj103',
        'subj104',
        # 'subj105',
        'subj106',
        'subj107',
    ]
}

natDir = 'natural'
chordDir = 'chords'

days = ['1', '2', '3', '4', '5']

channels = {
    'force': ['thumb',
              'index',
              'middle',
              'ring',
              'pinkie'],
    'emg': ['emg_hold_avg_e1',
            'emg_hold_avg_e2',
            'emg_hold_avg_e3',
            'emg_hold_avg_e4',
            'emg_hold_avg_e5',
            'emg_hold_avg_f1',
            'emg_hold_avg_f2',
            'emg_hold_avg_f3',
            'emg_hold_avg_f4',
            'emg_hold_avg_f5']
}
