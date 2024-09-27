import os
from pathlib import Path
import numpy as np

fthresh = 1.2  # threshold to exit the baseline area
fGain = np.array([1, 1, 1, 1.5, 1.5])
hold_time = .6  # in seconds

fsample = {
    'emg': 2148,
    'force': 500,
}

nblocks = 8

chordID = [29212, 92122, 91211, 22911, 21291, 12129, 12291, 19111]
# trained = [29212, 92122, 91211, 22911]
# untrained = [21291, 12129, 12291, 19111]

movCols = ['state', 'timeReal', 'time',
           'eThumb', 'eIndex', 'eMiddle', 'eRing', 'ePinkie',
           'fThumb', 'fIndex', 'fMiddle', 'fRing', 'fPinkie',
           'Thumb', 'Index', 'Middle', 'Ring', 'Pinkie',  # 13, 14, 15, 16, 17
           'vThumb', 'vIndex', 'vMiddle', 'vRing', 'vPinkie']
diffCols = [13, 14, 15, 16, 17]

Dirs = ["/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/ExtFlexChord",
        "/cifs/diedrichsen/data/SequenceAndChord/ExtFlexChord"]

# natChord_chordDir = "/Users/mnlmrc/Downloads/natChord_chord.tsv"


baseDir = next((Dir for Dir in Dirs if Path(Dir).exists()), None)

if baseDir:
    print(f"Base directory found: {baseDir}")
else:
    print("No valid base directory found.")

participants = {
    'efc0': ['subj01',
             'subj02',
             'subj03',
             'subj04',
             'subj05',
             'subj06',
             'subj07',
             'subj08',
             'subj09',
             'subj10',
             'subj11',
             'subj10',
             'subj10',
             'subj10'],
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
        'subj108',
        'subj109',
        'subj110',
        'subj111',
        'subj112',
        'subj113'
    ],
    'efc3': [
        'subj100',
        'subj114'
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
            'emg_hold_avg_f5'],
    'emgTMS': [
        # 'ring_ext'  , 'pinkie_ext'  ,'thumb_flex'  ,'pinkie_flex' , 'index_flex' , 'middle_flex' ,  'thumb_ext' ,  'index_ext' , 'EMG 1' ,'EMG 2' , 'thumb_ext' , 'index_ext' , 'thumb_ext' ,  'index_ext',

        'ext_D3',
        'ext_D4',
        'ext_D5',
        'ext_D1',
        'ext_D2',
        'flx_D3',
        'flx_D4',
        'flx_D2',
        'flx_D1',
        'flx_D5',
        'FDI',
        'lum1',
        'lum2',
        'lum3',
    ]
}

removeEMG = {
    'efc3': {
        'subj100': ['flx_D5'],
        'subj114': ['lum3']
    }
}
