import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

Dirs = ["/Volumes/diedrichsen_data$/data/Chord_exp/ExtFlexChord",
        "/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/ExtFlexChord",
        "/cifs/diedrichsen/data/Chord_exp/ExtFlexChord"]

# natChord_chordDir = "/Users/mnlmrc/Downloads/natChord_chord.tsv"
baseDir = next((Dir for Dir in Dirs if Path(Dir).exists()), None)
natDir = 'natural'
chordDir = 'chords'
behavDir = 'behavioural'
glmDir = 'glm'
roiDir = 'ROI'
rdmDir = 'rdm'
wbDir = 'surfaceWB'

if baseDir is not None:
    print(f"Base directory found: {baseDir}")
else:
    print("No valid base directory found.")

fthresh = 1.2  # threshold to exit the baseline area
ftarget = 2
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

# movCols = ['state', 'timeReal', 'time',
#            'eThumb', 'eIndex', 'eMiddle', 'eRing', 'ePinkie',
#            'fThumb', 'fIndex', 'fMiddle', 'fRing', 'fPinkie',
#            'Thumb', 'Index', 'Middle', 'Ring', 'Pinkie',  # 13, 14, 15, 16, 17
#            'vThumb', 'vIndex', 'vMiddle', 'vRing', 'vPinkie']
diffCols = [18, 19, 20, 21, 22]  # [13, 14, 15, 16, 17]  # [14, 15, 16, 17, 18] #

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

# removeEMG = {
#     'efc3': {
#         'subj100': ['lum1', 'lum2', 'lum3']
#     }
# }

# flatmap stuff
borders = {'L': '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_L/fs_LR.32k.L.border',
           'R': '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_R/fs_LR.32k.R.border'}

rois = {
    'Desikan': [
        'rostralmiddlefrontal',
        'caudalmiddlefrontal',
        'precentral',
        'postcentral',
        'superiorparietal',
        'pericalcarine'
    ],
    'BA_handArea': [
        'ba4a', 'ba4p', 'ba3A', 'ba3B', 'ba1', 'ba2'
    ],
    'ROI': [
        'SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1'
    ]
}

### colours ###
cmap = plt.get_cmap("Set2")
colors = [cmap(i) for i in np.linspace(0, 1, 8)]

colour_mapping = {'glm1':
                      {'chordID:12129': colors[0],
                       'chordID:12291': colors[1],
                       'chordID:19111': colors[2],
                       'chordID:21291': colors[3],
                       'chordID:22911': colors[4],
                       'chordID:29212': colors[5],
                       'chordID:91211': colors[6],
                       'chordID:92122': colors[7],
                       }
                  }

###############
