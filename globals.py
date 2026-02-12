import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

ROOT =  str(Path().resolve().parent)+ '/extension-flexion-chords/'
sys.path.append(ROOT)

Dirs = ["/Volumes/diedrichsen_data$/data/Chord_exp/EFC_learningfMRI",
        "/cifs/diedrichsen/data/Chord_exp/EFC_learningfMRI"]

# natChord_chordDir = "/Users/mnlmrc/Downloads/natChord_chord.tsv"
baseDir = next((Dir for Dir in Dirs if Path(Dir).exists()), None)
natDir = 'natural'
imagingDir = 'imaging_data'
anatDir = 'anatomicals'
chordDir = 'chords'
behavDir = 'behavioural'
glmDir = 'glm'
roiDir = 'ROI'
rdmDir = 'rdm'
surfDir = 'surfaceWB'
pcmDir = 'pcm'

atlasDir = os.path.join(ROOT, 'atlases') #next((Dir for Dir in atlasDirs if Path(Dir).exists()), None)

if baseDir is not None:
    print(f"Base directory found: {baseDir}")
    print(f"Atlas directory found: {atlasDir}")
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

chordID = np.sort(np.array(['21911', '92122', '91211', '22911', '21291', '12129', '12291', '11911']))
diffCols = [18, 19, 20, 21, 22]
wait_exec = 4


trialPoint_mapping = {
    1: 'success',
    0: 'unsuccess',
}

borderDirs = ["/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/",
        "/home/UWO/memanue5/Documents/GitHub/surfAnalysisPy/standard_mesh/",]

borderDirs = next((Dir for Dir in borderDirs if Path(Dir).exists()), None)

# borders = {'L': os.path.join(borderDirs, 'fs_L', 'fs_LR.32k.L.border'),
#            'R': os.path.join(borderDirs, 'fs_L', 'fs_LR.32k.L.border')}

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
