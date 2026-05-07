import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

baseDir = "/cifs/diedrichsen/data/Chord_exp/EFC_learningfMRI"
if not os.path.exists(baseDir):
     baseDir = "/Volumes/diedrichsen_data$/data/Chord_exp/EFC_learningfMRI"

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

atlasDir = 'atlases' #next((Dir for Dir in atlasDirs if Path(Dir).exists()), None)

figDir='/Users/jdiedrichsen/Dropbox/Grants/CIHR_chords_2026/Figures'
if not os.path.exists(figDir):
    figDir = 'figures/'

print(f'base directory found: {baseDir}')
print(f"Atlas directory found: {atlasDir}")
print(f'figure directory found: {figDir}')

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

TR = 1
nTR = 336

borders = {'L': os.path.join(atlasDir, 'fs_LR.32k.L.border'),
           'R': os.path.join(atlasDir, 'fs_LR.32k.L.border')}

Hem = ['L', 'R']
struct = ['CortexLeft', 'CortexRight']
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
    ],
    'ROI_grouped': [
        'premotor', 'M1-S1', 'parietal', 'V1'
    ]
}