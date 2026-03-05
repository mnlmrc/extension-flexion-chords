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
    figDir = '../figures/'

print(f'base directory found: {baseDir}')
print(f"Atlas directory found: {atlasDir}")
print(f'figure directory found: {figDir}')