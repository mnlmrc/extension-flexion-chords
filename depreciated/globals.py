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

print(f"Atlas directory found: {atlasDir}")







# cmap = plt.get_cmap("Set2")
# colors = [cmap(i) for i in np.linspace(0, 1, 8)]
#
# colour_mapping = {'glm1':
#                       {'chordID:12129': colors[0],
#                        'chordID:12291': colors[1],
#                        'chordID:19111': colors[2],
#                        'chordID:21291': colors[3],
#                        'chordID:22911': colors[4],
#                        'chordID:29212': colors[5],
#                        'chordID:91211': colors[6],
#                        'chordID:92122': colors[7],
#                        }
#                   }


