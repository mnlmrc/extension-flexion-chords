import os
from pathlib import Path

fsample = 500
fthresh = 1.2  # threshold to exit the baseline area
hold_time = .6  # in seconds

nblocks = 8

trained = [29212, 92122, 91211, 22911]
untrained = [21291, 12129, 12291, 19111]

movCols = ['state', 'timeReal', 'time',
           'eThumb', 'eIndex', 'eMiddle', 'eRing', 'ePinkie',
           'fThumb', 'fIndex', 'fMiddle', 'fRing', 'fPinkie',
           'Thumb', 'Index', 'Middle', 'Ring', 'Pinkie',  # 13, 14, 15, 16, 17
           'vThumb', 'vIndex', 'vMiddle', 'vRing', 'vPinkie']
diffCols = [13, 14, 15, 16, 17]

baseDir = [
           "/Volumes/diedrichsen_data$/data/SequenceAndChord/ExtFlexChord"
           ]
for Dir in baseDir:
    if Path(Dir).exists():
        baseDir = Dir
        break