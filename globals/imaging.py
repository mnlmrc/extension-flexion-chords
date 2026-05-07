import os
from globals.path import atlasDir

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