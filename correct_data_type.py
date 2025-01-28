import os.path

import nitools as nt

import globals

sn = 104
BN = 1

infile = os.path.join(globals.baseDir, 'imaging_data_raw', f'subj{sn}', f'subj{sn}_run_{BN:02}')
outfile = os.path.join(globals.baseDir, 'imaging_data_raw', f'subj{sn}', f'subj{sn}_run_{BN:02}')

nt.volume.change_nifti_numformat(infile, outfile, new_numformat="uint16", typecast_data=True)

