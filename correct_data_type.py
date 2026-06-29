import os.path

import nitools as nt

import EFC_learningfMRI.globals as gl

sn = 104
BN = 1

infile = os.path.join(gl.baseDir, 'imaging_data_raw', f'subj{sn}', f'subj{sn}_run_{BN:02}')
outfile = os.path.join(gl.baseDir, 'imaging_data_raw', f'subj{sn}', f'subj{sn}_run_{BN:02}')

nt.volume.change_nifti_numformat(infile, outfile, new_numformat="uint16", typecast_data=True)

