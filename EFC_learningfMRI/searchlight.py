import argparse
from imaging_pipelines.searchlight import searchlight_surf
import EFC_learningfMRI.globals as gl
import time
import os
import numpy as np


def make_searchlight(sn):
    path_surf = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
    white = [os.path.join(path_surf, f'subj{sn}.{H}.white.32k.surf.gii') for H in gl.Hem]
    pial = [os.path.join(path_surf, f'subj{sn}.{H}.pial.32k.surf.gii') for H in gl.Hem]
    mask = [os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'Hem.{H}.nii') for H in gl.Hem]
    savedir = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
    searchlight_surf(white, pial, mask, savedir, maxradius=10, maxvoxels=100)


