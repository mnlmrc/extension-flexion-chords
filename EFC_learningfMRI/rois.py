import os
import EFC_learningfMRI.globals as gl
from imaging_pipelines import rois
import nibabel as nb
import numpy as np
import nitools as nt

import argparse
import time

def make_cortical_rois(sn, atlas_name='ROI', glm=1):
    exclude = {
        'ROI': [(1, 2), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5), (2, 7), (3, 4), (3, 5), (7, 8)],
        'BA_handArea': [],
        'ROI_grouped': []
    }
    path_surf = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
    white = [os.path.join(path_surf, f'subj{sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
    pial = [os.path.join(path_surf, f'subj{sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
    mask = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'mask.nii')
    atlas_dir = gl.atlasDir
    rois_dir = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
    Rois = rois.SurfRois(white, pial, mask, rois_dir, atlas_name=atlas_name, atlas_dir=atlas_dir)
    Rois.make_rois(exclude=exclude[atlas_name])

def make_hemispheres(sn, glm):
    path_surf = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
    white = [os.path.join(path_surf, f'subj{sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
    pial = [os.path.join(path_surf, f'subj{sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
    mask = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'mask.nii')
    rois_dir = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
    Rois = rois.SurfRois(white, pial, mask, rois_dir)
    Rois.make_hemispheres()

def make_roi_grouped():
    for H, struct in zip(gl.Hem, gl.struct):
        gifti = nb.load(os.path.join('..', gl.atlasDir, f'ROI.32k.{H}.label.gii'))
        labels = nt.get_gifti_labels(gifti)
        data = nt.get_gifti_data_matrix(gifti)
        data_new = np.zeros_like(data,)
        data_new[(data == 3) | (data == 4) | (data == 5)] = 1
        data_new[(data == 1) | (data == 2)] = 2  # M1-S1
        data_new[(data == 8) | (data == 7)] = 3  # SPLa, SPLp
        data_new[data == 6] = 4  # V1
        gifti_new = nt.make_label_gifti(data_new,
                                        anatomical_struct=struct,
                                        label_RGBA = [np.array([0, 0, 0, 0]),
                                                      np.array([1, 0, 0, 1]),
                                                      np.array([0, 1, 0, 1]),
                                                      np.array([0, 0, 1, 1]),
                                                      np.array([1, 1, 0, 1]),],
                                        label_names=['', 'premotor', 'M1-S1', 'parietal', 'V1'])
        nb.save(gifti_new, os.path.join('..', gl.atlasDir, f'ROI_grouped.32k.{H}.label.gii'))


if "__main__" == __name__:
    make_roi_grouped()