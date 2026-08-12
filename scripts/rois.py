import os
import EFC_learningfMRI.globals as gl
from imaging_pipelines import rois
import nibabel as nb
import numpy as np
import nitools as nt

def make_cortical_rois(sn, glm=3, atlas_name='ROI'):

    exclude = {
        'ROI': [(1, 2), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5), (2, 7), (3, 4), (3, 5)],
        'BA_handArea': [],
        'ROI_grouped': []
    }

    path_surf = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
    white     = [os.path.join(path_surf, f'subj{sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
    pial      = [os.path.join(path_surf, f'subj{sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
    mask      = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'mask.nii')
    atlas_dir = gl.atlasDir
    rois_dir  = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
    Rois      = rois.SurfRois(white, pial, mask, rois_dir, atlas_name=atlas_name, atlas_dir=atlas_dir)

    Rois.make_rois(exclude=exclude[atlas_name])

def make_hemispheres(sn, glm):
    path_surf = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
    white     = [os.path.join(path_surf, f'subj{sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
    pial      = [os.path.join(path_surf, f'subj{sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
    mask      = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'mask.nii')
    rois_dir  = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
    Rois      = rois.SurfRois(white, pial, mask, rois_dir)

    Rois.make_hemispheres()


if "__main__" == __name__:
    sns = [117]
    glm = 3
    for sn in sns:
        make_cortical_rois(sn, glm)
        make_hemispheres(sn, glm)