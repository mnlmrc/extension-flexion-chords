import os
import numpy as np

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.betas import roi_avg
from EFC_learningfMRI.surface import project_cifti_to_surface, smooth_cifti_contrasts, average_smoothed_contrasts
from EFC_learningfMRI.repetition_suppression import r2_v_r1_slope, pc1_theta

if __name__ == "__main__":
    glm = 2
    sns = [101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113]

    # for sn in sns: project_cifti_to_surface(sn, glm, stat='rep_suppr')

    # for sn in sns: smooth_cifti_contrasts(sn, glm, stat='rep_suppr')
    # average_smoothed_contrasts(sns, glm, stat='rep_suppr')
    
    # roi_avg(sns=sns, atlas_name='ROI', glm=glm, fname='contrast.dscalar.nii', cond_names=['chordID', 'session', 'rep'])
    #roi_avg(sns=sns, atlas_name='BA_handArea', glm=glm, fname='contrast.dscalar.nii', cond_names=['chordID', 'session', 'rep'])
    pc1_theta(sns=sns, atlas_name='ROI', glm=glm, fname='contrast.dscalar.nii')
    # r2_v_r1_slope(sns=sns, atlas_name='ROI', glm=glm, fname='contrast.dscalar.nii')

    #long_contrast = save_rep_contrast_long(sns, glm, atlas_name='ROI')
    #fit_lme(long_contrast, glm, atlas_name='ROI')

