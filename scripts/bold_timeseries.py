import os
import numpy as np
from nitools import spm
import EFC_learningfMRI.globals as gl
import nibabel as nb
import nitools as nt

def save_bold_rois(sn, glm, atlas='ROI', rois=None):

    if rois is None:
        rois = gl.rois[atlas]

    path_glm  = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}')
    path_rois = os.path.join(gl.baseDir, 'ROI', f'subj{sn}')
    SPM       = spm.SpmGlm(path_glm)

    SPM.get_info_from_spm_mat()

    for H in gl.Hem:
        for roi in rois:
            print(f'doing participant {sn}, {H}, {roi}')

            roi_img = nb.load(os.path.join(path_rois, f'{atlas}.{H}.{roi}.nii'))
            coords  = nt.get_mask_coords(roi_img)
            y_raw   = nt.sample_images(SPM.rawdata_files, coords)
            y_scl   = y_raw * SPM.gSF[:, None] # rescale y_raw

            _, info, _, data_hat, data_adj, _ = SPM.rerun_glm(y_scl)

            np.save(os.path.join(path_glm, f'BOLD.hat.{H}.{roi}.npy'), data_hat)
            np.save(os.path.join(path_glm, f'BOLD.raw.{H}.{roi}.npy'), y_scl)
            np.save(os.path.join(path_glm, f'BOLD.adj.{H}.{roi}.npy'), data_adj)


if __name__=='__main__':
    sns   = [117] # gl.participants
    glm   = 3
    atlas = 'ROI'
    rois  = gl.rois[atlas]

    for sn in sns:
        save_bold_rois(sn, glm, rois=rois)