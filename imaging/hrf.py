import os
import numpy as np
import pandas as pd
import globals.path as pth
import globals.imaging as im
from nitools import spm
import nibabel as nb
import nitools as nt
from imaging_pipelines import hrf
from util.util import calc_R2


def optimise_hrf(sn, glm, H='L', rois=['M1'], atlas_name='ROI'):
    path_glm = os.path.join(pth.baseDir, f'glm{glm}', f'subj{sn}')
    path_hrf = os.path.join(pth.baseDir, 'hrf', f'subj{sn}')
    SPM = spm.SpmGlm(path_glm)
    SPM.get_info_from_spm_mat()

    print(f'subj{sn}, loading raw data...')
    y_raw = np.load(os.path.join(path_hrf, f'BOLD.{atlas_name}.{H}.{".".join(rois)}.npy')) #nt.sample_images(SPM.rawdata_files, coords)
    y_scl = y_raw * SPM.gSF[:, None]  # rescale y_raw

    print('optimising HRF parameters...')
    grid = {
        0: np.array([3., 4., 5., 6., 7.]),  # delay response
        1: np.array([9., 10., 12., 14., 16., 18.]),  # delay undershoot
        2: np.array([1.0]),  # dispersion response
        3: np.array([1.0]),  # dispersion undershoot
        4: np.array([6.]),  # ratio
        5: np.array([0.0]),  # onset
        6: np.array([32.0])  # length
    }
    P, _, res = hrf.grid_search_hrf(SPM, y_scl, TR=1, grid=grid)
    print(f'optimisation complete, P={P}')

    df = pd.DataFrame(P)
    df.to_csv(os.path.join(path_glm, 'P.tsv'), sep='\t', index=False)

def save_BOLD(sn, glm, atlas_name='ROI'):
    path_glm = os.path.join(pth.baseDir, f'glm{glm}', f'subj{sn}')
    path_rois = os.path.join(pth.baseDir, 'ROI', f'subj{sn}')
    path_hrf = os.path.join(pth.baseDir, 'hrf', f'subj{sn}')
    os.makedirs(path_hrf, exist_ok=True)
    SPM = spm.SpmGlm(path_glm)
    SPM.get_info_from_spm_mat()
    coords = []
    for H in im.Hem:
        for roi in im.rois[atlas_name]:
            print(f'subj{sn}, {H}, {roi}, saving BOLD timeseries...')
            roi_img = nb.load(os.path.join(path_rois, f'{atlas_name}.{H}.{roi}.nii'))
            coords = nt.get_mask_coords(roi_img)
            y_raw = nt.sample_images(SPM.rawdata_files, coords)
            y_scl = y_raw * SPM.gSF[:, None]
            _, info, y_filt, y_hat, y_adj, _ = SPM.rerun_glm(y_scl)
            np.save(os.path.join(path_hrf, f'y_adj.{atlas_name}.{H}.{roi}.npy'), y_adj)
            np.save(os.path.join(path_hrf, f'y_hat.{atlas_name}.{H}.{roi}.npy'), y_hat)
            np.save(os.path.join(path_hrf, f'y_filt.{atlas_name}.{H}.{roi}.npy'), y_filt)


def calc_R2_adj_hat(sns, glm, atlas_name='ROI'):
    r2 = {'R2': [], 'glm': [], 'Hem': [], 'roi': [], 'sn': []}
    for glm in [1, 3]:
        for H in im.Hem:
            for roi in im.rois[atlas_name]:
                for sn in sns:
                    print(f'doing participant {sn}, {H}, {roi}, glm{glm}...')
                    y_adj = np.load(os.path.join(pth.baseDir, 'hrf', f'subj{sn}', f'y_adj.glm{glm}.ROI.{H}.{roi}.npy'))
                    y_hat = np.load(os.path.join(pth.baseDir, 'hrf', f'subj{sn}', f'y_hat.glm{glm}.ROI.{H}.{roi}.npy'))
                    R2 = calc_R2(y_adj, y_hat)
                    r2['R2'].append(R2)
                    r2['glm'].append(glm)
                    r2['Hem'].append(H)
                    r2['roi'].append(roi)
                    r2['sn'].append(sn)
    df_r2 = pd.DataFrame(r2)
    df_r2.to_csv(os.path.join(pth.baseDir, 'hrf', f'R2.{atlas_name}.tsv'), sep='\t', index=False)


