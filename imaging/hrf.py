import os
import numpy as np
import pandas as pd
import globals.path as pth
import globals.imaging as im
from nitools import spm
import nibabel as nb
import nitools as nt
from imaging_pipelines import hrf


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

# def save_BOLD_cut(sns, glm, H='L', roi='M1', atlas_name='ROI', TR=1, nTR=336):
#     pinfo = pd.read_csv(os.path.join(pth.baseDir, 'participants.tsv'), sep='\t')
#     y_cut_filt, y_cut_adj, y_cut_hat = [], [], []
#     for sn in sns:
#         print(f'doing participant {sn}')
#         func_runs = pinfo.loc[pinfo.participant_id == f"subj{sn}", "FuncRuns_day3"].iloc[0].split('.')
#         P = np.array(pinfo.loc[pinfo.participant_id == f"subj{sn}", "P"].iloc[0].split(','), dtype=float)
#         func_runs = np.array(func_runs, dtype=int)
#         func_runs = np.array([func_runs + func_runs.size * i for i in range(3)]).flatten()
#         path_glm = os.path.join(pth.baseDir, f'glm{glm}', f'subj{sn}')
#         path_hrf = os.path.join(pth.baseDir, 'hrf', f'subj{sn}')
#         events = pd.read_csv(os.path.join(pth.baseDir, pth.behavDir, 'day3', f'efc4_subj{sn}_glm{glm}_events.tsv'),
#                              sep='\t')
#         events = events[events.BN.isin(func_runs)]
#         BN = events.BN.to_numpy() - 1
#         onset_b = events.Onset.to_numpy()
#         onset = (np.round(onset_b * TR) + BN * nTR).astype(int)
#         onset = np.sort(onset)
#         SPM = spm.SpmGlm(path_glm)
#         SPM.get_info_from_spm_mat()
#         y_raw = np.load(os.path.join(path_hrf, f'BOLD.ROI.{H}.{roi}.npy'))
#         y_scl = y_raw * SPM.gSF[:, None]  # rescale y_raw
#         _, info, data_filt, data_hat, data_adj, _ = SPM.rerun_glm(y_scl)
#         y_cut_filt.append(spm.cut(data_filt, pre=6, at=onset, post=12, padding='last').mean(axis=(0, -1)))
#         y_cut_adj.append(spm.cut(data_adj, pre=6, at=onset, post=12, padding='last').mean(axis=(0, -1)))
#         y_cut_hat.append(spm.cut(data_hat, pre=6, at=onset, post=12, padding='last').mean(axis=(0, -1)))
#
#     y_cut_filt = np.vstack(y_cut_filt)
#     y_cut_adj = np.vstack(y_cut_adj)
#     y_cut_hat = np.vstack(y_cut_hat)
#
#     np.save(os.path.join(pth.baseDir, f'glm{glm}',  f'y_filt.{roi}.npy'), y_cut_filt)
#     np.save(os.path.join(pth.baseDir, f'glm{glm}', f'y_adj.{roi}.npy'), y_cut_adj)
#     np.save(os.path.join(pth.baseDir, f'glm{glm}', f'y_hat.{roi}.npy'), y_cut_hat)
