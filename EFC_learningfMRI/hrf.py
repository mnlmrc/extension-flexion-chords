import os
import numpy as np
import pandas as pd
import EFC_learningfMRI.globals as gl
from nitools import spm
import nibabel as nb
import nitools as nt
import time
from imaging_pipelines import hrf
from EFC_learningfMRI.util import calc_R2, load_glm_onset
from scipy.optimize import minimize


class Optimise_HRF:

    def __init__(self, sn, glm, H='L', rois=['M1'], atlas_name='ROI', P=None, TR=1, nTR=336):
        
        self.sn = sn
        self.glm = glm
        self.P0 = np.array([6., 16., 1., 1., 6., 0., 32.], dtype=float)
        self.glm_path = os.path.join(gl.baseDir, f'glm{glm}',)
        self.TR= TR
        self.nTR = nTR
        self.rois = rois
        self.H = H
        self.onset = load_glm_onset(sn, glm)
        self.SPM = spm.SpmGlm(os.path.join(self.glm_path, f'subj{sn}'))
        self.SPM.get_info_from_spm_mat()

    def manual(self, P, roi='M1', pre=6, post=12):
        hrf, _ = spm.spm_hrf(1, P=P)
        self.SPM.convolve_glm(hrf)
        y_raw = np.load(os.path.join(self.glm_path, f'subj{self.sn}', f'BOLD.raw.{self.H}.{roi}.npy'))
        _, info, _, y_hat, y_adj, _ = self.SPM.rerun_glm(y_raw)
        y_cut_hat = spm.cut(y_hat, pre=pre, at=self.onset, post=post, padding='last')
        y_cut_adj = spm.cut(y_adj, pre=pre, at=self.onset, post=post, padding='last')
        y_cut_raw = spm.cut(y_raw, pre=pre, at=self.onset, post=post, padding='last')
        return y_hat, y_adj, y_raw, y_cut_hat, y_cut_adj, y_cut_raw 

    def cut(self, roi='M1', pre=6, post=12):
        y_raw = np.load(os.path.join(self.glm_path, f'subj{self.sn}', f'BOLD.raw.{self.H}.{roi}.npy'))
        y_hat = np.load(os.path.join(self.glm_path, f'subj{self.sn}', f'BOLD.hat.{self.H}.{roi}.npy'))
        y_adj = np.load(os.path.join(self.glm_path, f'subj{self.sn}', f'BOLD.adj.{self.H}.{roi}.npy'))
        y_cut_hat = spm.cut(y_hat, pre=pre, at=self.onset, post=post, padding='last')
        y_cut_adj = spm.cut(y_adj, pre=pre, at=self.onset, post=post, padding='last')
        y_cut_raw = spm.cut(y_raw, pre=pre, at=self.onset, post=post, padding='last')       
        return y_cut_hat, y_cut_adj, y_cut_raw

    def _gridsearch_in_roi(self, roi):

        print('optimising HRF parameters...')

        grid = {
            0: np.array([4., 5., 6., 7., 8., 9.]),  # delay response
            1: np.array([10., 12., 14., 16., 18., 20.]),  # delay undershoot
            2: np.array([1.0]),  # dispersion response
            3: np.array([1.0]),  # dispersion undershoot
            4: np.array([2., 3., 4., 5., 6., 7.]),  # ratio
            5: np.array([0.0]),  # onset
            6: np.array([32.0])  # length
        }

        y_raw = np.load(os.path.join(self.glm_path, f'subj{self.sn}', f'BOLD.raw.{self.H}.{roi}.npy'))
        P, _, params_gridsearch = hrf.grid_search_hrf(self.SPM, y_raw, TR=gl.TR, grid=grid)
        print(f'optimisation complete, P={P}')
        return params_gridsearch


    def gridsearch(self):
        params_gridsearch = pd.DataFrame()
        for roi in self.rois:
            grid = self._gridsearch_in_roi(roi)
            grid['roi'] = roi
            params_gridsearch = pd.concat([params_gridsearch, grid], axis=0)
        params_gridsearch.to_csv(os.path.join(self.glm_path, f'subj{self.sn}', 'gridsearch_hrf.tsv'), sep='\t', index=False)


def save_bold_rois(sn, glm, atlas='ROI', H='L', rois=None):
    if rois is None:
        rois = gl.rois[atlas]
    path_glm = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}')
    path_rois = os.path.join(gl.baseDir, 'ROI', f'subj{sn}')
    SPM = spm.SpmGlm(path_glm)
    SPM.get_info_from_spm_mat()
    for H in ['L']: #gl.Hem:
        for roi in rois:
            print(f'doing participant {sn}, {H}, {roi}')
            roi_img = nb.load(os.path.join(path_rois, f'{atlas}.{H}.{roi}.nii'))
            coords = nt.get_mask_coords(roi_img)
            y_raw = nt.sample_images(SPM.rawdata_files, coords)
            y_scl = y_raw * SPM.gSF[:, None]  # rescale y_raw
            _, info, _, data_hat, data_adj, _ = SPM.rerun_glm(y_scl)
            np.save(os.path.join(path_glm, f'BOLD.hat.{H}.{roi}.npy'), data_hat)
            np.save(os.path.join(path_glm, f'BOLD.raw.{H}.{roi}.npy'), y_scl)
            np.save(os.path.join(path_glm, f'BOLD.adj.{H}.{roi}.npy'), data_adj)




