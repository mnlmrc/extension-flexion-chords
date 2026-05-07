import os
import numpy as np
import pandas as pd
import globals.path as pth
import globals.imaging as im
from nitools import spm
import nibabel as nb
import nitools as nt
import time
from imaging_pipelines import hrf
from util.util import calc_R2, load_glm_onset
from scipy.optimize import minimize


class Optimise_HRF:

    def __init__(self, sn, glm, H='L', roi='M1', atlas_name='ROI', P=None, TR=1, nTR=336):
        
        self.sn = sn
        self.glm = glm
        self.P0 = np.array([6., 16., 1., 1., 6., 0., 32.], dtype=float)
        self.glm_path = os.path.join(pth.baseDir, f'glm{glm}',)
        self.TR= TR
        self.nTR = nTR
        self.df = self._make_hrf_table()
        self.onset = load_glm_onset(sn, glm)
        self.SPM = spm.SpmGlm(os.path.join(pth.baseDir, f'glm{glm}', f'subj{sn}'))
        self.SPM.get_info_from_spm_mat()
        self.y_raw = np.load(os.path.join(pth.baseDir, 'hrf', f'subj{sn}', f'y_raw.glm{glm}.{atlas_name}.{H}.{roi}.npy'))

    def _make_hrf_table(self):
        """
        Create hrf_params.tsv if missing and ensure the current participant row exists.
        """

        # Columns for the 7 SPM HRF parameters
        cols = ['sn', 'P']

        path = os.path.join(self.glm_path, 'hrf_params.tsv')

        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
        else:
            df = pd.DataFrame(columns=cols)

        # Add missing columns if file exists but is incomplete
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        # Add participant row if missing
        if self.sn not in df.sn.values:
            new_row = {c: np.nan for c in cols}
            new_row['sn'] = self.sn
            new_row['P'] = ",".join(map(str, self.P0))
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(path, sep='\t', index=False)

        return df

    def save_P_to_table(self, P):
        """
        Save optimized P for this participant into hrf_params.tsv.
        """

        if P.shape != (7,):
            raise ValueError("P must have shape (7,)")

        idx = self.df.index[self.df.sn == self.sn]
        self.df.loc[idx, 'P'] = ",".join(map(str, P))
        self.df.to_csv(os.path.join(self.glm_path, 'hrf_params.tsv'), sep='\t', index=False)

    def manual(self, P, pre=6, post=18):
        hrf, _ = spm.spm_hrf(1, P=P)
        self.SPM.convolve_glm(hrf)
        _, info, _, y_hat, y_adj, _ = self.SPM.rerun_glm(self.y_raw)
        y_cut_hat = spm.cut(y_hat, pre=pre, at=self.onset, post=post, padding='last')
        y_cut_adj = spm.cut(y_adj, pre=pre, at=self.onset, post=post, padding='last')

        return y_hat, y_adj, y_cut_hat, y_cut_adj

    def gridsearch(self):

        print('optimising HRF parameters...')

        grid = {
            0: np.array([3., 4., 5., 6., 7., 8., 9.]),  # delay response
            1: np.array([10., 12., 14., 16., 18.]),  # delay undershoot
            2: np.array([.6, .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]),  # dispersion response
            3: np.array([.6, .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]),  # dispersion undershoot
            4: np.array([1., 2., 3., 4., 5., 6.]),  # ratio
            5: np.array([0.0]),  # onset
            6: np.array([32.0])  # length
        }
        P, _, _ = hrf.grid_search_hrf(self.SPM, self.y_raw, TR=im.TR, grid=grid)
        print(f'optimisation complete, P={P}')

        df = pd.DataFrame(P)
        #df.to_csv(os.path.join(self.glm_path, 'P.tsv'), sep='\t', index=False)

    def powell(self, P0=None):
        if P0 is None:
            P0 = self.P0

        # Precompute once: this does NOT depend on HRF parameters
        y_filt = self.SPM.spm_filter(self.SPM.weight @ self.y_raw)

        # Choose regressors of interest explicitly
        idx = self.SPM.reg_of_interest[:-1]

        def _objective(P):
            P_star = P0.copy()
            P_star[[0, 1, 2, 3, 4]] = P

            hrf, _ = spm.spm_hrf(self.TR, P=P_star)
            self.SPM.convolve_glm(hrf)

            beta = self.SPM.pinvX @ y_filt
            y_hat = self.SPM.design_matrix[:, idx] @ beta[idx, :]
            residuals = y_filt - self.SPM.design_matrix @ beta
            y_adj = y_hat + residuals

            # intervals = [(0,12), (102,124), (214, 236)]
            # for start, end in intervals:
            #     y_adj[start:end] = np.nan
            #     y_hat[start:end] = np.nan

            R2 = calc_R2(y_adj, y_hat)

            print(f"Testing P={P_star}, R2={R2:.5f}")

            return -R2

        res = minimize(
            _objective,
            x0=P0[[0, 1, 2, 3, 4]],
            method="Powell",
            bounds=[(3, 9), (10, 24), (.6, 2.4), (.6, 2.4), (1, 8)],
            options={'disp': True, 
                    "maxiter": 30,
                    "maxfev": 5000,
                    "xtol": 1e-7,
                    "ftol": 1e-7,},
            
        )

        best_P = P0.copy()
        best_P[[0, 1, 2, 3, 4]] = res.x
        best_R2 = -res.fun
        print(f'optimisation complete, P={best_P}')

        self.save_P_to_table(best_P)

        return best_P, best_R2, res


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
            np.save(os.path.join(path_hrf, f'y_raw.glm{glm}.{atlas_name}.{H}.{roi}.npy'), y_scl)
            np.save(os.path.join(path_hrf, f'y_adj.glm{glm}.{atlas_name}.{H}.{roi}.npy'), y_adj)
            np.save(os.path.join(path_hrf, f'y_hat.glm{glm}.{atlas_name}.{H}.{roi}.npy'), y_hat)
            np.save(os.path.join(path_hrf, f'y_filt.glm{glm}.{atlas_name}.{H}.{roi}.npy'), y_filt)


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


