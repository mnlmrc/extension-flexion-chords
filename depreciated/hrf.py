import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.hrf import Optimise_HRF
from EFC_learningfMRI.util import load_glm_onset
import nitools.spm as spm
import numpy as np
import pandas as pd
import os

if __name__=='__main__':

    tAx = np.arange(-3, 17)

    sns = [101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113]

    glm = 3

    # ### save best HRF parameters
    # gridsearch = pd.DataFrame()
    # hrf_params = {'sn': [], 'P': []}
    # for sn in sns:
    #     print(f'doing participant {sn}, glm {glm}...')
    #     gs = pd.read_csv(os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'gridsearch_hrf.tsv'), sep='\t')
    #     gs['sn'] = sn
    #     gridsearch = pd.concat([gridsearch, gs], axis=0)

    #     # find best parameters
    #     gs_avg = gs.groupby(gl.hrf_params)['R_squared'].mean().reset_index()
    #     idxmax = gs_avg.R_squared.argmax()
    #     P = gs_avg.loc[idxmax][gl.hrf_params].to_numpy()
    #     hrf_params['sn'].append(sn)
    #     hrf_params['P'].append(",".join(map(str, P)))
        
    # # save grid search
    # gridsearch.to_csv(os.path.join(gl.baseDir, f'glm{glm}', 'hrf_gridsearch.tsv'), sep='\t', index=False)

    # # save best parameters
    # hrf_params = pd.DataFrame(hrf_params)
    # hrf_params.to_csv(os.path.join(gl.baseDir, f'glm{glm}', 'hrf_params.tsv'), sep='\t', index=False)

    ### save segmented BOLD 
    df = pd.DataFrame()
    for H in ['L']: #gl.Hem:
        for r, roi in enumerate(gl.rois['ROI']):
            adj, hat, raw = [], [], []
            for sn in sns:
                print(f'doing participant {sn}, {H}, {roi}, glm {glm}, fitted')

                # retrieve onsets
                onset = load_glm_onset(sn, glm)

                HRF = Optimise_HRF(sn=sn, glm=glm, H=H)
                y_cut_hat, y_cut_adj, y_cut_raw = HRF.cut(pre=3, roi=roi, post=16)

                raw.append(y_cut_raw.mean(axis=(0, 2)))
                hat.append(y_cut_hat.mean(axis=(0, 2)))
                adj.append(y_cut_adj.mean(axis=(0, 2)))
            
            raw = np.array(raw)
            adj = np.array(adj)
            hat = np.array(hat)

            df = pd.concat([df, pd.DataFrame(np.c_[raw.T, tAx], columns=sns + ['time']).assign(kind='raw', Hem=H, roi=roi)])
            df = pd.concat([df, pd.DataFrame(np.c_[adj.T, tAx], columns=sns + ['time']).assign(kind='adj', Hem=H, roi=roi)])
            df = pd.concat([df, pd.DataFrame(np.c_[hat.T, tAx], columns=sns + ['time']).assign(kind='hat', Hem=H, roi=roi)])
            
    df.to_csv(os.path.join(gl.baseDir, f'glm{glm}', 'hrf_fitted.tsv'), sep='\t', index=False)
