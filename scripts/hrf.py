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

    GLMs = [1]
    for glm in GLMs:

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
                adj, hat = [], []
                for sn in sns:
                    print(f'doing participant {sn}, {H}, {roi}, glm {glm}, fitted')

                    # retrieve onsets
                    onset = load_glm_onset(sn, glm)

                    HRF = Optimise_HRF(sn=sn, glm=glm, H=H)
                    y_cut_hat, y_cut_adj = HRF.cut(pre=3, roi=roi, post=16)

                    hat.append(y_cut_hat.mean(axis=(0, 2)))
                    adj.append(y_cut_adj.mean(axis=(0, 2)))
                    
                adj = np.array(adj)
                hat = np.array(hat)

                df = pd.concat([df, pd.DataFrame(np.c_[adj.T, tAx], columns=sns + ['time']).assign(kind='adj', Hem=H, roi=roi)])
                df = pd.concat([df, pd.DataFrame(np.c_[hat.T, tAx], columns=sns + ['time']).assign(kind='hat', Hem=H, roi=roi)])
                
        df.to_csv(os.path.join(gl.baseDir, f'glm{glm}', 'hrf_fitted.tsv'), sep='\t', index=False)

    # sns = [101, 102, 103, 104, 105, 106, 107, 108]
    # atlas = 'ROI'
    # TR = 1
    # nTR = 336   
    # records = []
    # for glm in [1, 3]:
    #     for H in gl.Hem:
    #         for roi in gl.rois[atlas]:
    #             y_cut_adj = np.zeros((31, len(sns)))
    #             y_cut_hat = np.zeros_like(y_cut_adj)
    #             for s, sn in enumerate(sns):

    #                 print(f'doing participant {sn}, {H}, {roi}, glm{glm}...')
    #                 pinfo = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    #                 func_runs = pinfo.loc[pinfo.participant_id == f"subj{sn}", "FuncRuns_day3"].iloc[0].split('.')
    #                 func_runs = np.array(func_runs, dtype=int)
    #                 func_runs = np.array([func_runs + func_runs.size * i for i in range(3)]).flatten()
    #                 events = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, 'day3', f'efc4_subj{sn}_glm{glm}_events.tsv'), sep='\t')
    #                 events = events[events.BN.isin(func_runs)]
    #                 BN = events.BN.to_numpy() - 1
    #                 onset_b = events.Onset.to_numpy()
    #                 onset = (np.round(onset_b * TR) + BN * nTR).astype(int)
    #                 onset = np.sort(onset)

    #                 y_adj = np.load(os.path.join(gl.baseDir, 'hrf', f'subj{sn}', f'y_adj.glm{glm}.ROI.{H}.{roi}.npy'))
    #                 y_hat = np.load(os.path.join(gl.baseDir, 'hrf', f'subj{sn}', f'y_hat.glm{glm}.ROI.{H}.{roi}.npy'))
    #                 y_cut_adj[:, s] = spm.cut(y_adj, pre=6, at=onset, post=24, padding='last').mean(axis=(0, 2))
    #                 y_cut_hat[:, s] = spm.cut(y_hat, pre=6, at=onset, post=24, padding='last').mean(axis=(0, 2))

    #             col_adj = [f'subj{sn}_y_adj' for sn in sns]
    #             col_hat = [f'subj{sn}_y_hat' for sn in sns]

    #             df_block = pd.DataFrame(np.hstack([y_cut_adj, y_cut_hat]), columns=col_adj+col_hat)
    #             df_block['time'] = np.linspace(-6, 24, y_cut_adj.shape[0])
    #             df_block['Hem'] = H
    #             df_block['roi'] = roi

    #             records.append(df_block)

    #     df = pd.concat(records, ignore_index=True)
    #     col_adj = [f'subj{sn}_y_adj' for sn in sns]
    #     col_hat = [f'subj{sn}_y_hat' for sn in sns]
    #     df_melt_adj = pd.melt(df, id_vars=['roi', 'Hem', 'time'], value_vars=col_adj, var_name='participant_id', value_name='hrf')
    #     df_melt_hat = pd.melt(df, id_vars=['roi', 'Hem', 'time'], value_vars=col_hat, var_name='participant_id', value_name='hrf')
    #     df_melt_adj['type'] = 'y_adj'
    #     df_melt_hat['type'] = 'y_hat'
    #     df_melt = pd.concat([df_melt_adj, df_melt_hat])
    #     df_melt.to_csv(os.path.join(gl.baseDir, 'hrf', f'hrf.glm{glm}.{atlas}.tsv'), sep='\t', index=False)
