import globals.path as pth
import globals.imaging as im
import nitools.spm as spm
import numpy as np
import pandas as pd
import os

if __name__=='__main__':
    sns = [101, 102, 103, 104, 105, 106, 107, 108]
    atlas = 'ROI'
    TR = 1
    nTR = 336   
    records = []

    for glm in [1, 3]:
        for H in im.Hem:
            for roi in im.rois[atlas]:
                y_cut_adj = np.zeros((31, len(sns)))
                y_cut_hat = np.zeros_like(y_cut_adj)
                for s, sn in enumerate(sns):

                    print(f'doing participant {sn}, {H}, {roi}, glm{glm}...')
                    pinfo = pd.read_csv(os.path.join(pth.baseDir, 'participants.tsv'), sep='\t')
                    func_runs = pinfo.loc[pinfo.participant_id == f"subj{sn}", "FuncRuns_day3"].iloc[0].split('.')
                    func_runs = np.array(func_runs, dtype=int)
                    func_runs = np.array([func_runs + func_runs.size * i for i in range(3)]).flatten()
                    events = pd.read_csv(os.path.join(pth.baseDir, pth.behavDir, 'day3', f'efc4_subj{sn}_glm{glm}_events.tsv'), sep='\t')
                    events = events[events.BN.isin(func_runs)]
                    BN = events.BN.to_numpy() - 1
                    onset_b = events.Onset.to_numpy()
                    onset = (np.round(onset_b * TR) + BN * nTR).astype(int)
                    onset = np.sort(onset)

                    y_adj = np.load(os.path.join(pth.baseDir, 'hrf', f'subj{sn}', f'y_adj.glm{glm}.ROI.{H}.{roi}.npy'))
                    y_hat = np.load(os.path.join(pth.baseDir, 'hrf', f'subj{sn}', f'y_hat.glm{glm}.ROI.{H}.{roi}.npy'))
                    y_cut_adj[:, s] = spm.cut(y_adj, pre=6, at=onset, post=24, padding='last').mean(axis=(0, 2))
                    y_cut_hat[:, s] = spm.cut(y_hat, pre=6, at=onset, post=24, padding='last').mean(axis=(0, 2))

                col_adj = [f'subj{sn}_y_adj' for sn in sns]
                col_hat = [f'subj{sn}_y_hat' for sn in sns]

                df_block = pd.DataFrame(np.hstack([y_cut_adj, y_cut_hat]), columns=col_adj+col_hat)
                df_block['time'] = np.linspace(-6, 24, y_cut_adj.shape[0])
                df_block['Hem'] = H
                df_block['roi'] = roi

                records.append(df_block)

        df = pd.concat(records, ignore_index=True)
        col_adj = [f'subj{sn}_y_adj' for sn in sns]
        col_hat = [f'subj{sn}_y_hat' for sn in sns]
        df_melt_adj = pd.melt(df, id_vars=['roi', 'Hem', 'time'], value_vars=col_adj, var_name='participant_id', value_name='hrf')
        df_melt_hat = pd.melt(df, id_vars=['roi', 'Hem', 'time'], value_vars=col_hat, var_name='participant_id', value_name='hrf')
        df_melt_adj['type'] = 'y_adj'
        df_melt_hat['type'] = 'y_hat'
        df_melt = pd.concat([df_melt_adj, df_melt_hat])
        df_melt.to_csv(os.path.join(pth.baseDir, 'hrf', f'hrf.glm{glm}.{atlas}.tsv'), sep='\t', index=False)
