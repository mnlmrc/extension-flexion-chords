import pandas as pd
import os
import globals.path as pth

if __name__=='__main__':
    for atlas in ['ROI', 'BA_handArea']:
        df = pd.read_csv(os.path.join(pth.baseDir, f'glm2', f'{atlas}.con.avg.tsv'), sep='\t')
        df_1 = df[df['rep']==1].reset_index(drop=True)
        df_2 = df[df['rep']==2].reset_index(drop=True)
        df_rep = df_1.copy()
        df_rep['con'] = df_2['con'] - df_1['con']
        df_rep.to_csv(os.path.join(pth.baseDir, f'glm2', f'{atlas}.repetition_suppression.tsv'), sep='\t', index=False)