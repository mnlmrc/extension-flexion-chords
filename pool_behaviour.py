import globals.path as pth
import pandas as pd
import os

if __name__ == '__main__':
    sns = [101, 102, 103, 104, 105, 106, 107, 108]
    data_pooled = pd.DataFrame()
    for sn in sns:
        for sess in range(24):
            print(f'doing participant {sn}, day {sess + 1}')
            data = pd.read_csv(os.path.join(pth.baseDir, pth.behavDir, f'day{sess+1}',
                         f'efc4_{sn}_single_trial.tsv'), sep = '\t',)
            data_pooled = pd.concat([data_pooled, data])
    data_pooled.to_csv(os.path.join(pth.baseDir, pth.behavDir, f'single_trial_behaviour.tsv'), sep = '\t', index=False)