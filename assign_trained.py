import pandas as pd
import os
import random
import globals as gl

# Load the data
pinfo = pd.read_csv(os.path.join(gl.baseDir, 'efc4', 'participants.tsv'), sep='\t')

# Filter participants
pinfo = pinfo[pinfo.good == 1].reset_index(drop=True)

# Iterate through rows
for r, row in pinfo.iterrows():
    trained_idx = random.sample(range(len(gl.chordID)), 4)
    trained = [gl.chordID[i] for i in trained_idx]
    untrained = [gl.chordID[i] for i in range(len(gl.chordID)) if i not in trained_idx]

    trained_str = ".".join(trained)
    untrained_str = ".".join(untrained)

    # Proper assignment using .at[] or .loc[]
    pinfo.at[r, 'trained'] = trained_str
    pinfo.at[r, 'untrained'] = untrained_str

pinfo.to_csv(os.path.join(gl.baseDir, 'efc4', 'participants_tmp.tsv'), sep='\t', index=False)

