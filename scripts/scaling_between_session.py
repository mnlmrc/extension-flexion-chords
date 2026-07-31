from EFC_learningfMRI.G_matrix import G_scaling
import itertools
import EFC_learningfMRI.globals as gl
import os
import numpy as np
import pandas as pd

if __name__=='__main__':
    sns = gl.participants
    glm = 3
    atlas_name = 'ROI'
    rois = gl.rois[atlas_name]

    scaling = pd.DataFrame()
    for session, sn, H, roi in itertools.product(gl.sessions, sns, gl.Hem, rois):

        print(f'doing participant {sn}, session {session}, {H}, {roi}...')

        G_ref = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs.within_session.3.glm{glm}.{H}.{roi}.npy'))
        G_tar = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs.within_session.{session}.glm{glm}.{H}.{roi}.npy'))

        row_all       = G_scaling(G_ref, G_tar).assign(chord='all', session=session, Hem=H, roi=roi, sn=sn)
        row_trained   = G_scaling(G_ref[:4, :4], G_tar[:4, :4]).assign(chord='trained', session=session, Hem=H, roi=roi, sn=sn)
        row_untrained = G_scaling(G_ref[4:, 4:], G_tar[4:, 4:]).assign(chord='untrained', session=session, Hem=H, roi=roi, sn=sn)

        scaling = pd.concat([scaling, row_all, row_trained, row_untrained])

    scaling.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'scaling.between_session.glm{glm}.{atlas_name}.tsv'), index=False, sep='\t')