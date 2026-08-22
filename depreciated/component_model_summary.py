import pandas as pd
import os
import pickle
import numpy as np
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.pcm import make_models
import itertools

def make_component_model_dataframe(sns=None, glm=3, atlas_name='ROI'):
    """Build the component model dataframe from the pickled ``theta_in`` files.

    """
    sns           = gl.participants if sns is None else sns
    _, comp_names = make_models(sns[0])          # component names don't depend on the subject
    comp_names    = comp_names + ['base']

    df = pd.DataFrame()
    for sn, session, H, roi in itertools.product(sns, gl.sessions, gl.Hem, gl.rois[atlas_name]):
        
        path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', 
            f'component_model.theta_in.{atlas_name}.glm{glm}.{session}.{H}.{roi}.p')
        with open(path, 'rb') as f: 
            theta_in = pickle.load(f)

        weight = np.asarray(np.exp(theta_in)).ravel()
        if len(weight) > len(comp_names):                    # trailing params are noise scales
            labels = comp_names + ['noise'] * (len(weight) - len(comp_names))
        else:
            labels = comp_names[:len(weight)]

        df_tmp = pd.DataFrame({'weight': weight, 'component': labels})
        df_tmp['sn']      = sn
        df_tmp['Hem']     = H
        df_tmp['roi']     = roi
        df_tmp['session'] = session
        df = pd.concat([df, df_tmp], ignore_index=True)

    return df


if __name__=='__main__':
    sns        = gl.participants
    glm        = 3
    atlas_name = 'ROI'
    comp_model = make_component_model_dataframe(sns=sns, glm=glm, atlas_name=atlas_name)

    comp_model.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'component_model.{atlas_name}.glm{glm}.tsv'), sep='\t')