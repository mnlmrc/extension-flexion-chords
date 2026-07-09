import pandas as pd
import nibabel as nb
import os
import EFC_learningfMRI.globals as gl
import nitools as nt
from EFC_learningfMRI.util import get_trained_and_untrained
import statsmodels.formula.api as smf


import re
import numpy as np
import pandas as pd
import itertools

def suppression_table(result):
    """
    Computes (1 - slope) and its SE for every session x chord cell,
    inferring factor levels directly from the fitted model.
    """

    # grab the original dataframe the model was fit on
    df = result.model.data.frame

    # all coefficient names in the fitted model
    param_names = result.params.index

    # regex to pull out non-reference levels mentioned in coefficient names
    # e.g. 'session[T.sess09]' -> 'sess09'
    session_in_names = set(re.findall(r'session\[T\.(\w+)\]', ' '.join(param_names)))
    chord_in_names = set(re.findall(r'chord\[T\.(\w+)\]', ' '.join(param_names)))

    # all actual levels present in the data
    all_sessions = df['session'].unique()
    all_chords = df['chord'].unique()

    # the reference level is whichever one does NOT appear in the coefficient names
    session_ref = [s for s in all_sessions if s not in session_in_names][0]
    chord_ref = [c for c in all_chords if c not in chord_in_names][0]

    rows = []

    # loop over every combination of session and chord
    for session, chord in itertools.product(all_sessions, all_chords):

        # start with just the R1 main effect, which is always in the slope
        coefs = {'R1': 1}

        # if this session is not the reference, add its R1:session interaction term
        if session != session_ref:
            coefs[f'R1:session[T.{session}]'] = 1

        # if this chord is not the reference, add its R1:chord interaction term
        if chord != chord_ref:
            coefs[f'R1:chord[T.{chord}]'] = 1

        # if BOTH session and chord are non-reference, add the three-way term too
        if session != session_ref and chord != chord_ref:
            coefs[f'R1:session[T.{session}]:chord[T.{chord}]'] = 1

        # build a weight vector L that picks out exactly these coefficients
        L = np.zeros(len(result.params))
        for name, weight in coefs.items():
            L[result.params.index.get_loc(name)] = weight

        # slope estimate for this cell = weighted sum of the relevant coefficients
        slope_est = L @ result.params

        # SE of that sum, using the full covariance matrix
        slope_se = np.sqrt(L @ result.cov_params() @ L)

        rows.append({
            'session': session,
            'chord': chord,
            'suppression': 1 - slope_est,
            'se': slope_se
        })

    return pd.DataFrame(rows)



def _make_long_format(cifti, mask):

    coords = nt.get_mask_coords(mask)

    vol = nt.volume_from_cifti(cifti)
    data = nt.sample_image(vol, coords[0], coords[1], coords[2], interpolation=0).T

    cond = pd.Series(cifti.header.get_axis(0).name)[::2]
    chordID_sess = cond.str.split(',', expand=True)
    chordID, sess = chordID_sess[0].to_numpy(), chordID_sess[1].to_numpy()

    R1 = data[0::2, :]
    R2 = data[1::2, :]
    n_cond, n_vox = R1.shape

    rows = []
    for c in range(n_cond):
        for v in range(n_vox):
            rows.append({
                'chordID': chordID[c],
                'session': sess[c],
                'voxel': v,
                'R1': R1[c, v],
                'R2': R2[c, v]
            })

    return pd.DataFrame(rows)


def _concat_long(sns, glm, atlas='ROI'):

    rois = gl.rois[atlas]

    df = pd.DataFrame()
    for sn in sns:
        cifti = nb.load(os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'contrast.dscalar.nii'))
        for H in gl.Hem:
            for roi in rois:
                
                print(f'doing participant {sn}, {H}, {roi}...')

                mask = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'ROI.{H}.{roi}.nii')
                df_tmp = _make_long_format(cifti, mask)
                df_tmp['sn'] = sn
                df_tmp['Hem'] = H
                df_tmp['roi'] = roi
        
                trained = get_trained_and_untrained(sn)[:4]
                df_tmp['chord'] = 'untrained'
                df_tmp.loc[df_tmp.chordID.isin(trained), 'chord'] = 'trained'

                df = pd.concat([df, df_tmp])

    return df


def save_rep_contrast_long(sns, glm, atlas_name='ROI'):
    df = _concat_long(sns, glm)
    df.to_csv(os.path.join(gl.baseDir, f'glm{glm}', f'{atlas_name}.rep_suppr.long.tsv'), sep='\t', index=False)
    return df


def _fit_lme_in_roi(df, H, roi):

    df_g = df.groupby(['session', 'voxel', 'Hem', 'roi', 'chord']).mean(numeric_only=True).reset_index()

    df_g = df_g[(df_g.roi==roi) & (df_g.Hem==H)].reset_index()
    df_g = df_g.dropna(subset=['R2', 'R1', 'session', 'chord', 'sn'])

    model = smf.mixedlm(
        "R2 ~ R1 * session * chord",
        data=df_g,
        groups=df_g["sn"],
        #re_formula="~R1"   # random slope + intercept for R1 by subject
    ).fit()

    print('===================================================================================')
    print(f'==================================={roi},{H}======================================')
    print('===================================================================================')
    print(model.summary())

    return model

def fit_lme(df, glm, atlas_name='ROI'):
    suppr_est = pd.DataFrame()
    for H in gl.Hem:
        for roi in gl.rois[atlas_name]:
            model=_fit_lme_in_roi(df, H, roi)
            tmp = suppression_table(model)
            tmp['roi'] = roi
            tmp['Hem'] = H
            suppr_est = pd.concat([tmp, suppr_est])

    suppr_est.to_csv(os.path.join(gl.baseDir, f'glm{glm}', f'{atlas_name}.rep_suppr_lme.tsv'), sep='\t', index=False)


def params_to_df(model):
    """Collect the estimated parameters of a fitted (Mixed)LM into a tidy DataFrame.

    One row per term (fixed effects plus any variance components), with the
    coefficient, standard error, z-statistic, p-value and 95% CI.
    """
    ci = model.conf_int()
    out = pd.DataFrame({
        'term': model.params.index,
        'coef': model.params.to_numpy(),
        'se': model.bse.to_numpy(),
        'z': model.tvalues.to_numpy(),
        'pval': model.pvalues.to_numpy(),
        'ci_low': ci.iloc[:, 0].to_numpy(),
        'ci_high': ci.iloc[:, 1].to_numpy(),
    }).reset_index(drop=True)
    return out






