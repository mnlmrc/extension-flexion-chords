import pandas as pd
import nibabel as nb
import os
import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import get_trained_and_untrained
from EFC_learningfMRI.betas import RegInfo
import nitools as nt
from imaging_pipelines.model import calc_prewhitened_betas
import numpy as np
import pandas as pd
import PcmPy as pcm


def fit_pc1(data):

    C, V = data.shape

    theta = []
    for c in range(0, C, 2):
        r1       = data[c]
        r2       = data[c+1]
        R        = np.c_[r1, r2].T
        R        = R[:, ~np.isnan(R).any(axis=0)]
        G        = R @ R.T
        theta.append(0.5 * np.atan2(2*G[0, 1], G[0, 0] - G[1, 1]))

    return np.array(theta)


def axial_mean(theta):
    """Mean of axis orientations (angles defined modulo pi).

    ``fit_pc1`` returns the orientation of a principal *axis*, which is defined
    modulo pi (a line at ``theta`` and one at ``theta + pi`` are identical). A
    plain arithmetic mean is therefore wrong: it mishandles the wraparound and
    collapses near-orthogonal axes. The axial mean doubles the angles (mapping
    them onto the full circle), takes the circular mean, and halves the result.

    Parameters
    ----------
    theta : array_like
        Axis orientations in radians (each in ``(-pi/2, pi/2]``).

    Returns
    -------
    float
        The mean orientation in radians, in ``(-pi/2, pi/2]``.
    """
    return 0.5 * np.arctan2(np.mean(np.sin(2 * theta)), np.mean(np.cos(2 * theta)))


def pc1_theta(sns, glm=None, fname='contrast.dscalar.nii', atlas_name='ROI'):
    """Mean PC1 orientation (theta) per subject / ROI / session / chord group.

    For every ROI the rep1-vs-rep2 point cloud of each chord is summarised by the
    orientation of its first principal axis (``fit_pc1``), an angle defined modulo
    pi. These per-chord orientations are aggregated within each trained/untrained
    group using the axial (double-angle) circular mean (``axial_mean``) rather than
    a plain arithmetic mean, which is invalid for axis orientations.

    This mirrors the per-chord-then-axial-mean logic used to draw the summary fit
    lines in ``notebooks/scratch.ipynb`` (as opposed to ``r2_v_r1_slope``, which
    averages the rep patterns across chords before fitting a single orientation).

    Parameters
    ----------
    sns : iterable of int
        Subject numbers.
    glm : int or str
        GLM number.
    fname : str, optional
        Contrast CIFTI file name (default ``'contrast.dscalar.nii'``).
    atlas_name : str, optional
        ROI atlas key into ``gl.rois`` (default ``'ROI'``).

    Returns
    -------
    pandas.DataFrame
        Columns ``['sn', 'Hem', 'roi', 'session', 'chord', 'theta']``, one row per
        subject / hemisphere / ROI / session / chord group. ``theta`` is the mean
        PC1 orientation in radians.
    """

    path_glm = os.path.join(gl.baseDir, f'glm{glm}')

    df_list = []
    for sn in sns:
        cifti     = nb.load(os.path.join(path_glm, f'subj{sn}', fname))
        contrasts = nt.volume_from_cifti(cifti)
        reginfo = RegInfo(sn, glm)
        chordID = reginfo.condition_unique.loc[::2, 0]
        chordID = chordID.astype(int).map(reginfo.make_chord_mapping)
        session = reginfo.condition_unique.loc[::2, 1]
        for H in gl.Hem:
            for roi in gl.rois[atlas_name]:
                print(f'doing participant {sn}, {H}, {roi}...')
                mask          = nb.load(os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'ROI.{H}.{roi}.nii'))
                coords        = nt.get_mask_coords(mask)
                contrasts_roi = nt.sample_image(contrasts, coords[0], coords[1], coords[2], interpolation=0).T
                theta         = fit_pc1(contrasts_roi)  # one PC1 orientation per chord
                for sess in np.unique(session):
                    for chord in ('trained', 'untrained'):
                        sel = (session == sess) & (chordID < 4 if chord=='trained' else chordID >= 4)
                        df_list.append(pd.DataFrame({
                            'sn'     : [sn],
                            'Hem'    : [H],
                            'roi'    : [roi],
                            'session': [sess],
                            'chord'  : [chord],
                            'theta'  : [axial_mean(theta[sel])],
                        }))

    df = pd.concat(df_list, ignore_index=True)
    df.session = df.session.map({f'sess{s:02d}': s for s in gl.sessions})
    df.to_csv(os.path.join(gl.baseDir, f'glm{glm}', f'{atlas_name}.pc1_theta.tsv'), index=False, sep='\t')
    return df


if __name__=='__main__':
    sns = gl.participants
    glm = 2
    pc1_theta(sns, glm)


    