import os

import numpy as np
import nibabel as nb
import nitools as nt

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.betas import roi_contrasts
from EFC_learningfMRI.util import get_trained_and_untrained


def _chord_of(col):
    """Chord ID encoded in a CIFTI scalar-row name, e.g. 'sess03_21911,...'."""
    return col.split('_')[1].split(',')[0]


def _rep_of(col):
    """Repetition index encoded in a CIFTI scalar-row name."""
    return int(col.split(',')[2].split('.')[0])


def _session_groups(row_axis, glm, trained, untrained):
    """Group scalar-row names by session x (trained / untrained)[ x rep].

    Returns ``(labels, groups)`` kept in sync: ``groups[i]`` is the list of
    column names that belong to ``labels[i]``, so the averaged data rows line
    up with the output ``ScalarAxis``.
    """
    if glm == 2:
        reps = [1, 2]
    elif glm == 3:
        reps = [None]
    else:
        raise ValueError(f"GLM {glm} not found")

    chord_sets = {'trained': trained, 'untrained': untrained}

    labels, groups = [], []
    for rep in reps:
        for s in gl.sessions:
            for name, chords in chord_sets.items():
                cols = [col for col in row_axis
                        if f'sess{s:02d}' in col
                        and _chord_of(col) in chords
                        and (rep is None or _rep_of(col) == rep)]
                label = f'sess{s:02d},{name}'
                if rep is not None:
                    label += f',{rep}'
                labels.append(label)
                groups.append(cols)
    return labels, groups


def smooth_cifti_contrasts(sns, glm, stat='con'):
    data = []
    labels = None
    brain_axis = None
    for i, sn in enumerate(sns):
        print(f'Processing participant {sn}')
        path = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
        giftis = [os.path.join(path, f'glm{glm}.{stat}.{H}.func.gii') for H in gl.Hem]
        cifti_img = nt.join_giftis_to_cifti(giftis)

        trained_untrained = get_trained_and_untrained(sn)
        trained, untrained = trained_untrained[:4], trained_untrained[4:] 

        row_axis = cifti_img.header.get_axis(0).name
        cond, regr = _session_groups(row_axis, glm, trained, untrained) # cond[i] is the condition label (e.g., trained,sess03) for regressors in regr[i]

        fdata = cifti_img.get_fdata()
        subj_data = [fdata[np.isin(row_axis, r)].mean(axis=0) for r in regr]
        data.append(np.vstack(subj_data))

        if i == 0:
            brain_axis = cifti_img.header.get_axis(1)

    data = np.array(data).mean(axis=0) # dimord subj_condition_vertices

    header = nb.Cifti2Header.from_axes((nb.cifti2.ScalarAxis(cond), brain_axis))
    cifti_img = nb.Cifti2Image(dataobj=data, header=header)

    session_file = os.path.join(gl.baseDir, gl.surfDir, f'glm{glm}.{stat}.session.dscalar.nii')
    smooth_file = os.path.join(gl.baseDir, gl.surfDir, f'glm{glm}.{stat}.session.smooth.dscalar.nii')
    nb.save(cifti_img, session_file)
    nt.smooth_cifti(session_file, smooth_file,
                    os.path.join(gl.atlasDir, 'fs_LR.32k.L.flat.surf.gii'),
                    os.path.join(gl.atlasDir, 'fs_LR.32k.R.flat.surf.gii'))
    
    data_diff = data[::2] - data[1::2]

    cond = ['sess03', 'sess09', 'sess23']
    header = nb.Cifti2Header.from_axes((nb.cifti2.ScalarAxis(cond), brain_axis))
    cifti_img = nb.Cifti2Image(dataobj=data_diff, header=header)
    
    session_file = os.path.join(gl.baseDir, gl.surfDir, f'glm{glm}.{stat}.trained_vs_untrained.dscalar.nii')
    smooth_file = os.path.join(gl.baseDir, gl.surfDir, f'glm{glm}.{stat}.trained_vs_untrained.smooth.dscalar.nii')
    nb.save(cifti_img, session_file)
    nt.smooth_cifti(session_file, smooth_file,
                    os.path.join(gl.atlasDir, 'fs_LR.32k.L.flat.surf.gii'),
                    os.path.join(gl.atlasDir, 'fs_LR.32k.R.flat.surf.gii'))


if __name__ == "__main__":
    glm = 3
    sns = [101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113]
    smooth_cifti_contrasts(sns=sns, glm=glm, stat='con')
    roi_contrasts(sns=sns, atlas_name='ROI', glm=glm)
