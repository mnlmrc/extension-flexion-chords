import os
import re

import numpy as np
import nibabel as nb
import nitools as nt
import SUITPy.flatmap as flatmap

import EFC_learningfMRI.globals as gl

from EFC_learningfMRI.util import get_trained_and_untrained


def project_cifti_to_surface(sn, glm, stat='rep_suppr'):

    cifti = nb.load(os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', f'{stat}.dscalar.nii'))
    vol = nt.volume_from_cifti(cifti, struct_names=gl.struct_cortex)
    cols = cifti.header.get_axis(0).name

    whiteL = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'subj{sn}.L.white.32k.surf.gii')
    pialL = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'subj{sn}.L.pial.32k.surf.gii')
    surfdataL = flatmap.vol_to_surf(vol, inner_surf_gifti=whiteL, outer_surf_gifti=pialL)
    giftiL  = nt.make_func_gifti(surfdataL,column_names=cols, anatomical_struct='CortexLeft')

    nb.save(giftiL, os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'glm{glm}.{stat}.L.func.gii'))

    whiteR = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'subj{sn}.R.white.32k.surf.gii')
    pialR = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'subj{sn}.R.pial.32k.surf.gii')
    surfdataR = flatmap.vol_to_surf(vol, inner_surf_gifti=whiteR, outer_surf_gifti=pialR)
    giftiR  = nt.make_func_gifti(surfdataR,column_names=cols, anatomical_struct='CortexRight')

    nb.save(giftiR, os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'glm{glm}.{stat}.R.func.gii'))


def _chord_of(col):
    """Chord ID (a 5-digit sequence) encoded in a CIFTI scalar-row name, e.g. 'sess03_21911,...'."""
    match = re.search(r'\d{5}', col)
    if match is None:
        raise ValueError(f"No 5-digit chord ID found in column name: {col!r}")
    return match.group()


def _session_groups(row_axis, trained, untrained):
    """Group scalar-row names by session x (trained / untrained).

    Returns ``(labels, groups)`` kept in sync: ``groups[i]`` is the list of
    column names that belong to ``labels[i]``, so the averaged data rows line
    up with the output ``ScalarAxis``.
    """

    chord_sets = {'trained': trained, 'untrained': untrained}

    labels, groups = [], []
    for s in gl.sessions:
        for name, chords in chord_sets.items():
            cols = [col for col in row_axis
                    if f'sess{s:02d}' in col
                    and _chord_of(col) in chords]
            if len(cols) == 0:
                raise ValueError(f'No {name} regressors found for sess{s:02d}')
            labels.append(f'sess{s:02d},{name}')
            groups.append(cols)

    return labels, groups


def _smooth_on_flat(cifti_input, cifti_output):
    """Smooth a dscalar along the fs_LR.32k flat surfaces."""
    nt.smooth_cifti(cifti_input, cifti_output,
                    os.path.join(gl.atlasDir, 'fs_LR.32k.L.flat.surf.gii'),
                    os.path.join(gl.atlasDir, 'fs_LR.32k.R.flat.surf.gii'))


def _session_file(glm, stat, sn=None, smooth=False):
    """Path to the session x (trained / untrained) dscalar, for one participant or the group."""
    path = os.path.join(gl.baseDir, gl.surfDir)
    if sn is not None:
        path = os.path.join(path, f'subj{sn}')
    suffix = '.smooth' if smooth else ''
    return os.path.join(path, f'glm{glm}.{stat}.session{suffix}.dscalar.nii')


def smooth_cifti_contrasts(sn, glm, stat='con'):
    """Average one participant's contrasts within session x (trained / untrained) and smooth them.

    Writes ``glm{glm}.{stat}.session.dscalar.nii`` and its smoothed counterpart
    to the participant's surface folder. Averaging across participants is done
    separately by :func:`average_smoothed_contrasts`.
    """

    print(f'Processing participant {sn}')
    path = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
    giftis = [os.path.join(path, f'glm{glm}.{stat}.{H}.func.gii') for H in gl.Hem]
    cifti_img = nt.join_giftis_to_cifti(giftis)

    trained_untrained  = get_trained_and_untrained(sn)
    trained, untrained = trained_untrained[:4], trained_untrained[4:]

    row_axis = cifti_img.header.get_axis(0).name
    cond, regr = _session_groups(row_axis, trained, untrained) # cond[i] is the condition label (e.g., sess03,trained) for regressors in regr[i]

    fdata = cifti_img.get_fdata()
    data = np.vstack([fdata[np.isin(row_axis, r)].mean(axis=0) for r in regr]) # dimord condition_vertices

    header = nb.Cifti2Header.from_axes((nb.cifti2.ScalarAxis(cond), cifti_img.header.get_axis(1)))

    session_file = _session_file(glm, stat, sn=sn)
    smooth_file = _session_file(glm, stat, sn=sn, smooth=True)
    nb.save(nb.Cifti2Image(dataobj=data, header=header), session_file)
    _smooth_on_flat(session_file, smooth_file)

    return smooth_file


def average_smoothed_contrasts(sns, glm, stat='con'):
    """Average the smoothed per-participant maps written by :func:`smooth_cifti_contrasts`.

    Writes the group session map and the trained - untrained difference in each
    session to the surface folder.
    """

    data, cond = [], None
    for sn in sns:
        cifti_img = nb.load(_session_file(glm, stat, sn=sn, smooth=True))
        labels = list(cifti_img.header.get_axis(0).name)

        if cond is None:
            cond, brain_axis = labels, cifti_img.header.get_axis(1)
        elif labels != cond:
            raise ValueError(f'Conditions of participant {sn} ({labels}) do not match {cond}')

        data.append(cifti_img.get_fdata())

    data = np.nanmean(np.stack(data, axis=0), axis=0) # dimord subj_condition_vertices -> condition_vertices

    header = nb.Cifti2Header.from_axes((nb.cifti2.ScalarAxis(cond), brain_axis))
    nb.save(nb.Cifti2Image(dataobj=data, header=header), _session_file(glm, stat, smooth=True))

    # difference between trained and untrained
    diff_cond = [f'sess{s:02d}' for s in gl.sessions]
    data_diff = np.vstack([data[cond.index(f'{c},trained')] - data[cond.index(f'{c},untrained')]
                           for c in diff_cond])

    header = nb.Cifti2Header.from_axes((nb.cifti2.ScalarAxis(diff_cond), brain_axis))
    nb.save(nb.Cifti2Image(dataobj=data_diff, header=header),
            os.path.join(gl.baseDir, gl.surfDir, f'glm{glm}.{stat}.trained_vs_untrained.smooth.dscalar.nii'))