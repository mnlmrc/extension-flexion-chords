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


def smooth_cifti_contrasts(sns, glm, stat='con'):

    def _chord_of(col):
        """Chord ID (a 5-digit sequence) encoded in a CIFTI scalar-row name, e.g. 'sess03_21911,...'."""
        match = re.search(r'\d{5}', col)
        if match is None:
            raise ValueError(f"No 5-digit chord ID found in column name: {col!r}")
        return match.group()


    def _session_groups(row_axis, glm, trained, untrained):
        """Group scalar-row names by session x (trained / untrained)[ x rep].

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
                label = f'sess{s:02d},{name}'
                labels.append(label)
                groups.append(cols)

        return labels, groups

    data = []
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
    
    # difference between trained and untrained
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