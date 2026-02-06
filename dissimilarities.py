import PcmPy as pcm
import globals as gl
import os
import argparse
import pandas as pd
import numpy as np
from imaging_pipelines.model import calc_prewhitened_betas
import nibabel as nb
import nitools as nt
import xarray as xr
import time
from util import get_trained_and_untrained

def calc_G_chord_set(sn, H, roi, path_glm, path_roi, type='set', n_sess=None):
    # get trained chords
    trained_untrained = np.array(get_trained_and_untrained(sn)).astype(int)
    label = ['trained'] * 4 + ['untrained'] * 4
    if type == 'set':
        chordID_mapping = dict(zip(trained_untrained, label))
    elif type == 'chord':
        chordID_mapping = dict(zip(trained_untrained, np.arange(8)))
    else:
        raise ValueError("Wrong type. Use 'set; for trained vs. untrained and 'chord' for individual chords.")

    # load reginfo
    reginfo = pd.read_csv(os.path.join(path_glm, f'subj{sn}', 'reginfo.tsv'), sep='\t')
    sess = reginfo.name.str.split(',', n=1, expand=True)[0]
    chordID = reginfo.name.str.split(',', n=1, expand=True)[1]
    chord = chordID.astype(int).map(chordID_mapping)
    part_vec = (reginfo.run % 10).to_numpy()
    cond_vec = sess + ',' + chord.astype(str)

    # load betas
    betas = nb.load(os.path.join(path_glm, f'subj{sn}', 'beta.dscalar.nii'))
    betas = nt.volume_from_cifti(betas)

    # load residuals
    residuals = nb.load(os.path.join(path_glm, f'subj{sn}', f'ResMS.nii'))

    mask = nb.load(os.path.join(path_roi, f'subj{sn}', f'ROI.{H}.{roi}.nii'))
    betas_prewhitened = calc_prewhitened_betas(betas, residuals, mask)
    if n_sess is not None:
        sess = sess.to_numpy().astype(int)
        betas_prewhitened = betas_prewhitened[sess==n_sess]
        part_vec = part_vec[sess==n_sess]
        cond_vec = cond_vec[sess==n_sess]

    G, _ = pcm.est_G_crossval(betas_prewhitened, cond_vec, part_vec, X=pcm.indicator(part_vec))

    return G

def main(args):
    atlas = 'ROI'
    Hem = ['L', 'R']
    rois = gl.rois[atlas]
    path_glm = os.path.join(gl.baseDir, f'{gl.glmDir}{args.glm}')
    path_roi = os.path.join(gl.baseDir, gl.roiDir)
    path_pcm = os.path.join(gl.baseDir, gl.pcmDir)
    if args.what=='calc_G_set':
        for h, H in enumerate(Hem):
            for r, roi in enumerate(rois):
                G = []
                for s, sn in enumerate(args.sns):
                    print(f'doing participant {sn}...')
                    G_tmp = calc_G_chord_set(sn, H, roi, path_glm, path_roi, type='set')
                    G.append(G_tmp)
                G = np.array(G)
                np.save(os.path.join(path_pcm, f'G_obs.trained-untrained.{H}.{roi}.npy'), G)
    if args.what=='calc_G_chord':
        for h, H in enumerate(Hem):
            for r, roi in enumerate(rois):
                G = []
                for s, sn in enumerate(args.sns):
                    print(f'doing participant {sn}...')
                    G_tmp = calc_G_chord_set(sn, H, roi, path_glm, path_roi, type='chord')
                    G.append(G_tmp)
                G = np.array(G)
                np.save(os.path.join(path_pcm, f'G_obs.chord-session.{H}.{roi}.npy'), G)
    if args.what=='calc_G_pattern':
        for sess in [3, 9, 23]:
            for h, H in enumerate(Hem):
                for r, roi in enumerate(rois):
                    G = []
                    for s, sn in enumerate(args.sns):
                        print(f'doing participant {sn}, {H}, {roi}...')
                        G_tmp = calc_G_chord_set(sn, H, roi, path_glm, path_roi, type='chord', n_sess=sess)
                        G.append(G_tmp)
                    G = np.array(G)
                    np.save(os.path.join(path_pcm, f'G_obs.chord.{sess}.{H}.{roi}.npy'), G)

    if args.what=='representation_stability':
        repr_dict = {
            'sn': [],
            'roi': [],
            'Hem': [],
            'corr': [],
            'corr_type': [],
            'encoding': [],
        }
        N = len(args.sns)
        mask = np.tri(8, k=-1, dtype=bool)
        for h, H in enumerate(Hem):
            for r, roi in enumerate(rois):
                D = []
                for sess in [3, 9, 23]:
                    G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'G_obs.chord.{sess}.{H}.{roi}.npy'))
                    Dd = pcm.G_to_dist(G)
                    D.append(Dd[:, mask])
                D = np.vstack(D)
                encoding = D.mean(axis=1)
                corr = np.corrcoef(D)
                corr12 = np.diagonal(corr[:N, N:N * 2])
                corr23 = np.diagonal(corr[N:N * 2, N * 2:N * 3])
                corr13 = np.diagonal(corr[:N, N * 2:N * 3])
                corr_s = np.hstack((corr12, corr13, corr23)).T
                repr_dict['sn'].extend(args.sns * 3)
                repr_dict['roi'].extend([roi] * (N * 3))
                repr_dict['Hem'].extend([H] * (N * 3))
                repr_dict['corr'].extend(corr_s)
                repr_dict['encoding'].extend(encoding)
                repr_dict['corr_type'].extend(['sess 3 vs. 9'] * N + ['sess 3 vs. 23'] * N + ['sess 9 vs. 23'] * N)
        repr = pd.DataFrame(repr_dict)
        repr.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'representational_stability.BOLD.tsv'),sep='\t',index=False)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='EFC_learningfMRI')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[101, 102, 103, 104, 105, 106, 107])
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()
    main(args)
    finish = time.time()

    print(f'Time elapsed: {finish - start} seconds')