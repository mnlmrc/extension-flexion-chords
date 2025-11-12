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


def main(args):
    atlas = 'ROI'
    Hem = ['L', 'R']
    rois = gl.rois[atlas]
    path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
    path_roi = os.path.join(gl.baseDir, args.experiment, gl.roiDir)
    if args.what=='calc_G_set':
        G = np.zeros((len(args.sns), 2, len(rois), 6, 6))
        for s, sn in enumerate(args.sns):
            print(f'doing participant {sn}...')

            # get trained chords
            trained_untrained = np.array(get_trained_and_untrained('EFC_learningfMRI', sn)).astype(int)
            label = ['trained'] * 4 + ['untrained'] * 4
            chordID_mapping = dict(zip(trained_untrained, label))

            # load reginfo
            reginfo = pd.read_csv(os.path.join(path_glm, f'subj{sn}', 'reginfo.tsv'), sep='\t')
            day = reginfo.name.str.split(',', n=1, expand=True)[0]
            chordID = reginfo.name.str.split(',', n=1, expand=True)[1]
            chord = chordID.astype(int).map(chordID_mapping)
            part_vec = (reginfo.run % 10).to_numpy()
            cond_vec = day + ',' + chord.astype(str)

            # load betas
            betas = nb.load(os.path.join(path_glm, f'subj{sn}', 'beta.dscalar.nii'))
            betas = nt.volume_from_cifti(betas)
            # betas = betas[..., idx_valid]

            # load residuals
            residuals = nb.load(os.path.join(path_glm, f'subj{sn}', f'ResMS.nii'))

            # loop through rois
            for h, H in enumerate(Hem):
                for r, roi in enumerate(rois):
                    mask = nb.load(os.path.join(path_roi, f'subj{sn}', f'{atlas}.{H}.{roi}.nii'))
                    betas_prewhitened = calc_prewhitened_betas(betas, residuals, mask)
                    # betas_prewhitened = betas_prewhitened[idx_valid]
                    G[s, h, r], _ = pcm.est_G_crossval(betas_prewhitened, cond_vec, part_vec, X=pcm.indicator(part_vec))
        G = xr.DataArray(data=G, dims=('subj', 'Hem', 'roi', 'regr1', 'regr2'),
                         coords={'subj': args.sns, 'Hem': Hem, 'roi': rois,})
        G.to_netcdf(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, 'G_obs.set.h5'), engine="h5netcdf")

    if args.what=='calc_G_chord':
        G = np.zeros((len(args.sns), 2, len(rois), 24, 24))
        for s, sn in enumerate(args.sns):
            print(f'doing participant {sn}...')

            # get trained chords
            trained_untrained = np.array(get_trained_and_untrained('EFC_learningfMRI', sn)).astype(int)
            chordID_mapping = dict(zip(trained_untrained, np.arange(8)))

            # load reginfo
            reginfo = pd.read_csv(os.path.join(path_glm, f'subj{sn}', 'reginfo.tsv'), sep='\t')
            day = reginfo.name.str.split(',', n=1, expand=True)[0]
            chordID = reginfo.name.str.split(',', n=1, expand=True)[1]
            chordID = chordID.astype(int).map(chordID_mapping)
            part_vec = (reginfo.run % 10).to_numpy()
            cond_vec = day + ',' + chordID.astype(str)

            # load betas
            betas = nb.load(os.path.join(path_glm, f'subj{sn}', 'beta.dscalar.nii'))
            betas = nt.volume_from_cifti(betas)
            # betas = betas[..., idx_valid]

            # load residuals
            residuals = nb.load(os.path.join(path_glm, f'subj{sn}', f'ResMS.nii'))

            # loop through rois
            for h, H in enumerate(Hem):
                for r, roi in enumerate(rois):
                    mask = nb.load(os.path.join(path_roi, f'subj{sn}', f'{atlas}.{H}.{roi}.nii'))
                    betas_prewhitened = calc_prewhitened_betas(betas, residuals, mask)
                    # betas_prewhitened = betas_prewhitened[idx_valid]
                    G[s, h, r], _ = pcm.est_G_crossval(betas_prewhitened, cond_vec, part_vec, X=pcm.indicator(part_vec))
        G = xr.DataArray(data=G, dims=('subj', 'Hem', 'roi', 'regr1', 'regr2'),
                         coords={'subj': args.sns, 'Hem': Hem, 'roi': rois, })
        G.to_netcdf(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, 'G_obs.chord.h5'), engine="h5netcdf")

    if args.what=='calc_G_chord_pattern':
        chordID_mapping = dict(zip(gl.chordID, np.arange(8)))
        G = np.zeros((len(args.sns), 2, len(rois), 8, 8))
        for s, sn in enumerate(args.sns):
            print(f'doing participant {sn}...')

            # load reginfo
            reginfo = pd.read_csv(os.path.join(path_glm, f'subj{sn}', 'reginfo.tsv'), sep='\t')
            chordID = reginfo.name.str.split(',', n=1, expand=True)[1]
            day = reginfo.name.str.split(',', n=1, expand=True)[0].astype(int)
            cond_vec = chordID.map(chordID_mapping)
            part_vec = (reginfo.run % 10).to_numpy()

            # load betas
            betas = nb.load(os.path.join(path_glm, f'subj{sn}', 'beta.dscalar.nii'))
            betas = nt.volume_from_cifti(betas)

            # load residuals
            residuals = nb.load(os.path.join(path_glm, f'subj{sn}', f'ResMS.nii'))

            # loop through rois
            for h, H in enumerate(Hem):
                for r, roi in enumerate(rois):
                    mask = nb.load(os.path.join(path_roi, f'subj{sn}', f'{atlas}.{H}.{roi}.nii'))
                    betas_prewhitened = calc_prewhitened_betas(betas, residuals, mask)
                    G[s, h, r], _ = pcm.est_G_crossval(betas_prewhitened, cond_vec, part_vec,
                                                       X=pcm.indicator(part_vec))
        G = xr.DataArray(data=G, dims=('subj', 'Hem', 'roi', 'regr1', 'regr2'),
                         coords={'subj': args.sns, 'Hem': Hem, 'roi': rois, })
        G.to_netcdf(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, 'G_obs.chord_pattern.h5'), engine="h5netcdf")

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='EFC_learningfMRI')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()
    main(args)
    finish = time.time()

    print(f'Time elapsed: {finish - start} seconds')