import os
import globals as gl
import nitools.spm as spm
import time
import argparse
import numpy as np
import pandas as pd
import nibabel as nb
import imaging_pipelines.betas as bt
import nitools as nt

def main(args):
    nSess = 3
    atlas = 'ROI'
    Hem = ['L', 'R']
    struct = ['CortexLeft', 'CortexRight']
    rois = gl.rois[atlas]
    path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
    path_rois = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}')
    path_alignment = os.path.join(gl.baseDir, args.experiment, 'alignment', f'subj{args.sn}')
    os.makedirs(path_alignment, exist_ok=True)
    if args.what=='intercept_ROI':
        cifti = nb.load(os.path.join(path_glm, 'intercept.dscalar.nii'))
        betas = nt.volume_from_cifti(cifti)
        coords_merged = []
        for h, H in enumerate(Hem):
            for r, roi in enumerate(rois):
                mask = nb.load(os.path.join(path_rois, f'{atlas}.{H}.{roi}.nii'))
                coords = nt.get_mask_coords(mask)
                coords_merged.append(coords)
                X = nt.sample_image(betas, coords[0], coords[1], coords[2], interpolation=0).T
                X = X[:, np.isnan(X).sum(axis=0)==0]
                Xc = X - X.mean(axis=1, keepdims=True)
                cov = Xc @ Xc.T
                std = np.sqrt(np.diag(cov))  # vector of standard deviations
                r = cov / np.outer(std, std)  # elementwise division
                np.save(os.path.join(gl.baseDir, path_alignment, f'corr.intercept.{H}.{roi}.npy'), r)
        coords = np.concatenate(coords_merged, axis=1)
        X = nt.sample_image(betas, coords[0], coords[1], coords[2], interpolation=0).T
        X = X[:, (np.isnan(X).sum(axis=0) == 0) & (np.any(np.abs(X) > 1e-8, axis=0))]
        Xc = X - X.mean(axis=1, keepdims=True)
        cov = Xc @ Xc.T
        std = np.sqrt(np.diag(cov))  # vector of standard deviations
        r = cov / np.outer(std, std)  # elementwise division
        np.save(os.path.join(gl.baseDir, path_alignment, f'corr.intercept.merged.npy'), r)
    if args.what == 'intercept_ROI_all':
        for sn in args.sns:
            print(f'doing participant {sn}')
            args = argparse.Namespace(
                what='intercept_ROI',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='EFC_learningfMRI')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', default=[101, 102, 103, 104, 105], type=int)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()

    main(args)
    end = time.time()
    print(f'Finished in {end - start} seconds')


