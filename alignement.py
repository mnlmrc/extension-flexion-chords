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
    #nSess = 3
    atlas = 'Hem'
    Hem = ['L', 'R']
    #struct = ['CortexLeft', 'CortexRight']
    path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
    path_rois = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}')
    path_alignment = os.path.join(gl.baseDir, args.experiment, 'alignment', f'subj{args.sn}')
    os.makedirs(path_alignment, exist_ok=True)
    if args.what=='intercept':
        cifti = nb.load(os.path.join(path_glm, 'intercept.dscalar.nii'))
        betas = nt.volume_from_cifti(cifti)
        coords_merged = []
        for h, H in enumerate(Hem):
            #for r, roi in enumerate(rois):
            mask = nb.load(os.path.join(path_rois, f'{atlas}.{H}.nii'))
            coords = nt.get_mask_coords(mask)
            coords_merged.append(coords)
        coords = np.concatenate(coords_merged, axis=1)
        X = nt.sample_image(betas, coords[0], coords[1], coords[2], interpolation=0).T
        X = X[:, (np.isnan(X).sum(axis=0) == 0) & (np.any(np.abs(X) > 1e-8, axis=0))]
        Xc = X - X.mean(axis=1, keepdims=True)
        cov = Xc @ Xc.T
        std = np.sqrt(np.diag(cov))  # vector of standard deviations
        r = cov / np.outer(std, std)  # elementwise division
        np.save(os.path.join(gl.baseDir, path_alignment, f'corr.intercept.merged.npy'), r)
    if args.what=='intercept_SML':
        sns = [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        for sn in sns:
            path = os.path.join(gl.baseDir, args.experiment,'SML', 'glmSess1', f's{sn:02d}')
            cifti = nb.load(os.path.join(path, 'intercept.dscalar.nii'))
            betas = nt.volume_from_cifti(cifti)
            coords_merged = []
            for h, H in enumerate(Hem):
                mask = nb.load(os.path.join(path, f'{atlas}.{H}.nii'))
                coords = nt.get_mask_coords(mask)
                coords_merged.append(coords)
            coords = np.concatenate(coords_merged, axis=1)
            X = nt.sample_image(betas, coords[0], coords[1], coords[2], interpolation=0).T
            X = X[:, (np.isnan(X).sum(axis=0) == 0) & (np.any(np.abs(X) > 1e-8, axis=0))]
            Xc = X - X.mean(axis=1, keepdims=True)
            cov = Xc @ Xc.T
            std = np.sqrt(np.diag(cov))  # vector of standard deviations
            r = cov / np.outer(std, std)  # elementwise division
            path_save = os.path.join(gl.baseDir,args.experiment, 'SML', 'alignment', f's{sn:02d}')
            os.makedirs(path_save, exist_ok=True)
            np.save(os.path.join(path_save, f'corr.intercept.merged.npy'), r)
    if args.what == 'intercept_all':
        for sn in args.sns:
            print(f'doing participant {sn}')
            args = argparse.Namespace(
                what='intercept',
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


