import sys
import os
import globals as gl
from imaging_pipelines import rois

import argparse
import time

import Functional_Fusion.atlas_map as am


def main(args):

    exclude = {
        'ROI': [(1, 2), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5), (2, 7),(3, 4), (3, 5), (7, 8)]
    }

    if args.what=='make_cortical_rois':
        path_surf = os.path.join(gl.baseDir,  gl.surfDir, f'subj{args.sn}')
        white = [os.path.join(path_surf, f'subj{args.sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
        pial = [os.path.join(path_surf, f'subj{args.sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
        mask = os.path.join(gl.baseDir,  f'{gl.glmDir}{args.glm}', f'subj{args.sn}', 'mask.nii')
        atlas_name = 'ROI'
        atlas_dir = gl.atlasDir
        rois_dir = os.path.join(gl.baseDir, gl.roiDir, f'subj{args.sn}')
        Rois = rois.SurfRois(atlas_name, white, pial, mask, atlas_dir, rois_dir)
        Rois.make_hemispheres()
        Rois.make_rois(exclude=exclude[atlas_name])
    if args.what == 'make_cortical_rois_all':
        for sn in args.sns:
            args = argparse.Namespace(
                what='make_cortical_rois',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm,
                atlas=args.atlas,

            )
            main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='make_rois')
    parser.add_argument('--experiment', type=str, default='efc4')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int,default=[101, 102, 103, 104, 105])
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()

    start = time.time()
    main(args)
    finish = time.time()

    print(f'Execution time:{finish-start} s')

