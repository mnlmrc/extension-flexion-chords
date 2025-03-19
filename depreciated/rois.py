import sys
import os
import globals as gl
import nitools as nt
import nibabel as nb

import argparse
import time

import Functional_Fusion.atlas_map as am

import numpy as np


def main(args):

    atlas, _ = am.get_atlas('fs32k')

    if args.what=='make_rois':

        Hem = ['L', 'R']

        for h, H in enumerate(Hem):

            g_atlas = nb.load(os.path.join(gl.atlas_dir, f'{args.atlas}.32k.{Hem[h]}.label.gii'))

            labels = {
                ele.key: getattr(ele, 'label', '')
                for ele in g_atlas.labeltable.labels
            }

            amap = list()
            for nlabel, label in enumerate(labels.values()):
                print(f'making ROI: {label}, {H}')

                atlas_hem = atlas.get_hemisphere(h)
                subatlas = atlas_hem.get_subatlas_image(os.path.join(gl.atlas_dir,
                                                                     f'{args.atlas}.32k.{H}.label.gii'), nlabel)

                subj_dir = os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{args.sn}')
                white = os.path.join(subj_dir, f'subj{args.sn}.{H}.white.32k.surf.gii')
                pial = os.path.join(subj_dir, f'subj{args.sn}.{H}.pial.32k.surf.gii')
                mask = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}',f'subj{args.sn}', 'mask.nii')
                amap_tmp = am.AtlasMapSurf(subatlas.vertex[0], white, pial, mask)
                amap_tmp.build()

                # add roi name
                amap_tmp.name = label

                # add number of voxels
                amap_tmp.n_voxels = len(np.unique(amap_tmp.vox_list))

                amap.append(amap_tmp)

            print('excluding voxels...')
            amap = am.exclude_overlapping_voxels(amap, exclude=[(1, 2), (1, 6), (1, 7),
                                                             (2, 3), (2, 4), (2, 5), (2, 7),
                                                             (3, 4), (3, 5),
                                                             (7, 8)])

            roiMasks = []
            for amap_tmp in amap:
                print(f'saving ROI {amap_tmp.name}, {H}')
                mask_out = amap_tmp.save_as_image(os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'day{args.day}',
                                                               f'subj{args.sn}', f'{args.atlas}.{H}.{amap_tmp.name}.nii'))
                if len(amap_tmp.name) > 0:
                    roiMasks.append(os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'day{args.day}', f'subj{args.sn}',
                                                               f'{args.atlas}.{H}.{amap_tmp.name}.nii'))


            am.parcel_combine(roiMasks,os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'day{args.day}', f'subj{args.sn}',
                                                                   f'{args.atlas}.{H}.nii'))

    if args.what=='make_hemispheres':

        Hem = ['L', 'R']

        amap = []
        for h, H in enumerate(Hem):
            atlas_hem = atlas.get_hemisphere(h)

            subj_dir = os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{args.sn}')
            white = os.path.join(subj_dir, f'subj{args.sn}.{H}.white.32k.surf.gii')
            pial = os.path.join(subj_dir, f'subj{args.sn}.{H}.pial.32k.surf.gii')
            mask = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'day{args.day}',f'subj{args.sn}', 'mask.nii')
            amap_tmp = am.AtlasMapSurf(atlas_hem.vertex[0], white, pial, mask)

            print(f'building hemisphere: {H}')
            amap_tmp.build()

            # add hem name
            amap_tmp.name = H

            # add number of voxels
            amap_tmp.n_voxels = len(np.unique(amap_tmp.vox_list))

            amap.append(amap_tmp)

        print('excluding overlapping voxels...')
        amap = am.exclude_overlapping_voxels(amap, exclude=[(0, 1)])
        for amap_tmp, H in zip(amap, Hem):
            print(f'saving hemisphere {amap_tmp.name}')
            mask_out = amap_tmp.save_as_image(os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'day{args.day}' ,
                                                           f'subj{args.sn}', f'Hem.{H}.nii'))




if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='make_rois')
    parser.add_argument('--experiment', type=str, default='efc4')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--day', type=int, default=1)
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()

    main(args)
    finish = time.time()

    print(f'Execution time:{finish-start} s')

