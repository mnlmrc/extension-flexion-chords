import argparse
from imaging_pipelines.searchlight import searchlight_surf
import globals as gl
import time
import os
import numpy as np

def main(args):
    if args.what == 'make_searchlight':
        path_surf = os.path.join(gl.baseDir, gl.surfDir, f'subj{args.sn}')
        white = [os.path.join(path_surf, f'subj{args.sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
        pial = [os.path.join(path_surf, f'subj{args.sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
        mask = os.path.join(gl.baseDir, f'{gl.glmDir}{args.glm}', f'subj{args.sn}', 'mask.nii')
        savedir = os.path.join(gl.baseDir, f'{gl.roiDir}', f'subj{args.sn}')
        searchlight_surf(white, pial, mask, savedir, maxradius=np.inf, maxvoxels=100)
    if args.what == 'make_searchlight_all':
        for sn in args.sns:
            print(f'Doing participant {sn}...')
            main(argparse.Namespace(
                    what='make_searchlight',
                    glm=args.glm,
                    sn=sn))


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--what', type=str, default='make_searchlight')
    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+',
                        default=[101, 102, 103, 104, 105, 106, 107,])
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()
    main(args)
    finish = time.time()

    print(f'Execution time:{finish - start} s')