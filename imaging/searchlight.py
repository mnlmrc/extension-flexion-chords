import argparse
from imaging_pipelines.searchlight import searchlight_surf
import globals.path as pth
import time
import os
import numpy as np


def make_searchlight(sn, glm=1):
    path_surf = os.path.join(pth.baseDir, pth.surfDir, f'subj{sn}')
    white = [os.path.join(path_surf, f'subj{sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
    pial = [os.path.join(path_surf, f'subj{sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
    mask = os.path.join(pth.baseDir, f'glm{glm}', f'subj{sn}', 'mask.nii')
    savedir = os.path.join(pth.baseDir, f'{pth.roiDir}', f'subj{sn}')
    searchlight_surf(white, pial, mask, savedir, maxradius=10, maxvoxels=100)


def main(args):
    if args.what == 'make_searchlight':
        path_surf = os.path.join(pth.baseDir, pth.surfDir, f'subj{args.sn}')
        white = [os.path.join(path_surf, f'subj{args.sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
        pial = [os.path.join(path_surf, f'subj{args.sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
        mask = os.path.join(pth.baseDir, f'{pth.glmDir}{args.glm}', f'subj{args.sn}', 'mask.nii')
        savedir = os.path.join(pth.baseDir, f'{pth.roiDir}', f'subj{args.sn}')
        searchlight_surf(white, pial, mask, savedir, maxradius=10, maxvoxels=100)
    if args.what == 'make_searchlight_all':
        for sn in args.sns:
            print(f'Doing participant {sn}...')
            main(argparse.Namespace(
                    what='make_searchlight',
                    glm=args.glm,
                    sn=sn))
