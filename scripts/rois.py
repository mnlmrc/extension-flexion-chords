import argparse
import os

import EFC_learningfMRI.globals as gl
import imaging_pipelines.rois as rois


def _make_cortical_rois(sn, glm=3, atlas_name='ROI'):
    """Build one participant's cortical ROIs from the surfaces and the glm mask."""
    exclude = {
        'ROI': [(1, 2), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5), (2, 7), (3, 4), (3, 5)],
        'BA_handArea': [],
        'ROI_grouped': []
    }

    path_surf = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
    white     = [os.path.join(path_surf, f'subj{sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
    pial      = [os.path.join(path_surf, f'subj{sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
    mask      = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'mask.nii')
    atlas_dir = gl.atlasDir
    rois_dir  = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
    Rois      = rois.SurfRois(white, pial, mask, rois_dir, atlas_name=atlas_name, atlas_dir=atlas_dir)

    Rois.make_rois(exclude=exclude[atlas_name])


def _make_hemispheres(sn, glm=3):
    """Build one participant's hemisphere masks from the surfaces and the glm mask."""
    path_surf = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
    white     = [os.path.join(path_surf, f'subj{sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
    pial      = [os.path.join(path_surf, f'subj{sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
    mask      = os.path.join(gl.baseDir, f'glm{glm}', f'subj{sn}', 'mask.nii')
    rois_dir  = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
    Rois      = rois.SurfRois(white, pial, mask, rois_dir)

    Rois.make_hemispheres()


def make_rois(sns=gl.participants, glm=3, atlas_name='ROI'):
    """Build the cortical ROIs for each participant."""
    for sn in sns:
        print(f'rois, doing participant {sn}...')
        _make_cortical_rois(sn, glm=glm, atlas_name=atlas_name)


def make_hemispheres(sns=gl.participants, glm=3):
    """Build the hemisphere masks for each participant."""
    for sn in sns:
        print(f'hemispheres, doing participant {sn}...')
        _make_hemispheres(sn, glm=glm)


# Step name -> function. Both steps are independent, built from the surfaces and glm mask.
FUNC = {
    'make_rois'       : make_rois,
    'make_hemispheres': make_hemispheres,
}


def main(what=None, **kwargs):
    """Run one step. `kwargs` (`sns`, `glm`, `atlas_name`) are forwarded to the step."""
    if what is not None:
        FUNC[what](**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build cortical ROIs and hemisphere masks from the surfaces.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--glm', type=int, default=None, help='GLM whose mask.nii bounds the ROIs (default: the step default, 3)')
    parser.add_argument('--sn', nargs='+', type=int, default=None, dest='sns',
                        help='participant numbers (default: all participants)')
    parser.add_argument('--atlas_name', default=None, help='atlas, make_rois only (default: the step default, ROI)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        main('make_rois')
        main('make_hemispheres')
