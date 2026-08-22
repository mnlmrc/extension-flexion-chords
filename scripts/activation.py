import argparse

from EFC_learningfMRI.betas import roi_avg
from EFC_learningfMRI.surface import (smooth_cifti_contrasts, average_smoothed_contrasts,
                                      average_smoothed_contrasts_difference)
import EFC_learningfMRI.globals as gl


def roi_activation(sns=gl.participants, atlas_name='ROI', glm=3, cond_names=('chordID', 'session',)):
    """ROI-averaged activation per participant, condition and region."""
    roi_avg(sns=sns, atlas_name=atlas_name, glm=glm, cond_names=cond_names)


def smooth_contrasts(sns=gl.participants, glm=3, stat='con'):
    """Smooth each participant's cifti contrast maps on the surface."""
    for sn in sns:
        smooth_cifti_contrasts(sn=sn, glm=glm, stat=stat)


def average_contrasts(sns=gl.participants, glm=3, stat='con'):
    """Group-average the smoothed contrast maps written by `smooth_contrasts`."""
    average_smoothed_contrasts(sns=sns, glm=glm, stat=stat)


def average_contrasts_difference(glm=3, stat='con'):
    """Group trained - untrained difference from the map written by `average_contrasts`."""
    average_smoothed_contrasts_difference(glm=glm, stat=stat)


FUNC = {
    'roi'                      : roi_activation,
    'subj.smooth.surface'      : smooth_contrasts,
    'average.smooth.surface'   : average_contrasts,
    'difference.smooth.surface': average_contrasts_difference,
}


def main(what=None, **kwargs):
    """Run one step. `kwargs` (`glm`, ...) are forwarded to the step."""
    if what is not None:
        FUNC[what](**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROI-averaged and surface-smoothed activation maps.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas/contrasts come from (default: the step default, 3)')
    parser.add_argument('--cond_names', nargs='+', default=('chordID', 'session'), dest='cond_names',
                        help='condition columns the ROI table is indexed by (default: the step default, chordID session)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        main('roi')
        main('subj.smooth.surface')
        main('average.smooth.surface')
        main('difference.smooth.surface')
