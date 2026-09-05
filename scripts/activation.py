import argparse
import inspect
import EFC_learningfMRI.betas as betas
import EFC_learningfMRI.surface as surface
import EFC_learningfMRI.globals as gl


def roi_activation(sns=gl.participants, atlas_name='ROI', glm=3, cond_names=('chordID', 'session',)):
    """ROI-averaged activation per participant, condition and region."""
    betas.roi_avg(sns=sns, atlas_name=atlas_name, glm=glm, cond_names=cond_names)


def smooth_contrasts(sns=gl.participants, glm=3, stat='con'):
    """Smooth each participant's cifti contrast maps on the surface."""
    for sn in sns:
        surface.smooth_cifti_contrasts(sn=sn, glm=glm, stat=stat)


def average_contrasts(sns=gl.participants, glm=3, stat='con'):
    """Group-average the smoothed contrast maps written by `smooth_contrasts`."""
    surface.average_smoothed_contrasts(sns=sns, glm=glm, stat=stat)


def average_contrasts_difference(sns=None, glm=3, stat='con'):
    """Group trained - untrained difference from the map written by `average_contrasts`.

    `sns` is unused here.
    """
    surface.average_smoothed_contrasts_difference(glm=glm, stat=stat)


FUNC = {
    'avg_activation_rois'     : roi_activation,
    'smooth_contrast_in_subjs'   : smooth_contrasts,
    'smoothened_avg_contrast'    : average_contrasts,
    'smoothened_avg_diff_tr_untr': average_contrasts_difference,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`glm=`, `metrics=`, `repetitions=`, ...).
    """

    if what is not None:
        func = FUNC[what] # select function
        accepted = inspect.signature(func).parameters # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted}) # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROI-averaged and surface-smoothed activation maps.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas/contrasts come from (default: the step default, 3)')
    parser.add_argument('--sns', nargs='+', type=int, default=gl.participants, help='participant IDs to include')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)
