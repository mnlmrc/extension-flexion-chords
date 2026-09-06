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


def average_contrasts_difference(glm=3, stat='con'):
    """Group trained - untrained difference from the map written by `average_contrasts`.

    No `sns`: it reads the group map, which is already pooled over participants. `main`
    filters the extra kwarg out.
    """
    surface.average_smoothed_contrasts_difference(glm=glm, stat=stat)


# Step name -> function, in the order the full run does them.
FUNC = {
    'activation_rois'    : roi_activation,
    'smooth_contrasts'   : smooth_contrasts,
    'contrast_group'     : average_contrasts,
    'contrast_group_diff': average_contrasts_difference,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`sns=`, `glm=`, `stat=`), but only the ones it
    accepts.
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
    parser.add_argument('--atlas_name', default=None, help='atlas, activation_rois only (default: the step default, ROI)')
    parser.add_argument('--stat', default=None, help='contrast statistic, the contrast_* steps only (default: the step default, con)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        main('activation_rois',     **kwargs)
        main('contrast_smooth',     **kwargs)
        main('contrast_group',      **kwargs)
        main('contrast_group_diff', **kwargs)
