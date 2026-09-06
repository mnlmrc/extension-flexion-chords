import argparse
import inspect

import EFC_learningfMRI.betas as betas
import EFC_learningfMRI.globals as gl


def beta(sns=gl.participants, glm=3):
    """Write each participant's beta maps as cifti."""
    for sn in sns:
        betas.CiftiCortex(sn, glm).beta()


def contrast(sns=gl.participants, glm=3):
    """Write each participant's contrast maps as cifti."""
    for sn in sns:
        betas.CiftiCortex(sn, glm).contrast()


def residual(sns=gl.participants, glm=3):
    """Write each participant's residual maps as cifti."""
    for sn in sns:
        betas.CiftiCortex(sn, glm).residual()


# Step name -> function. Each step is independent, built straight from the glm betas.
FUNC = {
    'cifti_beta'    : beta,
    'cifti_contrast': contrast,
    'cifti_residual': residual,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`sns=`, `glm=`), but only the ones it accepts.
    """
    if what is not None:
        func     = FUNC[what]                                       # select function
        accepted = inspect.signature(func).parameters               # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted})  # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write cifti beta, contrast and residual maps.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas come from (default: the step default, 3)')
    parser.add_argument('--sns', nargs='+', type=int, default=[116], help='participant numbers (default: all participants)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        main('cifti_beta',     **kwargs)
        main('cifti_contrast', **kwargs)
        main('cifti_residual', **kwargs)
