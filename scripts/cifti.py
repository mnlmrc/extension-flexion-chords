import argparse

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
    'beta'    : beta,
    'contrast': contrast,
    'residual': residual,
}


def main(what=None, **kwargs):
    """Run one step. `kwargs` (`sns`, `glm`) are forwarded to the step."""
    if what is not None:
        FUNC[what](**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write cifti beta, contrast and residual maps.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas come from (default: the step default, 3)')
    parser.add_argument('--sn', nargs='+', type=int, default=None, dest='sns',
                        help='participant numbers (default: all participants)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        main('beta', **kwargs)
        main('contrast', **kwargs)
        main('residual', **kwargs)
