import rsatoolbox as rsa


def calc_rdm_unbalanced(data, channels, cond_vec, run_vec, method='crossnobis'):
    """

    Args:
        data:
        channels:
        cond_vec:
        run_vec:
        method:

    Returns:

    """

    dataset = rsa.data.Dataset(
        data,
        channel_descriptors={'channels': channels},
        obs_descriptors={'cond': cond_vec, 'run': run_vec},
    )

    noise = rsa.data.noise.prec_from_unbalanced(dataset,
                                                obs_desc='cond',
                                                method='diag')

    rdm = rsa.rdm.calc_rdm_unbalanced(dataset,
                                      method=method,
                                      descriptor='cond',
                                      noise=noise,
                                      cv_descriptor='run')
    rdm.reorder(rdm.pattern_descriptors['cond'].argsort())

    return rdm
