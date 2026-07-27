"""Write simulated single-ROI betas as CIFTI (+ dummy mask / reginfo) per subject.

Two phases, because the trained/untrained split now comes from participants.tsv (as for real
subjects):

  1. ``Simulation(...).run()`` writes split-agnostic H0 data to ``{gl.baseDir}/sim{sim}/subj{sn}/``
     (beta.dscalar.nii + mask.nii + reginfo.tsv). Needs no participants.tsv.
  2. Add rows for the sim subjects to participants.tsv (trained/untrained chords). THEN load,
     derive H1s, make contrasts, and run the usual analyses -- ``load`` reads the split from
     participants.tsv via ``get_trained_and_untrained``.

    import PcmPy as pcm
    from EFC_learningfMRI.simulation import load
    from EFC_learningfMRI.util import runs_to_keep
    from EFC_learningfMRI.geometry import split_trained

    betas, cond_vec, part_vec, chords = load(1, 9001)   # split from participants.tsv
    keep = runs_to_keep(3, part_vec.size)
    G, _ = pcm.est_G_crossval(betas[keep], cond_vec[keep], part_vec[keep],
                              X=pcm.indicator(part_vec[keep]))
    _, trained, untrained = split_trained(pcm.G_to_dist(G))
"""

from EFC_learningfMRI.simulation import Simulation, h1_trained_hypoactive, make_sim_contrasts
from EFC_learningfMRI.betas import roi_avg

if __name__ == '__main__':
    sns = list(range(9001, 9016))

    # phase 1 -- generate H0 (no participants.tsv needed)
    Simulation(
        sim      = 1,
        sns      = sns,
        n_run    = 10,
        n_vox    = 500,
        signal   = 1.0,
        noise    = 1.0,
        subj_var = 0.0,
        seed     = 0,
    ).run()

    make_sim_contrasts(sim=1, sns=sns)

    #roi_avg(sns=sns, glm=101)         
    # h1_trained_hypoactive(1, 2, amount=1.0)  # trained hypoactivity: read sim1 -> write sim2
