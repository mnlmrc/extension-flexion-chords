import numpy as np
import PcmPy as pcm
import time
import argparse
import os
import globals as gl
import pickle


def normalize_Ac(Ac):
    for a in range(Ac.shape[0]):
        tr = np.trace(Ac[a] @ Ac[a].T)
        Ac[a] = Ac[a] / np.sqrt(tr)
    return Ac


def make_models():
    C = pcm.centering(24)

    day_eq = C @ np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]).T
    day_ln = C @ np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3])
    chord = C @ np.array([[1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]]).T

    Ac = np.zeros((5, 24, 6))
    Ac[0, :, :3] = day_eq
    Ac[1, :, 3:] = day_eq
    Ac[2, :, 3] = chord[:, 0]
    Ac[3, :, 4] = chord[:, 1]
    Ac[4, :, 5] = chord[:, 2]

    Ac = normalize_Ac(Ac)

    G_day_eq = day_eq @ day_eq.T
    G_day_ln = np.outer(day_ln, day_ln)
    G_chord = chord @ chord.T
    G_component = np.array([G_day_eq / np.trace(G_day_eq),
                            G_day_ln / np.trace(G_day_ln),
                            G_chord / np.trace(G_chord)
                            ])

    M = []
    M.append(pcm.FixedModel('null', np.eye(24)))
    M.append(pcm.FixedModel('day_eq', G_day_eq))
    M.append(pcm.FixedModel('day_ln', G_day_ln))
    M.append(pcm.FixedModel('chord', G_chord))
    M.append(pcm.ComponentModel('component', G_component))
    M.append(pcm.FeatureModel('feature', Ac))
    M.append(pcm.FreeModel('ceil', 24))

    return M



def main(args):
    if args.what == "corr":
        # nsteps = 20
        # M = []
        # for r in np.linspace(0, 1, nsteps):
        #     M.append(pcm.CorrelationModel(f"{r:0.2f}", num_items=4, corr=r, cond_effect=True))
        Mflex = pcm.CorrelationModel("flex", num_items=4, corr=None, cond_effect=True)
        # M.append(Mflex)
        f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.corr.p'), "wb")
        pickle.dump(Mflex, f)
    if args.what == "corr_chord":
        # nsteps = 50
        # M = []
        # for r in np.linspace(0, 1, nsteps):
        #     M.append(pcm.CorrelationModel(f"{r:0.2f}", num_items=1, corr=r, cond_effect=False))
        Mflex = pcm.CorrelationModel("flex", num_items=1, corr=None, cond_effect=False)
        # M.append(Mflex)
        f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.corr_chord.p'), "wb")
        pickle.dump(Mflex, f)
    if args.what == "across_sessions":
        M = make_models()
        f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.across_sessions.p'), "wb")
        pickle.dump(M, f)

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)

    args = parser.parse_args()

    make_models()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')