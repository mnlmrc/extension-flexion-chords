import numpy as np
import PcmPy as pcm
import time
import argparse
import os
import EFC_learningfMRI.globals as gl
import pickle


def normalize_Ac(Ac):
    for a in range(Ac.shape[0]):
        tr = np.trace(Ac[a] @ Ac[a].T)
        Ac[a] = Ac[a] / np.sqrt(tr)
    return Ac


def make_models_sess():
    C = pcm.centering(8)

    trained_untrained_bt = C @ np.array([1, 1, 1, 1, -1, -1, -1, -1])

    G_trained_untrained_bt = np.outer(trained_untrained_bt, trained_untrained_bt)
    G_I = np.eye(8)
    G_component = np.array([G_trained_untrained_bt / np.trace(G_trained_untrained_bt),
                            G_trained_untrained_wt / np.trace(G_trained_untrained_wt),
                            G_I / np.trace(G_I)])

    M = []
    M.append(pcm.ComponentModel('component', G_component))

    return M


def make_models_chord():
    C = pcm.centering(8)
    
    pass



def main(args):
    if args.what == "correlation_across_sessions":
        Mflex = pcm.CorrelationModel("flex", num_items=4, corr=None, cond_effect=False)
        f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.corr.p'), "wb")
        pickle.dump(Mflex, f)
    if args.what == "trained_untrained_sess":
        M = make_models_sess()
        f = open(os.path.join(gl.baseDir, gl.pcmDir, f'M.trained_untrained.p'), "wb")
        pickle.dump(M, f)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')