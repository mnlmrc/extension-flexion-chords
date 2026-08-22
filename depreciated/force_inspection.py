import EFC_learningfMRI.globals as gl
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import EFC_learningfMRI.vis as vis
import EFC_learningfMRI.force as force


if __name__=='__main__':

    sn = 106
    n_block = [1, 1]
    sessions  = [3, 23]
    n_trial = [22, 41]  # chord 92122
    ch_idx = np.array(gl.diffCols)

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(6, 3), constrained_layout=True,)

    vis.plot_example_trials(fig, axs, sessions, sn=101, n_block=n_block, n_trial=n_trial, plot_derivative=True)

    plt.show()
