import os

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt, firwin
from scipy.special import expit
import EFC_learningfMRI.globals as gl
from scipy.stats import linregress, t
#import EFC_learningfMRI.globals as gl


def make_chord_mapping(sn, type='trained-untrained'):

    trained_untrained = np.array(get_trained_and_untrained(sn)).astype(int)
    if type == 'trained-untrained':
        label = [1, 1, 1, 1, 2, 2, 2, 2,]
        chordID_mapping = dict(zip(trained_untrained, label))
    elif type == 'chord-session':
        chordID_mapping = dict(zip(trained_untrained, np.arange(8)))
    else:
        raise ValueError("Wrong type. Use 'trained-untrained' for trained vs. untrained and 'chord-session' for "
                         "individual chords in each session.")
    
    return chordID_mapping

def get_cond_part(sn, glm, type='trained-untrained'):

    path_glm = os.path.join(gl.baseDir, f'glm{glm}')

    reginfo = pd.read_csv(os.path.join(path_glm, f'subj{sn}', 'reginfo.tsv'), sep='\t')

    chordID_mapping = make_chord_mapping(sn, type)

    # make cond and part
    sess = reginfo.name.str.split(',', n=1, expand=True).loc[:, 1]
    sess = sess.map(gl.sess_mapping)
    chordID = reginfo.name.str.split(',', n=1, expand=True).loc[:, 0]
    chord = chordID.astype(int).map(chordID_mapping)
    part_vec = (reginfo.run % 10).to_numpy()
    cond_vec = (sess.astype(str) + ',' + chord.astype(str)).to_numpy()

    return part_vec, cond_vec


def get_trained_and_untrained(sn):
    """
    Retrieve which chords where trained and untrained in participant sn.
    Args:
        sn: participant number

    Returns:
        list of chordIDs. First four are trained, last four untrained
    """

    pinfo = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    trained = pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.')
    untrained = pinfo[pinfo.sn == sn].reset_index()['untrained'][0].split('.')
    chords = list()
    chords.extend(trained)
    chords.extend(untrained)

    return chords


def linear_fit(x, y, alternative_slope='two-sided', alternative_intercept='greater'):
    slope, intercept, r_value, p_slope, std_err = linregress(x, y, alternative=alternative_slope)

    R2 = r_value ** 2

    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = slope * x_fit + intercept

    # Compute confidence intervals
    n = len(x)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    dof = n - 2
    t_val = t.ppf(0.975, dof)

    se_line = np.sqrt(
        np.sum(residuals ** 2) / dof * (1 / n + (x_fit - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )
    ci = t_val * se_line

    # Check confidence interval at x = 0
    ix_0 = np.argmin(np.abs(x_fit - 0))
    lower_bound = y_fit[ix_0] - ci[ix_0]
    upper_bound = y_fit[ix_0] + ci[ix_0]

    MSE = np.sum(residuals ** 2) / dof
    SE_intercept = np.sqrt(MSE * (1 / n + np.mean(x) ** 2 / np.sum((x - np.mean(x)) ** 2)))
    t_intercept = intercept / SE_intercept
    if alternative_intercept == 'two-sided':
        p_intercept = 2 * (1 - t.cdf(t_intercept, df=dof))
    elif alternative_intercept == 'greater':
        p_intercept = 1 - t.cdf(t_intercept, df=dof)
    elif alternative_intercept == 'less':
        p_intercept = t.cdf(t_intercept, df=dof)

    # print(f'slope: {slope}, p = {p_slope:.3f}')
    # print(f'intercept: {intercept}, p_intercept = {p_intercept:.3f}')
    # print(f'R2 = {R2:.3f}')

    return x_fit, y_fit, ci, slope, p_slope, intercept, p_intercept, R2



def load_nat_emg(file_path):
    # Load the .mat file
    mat = scipy.io.loadmat(file_path)

    # Extract the 'dist' cell array from 'emg_natural_dist'
    emg_nat = mat['emg_natural_dist']
    emg_nat = emg_nat['dist'][0, 0]

    emg_nat_list = []
    for e in emg_nat:
        emg_nat_list.append(e[0])

    return emg_nat_list


def lowpass_butter(signal=None, cutoff=None, fsample=None, order=5, axis=-1):
    """
    Apply a low-pass filter to a 5-by-t signal array.

    Parameters:
    signal (np.ndarray): 5-by-t array where each row is a signal to be filtered.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling frequency of the signal.
    order (int): The order of the Butterworth filter (default is 5).

    Returns:
    np.ndarray: The filtered 5-by-t signal array.
    """
    # Design a Butterworth low-pass filter
    nyquist = .5 * fsample
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    filtered_signal = filtfilt(b, a, signal, axis=axis)

    return filtered_signal



def lowpass_fir(data, n_ord=None, cutoff=None, fsample=None, padlen=None, axis=-1):
    """
    Low-pass filter to remove high-frequency noise from the EMG signal.

    :param data: Input signal to be filtered.
    :param n_ord: Filter order.
    :param cutoff: Cutoff frequency of the low-pass filter.
    :param fsample: Sampling frequency of the input signal.
    :return: Filtered signal.
    """
    numtaps = int(n_ord * fsample / cutoff)
    b = firwin(numtaps + 1, cutoff, fs=fsample, pass_zero='lowpass')
    filtered_data = filtfilt(b, 1, data, axis=axis, padlen=padlen)

    return filtered_data


def calc_R2(Y, Yhat):
    ss_res = np.nansum((Y - Yhat) ** 2)
    ss_tot = np.nansum((Y - np.nanmean(Y)) ** 2)

    return 1 - ss_res / ss_tot



def load_glm_onset(sn, glm):
    pinfo = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    func_runs = pinfo.loc[pinfo.participant_id == f"subj{sn}", "FuncRuns_day3"].iloc[0].split('.')
    func_runs = np.array(func_runs, dtype=int)
    func_runs = np.array([func_runs + func_runs.size * i for i in range(3)]).flatten()
    events = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, 'day3', f'efc4_subj{sn}_glm{glm}_events.tsv'), sep='\t')
    events = events[(events.BN.isin(func_runs)) & (events.chordID != 99999)]
    BN = events.BN.to_numpy() - 1
    onset_b = events.Onset.to_numpy()
    onset = (np.round(onset_b * gl.TR) + BN * gl.nTR).astype(int)
    onset = np.sort(onset)
    return onset

