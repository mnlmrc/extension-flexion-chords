import os

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, filtfilt, firwin


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

def calc_success(X):

    success = X.groupby(['subNum', 'chordID', 'day', 'chord'])['trialPoint'].mean().reset_index()
    success.rename(columns={'trialPoint': 'success'}, inplace=True)

    success.sort_values(by='chord', inplace=True)

    return success

def calc_avg(X, columns=None, by=None):
    """
        Computes the average value of variables in <columns> grouped by <by>.

        Parameters:
        data (pd.DataFrame): The input dataframe.

        Returns:
        pd.DataFrame: A dataframe with averaged values.
        """

    if isinstance(columns, str):
        columns = [columns]

    # Group by 'chordID' and compute the mean of 'MD'
    if X is str:
        data = pd.read_csv(X)
    elif isinstance(X, pd.DataFrame):
        data = X
    else:
        data = None

    columns = {col: 'mean' for col in columns}
    avg = data.groupby(by).agg(columns).reset_index()
    # md.rename(columns={'MD': 'average_MD'}, inplace=True)

    return avg


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


def savefig(path, fig):
    # Check if file exists
    if os.path.exists(path):
        response = input(f"The file {path} already exists. Do you want to overwrite it? (y/n): ")
        if response.lower() == 'y':
            fig.savefig(path, dpi=600)
            print(f"File {path} has been overwritten.")
        else:
            print("File not saved. Please choose a different name or path.")
    else:
        fig.savefig(path, dpi=600)
        print(f"File saved as {path}.")

def time_to_seconds(t):
    minutes, seconds = map(float, t.split(':'))
    return minutes * 60 + seconds


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
