import os

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt, firwin
from scipy.special import expit
import globals as gl


def get_trained_and_untrained(experiment, sn):
    """
    Retrieve which chords where trained and untrained in participant sn.
    Args:
        sn: participant number

    Returns:
        list of chordIDs. First four are trained, last four untrained
    """

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
    trained = pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.')
    untrained = pinfo[pinfo.sn == sn].reset_index()['untrained'][0].split('.')
    chords = list()
    chords.extend(trained)
    chords.extend(untrained)

    return chords


def load_matlab_hrf(path):
    mat_contents = scipy.io.loadmat(path)
    mat_struct = mat_contents['T'][0, 0]  # Assuming 1x1 struct
    T = {field: mat_struct[field] for field in mat_struct.dtype.names}

    T['day'] = T['day'].flatten()
    T['block'] = T['block'].flatten()
    T['ons'] = T['ons'].flatten()
    T['chordID'] = T['chordID'].flatten()
    T['SN'] = T['SN'].flatten()
    T['region'] = T['region'].flatten()
    T['name'] = T['name'].flatten()
    T['hem'] = T['hem'].flatten()

    return T

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


def pearsonr_vec(y, X):
    """
    Computes the Pearson correlation between a 1x13 vector and each row of an nx13 matrix.

    Parameters:
    - vec: numpy array of shape (13,), the 1x13 vector.
    - mat: numpy array of shape (n, 13), the nx13 matrix.

    Returns:
    - correlations: numpy array of shape (n,), containing Pearson correlations
                    between the vector and each row of the matrix.
    """
    # Center the vector and the matrix by subtracting the mean
    vec_mean = y - np.mean(y)
    mat_mean = X - np.mean(X, axis=1)[:, np.newaxis]

    # Compute the covariance (dot product of the centered data)
    covariance = np.dot(mat_mean, vec_mean)

    # Compute the standard deviations
    vec_std = np.linalg.norm(vec_mean)  # Standard deviation of the vector
    mat_std = np.linalg.norm(mat_mean, axis=1)  # Standard deviation of each row in the matrix

    # Calculate Pearson correlation as the covariance divided by the product of std devs
    stat = covariance / (vec_std * mat_std)

    return stat


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


def calc_distance_from_distr(pattern, distr, d_type='project_to_nSphere', lambda_val=None):
    """
    Calculates the distance of the pattern from every chord in nat_dist.

    Parameters:
        pattern: numpy array, K by 1 vector representing the EMG pattern.
        distr: numpy array, matrix where rows are observations (chords) and columns are EMG channels.
        d_type: string, type of distance to use ('Euclidean', 'project_to_nSphere', 'oval'). Default is 'Euclidean'.
        lambda_val: float, parameter for the 'oval' distance type. Default is None.

    Returns:
        d: numpy array, sorted vector of distances between the pattern and each observation in nat_dist.
    """
    # Ensure pattern is a column vector
    if not isinstance(pattern, np.ndarray) or len(pattern.shape) != 1:
        raise ValueError('pattern must be a 1D numpy array')

    pattern = pattern.reshape(-1, 1)  # Convert to column vector if necessary

    # Default values for distance type
    if d_type == 'Euclidean':
        lambda_val = 0
    elif d_type == 'oval':
        if lambda_val is None:
            print("Warning: When using oval distance option, lambda must be provided. Setting lambda to 1.")
            lambda_val = 1
    elif d_type == 'project_to_nSphere':
        lambda_val = 20000
    else:
        raise ValueError(f'Distance type {d_type} does not exist.')

    # Distance container
    d = np.zeros(distr.shape[0])

    # Covariance matrix of the chord pattern
    cov_pattern = np.dot(pattern, pattern.T)

    # Distance weights
    sigma = np.eye(cov_pattern.shape[0]) + lambda_val * cov_pattern

    # Looping through points and calculating distances
    for i in range(distr.shape[0]):
        # Sample in natural distribution
        x = distr[i, :].reshape(-1, 1)

        # Squared distance
        diff = x - pattern
        d[i] = np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff)

    # Take the square root of the distances and sort them
    d = np.sqrt(d.flatten())
    return np.sort(d)


def sigmoid(t, k, t0):
    return expit(k * (t - t0))


def calc_sigmoid_sse(params, t, F, N):
    # Extract sigmoid parameters from params
    k_values = params[:N]  # Slopes of the N sigmoids
    t0_values = params[N:2 * N]  # Onsets of the N sigmoids
    weights = params[2 * N:].reshape(N, 4)  # Weights (N x 4 matrix)

    # Generate the sigmoid matrix S (t x N)
    S = np.array([[sigmoid(t_i, k, t0) for k, t0 in zip(k_values, t0_values)] for t_i in t])

    # Reconstruct F using S and W (N x 4)
    F_hat = S @ weights

    # r2 = calc_r2(F, F_hat)

    return (F - F_hat).ravel()  # np.sum((F - F_hat) ** 2)

# def fit_sigmoids(F, t, N, init_params):
#     result = least_squares(
#         calc_sigmoid_sse, init_params, args=(t, F, N), method='lm',
#     )
#     return result

def fit_sigmoids(F, t, N):
    # init_k = np.ones(N)  # Initial slopes (1 for all)
    # init_t0 = np.zeros(N) + 200  # Evenly spaced initial onsets
    init_k = np.random.uniform(low=0.5, high=2.0, size=N)  # Random slopes between 0.5 and 2.0
    init_t0 = np.random.uniform(low=np.min(t), high=np.max(t), size=N)
    init_weights = np.random.rand(N, 4)  # Random weights

    init_params = np.hstack([init_k, init_t0, init_weights.flatten()])

    # Perform optimization
    result = least_squares(calc_sigmoid_sse, init_params, args=(t, F, N), method='lm')

    return result


def calc_planTime(experiment, p, session, day):
    from force import load_mov
    sn = int(''.join([c for c in p if c.isdigit()]))
    path = os.path.join(gl.baseDir, experiment, session, f"day{day}")

    len_planTime = []
    for block in range(7):

        filename = os.path.join(path, f'{experiment}_{sn}_{block + 1:02d}.mov')

        mov = load_mov(filename)

        for tr in range(len(mov)):

            len_planTime.append(np.sum(mov[tr][:, 1] == 3) / gl.fsample['force'])

    return np.array(len_planTime)
