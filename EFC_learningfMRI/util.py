import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt, firwin
from scipy.special import expit
import EFC_learningfMRI.globals as gl
from scipy.stats import linregress, t
#import EFC_learningfMRI.globals as gl


@dataclass
class RegInfo:
    """The reginfo.tsv of one subject/glm, parsed into partition and condition vectors.

    The file is read and parsed on first access, then cached, so a pipeline that
    needs ``cond_vec`` and ``part_vec`` more than once does not re-read the TSV
    each time (which is what repeated ``get_cond_part`` calls do)::

        reg = RegInfo(sn, glm)
        reg.cond_vec, reg.part_vec        # parsed once here, reused after
        reg.reginfo                       # the raw DataFrame, if you need it
    """

    sn: int
    glm: int

    @cached_property
    def reginfo(self):
        """The raw reginfo.tsv as a DataFrame."""
        path = os.path.join(gl.baseDir, f'glm{self.glm}', f'subj{self.sn}', 'reginfo.tsv')
        return pd.read_csv(path, sep='\t')

    @cached_property
    def part_vec(self):
        """Partition (run) vector, one entry per regressor."""
        return self.reginfo.run.to_numpy()

    @cached_property
    def condition(self):
        return parse_regressor_name(self.reginfo.name)

    @cached_property
    def condition_unique(self):
        return self.condition.drop_duplicates(ignore_index=True)

    @cached_property
    def make_chord_mapping(self):
        trained_untrained = get_trained_and_untrained(self.sn)
        trained_untrained = np.asarray(trained_untrained, dtype=int)
        return dict(zip(trained_untrained, np.arange(8)))

    @cached_property
    def cond_vec(self):
        """
        transform condition labels into number for correct ordering in G matrix
        """
        sess    = self.condition[1].map(gl.sess_mapping)
        chordID = self.condition[0].astype(int).map(self.make_chord_mapping)

        cond_vec = sess.astype(str) + ',' + chordID.astype(str)

        if self.condition.shape[1] > 2:
            return (cond_vec + ',' + self.condition[2].astype(str)).to_numpy()
        else:
            return cond_vec.to_numpy()


def parse_regressor_name(name, sep=','):
    """Split a contrast regressor name into its component condition labels.

    Regressor names are ``sep``-delimited condition labels, e.g.
    ``'chordID,sess'`` or ``'chordID,sess,rep'`` with the default ``sep=','``.

    Parameters
    ----------
    name : pandas.Series of str
        The regressor names to parse.
    sep : str, optional
        Field separator (default ``','``).

    Returns
    -------
    tuple of pandas.Series
        One ``Series`` per delimited component, in order. The tuple has as many
        elements as the number of parts found, so it can be unpacked directly,
        e.g. ``chordID, sess = _parse_regressor_name(regressor)``.
    """
    parts = name.str.split(sep, expand=True)
    return pd.DataFrame(tuple(parts[col] for col in parts.columns)).T


def get_trained_and_untrained(sn):
    """
    Retrieve which chords where trained and untrained in participant sn.
    Args:
        sn: participant number

    Returns:
        list of chordIDs. First four are trained, last four untrained
    """

    pinfo     = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    trained   = pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.')
    untrained = pinfo[pinfo.sn == sn].reset_index()['untrained'][0].split('.')
    chords    = list()
    chords.extend(trained)
    chords.extend(untrained)

    return chords


def runs_to_keep(session, totRuns):

        session_dict = {3: 0, 9: 1, 23: 2}

        if session=='all':
            keep = np.ones(totRuns, dtype=bool)
        else:
            session                                   = session_dict[session]
            nRuns                                     = totRuns // 3
            keep                                      = np.zeros(totRuns, dtype=bool)
            keep[session * nRuns:(session + 1)*nRuns] = True
        return keep


def add_chord_column(df, chordID_col='chordID', sn_col='sn', out_col='chord'):
    """Add a ``chord`` column labelling each row 'trained' or 'untrained'.

    For every participant in ``sn_col`` the trained/untrained chord IDs are
    looked up with :func:`get_trained_and_untrained` (first four trained, last
    four untrained), and each row's ``chordID_col`` value is labelled
    accordingly.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing a chord-ID column and a participant column.
    chordID_col : str, optional
        Name of the chord-ID column (default ``'chordID'``).
    sn_col : str, optional
        Name of the participant column (default ``'sn'``); needed because the
        trained/untrained split differs per participant.
    out_col : str, optional
        Name of the column to add (default ``'chord'``).

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the ``out_col`` column added.
    """
    df = df.copy()
    df[out_col] = 'untrained'
    for sn in df[sn_col].unique():
        trained = np.array(get_trained_and_untrained(sn)[:4]).astype(int)
        mask = (df[sn_col] == sn) & (df[chordID_col].astype(int).isin(trained))
        df.loc[mask, out_col] = 'trained'

    return df


def add_flexion_imbalance(pair, sep="-"):
    """
    Flexion difference for a chord pair, e.g. "91211-22911".
    Each chord is 5 fingers coded 1=extension, 2=flexion, 9=neutral.
    Returns |n_flexed(A) - n_flexed(B)|; NaN if the entry is malformed.

    Accepts a single string or a whole column: pass a pandas Series (e.g.
    ``df['pair']``) and a Series of the same length is returned, so it can be
    assigned straight back with ``df['flexion_imbalance'] = add_flexion_imbalance(df['pair'])``.
    """
    if isinstance(pair, pd.Series):
        return pair.map(lambda p: add_flexion_imbalance(p, sep))
    if not isinstance(pair, str):
        return float("nan")
    parts = pair.split(sep)
    if len(parts) != 2 or any("99999" in p for p in parts):
        return float("nan")
    a, b = parts
    return abs(a.count("2") - b.count("2"))   # 2 == flexion


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



def load_glm_onset(sn, glm, output_events=False):
    pinfo     = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    func_runs = pinfo.loc[pinfo.participant_id == f"subj{sn}", "FuncRuns_day3"].iloc[0].split('.')
    func_runs = np.array(func_runs, dtype=int)
    func_runs = np.array([func_runs + func_runs.size * i for i in range(3)]).flatten()
    events    = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, 'day3', f'efc4_subj{sn}_glm{glm}_events.tsv'), sep='\t')
    events    = events[(events.BN.isin(func_runs)) & (events.chordID != 99999)]
    BN        = events.BN.to_numpy() - 1
    onset_b   = events.Onset.to_numpy()
    onset     = (np.round(onset_b * gl.TR) + BN * gl.nTR).astype(int)
    onset     = np.sort(onset)
    if output_events:
        return onset, events
    else:
        return onset

