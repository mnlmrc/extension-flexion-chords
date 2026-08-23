import itertools
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
        session  = self.condition[1].map(gl.sess_mapping)
        chordID  = self.condition[0].astype(int).map(self.make_chord_mapping)

        if self.condition.shape[1] > 2:
            repetition = self.condition[2]
            cond_vec   = session.astype(str) + ',' + repetition.astype(str) + ',' + chordID.astype(str)
        else:
            cond_vec = session.astype(str) + ',' + chordID.astype(str)
        
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


def runs_to_keep(totRuns, session='all', repetition='all', rep_vec=None):
    """Boolean mask over ``totRuns`` regressor rows, selecting a session and/or repetition.

    The two filters are independent and combined with AND, so you can ask for one,
    the other, or both.

    Args:
        totRuns:    number of rows the mask covers, e.g. ``part_vec.size``.
        session:    3, 9 or 23 to keep only that session's rows, or ``'all'``.
                    Sessions are assumed to split the rows into three equal blocks.
        repetition: repetition label to keep, or ``'all'``. Needs ``rep_vec``.
        rep_vec:    repetition label per row, length ``totRuns``. Only required
                    when ``repetition != 'all'``.

    Returns:
        Boolean array of length ``totRuns``, True for rows passing both filters.
    """
    session_dict = {3: 0, 9: 1, 23: 2}

    if session == 'all':
        keep_sess = np.ones(totRuns, dtype=bool)
    else:
        if session not in session_dict:
            raise ValueError(f"session must be 'all' or one of {sorted(session_dict)}, got {session!r}")
        block                                        = session_dict[session]
        nRuns                                        = totRuns // 3
        keep_sess                                    = np.zeros(totRuns, dtype=bool)
        keep_sess[block * nRuns:(block + 1) * nRuns] = True

    if repetition == 'all':
        keep_rep = np.ones(totRuns, dtype=bool)
    else:
        if rep_vec is None:
            raise ValueError('rep_vec must be passed if a specific repetition is requested')
        rep_vec = np.asarray(rep_vec)
        if rep_vec.size != totRuns:
            raise ValueError(f'rep_vec has {rep_vec.size} entries but totRuns is {totRuns}')
        keep_rep = rep_vec == repetition
        if not keep_rep.any():
            raise ValueError(f'no rows match repetition {repetition!r}; '
                             f'rep_vec holds {np.unique(rep_vec).tolist()}')

    return keep_sess & keep_rep


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


def add_behav_pair_diff(df, measures=('ET', 'MD', 'force_abs', 'force_der', 'force_der_peak'), behav=None, pair_col='pair',
                        sn_col='sn', session_col='session', behav_sn_col='subNum', chordID_col='chordID'):

    """Behavioural difference between the two chords of each pair, as a column per measure.

    For every row the two chords in ``pair_col`` are looked up in the trial-level
    behavioural table for that participant and session, averaged over trials, and
    ``|measure(A) - measure(B)|`` is written to ``f'{measure}{suffix}'``. Lets a
    pair's crossnobis distance be regressed on how differently its two chords were
    actually executed.

    The difference is absolute because ``pair`` is unordered -- the two chord IDs
    are sorted when the dissimilarity table is built -- so a signed difference
    would flip meaning from row to row. Same convention as
    :func:`add_flexion_imbalance`.

    Parameters
    ----------
    df : pandas.DataFrame
        Dissimilarity table, one row per chord pair, e.g.
        ``dissimilarity.within_session.{atlas}.glm{glm}.tsv``.
    measures : sequence of str, optional
        Behavioural columns to difference.
    behav : pandas.DataFrame, optional
        Trial-level behaviour. Defaults to ``force.success.tsv``, which is the
        table to use here: the ``*.session.success.tsv`` files are already
        collapsed to trained/untrained, and their ``chordID`` is an average of
        chord IDs rather than a chord.
    pair_col, sn_col, session_col : str, optional
        Columns of ``df`` holding the pair, the participant and the session.
    behav_sn_col, chordID_col : str, optional
        Participant and chord-ID columns of ``behav`` (named differently there).
    sep : str, optional
        Separator between the two chord IDs in ``pair_col``.
    suffix : str, optional
        Appended to each measure name to form the new column name.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with one ``f'{measure}{suffix}'`` column per measure.
        Pairs whose chords are missing from ``behav``, or that are malformed
        (e.g. containing '99999'), come out as NaN.
    """

    measures = list(measures)
    absent   = [m for m in measures if m not in behav.columns]
    if absent:
        raise KeyError(f'{absent} not in the behavioural table; it has {sorted(behav.columns)}')

    # one value per (participant, session, chord), averaged over successful trials
    behav = behav.copy()
    behav[chordID_col] = pd.to_numeric(behav[chordID_col], errors='coerce').astype(float)
    lut = behav.groupby([behav_sn_col, session_col, chordID_col])[measures].mean()

    chords = df[pair_col].str.split('-', expand=True)
    sides  = [pd.to_numeric(chords[i], errors='coerce').astype(float) for i in (0, 1)]

    df = df.copy()
    # reindex rather than merge, so row order and duplicate pairs are preserved
    vals = [lut.reindex(pd.MultiIndex.from_arrays([df[sn_col], df[session_col], c])) for c in sides]
    for m in measures:
        df[m] = np.abs(vals[0][m].to_numpy() - vals[1][m].to_numpy())

    return df


def split_trained_untrained(M):
    """Mean of the trained (first 4) and untrained (last 4) condition blocks."""
    mask4 = np.tri(4, k=-1, dtype=bool)
    mask8 = np.tri(8, k=-1, dtype=bool)
    if M.ndim==2:
        trained   = M[:4, :4][mask4]
        untrained = M[4:, 4:][mask4]
        tot       = M[mask8].mean()
    elif M.ndim==3:
        trained   = M[:, :4, :4][:, mask4]
        untrained = M[:, 4:, 4:][:, mask4]
        tot       = M[:, mask8].mean(axis=1)
    return tot, trained, untrained



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



def ttest_im_sess(df, rois=None, x='session', y=None, hue=None, hue_order=None, roi_col='roi', subject_col='sn', paired=True, alternative='two-sided'):
    """Paired t-tests between ``hue`` levels, one per level of ``x`` (and per ROI).

    The statistical counterpart of the ``plot_*_sess`` functions in
    :mod:`EFC_learningfMRI.vis`: it takes the same long-format table and the same
    ``rois``/``x``/``y``/``hue`` arguments, and tests, within each session, the
    ``y`` values of one ``hue`` level against another. With more than two levels
    every pair is tested (columns ``A`` and ``B`` say which), so the
    trained/untrained case is simply the two-level special case::

        ttest = ttest_im_sess(dist_chord_sess, rois, y='crossnobis', hue='chord',
                              hue_order=['trained', 'untrained'])
        print(ttest[['T', 'dof', 'p_val', 'CI95', 'roi', 'session']])

    Set ``roi_col=None`` for a table with no ROI column -- the behavioural ones,
    say, where the test is over sessions only::

        ttest = ttest_im_sess(perf[perf.session.isin(gl.sessions)], y='ET', hue='chord',
                              hue_order=['trained', 'untrained'], roi_col=None, subject_col='subNum')
        print(ttest[['session', 'A', 'B', 'T', 'dof', 'p_val', 'CI95', 'cohen_d']])

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format table with one row per subject / ROI / ``x`` / ``hue`` cell.
        Only the levels of ``x`` present in ``df`` are tested, so restrict the
        sessions by filtering ``df``. Duplicate rows within a cell raise, rather
        than being averaged silently.
    rois : list, optional
        ROIs to test, in order (default: all in ``roi_col``).
    x : str, optional
        Column whose levels are tested separately (default ``'session'``).
    y : str
        Column holding the values to test.
    hue : str
        Column whose levels are compared against each other.
    hue_order : list, optional
        Levels of ``hue`` to compare, in order; the t-test is signed as the
        first level minus the second (default: order of appearance in ``df``).
    roi_col : str, optional
        Column holding the ROI names (default ``'roi'``); ``None`` to test the
        whole table at once, without splitting it by ROI.
    subject_col : str, optional
        Column identifying the subject, used to pair observations across
        ``hue`` levels (default ``'sn'``; the behavioural tables use ``'subNum'``).
    paired : bool, optional
        Whether to pair the two samples by ``subject_col`` (default True).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Passed to pingouin.

    Returns
    -------
    pandas.DataFrame
        One row per ROI / ``x`` level / pair of ``hue`` levels, with columns
        ``A``, ``B``, ``T``, ``dof``, ``p_val``, ``CI95``, ``cohen_d``, ``BF10``,
        ``power``, plus the ``x`` column and, unless ``roi_col`` is None, ``roi_col``.
    """
    import pingouin as pg   # heavy import; only this helper needs it

    if y is None or hue is None:
        raise ValueError("y (the values to test) and hue (the levels to compare) are both required")

    if hue_order is None:
        hue_order = list(df[hue].unique())

    if roi_col is None:
        rois = [None]           # one pass over the whole table
    elif rois is None:
        rois = df[roi_col].unique()

    x_levels = np.sort(df[x].unique())

    rows = []
    for roi in rois:

        df_roi = df if roi_col is None else df[df[roi_col]==roi]

        for level in x_levels:

            df_level   = df_roi[df_roi[x]==level]
            piv        = df_level.pivot(index=subject_col, columns=hue, values=y)
            hue_levels = [h for h in hue_order if h in piv.columns]

            for a, b in itertools.combinations(hue_levels, 2):
                pair = piv[[a, b]].dropna()
                res  = pg.ttest(pair[a], pair[b], paired=paired, alternative=alternative).iloc[0]
                row  = {x        : level,
                        'A'      : a,
                        'B'      : b,
                        'T'      : res['T'],
                        'dof'    : res['dof'],
                        'p_val'  : res['p_val'],
                        'CI95'   : res['CI95'],       # of the A - B difference; pingouin rounds it to 2 decimals
                        'cohen_d': res['cohen_d'],    # unsigned -- the sign of the effect is in T
                        'BF10'   : res['BF10'],
                        'power'  : res['power']}
                rows.append(row if roi_col is None else {roi_col: roi, **row})

    return pd.DataFrame(rows)


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

