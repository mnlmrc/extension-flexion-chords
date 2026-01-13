import numpy as np
import os
import pandas as pd
import xarray as xr
import globals as gl
import PcmPy as pcm
import argparse
from util import get_trained_and_untrained
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal import resample
from sigproc.filter_hilbert import FilterHilbert
from sigproc.util import resample_to_fs, hp_filter
from sigproc.coherence import plv_between_channels, plv_between_channels_shuffle, calc_plv_z, demodulate, calc_plv_aligned, calc_mscohere

exclude_blocks = {
    'day1': {
        'subj100': [],
        'subj101': [1],
        'subj102': [],
        'subj103': [],
        'subj104': [],
        'subj105': [],
        'subj106': [],
    },
    'day5': {
        'subj100': [4, 6],
        'subj101': [],
        'subj102': [],
        'subj103': [],
        'subj104': [],
        'subj105': [],
        'subj106': [],
    }
}


def detect_trig(trig_sig, time_trig, amp_threshold=None, edge='rising', min_duration=.05):
    """
    Detect trigger onsets with optional minimum duration filtering.

    :param trig_sig: signal array
    :param time_trig: time vector (same length as trig_sig)
    :param amp_threshold: threshold to binarize signal
    :param debugging: if True, plots signal and detected triggers
    :param edge: 'rising' or 'falling'
    :param min_duration: minimum duration (in seconds) that the signal must remain above/below threshold
    :return: rise_times, rise_idx
    """

    if edge == 'rising':
        trig_bin = (trig_sig > amp_threshold).astype(int)
    elif edge == 'falling':
        trig_bin = (trig_sig < amp_threshold).astype(int)
    else:
        raise ValueError('edge must be either "rising" or "falling"')

    diff_trig = np.diff(trig_bin)
    rise_idx = np.where(diff_trig == 1)[0]
    fall_idx = np.where(diff_trig == -1)[0]

    # Make sure each rising edge has a corresponding falling edge
    if fall_idx.size == 0 or rise_idx.size == 0:
        return np.array([]), np.array([])

    # Ensure proper ordering
    if fall_idx[0] < rise_idx[0]:
        fall_idx = fall_idx[1:]
    if rise_idx[-1] > fall_idx[-1]:
        rise_idx = rise_idx[:-1]

    valid_rise_idx = []
    for r_idx, f_idx in zip(rise_idx, fall_idx):
        duration = time_trig[f_idx] - time_trig[r_idx]
        if duration >= min_duration:
            valid_rise_idx.append(r_idx)

    rise_idx = np.array(valid_rise_idx)
    rise_times = np.array([float(time_trig[idx]) for idx in rise_idx])

    # Remove triggers that are <4s apart
    if len(rise_idx) > 0:
        filtered_idx = [rise_idx[0]]
        last_time = rise_times[0]
        for i in range(1, len(rise_idx)):
            if rise_times[i] - last_time >= 4:
                filtered_idx.append(rise_idx[i])
                last_time = time_trig[rise_idx[i]]
        rise_idx = np.array(filtered_idx)
        rise_times = np.array([float(time_trig[idx]) for idx in rise_idx])

    return rise_times, rise_idx


import numpy as np

def segment(data, start_idx, prestim, poststim, fsample):
    """
    Segment a (freq, chan, samples) array into fixed-length trials with no padding.

    Parameters
    ----------
    data : np.ndarray, shape (F, C, T)
    start_idx : array-like of int
        Trial start indices in samples (0-based).
    prestim : float
        Time in seconds before (if negative) or after (if positive) start_idx to begin the segment.
    poststim : float
        Time in seconds after start_idx to end the segment.
    fsample : float
        Sampling frequency (Hz).

    Returns
    -------
    segments : np.ndarray
        Shape (N, F, C, L), where L = (abs(prestim) + abs(poststim)) * fsample.
    kept_idx : np.ndarray
        Indices (into start_idx) of trials kept (i.e., fully in-bounds).
    """
    data = np.asarray(data)
    if data.ndim == 2:
        C, S = data.shape
    elif data.ndim == 3:
        F, C, S = data.shape
    T = len(start_idx)
    L = int((np.abs(prestim) + np.abs(poststim)) * fsample)

    segs = []
    kept = []
    for tr, idx in enumerate(start_idx):
        if prestim < 0:
            start = idx + int(prestim * fsample)
            end   = idx + int(poststim * fsample)
        else:  # prestim >= 0
            start = idx + int(prestim * fsample)
            end   = idx + int((prestim + poststim) * fsample)
        segs.append(data[..., start:end])

    segs = np.stack(segs, axis=0) if segs else np.zeros((0, data.shape), dtype=data.dtype)

    return segs



def load_delsys(filepath, trigger_name=None, muscle_names=None):
    """returns a pandas DataFrame with the raw EMG data recorded using the Delsys system

    :param participant_id:
    :param experiment:
    :param block:
    :param muscle_names:
    :param trigger_name:
    :return:
    """
    # fname = f"{experiment}_{participant_id}_{block}.csv"
    # filepath = os.path.join(gl.make_dirs(experiment, "emg", participant_id), fname)

    # read data from .csv file (Delsys output)
    with open(filepath, 'rt') as fid:
        A = []
        for line in fid:
            # Strip whitespace and newline characters, then split
            split_line = [elem.strip() for elem in line.strip().split(',')]
            A.append(split_line)

    # identify columns with data from each muscle
    muscle_columns = {}
    for muscle in muscle_names:
        for c, col in enumerate(A[3]):
            if muscle in col:
                muscle_columns[muscle] = c + 1  # EMG is on the right of Timeseries data (that's why + 1)
                break
        for c, col in enumerate(A[5]):
            if muscle in col:
                muscle_columns[muscle] = c + 1
                break

    df_raw = pd.DataFrame(A[8:])  # get rid of header
    df_out = pd.DataFrame()  # init final dataframe

    for muscle in muscle_columns:
        df_out[muscle] = pd.to_numeric(df_raw[muscle_columns[muscle]],
                                       errors='coerce').replace('', np.nan).dropna()  # add EMG to dataframe

    # add trigger column
    trigger_column = None
    for c, col in enumerate(A[3]):
        if trigger_name in col:
            trigger_column = c + 1

    try:
        trigger = df_raw[trigger_column]
        trigger = resample(trigger.values, len(df_out))
    except IOError as e:
        raise IOError("Trigger not found") from e

    df_out[trigger_name] = trigger

    # add time column
    df_out['time'] = df_raw.loc[:, 0]

    return df_out


def main(args):
    channels_emg = ['flx_D1', 'flx_D2', 'flx_D3', 'flx_D4', 'flx_D5', 'ext_D1', 'ext_D2', 'ext_D3', 'ext_D4', 'ext_D5']
    single_finger = [19999, 29999, 91999, 92999, 99199, 99299, 99919, 99929, 99991, 99992]
    pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')
    if args.sn is not None:
        trained = pinfo[pinfo['sn'] == args.sn]['trained'].reset_index(drop=True)[0].split('.')
    # fs_old, fs_new = 2148, 600
    fsample = 2148
    if args.what=='raw':
        for day in [1, 5]:
            savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
            os.makedirs(savedir, exist_ok=True)
            for bl in range(10):
                excl = exclude_blocks[f'day{day}'][f'subj{args.sn}']
                if bl+1 not in excl:
                    print(f'subj{args.sn} - block {bl+1}')

                    # load delsys files
                    filepath = os.path.join(gl.baseDir, args.experiment, 'emg', f'day{day}', f'subj{args.sn}',
                                            f'{args.experiment}_{args.sn}_{bl+1}.csv')
                    df_out = load_delsys(filepath, trigger_name='Trigger', muscle_names=channels_emg)

                    # calc emg envelope/rect
                    emg_raw = df_out[channels_emg].to_numpy().T
                    # emg_resampled = resample_to_fs(emg_raw, fs_old=fs_old, fs_new=fs_new)
                    np.save(os.path.join(savedir, f'emg_raw_{bl+1}.npy'), emg_raw)
    if args.what=='rectify':
        for day in [1, 5]:
            savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
            os.makedirs(savedir, exist_ok=True)
            for bl in range(10):
                excl = exclude_blocks[f'day{day}'][f'subj{args.sn}']
                if bl+1 not in excl:
                    print(f'subj{args.sn} - block {bl+1}')

                    # load delsys files
                    filepath = os.path.join(gl.baseDir, args.experiment, 'emg', f'day{day}', f'subj{args.sn}',
                                            f'{args.experiment}_{args.sn}_{bl+1}.csv')
                    df_out = load_delsys(filepath, trigger_name='Trigger', muscle_names=channels_emg)

                    # calc emg envelope/rect
                    emg_raw = df_out[channels_emg].to_numpy().T
                    emg_hp = hp_filter(emg_raw, n_ord=4, cutoff=20, fsample=fsample, axis=-1)
                    emg_rect = np.abs(emg_hp - emg_hp.mean(axis=-1, keepdims=True))
                    np.save(os.path.join(savedir, f'emg_rect_{bl+1}.npy'), emg_rect)
    if args.what=='rect_avg':
        for day in [1, 5]:
            savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
            emg_rect = np.load(os.path.join(savedir, f'emg_rect_segmented.npy'))
            emg_rect_avg = emg_rect.mean(axis=-1)
            np.save(os.path.join(savedir, f'emg_rect_avg.npy'), emg_rect_avg)
    if args.what=='timestamps':
        for sn in args.sns:
            for day in [1, 5]:
                savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{sn}')
                for bl in range(10):
                    excl = exclude_blocks[f'day{day}'][f'subj{sn}']
                    if bl + 1 not in excl:
                        filepath = os.path.join(gl.baseDir, args.experiment, 'emg', f'day{day}', f'subj{sn}',
                                                f'{args.experiment}_{sn}_{bl + 1}.csv')
                        df_out = load_delsys(filepath, trigger_name='Trigger', muscle_names=channels_emg)
                        trig_sig = np.abs(df_out.Trigger.to_numpy())
                        trig_time = df_out.time.astype(float).to_numpy()
                        _, timestamp = detect_trig(trig_sig, trig_time, amp_threshold=args.thresh,
                                                   edge=args.edge)
                        print(f'Found {len(timestamp)} trials in block {bl + 1}...')
                        np.save(os.path.join(savedir, f'timestamp_{bl + 1}.npy'), timestamp)
    if args.what=='segment_raw':
        for day in [1, 5]:
            savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
            emg_rect_seg = []
            for bl in range(10):
                excl = exclude_blocks[f'day{day}'][f'subj{args.sn}']
                if bl + 1 not in excl:
                    print(f'subj{args.sn} - block {bl+1}')
                    timestamp = np.load(os.path.join(savedir, f'timestamp_{bl + 1}.npy'))
                    emg_rect = np.load(os.path.join(savedir, f'emg_raw_{bl + 1}.npy'))
                    emg_rect_seg.append(segment(emg_rect, timestamp, 1, 3.5, fsample))
            emg_rect_seg = np.vstack(emg_rect_seg)
            np.save(os.path.join(savedir, f'emg_raw_segmented.npy'), emg_rect_seg)
    if args.what=='segment_rect':
        for day in [1, 5]:
            savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
            emg_rect_seg = []
            for bl in range(10):
                excl = exclude_blocks[f'day{day}'][f'subj{args.sn}']
                if bl + 1 not in excl:
                    print(f'subj{args.sn} - block {bl+1}')
                    timestamp = np.load(os.path.join(savedir, f'timestamp_{bl + 1}.npy'))
                    emg_rect = np.load(os.path.join(savedir, f'emg_rect_{bl + 1}.npy'))
                    emg_rect_seg.append(segment(emg_rect, timestamp, 1, 3.5, fsample))
            emg_rect_seg = np.vstack(emg_rect_seg)
            np.save(os.path.join(savedir, f'emg_rect_segmented.npy'), emg_rect_seg)
    if args.what=='mscohere':
        for day in [1, 5]:
            savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
            emg_rect = np.load(os.path.join(savedir, f'emg_raw_segmented.npy'))
            f, coh = calc_mscohere(emg_rect, fsample=fsample)
            np.savez(os.path.join(savedir, f'mscohere.npz'), coh=coh, f=f)

    if args.what=='mscohere_pipeline':
        for sn in args.sns:
            print(f'doing participant {sn}')
            main(argparse.Namespace(
                what='raw',
                experiment=args.experiment,
                sn=sn,
                thresh=args.thresh,
                edge=args.edge, ))
            main(argparse.Namespace(
                what='segment_raw',
                experiment=args.experiment,
                sn=sn,))
            main(argparse.Namespace(
                what='mscohere',
                experiment=args.experiment,
                sn=sn,))
    if args.what=='mscohere_pattern':
        D, F, S = 2, 513, len(args.sns)
        Dist = np.zeros((D, S, F, 8, 8))
        Cosine = np.zeros_like(Dist)
        for d, day in enumerate([1, 5]):
            for s, sn in enumerate(args.sns):
                print(f'doing participant {sn}, day {day}...')
                chordID = np.array(get_trained_and_untrained('EFC_learningEMG', sn)).astype(int)
                data = np.load(os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{sn}',
                                            'mscohere.npz'))
                dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'behavioural', f'day{day}',
                                 f'{args.experiment}_{sn}.dat'), sep='\t')
                dat = dat[~dat['chordID'].isin(single_finger)].reset_index(drop=True)
                chordID_mapping = {ch: i for i, ch in enumerate(chordID)}
                dat.chordID = dat.chordID.map(chordID_mapping)
                coh = data['coh']
                z_coh = np.arctanh(np.sqrt(np.clip(coh, 1e-12, 1 - 1e-12)))
                mask = np.tri(10, k=-1, dtype=bool)
                z_coh_grouped, cond_vec, part_vec = pcm.group_by_condition(z_coh, dat.chordID, dat.BN, axis=0)
                obs_des = {'cond_vec': cond_vec, 'part_vec': part_vec}
                z_coh_grouped = z_coh_grouped[:, mask, :]
                T, C, _ = z_coh_grouped.shape
                for f in range(F):
                    data = z_coh_grouped[:, :, f]
                    err = data - data.mean(axis=0, keepdims=True)
                    cov = (err.T @ err) / data.shape[0]
                    data_prewhitened = data / np.sqrt(np.diag(cov))
                    Y = pcm.Dataset(data, obs_des)
                    G, _ = pcm.est_G_crossval(Y.measurements, Y.descriptors['cond_vec'], Y.descriptors['part_vec'])
                    Dist[d, s, f] = pcm.G_to_dist(G)
                    Cosine[d, s, f] = pcm.G_to_cosine(G)
        np.save(os.path.join(gl.baseDir, args.experiment, 'coherence', 'Dissimilarity.z_coh.npy'), Dist)
        np.save(os.path.join(gl.baseDir, args.experiment, 'coherence', 'Cosine.z_coh.npy'), Cosine)

    if args.what=='mscohere_avg':
        D, F, S = 2, 513, len(args.sns)
        z_coh_avg = np.zeros((D, F, S, 2))
        for d, day in enumerate([1, 5]):
            for s, sn in enumerate(args.sns):
                print(f'doing participant {sn}, day {day}...')
                chordID = np.array(get_trained_and_untrained('EFC_learningEMG', sn)).astype(int)
                data = np.load(os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{sn}',
                                            'mscohere.npz'))
                dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'behavioural', f'day{day}',
                                 f'{args.experiment}_{sn}.dat'), sep='\t')
                dat = dat[~dat['chordID'].isin(single_finger)].reset_index(drop=True)
                trained = dat.chordID.isin(chordID[:4])
                coh = data['coh']
                z_coh = np.arctanh(np.sqrt(np.clip(coh, 1e-12, 1 - 1e-12)))
                mask = np.tri(10, k=-1, dtype=bool)
                z_coh = z_coh[:, mask, :].mean(axis=1)
                z_coh_avg[d, :, s, 0] = z_coh[trained].mean(axis=0)
                z_coh_avg[d, :, s, 1] = z_coh[~trained].mean(axis=0)
        da = xr.DataArray(
            z_coh_avg,
            dims=('day', 'freq', 'sn', 'chord'),
            coords={'day': [1, 5], 'freq': data['f'], 'sn': args.sns, 'chord': ['trained', 'untrained']},
        )
        da.to_netcdf(os.path.join(gl.baseDir, args.experiment, 'coherence', 'mscohere.h5'), engine="h5netcdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[100, 101, 102, 103, 104, 105, ])
    parser.add_argument('--experiment', type=str, default='EFC_learningEMG')
    parser.add_argument('--thresh', type=float, default=2)
    parser.add_argument('--edge', type=str, default='rising')

    args = parser.parse_args()

    main(args)

    # if args.what=='time-resolved_plv':
    #     for day in [1, 5]:
    #         savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
    #         for bl in range(10):
    #             excl = exclude_blocks[f'day{day}'][f'subj{args.sn}']
    #             if bl + 1 not in excl:
    #                 print(f'doing block {bl+1}...')
    #                 x_hilb = np.load(os.path.join(savedir, f'emg_hilbert_{bl + 1}.npy'))
    #                 timestamp = np.load(os.path.join(savedir, f'emg_timestamp_{bl + 1}.npy'))
    #                 timestamp_s = timestamp / fs_new
    #                 phi = np.angle(x_hilb)
    #                 plv, lag, bins = calc_plv_aligned(phi, fsample=fs_new, timestamp_s=timestamp_s, rel_range_s=(0, 4.5))
    #                 np.savez(os.path.join(savedir, f'emg_plv_binned_{bl + 1}.npz'), plv=plv, lag=lag, bins=bins)
    # if args.what=='time-resolved_plv_contrast':
    #     for day in [1, 5]:
    #         savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
    #         dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'behavioural', f'day{day}',
    #                                        f'{args.experiment}_{args.sn}.dat'), sep='\t')
    #         dat = dat[~dat['chordID'].isin(single_finger)]
    #         chord = pd.array(['trained' if str(chordID) in trained else 'untrained' for chordID in dat['chordID']])
    #         plv_b = []
    #         for bl in range(10):
    #             excl = exclude_blocks[f'day{day}'][f'subj{args.sn}']
    #             if bl + 1 not in excl:
    #                 print(f'doing block {bl+1}...')
    #                 data = np.load(os.path.join(savedir, f'emg_plv_binned_{bl + 1}.npz'))
    #                 plv, lag, bins = data['plv'], data['lag'], data['bins']
    #                 lag0 = lag[lag.size // 2]
    #                 mask = lag == lag0
    #                 plv = plv[..., mask] - np.nanmean(plv[..., ~mask], axis=-1, keepdims=True)
    #                 plv = np.nanmean(plv, axis=(3, 4, 5))
    #                 plv_b.append(plv)
    #         plv_b = np.vstack(plv_b)
    #         plv_c = np.nanmean(plv_b[chord=='trained'], axis=0) - np.nanmean(plv_b[chord=='untrained'], axis=0)
    #         np.savez(os.path.join(savedir, f'emg_plv_binned_tr_vs_untr.npz'), plv=plv_c, bins=bins)
    # if args.what=='demodulate':
    #     for day in [1, 5]:
    #         savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
    #         os.makedirs(savedir, exist_ok=True)
    #         for bl in range(10):
    #             excl = exclude_blocks[f'day{day}'][f'subj{args.sn}']
    #             if bl+1 not in excl:
    #                 print(f'subj{args.sn} - block {bl+1}')
    #                 emg_rect = np.load(os.path.join(savedir, f'emg_rect_{bl + 1}.npy'))
    #                 emg_demod = demodulate(emg_rect)
    #                 np.save(os.path.join(savedir, f'emg_demod_{bl + 1}.npy'), emg_demod)
    # if args.what=='filter_hilbert_raw':
    #     fh = FilterHilbert(fsample=fs_new, nfreq=60, freq_lim=(1, 60), width_lim=(.5, 3), n_ord=4)
    #     frequencies = fh.frequencies
    #     for day in [1, 5]:
    #         savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
    #         np.save(os.path.join(savedir, 'frequencies.npy'), frequencies)
    #         for bl in range(10):
    #             excl = exclude_blocks[f'day{day}'][f'subj{args.sn}']
    #             if bl + 1 not in excl:
    #                 print(f'subj{args.sn} - block {bl + 1}')
    #                 env = np.load(os.path.join(savedir, f'emg_raw_{bl + 1}.npy'))
    #                 x_hilb = fh.filter_hilbert(env, verbose=True)
    #                 np.save(os.path.join(savedir, f'emg_hilbert_{bl + 1}.npy'), x_hilb)
    # if args.what=='segment_hilbert_raw':
    #     for day in [1, 5]:
    #         savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
    #         x_hilb_segmented = []
    #         for bl in range(10):
    #             excl = exclude_blocks[f'day{day}'][f'subj{args.sn}']
    #             if bl + 1 not in excl:
    #                 # load delsys files
    #                 filepath = os.path.join(gl.baseDir, args.experiment, 'emg', f'day{day}', f'subj{args.sn}',
    #                                         f'{args.experiment}_{args.sn}_{bl + 1}.csv')
    #                 df_out = load_delsys(filepath, trigger_name='Trigger', muscle_names=channels_emg)
    #
    #                 # timestamp
    #                 trig_sig = np.abs(df_out.Trigger.to_numpy())
    #                 trig_time = df_out.time.astype(float).to_numpy()
    #                 trig_resampled = resample_to_fs(trig_sig, fs_old=fs_old, fs_new=fs_new)
    #                 time_resampled = resample_to_fs(trig_time, fs_old=fs_old, fs_new=fs_new)
    #                 _, timestamp = detect_trig(trig_resampled, time_resampled, amp_threshold=args.thresh,
    #                                            edge=args.edge)
    #                 print(f'Found {len(timestamp)} trials in block {bl + 1}...')
    #                 np.save(os.path.join(savedir, f'emg_timestamp_{bl + 1}.npy'), timestamp)
    #
    #                 x_hilb = np.load(os.path.join(savedir, f'emg_hilbert_{bl + 1}.npy'))
    #                 x_hilb_segmented.append(segment(x_hilb, timestamp, 1, 3, fs_new))
    #
    #         x_hilb_segmented = np.vstack(x_hilb_segmented)
    #         np.save(os.path.join(savedir, 'emg_hilbert.npy'), x_hilb_segmented)
    # if args.what=='plv':
    #     for day in [1, 5]:
    #         path = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
    #         x_hilb = np.load(os.path.join(path, 'emg_hilbert.npy'))
    #
    #         # calc ph consistency at lags
    #         plv, lag = phase_consistency_channels(x_hilb)
    #         np.save(os.path.join(path, 'emg_phase_consistency.npy'), plv)
    #         np.save(os.path.join(path, 'lag.npy'), lag)
    #
    #         # calc ph consistency surrogate
    #         plv_shuffle = phase_consistency_channels_shuffle(x_hilb)
    #         np.save(os.path.join(path, 'emg_phase_consistency_surrogate.npy'), plv_shuffle)
    # if args.what=='plv_avg':
    #     for day in [1, 5]:
    #         # load dat and exclude single finger trials
    #         dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'behavioural', f'day{day}',
    #                            f'{args.experiment}_{args.sn}.dat'), sep='\t')
    #         dat = dat[~dat['chordID'].isin(single_finger)]
    #
    #         # load plv and freqs
    #         path = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
    #         plv = np.load(os.path.join(path, 'emg_phase_consistency.npy'))
    #         freq = np.load(os.path.join(path, 'frequencies.npy'))
    #         nchan = plv.shape[-2]
    #
    #         plv_shuffle = np.load(os.path.join(path, 'emg_phase_consistency_surrogate.npy'))
    #         plv_shuffle_avg = plv_shuffle.mean(axis=(-2, -1))
    #
    #         # calc avg across channels
    #         mask = np.triu(np.ones((nchan, nchan), dtype=bool), k=1)
    #         plv_raw = plv[..., mask, :].mean(-2)
    #         plv_corr = plv_raw[..., 200] - plv_raw.mean(axis=-1)
    #         plv_z = calc_plv_z(plv_raw, plv_shuffle_avg)
    #
    #         chord = ['trained' if str(chordID) in trained else 'untrained' for chordID in dat['chordID']]
    #         df_raw = pd.DataFrame(plv_raw[..., 200], columns=freq)
    #         df_corr = pd.DataFrame(plv_corr, columns=freq)
    #         df_z = pd.DataFrame(plv_z[...,200], columns=freq)
    #         df_meta = dat[['chordID', 'trialPoint', 'day', 'week', 'session']].reset_index(drop=True)
    #         assert df_meta.shape[0]==df_raw.shape[0], f"Trial number mismatch on day{day}"
    #         df = pd.concat([df_raw, df_meta], axis=1)
    #         df['chord'] = chord
    #         df['sn'] = args.sn
    #         df_melt = pd.melt(df, id_vars=['chordID', 'trialPoint', 'day', 'week', 'session', 'chord', 'sn'],
    #                           value_vars=freq, value_name='plv_raw', var_name='frequency')
    #         df_corr = pd.melt(df_corr, value_vars=freq, value_name='plv0', var_name='frequency')
    #         df_z = pd.melt(df_z, value_vars=freq, value_name='z-plv', var_name='frequency')
    #         df_melt['plv0'] = df_corr['plv0']
    #         df_melt['z-plv'] = df_z['z-plv']
    #         df_melt.to_csv(os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}',
    #                                'phase_consistency.tsv'), sep='\t', index=False)
    # if args.what=='plv_pipeline':
    #     for sn in args.sns:
    #         print(f'doing participant {sn}')
    #         main(argparse.Namespace(
    #             what='raw',
    #             experiment=args.experiment,
    #             sn=sn,
    #             thresh=args.thresh,
    #             edge=args.edge,))
    #         main(argparse.Namespace(
    #             what='filter_hilbert_raw',
    #             experiment=args.experiment,
    #             sn=sn))
    #         main(argparse.Namespace(
    #             what='segment_hilbert_raw',
    #             experiment=args.experiment,
    #             sn=sn,
    #             thresh=args.thresh,
    #             edge=args.edge,))
    #         main(argparse.Namespace(
    #             what='plv',
    #             experiment=args.experiment,
    #             sn=sn))
    #         main(argparse.Namespace(
    #             what='plv_avg',
    #             experiment=args.experiment,
    #             sn=sn))
    #     main(argparse.Namespace(
    #         what='plv_pooled_df',
    #         experiment=args.experiment))
    # if args.what=='envelope':
    #     for day in [1, 5]:
    #         savedir = os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{args.sn}')
    #         os.makedirs(savedir, exist_ok=True)
    #         for bl in range(10):
    #             excl = exclude_blocks[f'day{day}'][f'subj{args.sn}']
    #             if bl+1 not in excl:
    #                 print(f'subj{args.sn} - block {bl+1}')
    #
    #                 # load delsys files
    #                 filepath = os.path.join(gl.baseDir, args.experiment, 'emg', f'day{day}', f'subj{args.sn}',
    #                                         f'{args.experiment}_{args.sn}_{bl+1}.csv')
    #                 df_out = load_delsys(filepath, trigger_name='Trigger', muscle_names=channels_emg)
    #
    #                 # calc emg envelope/rect
    #                 emg_raw = df_out[channels_emg].to_numpy().T
    #                 emg_resampled = resample_to_fs(emg_raw, fs_old=fs_old, fs_new=fs_new)
    #                 emg_hp = hp_filter(emg_resampled, n_ord=4, cutoff=20, fsample=fs_new, axis=-1)
    #                 env = np.abs(hilbert(emg_hp, axis=-1))
    #                 np.save(os.path.join(savedir, f'emg_envelope_{bl+1}.npy'), env)
    # if args.what=='mscohere_pooled':
    #     coh, info = [], pd.DataFrame()
    #     for d, day in enumerate([1, 5]):
    #         for sn in args.sns:
    #             dat = pd.read_csv(
    #                 os.path.join(gl.baseDir, args.experiment, 'behavioural', f'day{day}', f'{args.experiment}_{sn}.dat'),
    #                 sep='\t')
    #             dat = dat[~dat['chordID'].isin(single_finger)].reset_index(drop=True)
    #
    #             trained = pinfo.loc[pinfo['sn'] == sn, 'trained'].reset_index(drop=True)[0].split('.')
    #             chord = np.array(['trained' if str(cid) in trained else 'untrained' for cid in dat['chordID']])
    #             dat['chord'] = chord
    #
    #             data = np.load(os.path.join(gl.baseDir, args.experiment, 'coherence', f'day{day}', f'subj{sn}', 'mscohere.npz'))
    #             coh_tmp = data['coh']  # shape: (trials, freq, chan, chan)
    #             f = data['f']
    #
    #             coh.append(coh_tmp)
    #             info = pd.concat([info, dat[['subNum', 'chordID', 'day', 'chord', 'BN']]])
    #
    #     coh = np.vstack(coh)
    #     assert coh.shape[0] == info.shape[0], "data and info have different shape"
    #     np.savez(os.path.join(gl.baseDir, args.experiment, 'coherence', 'mscohere.npz'), coh=coh, f=f)
    #     info.to_csv(os.path.join(gl.baseDir, args.experiment, 'coherence', 'info.tsv'), sep='\t', index=False)