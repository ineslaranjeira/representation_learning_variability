import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from scipy.signal import butter, filtfilt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from Functions.video_functions import lick_psth
except ImportError:
    raise ImportError("Could not import lick_psth from Functions.video_functions. Ensure the workspace root is on sys.path.")


def extended_qc(one, eids):
    """Retrieve extended QC data for a list of session eids."""
    df = pd.DataFrame()
    for eid in eids:
        extended_qc = one.get_details(eid, True)['extended_qc']
        transposed_df = pd.DataFrame.from_dict(extended_qc, orient='index').T
        transposed_df['eid'] = eid
        df = pd.concat([df, transposed_df], ignore_index=True)
    return df


def get_feature_event_times(dlc, dlc_t, features):
    """Detect event times from keypoint feature traces."""
    for i, feat in enumerate(features):
        f = dlc[feat]
        threshold = np.nanstd(np.diff(f)) / 4
        if i == 0:
            events = np.where(np.abs(np.diff(f)) > threshold)[0]
        else:
            events = np.r_[events, np.where(np.abs(np.diff(f)) > threshold)[0]]
    return dlc_t[np.unique(events)]


def merge_licks(poses, features, common_fs):
    """Merge lick detection from left and right camera pose features."""
    duration_sec = max(list(poses['leftCamera']['times'])[-1], list(poses['rightCamera']['times'])[-1])
    t_common = np.arange(0, duration_sec, 1 / common_fs)
    lick_trace_left = np.zeros_like(t_common, dtype=int)
    lick_trace_right = np.zeros_like(t_common, dtype=int)

    left_lick_times = get_feature_event_times(poses['leftCamera'], poses['leftCamera']['times'], features)
    right_lick_times = get_feature_event_times(poses['rightCamera'], poses['rightCamera']['times'], features)

    left_indices = np.searchsorted(t_common, left_lick_times)
    right_indices = np.searchsorted(t_common, right_lick_times)
    lick_trace_left[left_indices[left_indices < len(t_common)]] = 1
    lick_trace_right[right_indices[right_indices < len(t_common)]] = 1

    combined_licks = np.maximum(lick_trace_left, lick_trace_right)
    return t_common, combined_licks


def resample_common_time(reference_time, timestamps, data, kind, fill_gaps=None):
    """Resample a signal onto a common timeline."""
    yinterp = interpolate.interp1d(timestamps, data, kind=kind, fill_value='extrapolate')(reference_time)
    if fill_gaps:
        gaps, = np.where(np.diff(timestamps) >= fill_gaps)
        for i in gaps:
            yinterp[(reference_time >= timestamps[i]) & (reference_time < timestamps[i + 1])] = data[i]
    return yinterp, reference_time


def lowpass_filter(data, cutoff, fs, order=4):
    """Apply a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def low_pass(signal, cutoff, sf):
    """Low-pass filter a signal while preserving NaN positions."""
    not_nan = signal[np.where(~np.isnan(signal))]
    filtered = lowpass_filter(not_nan, cutoff, fs=sf, order=4)
    signal[np.where(~np.isnan(signal))] = filtered
    return signal


def interpolate_nans(pose, fr):
    """Interpolate short NaN gaps in pose traces."""
    nan_thresh = 0.1
    t = np.diff(1 * np.isnan(np.array(pose)))
    begs = np.where(t == 1)[0]
    ends = np.where(t == -1)[0]
    if np.isnan(np.array(pose)[0]):
        begs = begs[:-1]
        ends = ends[1:]
    if begs.shape[0] > ends.shape[0]:
        begs = begs[:ends.shape[0]]

    interp_pose = pose.copy()
    interp_pose = np.array(interp_pose.interpolate(method='cubic'))

    for b, e in zip(begs, ends):
        if (e - b) > (fr * nan_thresh):
            interp_pose[(b + 1):(e + 1)] = np.nan
    return interp_pose


def get_speed(poses, times, camera, sampling_rate, split, feature):
    RESOLUTION = {'left': 2, 'right': 1, 'body': 1}
    sampling_rate = 60
    speeds = {}
    times = np.array(times)
    x = poses[f'{feature}_x'] / RESOLUTION[camera]
    y = poses[f'{feature}_y'] / RESOLUTION[camera]
    dt = np.diff(times)
    tv = times[:-1] + dt / 2
    if split:
        s_x = np.diff(x) * sampling_rate
        s_y = np.diff(y) * sampling_rate
        if tv.size > 1:
            ifcn_x = interpolate.interp1d(tv, s_x, fill_value="extrapolate")
            ifcn_y = interpolate.interp1d(tv, s_y, fill_value="extrapolate")
            speeds = [times, ifcn_x(times), ifcn_y(times)]
        else:
            speeds = [times, s_x, s_y]
    else:
        s = ((np.diff(x)**2 + np.diff(y)**2)**0.5) * sampling_rate
        if tv.size > 1:
            ifcn = interpolate.interp1d(tv, s, fill_value="extrapolate")
            speeds = [times, ifcn(times)]
        else:
            speeds = [times, s]
    return speeds


def event_locked_signal(t, v, events, window=(-.5, 1)):
    fs = 1 / np.median(np.diff(t))
    w = np.arange(int(window[0] * fs), int(window[1] * fs))
    idx = np.searchsorted(t, events)
    X = np.stack([v[i + w] for i in idx if i + w.min() >= 0 and i + w.max() < len(v)])
    return X, w / fs


def plot_licks_PSTH(eid, trial_df, lick_times, save_path):
    data = lick_psth(trial_df, lick_times, 1, 2, event='feedback_times')
    if len(data) > 0:
        trials = data['trial']
        num_trials = int(np.max(trials))
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=True, figsize=[9, 5])
        for t, trial in enumerate(range(num_trials)):
            licks_correct = np.array(data.loc[(data['trial'] == t) & (data['correct'] == 1), 'lick_times'])
            licks_incorrect = np.array(data.loc[(data['trial'] == t) & (data['correct'] == -1), 'lick_times'])
            ax.scatter(licks_correct, np.full_like(licks_correct, t), color='green', s=1)
            ax.scatter(licks_incorrect, np.full_like(licks_incorrect, t), color='red', s=1)
        ax.set_xlabel('Time from feedback')
        ax.set_ylabel('Trial')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(str(Path(save_path) / f"{eid}_lick_psth_-12_feedback.png"), format='png')
        plt.close(fig)
    else:
        print(f'No PSTH for session {eid}')


def plot_whisker_psth(mat, design_matrix, trial_df, event, save_path):
    X, tscale = event_locked_signal(np.array(design_matrix['Bin']), np.array(design_matrix['avg_whisker_me']), trial_df[event])
    m = np.nanmean(X, axis=0)
    s = np.nanstd(X, axis=0) / np.sqrt(np.sum(~np.isnan(X), axis=0))
    fig, ax = plt.subplots()
    ax.plot(tscale, m)
    ax.fill_between(tscale, m - s, m + s, alpha=0.3)
    ax.axvline(0, color='k')
    ax.set_xlabel('Time from event (s)')
    ax.set_ylabel('Whisker ME')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(Path(save_path) / f"{mat}_me_psth.png"), format='png')
    plt.close(fig)


def plot_paw_hist(mat, design_matrix, save_path):
    left_speeds = get_speed(design_matrix, design_matrix['Bin'], 'left', 60, split=False, feature='l_paw')
    right_speeds = get_speed(design_matrix, design_matrix['Bin'], 'right', 60, split=False, feature='r_paw')
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[1:4, 0:3])
    x_hist = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
    y_hist = fig.add_subplot(grid[1:4, 3], sharey=main_ax)
    main_ax.scatter(np.abs(design_matrix['avg_wheel_vel']), left_speeds[1], alpha=0.2, s=.1, label='Left paw', color='C0')
    main_ax.scatter(np.abs(design_matrix['avg_wheel_vel']), right_speeds[1], alpha=0.2, s=.1, label='Right paw', color='C1')
    main_ax.legend()
    y_hist.hist([left_speeds[1], right_speeds[1]], bins=100, color=['C0', 'C1'], alpha=0.6, orientation='horizontal')
    x_hist.hist([np.abs(design_matrix['avg_wheel_vel']), np.abs(design_matrix['avg_wheel_vel'])], bins=100, color=['C0', 'C1'], alpha=0.6)
    main_ax.set_ylim([-5, int(np.nanmax([left_speeds[1], right_speeds[1]]) / 2)])
    main_ax.set_xlim([-.1, int(np.nanmax(np.abs(design_matrix['avg_wheel_vel'])) / 2)])
    main_ax.set_xlabel('Wheel velocity')
    main_ax.set_ylabel('Paw velocity')
    main_ax.set_title(mat)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(Path(save_path) / f"{mat}_paw_hist.png"), format='png')
    plt.close(fig)


def plot_choice_psth(mat, design_matrix, trial_df, save_path, event_key, feature, camera_split=False, title=None, suffix=''):
    left_speeds = get_speed(design_matrix, design_matrix['Bin'], 'left', 60, split=camera_split, feature='l_paw' if 'paw' in feature else feature)
    right_speeds = get_speed(design_matrix, design_matrix['Bin'], 'right', 60, split=camera_split, feature='r_paw' if 'paw' in feature else feature)
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=[9, 5])
    for c, choice in enumerate([-1.0, 1.0]):
        use_data = trial_df.loc[trial_df['choice'] == choice]
        if 'paw' in feature:
            right_paw, r_tscale = event_locked_signal(right_speeds[0], right_speeds[1], use_data['stimOn_times'])
            left_paw, l_tscale = event_locked_signal(left_speeds[0], left_speeds[1], use_data['stimOn_times'])
            l = np.nanmean(left_paw, axis=0)
            s_l = np.nanstd(left_paw, axis=0) / np.sqrt(np.sum(~np.isnan(l), axis=0))
            r = np.nanmean(right_paw, axis=0)
            s_r = np.nanstd(right_paw, axis=0) / np.sqrt(np.sum(~np.isnan(r), axis=0))
            ax[c].plot(l_tscale, l, label='left')
            ax[c].fill_between(l_tscale, l - s_l, l + s_l, alpha=0.3)
            ax[c].plot(r_tscale, r, label='right')
            ax[c].fill_between(r_tscale, r - s_r, r + s_r, alpha=0.3)
        else:
            signal, tscale = event_locked_signal(np.array(design_matrix['Bin']), np.array(design_matrix[feature]), use_data['stimOn_times'])
            l = np.nanmean(np.abs(signal), axis=0)
            s_l = np.nanstd(np.abs(signal), axis=0) / np.sqrt(np.sum(~np.isnan(l), axis=0))
            ax[c].plot(tscale, l, label=choice)
            ax[c].fill_between(tscale, l - s_l, l + s_l, alpha=0.3)
        ax[c].axvline(0, linestyle='dashed', color='k')
        ax[c].set_title(title or f'Choice {choice}')
        ax[c].set_xlabel('Time from event (s)')
        ax[c].set_xlim([-0.25, 0.75])
        ax[c].set_ylabel('Velocity')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(Path(save_path) / f"{mat}{suffix}.png"), format='png')
    plt.close(fig)


def plot_paw_feedback_psth(mat, design_matrix, trial_df, save_path, suffix='paw_feedback_psth'):
    plot_choice_psth(mat, design_matrix, trial_df, save_path, event_key='feedbackType', feature='paw', camera_split=True, title='Paw Feedback PSTH', suffix=suffix)


def plot_paw_left_feedbackpsth(mat, design_matrix, trial_df, save_path):
    left_speeds = get_speed(design_matrix, design_matrix['Bin'], 'left', 60, split=True, feature='l_paw')
    fig, ax = plt.subplots(figsize=(5, 5))
    for choice in [-1.0, 1.0]:
        use_data = trial_df.loc[trial_df['feedbackType'] == choice]
        left_paw, l_tscale = event_locked_signal(left_speeds[0], left_speeds[1], use_data['stimOn_times'])
        l = np.nanmean(np.abs(left_paw), axis=0)
        s_l = np.nanstd(np.abs(left_paw), axis=0) / np.sqrt(np.sum(~np.isnan(l), axis=0))
        ax.plot(l_tscale, l, label=choice)
        ax.fill_between(l_tscale, l - s_l, l + s_l, alpha=0.3)
    ax.axvline(0, linestyle='dashed', color='k')
    ax.set_title('Left paw')
    ax.set_xlabel('Time from event (s)')
    ax.set_xlim([-0.25, 0.75])
    ax.set_ylabel('Velocity')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(Path(save_path) / f"{mat}_paw_left_psth.png"), format='png')
    plt.close(fig)


def plot_paw_right_feedbackpsth(mat, design_matrix, trial_df, save_path):
    right_speeds = get_speed(design_matrix, design_matrix['Bin'], 'right', 60, split=True, feature='r_paw')
    fig, ax = plt.subplots(figsize=(5, 5))
    for choice in [-1.0, 1.0]:
        use_data = trial_df.loc[trial_df['feedbackType'] == choice]
        right_paw, l_tscale = event_locked_signal(right_speeds[0], right_speeds[1], use_data['stimOn_times'])
        l = np.nanmean(np.abs(right_paw), axis=0)
        s_l = np.nanstd(np.abs(right_paw), axis=0) / np.sqrt(np.sum(~np.isnan(l), axis=0))
        ax.plot(l_tscale, l, label=choice)
        ax.fill_between(l_tscale, l - s_l, l + s_l, alpha=0.3)
    ax.axvline(0, linestyle='dashed', color='k')
    ax.set_title('Right paw')
    ax.set_xlabel('Time from event (s)')
    ax.set_xlim([-0.25, 0.75])
    ax.set_ylabel('Velocity')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(Path(save_path) / f"{mat}_paw_right_psth.png"), format='png')
    plt.close(fig)


def plot_wheel_choicepsth(mat, design_matrix, trial_df, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for choice in [-1.0, 1.0]:
        use_data = trial_df.loc[trial_df['choice'] == choice]
        signal, tscale = event_locked_signal(np.array(design_matrix['Bin']), np.array(design_matrix['avg_wheel_vel']), use_data['stimOn_times'])
        l = np.nanmean(np.abs(signal), axis=0)
        s_l = np.nanstd(np.abs(signal), axis=0) / np.sqrt(np.sum(~np.isnan(l), axis=0))
        ax.plot(tscale, l, label=choice)
        ax.fill_between(tscale, l - s_l, l + s_l, alpha=0.3)
    ax.axvline(0, linestyle='dashed', color='k')
    ax.set_title('Wheel')
    ax.set_xlabel('Time from event (s)')
    ax.set_xlim([-0.25, 0.75])
    ax.set_ylabel('Velocity')
    ax.legend()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(Path(save_path) / f"{mat}_wheel_choice_psth.png"), format='png')
    plt.close(fig)
