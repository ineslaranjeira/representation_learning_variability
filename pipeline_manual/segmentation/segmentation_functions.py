import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as interpolate
from scipy.signal import butter, filtfilt
from joblib import Parallel, delayed
from scipy.fftpack import fft, ifft, fftshift
# from jax import vmap
# import jax.numpy as jnp
from matplotlib.patches import Rectangle
from sklearn import mixture
from scipy.stats import mode


"""
SCRIPT 1: QUERY BWM DATA WITH QC
"""

def extended_qc(one, eids):
    
    # Initialize df
    df = pd.DataFrame()

    for e, eid in enumerate(eids):
        
        extended_qc = one.get_details(eid, True)['extended_qc']
        transposed_df = pd.DataFrame.from_dict(extended_qc, orient='index').T
        transposed_df['eid'] = eid
        df = pd.concat([df, transposed_df])
    
    return df


"""
SCRIPT 2: DESIGN MATRIX
"""

## LICKS
def get_feature_event_times(dlc, dlc_t, features):
    """
    Detect events from the dlc traces. Based on the standard deviation between frames
    :param dlc: dlc pqt table
    :param dlc_t: dlc times
    :param features: features to consider
    :return:
    """

    for i, feat in enumerate(features):
        f = dlc[feat]
        threshold = np.nanstd(np.diff(f)) / 4
        if i == 0:
            events = np.where(np.abs(np.diff(f)) > threshold)[0]
        else:
            events = np.r_[events, np.where(np.abs(np.diff(f)) > threshold)[0]]

    return dlc_t[np.unique(events)]


def merge_licks(poses, features, common_fs):
    
    # Define total duration (max of both videos)
    duration_sec = max(list(poses['leftCamera']['times'])[-1], list(poses['rightCamera']['times'])[-1])  # in seconds

    # Set common sampling rate (high rather than low)
    t_common = np.arange(0, duration_sec, 1/common_fs)  # uniform timestamps
    
    lick_trace_left = np.zeros_like(t_common, dtype=int)
    lick_trace_right = np.zeros_like(t_common, dtype=int)

    left_lick_times = get_feature_event_times(poses['leftCamera'], poses['leftCamera']['times'], features)
    right_lick_times = get_feature_event_times(poses['rightCamera'], poses['rightCamera']['times'], features)

    # Round licks to nearest timestamp in t_common
    left_indices = np.searchsorted(t_common, left_lick_times)
    right_indices = np.searchsorted(t_common, right_lick_times)

    # Set licks to 1
    lick_trace_left[left_indices[left_indices < len(t_common)]] = 1
    lick_trace_right[right_indices[right_indices < len(t_common)]] = 1

    combined_licks = np.maximum(lick_trace_left, lick_trace_right)

    return t_common, combined_licks 


def resample_common_time(reference_time, timestamps, data, kind, fill_gaps=None):
    # Function inspired on wh.interpolate from here: https://github.com/int-brain-lab/ibllib/blob/master/brainbox/behavior/wheel.py#L28 
    yinterp = interpolate.interp1d(timestamps, data, kind=kind, fill_value='extrapolate')(reference_time)
    
    if fill_gaps:
        #  Find large gaps and forward fill @fixme This is inefficient
        gaps, = np.where(np.diff(timestamps) >= fill_gaps)

        for i in gaps:
            yinterp[(reference_time >= timestamps[i]) & (reference_time < timestamps[i + 1])] = data[i]
            
    return yinterp, reference_time


def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def low_pass(signal, cutoff, sf):    
    not_nan = signal[np.where(~np.isnan(signal))]
    low_pass = lowpass_filter(not_nan, cutoff, fs=sf, order=4)
    signal[np.where(~np.isnan(signal))] = low_pass
    return signal


def interpolate_nans(pose, fr):

    # threshold (in seconds) above which we will not interpolate nans,
    # but keep them (for long stretches interpolation may not be appropriate)
    nan_thresh = .1
    # SAMPLING = {'left': 60,
    #             'right': 150,
    #             'body': 30}
    # fr = SAMPLING[camera]

    # don't interpolate long strings of nans
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

    # Restore long NaNs
    for b, e in zip(begs, ends):
        if (e - b) > (fr * nan_thresh):
            interp_pose[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff
        
    return interp_pose


def lick_psth(trials, licks, t_init, t_end, event='feedback_times'):
    
    event_times = trials[event]
    feedback_type = trials['feedbackType']

    licks_df = pd.DataFrame(columns=['trial', 'lick_times', 'correct'])

    for t, trial in enumerate(event_times):
        event_time = event_times[t]
        correct = feedback_type[t]
        start = event_time - t_init
        end = event_time + t_end
        trial_licks = licks[(licks>start) & (licks<end)]
        aligned_lick_times = trial_licks - event_time
        
        # Temp dataframe
        temp_df = pd.DataFrame(columns=['trial', 'lick_times', 'correct'])
        temp_df['lick_times'] = aligned_lick_times
        temp_df['trial'] = np.ones(len(aligned_lick_times)) * t
        temp_df['correct'] = correct
        
        licks_df = pd.concat([licks_df, temp_df])
        
    return licks_df


def plot_licks_PSTH(eid, trial_df, lick_times, save_path):
    
    data = lick_psth(trial_df, lick_times, 1, 2, event='feedback_times')
    
    if len(data) > 0:
        # data = data[0]
        trials = data['trial']
        num_trials = int(np.max(trials))
        # Plot data
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=True, figsize=[9, 5])
        for t, trial in enumerate(range(num_trials)):
            licks_correct = np.array(data.loc[(data['trial']==t) & (data['correct']==1), 'lick_times'])
            licks_incorrect = np.array(data.loc[(data['trial']==t) & (data['correct']==-1), 'lick_times'])
            ax.scatter(licks_correct, np.full_like(licks_correct, t), color='green', s=1)
            ax.scatter(licks_incorrect, np.full_like(licks_incorrect, t), color='red', s=1)

        ax.set_xlabel('Time from feedback')
        ax.set_ylabel('Trial')
        plt.savefig(str(save_path + eid + 'lick_psth_-12_feedback.png'), format='png')
        plt.show()
        
    else:
        print('No PSTH for session ' + eid)
        
# This function uses get_XYs, not smoothing, is closer to brainbox function: https://github.com/int-brain-lab/ibllib/blob/78e82df8a51de0be880ee4076d2bb093bbc1d2c1/brainbox/behavior/dlc.py#L63
def get_speed(poses, times, camera, sampling_rate, split, feature):
    """
    FIXME Document and add unit test!

    :param dlc: dlc pqt table
    :param dlc_t: dlc time points
    :param camera: camera type e.g 'left', 'right', 'body'
    :param feature: dlc feature to compute speed over
    :return:
    """

    RESOLUTION = {'left': 2,
                  'right': 1,
                  'body': 1}
    sampling_rate = 60

    speeds = {}
    times = np.array(times)
    x = poses[f'{feature}_x'] / RESOLUTION[camera]
    y = poses[f'{feature}_y'] / RESOLUTION[camera]

    dt = np.diff(times)
    tv = times[:-1] + dt / 2


    # Calculate velocity for x and y separately if split is true
    if split == True:
        s_x = np.diff(x) * sampling_rate
        s_y = np.diff(y) * sampling_rate
        speeds = [times, s_x, s_y]
        # interpolate over original time scale
        if tv.size > 1:
            ifcn_x = interpolate.interp1d(tv, s_x, fill_value="extrapolate")
            ifcn_y = interpolate.interp1d(tv, s_y, fill_value="extrapolate")
            speeds = [times, ifcn_x(times), ifcn_y(times)]
    else:
        # Speed vector is given by the Pitagorean theorem
        s = ((np.diff(x)**2 + np.diff(y)**2)**.5) * sampling_rate
        speeds = [times, s]
        # interpolate over original time scale
        if tv.size > 1:
            ifcn = interpolate.interp1d(tv, s, fill_value="extrapolate")
            speeds = [times, ifcn(times)]

    return speeds  


def event_locked_signal(t, v, events, window=(-.5,1)):
    fs = 1/np.median(np.diff(t))
    w = np.arange(int(window[0]*fs), int(window[1]*fs))
    idx = np.searchsorted(t, events)
    X = np.stack([v[i+w] for i in idx if i+w.min()>=0 and i+w.max()<len(v)])
    return X, w/fs


"""
SCRIPT 2.1: DESIGN MATRIX INSPECTION
"""

def plot_whisker_psth(mat, design_matrix, trial_df, event, save_path):

    X, tscale = event_locked_signal(np.array(design_matrix['Bin']), np.array(design_matrix['avg_whisker_me']), trial_df[event])
    m = np.nanmean(X, axis=0)
    s = np.nanstd(X, axis=0) / np.sqrt(np.sum(~np.isnan(X), axis=0))
    plt.plot(tscale, m)
    plt.fill_between(tscale, m - s, m + s, alpha=0.3)
    plt.axvline(0, color='k')
    plt.xlabel("Time from event (s)")
    plt.ylabel("Whisker ME")

    # Save the plot as a PNG file
    plt.savefig(str(save_path + mat + '_me_psth.png'), format='png')
    plt.show()


def plot_paw_hist(mat, design_matrix, save_path):

    left_speeds = get_speed(design_matrix, design_matrix['Bin'], 'left', 60, split=False, feature='l_paw')
    right_speeds = get_speed(design_matrix, design_matrix['Bin'], 'right', 60, split=False, feature='r_paw')
    
    # Create main figure
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

    # Define areas
    main_ax = fig.add_subplot(grid[1:4, 0:3])
    x_hist = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
    y_hist = fig.add_subplot(grid[1:4, 3], sharey=main_ax)

    # Scatterplot (two hues)
    main_ax.scatter(np.abs(design_matrix['avg_wheel_vel']), left_speeds[1], alpha=0.2, s=.1, label='Left paw', color='C0')
    main_ax.scatter(np.abs(design_matrix['avg_wheel_vel']), right_speeds[1], alpha=0.2, s=.1, label='Right paw', color='C1')
    main_ax.legend()
    
    # Marginal histograms
    y_hist.hist([left_speeds[1],right_speeds[1]], bins=100, color=['C0', 'C1'], alpha=0.6, orientation='horizontal')
    x_hist.hist([np.abs(design_matrix['avg_wheel_vel']), np.abs(design_matrix['avg_wheel_vel'])], bins=100, color=['C0', 'C1'], alpha=0.6)

    # Hide tick labels on shared axes
    main_ax.set_ylim([-5, int(np.nanmax([left_speeds[1], right_speeds[1]])/2)])
    main_ax.set_xlim([-.1, int(np.nanmax(np.abs(design_matrix['avg_wheel_vel']))/2)])
    main_ax.set_xlabel('Wheel velocity')
    main_ax.set_ylabel('Paw velocity')
    main_ax.set_title(mat)
    plt.tight_layout()
    # Save the plot as a PNG file
    plt.savefig(str(save_path + mat + '_paw_hist.png'), format='png')
    plt.show()

def plot_paw_choice_psth(mat, design_matrix, trial_df, save_path):

    left_speeds = get_speed(design_matrix, design_matrix['Bin'], 'left', 60, split=True, feature='l_paw')
    right_speeds = get_speed(design_matrix, design_matrix['Bin'], 'right', 60, split=True, feature='r_paw')

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=[9, 5])

    for c, choice in enumerate([-1.0, 1.0]):

        use_data = trial_df.loc[trial_df['choice']==choice]
        right_paw, r_tscale = event_locked_signal(right_speeds[0], right_speeds[1], use_data['stimOn_times'])
        left_paw, l_tscale = event_locked_signal(left_speeds[0], left_speeds[1], use_data['stimOn_times'])

        l = np.nanmean(left_paw, axis=0)
        s_l = np.nanstd(left_paw, axis=0) / np.sqrt(np.sum(~np.isnan(l), axis=0))
        ax[c].plot(l_tscale, l, label='left')
        ax[c].fill_between(l_tscale, l - s_l, l + s_l, alpha=0.3)
        ax[c].legend()
        r = np.nanmean(right_paw, axis=0)
        s_r = np.nanstd(right_paw, axis=0) / np.sqrt(np.sum(~np.isnan(r), axis=0))
        ax[c].plot(r_tscale, r, label='right')
        ax[c].fill_between(r_tscale, r - s_r, r + s_r, alpha=0.3)
        ax[c].axvline(0, linestyle='dashed', color='k')
        ax[c].set_title('Choice'+str(choice))
        ax[c].set_xlabel("Time from event (s)")
        ax[c].set_xlim([-0.25, 0.75])
        ax[c].set_ylabel("Velocity")

    plt.savefig(str(save_path + mat + 'paw_choice_psth.png'), format='png')
    plt.show()

def plot_paw_feedback_psth(mat, design_matrix, trial_df, save_path):

    left_speeds = get_speed(design_matrix, design_matrix['Bin'], 'left', 60, split=True, feature='l_paw')
    right_speeds = get_speed(design_matrix, design_matrix['Bin'], 'right', 60, split=True, feature='r_paw')

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=[9, 5])

    for c, choice in enumerate([-1.0, 1.0]):

        use_data = trial_df.loc[trial_df['feedbackType']==choice]
        right_paw, r_tscale = event_locked_signal(right_speeds[0], right_speeds[1], use_data['stimOn_times'])
        left_paw, l_tscale = event_locked_signal(left_speeds[0], left_speeds[1], use_data['stimOn_times'])

        l = np.nanmean(left_paw, axis=0)
        s_l = np.nanstd(left_paw, axis=0) / np.sqrt(np.sum(~np.isnan(l), axis=0))
        ax[c].plot(l_tscale, l, label='left')
        ax[c].fill_between(l_tscale, l - s_l, l + s_l, alpha=0.3)
        ax[c].legend()
        r = np.nanmean(right_paw, axis=0)
        s_r = np.nanstd(right_paw, axis=0) / np.sqrt(np.sum(~np.isnan(r), axis=0))
        ax[c].plot(r_tscale, r, label='right')
        ax[c].fill_between(r_tscale, r - s_r, r + s_r, alpha=0.3)
        ax[c].axvline(0, linestyle='dashed', color='k')
        ax[c].set_title('Choice'+str(choice))
        ax[c].set_xlabel("Time from event (s)")
        ax[c].set_xlim([-0.25, 0.75])
        ax[c].set_ylabel("Velocity")

    plt.savefig(str(save_path + mat + 'paw_psth_feedback.png'), format='png')
    plt.show()

def plot_paw_left_feedbackpsth(mat, design_matrix, trial_df, save_path):

    left_speeds = get_speed(design_matrix, design_matrix['Bin'], 'left', 60, split=True, feature='l_paw')

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=True, figsize=[5, 5])

    for c, choice in enumerate([-1.0, 1.0]):

        use_data = trial_df.loc[trial_df['feedbackType']==choice]
        left_paw, l_tscale = event_locked_signal(left_speeds[0], left_speeds[1], use_data['stimOn_times'])

        l = np.nanmean(np.abs(left_paw), axis=0)
        s_l = np.nanstd(np.abs(left_paw), axis=0) / np.sqrt(np.sum(~np.isnan(l), axis=0))
        ax.plot(l_tscale, l, label=choice)
        ax.fill_between(l_tscale, l - s_l, l + s_l, alpha=0.3)
        ax.legend()

        ax.axvline(0, linestyle='dashed', color='k')
        ax.set_title('Left paw')
        ax.set_xlabel("Time from event (s)")
        ax.set_xlim([-0.25, 0.75])
        ax.set_ylabel("Velocity")

    plt.savefig(str(save_path + mat + 'paw_left_psth.png'), format='png')
    plt.show()

def plot_wheel_choicepsth(mat, design_matrix, trial_df, save_path):

    # left_speeds = get_speed(design_matrix, design_matrix['Bin'], 'left', 60, split=True, feature='l_paw')

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=True, figsize=[5, 5])

    for c, choice in enumerate([-1.0, 1.0]):

        use_data = trial_df.loc[trial_df['choice']==choice]
        left_paw, l_tscale = event_locked_signal(np.array(design_matrix['Bin']), np.array(design_matrix['avg_wheel_vel']), use_data['stimOn_times'])

        l = np.nanmean(np.abs(left_paw), axis=0)
        s_l = np.nanstd(np.abs(left_paw), axis=0) / np.sqrt(np.sum(~np.isnan(l), axis=0))
        ax.plot(l_tscale, l, label=choice)
        ax.fill_between(l_tscale, l - s_l, l + s_l, alpha=0.3)
        ax.legend()

        ax.axvline(0, linestyle='dashed', color='k')
        ax.set_title('Wheel')
        ax.set_xlabel("Time from event (s)")
        ax.set_xlim([-0.25, 0.75])
        ax.set_ylabel("Velocity")

    plt.savefig(str(save_path + mat + 'wheel_choice_psth.png'), format='png')
    plt.show()

def plot_paw_right_feedbackpsth(mat, design_matrix, trial_df, save_path):

    right_speeds = get_speed(design_matrix, design_matrix['Bin'], 'right', 60, split=True, feature='r_paw')

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=True, figsize=[5, 5])

    for c, choice in enumerate([-1.0, 1.0]):

        use_data = trial_df.loc[trial_df['feedbackType']==choice]
        left_paw, l_tscale = event_locked_signal(right_speeds[0], right_speeds[1], use_data['stimOn_times'])

        l = np.nanmean(np.abs(left_paw), axis=0)
        s_l = np.nanstd(np.abs(left_paw), axis=0) / np.sqrt(np.sum(~np.isnan(l), axis=0))
        ax.plot(l_tscale, l, label=choice)
        ax.fill_between(l_tscale, l - s_l, l + s_l, alpha=0.3)
        ax.legend()

        ax.axvline(0, linestyle='dashed', color='k')
        ax.set_title('Right paw')
        ax.set_xlabel("Time from event (s)")
        ax.set_xlim([-0.25, 0.75])
        ax.set_ylabel("Velocity")

    plt.savefig(str(save_path + mat + 'paw_right_psth.png'), format='png')
    plt.show()


"""
DESIGN_MATRIX LOADING
"""

def idxs_from_files(design_matrices):
    
    idxs = []
    mouse_names = []
    for m, mat in enumerate(design_matrices):
        mouse_name = design_matrices[m][51:]
        eid = design_matrices[m][14:50]
        idx = str(eid + '_' + mouse_name)

        if len(idxs) == 0:
            idxs = idx
            mouse_names = mouse_name
        else:
            idxs = np.hstack((idxs, idx))
            mouse_names = np.hstack((mouse_names, mouse_name))
            
    return idxs, mouse_names

"""
SCRIPT 3.1: PAW WAVELET DECOMPOSITION
"""

# This function uses get_XYs, not smoothing, is closer to brainbox function: https://github.com/int-brain-lab/ibllib/blob/78e82df8a51de0be880ee4076d2bb093bbc1d2c1/brainbox/behavior/dlc.py#L63
def get_speed(poses, times, camera, sampling_rate, split, feature):
    """
    FIXME Document and add unit test!

    :param dlc: dlc pqt table
    :param dlc_t: dlc time points
    :param camera: camera type e.g 'left', 'right', 'body'
    :param feature: dlc feature to compute speed over
    :return:
    """

    RESOLUTION = {'left': 2,
                  'right': 1,
                  'body': 1}
    sampling_rate = 60

    speeds = {}
    times = np.array(times)
    x = poses[f'{feature}_x'] / RESOLUTION[camera]
    y = poses[f'{feature}_y'] / RESOLUTION[camera]

    dt = np.diff(times)
    tv = times[:-1] + dt / 2

    # Calculate velocity for x and y separately if split is true
    if split == True:
        s_x = np.diff(x) * sampling_rate
        s_y = np.diff(y) * sampling_rate
        speeds = [times, s_x, s_y]
        # interpolate over original time scale
        if tv.size > 1:
            ifcn_x = interpolate.interp1d(tv, s_x, fill_value="extrapolate")
            ifcn_y = interpolate.interp1d(tv, s_y, fill_value="extrapolate")
            speeds = [times, ifcn_x(times), ifcn_y(times)]
    else:
        # Speed vector is given by the Pitagorean theorem
        s = ((np.diff(x)**2 + np.diff(y)**2)**.5) * sampling_rate
        speeds = [times, s]
        # interpolate over original time scale
        if tv.size > 1:
            ifcn = interpolate.interp1d(tv, s, fill_value="extrapolate")
            speeds = [times, ifcn(times)]

    return speeds  

# WAVELET DECOMPOSITION
def morlet_conj_ft(omega_vals, omega0):
    """
    Computes the conjugate Fourier transform of the Morlet wavelet.

    Parameters:
    - w: Angular frequency values (array or scalar)
    - omega0: Dimensionless Morlet wavelet parameter

    Returns:
    - out: Conjugate Fourier transform of the Morlet wavelet
    """

    return np.pi**(-1/4) * np.exp(-0.5 * (omega_vals - omega0)**2)


def fast_wavelet_morlet_convolution_parallel(x, f, omega0, dt):
    """
    Fast Morlet wavelet transform using parallel computation.

    Args:
        x (array): 1D array of projection values to transform.
        f (array): Center frequencies of the wavelet frequency channels (Hz).
        omega0 (float): Dimensionless Morlet wavelet parameter.
        dt (float): Sampling time (seconds).

    Returns:
        amp (array): Wavelet amplitudes.
        W (array): Wavelet coefficients (complex-valued, optional).
    """
    N = len(x)
    L = len(f)
    amp = np.zeros((L, N))
    Q = np.zeros((L, N))

    # Ensure N is even
    if N % 2 == 1:
        x = np.append(x, 0)
        N += 1
        test = True
    else:
        test = False

    # Add zero padding to x
    # Zero padding serves to compensate for the fact that the kernel does not have the same size as 
    # 
    x = np.concatenate((np.zeros(N // 2), x, np.zeros(N // 2)))
    M = N
    N = len(x)

    # Compute scales
    scales = (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * f)
    # angular frequencies to compute FT for (depends on sampling frequency); is as long as N 
    omega_vals = 2 * np.pi * np.arange(-N // 2, N // 2) / (N * dt)  

    # Fourier transform of x; shift folds it around zero so that it is more interpretable (frequencies at the right of nyquist become negative)
    x_hat = fftshift(fft(x))

    # Index for truncation to recover the actual x without padding
    if test:
        idx = np.arange(M // 2, M // 2 + M - 1)
    else:
        idx = np.arange(M // 2, M // 2 + M)

    # Function for parallel processing
    def process_frequency(i):
        # Take the Morlet conjugate of the Fourier transform
        m = morlet_conj_ft(-omega_vals * scales[i], omega0)
        # Convolution on the Fourier domain (as opposed to time domain in DWT)
        conv = m * x_hat
        # Inverse Fourier transform (normalized?)
        # q are the wavelet coefficients; normalized to ensure the energy of the wavelet is preserved across different scales
        q = ifft(conv) * np.sqrt(scales[i])
        # Recover q without padding
        q = q[idx]
        amp_row = np.abs(q) * np.pi**-0.25 * np.exp(0.25 * (omega0 - np.sqrt(omega0**2 + 2))**2) / np.sqrt(2 * scales[i])
        return amp_row, q

    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_frequency)(i) for i in range(L))

    for i, (amp_row, q) in enumerate(results):
        amp[i, :] = amp_row
        Q[i, :] = q

    return amp, Q, x_hat


"""
SCRIPT 3.2: PAW WAVELET SUBSAPLING
"""

def plot_kde(X_embedded, kernel):
    xmin = -150
    xmax = 150
    ymin=-150
    ymax=150
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
            extent=[xmin, xmax, ymin, ymax])
    ax.plot(X_embedded[:, 0], X_embedded[:, 1], 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.show()
    
    
# def GMM_neg_log_likelihood(embedding, components):
    
#     LL = np.zeros(len(components)) * np.nan
    
#     for i, k in enumerate(components):
#         # g = mixture.GaussianMixture(n_components=k)
#         # generate random sample, two components
#         np.random.seed(0)

#         # concatenate the two datasets into the final training set
#         cutoff = int(np.shape(embedding)[0]*0.8)
#         train_indices = np.random.choice(embedding.shape[0], cutoff, replace=False)
#         X_train = np.vstack([embedding[train_indices, 0], embedding[train_indices, 1]]).T

#         # fit a Gaussian Mixture Model with two components
#         clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
#         clf.fit(X_train)

#         all_indices = np.arange(0, embedding.shape[0], 1)
#         test_indices = [idx for idx in all_indices if idx not in train_indices]
#         X_test = np.vstack([embedding[test_indices, 0], embedding[test_indices, 1]])
#         LL[i] = -clf.score(X_test.T)
        
#     return LL


"""
SCRIPT 3.2: HMM fits
"""

"""" AR-HMM FITTING FUNCTIONS """

def cross_validate_armodel(model, key, train_emissions, train_inputs, method_to_use, num_train_batches, method, num_iters=100):
    # Initialize the parameters using K-Means on the full training set
    #init_params, props = model.initialize(key=key, method="kmeans", emissions=train_emissions)
    init_params, props = model.initialize(key=key, method=method_to_use, emissions=train_emissions)

    # Split the training data and the training inputs matrix_all[ses][0]into folds.
    # Note: this is memory inefficient but it highlights the use of vmap.
    folds = jnp.stack([
        jnp.concatenate([train_emissions[:i], train_emissions[i+1:]])
        for i in range(num_train_batches)])
    
    inpt_folds = jnp.stack([
        jnp.concatenate([train_inputs[:i], train_inputs[i+1:]])
        for i in range(num_train_batches)])
    
    # Baseline model has the same number of states but random initialization
    def _fit_fold_baseline(y_val, inpts):
        return model.marginal_log_prob(init_params, y_val, inpts) # np.shape(y_val)[1]

    baseline_val_lls = vmap(_fit_fold_baseline)(train_emissions, train_inputs)
    
    # Then actually fit the model to data
    if method == 'em':
        def _fit_fold(y_train, y_val, inpt_folds, inpts):
            fit_params, train_lps = model.fit_em(init_params, props, y_train, inpt_folds, 
                                                num_iters=num_iters, verbose=False)
            return model.marginal_log_prob(fit_params, y_val, inpts) , fit_params 
    elif method == 'sgd':
        def _fit_fold(y_train, y_val, inpt_folds, inpts):
            fit_params, train_lps = model.fit_sgd(init_params, props, y_train, inpt_folds, 
                                                num_epochs=num_iters)
            return model.marginal_log_prob(fit_params, y_val, inpts) , fit_params  
    
    val_lls, fit_params = vmap(_fit_fold)(folds, train_emissions, inpt_folds, train_inputs)
    
    return val_lls, fit_params, init_params, baseline_val_lls


def compute_inputs(emissions, num_lags, emission_dim):
    """Helper function to compute the matrix of lagged emissions.

    Args:
        emissions: $(T \times N)$ array of emissions
        prev_emissions: $(L \times N)$ array of previous emissions. Defaults to zeros.

    Returns:
        $(T \times N \cdot L)$ array of lagged emissions. These are the inputs to the fitting functions.
    """
    prev_emissions = jnp.zeros((num_lags, emission_dim))

    padded_emissions = jnp.vstack((prev_emissions, emissions))
    num_timesteps = len(emissions)
    return jnp.column_stack([padded_emissions[lag:lag+num_timesteps]
                                for lag in reversed(range(num_lags))])
    
"""" POISSON-HMM FITTING FUNCTIONS """

def cross_validate_poismodel(model, key, train_emissions, num_train_batches, fit_method, num_iters=100):
    # Initialize the parameters using K-Means on the full training set
    init_params, props = model.initialize(key=key)

    # Split the training data into folds.
    # Note: this is memory inefficient but it highlights the use of vmap.
    folds = jnp.stack([
        jnp.concatenate([train_emissions[:i], train_emissions[i+1:]])
        for i in range(num_train_batches)])

    # Baseline model has the same number of states but random initialization
    def _fit_fold_baseline(y_train, y_val):
        return model.marginal_log_prob(init_params, y_val) # np.shape(y_val)[1]
    
    # Then actually fit the model to data
    if fit_method == 'em':
        def _fit_fold(y_train, y_val):
            fit_params, train_lps = model.fit_em(init_params, props, y_train, 
                                                num_iters=num_iters, verbose=False)
            return model.marginal_log_prob(fit_params, y_val) , fit_params  
    elif fit_method == 'sgd':
        def _fit_fold(y_train, y_val):
            fit_params, train_lps = model.fit_sgd(init_params, props, y_train, 
                                                num_epochs=num_iters)
            return model.marginal_log_prob(fit_params, y_val) , fit_params  
    
    val_lls, fit_params = vmap(_fit_fold)(folds, train_emissions)
    
    baseline_val_lls = vmap(_fit_fold_baseline)(folds, train_emissions)

    return val_lls, fit_params, init_params, baseline_val_lls

""" Model comparison """

def conditional_nanmean(arr, axis):
    arr = np.asanyarray(arr)
    
    # Count NaNs along axis
    nan_count = np.isnan(arr).sum(axis=axis)
    total_count = arr.shape[axis]
    
    # Boolean mask: True where we should return NaN
    too_many_nans = nan_count >= (total_count / 2)
    
    # Compute nanmean
    mean_vals = np.nanmean(arr, axis=axis)
    
    # Where too_many_nans, set result to NaN
    result = np.where(too_many_nans, np.nan, mean_vals)
    
    return result


def get_bits_LL_kappa(all_lls, all_baseline_lls, design_matrix, num_train_batches, params):
    
    all_LL = np.ones((len(params), num_train_batches)) * np.nan
    all_baseline_LL = np.ones((len(params), num_train_batches)) * np.nan
    best_fold = np.ones((len(params))) * np.nan
    
    # Reshape
    for k_index, k in enumerate(params):
        all_LL[k_index, :] = all_lls[k]
        all_baseline_LL[k_index, :] = all_baseline_lls[k]
        
    # Get size of folds
    num_timesteps = np.shape(design_matrix)[0]
    shortened_array = np.array(design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    fold_len =  len(shortened_array)/num_train_batches
    
    bits_LL = (np.array(all_LL) - np.array(all_baseline_LL)) / fold_len * np.log(2)

    for k_index, k in enumerate(params):
        # Best fold for each kappa based on bits_LL
        if np.sum(np.isnan(bits_LL[k_index])) < len(bits_LL[k_index])/2:  # nan if half of the folds are nan
            best_fold[k_index] = np.where(bits_LL[k_index]==np.nanmax(bits_LL[k_index]))[0][0]
        else:
            best_fold[k_index] = np.nan
                
    return bits_LL, all_LL, best_fold


def get_bits_LL_kappa_Lag(all_lls, all_baseline_lls, design_matrix, num_train_batches, kappas, Lags):
    
    all_LL = np.ones((len(kappas), len(Lags), num_train_batches)) * np.nan
    all_baseline_LL = np.ones((len(kappas), len(Lags), num_train_batches)) * np.nan
    best_fold = np.ones((len(kappas), len(Lags))) * np.nan
    
    # Reshape
    for k_index, k in enumerate(kappas):
        for lag_index, l in enumerate(Lags):
            all_LL[k_index, lag_index :] = all_lls[l][k]
            all_baseline_LL[k_index, lag_index, :] = all_baseline_lls[l][k]
        
    # Get size of folds
    num_timesteps = np.shape(design_matrix)[0]
    shortened_array = np.array(design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    fold_len =  len(shortened_array)/num_train_batches
    
    bits_LL = (np.array(all_LL) - np.array(all_baseline_LL)) / fold_len * np.log(2)
    
    for k_index, k in enumerate(kappas):
        for lag_index, l in enumerate(Lags):
            # Best fold for each kappa
            if np.sum(np.isnan(bits_LL[k_index, lag_index])) < len(bits_LL[k_index, lag_index])/2:  # nan if half of the folds are nan
                best_fold[k_index, lag_index] = np.where(bits_LL[k_index, lag_index]==np.nanmax(bits_LL[k_index, lag_index]))[0][0]
            else:
                best_fold[k_index, lag_index] = np.nan
    
    return bits_LL, all_LL, best_fold


def get_bits_LL(all_lls, all_baseline_lls, design_matrix, num_train_batches, params, param_num):
    _, Lags, kappas  = params
    if param_num == 1:
        bits_LL, all_LL, best_fold = get_bits_LL_kappa(all_lls, all_baseline_lls, design_matrix, num_train_batches, kappas)
    elif param_num == 2:
        bits_LL, all_LL, best_fold = get_bits_LL_kappa_Lag(all_lls, all_baseline_lls, design_matrix, num_train_batches, kappas, Lags)
    
    return bits_LL, all_LL, best_fold


def find_1_best_param(bits_LL, param):
    # Find param which minimizes complexity 
    # while leading to LL not significantly different from best
    
    mean_bits_LL = conditional_nanmean(bits_LL, axis=1)
    index_best_param = np.where(mean_bits_LL==np.nanmax(mean_bits_LL))[0][0]
    max_param = param[index_best_param]
    
    ci_95 = np.nanstd(bits_LL, axis=1) / (np.sqrt(np.shape(bits_LL)[1]))*1.96
    upper_lims = mean_bits_LL + ci_95
    lower_lims = mean_bits_LL - ci_95

    # If best kappa is in left boundary
    if max_param == np.min(param):
        best_param = max_param
    
    # If best kappa is in right boundary
    elif max_param == np.max(param):
        # Check if significantly higher than lower kappa
        max_param_lim = lower_lims[index_best_param]
        pre_param_lim = upper_lims[index_best_param-1]
        if pre_param_lim < max_param_lim:
            print('Best parameter at the boundary')
        else:
            not_significantly_different = np.where(upper_lims >= lower_lims[index_best_param])
            minimize_complexity = np.min(not_significantly_different[0])
            best_param = param[minimize_complexity]
    else:
        not_significantly_different = np.where(upper_lims >= lower_lims[index_best_param])
        minimize_complexity = np.min(not_significantly_different[0])
        best_param = param[minimize_complexity]
        
    return best_param, mean_bits_LL


def find_2_best_param(bits_LL, kappas, Lags):
    # Find param which minimizes complexity 
    # while leading to LL not significantly different from best
    parameters = kappas, Lags
    mean_bits_LL = conditional_nanmean(bits_LL, axis=2)
    index_best_kappa = np.where(mean_bits_LL==np.nanmax(mean_bits_LL))[0][0]
    index_best_lag = np.where(mean_bits_LL==np.nanmax(mean_bits_LL))[1][0]
    
    best_params = np.zeros((2))*np.nan
    for i, index_best_param in enumerate([index_best_kappa, index_best_lag]):
        params = parameters[i]
        max_param = params[index_best_param]
        if i == 0:
            use_bits_LL = conditional_nanmean(bits_LL, axis=1) # average over lags
            use_mean_bits_LL = conditional_nanmean(use_bits_LL, axis=1)
        elif i == 1:
            use_bits_LL = conditional_nanmean(bits_LL, axis=0)  # average over kappa
            use_mean_bits_LL = conditional_nanmean(use_bits_LL, axis=1)

        ci_95 = np.nanstd(use_bits_LL, axis=1) / (np.sqrt(np.shape(use_bits_LL)[1]))*1.96
        upper_lims = use_mean_bits_LL + ci_95
        lower_lims = use_mean_bits_LL - ci_95

        # If best kappa is in left boundary
        if max_param == np.min(params):
            best_param = max_param
        
        # If best kappa is in right boundary
        elif max_param == np.max(params):
            # Check if significantly higher than lower kappa
            max_param_lim = lower_lims[index_best_param]
            pre_param_lim = upper_lims[index_best_param-1]
            if pre_param_lim < max_param_lim:
                print('Best parameter at the boundary')
            else:
                not_significantly_different = np.where(upper_lims >= lower_lims[index_best_param])
                minimize_complexity = np.min(not_significantly_different[0])
                best_param = params[minimize_complexity]
        else:
            not_significantly_different = np.where(upper_lims >= lower_lims[index_best_param])
            minimize_complexity = np.min(not_significantly_different[0])
            best_param = params[minimize_complexity]
        
        best_params[i] = int(best_param)
        
        
    return int(best_params[0]), int(best_params[1]), mean_bits_LL


def find_best_param(bits_LL, params, param_num):
    _, Lags, kappas  = params
    if param_num == 1:
        best_kappa, mean_bits_LL = find_1_best_param(bits_LL, kappas)
        best_lag = []
    elif param_num == 2:
        best_kappa, best_lag, mean_bits_LL = find_2_best_param(bits_LL, kappas, Lags)
        
    return best_kappa, best_lag, mean_bits_LL
    
    
def plot_grid_search(best_kappa, best_lag, mean_bits_LL, kappas, Lags, mouse_name, var_interest):
    
    best_kappa_idx = np.where(np.array(kappas)==best_kappa)[0][0]
    best_lag_idx = np.where(np.array(Lags)==best_lag)[0][0]

    # Coordinates of the square to highlight (row, column)
    highlight_square = (best_lag_idx, best_kappa_idx)
    # Size of the square (assuming square is 1x1)
    square_size = 1
    # Create the plot
    fig, ax = plt.subplots()
    # Display the matrix
    cax = ax.imshow(mean_bits_LL, cmap='viridis')
    # Add the rectangle to highlight the square
    rect = Rectangle((highlight_square[1] - 0.5, highlight_square[0] - 0.5), 
                    square_size, square_size, 
                    linewidth=2, edgecolor='r', facecolor='none')
    # Add the rectangle patch to the plot
    ax.add_patch(rect)
    # Add color bar
    cbar = plt.colorbar(cax)
    cbar.set_label('Delta LL')
    ax.set_xticks(np.arange(0, len(kappas), 1), kappas)
    ax.set_yticks(np.arange(0, len(Lags), 1), Lags)
    plt.xlabel('Kappa')
    plt.ylabel('Lag')
    plt.title(mouse_name + ' ' + var_interest)
    # Display the plot
    plt.show()
    
    
def get_metadata(one, sessions):
    metadata = pd.DataFrame(columns=['session', 'lab'], index=range(len(sessions)))
    for s, session in enumerate(sessions):
        session_details = one.get_details(session, full=False)
        metadata['session'][s] = session
        metadata['lab'][s] = session_details['lab']
    return metadata


"""
SCRIPT 5: SYLLABLE GENERATION
"""

def prepro(trials):

    """ Performance """
    # Some preprocessing
    trials['contrastLeft'] = trials['contrastLeft'].fillna(0)
    trials['contrastRight'] = trials['contrastRight'].fillna(0)
    trials['signed_contrast'] = - trials['contrastLeft'] + trials['contrastRight']
    trials['contrast'] = trials['contrastLeft'] + trials['contrastRight']
    trials['correct_easy'] = trials['feedbackType']
    trials.loc[trials['correct_easy']==-1, 'correct_easy'] = 0
    trials['correct'] = trials['feedbackType']
    trials.loc[trials['contrast']<.5, 'correct_easy'] = np.nan
    trials.loc[trials['correct']==-1, 'correct'] = 0

    """ Response/ reaction times """
    trials['response'] = trials['response_times'] - trials['goCue_times']
    trials['reaction'] = trials['firstMovement_times'] - trials['goCue_times']
    """ Quiescence elongation """
    trials['elongation'] = trials['goCue_times'] - trials['quiescencePeriod'] - trials['intervals_0']
    """ Win stay lose shift """
    trials['prev_choice'] = trials['choice'] * np.nan
    trials['prev_choice'][1:] = trials['choice'][:-1]
    trials['prev_feedback'] = trials['feedbackType'] * np.nan
    trials['prev_feedback'][1:] = trials['feedbackType'][:-1]
    trials['wsls'] = trials['choice'] * np.nan
    trials.loc[(trials['prev_feedback']==1.) & (trials['choice']==trials['prev_choice']), 'wsls'] = 'wst'
    trials.loc[(trials['prev_feedback']==1.) & (trials['choice']!=trials['prev_choice']), 'wsls'] = 'wsh'
    trials.loc[(trials['prev_feedback']==-1.) & (trials['choice']!=trials['prev_choice']), 'wsls'] = 'lsh'
    trials.loc[(trials['prev_feedback']==-1.) & (trials['choice']==trials['prev_choice']), 'wsls'] = 'lst'
    #TODO : trials['days_to_trained'] = trials['training_time']

    return trials


# def state_identifiability_old(combined_states, design_matrix_heading, use_sets):
    
#     unique_states = np.unique(combined_states)
#     new_states = unique_states.copy()

#     # Create new mapping depending on empirical data for each state
#     for v, var in enumerate(use_sets):
#         zeros = [s[v] == '0' if s != 'nan' else False for s in combined_states]
#         ones = [s[v] == '1' if s != 'nan' else False for s in combined_states]
        
#         # For an empty variable, do not make changes (wavelet)
#         if len(var) == 0:
#             var_0 = np.nan
#             var_1 = np.nan
#         elif var == ['avg_wheel_vel']:
#             var_0 = np.array(np.abs(design_matrix_heading[var]))[zeros]
#             var_1 = np.array(np.abs(design_matrix_heading[var]))[ones]
#         elif var == ['left_X', 'left_Y', 'right_X', 'right_Y']:
#             var_0 = np.array(np.abs(np.diff(design_matrix_heading[var], axis=0)))[zeros[1:]]
#             var_1 = np.array(np.abs(np.diff(design_matrix_heading[var], axis=0)))[ones[1:]]
#         elif var == ['nose_x', 'nose_Y']:
#             print('Not implemented yet')
#         else:
#             var_0 = np.array(design_matrix_heading[var])[zeros]
#             var_1 = np.array(design_matrix_heading[var])[ones]
        
#         if np.nanmean(var_0)> np.nanmean(var_1):
#             var_state_0 = [s[v] == '0' if s != 'nan' else False for s in unique_states]
#             new_states[var_state_0] = np.array([s[:v] + '1' + s[v+1:] for s in new_states[var_state_0]])
#             var_state_1 = [s[v] == '1' if s != 'nan' else False for s in unique_states]
#             new_states[var_state_1] = np.array([s[:v] + '0' + s[v+1:] for s in new_states[var_state_1]])

#     identifiable_mapping = {unique: key for unique, key in zip(unique_states, new_states)}

#     # Use np.vectorize to apply the mapping
#     replace_func = np.vectorize(identifiable_mapping.get)
#     identifiable_states = replace_func(combined_states)
    
#     return identifiable_states


def state_identifiability(session_states, use_sets):
    # Create new mapping depending on empirical data for each state
    for v, var in enumerate(use_sets):
        var_states = var+'_states'
        
        # For an empty variable, do not make changes (wavelet)
        if len(var) == 0:
            var_0 = np.nan
            var_1 = np.nan
        elif var == ['avg_wheel_vel']:
            var_0 = np.nanmean(np.abs(session_states.loc[session_states[var_states]==0, var]))
            var_1 = np.nanmean(np.abs(session_states.loc[session_states[var_states]==1, var]))
        elif var == ['left_X', 'left_Y', 'right_X', 'right_Y']:
            var_0 = np.array(np.abs(np.diff(session_states.loc[session_states[var_states]==0, var])))
            var_1 = np.array(np.abs(np.diff(session_states.loc[session_states[var_states]==0, var])))
        elif var == ['nose_x', 'nose_Y']:
            print('Not implemented yet')
        else:
            var_0 = session_states.loc[session_states[var_states]==0, var]
            var_1 = session_states.loc[session_states[var_states]==1, var]
        
        if np.nanmean(var_0)> np.nanmean(var_1):
            session_states[var_states] = session_states[var_states] * -1 + 1
    return session_states


def align_bin_design_matrix (init, end, event_type_list, session_trials, design_matrix, most_likely_states, multiplier):
    
    for e, this_event in enumerate(event_type_list):
        
        # Initialize variables
        # Before there was a function for keeping validation set apart, now deprecated
        reduced_design_matrix = design_matrix.copy()
        reduced_design_matrix['most_likely_states'] = most_likely_states
        reduced_design_matrix['new_bin'] = reduced_design_matrix['Bin'] * np.nan
        reduced_design_matrix['correct'] = reduced_design_matrix['Bin'] * np.nan
        reduced_design_matrix['choice'] = reduced_design_matrix['Bin'] * np.nan
        reduced_design_matrix['contrast'] = reduced_design_matrix['Bin'] * np.nan        
        reduced_design_matrix['block'] = reduced_design_matrix['Bin'] * np.nan        

        feedback = session_trials['feedbackType']
        choice = session_trials['choice']
        contrast = np.abs(prepro(session_trials)['signed_contrast'])
        block = session_trials['probabilityLeft']
        reaction = prepro(session_trials)['reaction']
        response = prepro(session_trials)['response']
        elongation = prepro(session_trials)['elongation']
        wsls = prepro(session_trials)['wsls']
        trial_id = session_trials['index'] 

        events = session_trials[this_event]
                
        for t, trial in enumerate(events[:-1]):
            event = events[t]
            trial_start = session_trials['intervals_0'][t]
            trial_end = session_trials['intervals_0'][t+1]
            
            # Check feedback
            if feedback[t] ==1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'correct'] = 1
            elif feedback[t] == -1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'correct'] = 0
            # Check choice
            if choice[t] ==1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'choice'] = 'right'
            elif choice[t] == -1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'choice'] = 'left'
            
            # Check reaction
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'reaction'] = reaction[t]
            # Check response
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'response'] = response[t]
            # Check elongation
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']>trial_start*multiplier), 
                                            'elongation'] = elongation[t]

            # Check contrast
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'contrast'] = contrast[t]

            # Check block
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'block'] = block[t]
            
            # Check wsls
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'wsls'] = wsls[t]

            # Check trial id
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'trial_id'] = trial_id[t]
            
            # Add reliable timestamp to identify trial
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            this_event] = event
            
            # Rename bins so that they are aligned on stimulus onset
            if event > 0:
                event_window = reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) &
                                                         (reduced_design_matrix['Bin']> event*multiplier + init)]
                onset_bin = reduced_design_matrix.loc[reduced_design_matrix['Bin']>= event*multiplier, 'Bin']
                if (len(event_window)>0) & len(onset_bin)>0:
                    bin = list(onset_bin)[0]
                    reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) &
                                            (reduced_design_matrix['Bin']> event*multiplier + init), 
                                            'new_bin'] = reduced_design_matrix.loc[(reduced_design_matrix['Bin']< event*multiplier + end) & 
                                            (reduced_design_matrix['Bin']>= event*multiplier + init), 'Bin'] - bin
                else:
                    reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) & 
                                              (reduced_design_matrix['Bin']> event*multiplier + init), 'new_bin'] = np.nan
            else:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) & 
                                          (reduced_design_matrix['Bin']> event*multiplier + init), 'new_bin'] = np.nan
                
    return reduced_design_matrix


def states_per_trial_phase(reduced_design_matrix, session_trials, multiplier):
    
    # Split session into trial phases and gather most likely states of those trial phases
    # use_data = reduced_design_matrix.dropna()
    use_data = reduced_design_matrix.copy()
    use_data['label'] = use_data['Bin'] * np.nan
    trial_num = len(session_trials)

    # Pre-quiescence 
    pre_qui_init = session_trials['intervals_0']
    pre_qui_end = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']

    # Quiescence
    qui_init = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']
    qui_end = session_trials['goCueTrigger_times']
    
    # ITI
    iti_init = session_trials['feedback_times']
    iti_end_correct = session_trials['intervals_1']
    iti_end_incorrect = session_trials['intervals_1'] - 1
    
    # Reaction time 
    rt_init = session_trials['goCueTrigger_times']
    rt_end = session_trials['firstMovement_times']

    # Movement time 
    move_init = session_trials['firstMovement_times']
    move_end = session_trials['feedback_times']
    

    for t in range(trial_num):
          
        # Pre-quiescence
        use_data.loc[(use_data['Bin'] <= pre_qui_end[t]*multiplier) & 
                     (use_data['Bin'] > pre_qui_init[t]*multiplier), 'label'] = 'Pre-quiescence'

        # Quiescence
        use_data.loc[(use_data['Bin'] <= qui_end[t]*multiplier) &
                     (use_data['Bin'] > qui_init[t]*multiplier), 'label'] = 'Quiescence'
        
        # ITI
        if session_trials['feedbackType'][t] == -1.:
            use_data.loc[(use_data['Bin'] <= iti_end_incorrect[t]*multiplier) & 
                            (use_data['Bin'] > iti_init[t]*multiplier), 'label'] = 'ITI'
        elif session_trials['feedbackType'][t] == 1.:
            use_data.loc[(use_data['Bin'] <= iti_end_correct[t]*multiplier) & 
                            (use_data['Bin'] > iti_init[t]*multiplier), 'label'] = 'ITI'
        # Move
        if session_trials['choice'][t] == -1:
            use_data.loc[(use_data['Bin'] <= move_end[t]*multiplier) & 
                         (use_data['Bin'] > move_init[t]*multiplier), 'label'] = 'Left choice'
        elif session_trials['choice'][t] == 1.:
            use_data.loc[(use_data['Bin'] <= move_end[t]*multiplier) & 
                         (use_data['Bin'] > move_init[t]*multiplier), 'label'] = 'Right choice'
            
        # React        
        if prepro(session_trials)['signed_contrast'][t] < 0:
            use_data.loc[(use_data['Bin'] <= rt_end[t]*multiplier) & 
                         (use_data['Bin'] > rt_init[t]*multiplier), 'label'] = 'Stimulus left'
        elif prepro(session_trials)['signed_contrast'][t] > 0:
            use_data.loc[(use_data['Bin'] <= rt_end[t]*multiplier) & 
                         (use_data['Bin'] > rt_init[t]*multiplier), 'label'] = 'Stimulus right'
    return use_data


def broader_label(df):
    
    df['broader_label'] = df['label']
    # df.loc[df['broader_label']=='Stimulus right', 'broader_label'] = 'Stimulus'
    # df.loc[df['broader_label']=='Stimulus left', 'broader_label'] = 'Stimulus'
    df.loc[df['broader_label']=='Stimulus right', 'broader_label'] = 'Choice'
    df.loc[df['broader_label']=='Stimulus left', 'broader_label'] = 'Choice'
    df.loc[df['broader_label']=='Quiescence', 'broader_label'] = 'Quiescence'
    df.loc[df['broader_label']=='Pre-quiescence', 'broader_label'] = 'Pre-quiescence'
    df.loc[df['broader_label']=='Left choice', 'broader_label'] = 'Choice'
    df.loc[df['broader_label']=='Right choice', 'broader_label'] = 'Choice'
    df.loc[df['broader_label']=='Correct feedback', 'broader_label'] = 'ITI'
    df.loc[df['broader_label']=='Incorrect feedback', 'broader_label'] = 'ITI'
    df.loc[df['broader_label']=='ITI_correct', 'broader_label'] = 'ITI'
    df.loc[df['broader_label']=='ITI_incorrect', 'broader_label'] = 'ITI'
    
    return df


def define_trial_types(states_trial_type, trial_type_agg):
    
    """ Define trial types"""
    states_trial_type['correct_str'] = states_trial_type['correct']
    states_trial_type.loc[states_trial_type['correct_str']==1., 'correct_str'] = 'correct'
    states_trial_type.loc[states_trial_type['correct_str']==0., 'correct_str'] = 'incorrect'
    states_trial_type['contrast_str'] = states_trial_type['contrast'].astype(str)
    states_trial_type['block_str'] = states_trial_type['block'].astype(str)
    states_trial_type['perseverence'] = states_trial_type['wsls'].copy()
    states_trial_type.loc[states_trial_type['wsls'].isin(['wst', 'lst']), 'perseverence']  = 'stay'
    states_trial_type.loc[states_trial_type['wsls'].isin(['wsh', 'lsh']), 'perseverence']  = 'shift'
    states_trial_type['trial_type'] = states_trial_type[trial_type_agg].fillna('unknown').agg(' '.join, axis=1)
    # states_trial_type['trial_type'] = states_trial_type[trial_type_agg].agg(' '.join, axis=1)
    states_trial_type['trial_str'] = states_trial_type['trial_id'].astype(str)
    states_trial_type['sample'] = states_trial_type[['session', 'trial_str']].agg(' '.join, axis=1)
    if 'ballistic' in states_trial_type.columns:
        states_trial_type.loc[states_trial_type['ballistic']==True, 'ballistic'] = 1
        states_trial_type.loc[states_trial_type['ballistic']==False, 'ballistic'] = 0
    return states_trial_type


def rescale_sequence(seq, target_length, estimator):
    """
    Rescales a categorical sequence to a fixed target length.
    
    - If `target_length` is smaller than the original length, it takes the mode of each bin.
    - If `target_length` is larger, it repeats values evenly.
    
    Parameters:
        seq (array-like): The original categorical sequence.
        target_length (int): The desired length of the output sequence.
    
    Returns:
        np.ndarray: The transformed sequence with the specified target length.
    """
    original_length = len(seq)

    if original_length == target_length:
        return np.array(seq)  # No change needed

    if target_length < original_length:
        # Compression: Split into bins and take mode of each bin
        bins = np.array_split(seq, target_length)
        # return np.array([mode(b)[0][0] for b in bins])  # Extract mode from result
        if estimator == 'mode':
            result = np.array([mode(b)[0] for b in bins])
        elif estimator == 'mean':
            result = np.array([np.mean(b) for b in bins])
        return result  

    else:
        # Stretching: Repeat values to fit new size
        stretched_indices = np.floor(np.linspace(0, original_length - 1, target_length)).astype(int)
        return np.array(seq)[stretched_indices]  # Map stretched indices to original values
    

def plot_binned_sequence(df_grouped, index, states_to_append, var_name, palette):
        title = df_grouped['broader_label'][index]
        fig, axs = plt.subplots(2, 1, sharex=False, sharey=True, figsize=(5, 2))
        axs[0].imshow(np.concatenate([df_grouped[var_name+'_sequence'][index], states_to_append])[None,:],  
                extent=(0, len(np.concatenate([df_grouped[var_name+'_sequence'][index], states_to_append])), 
                        0, 1),
                aspect="auto",
                cmap=palette,
                alpha=0.7) 
        axs[0].set_xlim([0, len(df_grouped[var_name+'_sequence'][index])])

        axs[1].imshow(np.concatenate([df_grouped[var_name+'_binned_sequence'][index], states_to_append])[None,:],  
                extent=(0, len(np.concatenate([df_grouped[var_name+'_binned_sequence'][index], states_to_append])), 
                        0, 1),
                aspect="auto",
                cmap=palette,
                alpha=0.7) 
        axs[1].set_xlim([0, len(df_grouped[var_name+'_binned_sequence'][index])])
        axs[0].set_title(title)
        plt.tight_layout()