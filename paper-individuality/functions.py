import pandas as pd
import os
import numpy as np
from one.api import ONE
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import brainbox.behavior.wheel as wh
from scipy.stats import zscore
import concurrent.futures
from brainbox.io.one import SessionLoader
import scipy.interpolate as interpolate
from joblib import Parallel, delayed
from scipy.fftpack import fft, ifft, fftshift
from sklearn.preprocessing import StandardScaler, Normalizer
import gc
import os
import uuid
from pathlib import Path
from one.alf.files import add_uuid_string
from one.remote import aws


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


# Function written by Julia 
def download_subjectTables(one, subject=None, trials=True, training=True,
                           target_path=None, tag=None, overwrite=False, check_updates=True):
    """
    Function to download the aggregated clusters information associated with the given data release tag from AWS.
    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to database.
    trials: bool
        Whether to download the subjectTrials.table.pqt, default is True
    training: bool
        Whether to donwnload the subjectTraining.table.pqt, defaults is True
    subject: str, uuid or None
        Nickname or UUID of the subject to download all trials from. If None, download all available trials tables
        (associated with 'tag' if one is given)
    target_path: str or pathlib.Path
        Directory to which files should be downloaded. If None, downloads to one.cache_dir/aggregates
    tag: str
        Data release tag to download _ibl_subjectTrials.table datasets from. Default is None.
    overwrite : bool
        If True, will re-download files even if file exists locally and file sizes match.
    check_updates : bool
        If True, will check if file sizes match and skip download if they do. If False, will just return the paths
        and not check if the data was updated on AWS.
    Returns
    -------
    trials_tables: list of pathlib.Path
        Paths to the downloaded subjectTrials files
    training_tables: list of pathlib.Path
        Paths to the downloaded subjectTraining files
    """

    if target_path is None:
        target_path = Path(one.cache_dir).joinpath('aggregates')
        target_path.mkdir(exist_ok=True)
    else:
        assert target_path.exists(), 'The target_path you passed does not exist.'

    # Get the datasets
    trials_ds = []
    training_ds = []
    if subject:
        try:
            subject_id = uuid.UUID(subject)
        except ValueError:
            subject_id = one.alyx.rest('subjects', 'list', nickname=subject)[0]['id']
        if trials:
            trials_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTrials.table.pqt',
                                           django=f'object_id,{subject_id}'))
        if training:
            training_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTraining.table.pqt',
                                             django=f'object_id,{subject_id}'))
    else:
        if tag:
            if trials:
                trials_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTrials.table.pqt', tag=tag))
            if training:
                training_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTraining.table.pqt', tag=tag))
        else:
            if trials:
                trials_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTrials.table.pqt'))
            if training:
                training_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTraining.table.pqt'))

    # Set up the bucket
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)

    all_out = []
    for ds_list in [trials_ds, training_ds]:
        out_paths = []
        for ds in ds_list:
            relative_path = add_uuid_string(ds['file_records'][0]['relative_path'], ds['url'][-36:])
            src_path = 'aggregates/' + str(relative_path)
            dst_path = target_path.joinpath(relative_path)
            if check_updates:
                out = aws.s3_download_file(src_path, dst_path, s3=s3, bucket_name=bucket_name, overwrite=overwrite)
            else:
                out = dst_path

            if out and out.exists():
                out_paths.append(out)
            else:
                print(f'Downloading of {src_path} table failed.')
        all_out.append(out_paths)

    return all_out[0], all_out[1]



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

# def get_lick_times(poses, combine=False, video_type='left'):
    
#     if combine:    
#         # combine licking events from left and right cam
#         lick_times = []
#         for video_type in ['right','left']:
#             camera_name = str(video_type+'Camera')
#             camera_licks = get_feature_event_times(poses.pose[camera_name], 
#                                                  poses.pose[camera_name]['times'], features)        
#             lick_times.append(camera_licks)
        
#         lick_times = np.array(sorted(np.concatenate(lick_times)))
        
#     else:
#         lick_times = get_feature_event_times(poses.pose[video_type], 
#                                              poses.pose[video_type]['times'], features)        

#     return lick_times


def resample_common_time(reference_time, timestamps, data, kind, fill_gaps=None):
    # Function inspired on wh.interpolate from here: https://github.com/int-brain-lab/ibllib/blob/master/brainbox/behavior/wheel.py#L28 
    yinterp = interpolate.interp1d(timestamps, data, kind=kind, fill_value='extrapolate')(reference_time)
    
    if fill_gaps:
        #  Find large gaps and forward fill @fixme This is inefficient
        gaps, = np.where(np.diff(timestamps) >= fill_gaps)

        for i in gaps:
            yinterp[(reference_time >= timestamps[i]) & (reference_time < timestamps[i + 1])] = data[i]
            
    return yinterp, reference_time


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


# This function uses get_XYs, not smoothing, is closer to brainbox function: https://github.com/int-brain-lab/ibllib/blob/78e82df8a51de0be880ee4076d2bb093bbc1d2c1/brainbox/behavior/dlc.py#L63
def keypoint_speed_one_camera(one, eid, ephys, video_type, body_part, split, lp):

    if ephys ==True:
        fs = {'right':150,'left':60}   
    else:
        fs = {'right':150,'left':30}

    # if it is the paw, take speed from right paw only, i.e. closer to cam  
    # for each video
    speeds = {}
    times, XYs = get_XYs(one, eid, video_type, likelihood_thresh=0.9, lp=lp)
    
    # Pupil requires averaging 4 keypoints
    x = XYs[body_part][:, 0]
    y = XYs[body_part][:, 1]
    # _, x, _, y = get_raw_and_smooth_position(one, eid, video_type, ephys, body_part, lp)

    if video_type == 'left': #make resolution same
        x = x/2
        y = y/2
        
    dt = np.diff(times)
    tv = times[:-1] + dt / 2
    
    # Calculate velocity for x and y separately if split is true
    if split == True:
        s_x = np.diff(x)*fs[video_type]
        s_y = np.diff(y)*fs[video_type]
        speeds[video_type] = [times, s_x, s_y]

        # interpolate over original time scale
        if tv.size > 1:
            ifcn_x = interpolate.interp1d(tv, s_x, fill_value="extrapolate")
            ifcn_y = interpolate.interp1d(tv, s_y, fill_value="extrapolate")

            speeds[video_type] = [times, ifcn_x(times), ifcn_y(times)]

    else:
        # Speed vector is given by the Pitagorean theorem
        s = ((np.diff(x)**2 + np.diff(y)**2)**.5)*fs[video_type]
        speeds[video_type] = [times,s]

        # interpolate over original time scale
        if tv.size > 1:
            ifcn = interpolate.interp1d(tv, s, fill_value="extrapolate")
            
            speeds[video_type] = [times,ifcn(times)]
        
    return speeds


## This function is the brainbox function, should use for SessionLoader dataformat
SAMPLING = {'left': 60,
            'right': 150,
            'body': 30}
RESOLUTION = {'left': 2,
              'right': 1,
              'body': 1}

def get_speed(dlc, dlc_t, camera, feature='paw_r'):
    """
    FIXME Document and add unit test!

    :param dlc: dlc pqt table
    :param dlc_t: dlc time points
    :param camera: camera type e.g 'left', 'right', 'body'
    :param feature: dlc feature to compute speed over
    :return:
    """
    x = dlc[f'{feature}_x'] / RESOLUTION[camera]
    y = dlc[f'{feature}_y'] / RESOLUTION[camera]

    # get speed in px/sec [half res]
    s = ((np.diff(x) ** 2 + np.diff(y) ** 2) ** .5) * SAMPLING[camera]

    dt = np.diff(dlc_t)
    tv = dlc_t[:-1] + dt / 2

    # interpolate over original time scale
    if tv.size > 1:
        ifcn = interpolate.interp1d(tv, s, fill_value="extrapolate")
        return ifcn(dlc_t)
    
    