
## All functions from brainbox (and potentially adapted by Ines)

import numpy as np
import pandas as pd
from one.api import ONE
from scipy.interpolate import interp1d
import math
import matplotlib.pyplot as plt

one = ONE() 


def get_dlc_XYs(eid, view, likelihood_thresh=0.9):

    try:
        times = one.load_dataset(eid, '_ibl_%sCamera.times.npy' % view)
        cam = one.load_dataset(eid, '_ibl_%sCamera.dlc.pqt' % view)
    except KeyError:
        print('not all dlc data available')
        return None, None
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array([x, y]).T
    return times, XYs


""" LICKs """

def get_dlc_XYs_lick(eid, video_type, query_type='remote'):

    #video_type = 'left'    
    # Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy',
    #                          query_type=query_type) 
    # cam = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.dlc.pqt', 
    #                        query_type=query_type)
    try:
        Times = one.load_dataset(eid, '_ibl_%sCamera.times.npy' % video_type)
        cam = one.load_dataset(eid, '_ibl_%sCamera.dlc.pqt' % video_type)
    except KeyError:
        print('not all dlc data available')
        return None, None
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])

    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x, y])    

    return Times, XYs  


def get_licks(XYs):

    '''
    define a frame as a lick frame if
    x or y for left or right tongue point
    change more than half the sdt of the diff
    '''  
    
    licks = []
    for point in ['tongue_end_l', 'tongue_end_r']:
        for c in XYs[point]:
           thr = np.nanstd(np.diff(c))/4
           licks.append(set(np.where(abs(np.diff(c))>thr)[0]))
    return sorted(list(set.union(*licks))) 


def get_lick_times(eid, combine=False, video_type='left'):
    
    if combine:    
        # combine licking events from left and right cam
        lick_times = []
        for video_type in ['right','left']:
            times, XYs = get_dlc_XYs_lick(eid, video_type, query_type='remote')
            r = get_licks(XYs)
            # cover case that there are less times than DLC points            
            idx = np.where(np.array(r)<len(times))[0][-1]            
            lick_times.append(times[r[:idx]])
        
        lick_times = sorted(np.concatenate(lick_times))
        
    else:
        times, XYs = get_dlc_XYs_lick(eid, video_type, query_type='remote')    
        r = get_licks(XYs)
        # cover case that there are less times than DLC points
        idx = np.where(np.array(r)<len(times))[0][-1]              
        lick_times = times[r[:idx]]

    return lick_times
        

# By Ines
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
        
        licks_df = licks_df.append(temp_df)
        
    return licks_df


""" PUPIL """
"""
def get_dlc_XYs_pupil(one, eid, view, likelihood_thresh=0.9):
    dataset_types = ['camera.dlc', 'camera.times']
    try:
        times = one.load_dataset(eid, '_ibl_%sCamera.times.npy' % view)
        cam = one.load_dataset(eid, '_ibl_%sCamera.dlc.pqt' % view)
    except KeyError:
        print('not all dlc data available')
        return None, None
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array([x, y]).T
    return times, XYs
"""
# Brainbox

def motion_energy_PSTH(eid):

    '''
    ME PSTH
    canonical session
    eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
    '''

    rt = 2  # duration of window
    st = -0.5  # lag of window wrt to stype 
    stype = 'stimOn_times'

    ME = {}
    one = ONE()
    trials = one.load_object(eid,'trials')
    ts = trials.intervals[0][0]
    te = trials.intervals[-1][1]

    try:
        for video_type in ['left','right','body']:
            t,m = get_ME(eid, video_type)       
            m = zscore(m,nan_policy='omit') 

            sta, end = find_nearest(t,ts), find_nearest(t,te) 
            t = t[sta:end]
            m = m[sta:end]

            ME[video_type] = [t,m]

        # align to body cam
        for video_type in ['left','right']:

            # align time series camera/neural
            interpolater = interp1d(
                ME[video_type ][0],
                np.arange(len(ME[video_type ][0])),
                kind="cubic",
                fill_value="extrapolate")

            idx_aligned = np.round(interpolater(ME['body'][0])).astype(int)
            ME[video_type] = [ME['body'][0], ME[video_type][1][idx_aligned]]

      
        D = {}
     
        fs = 30
        xs = np.arange(rt*fs)  # number of frames 
        xs = np.concatenate([-1*np.array(list(reversed(xs[:int(abs(st)*fs)]))),
                              np.arange(rt*fs)[1:1+len(xs[int(abs(st)*fs):])]])
        xs = xs /float(fs)
        
        cols = {'left':'r','right':'b','body':'g'}
            
        for video_type in ME:
            # that's centered at feedback time
            
            
            D[video_type] = []
            
            times,s = ME[video_type]

            trs = trials[stype][20:-20]    
            for i in trs:

                start_idx = int(find_nearest(times,i) + st*30)
                end_idx = int(start_idx  + rt*30)  

                D[video_type].append(s[start_idx:end_idx])

        
            MEAN = np.mean(D[video_type],axis=0)
            STD = np.std(D[video_type],axis=0)/np.sqrt(len(trs)) 
           

            plt.plot(xs, MEAN, label=video_type, 
                     color = cols[video_type], linewidth = 2)
            plt.fill_between(xs, MEAN + STD, MEAN - STD, color = cols[video_type],
                             alpha=0.2)
            
        ax = plt.gca()
        ax.axvline(x=0, label='stimOn', linestyle = '--', c='k')
        plt.title('Motion Energy PSTH')
        plt.xlabel('time [sec]')
        plt.ylabel('z-scored motion energy [a.u.]') 
        plt.legend(loc='lower right')
        plt.show()        
        
    except:
        plt.title('No motion energy available!')


def get_ME(eid, video_type):

    #video_type = 'left'    
    one = ONE()       
    
    
    #Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy') 
    Times = one.load_dataset(eid, '_ibl_%sCamera.times.npy' % video_type)
    ME = one.load_dataset(eid,f'alf/{video_type}Camera.ROIMotionEnergy.npy')
    

    return Times, ME


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx   
    
    
def smooth_interpolate_signal_sg(signal, window=31, order=3, interp_kind='cubic'):
    """Run savitzy-golay filter on signal, interpolate through nan points.
    
    Parameters
    ----------
    signal : np.ndarray
        original noisy signal of shape (t,), may contain nans
    window : int
        window of polynomial fit for savitzy-golay filter
    order : int
        order of polynomial for savitzy-golay filter
    interp_kind : str
        type of interpolation for nans, e.g. 'linear', 'quadratic', 'cubic'

    Returns
    -------
    np.array
        smoothed, interpolated signal for each time point, shape (t,)
        
    """

    signal_noisy_w_nans = np.copy(signal)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    good_idxs = np.where(~np.isnan(signal_noisy_w_nans))[0]
    # perform savitzky-golay filtering on non-nan points
    signal_smooth_nonans = non_uniform_savgol(
        timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
    signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
    signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
    # interpolate nan points
    interpolater = interp1d(
        timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')

    signal = interpolater(timestamps)

    return signal


def non_uniform_savgol(x, y, window, polynom):
    """Applies a Savitzky-Golay filter to y with non-uniform spacing as defined in x.

    This is based on 
    https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    https://dsp.stackexchange.com/a/64313

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array
        The smoothed y values
    """

    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]
        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]
        # Multiply the two matrices
        tAA = np.matmul(tA, A)
        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)
        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)
        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]
        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]
    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]
    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]
    return y_smoothed


# Position functions
def get_pupil_diameter(XYs):
    """Estimate pupil diameter by taking median of different computations.
    
    In the two most obvious ways:
    d1 = top - bottom, d2 = left - right
    
    In addition, assume the pupil is a circle and estimate diameter from other pairs of 
    points
    
    Author: Michael Schartner
    
    Parameters
    ----------
    XYs : dict
        keys should include `pupil_top_r`, `pupil_bottom_r`, 
        `pupil_left_r`, `pupil_right_r`

    Returns
    -------
    np.array
        pupil diameter estimate for each time point, shape (n_frames,)
    
    """
    
    # direct diameters
    t = XYs['pupil_top_r'][:, :2]
    b = XYs['pupil_bottom_r'][:, :2]
    l = XYs['pupil_left_r'][:, :2]
    r = XYs['pupil_right_r'][:, :2]

    def distance(p1, p2):
        return ((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2) ** 0.5

    # get diameter via top-bottom and left-right
    ds = []
    ds.append(distance(t, b))
    ds.append(distance(l, r))

    def dia_via_circle(p1, p2):
        # only valid for non-crossing edges
        u = distance(p1, p2)
        return u * (2 ** 0.5)

    # estimate diameter via circle assumption
    for side in [[t, l], [t, r], [b, l], [b, r]]:
        ds.append(dia_via_circle(side[0], side[1]))
    diam = np.nanmedian(ds, axis=0)


    return diam

# Find pupil center
def pupil_center(XYs):

    # direct diameters
    t = XYs['pupil_top_r'][:, :2]
    b = XYs['pupil_bottom_r'][:, :2]
    l = XYs['pupil_left_r'][:, :2]
    r = XYs['pupil_right_r'][:, :2]

    X = np.nanmean(np.stack((l[:, 0], r[:, 0]), axis=0), axis=0)
    Y = np.nanmean(np.stack((t[:, 1], b[:, 1]), axis=0), axis=0)

    return X, Y


# Find nose tip
def nose_tip(XYs):

    X = XYs['nose_tip'][:, 0]
    Y = XYs['nose_tip'][:, 1]

    return Y, X


# Find tongue tip
def tongue_tip(XYs):

    left_tongue = XYs['tongue_end_l'][:, 0]
    right_tongue = XYs['tongue_end_r'][:, 1]

    X = np.nanmean(np.stack((left_tongue, right_tongue), axis=0), axis=0)
    Y = np.nanmean(np.stack((left_tongue, right_tongue), axis=0), axis=0)

    return Y, X


# Find left paw
def left_paw(XYs):

    X = XYs['paw_l'][:, 0]
    Y = XYs['paw_l'][:, 1]

    return Y, X


# Find right paw
def right_paw(XYs):

    X = XYs['paw_r'][:, 0]
    Y = XYs['paw_r'][:, 1]

    return Y, X

"""
def get_raw_and_smooth_position(eid, video_type, ephys, position_function):

    # likelihood threshold
    l_thresh = 0.9

    # camera view
    view = video_type

    # threshold (in standard deviations) beyond which a point is labeled as an outlier
    std_thresh = 5

    # threshold (in seconds) above which we will not interpolate nans, but keep them
    # (for long stretches interpolation may not be appropriate)
    nan_thresh = 1

    # compute framerate of camera
    if view == 'left':
        if ephys == True:
            fr = 60  # set by hardware
            window = 31  # works well empirically
        else:
            fr = 30
            window = 31 # TODO: need to validate this
    elif view == 'right':
        fr = 150  # set by hardware
        window = 75  # works well empirically
    else:
        raise NotImplementedError

    # load markers
    _, markers = get_dlc_XYs(eid, view, likelihood_thresh=l_thresh)

    # compute diameter using raw values of 4 markers (will be noisy and have missing data)
    if position_function == get_pupil_diameter:
        
        X_center0 = position_function(markers)
        
        # XX
        # run savitzy-golay filter on non-nan timepoints to denoise
        X_center_sm0 = smooth_interpolate_signal_sg(
            X_center0, window=window, order=3, interp_kind='linear')

        # find outliers, set to nan
        errors = X_center0 - X_center_sm0
        std = np.nanstd(errors)
        X_center1 = np.copy(X_center0)
        X_center1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan
        # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
        X_center_sm1 = smooth_interpolate_signal_sg(
            X_center1, window=window, order=3, interp_kind='linear')

        # don't interpolate long strings of nans
        t = np.diff(1 * np.isnan(X_center1))
        begs = np.where(t == 1)[0]
        ends = np.where(t == -1)[0]
        if begs.shape[0] > ends.shape[0]:
            begs = begs[:ends.shape[0]]
        for b, e in zip(begs, ends):
            if (e - b) > (fr * nan_thresh):
                X_center_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff
                
        Y_center0 = []
        Y_center_sm1 = []
    else:
        X_center0, Y_center0 = position_function(markers)

        # XX
        # run savitzy-golay filter on non-nan timepoints to denoise
        X_center_sm0 = smooth_interpolate_signal_sg(
            X_center0, window=window, order=3, interp_kind='linear')

        # find outliers, set to nan
        errors = X_center0 - X_center_sm0
        std = np.nanstd(errors)
        X_center1 = np.copy(X_center0)
        X_center1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan
        # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
        X_center_sm1 = smooth_interpolate_signal_sg(
            X_center1, window=window, order=3, interp_kind='linear')

        # don't interpolate long strings of nans
        t = np.diff(1 * np.isnan(X_center1))
        begs = np.where(t == 1)[0]
        ends = np.where(t == -1)[0]
        if begs.shape[0] > ends.shape[0]:
            begs = begs[:ends.shape[0]]
        for b, e in zip(begs, ends):
            if (e - b) > (fr * nan_thresh):
                X_center_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff
            
        # YY
        # run savitzy-golay filter on non-nan timepoints to denoise
        Y_center_sm0 = smooth_interpolate_signal_sg(
            Y_center0, window=window, order=3, interp_kind='linear')

        # find outliers, set to nan
        errors = Y_center0 - Y_center_sm0
        std = np.nanstd(errors)
        Y_center1 = np.copy(Y_center0)
        Y_center1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan
        # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
        Y_center_sm1 = smooth_interpolate_signal_sg(
            Y_center1, window=window, order=3, interp_kind='linear')

        # don't interpolate long strings of nans
        t = np.diff(1 * np.isnan(Y_center1))
        begs = np.where(t == 1)[0]
        ends = np.where(t == -1)[0]
        if begs.shape[0] > ends.shape[0]:
            begs = begs[:ends.shape[0]]
        for b, e in zip(begs, ends):
            if (e - b) > (fr * nan_thresh):
                Y_center_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff
            
            
    # diam_sm1 is the final smoothed pupil diameter estimate
    return X_center0, X_center_sm1, Y_center0, Y_center_sm1
"""

def get_raw_and_smooth_position(eid, video_type, ephys, position_function):
    """Params
    position_function: get_pupil_diameter, keypoint, ..."""

    # likelihood threshold
    l_thresh = 0.9

    # camera view
    view = video_type

    # threshold (in standard deviations) beyond which a point is labeled as an outlier
    std_thresh = 5

    # threshold (in seconds) above which we will not interpolate nans, but keep them
    # (for long stretches interpolation may not be appropriate)
    nan_thresh = 1

    # compute framerate of camera
    if view == 'left':
        if ephys == True:
            fr = 60  # set by hardware
            window = 31  # works well empirically
        else:
            fr = 30
            window = 31 # TODO: need to validate this
    elif view == 'right':
        fr = 150  # set by hardware
        window = 75  # works well empirically
    else:
        raise NotImplementedError

    """ Load markers"""
    _, markers = get_dlc_XYs(eid, view, likelihood_thresh=l_thresh)

    # Get XY position directly based on string
    if type(position_function) == str:
        X_center0 = markers[position_function].T[0]
        Y_center0 = markers[position_function].T[1]   
    
        """ Smooth YYs """
        # run savitzy-golay filter on non-nan timepoints to denoise
        Y_center_sm0 = smooth_interpolate_signal_sg(
            Y_center0, window=window, order=3, interp_kind='linear')

        # find outliers, set to nan
        errors = Y_center0 - Y_center_sm0
        std = np.nanstd(errors)
        Y_center1 = np.copy(Y_center0)
        Y_center1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan
        # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
        Y_center_sm1 = smooth_interpolate_signal_sg(
            Y_center1, window=window, order=3, interp_kind='linear')

        # don't interpolate long strings of nans
        t = np.diff(1 * np.isnan(Y_center1))
        begs = np.where(t == 1)[0]
        ends = np.where(t == -1)[0]
        if begs.shape[0] > ends.shape[0]:
            begs = begs[:ends.shape[0]]
        for b, e in zip(begs, ends):
            if (e - b) > (fr * nan_thresh):
                Y_center_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff
                
    # Or based on the mean of multiple keypoints
    elif (type(position_function) != str) and (position_function != get_pupil_diameter):
        
        X_center0, Y_center0 = position_function(markers)
            
        """ Smooth YYs """
        # run savitzy-golay filter on non-nan timepoints to denoise
        Y_center_sm0 = smooth_interpolate_signal_sg(
            Y_center0, window=window, order=3, interp_kind='linear')

        # find outliers, set to nan
        errors = Y_center0 - Y_center_sm0
        std = np.nanstd(errors)
        Y_center1 = np.copy(Y_center0)
        Y_center1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan
        # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
        Y_center_sm1 = smooth_interpolate_signal_sg(
            Y_center1, window=window, order=3, interp_kind='linear')

        # don't interpolate long strings of nans
        t = np.diff(1 * np.isnan(Y_center1))
        begs = np.where(t == 1)[0]
        ends = np.where(t == -1)[0]
        if begs.shape[0] > ends.shape[0]:
            begs = begs[:ends.shape[0]]
        for b, e in zip(begs, ends):
            if (e - b) > (fr * nan_thresh):
                Y_center_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff

    # If metric is pupil diameter, it is one D (XX)
    elif position_function == get_pupil_diameter:
        
        X_center0 = position_function(markers)

        # Save Y vars as empty arrays
        Y_center0 = []
        Y_center_sm1 = []
        
    """ Smooth XXs """
    # run savitzy-golay filter on non-nan timepoints to denoise
    X_center_sm0 = smooth_interpolate_signal_sg(
        X_center0, window=window, order=3, interp_kind='linear')

    # find outliers, set to nan
    errors = X_center0 - X_center_sm0
    std = np.nanstd(errors)
    X_center1 = np.copy(X_center0)
    X_center1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan
    # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
    X_center_sm1 = smooth_interpolate_signal_sg(
        X_center1, window=window, order=3, interp_kind='linear')

    # don't interpolate long strings of nans
    t = np.diff(1 * np.isnan(X_center1))
    begs = np.where(t == 1)[0]
    ends = np.where(t == -1)[0]
    if begs.shape[0] > ends.shape[0]:
        begs = begs[:ends.shape[0]]
    for b, e in zip(begs, ends):
        if (e - b) > (fr * nan_thresh):
            X_center_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff
        
            
    # diam_sm1 is the final smoothed pupil diameter estimate
    return X_center0, X_center_sm1, Y_center0, Y_center_sm1


def get_raw_and_smooth_ME(X_center0, video_type, ephys):
    """Params
    position_function: get_pupil_diameter, keypoint, ..."""

    # likelihood threshold
    l_thresh = 0.9

    # camera view
    view = video_type

    # threshold (in standard deviations) beyond which a point is labeled as an outlier
    std_thresh = 5

    # threshold (in seconds) above which we will not interpolate nans, but keep them
    # (for long stretches interpolation may not be appropriate)
    nan_thresh = 1

    # compute framerate of camera
    if view == 'left':
        if ephys == True:
            fr = 60  # set by hardware
            window = 31  # works well empirically
        else:
            fr = 30
            window = 31 # TODO: need to validate this
    elif view == 'right':
        fr = 150  # set by hardware
        window = 75  # works well empirically
    else:
        raise NotImplementedError
    
    # Save Y vars as empty arrays
    Y_center0 = []
    Y_center_sm1 = []
        
    """ Smooth XXs """
    # run savitzy-golay filter on non-nan timepoints to denoise
    X_center_sm0 = smooth_interpolate_signal_sg(
        X_center0, window=window, order=3, interp_kind='linear')

    # find outliers, set to nan
    errors = X_center0 - X_center_sm0
    std = np.nanstd(errors)
    X_center1 = np.copy(X_center0)
    X_center1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan
    # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
    X_center_sm1 = smooth_interpolate_signal_sg(
        X_center1, window=window, order=3, interp_kind='linear')

    # don't interpolate long strings of nans
    t = np.diff(1 * np.isnan(X_center1))
    begs = np.where(t == 1)[0]
    ends = np.where(t == -1)[0]
    if begs.shape[0] > ends.shape[0]:
        begs = begs[:ends.shape[0]]
    for b, e in zip(begs, ends):
        if (e - b) > (fr * nan_thresh):
            X_center_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff
        
            
    # diam_sm1 is the final smoothed pupil diameter estimate
    return X_center0, X_center_sm1, Y_center0, Y_center_sm1
 
    
def SNR(diam0, diam_sm1):

    # compute signal to noise ratio between raw and smooth dia
    good_idxs = np.where(~np.isnan(diam_sm1) & ~np.isnan(diam0))[0]
    snr = (np.var(diam_sm1[good_idxs]) / 
           np.var(diam_sm1[good_idxs] - diam0[good_idxs]))
           
    return snr


# By Ines
def stack_pupil(position, time, trials, event, t_init, t_end):
    
    time_step = np.median(np.diff(time))
    interval_length = int((t_end+t_init)/time_step + .25 * (t_end+t_init)/time_step) # This serves as an estimation for how large the data might be?
    onset_times = trials[event]

    # Initialize dataframe
    df = pd.DataFrame({'time':time, 'position':position})
    
    # Calculate max and min of the session for normalization
    max_pupil = np.max(df['position'])
    min_pupil = np.min(df['position'])

    # Normalize pupil size
    df['norm_position'] = (df['position'] - min_pupil) * 100 / (max_pupil - min_pupil)

    # Start a matrix with #trials x # enough entries for the interval # TODO: need to fix interval length
    stack = np.zeros((len(onset_times), interval_length)) * np.nan
    stack_time = np.zeros((len(onset_times), interval_length)) * np.nan

    for t, trial_onset in enumerate(onset_times):
        if np.isnan(trial_onset) == False:
            if len(df.loc[df['time'] > trial_onset-t_init, 'time']) > 0:

                # Calculate baseline (always in relation to StimOnTime)
                stim_on = trials['stimOn_times'][t]
                baseline = np.mean(df.loc[(df['time'] > stim_on-0.5) & (df['time'] < stim_on), 'norm_position'])

                # Populate dataframe with useful trial-aligned information
                temp_stack = df.loc[(df['time']> trial_onset-t_init) & (df['time'] <= trial_onset+t_end), 'norm_position'] - baseline
                stack[t, 0:len(temp_stack)] = temp_stack
                
                # Save time
                temp_time = df.loc[(df['time']> trial_onset-t_init) & (df['time'] <= trial_onset+t_end), 'time'] - stim_on
                stack_time[t, 0:len(temp_time)] = temp_time
    return stack, stack_time

"""
def keypoint_speed(eid, body_part):

    fs = {'right':150,'left':60}   

    # if it is the paw, take speed from right paw only, i.e. closer to cam  
    # for each video
    speeds = {}
    for video_type in ['right','left']:
        times, XYs = get_dlc_XYs(eid, video_type)
        
        # Pupil requires averaging 4 keypoints
        if body_part == 'pupil':
            x, y = pupil_center(XYs)
        else:
            x = XYs[body_part].T[0]
            y = XYs[body_part].T[1]   
            
        if video_type == 'left': #make resolution same
            x = x/2
            y = y/2
        
        # get speed in px/sec [half res]
        # Speed vector is given by the Pitagorean theorem
        s = ((np.diff(x)**2 + np.diff(y)**2)**.5)*fs[video_type]
        
        speeds[video_type] = [times,s]   
        
    return speeds
    """


def keypoint_speed(eid, ephys, body_part, split):

    if ephys ==True:
        fs = {'right':150,'left':60}   
    else:
        fs = {'right':150,'left':30}

    # if it is the paw, take speed from right paw only, i.e. closer to cam  
    # for each video
    speeds = {}
    for video_type in ['right','left']:
        times, _ = get_dlc_XYs(eid, video_type)
        
        # Pupil requires averaging 4 keypoints
        _, x, _, y = get_raw_and_smooth_position(eid, video_type, ephys, body_part)
        if body_part == get_pupil_diameter:
            if video_type == 'left': #make resolution same
                x = x/2
                
            # get speed in px/sec [half res]    
            s = np.diff(x)*fs[video_type]
            speeds[video_type] = [times,s]
        
        else:
            if video_type == 'left': #make resolution same
                x = x/2
                y = y/2
            
            # Calculate velocity for x and y separately if split is true
            if split == True:
                s_x = np.diff(x)*fs[video_type]
                s_y = np.diff(y)*fs[video_type]
                speeds[video_type] = [times, s_x, s_y]

            else:
                # Speed vector is given by the Pitagorean theorem
                s = ((np.diff(x)**2 + np.diff(y)**2)**.5)*fs[video_type]
                speeds[video_type] = [times,s]
        
    return speeds


def downsample (metric, to_shorten, reference):
    """ Make all arrays be the same size """

    for video_type in to_shorten:

        # align time series camera/neural
        interpolater = interp1d(
            metric[video_type ][0][:-1],
            np.arange(len(metric[video_type ][0][:-1])),
            kind="cubic",
            fill_value="extrapolate")

        idx_aligned = np.round(interpolater(metric[reference][0][:-1])).astype(int)
                
        if len(metric[video_type]) > 2:
            
            metric[video_type] = [metric[reference][0], metric[video_type][1][idx_aligned], metric[video_type][2][idx_aligned]]
        else:
            metric[video_type] = [metric[reference][0], metric[video_type][1][idx_aligned]]
            
    
    return metric
