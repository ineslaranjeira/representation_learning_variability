# %%
"""
Imports
"""
import pandas as pd
import pickle 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from one.api import ONE
import brainbox.behavior.wheel as wh
from scipy.stats import zscore
from scipy import stats
from scipy import signal

# Get my functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Functions/'
#functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Functions/'

os.chdir(functions_path)
from one_functions_generic import query_subjects_interest, subjects_interest_data, prepro, timeseries_PSTH
from video_functions import keypoint_speed, downsample, pupil_center, get_dlc_XYs, find_nearest, get_raw_and_smooth_position, get_pupil_diameter, pupil_center, nose_tip, tongue_tip
from design_functions import lick_rate, wheel_velocity, wheel_displacement, pupil_diam, cont_bin, align_stimOn
# %%
one = ONE()

# Choose a session with good QC
data_path = '/home/ines/repositories/representation_learning_variability/Video/'
#data_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Video/'

os.chdir(data_path)
pass_qc = pickle.load(open(data_path + "good_brainwide_sessions", "rb"))

# Parameters
bin_size = 0.1  # seconds
video_type = 'left'
save_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Exported figures/'
save_path = '/home/ines/repositories/representation_learning_variability/Exported figures/'


# %%
# Choose one pair of metrics for cross-correlogram
# choose from here: 'pupil + wheel_vel'; 'pupil + paw'; 'noseX + wheel_vel'
metric = 'noseX + wheel_vel'

if metric == 'pupil + wheel_vel':
    #### Pupil diameter and wheel velocity ####

    for s, session in enumerate(list(pass_qc)):
        # Trials data
        session_trials = one.load_object(session, obj='trials', namespace='ibl')
        session_trials = prepro(session_trials.to_df())

        # Wheel
        wheel = one.load_object(session, 'wheel', collection='alf')
        pos, wheel_times = wh.interpolate_position(wheel.timestamps, wheel.position)

        # Pupil diameter
        pupil_dia_raw, pupil_dia_smooth, _, _ = (
            get_raw_and_smooth_position(session, video_type, ephys=True, 
                                        position_function=get_pupil_diameter))
        pupil_t, XYs = get_dlc_XYs(session, view=video_type, likelihood_thresh=0.9)

        session_length = len(session_trials['stimOff_times'])
        n_bins = int(np.floor(session_length/bin_size))
        onsets = session_trials['stimOn_times']

        try: # TODO: need to solve individual issues and remove try
            # Initialize dataframe
            wheel_vel = wheel_velocity(bin_size, wheel_times, pos, session_trials)
            pupil = pupil_diam(pupil_t, pupil_dia_smooth, session_trials, bin_size, onset_subtraction=False)
            wheel_disp = wheel_displacement(wheel_times, pos, session_trials, bin_size, onset_subtraction=False)

            pupil = pupil.rename(columns={'pupil_final':'pupil_diameter'})

            # Merge data
            all_metrics = wheel_vel[['Bin', 'avg_wheel_vel']].merge(wheel_disp[['Bin', 'Onset times']], on='Bin', how='outer')
            all_metrics = all_metrics.merge(pupil[['Bin', 'pupil_diameter']], on='Bin', how='outer')

            # Remove wheel disp (was used just to get onset times alignment with bins)
            data_df = all_metrics.dropna().drop_duplicates()

            data_df['avg_wheel_vel'] = np.abs(data_df['avg_wheel_vel'])

            # Plot
            x = data_df['pupil_diameter']
            y = data_df['avg_wheel_vel']

            # Calculate the lag values corresponding to the cross-correlation
            lags = np.arange(-100, 100)

            # Compute the cross-correlation using np.correlate
            cross_correlation = np.zeros(len(lags)) * np.nan
            len_timeseries = len(x)
            for l, lag in enumerate(lags):
                if lag < 0:
                    x_chunk = np.array(x[-lag:])
                    y_chunk = np.array(y[:lag])
                elif lag == 0:
                    x_chunk = np.array(x)
                    y_chunk = np.array(y)
                elif lag > 0:
                    x_chunk = np.array(x[0:-lag])
                    y_chunk = np.array(y[lag:])
                cross_correlation[l] = stats.pearsonr(x_chunk, y_chunk).statistic

            # Plot the cross-correlation
            plt.scatter(lags, cross_correlation)
            plt.hlines(0, np.min(lags), np.max(lags), color='gray', linestyles='--')
            plt.vlines(0, -1, 1, color='gray', linestyles='--')
            plt.xlabel('Lag')
            plt.ylabel('Cross-Correlation')
            plt.title('Cross-Correlogram pupil diam - wheel velocity')
            plt.ylim([-.5, .5])
                    
            plt.tight_layout()

            # Save the plot as a PNG file
            plt.savefig(str(save_path + session + '.png'), format='png')

            # Display the plot
            plt.show()
        except:
            print(session)
    
elif metric == 'pupil + paw':

    #### Pupil diameter and paw speed ####

    for s, session in enumerate(list(pass_qc)[2:]):
        # Trials data
        session_trials = one.load_object(session, obj='trials', namespace='ibl')
        session_trials = prepro(session_trials.to_df())

        # Pupil diameter
        pupil_dia_raw, pupil_dia_smooth, _, _ = (
            get_raw_and_smooth_position(session, video_type, ephys=True, 
                                        position_function=get_pupil_diameter))
        pupil_t, XYs = get_dlc_XYs(session, view=video_type, likelihood_thresh=0.9)

        # Left paw velocity
        left_p_speeds = keypoint_speed(session, True, 'paw_r', True)
        left_p_times = left_p_speeds['left'][0][1:]
        left_p_speed_X = left_p_speeds['left'][1]
            
        session_length = len(session_trials['stimOff_times'])
        n_bins = int(np.floor(session_length/bin_size))
        onsets = session_trials['stimOn_times']

        try: # TODO: need to solve individual issues and remove try
            # Initialize dataframe
            pupil = pupil_diam(pupil_t, pupil_dia_smooth, session_trials, bin_size, onset_subtraction=False)
            left_vel_X = cont_bin(left_p_times, left_p_speed_X, session_trials, bin_size)

            pupil = pupil.rename(columns={'pupil_final':'pupil_diameter'})
            left_vel_X = left_vel_X.rename(columns={'Values':'l_paw_speed_X'})
            
            # Merge data
            all_metrics = pupil[['Bin', 'pupil_diameter']].merge(left_vel_X[['Bin', 'l_paw_speed_X']], on='Bin', how='outer')

            # Remove wheel disp (was used just to get onset times alignment with bins)
            data_df = all_metrics.dropna().drop_duplicates()

            # Plot
            x = data_df['pupil_diameter']
            y = data_df['l_paw_speed_X']

            # Calculate the lag values corresponding to the cross-correlation
            lags = np.arange(-100, 100)

            # Compute the cross-correlation using np.correlate
            cross_correlation = np.zeros(len(lags)) * np.nan
            len_timeseries = len(x)
            for l, lag in enumerate(lags):
                if lag < 0:
                    x_chunk = np.array(x[-lag:])
                    y_chunk = np.array(y[:lag])
                elif lag == 0:
                    x_chunk = np.array(x)
                    y_chunk = np.array(y)
                elif lag > 0:
                    x_chunk = np.array(x[0:-lag])
                    y_chunk = np.array(y[lag:])
                cross_correlation[l] = stats.pearsonr(x_chunk, y_chunk).statistic

            # Plot the cross-correlation
            plt.scatter(lags, cross_correlation)
            plt.hlines(0, np.min(lags), np.max(lags), color='gray', linestyles='--')
            plt.vlines(0, -1, 1, color='gray', linestyles='--')
            plt.xlabel('Lag')
            plt.ylabel('Cross-Correlation')
            plt.title('Cross-Correlogram pupil diam - paw velocity')
            plt.ylim([-.5, .5])
                    
            plt.tight_layout()

            # Save the plot as a PNG file
            plt.savefig(str(save_path + session + '.png'), format='png')

            # Display the plot
            plt.show()
        except:
            print(session)


elif metric == 'noseX + wheel_vel':
    #### Nose speed and wheel velocity ####

    for s, session in enumerate(list(pass_qc)):
        # Trials data
        session_trials = one.load_object(session, obj='trials', namespace='ibl')
        session_trials = prepro(session_trials.to_df())

        # Wheel
        wheel = one.load_object(session, 'wheel', collection='alf')
        pos, wheel_times = wh.interpolate_position(wheel.timestamps, wheel.position)

        # Nose velocity
        nose_speeds = keypoint_speed(session, True, 'nose_tip', True)
        nose_times = nose_speeds['left'][0][1:]
        nose_speed_X = nose_speeds['left'][1]

        session_length = len(session_trials['stimOff_times'])
        n_bins = int(np.floor(session_length/bin_size))
        onsets = session_trials['stimOn_times']

        #try: # TODO: need to solve individual issues and remove try
        # Initialize dataframe
        wheel_vel = wheel_velocity(bin_size, wheel_times, pos, session_trials)
        nose_vel_X = cont_bin(nose_times, nose_speed_X, session_trials, bin_size)

        nose_vel_X = nose_vel_X.rename(columns={'Values':'nose_speed_X'})

        # Merge data
        all_metrics = wheel_vel[['Bin', 
                                    'avg_wheel_vel']].merge(nose_vel_X[['Bin',
                                                                        'nose_speed_X']],
                                                            on='Bin', how='outer')
        # = align_stimOn(all_metrics, session_trials)

        # Remove wheel disp (was used just to get onset times alignment with bins)
        data_df = all_metrics.dropna().drop_duplicates()

        data_df['avg_wheel_vel'] = np.abs(data_df['avg_wheel_vel'])

        # Plot
        x = data_df['nose_speed_X']
        y = data_df['avg_wheel_vel']

        # Calculate the lag values corresponding to the cross-correlation
        lags = np.arange(-100, 100)

        # Compute the cross-correlation using np.correlate
        cross_correlation = np.zeros(len(lags)) * np.nan
        len_timeseries = len(x)
        for l, lag in enumerate(lags):
            if lag < 0:
                x_chunk = np.array(x[-lag:])
                y_chunk = np.array(y[:lag])
            elif lag == 0:
                x_chunk = np.array(x)
                y_chunk = np.array(y)
            elif lag > 0:
                x_chunk = np.array(x[0:-lag])
                y_chunk = np.array(y[lag:])
            cross_correlation[l] = stats.pearsonr(x_chunk, y_chunk).statistic

        # Plot the cross-correlation
        plt.scatter(lags, cross_correlation)
        plt.hlines(0, np.min(lags), np.max(lags), color='gray', linestyles='--')
        plt.vlines(0, -1, 1, color='gray', linestyles='--')
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.title('Cross-Correlogram pupil diam - wheel velocity')
        plt.ylim([-.5, .5])
                
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(str(save_path + session + '.png'), format='png')

        # Display the plot
        plt.show()
        #except:
        #    print(session)
    
# %%