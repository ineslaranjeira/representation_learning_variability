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

# Get my functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Functions/'
#functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Functions/'

os.chdir(functions_path)
from one_functions_generic import query_subjects_interest, subjects_interest_data, prepro, timeseries_PSTH
from video_functions import keypoint_speed, downsample, pupil_center, get_dlc_XYs, find_nearest, get_raw_and_smooth_position, get_pupil_diameter, pupil_center, nose_tip, tongue_tip
from design_functions import lick_rate, wheel_velocity, wheel_displacement, pupil_diam, cont_bin

# %%
one = ONE()

# Choose a session with good QC
data_path = '/home/ines/repositories/representation_learning_variability/Video/'
#data_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Video/'

os.chdir(data_path)
pass_qc = pickle.load(open(data_path + "good_brainwide_sessions", "rb"))

# %%
# Parameters
bin_size = 0.1  # seconds
video_type = 'left'
save_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Exported figures/'
save_path = '/home/ines/repositories/representation_learning_variability/Exported figures/'

# %% 
# Loop over sessions

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

    #session_length = list(session_trials['stimOff_times'][-1:])[0]
    session_length = len(session_trials['stimOff_times'])
    n_bins = int(np.floor(session_length/bin_size))
    onsets = session_trials['stimOn_times']

    #if np.sum(np.isnan(onsets)) < 5:
    try: # TODO: need to solve individual issues and remove try
        # Initialize dataframe
        wheel_vel = wheel_velocity(bin_size, wheel_times, pos, session_trials)
        pupil = pupil_diam(pupil_t, pupil_dia_smooth, session_trials, bin_size, onset_subtraction=True)
        wheel_disp = wheel_displacement(wheel_times, pos, session_trials, bin_size, onset_subtraction=False)

        pupil = pupil.rename(columns={'pupil_final':'pupil_diameter'})

        # Merge data
        all_metrics = wheel_vel[['Bin', 'avg_wheel_vel']].merge(wheel_disp[['Bin', 'Onset times']], on='Bin', how='outer')
        all_metrics = all_metrics.merge(pupil[['Bin', 'pupil_diameter']], on='Bin', how='outer')

        # Remove wheel disp (was used just to get onset times alignment with bins)
        data_df = all_metrics.dropna().drop_duplicates()
        
        # Bin data on both axes
        bin_edges = np.arange(-100, 100, 2.5)
        data_df['pupil_bin'] = pd.cut(data_df['pupil_diameter'], bins=bin_edges, labels=bin_edges[:-1])
        data_df['avg_wheel_vel'] = np.abs(data_df['avg_wheel_vel'])

        # Plot per session
        sns.barplot(x='pupil_bin', y='avg_wheel_vel', data=data_df)  # 
        plt.xlabel('Pupil diameter (%)')
        plt.ylabel('Wheel velocity')
        plt.title(session)
        plt.xticks(rotation=90)
        # Plot only where there is data
        b = np.arange(0, len(bin_edges), 1)
        min = b[np.where(bin_edges==np.min(data_df['pupil_bin']))][0]
        max = b[np.where(bin_edges==np.max(data_df['pupil_bin']))][0]
        plt.xlim([min-1, max+1])

        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(str(save_path + session + '.png'), format='png')

        # Display the plot
        plt.show()
    except:
        print(session)
# %%
