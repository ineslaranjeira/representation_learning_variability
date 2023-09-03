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
from video_functions import keypoint_speed, downsample, pupil_center, get_dlc_XYs, find_nearest, get_raw_and_smooth_ME, get_raw_and_smooth_position, get_ME, get_pupil_diameter, pupil_center, nose_tip, tongue_tip
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
event = 'stimOn_times'
t_init = 0.5
t_end = 2
    
save_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Exported figures/'
save_path = '/home/ines/repositories/representation_learning_variability/Exported figures/'

# %% 
# Loop over sessions

for s, session in enumerate(list(pass_qc)):

    # Trials data
    session_trials = one.load_object(session, obj='trials', namespace='ibl')
    session_trials = prepro(session_trials.to_df())

    try:
        
        # Motion energy of whisker pad
        me_times, motion_energy = get_ME(session, video_type)
        X_center0, smooth_me, _, _ = get_raw_and_smooth_ME(motion_energy, video_type='left', ephys=True)       
        smooth_me = zscore(smooth_me, nan_policy='omit') 

        # Align on stimulus onset
        me_aligned = timeseries_PSTH(me_times, smooth_me, session_trials, event, t_init, t_end, subtract_baseline=False)

        # Get peak of
        max_ME = pd.DataFrame(me_aligned.groupby(['feedback_time'])['value'].max())
        max_ME = max_ME.reset_index(level=[0])
        max_ME = max_ME.rename(columns={'value':'max_ME'})

        merged = me_aligned.merge(max_ME, on='feedback_time', how='inner')
        reduced = merged.loc[merged['value']==merged['max_ME']]

        # Plot per session
        plt.tight_layout()
        sns.lineplot(x='variable', y='value', data=me_aligned, alpha=0.5)
        plt.vlines(0, -1, 4, color='grey', linestyles='--')
        plt.xlabel('Time from stimulus onset (s)')
        plt.ylabel('Z-scored Motion energy')
        #plt.savefig(str(save_path + 'me_PSTH' + session + '.png'), format='png')
        plt.show()

        # Find maximum ME for each trial
        sns.scatterplot(x='variable', y='response_time', data=reduced)
        plt.ylim([0,t_end])
        plt.xlim([0,t_end])
        plt.xlabel('Motion energy peak (s)')
        plt.ylabel('Response time (s)')
        stats.pearsonr(reduced['variable'], reduced['response_time']).statistic

        # Save the plot as a PNG file
        plt.savefig(str(save_path + 'corr_me_rt' + session + '.png'), format='png')

        # Display the plot
        plt.show()
    
    except:
        print(session)

# %%
