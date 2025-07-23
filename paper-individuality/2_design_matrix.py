"""
2. Process raw data into design matrix
@author: Ines
"""
#%%

import os
import numpy as np
import pickle
import pandas as pd
from scipy.stats import zscore
from brainbox.io.one import SessionLoader
from sklearn.preprocessing import StandardScaler
import gc
import concurrent.futures

from functions import get_speed, merge_licks, resample_common_time, fast_wavelet_morlet_convolution_parallel

from one.api import ONE
one = ONE(mode='remote')

#%%

""" PARAMETERS """

bin_size = 0.017  # np.round(1/60, 3)  # No binning, number indicates sampling rate
video_type = 'left'    
first_90 = False  # Use full sessions #TODO consider removing this from here?

# Wavelet decomposition
f = np.array([.25, .5, 1, 2, 4, 8, 16])
omega0 = 5

#%%

""" Load BWM data post-QC """
prefix = '/home/ines/repositories/'
prefix = '/Users/ineslaranjeira/Documents/Repositories/'
data_path = prefix + '/representation_learning_variability/paper-individuality/'
filename = '1_bwm_qc_07-10-2025'



data_path = prefix + 'representation_learning_variability/Video and wheel/Video QC/'
filename = 'final_lp_qc07-10-2025'




bwm_query = pickle.load(open(data_path+filename, "rb"))

#%%

def process_design_matrix(session):    

    file_path = one.eid2path(session)
    if prefix == '/home/ines/repositories/':
        mouse_name = file_path.parts[8]
    else:
        mouse_name = file_path.parts[7]

    """ LOAD VARIABLES """
    sl = SessionLoader(eid=session, one=one)
    sl.load_pose(views=['left', 'right'], tracker='lightningPose')
    sl.load_session_data(trials=True, wheel=True, motion_energy=True)

    # Check if all data is available
    if np.sum(sl.data_info['is_loaded']) >= 4:
        
        # Motion energy
        me = sl.motion_energy
        me_time = np.array(me['leftCamera']['times'])
        motion_energy = np.array(zscore(me['leftCamera']['whiskerMotionEnergy'], nan_policy='omit'))
        
        # Licks 
        features = ['tongue_end_l_x', 'tongue_end_l_y','tongue_end_r_x', 'tongue_end_r_y']        
        common_fs = 150 
        poses = sl.pose
        licks_time, licks = merge_licks(poses, features, common_fs)

        # Paws
        paw_speeds = get_speed(poses['leftCamera'], poses['leftCamera']['times'], 'left', True, feature='paw_r')
        paw_time = np.array(paw_speeds['left'][0])
        paw_x = paw_speeds['left'][1]
        paw_y = paw_speeds['left'][2]
        
        # Wheel
        wheel = sl.wheel
        wheel_time = np.array(wheel['times'])
        wheel_vel = np.array(wheel['velocity'])
        
        """ COMMON TIMESTAMPS AND RESAMPLING"""
        # Use reference time, truncate and resample
        onset = np.max([np.min(poses['leftCamera']['times']), np.min(poses['rightCamera']['times']), np.min(wheel['times'])])
        offset = np.min([np.max(poses['leftCamera']['times']), np.max(poses['rightCamera']['times']), np.max(wheel['times'])])
        
        common_fs = 60
        # Set common sampling rate (high rather than low)
        reference_time = np.arange(onset, offset, 1/common_fs)  # uniform timestamps
        
        motion_energy = motion_energy[np.where((me_time >= onset) & (me_time <= offset))[0]]
        me_time = me_time[np.where((me_time >= onset) & (me_time <= offset))[0]]
        donwsampled_me, corrected_me_t = resample_common_time(reference_time, me_time, motion_energy, kind='linear', fill_gaps=None)

        wheel_vel = wheel_vel[np.where((wheel_time >= onset) & (wheel_time < offset))]
        wheel_time = wheel_time[np.where((wheel_time >= onset) & (wheel_time < offset))]
        donwsampled_wheel, corrected_wheel_t = resample_common_time(reference_time, wheel_time, wheel_vel, kind='linear', fill_gaps=None)

        licks = licks[np.where((licks_time >= onset) & (licks_time < offset))]
        licks_time = licks_time[np.where((licks_time >= onset) & (licks_time < offset))]
        donwsampled_lick, corrected_lick_t = resample_common_time(reference_time, licks_time, licks, kind='nearest', fill_gaps=None)

        paw_time = paw_time[np.where((paw_time >= onset) & (paw_time < offset))]
        paw_x = paw_x[np.where((paw_time >= onset) & (paw_time < offset))]
        donwsampled_paw_x, corrected_paw_x_t = resample_common_time(reference_time, paw_time, paw_x, kind='linear', fill_gaps=None)
        paw_y = paw_y[np.where((paw_time >= onset) & (paw_time < offset))]
        donwsampled_paw_y, corrected_paw_y_t = resample_common_time(reference_time, paw_time, paw_y, kind='linear', fill_gaps=None)

        # Check integrity of data
        assert (corrected_me_t == corrected_wheel_t).all(), print('Assertion error')
        assert (corrected_wheel_t == corrected_lick_t).all(), print('Assertion error')
        assert (corrected_lick_t == corrected_paw_x_t).all(), print('Assertion error')
        assert (corrected_paw_x_t == corrected_paw_y_t).all(), print('Assertion error')

        # Wavelet decomposition of wheel velocity
        dt = np.round(np.mean(np.diff(corrected_wheel_t)), 3)
        amp, Q, x_hat = fast_wavelet_morlet_convolution_parallel(donwsampled_wheel, f, omega0, dt)

        """ GROUP DATA INTO DESIGN MATRIX """
        design_matrix = pd.DataFrame(columns=['Bin', 'avg_wheel_vel', 'whisker_me', 'Lick count', 'paw_x', 'paw_y'])
        design_matrix['Bin'] = corrected_me_t.copy()
        design_matrix['Lick count'] = donwsampled_lick.copy()
        design_matrix['avg_wheel_vel'] = donwsampled_wheel.copy()
        design_matrix['whisker_me'] = donwsampled_me.copy()
        design_matrix['paw_x'] = donwsampled_paw_x.copy()
        design_matrix['paw_y'] = donwsampled_paw_y.copy()

        # Wavelet transforms
        for i, frequency in enumerate(f):
            # Create new column with frequency
            design_matrix[str(frequency)] = design_matrix['Bin'] * np.nan
            design_matrix[str(frequency)] = amp[i, :]
            
        # """ LOAD TRIAL DATA """
        session_trials = sl.trials
        session_start = list(session_trials['goCueTrigger_times'])[0]

        # Get time of last unbiased trial
        unbiased = session_trials.loc[session_trials['probabilityLeft']==0.5]
        time_trial_90 = list(unbiased['stimOff_times'])[-1]

        if first_90 == True:
            # Keep only first 90 trials
            design_matrix = design_matrix.loc[(design_matrix['Bin'] < time_trial_90) & 
                                                (design_matrix['Bin'] > session_start)]
            use_trials = session_trials.loc[session_trials['stimOff_times'] < time_trial_90]
        else:
            design_matrix = design_matrix.loc[(design_matrix['Bin'] > session_start)]
            use_trials = session_trials.copy()

        """ STANDARDIZE DATA """
        training_set = np.array(design_matrix).copy()[:, 1:]
        # Standardization using StandardScaler
        scaler = StandardScaler()
        std_design_matrix = scaler.fit_transform(training_set)
        # Keep licks unnormalized
        std_design_matrix[:, 2] = training_set[:, 2]  

        """ SAVE DATA """       
        # Save unnormalized design matrix
        data_path =  prefix + 'representation_learning_variability/DATA/Sub-trial/Design matrix/v6_21Jul2025/' + str(bin_size) + '/'
        filename = data_path + "design_matrix_" + str(session) + '_'  + mouse_name
        design_matrix.to_parquet(filename, compression='gzip')  

        # Save standardized design matrix
        data_path =  prefix + 'representation_learning_variability/DATA/Sub-trial/Design matrix/v6_21Jul2025/' + str(bin_size) + '/'
        filename = data_path + "standardized_design_matrix_" + str(session) + '_'  + mouse_name
        np.save(filename, std_design_matrix)

        # Save trials
        data_path =  prefix + 'representation_learning_variability/DATA/Sub-trial/Design matrix/v6_21Jul2025/' + str(bin_size) + '/'
        filename = data_path + "session_trials_" + str(session) + '_'  + mouse_name
        use_trials.to_parquet(filename, compression='gzip')  
        
        del design_matrix, std_design_matrix, use_trials, sl
        gc.collect()

    else:
        print('Data missing for session '+session)  


def parallel_process_data(sessions, function_name):
    with concurrent.futures.ThreadPoolExecutor() as executor:

        # Process each chunk in parallel
        executor.map(function_name, sessions)

#%%
# Loop through animals
function_name = process_design_matrix
sessions = bwm_query['eid'].unique()
data_path = prefix + 'representation_learning_variability/DATA/Sub-trial/Design matrix/v6_21Jul2025/' + str(bin_size) + '/'
os.chdir(data_path)
files = os.listdir()
sessions_to_process = []

for s, sess in enumerate(sessions):
    file_path = one.eid2path(sess)

    if prefix == '/home/ines/repositories/':
        mouse_name = file_path.parts[8]
    else:
        mouse_name = file_path.parts[7]

    filename = "design_matrix_" + str(sess) + '_'  + mouse_name
    if filename not in files:
        sessions_to_process.append((sess))

len(sessions_to_process)

#%%
for s, session in enumerate(sessions_to_process[:10]):
    process_design_matrix(session)

# %%
