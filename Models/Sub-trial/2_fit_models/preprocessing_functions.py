""" 
IMPORTS
"""
import autograd.numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, Normalizer


""" LOADING DESIGN MATRIX """


def get_frame_rate(eid, mouse_name, frame_rate):

    fr = int(frame_rate.loc[(frame_rate['subject_nickname']==mouse_name) & (frame_rate['session_uuid']==eid), 'frame_rate'])
    return fr


def idxs_from_files(one, design_matrices, frame_rate, data_path, bin_size):
    
    idxs = []
    mouse_names = []
    for m, mat in enumerate(design_matrices):
        
        mouse_name = design_matrices[m][51:-4]
        eid = design_matrices[m][14:50]
        idx = str(eid + '_' + mouse_name)
                    
        # Check frame rate
        fr = get_frame_rate(eid, mouse_name, frame_rate)
        
        if np.abs(fr-60)<2:
            
            # Check licks per trial
            session_trials = one.load_object(eid, obj='trials', namespace='ibl')
            session_trials = session_trials.to_df()
            filename = str(data_path + mat)  # + mouse_name + '_'
            design_matrix = pickle.load(open(filename, "rb"))
            
            # Require at least one lick per trial (this is an estimation, not very principled)
            if np.sum(design_matrix['Lick count']) > len(session_trials):
            
                if len(idxs) == 0:
                    idxs = idx
                    mouse_names = mouse_name
                else:
                    idxs = np.hstack((idxs, idx))
                    mouse_names = np.hstack((mouse_names, mouse_name))
            
    return idxs, mouse_names


def prepro_design_matrix(one, idxs, mouse_names, bin_size, var_names, data_path, first_90=True):

    # Save data of all sessions for latter
    matrix_all = defaultdict(list)
    matrix_all_unnorm = defaultdict(list)
    session_all = defaultdict(list)

    for m, mouse_name in enumerate(mouse_names):
        # Save results per mouse
        matrix_all[mouse_name] = {}
        session_all[mouse_name] = {}
        matrix_all_unnorm[mouse_name] = {}

    for m, mat in enumerate(idxs):
        if len(mat) > 35: 
                
            # Trials data
            session = mat[0:36]
            mouse_name = mat[37:]
            
            session_trials = one.load_object(session, obj='trials', namespace='ibl')
            session_trials = session_trials.to_df()
            session_end = list(session_trials['stimOff_times'][-1:])[0]  # TODO: this might not work if stimOff times are missing
            session_start = list(session_trials['stimOn_times'])[0]

            # Get time of last unbiased trial
            unbiased = session_trials.loc[session_trials['probabilityLeft']==0.5]
            time_trial_90 = list(unbiased['stimOff_times'])[-1]
            
            filename = str(data_path +'design_matrix_' + mat + '_'  + str(bin_size))  # + mouse_name + '_'
            big_design_matrix = pickle.load(open(filename, "rb"))
            design_matrix = big_design_matrix.groupby('Bin')[var_names].mean()  # 
            design_matrix = design_matrix.reset_index(level = [0])  # , 'Onset times'
            design_matrix = design_matrix.dropna()

            if first_90 == True:
                # Keep only first 90 trials
                design_matrix = design_matrix.loc[(design_matrix['Bin'] < time_trial_90 * 10) & (design_matrix['Bin'] > session_start * 10)]
                use_trials = session_trials.loc[session_trials['stimOff_times'] < time_trial_90]
            else:
                design_matrix = design_matrix.loc[design_matrix['Bin'] > session_start * 10]
                use_trials = session_trials.copy()
                
            training_set = np.array(design_matrix[var_names]).copy() 
            
            if (len(training_set) > 0) & (np.sum(big_design_matrix['Lick count']) > len(use_trials)):
                # Standardization using StandardScaler
                scaler = StandardScaler()
                standardized = scaler.fit_transform(training_set)
                # Normalize between 0 and 1
                normalizer = Normalizer()
                normalized = normalizer.fit_transform(standardized)
                
                if len(var_names)>1:
                    matrix_all[mouse_name][session] = normalized
                else:
                    matrix_all[mouse_name][session] = standardized
                if 'Lick count' in var_names:
                    matrix_all[mouse_name][session] = training_set
                
                session_all[mouse_name][session] = use_trials    
                matrix_all_unnorm[mouse_name][session] = design_matrix
                
            else:
                print(session)
        else:
            print(mat)
        
    return matrix_all, matrix_all_unnorm, session_all


def concatenate_sessions (mouse_names, matrix_all, matrix_all_unnorm, session_all):
    
    collapsed_matrices = defaultdict(list)
    collapsed_unnorm = defaultdict(list)
    collapsed_trials = defaultdict(list)

    # Collapse multiple sessions per mouse
    for mouse in np.unique(mouse_names):
        if len(np.where(mouse_names==mouse)[0]) > 1 and len(mouse) > 0:
            mouse_sessions = list(matrix_all[mouse].keys())
            collapsed_matrices[mouse] = np.concatenate([matrix_all[mouse][k] for k in mouse_sessions])
            collapsed_unnorm[mouse] = pd.concat(matrix_all_unnorm[mouse], ignore_index=True)
            collapsed_trials[mouse] = pd.concat(session_all[mouse], ignore_index=True)
            
    return collapsed_matrices, collapsed_unnorm, collapsed_trials


def fix_discontinuities(session_trials, design_matrix_heading):
    
    cum_timing_vars = ['intervals_bpod_0', 'intervals_bpod_1', 'stimOn_times',
                       'goCueTrigger_times', 'stimOff_times', 'goCue_times', 'response_times',
                       'feedback_times', 'firstMovement_times', 'intervals_0', 'intervals_1']
    
    time_discs = np.where(np.diff(np.array(session_trials[cum_timing_vars[0]]))<0)[0]
    bin_discs = np.where(np.diff(np.array(design_matrix_heading['Bin']))<0)[0]
    
    # Fix as many times as there are discontinuities
    for d in range(len(time_discs)):
        time_disc = time_discs[d]
        bin_disc = bin_discs[d]
        
        trial_reset_time = session_trials['intervals_bpod_0'][time_disc]
        
        # Fix timing variables
        for v in cum_timing_vars:
            session_trials[v][time_disc+1:] = session_trials[v][time_disc+1:] + session_trials['intervals_bpod_0'][time_disc]
            
        # Fix bin count
        design_matrix_heading['Bin'][bin_disc+1:] = design_matrix_heading['Bin'][bin_disc+1:] + trial_reset_time * 10
        
        
    return session_trials, design_matrix_heading