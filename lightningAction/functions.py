import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import mode
import scipy.interpolate as interpolate


"""
SCRIPT: Syllables per trial epoch
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


""" State post-processing """
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


def state_identifiability_old(combined_states, design_matrix_heading, use_sets):
    
    unique_states = np.unique(combined_states)
    new_states = unique_states.copy()

    # Create new mapping depending on empirical data for each state
    for v, var in enumerate(use_sets):
        zeros = [s[v] == '0' if s != 'nan' else False for s in combined_states]
        ones = [s[v] == '1' if s != 'nan' else False for s in combined_states]
        
        # For an empty variable, do not make changes (wavelet)
        if len(var) == 0:
            var_0 = np.nan
            var_1 = np.nan
        elif var == ['avg_wheel_vel']:
            var_0 = np.array(np.abs(design_matrix_heading[var]))[zeros]
            var_1 = np.array(np.abs(design_matrix_heading[var]))[ones]
        elif var == ['left_X', 'left_Y', 'right_X', 'right_Y']:
            var_0 = np.array(np.abs(np.diff(design_matrix_heading[var], axis=0)))[zeros[1:]]
            var_1 = np.array(np.abs(np.diff(design_matrix_heading[var], axis=0)))[ones[1:]]
        elif var == ['nose_x', 'nose_Y']:
            print('Not implemented yet')
        else:
            var_0 = np.array(design_matrix_heading[var])[zeros]
            var_1 = np.array(design_matrix_heading[var])[ones]
        
        if np.nanmean(var_0)> np.nanmean(var_1):
            var_state_0 = [s[v] == '0' if s != 'nan' else False for s in unique_states]
            new_states[var_state_0] = np.array([s[:v] + '1' + s[v+1:] for s in new_states[var_state_0]])
            var_state_1 = [s[v] == '1' if s != 'nan' else False for s in unique_states]
            new_states[var_state_1] = np.array([s[:v] + '0' + s[v+1:] for s in new_states[var_state_1]])

    identifiable_mapping = {unique: key for unique, key in zip(unique_states, new_states)}

    # Use np.vectorize to apply the mapping
    replace_func = np.vectorize(identifiable_mapping.get)
    identifiable_states = replace_func(combined_states)
    
    return identifiable_states


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
    use_data = reduced_design_matrix.dropna()
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
    states_trial_type['trial_type'] = states_trial_type[trial_type_agg].agg(' '.join, axis=1)
    states_trial_type['trial_str'] = states_trial_type['trial_id'].astype(str)
    states_trial_type['sample'] = states_trial_type[['session', 'trial_str']].agg(' '.join, axis=1)
    if 'ballistic' in states_trial_type.columns:
        states_trial_type.loc[states_trial_type['ballistic']==True, 'ballistic'] = 1
        states_trial_type.loc[states_trial_type['ballistic']==False, 'ballistic'] = 0
    return states_trial_type


def rescale_sequence(seq, target_length):
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
        return np.array([mode(b)[0] for b in bins])  # Extract mode from result

    else:
        # Stretching: Repeat values to fit new size
        stretched_indices = np.floor(np.linspace(0, original_length - 1, target_length)).astype(int)
        return np.array(seq)[stretched_indices]  # Map stretched indices to original values


def plot_binned_sequence(df_grouped, index, states_to_append, palette):
        title = df_grouped['broader_label'][index]
        fig, axs = plt.subplots(2, 1, sharex=False, sharey=True, figsize=(5, 2))
        axs[0].imshow(np.concatenate([df_grouped['sequence'][index], states_to_append])[None,:],  
                extent=(0, len(np.concatenate([df_grouped['sequence'][index], states_to_append])), 
                        0, 1),
                aspect="auto",
                cmap=palette,
                alpha=0.7) 
        axs[0].set_xlim([0, len(df_grouped['sequence'][index])])

        axs[1].imshow(np.concatenate([df_grouped['binned_sequence'][index], states_to_append])[None,:],  
                extent=(0, len(np.concatenate([df_grouped['binned_sequence'][index], states_to_append])), 
                        0, 1),
                aspect="auto",
                cmap=palette,
                alpha=0.7) 
        axs[1].set_xlim([0, len(df_grouped['binned_sequence'][index])])
        axs[0].set_title(title)
        plt.tight_layout()