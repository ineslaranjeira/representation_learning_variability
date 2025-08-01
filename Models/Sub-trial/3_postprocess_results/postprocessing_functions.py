"""
IMPORTS
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os 
from scipy.stats import mode

# # Custom functions
prefix = '/home/ines/repositories/'
# prefix = '/Users/ineslaranjeira/Documents/Repositories/'

functions_path = prefix + '/representation_learning_variability/Functions/'
os.chdir(functions_path)
from one_functions_generic import prepro


""" Model comparison """

def best_lag_kappa(all_lls, all_baseline_lls, design_matrix, num_train_batches, kappas, Lags, subtract_baseline=False):
    
    # Find best params per mouse
    best_lag = {}
    best_kappa = {}
    mean_bits_LL = {}
                  
    # Get size of folds
    num_timesteps = np.shape(design_matrix)[0]
    shortened_array = np.array(design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    fold_len =  len(shortened_array)/num_train_batches
    
    mean_bits_LL = np.ones((len(Lags), len(kappas))) * np.nan
    mean_LL = np.ones((len(Lags), len(kappas))) * np.nan
    best_fold = np.ones((len(Lags), len(kappas))) * np.nan
    
    for l, lag in enumerate(Lags):
        
        # Reshape
        lag_lls = []
        b_lag_lls = []
        b_fold = []
        
        for k in kappas:
            lag_lls.append(all_lls[lag][k])
            b_lag_lls.append(all_baseline_lls[lag][k])
            # Best fold
            if np.abs(np.nansum(all_lls[lag][k])) > 0:  # accounts for possibility that all folds are nan
                b_f = np.where(all_lls[lag][k]==np.nanmax(all_lls[lag][k]))[0][0]
            else:
                b_f = np.nan
            b_fold.append(b_f)
            
        avg_val_lls = np.array(lag_lls)
        baseline_lls = np.array(b_lag_lls)
        bits_LL = (np.array(avg_val_lls) - np.array(baseline_lls)) / fold_len * np.log(2)
        
        mean_bits_LL[l,:] = np.nanmean(bits_LL, axis=1)        
        mean_LL[l,:] = np.nanmean(avg_val_lls, axis=1)     
        best_fold[l, :] = b_fold
        
    # Save best params for the mouse
    best_lag = Lags[np.where(mean_bits_LL==np.nanmax(mean_bits_LL))[0][0]]
    if subtract_baseline:
        best_kappa = kappas[np.where(mean_bits_LL==np.nanmax(mean_bits_LL))[1][0]]
    else:
        best_kappa = kappas[np.where(mean_LL==np.nanmax(mean_LL))[1][0]]
    
    return best_lag, best_kappa, mean_bits_LL, mean_LL, best_fold


def best_lag_kappa_bic(all_lls, all_baseline_lls, design_matrix, num_train_batches, kappas, Lags, num_states):
    
    # Find best params per mouse
    best_lag = {}
    best_kappa = {}
    mean_bits_LL = {}
                  
    # Get size of folds
    num_timesteps = np.shape(design_matrix)[0]
    shortened_array = np.array(design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    fold_len =  len(shortened_array)/num_train_batches
    
    mean_bits_LL = np.ones((len(Lags), len(kappas))) * np.nan
    mean_LL = np.ones((len(Lags), len(kappas))) * np.nan
    all_bic = np.ones((len(Lags), len(kappas))) * np.nan
    best_fold = np.ones((len(Lags), len(kappas))) * np.nan
        
    for l, lag in enumerate(Lags):
        
        num_params = 3*num_states + num_states**2 + num_states * lag
        
        # Reshape
        lag_lls = []
        b_lag_lls = []
        b_fold = []
        
        for k in kappas:
            lag_lls.append(all_lls[lag][k])
            b_lag_lls.append(all_baseline_lls[lag][k])
            # Best fold
            if np.abs(np.nansum(all_lls[lag][k])) > 0:  # accounts for possibility that all folds are nan
                b_f = np.where(all_lls[lag][k]==np.nanmax(all_lls[lag][k]))[0][0]
            else:
                b_f = np.nan
            b_fold.append(b_f)
            
        avg_val_lls = np.array(lag_lls)
        baseline_lls = np.array(b_lag_lls)
        bits_LL = (np.array(avg_val_lls) - np.array(baseline_lls)) / fold_len * np.log(2)
        bic = num_params * np.log(fold_len*num_train_batches) - 2 * np.nansum(avg_val_lls, axis=1)
        bic = num_params * 2 - 2 * np.nansum(avg_val_lls, axis=1)

        mean_bits_LL[l,:] = np.nanmean(bits_LL, axis=1)        
        mean_LL[l,:] = np.nanmean(avg_val_lls, axis=1)  
        all_bic[l,:] = bic
        best_fold[l, :] = b_fold
        
    # Save best params for the mouse
    best_kappa = kappas[np.where(all_bic==np.nanmin(all_bic))[1][0]]
    best_lag = Lags[np.where(all_bic==np.nanmin(all_bic))[0][0]]
    
    return best_lag, best_kappa, mean_bits_LL, mean_LL, all_bic, best_fold


def best__kappa(all_lls, all_baseline_lls, design_matrix, num_train_batches, kappas, subtract_baseline=False):
    
    # Find best params per mouse
    best_kappa = {}
    mean_bits_LL = {}
                  
    # Get size of folds
    num_timesteps = np.shape(design_matrix)[0]
    shortened_array = np.array(design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    fold_len =  len(shortened_array)/num_train_batches

    # Initialize
    lls = []
    b_lls = []
    b_fold = []
        
    for k in kappas:
        lls.append(all_lls[k])
        b_lls.append(all_baseline_lls[k])
        # Best fold
        if np.abs(np.nansum(all_lls[k])) > 0:  # accounts for possibility that all folds are nan
            b_f = np.where(all_lls[k]==np.nanmax(all_lls[k]))[0][0]
        else:
            b_f = np.nan
            
        b_fold.append(b_f)

    avg_val_lls = np.array(lls)
    baseline_lls = np.array(b_lls)
    bits_LL = (np.array(avg_val_lls) - np.array(baseline_lls)) / fold_len * np.log(2)
    mean_bits_LL = np.nanmean(bits_LL, axis=1)       
    mean_LL = np.nanmean(lls, axis=1) 
    best_fold = b_fold
    
    if subtract_baseline:  
        # Save best params for the mouse
        best_kappa = kappas[np.where(mean_bits_LL==np.nanmax(mean_bits_LL))[0][0]]
    else:
        best_kappa = kappas[np.where(mean_LL==np.nanmax(mean_LL))[0][0]]
    
    return best_kappa, mean_bits_LL, mean_LL, best_fold


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


""" State post-processing """

def remove_states_str(most_likely_states, threshold):
    
    most_likely_states = np.array([''.join(map(str, row)) for row in most_likely_states])
    # Remove states below threshold
    unique, counts = np.unique(most_likely_states, return_counts=True)
    threshold_count = threshold * len(most_likely_states)
    excluded_bins = 0
    remaining_states = list(counts.copy())  # how many repetitions per state
    # Go through all states
    for state in unique:
        # Get least represented state
        size_smallest_state = np.nanmin(remaining_states)
        # Check if below threshold
        if size_smallest_state + excluded_bins < threshold_count:
            remaining_states[np.where(counts==size_smallest_state)[0][0]] = np.nan
            excluded_bins += size_smallest_state
    

    # which_states = np.array(most_likely_states).astype(float)
    # Find states tagged to remove
    exclude_states_idx = np.where(np.isnan(np.array(remaining_states)))[0].astype(int)
    exclude_states = unique[exclude_states_idx]
    # Create a boolean mask to identify values to replace
    mask = np.isin(most_likely_states, exclude_states)
    # Replace values in main_array with np.nan using the boolean mask
    new_states = most_likely_states
    new_states[mask] = np.nan
    
    return new_states


def remove_states_flt(most_likely_states, threshold):
    
    unique, counts = np.unique(most_likely_states, return_counts=True)
    threshold_count = threshold * len(most_likely_states)
    excluded_bins = 0
    remaining_states = list(counts.copy())
    for state in unique:
        size_smallest_state = np.nanmin(remaining_states)
        if size_smallest_state + excluded_bins < threshold_count:
            remaining_states[np.where(counts==size_smallest_state)[0][0]] = np.nan
            excluded_bins += size_smallest_state
    
    # Remove states below threshold
    new_states = np.array(most_likely_states).astype(float)
    exclude_states = np.where(np.isnan(np.array(remaining_states)))[0].astype(float)
    # Create a boolean mask to identify values to replace
    mask = np.isin(new_states, exclude_states)
    # Replace values in main_array with np.nan using the boolean mask
    new_states[mask] = np.nan

    
    return new_states



def state_identifiability(combined_states, design_matrix_heading, use_sets):
    
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


def identifiable_states_w_wheel(extended_states):
    
    """ Adds human-readable info from wheel heuristic-based states into identifiable states variable """
    
    extended_states['ori_states'] = extended_states['identifiable_states'].copy()

    # New wheel states
    # Wheel movement, no response
    move_no_resp = extended_states.loc[(extended_states['virtual_response'].isna()) &
                                    (extended_states['identifiable_states'].str[0]=='1'), 'identifiable_states']
    extended_states.loc[(extended_states['virtual_response'].isna()) &
                        (extended_states['identifiable_states'].str[0]=='1'), 
                        'identifiable_states'] = 'n' + move_no_resp.str[1:]
    # Response L, non ballistic
    left_non_ballistic = extended_states.loc[(extended_states['virtual_response']==1.) &
                                            (extended_states['ballistic']==False) &
                                            (extended_states['identifiable_states'].str[0]=='1'), 'identifiable_states']
    extended_states.loc[(extended_states['virtual_response']==1.) &
                        (extended_states['ballistic']==False) &
                        (extended_states['identifiable_states'].str[0]=='1'), 
                        'identifiable_states'] = 'l' + left_non_ballistic.str[1:]
    # Response R, non ballistic
    right_non_ballistic = extended_states.loc[(extended_states['virtual_response']==-1.) &
                                            (extended_states['ballistic']==False) &
                                            (extended_states['identifiable_states'].str[0]=='1'), 'identifiable_states']
    extended_states.loc[(extended_states['virtual_response']==-1.) &
                        (extended_states['ballistic']==False) &
                        (extended_states['identifiable_states'].str[0]=='1'), 
                        'identifiable_states'] = 'r' + right_non_ballistic.str[1:]

    # Response L, ballistic
    left_ballistic = extended_states.loc[(extended_states['virtual_response']==1.) &
                                        (extended_states['ballistic']==True) &
                                        (extended_states['identifiable_states'].str[0]=='1'), 'identifiable_states']
    extended_states.loc[(extended_states['virtual_response']==1.) &
                        (extended_states['ballistic']==True) &
                        (extended_states['identifiable_states'].str[0]=='1'), 
                        'identifiable_states'] = 'L' + left_ballistic.str[1:]

    # Response R, non ballistic
    right_ballistic = extended_states.loc[(extended_states['virtual_response']==-1.) &
                                        (extended_states['ballistic']==True) &
                                        (extended_states['identifiable_states'].str[0]=='1'), 'identifiable_states']
    extended_states.loc[(extended_states['virtual_response']==-1.) &
                        (extended_states['ballistic']==True) &
                        (extended_states['identifiable_states'].str[0]=='1'), 
                        'identifiable_states'] = 'R' + right_ballistic.str[1:]

    return extended_states


def get_index(var_timeseries, target_value):

    # Assume we find an instance where x is close to 5
    index = (np.abs(var_timeseries - target_value)).argmin()

    return index


def update_var(target_value, var_timeseries, var_name, mouse_dynamics, state):
    
    weights = mouse_dynamics[var_name]['weights'][state][0]
    lag_num = np.shape(mouse_dynamics[var_name]['weights'][state])[1]

    # Start with input variable value
    updated_var = 0
    closest_to_target = get_index(var_timeseries, target_value)
    # Loop through lags to complete update
    for l, lag in enumerate(range(lag_num)):
        # Weight for that lag
        lag_weight = weights[l]  
        # Get variable with corresponding lag
        if l == 0:
            use_var = target_value
        else:
            use_var = np.array(var_timeseries)[closest_to_target-l]
        updated_var += use_var * lag_weight
        
    return updated_var


def transition_probabilities(states, unique_states):
    
    transition_matrix = np.zeros((len(unique_states), len(unique_states))) * np.nan
    previous = states[:-1]
    current = states[1:]
    for s, st in enumerate(unique_states):
        for s_p, s_prev in enumerate(unique_states):
            interest_current = np.where(current==st)[0]
            interest_prev = np.where(previous==s_prev)[0]
            joint = len(np.intersect1d(interest_current, interest_prev))/ len(previous)
            marginal = len(interest_prev) / len(previous)
            transition_matrix[s_p, s] = joint / marginal if marginal > 0 else np.nan
        
    return transition_matrix


## This function is now unnecessary
def trans_mat_complete(mapping, state_label, unique_states, transition_matrix, states_template):
    matrix_df = np.zeros((len(states_template), len(states_template))) * np.nan
    for r, row in enumerate(states_template):
        for c, column in enumerate(states_template):
            if (row in state_label) & (column in state_label):
                state_c = mapping[column]
                state_c_mat = np.where(unique_states==state_c)
                state_r = mapping[row]
                state_r_mat = np.where(unique_states==state_r)
                # if (state_c in unique_states) & (state_r in unique_states):
                matrix_df[r, c] = transition_matrix[state_r_mat, state_c_mat]
    
    return matrix_df


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

def params_to_df (var_names, num_states, num_train_batches, fit_params, norm_results):

    learned_params = pd.DataFrame(columns=['fold', 'state', 'variable'])  # , index=np.arange(0, (num_states*num_train_batches), 1)

    for v, var in enumerate(var_names):
        for s, state in enumerate(range(num_states)):
            df = pd.DataFrame(columns=['fold', 'state', 'variable'])
            values = fit_params[2].means[:].T[v][s, :]
            
            # Normalize results 
            if norm_results == True:
                # Calculate min and max values
                min_value = np.min(values)
                max_value = np.max(values)
                # Normalize
                values = (values - min_value) / (max_value - min_value) * 2 - 1
                
            df['value'] = values
            df['fold'] = np.arange(1, num_train_batches+1, 1)
            df['variable'] = var
            df['state'] = state
            learned_params = learned_params.append(df)
            
    return learned_params


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
    iti_end = session_trials['intervals_1']
    
    # iti_end = np.concatenate((session_trials['intervals_0'][1:], 
    #                           np.array(session_trials['intervals_1'][-1:])), axis=0) 
    
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
        use_data.loc[(use_data['Bin'] <= iti_end[t]*multiplier) & 
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

# THIS function was to control for difference in ITI length for correct and incorrect
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
    # iti_end = session_trials['feedback_times'] + 1.5
    
    # iti_end = np.concatenate((session_trials['intervals_0'][1:], 
    #                           np.array(session_trials['intervals_1'][-1:])), axis=0) 
    
    # Reaction time 
    rt_init = session_trials['goCueTrigger_times']
    rt_end = session_trials['firstMovement_times']

    # Movement time 
    move_init = session_trials['firstMovement_times']
    move_end = session_trials['feedback_times']
    

    for t in range(trial_num):
        
        # # Pre-quiescence
        # use_data.loc[(use_data['Bin']+0.5 <= pre_qui_end[t]*multiplier) & 
        #              (use_data['Bin']+0.5 > pre_qui_init[t]*multiplier), 'label'] = 'Pre-quiescence'

        # # Quiescence
        # use_data.loc[(use_data['Bin']+0.5 <= qui_end[t]*multiplier) &
        #              (use_data['Bin']+0.5 > qui_init[t]*multiplier), 'label'] = 'Quiescence'
        
        # # ITI
        # if session_trials['feedbackType'][t] == -1.:
        #     use_data.loc[(use_data['Bin']+0.5 <= iti_end_incorrect[t]*multiplier) & 
        #                     (use_data['Bin']+0.5 > iti_init[t]*multiplier), 'label'] = 'ITI'
        # elif session_trials['feedbackType'][t] == 1.:
        #     use_data.loc[(use_data['Bin']+0.5 <= iti_end_correct[t]*multiplier) & 
        #                     (use_data['Bin']+0.5 > iti_init[t]*multiplier), 'label'] = 'ITI'
        # # Move
        # if session_trials['choice'][t] == -1:
        #     use_data.loc[(use_data['Bin']+0.5 <= move_end[t]*multiplier) & 
        #                  (use_data['Bin']+0.5 > move_init[t]*multiplier), 'label'] = 'Left choice'
        # elif session_trials['choice'][t] == 1.:
        #     use_data.loc[(use_data['Bin']+0.5 <= move_end[t]*multiplier) & 
        #                  (use_data['Bin']+0.5 > move_init[t]*multiplier), 'label'] = 'Right choice'
            
        # # React        
        # if prepro(session_trials)['signed_contrast'][t] < 0:
        #     use_data.loc[(use_data['Bin']+0.5 <= rt_end[t]*multiplier) & 
        #                  (use_data['Bin']+0.5 > rt_init[t]*multiplier), 'label'] = 'Stimulus left'
        # elif prepro(session_trials)['signed_contrast'][t] > 0:
        #     use_data.loc[(use_data['Bin']+0.5 <= rt_end[t]*multiplier) & 
        #                  (use_data['Bin']+0.5 > rt_init[t]*multiplier), 'label'] = 'Stimulus right'
    
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


# Clustering analyses

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

def state_relative_frequency(use_data):
    
    vars = ['sample', 'trial_type', 'broader_label', 'mouse_name', 'identifiable_states']

    # Step 1: Group and count occurrences
    count = pd.DataFrame(use_data.groupby(vars)['identifiable_states'].count())  #  'correct',

    # Step 2: Reset index to bring the grouping columns back into the DataFrame
    count = count.reset_index(level=list(np.arange(0, len(vars)-1, 1)))

    # Rename the count column
    count.rename(columns={'identifiable_states': 'count'}, inplace=True)

    # Step 3: Calculate the total counts for each group of 'mouse_name', 'session', 'label'
    count['total'] = count.groupby(vars[:-1])['count'].transform('sum')  # 'broader_label'

    # Step 4: Compute the relative frequency
    count['relative_frequency'] = count['count'] / count['total']

    # Drop the 'total' column if it's no longer needed
    count = count.drop(columns=['total'])
    
    # Pivot the DataFrame
    freq_df = count.reset_index().pivot(index=['sample', 'trial_type', 'mouse_name'], columns=['identifiable_states', 'broader_label'], values='count')
    # To flatten the column MultiIndex
    freq_df.columns = ['_'.join(col).strip() for col in freq_df.columns.values]
    freq_df[freq_df.isna()] = 0

    return count, freq_df


def trial_relative_frequency(use_data, vars):

    # Step 1: Group and count occurrences
    count = pd.DataFrame(use_data.groupby(vars)['cluster'].count())  #  'correct',

    # Step 2: Reset index to bring the grouping columns back into the DataFrame
    count = count.reset_index(level=list(np.arange(0, len(vars)-1, 1)))

    # Rename the count column
    count.rename(columns={'cluster': 'count'}, inplace=True)

    # Step 3: Calculate the total counts for each group of 'mouse_name', 'session', 'label'
    count['total'] = count.groupby(vars[:-1])['count'].transform('sum')  # 'broader_label'

    # Step 4: Compute the relative frequency
    count['relative_frequency'] = count['count'] / count['total']

    # Drop the 'total' column if it's no longer needed
    count = count.drop(columns=['total'])

    # Pivot the DataFrame
    # freq_df = count.reset_index().pivot(index=['mouse_name'], columns=['cluster'], values='relative_frequency')
    freq_df = count.reset_index().pivot(index=vars[:-1], columns=['cluster'], values='relative_frequency')
    freq_df[freq_df.isna()] = 0
    
    return count, freq_df


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
        return np.array([mode(b)[0][0] for b in bins])  # Extract mode from result

    else:
        # Stretching: Repeat values to fit new size
        stretched_indices = np.floor(np.linspace(0, original_length - 1, target_length)).astype(int)
        return np.array(seq)[stretched_indices]  # Map stretched indices to original values


def plot_cm(decoding_result, trial_epochs, control=False):
    """
    PLOT RESULTS
    """

    # -- Confusion Matrix
    # labels = trial_epochs

    # Results on original model
    plt.rc('font', size=12) 
    plt.figure(figsize=[6.2, 5.2])
    hmap = sns.color_palette("mako", as_cmap=True)
    data = decoding_result.loc[decoding_result['shuffle'] == 0]
    sns.heatmap(data['confusion_matrix'].mean(), annot=True, square=True,
        yticklabels=trial_epochs, xticklabels=trial_epochs, 
        cmap= hmap, vmin=0, vmax=1, fmt=".2f") 

    plt.xticks([.5, 1.5, 2.5, 3.5], trial_epochs)
    plt.yticks([.5, 1.5, 2.5, 3.5], trial_epochs)
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('full_cm.svg',dpi=500)
    plt.show()
    print('F1 results', data['f1'].mean())

    if control:
        # Results from shuffled model
        plt.rc('font', size=12) 
        plt.figure(figsize=[6.2, 5.2])
        data = decoding_result.loc[decoding_result['shuffle'] >0]
        sns.heatmap(data['confusion_matrix'].mean(), annot=True, square=True,
            yticklabels=trial_epochs, xticklabels=trial_epochs, 
            cmap= hmap, vmin=0, vmax=1, fmt=".2f")

        plt.xticks([.5, 1.5, 2.5, 3.5], trial_epochs)
        plt.yticks([.5, 1.5, 2.5, 3.5], trial_epochs)
        plt.xticks(rotation = 90)
        plt.yticks(rotation = 0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()
        print('F1 shuffled results',  data['f1'].mean())


def plot_f1(decoding_result):
    # -- F1 score per model, original and shuffled
    data = decoding_result.copy()
    data['f1'] = data['f1'].astype(float)

    data.loc[data['shuffle'] >= 1, 'shuffle'] = 'Shuffled'
    data.loc[data['shuffle'] == 0, 'shuffle'] = 'Original'
    data = data.rename(columns={'shuffle': 'Dataset'})

    plt.rc('font', size=12) 
    plt.figure(figsize=[4.5, 4])
    sns.boxplot(y='f1', x='Dataset', data=data, color='grey') 
    sns.swarmplot(y='f1', x='Dataset', data=data, color='black', dodge=True, alpha=0.7, size=3)
    plt.ylim([0,1])
    plt.ylabel('Accuracy score (F1)')
    plt.legend(bbox_to_anchor=(1.05, 1))
    sns.despine(top=True, right=True)
    #plt.savefig('violin.svg',dpi=500)
    

def decoder(mat, var, model, mapping, trial_epochs, shufflings, plot = False):
    
    """
    RUN MODEL
    """

    # Generate random states for each iteration with a fixed seed
    # Loop over iterations of random draws of mice
    # Create empty dataframes to save results
    decoding_result = pd.DataFrame(columns=['shuffle', 'f1', 'confusion_matrix'])

    # Decoding function with 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    use_mat = mat.copy() # test using all mice

    # Find minimum number of samples per label
    original_labels = np.array(use_mat[var])
    # Use np.vectorize to apply the mapping
    replace_func = np.vectorize(mapping.get)
    labels = replace_func(original_labels)
    labels_nr = np.arange(labels.shape[0])
    min_freq = np.min(use_mat['y'].value_counts())

    # Randomly select N mice from each lab to equalize classes
    use_index = np.empty(0, dtype=int)
    for j, epoch in enumerate(np.unique(labels)):
        use_index = np.concatenate([use_index, np.random.choice(labels_nr[labels == epoch],
                                                                min_freq, replace=False)])

    new_mat = use_mat.iloc[use_index].reset_index().drop(columns=['index']).copy()
            
    # -- ORIGINAL DATASET

    dec_result = pd.DataFrame(columns=['shuffle', 'f1', 'confusion_matrix'])
    y_pred = np.zeros(len(new_mat), dtype=int) 
    exog = new_mat[new_mat.columns.difference([var])]
    endog = new_mat.loc[:, var].copy()
    endog[:] = replace_func(endog)

    for train_index, test_index in kf.split(new_mat):
        model.fit(exog.iloc[train_index], endog.iloc[train_index].astype(int))
        y_pred[test_index] = model.predict(exog.iloc[test_index])

    # Calculate f1 score and confusion matrix
    f1 = f1_score(endog.astype(int), y_pred.astype('int'), average='micro')
    cm = confusion_matrix(endog.astype(int), y_pred.astype('int'), normalize='true')

    # Save results
    dec_result.loc[0, 'f1'] = f1
    dec_result.loc[0, 'confusion_matrix'] = cm
    dec_result.loc[0, 'shuffle'] = 0

    decoding_result = pd.concat([decoding_result, dec_result])

    # -- SHUFFLED DATASET 
    shuffle_result = pd.DataFrame(columns=['shuffle', 'f1', 'confusion_matrix'])
    for s in range(shufflings):
        if np.mod(s+1, 10) == 0:
            print('Shuffling %d of %d' % (s+1, shufflings))   

        shuffle_y_pred = np.zeros(len(new_mat), dtype=int) 
        shuffle_endog = endog.copy()
        np.random.shuffle(shuffle_endog.values)

        for train_index, test_index in kf.split(new_mat):
            model.fit(exog.iloc[train_index], list(shuffle_endog.iloc[train_index].astype(int)))
            shuffle_y_pred[test_index] = model.predict(exog.iloc[test_index])   

        shuffle_f1 = f1_score(shuffle_endog.astype(int), shuffle_y_pred.astype('int'), average='micro')
        shuffle_cm = confusion_matrix(shuffle_endog.astype(int), shuffle_y_pred.astype('int'), normalize='true')

        # SAVE
        shuffle_result.loc[s, 'f1'] = shuffle_f1
        shuffle_result.loc[s, 'confusion_matrix'] = shuffle_cm
        shuffle_result.loc[s, 'shuffle'] = s + 1

    decoding_result = pd.concat([decoding_result, shuffle_result])

    if plot:
        # Plot
        plot_cm(decoding_result, trial_epochs)
        plot_f1(decoding_result)
        
    return decoding_result