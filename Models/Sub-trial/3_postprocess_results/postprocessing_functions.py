"""
IMPORTS
"""
import autograd.numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, Normalizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


""" Model comparison """

def best_lag_kappa(all_lls, all_baseline_lls, design_matrix, num_train_batches, kappas, Lags):
    
    # Find best params per mouse
    best_lag = {}
    best_kappa = {}
    mean_bits_LL = {}
                  
    # Get size of folds
    num_timesteps = np.shape(design_matrix)[0]
    shortened_array = np.array(design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    fold_len =  len(shortened_array)/num_train_batches
    
    mean_bits_LL = np.ones((len(Lags), len(kappas))) * np.nan
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
        best_fold[l, :] = b_fold
        
    # Save best params for the mouse
    best_lag = Lags[np.where(mean_bits_LL==np.nanmax(mean_bits_LL))[0][0]]
    best_kappa = kappas[np.where(mean_bits_LL==np.nanmax(mean_bits_LL))[1][0]]
    
    return best_lag, best_kappa, mean_bits_LL, best_fold


def best__kappa(all_lls, all_baseline_lls, design_matrix, num_train_batches, kappas):
    
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
    best_fold = b_fold
    
    # Save best params for the mouse
    best_kappa = kappas[np.where(mean_bits_LL==np.nanmax(mean_bits_LL))[0][0]]
    
    return best_kappa, mean_bits_LL, best_fold


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
    plt.xlabel('Kappa')
    plt.ylabel('Lag')
    plt.title(mouse_name + ' ' + var_interest)
    # Display the plot
    plt.show()


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
            transition_matrix[s_p, s] = joint / marginal
        
    return transition_matrix


def trans_mat_complete(mapping, state_label, unique_states, transition_matrix):
    states_template = ['000', '001', '010', '100', '110', '101', '011', '111']
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


def state_identifiability(combined_states, design_matrix_heading):
    
    unique_states = np.unique(combined_states)
    new_states = unique_states.copy()
    
    # Create new mapping depending on empirical data for each state
    for v, var in enumerate(['avg_wheel_vel', 'Lick count', 'whisker_me']):
        zeros = [s[v] == '0' for s in combined_states]
        ones = [s[v] == '1' for s in combined_states]
        if var == 'avg_wheel_vel':
            var_0 = np.array(np.abs(design_matrix_heading[var]))[zeros]
            var_1 = np.array(np.abs(design_matrix_heading[var]))[ones]
        else:
            var_0 = np.array(design_matrix_heading[var])[zeros]
            var_1 = np.array(design_matrix_heading[var])[ones]
        
        if np.nanmean(var_0)> np.nanmean(var_1):
            var_state_0 = [s[v] == '0' for s in unique_states]
            new_states[var_state_0] = np.array([s[:v] + '1' + s[v+1:] for s in new_states[var_state_0]])
            var_state_1 = [s[v] == '1' for s in unique_states]
            new_states[var_state_1] = np.array([s[:v] + '0' + s[v+1:] for s in new_states[var_state_1]])
    

    identifiable_mapping = {unique: key for unique, key in zip(unique_states, new_states)}

    # Use np.vectorize to apply the mapping
    replace_func = np.vectorize(identifiable_mapping.get)
    identifiable_states = replace_func(combined_states)
    
    return identifiable_states