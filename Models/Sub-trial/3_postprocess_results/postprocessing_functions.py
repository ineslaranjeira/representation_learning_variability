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
            b_f = np.where(all_lls[lag][k]==np.nanmax(all_lls[lag][k]))[0][0]
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
        b_f = np.where(all_lls[k]==np.nanmax(all_lls[k]))[0][0]
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
    plt.xlabel('Kappa')
    plt.ylabel('Lag')
    plt.title(mouse_name + ' ' + var_interest)
    # Display the plot
    plt.show()
    

""" State dynamics  - should move these to plotting functions!!"""

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


# def plot_trajectories(new_states, inverted_mapping, design_matrix_heading, x_var, y_var, movement_wheel, movement_whisker, axs, trajectory_num):
    
#     state_switches = np.diff(new_states)
#     switch_idxs = np.where(np.abs(state_switches)>0)[0][0:-1] 
#     switch_types = np.unique(new_states)
#     switch_types = switch_types[~np.isnan(switch_types)]
#     switch_type_idx = new_states[switch_idxs+1]

#     for t, type in enumerate(switch_types):
#         switch_interest = switch_idxs[np.where(switch_type_idx==type)]
#         switch_plot = np.random.choice(switch_interest, trajectory_num)
        
#         # Loop through switche types
#         for s, switch in enumerate(switch_plot):
#             trajectory_end = switch_idxs[switch_idxs>switch][0]
            
#             # Check if trajectory is long enough, if not, draw new switch
#             while (trajectory_end - switch) < 10:
#                 switch = np.random.choice(switch_interest, 1)[0]
#                 # print(switch, switch_idxs, switch_idxs[switch_idxs>switch])
#                 trajectory_end = switch_idxs[switch_idxs>switch][0]
            
#             current_state = new_states[switch+1]
#             state_label = inverted_mapping[current_state]
            
#             state_wheel = int(state_label[0])
#             state_whisker = int(state_label[2])

#             # flip so that state 0 is no movement and state 1 is movement
#             if movement_wheel == 0:
#                 state_wheel = -(state_wheel-1)
#             if movement_whisker == 0:
#                 state_whisker = -(state_whisker-1)
            
#             ax = axs[state_wheel, state_whisker]

#             color = sns.color_palette("viridis", len(switch_types))[int(current_state)]
#             label = state_label
#             xx = design_matrix_heading[x_var][switch+1:trajectory_end]
#             yy = design_matrix_heading[y_var][switch+1:trajectory_end]

#             if s == trajectory_num -1:
#                 ax.plot(xx, yy, alpha=0.5, color=color, label=label)
#                 ax.legend()
#             else:
#                 ax.plot(xx, yy, alpha=0.5, color=color)

#             ax.set_xlabel('Wheel velocity - state ' + str(state_wheel))
#             ax.set_ylabel('Whisker motion energy - state ' + str(state_whisker))


def plot_trajectories(new_states, inverted_mapping, design_matrix_heading, x_var, y_var, axs, trajectory_num):

    length_minimum = 3
    state_switches = np.diff(new_states)
    switch_idxs = np.where(np.abs(state_switches)>0)[0][0:-1] 
    switch_types = np.unique(new_states)
    switch_types = switch_types[~np.isnan(switch_types)]
    switch_type_idx = new_states[switch_idxs+1]

    for t, type in enumerate(switch_types):
        switch_interest = switch_idxs[np.where(switch_type_idx==type)]
        trajectory_length = np.zeros(len(switch_interest)) * np.nan
        # Check length of trajectories
        for s_t, switch_test in enumerate(switch_interest[:-1]):
            trajectory_length[s_t]= switch_idxs[switch_idxs>switch_test][0] - switch_test
        # Find long enough trajectories
        long_switches = switch_interest[np.where(trajectory_length> length_minimum)]
        if len(long_switches) >= trajectory_num:
            switch_plot = np.random.choice(long_switches, trajectory_num, replace=False)
        else:
            switch_plot = np.random.choice(long_switches, len(long_switches), replace=False)
        
        # Loop through switche types
        for s, switch in enumerate(switch_plot[:-1]):
            trajectory_end = switch_idxs[switch_idxs>switch][0]
            
            # Check if trajectory is long enough, if not, draw new switch
            # while (trajectory_end - switch) < 10:
            #     switch = np.random.choice(switch_interest, 1)[0]
            #     # print(switch, switch_idxs, switch_idxs[switch_idxs>switch])
            #     trajectory_end = switch_idxs[switch_idxs>switch][0]
            
            current_state = new_states[switch+1]
            state_label = inverted_mapping[current_state]
            
            state_wheel = int(state_label[0])
            state_whisker = int(state_label[2])

            # # flip so that state 0 is no movement and state 1 is movement
            # if movement_wheel == 1:
            #     state_wheel_ax = state_wheel
            # else:
            #     state_wheel_ax = -(state_wheel-1)
            # if movement_whisker == 1:
            #     state_whisker_ax = state_whisker
            # else:
            #     state_whisker_ax = -(state_whisker-1)
            state_wheel_ax = state_wheel
            state_whisker_ax = state_whisker
            ax = axs[state_wheel_ax, state_whisker_ax]

            color = sns.color_palette("viridis", len(switch_types))[int(current_state)]
            label = state_label
            xx = design_matrix_heading[x_var][switch+1:trajectory_end]
            yy = design_matrix_heading[y_var][switch+1:trajectory_end]

            if s == len(switch_plot) -2:
                ax.plot(xx, yy, alpha=0.5, color=color, label=label)
                ax.legend()
            else:
                ax.plot(xx, yy, alpha=0.5, color=color)

            ax.set_xlabel('Wheel velocity - state ' + str(state_wheel_ax))
            ax.set_ylabel('Whisker motion energy - state ' + str(state_whisker_ax))
            
            
def plot_x_y_dynamics(x_var, y_var, dynamics, mouse_name, new_states, design_matrix_heading, inverted_mapping, grid_density, trajectory_num, plot_traj=True):
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    unique_states = np.array(list(inverted_mapping.keys()))
    unique_states = unique_states[~np.isnan(unique_states)]
    mouse_dynamics = dynamics[mouse_name]
    
    # # Check dynamics to decide which state is movement and which state is not
    # dynamics_wheel_0 = mouse_dynamics[x_var]['weights'][0][0][0]
    # dynamics_wheel_1 = mouse_dynamics[x_var]['weights'][1][0][0]
    # dynamics_whisker_0 = mouse_dynamics[y_var]['weights'][0][0][0]
    # dynamics_whisker_1 = mouse_dynamics[y_var]['weights'][1][0][0]
    # if np.abs(dynamics_wheel_1) > np.abs(dynamics_wheel_0):
    #     movement_wheel = 1
    # else:
    #     movement_wheel = 0
    # if np.abs(dynamics_whisker_1) > np.abs(dynamics_whisker_0):
    #     movement_whisker = 1
    # else:
    #     movement_whisker = 0

    for s, state in enumerate(unique_states):
        
        x = np.linspace(np.min(design_matrix_heading[x_var]), 
                    np.max(design_matrix_heading[x_var]), grid_density)
        y = np.linspace(np.min(design_matrix_heading[y_var]), 
                        np.max(design_matrix_heading[y_var]), grid_density)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        where = np.where(np.array(list(inverted_mapping.keys())) == state)[0][0]
        state_wheel = int(inverted_mapping[where][0])
        state_whisker = int(inverted_mapping[where][2])

        # # flip so that state 0 is no movement and state 1 is movement
        # if movement_wheel == 1:
        #     state_wheel_ax = state_wheel
        # else:
        #     state_wheel_ax = -(state_wheel-1)
        # if movement_whisker == 1:
        #     state_whisker_ax = state_whisker
        # else:
    #     state_whisker_ax = -(state_whisker-1)
        state_wheel_ax = state_wheel
        state_whisker_ax = state_whisker
        ax = axs[state_wheel_ax, state_whisker_ax]

        for i in range(len(x)):
            for j in range(len(y)):
                U[i, j] = X[j, i] - update_var(X[j, i], design_matrix_heading[x_var], x_var, mouse_dynamics, 
                                    state_wheel)
                V[i, j] = Y[j, i] - update_var(Y[j, i], design_matrix_heading[y_var], y_var, mouse_dynamics, 
                                    state_whisker)

        # xy': Arrow direction in data coordinates, i.e. the arrows point from (x, y) to (x+u, y+v).     
        ax.quiver(X, Y, U, V, angles='xy')
        ax.set_xlabel('Wheel velocity - state ' + str(state_wheel_ax))
        ax.set_ylabel('Whisker motion energy - state ' + str(state_whisker_ax))
        ax.set_title(mouse_name)
        # ax.set_ylim([-1.5, 4])
        # ax.set_xlim([-4, 4])
        
    if plot_traj == True:
        plot_trajectories(new_states, inverted_mapping, design_matrix_heading, x_var, y_var, axs, trajectory_num)
        
    plt.tight_layout()
    
    plt.show()
    
    
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