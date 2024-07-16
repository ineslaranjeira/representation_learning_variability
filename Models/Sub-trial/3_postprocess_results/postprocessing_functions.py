""" 
IMPORTS
"""
import autograd.numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, Normalizer

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