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