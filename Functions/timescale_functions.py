""" 
IMPORTS
"""
import os
import autograd.numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from scipy.stats import ttest_1samp
from scipy.optimize import curve_fit

# Custom functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Functions/'
# functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Functions/'
os.chdir(functions_path)
from one_functions_generic import prepro

""" Correlation analyses """



def auto_correlogram(x, lags_to_plot):

    # Calculate the lag values corresponding to the cross-correlation
    lags = np.arange(-lags_to_plot, lags_to_plot)

    # Compute the cross-correlation using np.correlate
    auto_correlation = np.zeros(len(lags)) * np.nan
    
    for l, lag in enumerate(lags):
        if lag < 0:
            x_chunk = np.array(x[-lag:])
            y_chunk = np.array(x[:lag])
            
        elif lag == 0:
            x_chunk = np.array(x)
            y_chunk = np.array(x)

        elif lag > 0:
            x_chunk = np.array(x[0:-lag])
            y_chunk = np.array(x[lag:])
            
        auto_correlation[l] = stats.pearsonr(x_chunk, y_chunk).statistic

    return auto_correlation


def cross_correlogram(x, y, lags_to_plot):

    # Calculate the lag values corresponding to the cross-correlation
    lags = np.arange(-lags_to_plot, lags_to_plot)

    # Compute the cross-correlation using np.correlate
    cross_correlation = np.zeros(len(lags)) * np.nan
    
    for l, lag in enumerate(lags):
        if lag < 0:
            x_chunk = np.array(x[-lag:])
            y_chunk = np.array(y[:lag])
            
        elif lag == 0:
            x_chunk = np.array(x)
            y_chunk = np.array(y)

        elif lag > 0:
            x_chunk = np.array(x[0:-lag])
            y_chunk = np.array(y[lag:])
            
        cross_correlation[l] = stats.pearsonr(x_chunk, y_chunk).statistic

    return cross_correlation


def fit_tau(t, tau, c):
    return np.exp(-t / tau) + c


def shuffle_covariance(shuffle_iters, use_data, features):
    
    # Shuffled data
    shuffle_cov = {}
    for i, iter in enumerate(range(shuffle_iters)):
        
        #shuffle_cov[i] = {}
        # Shuffle data
        shuffled_data = use_data.copy()
        for feature in features:
            shuffled_data[feature] = list(shuffled_data[feature].sample(frac=1, axis=0).reset_index(drop=True))  # Shuffle rows of the column

        cov_df = shuffled_data.dropna().drop_duplicates()
        cov = cov_df.corr()

        # Save 
        for feature_x in features:
            for feature_y in features:
                eid = str(feature_x + feature_y)
                if i == 0:
                    shuffle_cov[eid] = [cov[feature_x][feature_y]]
                else:
                    shuffle_cov[eid] = np.concatenate((shuffle_cov[eid], [cov[feature_x][feature_y]]))

    return shuffle_cov


def cov_stats(cov, shuffle_cov, features):    

    p_values = cov * np.nan
    dif_cov = cov * np.nan 
    dif_sig_cov = cov * np.nan 

    # Loop through eids
    for feature_x in features:
        for feature_y in features:
            eid = str(feature_x + feature_y)

            shuffle_dist = shuffle_cov[eid]
            # t-test
            _, p_value = ttest_1samp(shuffle_dist, cov[feature_x][feature_y])

            # To avoid numerical error
            if p_value == 0:
                p_value = 0.00000001

            # Save
            p_values[feature_x][feature_y] = p_value
            dif_cov[feature_x][feature_y] = cov[feature_x][feature_y] - np.mean(shuffle_dist)
            
            if p_value < 0.01:
                dif_sig_cov[feature_x][feature_y] = dif_cov[feature_x][feature_y]
            else:
                dif_sig_cov[feature_x][feature_y] = np.nan

    return p_values, dif_cov, dif_sig_cov