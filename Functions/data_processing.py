""" 
IMPORTS
"""
import os
import autograd.numpy as np
import pandas as pd
from datetime import datetime
import json
import pickle
from one.api import ONE
import brainbox.behavior.wheel as wh
import sys
from joblib import Parallel, delayed

# Plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

# ML tools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft, fftshift

# Custom functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Functions/'
# functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Functions/'
os.chdir(functions_path)
from one_functions_generic import prepro


""" Design-matrix processing """

def bins_per_trial(design_matrix, session_trials):
    # Split session into trial phases and gather most likely states of those trial phases
    use_data = design_matrix.dropna()
    use_data['Trial'] = use_data['Bin'] * np.nan
    use_data['block'] = use_data['Bin'] * np.nan

    trial_num = len(session_trials)
    
    qui_init = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']
    iti_end = session_trials['intervals_1']
    
    block = session_trials['probabilityLeft']

    for t, trial in enumerate(range(trial_num)):
        
        # Trial number   
        use_data.loc[(use_data['Bin'] <= iti_end[t]*10) & (use_data['Bin'] > qui_init[t]*10), 'Trial'] = trial
        
        # Block 
        use_data.loc[(use_data['Bin'] <= iti_end[t]*10) & (use_data['Bin'] > qui_init[t]*10), 'block'] = block[t]
        
    return use_data


def group_per_phase(data, label = 'broader_label'):
    
    # Average trials per phase
    """ Get trial classes """
    # Mean per label
    mean_label = pd.DataFrame(data.groupby(['label', 'Trial'])['pc1', 'pc2'].mean())
    mean_label = mean_label.reset_index(level=[0,1])

    # Get correct and incorrect, left and right trials
    trials_correct = mean_label.loc[mean_label['label']=='Correct feedback', 'Trial']
    trials_incorrect = mean_label.loc[mean_label['label']=='Incorrect feedback', 'Trial']
    trials_left = mean_label.loc[mean_label['label']=='Left choice', 'Trial']
    trials_right = mean_label.loc[mean_label['label']=='Right choice', 'Trial']

    correct_left = np.intersect1d(trials_correct, trials_left)
    correct_right = np.intersect1d(trials_correct, trials_right)
    incorrect_left = np.intersect1d(trials_incorrect, trials_left)
    incorrect_right = np.intersect1d(trials_incorrect, trials_right)

    correct_left_data = data.loc[data['Trial'].isin(correct_left)]
    incorrect_left_data = data.loc[data['Trial'].isin(incorrect_left)]
    correct_right_data = data.loc[data['Trial'].isin(correct_right)]
    incorrect_right_data = data.loc[data['Trial'].isin(incorrect_right)]

    """ Mean per label per trial """
    mean_trial = pd.DataFrame(data.groupby([label, 'Trial'])['pc1', 'pc2'].mean())
    mean_trial = mean_trial.reset_index(level=[0, 1])

    # Mean per label
    correct_left_data_mean = pd.DataFrame(correct_left_data.groupby([label])['pc1', 'pc2'].mean())
    correct_left_data_mean = correct_left_data_mean.reset_index(level=[0])
    incorrect_left_data_mean = pd.DataFrame(incorrect_left_data.groupby([label])['pc1', 'pc2'].mean())
    incorrect_left_data_mean = incorrect_left_data_mean.reset_index(level=[0])
    correct_right_data_mean = pd.DataFrame(correct_right_data.groupby([label])['pc1', 'pc2'].mean())
    correct_right_data_mean = correct_right_data_mean.reset_index(level=[0])
    incorrect_right_data_mean = pd.DataFrame(incorrect_right_data.groupby([label])['pc1', 'pc2'].mean())
    incorrect_right_data_mean = incorrect_right_data_mean.reset_index(level=[0])

    # Mean per label per trial
    correct_left_data = pd.DataFrame(correct_left_data.groupby([label, 'Trial'])['pc1', 'pc2'].mean())
    correct_left_data = correct_left_data.reset_index(level=[0, 1])
    incorrect_left_data = pd.DataFrame(incorrect_left_data.groupby([label, 'Trial'])['pc1', 'pc2'].mean())
    incorrect_left_data = incorrect_left_data.reset_index(level=[0, 1])
    correct_right_data = pd.DataFrame(correct_right_data.groupby([label, 'Trial'])['pc1', 'pc2'].mean())
    correct_right_data = correct_right_data.reset_index(level=[0, 1])
    incorrect_right_data = pd.DataFrame(incorrect_right_data.groupby([label, 'Trial'])['pc1', 'pc2'].mean())
    incorrect_right_data = incorrect_right_data.reset_index(level=[0, 1])

    if label == 'broader_label':
        sort_dict = {'Pre-choice':0,'Choice':1,'Post-choice':2}
    elif label=='label':
        sort_dict = {'Quiescence':0,'Stimulus left':1, 'Stimulus right': 2, 
                     'Left choice': 3, 'Right choice': 4, 'Correct feedback': 5,
                     'Incorrect feedback': 6, 'ITI_correct': 7, 'ITI_incorrect': 8,
                     }
    correct_left_data_mean = correct_left_data_mean.iloc[correct_left_data_mean[label].map(sort_dict).sort_values().index]
    correct_right_data_mean = correct_right_data_mean.iloc[correct_right_data_mean[label].map(sort_dict).sort_values().index]
    incorrect_left_data_mean = incorrect_left_data_mean.iloc[incorrect_left_data_mean[label].map(sort_dict).sort_values().index]
    incorrect_right_data_mean = incorrect_right_data_mean.iloc[incorrect_right_data_mean[label].map(sort_dict).sort_values().index]

    grouped = [correct_left_data, incorrect_left_data, correct_right_data, incorrect_right_data]
    grouped_mean = [correct_left_data_mean, incorrect_left_data_mean, correct_right_data_mean, incorrect_right_data_mean]

    return grouped, grouped_mean


""" Processing of timeseries data """
# def time_intervals(session_trials):
    
#     session_trials = prepro(session_trials)
#     session_trials['ITI'] = session_trials['intervals_1'] - session_trials['stimOff_times']
#     session_trials['feedback_time'] = session_trials['stimOff_times'] - session_trials['feedback_times']
#     session_trials['movement_time'] = session_trials['feedback_times'] - session_trials['firstMovement_times']
#     session_trials['failing_quiescence'] = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod'] - session_trials['intervals_0']
#     session_trials['full_ITI'] = session_trials['ITI']*np.NaN
#     session_trials['full_ITI'][:-1] = np.array(session_trials['intervals_0'][1:]) - np.array(session_trials['stimOff_times'][:-1])
#     session_trials['feedback_ITI'] = session_trials['ITI']*np.NaN
#     session_trials['feedback_ITI'][:-1] = np.array(session_trials['intervals_0'][1:]) - np.array(session_trials['feedback_times'][:-1])
#     session_trials['prev_feedback'] = session_trials['feedbackType'] * np.nan
#     session_trials['prev_feedback'][1:] = session_trials['feedbackType'][:-1]   
#     session_trials['long_ITI'] =  session_trials['ITI']*np.NaN
#     session_trials['long_ITI'][:-1] =  np.array(session_trials['goCueTrigger_times'][1:]) - np.array(session_trials['feedback_times'][:-1])
#     session_trials['elongated_quiesc'] = session_trials['failing_quiescence']
#     session_trials.loc[session_trials['elongated_quiesc'] < 0.1, 'elongated_quiesc'] = 0
#     session_trials.loc[session_trials['elongated_quiesc'] >= 0.1, 'elongated_quiesc'] = 1
    
#     return session_trials


def process_quiescence(df):

    # Process data
    new = df[['trial', 'trial_epoch', 'feedback', 
                'next_feedback', 'signed_contrast', 
                'movement', 'response', 'reaction', 
                'choice', 'probabilityLeft']]

    # Identify consecutive duplicates
    consecutive_duplicates_mask = new.eq(new.shift())

    # Filter DataFrame to remove consecutive duplicates
    df_no_consecutive_duplicates = new[~consecutive_duplicates_mask.all(axis=1)].reset_index()

    time_df = df[['time']].reset_index()
    merged_df = df_no_consecutive_duplicates.merge(time_df, on='index')
    merged_df = merged_df.rename(columns={'time': 'movement_onset'})

    # Get trial onset
    epoch_onset = pd.DataFrame(merged_df.groupby(['trial', 'trial_epoch'])['movement_onset'].min())
    epoch_onset = epoch_onset.reset_index(level=[0, 1])
    epoch_onset = epoch_onset.rename(columns={'movement_onset': 'epoch_onset'})

    new_df = merged_df.merge(epoch_onset)
    new_df['movement_duration'] = new_df['movement_onset'] * np.NaN
    new_df['movement_duration'][1:] = np.diff(new_df['movement_onset'])
        
        
    actual_quiescence = pd.DataFrame(columns=['trial', 'quiesc_length', 'time_to_quiesc', 'time_to_quiesc_2', 
                                            'pre_quiesc_move_duration', 'pre_quiesc_move_count'], 
                                    index=new_df['trial'].unique())

    for t, trial in enumerate(new_df['trial'].unique()):
        
        trial_data = new_df.loc[new_df['trial']==trial]
        next_trial = new_df.loc[new_df['trial']==trial+1]
        
        try:
            # If next trial exists
            if len(next_trial) > 0:
                
                # Some timepoints
                response_time = list(trial_data.loc[trial_data['trial_epoch']=='post_choice', 'epoch_onset'])[0]
                next_quiescence = next_trial.loc[next_trial['trial_epoch']=='quiescence', 'epoch_onset']
                next_stimOn = next_trial.loc[next_trial['trial_epoch']=='stim_on', 'epoch_onset']
                next_movement_init = next_trial.loc[next_trial['trial_epoch']=='movement', 'epoch_onset']
                next_trial_onset = next_trial.loc[next_trial['trial_epoch']=='trial_start', 'epoch_onset']
                
                # Get inter-trial data
                iti_data_current = new_df.loc[((new_df['trial']==trial) & (new_df['trial_epoch']=='post_choice'))]
                # iti_data_next = new_df.loc[((new_df['trial']==trial+1) & (new_df['trial_epoch'].isin(['trial_start', 'quiescence'])))]
                iti_data_next = new_df.loc[((new_df['trial']==trial+1) & (new_df['trial_epoch']=='trial_start'))]
                
                # Find last stillness onset
                # If there is no stillness until quiescence, consider quiescence
                if list(iti_data_next['movement'])[-1] == 1.:
                    last_stillness_onset = list(next_quiescence)[0]
                else:
                    if 1 in list(iti_data_next['movement']):
                        # If there is a movement in the new trial but it doesn't preceed quiescence, look for last stillness
                        where_last_move = np.where(iti_data_next['movement']==1)[0][-1]
                        last_stillness_onset = list(iti_data_next['movement_onset'])[where_last_move+1]
                    elif 1 in list(iti_data_current['movement']):
                        where_last_move = np.where(iti_data_current['movement']==1)[0][-1]
                        last_stillness_onset = list(iti_data_current['movement_onset'])[where_last_move+1]
                    else:
                        # if there is no movement after choice, consider response
                        last_stillness_onset = response_time
                        
            
                # Save data
                actual_quiescence['trial'][t] = trial
                if len(next_stimOn) > 0:
                    actual_quiescence['quiesc_length'][t] = list(next_stimOn)[0] - last_stillness_onset
                else:
                    # If movement was detected before Stimulus onset
                    actual_quiescence['quiesc_length'][t] = list(next_movement_init)[0] - last_stillness_onset 
                
                # Time to quiescence goes from end of last 
                actual_quiescence['time_to_quiesc'][t] = last_stillness_onset - response_time
                if last_stillness_onset >= list(next_trial_onset)[0]:
                    actual_quiescence['time_to_quiesc_2'][t] = last_stillness_onset - list(next_trial_onset)[0]
                else:
                    actual_quiescence['time_to_quiesc_2'][t] = 0.
                
                actual_quiescence['pre_quiesc_move_duration'][t] = np.sum(trial_data.loc[(trial_data['trial_epoch']=='post_choice') & 
                                                                                (trial_data['movement']==1), 'movement_duration']) + np.sum(next_trial.loc[(next_trial['trial_epoch']=='trial_start') & 
                                                                                (next_trial['movement']==1), 'movement_duration']) 
                
                actual_quiescence['pre_quiesc_move_count'][t] = len(trial_data.loc[(trial_data['trial_epoch']=='post_choice') & 
                                                                                (trial_data['movement']==1)]) + len(next_trial.loc[(next_trial['trial_epoch']=='trial_start') & 
                                                                                (trial_data['movement']==1)])
                                                                        
            else:
                # Save data
                actual_quiescence['trial'][t] = trial
            
        except:
            print(trial)
        processed_df = new_df.merge(actual_quiescence)

    return processed_df


def interpolate(time_snippet, snippet, size, plot):
    x = np.arange(0, len(time_snippet))
    y = snippet
    f = interp1d(x, y, 'cubic')
    
    # New grid coordinates
    new_x = np.linspace(0, len(x)-1, size)  # Upscale to 6 columns

    # Interpolate values at new grid coordinates
    rescaled_array = f(new_x)
    
    if plot == True:
        # plt.plot(x, y, 'o', new_x, rescaled_array, '-')
        plt.plot(new_x, rescaled_array, '-')
        plt.plot(x, snippet)
        plt.xlabel('Time')
        plt.ylabel('Data')
        plt.show()
            
    return rescaled_array


def resample_common_time(reference_time, timestamps, data, kind, fill_gaps=None):
    
    # t = np.arange(t_init, t_end, 1 / freq)  # Evenly resample at frequency
    if reference_time[-1] > timestamps[-1]:
        reference_time = reference_time[:-1]  # Occasionally due to precision errors the last sample may be outside of range.
    yinterp = interpolate.interp1d(timestamps, data, kind=kind)(reference_time)
    
    if fill_gaps:
        #  Find large gaps and forward fill @fixme This is inefficient
        gaps, = np.where(np.diff(timestamps) >= fill_gaps)

        for i in gaps:
            yinterp[(reference_time >= timestamps[i]) & (reference_time < timestamps[i + 1])] = data[i]
            
    return yinterp, reference_time


""" DIMENSIONALITY REDUCTION FUNCTIONS """

def pca_behavior(use_mat, keep_pc, plot=False):
    
    """
    PRINCIPLE COMPONENT ANALYSES
    """
    # use_mat = epoch_matrix[var_names]
    X = np.array(use_mat) # (n_samples, n_features)

    # Mean centered and equal variance (redundant code)
    scaler = StandardScaler()
    new_X = scaler.fit_transform(X)
    
    # scaler = StandardScaler()
    # standardized = scaler.fit_transform(X)
    # # Normalize between 0 and 1
    # normalizer = Normalizer().fit(standardized)
    # new_X = normalizer.transform(standardized)
    
    pca = PCA() # svd_solver='full'
    X_reduced = pca.fit_transform(new_X)

    if plot == True:
        # Plot variance explained per principle component
        fig, ax = plt.subplots(figsize=[6,5])
        plt.rc('font', size=18)
        plt.bar(np.arange(1, keep_pc+1, 1), pca.explained_variance_ratio_[0:keep_pc], color='steelblue')
        plt.xticks(np.arange(1, keep_pc, 1))
        plt.xlabel('PCs')
        plt.ylabel('% Variance explained')
        plt.show()
    
    return X_reduced, X


def augment_data(X_reduced, epoch_matrix, keep_pc):
    
    if np.shape(X_reduced)[0] == np.shape(epoch_matrix)[0]:
        # Plot projections of datapoints into first 3 principal components
        augmented_data = pd.DataFrame(columns=['Bin'], index=range(np.shape(X_reduced)[0]))
        augmented_data['Bin'] = epoch_matrix['Bin']
        augmented_data['label'] = epoch_matrix['label']
        augmented_data['Trial'] = epoch_matrix['Trial']
        augmented_data['feedback'] = epoch_matrix['feedback']
        augmented_data['signed_contrast'] = epoch_matrix['signed_contrast']
        augmented_data['choice'] = epoch_matrix['choice']
        augmented_data['broader_label'] = epoch_matrix['broader_label']
        
        for p in range(keep_pc):
            augmented_data[str('pc' + str(p+1))] = X_reduced[:, p].transpose()
    else:
        print('Size does not match')

    return augmented_data


def plot_timeseries_pcs(X, augmented_data, var_names, init, range):
    scaler = StandardScaler()
    new_X = scaler.fit_transform(X)
    
    # scaler = StandardScaler()
    # standardized = scaler.fit_transform(X)
    # # Normalize between 0 and 1
    # normalizer = Normalizer().fit(standardized)
    # new_X = normalizer.transform(standardized)

    for v , var in enumerate(var_names):
        plt.plot(new_X[init:init+range, v], color='black', label='Data')
        plt.plot(augmented_data['pc1'], color='red', label='PC 1', alpha=0.5)
        plt.plot(augmented_data['pc2'], color='green', label='PC 2', alpha=0.5)
        plt.plot(augmented_data['pc3'], color='blue', label='PC 3', alpha=0.5)
        plt.legend()
        plt.title(var)

        plt.xlim([init, init+range])
        plt.show()
        
        
def save_and_log(file_to_save, filename, file_format, save_path, script_name):

    # current date
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y")
    
    # First, save file in desired location and with desired format
    os.chdir(save_path)
    if file_format == 'pickle':
        pickle.dump(file_to_save, open(filename+date_time, "wb"))
    elif file_format == 'parquet':
        assert isinstance(file_to_save, pd.DataFrame) 
        file_to_save.to_parquet(filename+date_time, compression='gzip')  
    else: 
        sys.exit("File format not implemented")        

    # Open log file json
    log_file_path =  '/home/ines/repositories/representation_learning_variability/DATA/' 
    with open(log_file_path + 'metadata_log.json', 'r') as openfile:
        # Reading from json file
        metadata_log = json.load(openfile)
    
    # # Then create entry for the log file
    # files = [f for f in os.listdir() if f.endswith('.ipynb')]
    # script_name = files[0]
    
    # Populate new entry
    new_log_entry = {
    "data_filename": filename,
    "script_name": script_name,
    "timestamp": date_time
    }
    
    # Update log dict
    order_last_entry = int(list(metadata_log.keys())[-1])
    metadata_log[order_last_entry+1] = new_log_entry
    updated_json = json.dumps(metadata_log)

    # Overwrite json log file
    with open(log_file_path+"metadata_log.json", "w") as outfile:
        outfile.write(updated_json)

    return metadata_log


""" WAVELETS """


def morlet_conj_ft(omega_vals, omega0):
    """
    Computes the conjugate Fourier transform of the Morlet wavelet.
    
    Parameters:
    - w: Angular frequency values (array or scalar)
    - omega0: Dimensionless Morlet wavelet parameter
    
    Returns:
    - out: Conjugate Fourier transform of the Morlet wavelet
    """
    
    return np.pi**(-1/4) * np.exp(-0.5 * (omega_vals - omega0)**2)


def fast_wavelet_morlet_convolution_parallel(x, f, omega0, dt):
    """
    Fast Morlet wavelet transform using parallel computation.

    Args:
        x (array): 1D array of projection values to transform.
        f (array): Center frequencies of the wavelet frequency channels (Hz).
        omega0 (float): Dimensionless Morlet wavelet parameter.
        dt (float): Sampling time (seconds).

    Returns:
        amp (array): Wavelet amplitudes.
        W (array): Wavelet coefficients (complex-valued, optional).
    """
    N = len(x)
    L = len(f)
    amp = np.zeros((L, N))
    Q = np.zeros((L, N))

    # Ensure N is even
    if N % 2 == 1:
        x = np.append(x, 0)
        N += 1
        test = True
    else:
        test = False

    # Add zero padding to x
    # Zero padding serves to compensate for the fact that the kernel does not have the same size as 
    # 
    x = np.concatenate((np.zeros(N // 2), x, np.zeros(N // 2)))
    M = N
    N = len(x)

    # Compute scales
    scales = (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * f)
    # angular frequencies to compute FT for (depends on sampling frequency); is as long as N 
    omega_vals = 2 * np.pi * np.arange(-N // 2, N // 2) / (N * dt)  

    # Fourier transform of x; shift folds it around zero so that it is more interpretable (frequencies at the right of nyquist become negative)
    x_hat = fftshift(fft(x))

    # Index for truncation to recover the actual x without padding
    if test:
        idx = np.arange(M // 2, M // 2 + M - 1)
    else:
        idx = np.arange(M // 2, M // 2 + M)

    # Function for parallel processing
    def process_frequency(i):
        # Take the Morlet conjugate of the Fourier transform
        m = morlet_conj_ft(-omega_vals * scales[i], omega0)
        # Convolution on the Fourier domain (as opposed to time domain in DWT)
        conv = m * x_hat
        # Inverse Fourier transform (normalized?)
        # q are the wavelet coefficients; normalized to ensure the energy of the wavelet is preserved across different scales
        q = ifft(conv) * np.sqrt(scales[i])
        # Recover q without padding
        q = q[idx]
        amp_row = np.abs(q) * np.pi**-0.25 * np.exp(0.25 * (omega0 - np.sqrt(omega0**2 + 2))**2) / np.sqrt(2 * scales[i])
        return amp_row, q

    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_frequency)(i) for i in range(L))

    for i, (amp_row, q) in enumerate(results):
        amp[i, :] = amp_row
        Q[i, :] = q

    return amp, Q, x_hat
