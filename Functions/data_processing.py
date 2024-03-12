""" 
IMPORTS
"""
import os
import autograd.numpy as np
import pandas as pd

from one.api import ONE
import brainbox.behavior.wheel as wh

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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


def time_intervals(session_trials):
    
    session_trials = prepro(session_trials)
    session_trials['ITI'] = session_trials['intervals_1'] - session_trials['stimOff_times']
    session_trials['feedback_time'] = session_trials['stimOff_times'] - session_trials['feedback_times']
    session_trials['movement_time'] = session_trials['feedback_times'] - session_trials['firstMovement_times']
    session_trials['failing_quiescence'] = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod'] - session_trials['intervals_0']
    session_trials['full_ITI'] = session_trials['ITI']*np.NaN
    session_trials['full_ITI'][:-1] = np.array(session_trials['intervals_0'][1:]) - np.array(session_trials['stimOff_times'][:-1])
    session_trials['feedback_ITI'] = session_trials['ITI']*np.NaN
    session_trials['feedback_ITI'][:-1] = np.array(session_trials['intervals_0'][1:]) - np.array(session_trials['feedback_times'][:-1])
    session_trials['prev_feedback'] = session_trials['feedbackType'] * np.nan
    session_trials['prev_feedback'][1:] = session_trials['feedbackType'][:-1]   
    session_trials['long_ITI'] =  session_trials['ITI']*np.NaN
    session_trials['long_ITI'][:-1] =  np.array(session_trials['goCueTrigger_times'][1:]) - np.array(session_trials['feedback_times'][:-1])
    session_trials['elongated_quiesc'] = session_trials['failing_quiescence']
    session_trials.loc[session_trials['elongated_quiesc'] < 0.1, 'elongated_quiesc'] = 0
    session_trials.loc[session_trials['elongated_quiesc'] >= 0.1, 'elongated_quiesc'] = 1
    
    return session_trials


def process_quiescence(df):
    
    # Process data
    new = df[['trial', 'trial_epoch', 'feedback', 'next_feedback', 'signed_contrast', 'movement']]

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
        
        
    actual_quiescence = pd.DataFrame(columns=['trial', 'quiesc_length', 'time_to_quiesc', 
                                            'pre_quiesc_move_duration', 'pre_quiesc_move_count'], 
                                    index=new_df['trial'].unique())

    for t, trial in enumerate(new_df['trial'].unique()[:-1]):
        
        trial_data = new_df.loc[new_df['trial']==t]
        next_trial = new_df.loc[new_df['trial']==t+1]
        
        # Some timepoints
        # next_onset = list(next_trial.loc[next_trial['trial_epoch']=='trial_start', 'epoch_onset'])[0]
        next_quiescence = list(next_trial.loc[next_trial['trial_epoch']=='quiescence', 'epoch_onset'])[0]
        
        post_choice_stillness = trial_data.loc[(trial_data['trial_epoch']=='post_choice') & 
                                                (trial_data['movement']==0), 'movement_onset']
        next_trial_init_stillness = next_trial.loc[(next_trial['trial_epoch']=='trial_start') & 
                                                (next_trial['movement']==0), 'movement_onset']
        
        if len(next_trial_init_stillness) > 0:
            last_stillness_onset = list(next_trial_init_stillness)[-1]
        else:
            last_stillness_onset = list(post_choice_stillness)[-1]
            
        response_time = list(trial_data.loc[trial_data['trial_epoch']=='post_choice', 'epoch_onset'])[0]
        # trial_data['movement_duration'] = trial_data['movement_onset'] * np.NaN
        # trial_data['movement_duration'][1:] = np.diff(trial_data['movement_onset'])
        
        # Save data
        actual_quiescence['trial'][t] = trial
        actual_quiescence['quiesc_length'][t] = next_quiescence - last_stillness_onset
        actual_quiescence['time_to_quiesc'][t] = last_stillness_onset - response_time
        actual_quiescence['pre_quiesc_move_duration'][t] = np.sum(trial_data.loc[(trial_data['trial_epoch']=='post_choice') & 
                                                                        (trial_data['movement']==1), 'movement_duration']) + np.sum(next_trial.loc[(next_trial['trial_epoch']=='trial_start') & 
                                                                        (next_trial['movement']==1), 'movement_duration']) 
        
        actual_quiescence['pre_quiesc_move_count'][t] = len(trial_data.loc[(trial_data['trial_epoch']=='post_choice') & 
                                                                        (trial_data['movement']==1)]) + len(next_trial.loc[(next_trial['trial_epoch']=='trial_start') & 
                                                                        (trial_data['movement']==1)])

    processed_df = new_df.merge(actual_quiescence)
    
    return processed_df