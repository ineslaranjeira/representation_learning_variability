""" 
IMPORTS
"""
import os
import sys
import autograd.numpy as np
import pickle
import pandas as pd

from itertools import count
from one.api import ONE
import brainbox.behavior.wheel as wh

from functools import partial
from jax import vmap
from pprint import pprint
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Custom functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Functions/'
#functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability//Models/Sub-trial//2_fit_models/'
os.chdir(functions_path)
from one_functions_generic import prepro



################################################### INES's FUNCTIONS ############################################################

def align_bin_design_matrix (init, end, event_type_list, session_trials, design_matrix, most_likely_states, multiplier):
    
    for e, this_event in enumerate(event_type_list):
        
        # Initialize variables
        # Before there was a function for keeping validation set apart, now deprecated
        #test_length = np.shape(test_set)[0]
        #reduced_design_matrix = design_matrix[:test_length*2].append(design_matrix[-test_length*2:])
        reduced_design_matrix = design_matrix.copy()
        reduced_design_matrix['most_likely_states'] = most_likely_states
        reduced_design_matrix['new_bin'] = reduced_design_matrix['Bin'] * np.nan
        reduced_design_matrix['correct'] = reduced_design_matrix['Bin'] * np.nan
        reduced_design_matrix['choice'] = reduced_design_matrix['Bin'] * np.nan
                
        events = session_trials[this_event]
        feedback = session_trials['feedbackType']
        choice = session_trials['choice']
        
        events = session_trials[this_event]
        #state_stack = np.zeros((len(events), end + init)) * np.nan
                
        for t, trial in enumerate(events[:-2]):
            event = events[t]
            
            # Check feedback
            if feedback[t] ==1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) &
                                            (reduced_design_matrix['Bin']> event*multiplier + init), 
                                            'correct'] = 1
            elif feedback[t] == -1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) &
                                            (reduced_design_matrix['Bin']> event*multiplier + init), 
                                            'correct'] = 0
            # Check choice
            if choice[t] ==1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) &
                                            (reduced_design_matrix['Bin']> event*multiplier + init), 
                                            'choice'] = 'right'
            elif choice[t] == -1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) &
                                            (reduced_design_matrix['Bin']> event*multiplier + init), 
                                            'choice'] = 'left'
            # Rename bins so that they are aligned on stimulus onset
            if event > 0:
                event_window = reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) &
                                                         (reduced_design_matrix['Bin']> event*multiplier + init)]
                onset_bin = reduced_design_matrix.loc[reduced_design_matrix['Bin']>= event*multiplier, 'Bin']
                if (len(event_window)>0): # & len(onset_bin)>0:
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
                
                
def plot_states_aligned(init, end, reduced_design_matrix, event_type_name, bin_size):

    for e, this_event in enumerate(event_type_name):
            
        # PLOT
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=[12, 7])
        plt.rc('font', size=12)
        use_data = reduced_design_matrix.dropna()
        use_data['new_bin'] = use_data['new_bin'] * bin_size
        #use_data = use_data[:test_length*2]
        
        # Correct left
        a = sns.histplot(x='new_bin', hue='most_likely_states', data=use_data.loc[(use_data['correct']==1) 
                                                                                & (use_data['choice']=='left')], stat='count', alpha=0.3, 
                        multiple="stack", binwidth=bin_size, binrange=(bin_size*init, bin_size*end), legend=False, ax = ax[0, 0], palette='viridis')
        # Correct right
        b = sns.histplot(x='new_bin', hue='most_likely_states', data=use_data.loc[(use_data['correct']==1) 
                                                                                & (use_data['choice']=='right')], stat='count', alpha=0.3, 
                        multiple="stack", binwidth=bin_size, binrange=(bin_size*init, bin_size*end), legend=False, ax = ax[0, 1], palette='viridis')
        # Incorrect left
        c = sns.histplot(x='new_bin', hue='most_likely_states', data=use_data.loc[(use_data['correct']==0) 
                                                                                & (use_data['choice']=='left')], stat='count', alpha=0.3,
                        multiple="stack", binwidth=bin_size, binrange=(bin_size*init, bin_size*end), legend=False, ax = ax[1, 0], palette='viridis')
        # Incorrect right
        d = sns.histplot(x='new_bin', hue='most_likely_states', data=use_data.loc[(use_data['correct']==0) 
                                                                                & (use_data['choice']=='right')], alpha=0.3, 
                        stat='count', multiple="stack", binwidth=bin_size, binrange=(bin_size*init, bin_size*end), legend=False, ax = ax[1, 1], palette='viridis')
        
        # ax[0, 0].set_title(str(this_event + ' - correct left'))
        # ax[0, 0].set_xlabel(str('Time from event (s)'))

        # ax[0, 1].set_title(str(this_event + ' - correct right'))
        # ax[0, 1].set_xlabel(str('Time from event (s)'))
        
        # ax[1, 0].set_title(str(this_event + ' - incorrect left'))
        # ax[1, 0].set_xlabel(str('Time from event (s)'))
        
        # ax[1, 1].set_title(str(this_event + ' - incorrect right'))
        # ax[1, 1].set_xlabel(str('Time from event (s)'))
        
        ax[0, 0].set_title(str('Correct left'))
        ax[0, 0].set_xlabel(str('Time from go cue (s)'))

        ax[0, 1].set_title(str('Correct right'))
        ax[0, 1].set_xlabel(str('Time from go cue (s)'))
        
        ax[1, 0].set_title(str('Incorrect left'))
        ax[1, 0].set_xlabel(str('Time from go cue (s)'))
        
        ax[1, 1].set_title(str('Incorrect right'))
        ax[1, 1].set_xlabel(str('Time from go cue (s)'))
        
        plt.tight_layout()
        plt.show()


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


def plot_transition_mat (init_params, fit_params):
    
    plt.title('Initial parameters')
    plot_transition_matrix(init_params[1].transition_matrix)
    init_params[1].transition_matrix

    plt.title('Fold 1')
    plot_transition_matrix(fit_params[1].transition_matrix[0])
    fit_params[1].transition_matrix[0]
    plt.title('Fold 2')
    plot_transition_matrix(fit_params[1].transition_matrix[1])
    fit_params[1].transition_matrix[1]
    plt.title('Fold 3')
    plot_transition_matrix(fit_params[1].transition_matrix[2])
    fit_params[1].transition_matrix[2]
    

def states_per_trial_phase(reduced_design_matrix, session_trials):
    
    # Split session into trial phases and gather most likely states of those trial phases
    use_data = reduced_design_matrix.dropna()
    trial_num = len(session_trials)

    # Quiescence
    qui_init = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']
    qui_end = session_trials['goCueTrigger_times']
    quiescence_states = []

    # ITI
    iti_init_correct = session_trials['feedback_times'] + 1
    iti_init_incorrect = session_trials['feedback_times'] + 2
    iti_end = session_trials['intervals_1']
    ITI_states_correct = []
    ITI_states_incorrect = []

    # Feedback
    feedback_init = session_trials['feedback_times']
    correct_end = session_trials['feedback_times'] + 1
    incorrect_end = session_trials['feedback_times'] + 2
    correct_states = []
    incorrect_states = []

    # Reaction time 
    rt_init = session_trials['goCueTrigger_times']
    rt_end = session_trials['firstMovement_times']
    stim_left_states = []
    stim_right_states = []

    # Movement time 
    move_init = session_trials['firstMovement_times']
    move_end = session_trials['feedback_times']
    left_states = []
    right_states = []

    for t, trial in enumerate(range(trial_num)):
        
        # Quiescence
        quiescence_data = use_data.loc[(use_data['Bin'] <= qui_end[t]*10) & (use_data['Bin'] > qui_init[t]*10)]
        quiescence_states = np.append(quiescence_states, quiescence_data['most_likely_states'])

        # Feedback        
        if session_trials['feedbackType'][t] == 1.:
            
            correct_data = use_data.loc[(use_data['Bin'] <= correct_end[t]*10) & (use_data['Bin'] > feedback_init[t]*10)]
            correct_states = np.append(correct_states, correct_data['most_likely_states'])
            
            # ITI correct
            ITI_data_correct = use_data.loc[(use_data['Bin'] <= iti_end[t]*10) & (use_data['Bin'] > iti_init_correct[t]*10)]
            ITI_states_correct = np.append(ITI_states_correct, ITI_data_correct['Bin'])
        
        elif session_trials['feedbackType'][t] == -1.:
            incorrect_data = use_data.loc[(use_data['Bin'] <= incorrect_end[t]*10) & (use_data['Bin'] > feedback_init[t]*10)]
            incorrect_states = np.append(incorrect_states, incorrect_data['most_likely_states'])

            # ITI incorrect
            ITI_data_incorrect = use_data.loc[(use_data['Bin'] <= iti_end[t]*10) & (use_data['Bin'] > iti_init_incorrect[t]*10)]
            ITI_states_incorrect = np.append(ITI_states_incorrect, ITI_data_incorrect['Bin'])
        
        # Move
        move_data = use_data.loc[(use_data['Bin'] <= move_end[t]*10) & (use_data['Bin'] > move_init[t]*10)]
        
        if session_trials['choice'][t] == -1:
            left_states = np.append(left_states, move_data['most_likely_states'])
        elif session_trials['choice'][t] == 1.:
            right_states = np.append(right_states, move_data['most_likely_states'])
            
        # React
        react_data = use_data.loc[(use_data['Bin'] <= rt_end[t]*10) & (use_data['Bin'] > rt_init[t]*10)]
        
        if prepro(session_trials)['signed_contrast'][t] < 0:
            stim_left_states = np.append(stim_left_states, react_data['most_likely_states'])
        elif prepro(session_trials)['signed_contrast'][t] > 0:
            stim_right_states = np.append(stim_right_states, react_data['most_likely_states'])
            
    # Save all in a dataframe
    quiescence_df = pd.DataFrame(quiescence_states)
    quiescence_df['label'] = 'Quiescence'

    left_stim_df = pd.DataFrame(stim_left_states)
    left_stim_df['label'] = 'Stimulus left'

    right_stim_df = pd.DataFrame(stim_right_states)
    right_stim_df['label'] = 'Stimulus right'
    
    iti_df_correct = pd.DataFrame(ITI_states_correct)
    iti_df_correct['label'] = 'ITI_correct'

    iti_df_incorrect = pd.DataFrame(ITI_states_incorrect)
    iti_df_incorrect['label'] = 'ITI_incorrect'

    correct_df = pd.DataFrame(correct_states)
    correct_df['label'] = 'Correct feedback'

    incorrect_df = pd.DataFrame(incorrect_states)
    incorrect_df['label'] = 'Incorrect feedback'

    left_df = pd.DataFrame(left_states)
    left_df['label'] = 'Left choice'

    right_df = pd.DataFrame(right_states)
    right_df['label'] = 'Right choice'

    all_df = quiescence_df.append(left_stim_df)
    all_df = all_df.append(right_stim_df)
    all_df = all_df.append(left_df)
    all_df = all_df.append(right_df)
    all_df = all_df.append(correct_df)
    all_df = all_df.append(incorrect_df)
    all_df = all_df.append(iti_df_correct)
    all_df = all_df.append(iti_df_incorrect)

    return all_df

def bins_per_trial_phase(design_matrix, session_trials):
    # Split session into trial phases and gather most likely states of those trial phases
    use_data = design_matrix.dropna()
    use_data['Trial'] = use_data['Bin'] * np.nan
    trial_num = len(session_trials)
    
    # Quiescence
    qui_init = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']
    qui_end = session_trials['goCueTrigger_times']
    quiescence_states = []

    # ITI
    iti_init_correct = session_trials['feedback_times'] + 1
    iti_init_incorrect = session_trials['feedback_times'] + 2
    iti_end = session_trials['intervals_1']
    ITI_states_correct = []
    ITI_states_incorrect = []

    # Feedback
    feedback_init = session_trials['feedback_times']
    correct_end = session_trials['feedback_times'] + 1
    incorrect_end = session_trials['feedback_times'] + 2
    correct_states = []
    incorrect_states = []

    # Reaction time 
    rt_init = session_trials['goCueTrigger_times']
    rt_end = session_trials['firstMovement_times']
    stim_left_states = []
    stim_right_states = []

    # Movement time 
    move_init = session_trials['firstMovement_times']
    move_end = session_trials['feedback_times']
    left_states = []
    right_states = []

    for t, trial in enumerate(range(trial_num)):
        
        # Trial number   
        use_data.loc[(use_data['Bin'] <= iti_end[t]*10) & (use_data['Bin'] > qui_init[t]*10), 'Trial'] = trial
        
        # Quiescence
        quiescence_data = use_data.loc[(use_data['Bin'] <= qui_end[t]*10) & (use_data['Bin'] > qui_init[t]*10)]
        quiescence_states = np.append(quiescence_states, quiescence_data['Bin'])
        
        # Feedback
        if session_trials['feedbackType'][t] == 1.:
            correct_data = use_data.loc[(use_data['Bin'] <= correct_end[t]*10) & (use_data['Bin'] > feedback_init[t]*10)]
            correct_states = np.append(correct_states, correct_data['Bin'])
            
            # ITI correct
            ITI_data_correct = use_data.loc[(use_data['Bin'] <= iti_end[t]*10) & (use_data['Bin'] > iti_init_correct[t]*10)]
            ITI_states_correct = np.append(ITI_states_correct, ITI_data_correct['Bin'])
        
        elif session_trials['feedbackType'][t] == -1.:
            incorrect_data = use_data.loc[(use_data['Bin'] <= incorrect_end[t]*10) & (use_data['Bin'] > feedback_init[t]*10)]
            incorrect_states = np.append(incorrect_states, incorrect_data['Bin'])
            
            # ITI incorrect
            ITI_data_incorrect = use_data.loc[(use_data['Bin'] <= iti_end[t]*10) & (use_data['Bin'] > iti_init_incorrect[t]*10)]
            ITI_states_incorrect = np.append(ITI_states_incorrect, ITI_data_incorrect['Bin'])

        # Move
        move_data = use_data.loc[(use_data['Bin'] <= move_end[t]*10) & (use_data['Bin'] > move_init[t]*10)]
        
        if session_trials['choice'][t] == -1:
            left_states = np.append(left_states, move_data['Bin'])
        elif session_trials['choice'][t] == 1.:
            right_states = np.append(right_states, move_data['Bin'])
            
        # React
        react_data = use_data.loc[(use_data['Bin'] <= rt_end[t]*10) & (use_data['Bin'] > rt_init[t]*10)]
        
        if prepro(session_trials)['signed_contrast'][t] < 0:
            stim_left_states = np.append(stim_left_states, react_data['Bin'])
        elif prepro(session_trials)['signed_contrast'][t] > 0:
            stim_right_states = np.append(stim_right_states, react_data['Bin'])
            
    # Save all in a dataframe
    quiescence_df = pd.DataFrame(quiescence_states)
    quiescence_df['label'] = 'Quiescence'

    left_stim_df = pd.DataFrame(stim_left_states)
    left_stim_df['label'] = 'Stimulus left'

    right_stim_df = pd.DataFrame(stim_right_states)
    right_stim_df['label'] = 'Stimulus right'

    iti_df_correct = pd.DataFrame(ITI_states_correct)
    iti_df_correct['label'] = 'ITI_correct'

    iti_df_incorrect = pd.DataFrame(ITI_states_incorrect)
    iti_df_incorrect['label'] = 'ITI_incorrect'

    correct_df = pd.DataFrame(correct_states)
    correct_df['label'] = 'Correct feedback'

    incorrect_df = pd.DataFrame(incorrect_states)
    incorrect_df['label'] = 'Incorrect feedback'

    left_df = pd.DataFrame(left_states)
    left_df['label'] = 'Left choice'

    right_df = pd.DataFrame(right_states)
    right_df['label'] = 'Right choice'
    
    
    all_df = quiescence_df.append(left_stim_df)
    all_df = all_df.append(right_stim_df)
    all_df = all_df.append(left_df)
    all_df = all_df.append(right_df)
    all_df = all_df.append(correct_df)
    all_df = all_df.append(incorrect_df)
    all_df = all_df.append(iti_df_correct)
    all_df = all_df.append(iti_df_incorrect)
    
    all_df = all_df.rename(columns={0: "Bin"})
    # Merge trials
    all_df = all_df.merge(use_data[['Bin', 'Trial']], on='Bin')
    return all_df

    
def plot_states_aligned_trial(trial_init, empirical_data, session_trials, bin_size, trials_to_plot, num_states):

    # PLOT
    fig, axs = plt.subplots(nrows=trials_to_plot, ncols=1, sharex=True, sharey=True, figsize=[8, 6])

    plt.rc('font', size=12)
    use_data = empirical_data.dropna()
    use_data['new_bin'] = use_data['new_bin'] * bin_size

    trials = empirical_data.loc[empirical_data['new_bin']==0]
    bins = list(trials['Bin'])
    
    for t, trial in enumerate(range(trials_to_plot)):
        
        trial_bin = bins[trial_init + t]
        bin_data = use_data.loc[(use_data['Bin']<trial_bin + 15) & (use_data['Bin']> trial_bin - 10)]
        trial_data = session_trials.loc[(session_trials['goCueTrigger_times']< trial_bin/10+2) & 
                                        (session_trials['goCueTrigger_times']> trial_bin/10-2)]
        
        # # Plot trial
        # axs[t].imshow(
        #     np.array(bin_data['most_likely_states'])[None,:], 
        #     extent=(0, len(bin_data['most_likely_states']), -1, 1),
        #     aspect="auto",
        #     cmap=colors,
        #     alpha=0.3) 
        # Hacky solution to make sure color palette is used properly
        attach_array = np.arange(0, num_states, 1)
        # axs[t].imshow(
        #     np.concatenate([bin_data['most_likely_states'], [0, 1, 2, 3, 4, 5]])[None,:], 
        #     extent=(0, len(np.concatenate([bin_data['most_likely_states'], [0, 1, 2, 3, 4, 5]])), -1, 1),
        #     aspect="auto",
        #     cmap='viridis',
        #     alpha=0.3) 
        axs[t].imshow(
            np.concatenate([bin_data['most_likely_states'], attach_array])[None,:], 
            extent=(0, len(np.concatenate([bin_data['most_likely_states'], attach_array])), -1, 1),
            aspect="auto",
            cmap='viridis',
            alpha=0.3) 


        axs[t].vlines(9,-1, 1, label='Stim On', color='Black', linewidth=2)
        axs[t].vlines(np.array(trial_data.loc[trial_data['feedbackType']==1, 'feedback_times'] * 10) - 
                      trial_bin + 10, -1, 1, label='Correct', color='Green', linewidth=2)
        axs[t].vlines(np.array(trial_data.loc[trial_data['feedbackType']==-1, 'feedback_times'] * 10) - 
                      trial_bin + 10, -1, 1, label='Incorrect', color='Red', linewidth=2)
        axs[t].vlines(np.array(trial_data['firstMovement_times'] * 10) - trial_bin + 10, -1, 1, 
                      label='First movement', color='Blue')
        axs[t].vlines(np.array((trial_data['goCueTrigger_times'] - trial_data['quiescencePeriod']) * 10) - 
                      trial_bin + 10, -1, 1, label='Quiescence start', color='Purple')

    axs[t].set_yticks([] ,[])
    axs[t].set_xticks([0, 9, 19] ,[-0.9, 0, 1])
    axs[t].set_xlabel(str('Time from go cue (s)'))
    axs[t].set_xlim([0, 24])

    axs[t].legend(loc='upper left', bbox_to_anchor=(1, -0.5))
    plt.show()


def traces_over_sates (init, design_matrix, most_likely_states, session_trials, columns_to_standardize):
    # Compute the most likely states

    end = init + 200

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(15, 8))

    use_data = design_matrix.copy()

    # Standardization using StandardScaler
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(use_data[columns_to_standardize])
    df_standardized = pd.DataFrame(standardized_data, columns=columns_to_standardize)

    # Normalization using MinMaxScaler
    min_max_scaler = MinMaxScaler()
    normalized_data = min_max_scaler.fit_transform(df_standardized)
    df_normalized = pd.DataFrame(normalized_data, columns=columns_to_standardize)

    df_normalized['Bin'] = design_matrix['Bin']

    shw = axs[0].imshow(
        most_likely_states[None,:], 
        extent=(0, len(most_likely_states), -1, 1),
        aspect="auto",
        cmap='viridis',
        alpha=0.3) 

    axs[0].vlines(np.array(session_trials['goCueTrigger_times'] * 10),-1, 1, label='Stim On', color='Black', linewidth=2)
    axs[0].vlines(np.array(session_trials.loc[session_trials['feedbackType']==1, 'feedback_times'] * 10), -1, 1, label='Correct', color='Green', linewidth=2)
    axs[0].vlines(np.array(session_trials.loc[session_trials['feedbackType']==-1, 'feedback_times'] * 10), -1, 1, label='Incorrect', color='Red', linewidth=2)
    axs[0].vlines(np.array(session_trials['firstMovement_times'] * 10), -1, 1, label='First movement', color='Blue')
    axs[0].vlines(np.array(session_trials['intervals_0'] * 10), -1, 1, label='Trial end', color='Grey', linewidth=2)
    axs[0].vlines(np.array((session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']) * 10), -1, 1, label='Quiescence start', color='Pink', linewidth=2)

    axs[0].hlines(0, init, end, color='Black', linestyles='dashed', linewidth=2)

    # Plot original values
    axs[0].plot(df_normalized['Bin'], df_normalized['whisker_me'], label='Whisker ME', linewidth=2)
    axs[0].plot(df_normalized['Bin'], df_normalized['nose_speed_X'], label='Nose speed', linewidth=2)
    axs[0].plot(df_normalized['Bin'], df_normalized['Gaussian_licks'], label='Licks', linewidth=2)

    axs[1].imshow(
        most_likely_states[None,:], 
        extent=(0, len(most_likely_states), -1, 1),
        aspect="auto",
        cmap='viridis',
        alpha=0.3) 

    axs[1].vlines(np.array(session_trials['goCueTrigger_times'] * 10),-1, 1, color='Black', linewidth=2)
    axs[1].vlines(np.array(session_trials.loc[session_trials['feedbackType']==1, 'feedback_times'] * 10), -1, 1, color='Green', linewidth=2)
    axs[1].vlines(np.array(session_trials.loc[session_trials['feedbackType']==-1, 'feedback_times'] * 10), -1, 1, color='Red', linewidth=2)
    axs[1].vlines(np.array(session_trials['firstMovement_times'] * 10), -1, 1, color='Blue')
    axs[1].vlines(np.array(session_trials['intervals_0'] * 10), -1, 1, color='Grey', linewidth=2)
    axs[1].vlines(np.array((session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']) * 10), -1, 1, color='Pink', linewidth=2)
    axs[1].hlines(0, init, end, color='Black', linestyles='dashed', linewidth=2)

    # Plot original values
    axs[1].plot(df_normalized['Bin'], df_normalized['avg_wheel_vel'], label='Wheel velocity', linewidth=2)
    axs[1].plot(df_normalized['Bin'], df_normalized['l_paw_speed_X'], label='Paw speed', linewidth=2)

    axs[2].imshow(
        most_likely_states[None,:], 
        extent=(0, len(most_likely_states), -1, 1),
        aspect="auto",
        cmap='viridis',
        alpha=0.3) 

    axs[2].vlines(np.array(session_trials['goCueTrigger_times'] * 10),-1, 1, color='Black', linewidth=2)
    axs[2].vlines(np.array(session_trials.loc[session_trials['feedbackType']==1, 'feedback_times'] * 10), -1, 1, color='Green', linewidth=2)
    axs[2].vlines(np.array(session_trials.loc[session_trials['feedbackType']==-1, 'feedback_times'] * 10), -1, 1, color='Red', linewidth=2)
    axs[2].vlines(np.array(session_trials['firstMovement_times'] * 10), -1, 1, color='Blue')
    axs[2].vlines(np.array(session_trials['intervals_0'] * 10), -1, 1, label='Trial end',linewidth=2)
    axs[2].vlines(np.array((session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']) * 10), -1, 1, color='Pink', linewidth=2)
    axs[2].hlines(0, init, end, color='Black', linestyles='dashed', linewidth=2)

    # Plot original values
    axs[2].plot(df_normalized['Bin'], df_normalized['pupil_diameter'], label='Pupil diameter', linewidth=2)
    axs[2].plot(df_normalized['Bin'], df_normalized['pupil_speed_X'], label='Pupil speed', linewidth=2)


    axs[0].set_ylim(0, 1)

    axs[0].set_ylabel("emissions")
    axs[1].set_ylabel("emissions")
    axs[2].set_ylabel("emissions")
    axs[2].set_xlabel("time (s)")
    axs[0].set_xlim(init, end)
    axs[0].set_xticks(np.arange(init, end+50, 50),np.arange(init/10, end/10+5, 5))

    axs[0].set_title("inferred states")
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()
    

    
#def compute_model_perf_metrics ():
    
    