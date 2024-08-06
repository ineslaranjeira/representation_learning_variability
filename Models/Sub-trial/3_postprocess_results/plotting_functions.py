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
        reduced_design_matrix['contrast'] = reduced_design_matrix['Bin'] * np.nan

        events = session_trials[this_event]
        feedback = session_trials['feedbackType']
        choice = session_trials['choice']
        contrast = np.abs(prepro(session_trials)['signed_contrast'])
        
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
            
            # Check contrast
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) &
                                            (reduced_design_matrix['Bin']> event*multiplier + init), 
                                            'contrast'] = contrast[t]
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
                        multiple="stack", binwidth=bin_size, binrange=(bin_size*init+0.01, bin_size*end), legend=False, ax = ax[0, 0], palette='viridis')
        # Correct right
        b = sns.histplot(x='new_bin', hue='most_likely_states', data=use_data.loc[(use_data['correct']==1) 
                                                                                & (use_data['choice']=='right')], stat='count', alpha=0.3, 
                        multiple="stack", binwidth=bin_size, binrange=(bin_size*init+0.01, bin_size*end), legend=False, ax = ax[0, 1], palette='viridis')
        # Incorrect left
        c = sns.histplot(x='new_bin', hue='most_likely_states', data=use_data.loc[(use_data['correct']==0) 
                                                                                & (use_data['choice']=='left')], stat='count', alpha=0.3,
                        multiple="stack", binwidth=bin_size, binrange=(bin_size*init+0.01, bin_size*end), legend=False, ax = ax[1, 0], palette='viridis')
        # Incorrect right
        d = sns.histplot(x='new_bin', hue='most_likely_states', data=use_data.loc[(use_data['correct']==0) 
                                                                                & (use_data['choice']=='right')], alpha=0.3, 
                        stat='count', multiple="stack", binwidth=bin_size, binrange=(bin_size*init+0.01, bin_size*end), legend=False, ax = ax[1, 1], palette='viridis')
        
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
    

def states_per_trial_phase(reduced_design_matrix, session_trials, multiplier):
    
    # Split session into trial phases and gather most likely states of those trial phases
    use_data = reduced_design_matrix.dropna()
    trial_num = len(session_trials)

    # Quiescence
    qui_init = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']
    qui_end = session_trials['goCueTrigger_times']
    quiescence_states = pd.DataFrame(columns=['correct', 'choice', 'contrast', 'most_likely_states', 'Bin'])
    
    # ITI
    iti_init_correct = session_trials['feedback_times'] + 1
    iti_init_incorrect = session_trials['feedback_times'] + 2
    iti_end = session_trials['intervals_1']
    ITI_states_correct = pd.DataFrame(columns=['correct', 'choice', 'contrast', 'most_likely_states', 'Bin'])
    ITI_states_incorrect = pd.DataFrame(columns=['correct', 'choice', 'contrast', 'most_likely_states', 'Bin'])

    # Feedback
    feedback_init = session_trials['feedback_times']
    correct_end = session_trials['feedback_times'] + 1
    incorrect_end = session_trials['feedback_times'] + 2
    correct_states = pd.DataFrame(columns=['correct', 'choice', 'contrast', 'most_likely_states', 'Bin'])
    incorrect_states = pd.DataFrame(columns=['correct', 'choice', 'contrast', 'most_likely_states', 'Bin'])

    # Reaction time 
    rt_init = session_trials['goCueTrigger_times']
    rt_end = session_trials['firstMovement_times']
    stim_left_states = pd.DataFrame(columns=['correct', 'choice', 'contrast', 'most_likely_states', 'Bin'])
    stim_right_states = pd.DataFrame(columns=['correct', 'choice', 'contrast', 'most_likely_states', 'Bin'])

    # Movement time 
    move_init = session_trials['firstMovement_times']
    move_end = session_trials['feedback_times']
    left_states = pd.DataFrame(columns=['correct', 'choice', 'contrast', 'most_likely_states', 'Bin'])
    right_states = pd.DataFrame(columns=['correct', 'choice', 'contrast', 'most_likely_states', 'Bin'])
    

    for t, trial in enumerate(range(trial_num)):
        
        # Quiescence
        quiescence_data = use_data.loc[(use_data['Bin'] <= qui_end[t]*multiplier) & (use_data['Bin'] > qui_init[t]*multiplier)]
        quiescence_states = quiescence_states.append(quiescence_data[['correct', 'choice', 'contrast', 'most_likely_states', 'Bin']])
        
        # Feedback        
        if session_trials['feedbackType'][t] == 1.:
            
            correct_data = use_data.loc[(use_data['Bin'] <= correct_end[t]*multiplier) & (use_data['Bin'] > feedback_init[t]*multiplier)]
            correct_states = correct_states.append(correct_data[['correct', 'choice', 'contrast', 'most_likely_states', 'Bin']])
            
            # ITI correct
            ITI_data_correct = use_data.loc[(use_data['Bin'] <= iti_end[t]*multiplier) & (use_data['Bin'] > iti_init_correct[t]*multiplier)]
            ITI_states_correct = ITI_states_correct.append(ITI_data_correct[['correct', 'choice', 'contrast', 'most_likely_states', 'Bin']])

        elif session_trials['feedbackType'][t] == -1.:
            incorrect_data = use_data.loc[(use_data['Bin'] <= incorrect_end[t]*multiplier) & (use_data['Bin'] > feedback_init[t]*multiplier)]
            incorrect_states =incorrect_states.append(incorrect_data[['correct', 'choice', 'contrast', 'most_likely_states', 'Bin']])

            # ITI incorrect
            ITI_data_incorrect = use_data.loc[(use_data['Bin'] <= iti_end[t]*multiplier) & (use_data['Bin'] > iti_init_incorrect[t]*multiplier)]
            ITI_states_incorrect = ITI_states_incorrect.append(ITI_data_incorrect[['correct', 'choice', 'contrast', 'most_likely_states', 'Bin']])

        # Move
        move_data = use_data.loc[(use_data['Bin'] <= move_end[t]*multiplier) & (use_data['Bin'] > move_init[t]*multiplier)]
        
        if session_trials['choice'][t] == -1:
            left_states = left_states.append(move_data[['correct', 'choice', 'contrast', 'most_likely_states', 'Bin']])
        elif session_trials['choice'][t] == 1.:
            right_states = right_states.append(move_data[['correct', 'choice', 'contrast', 'most_likely_states', 'Bin']])
            
        # React
        react_data = use_data.loc[(use_data['Bin'] <= rt_end[t]*multiplier) & (use_data['Bin'] > rt_init[t]*multiplier)]
        
        if prepro(session_trials)['signed_contrast'][t] < 0:
            stim_left_states = stim_left_states.append(react_data[['correct', 'choice', 'contrast', 'most_likely_states', 'Bin']])
        elif prepro(session_trials)['signed_contrast'][t] > 0:
            stim_right_states = stim_right_states.append(react_data[['correct', 'choice', 'contrast', 'most_likely_states', 'Bin']])

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
    use_data['feedback'] = use_data['Bin'] * np.nan
    use_data['signed_contrast'] = use_data['Bin'] * np.nan
    use_data['choice'] = use_data['Bin'] * np.nan

    trial_num = len(session_trials)
    
    # Quiescence
    qui_init = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']
    qui_end = session_trials['goCueTrigger_times']
    quiescence_states = []

    # ITI
    iti_init_correct = session_trials['feedback_times'] + 1
    iti_init_incorrect = session_trials['feedback_times'] + 1
    iti_end = session_trials['intervals_1']
    ITI_states_correct = []
    ITI_states_incorrect = []

    # Feedback
    feedback_init = session_trials['feedback_times']
    correct_end = session_trials['feedback_times'] + 1
    incorrect_end = session_trials['feedback_times'] + 1
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
        
        # Choice   
        choice = session_trials['choice'][t]
        use_data.loc[(use_data['Bin'] <= iti_end[t]*10) & (use_data['Bin'] > qui_init[t]*10), 'choice'] = choice
        
        # Correct   
        correct = prepro(session_trials)['correct'][t]
        use_data.loc[(use_data['Bin'] <= iti_end[t]*10) & (use_data['Bin'] > qui_init[t]*10), 'feedback'] = correct
        
        # Sided contrast  
        contrast = prepro(session_trials)['signed_contrast'][t]
        use_data.loc[(use_data['Bin'] <= iti_end[t]*10) & (use_data['Bin'] > qui_init[t]*10), 'signed_contrast'] = contrast
        
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
    all_df = all_df.merge(use_data[['Bin', 'Trial', 'feedback', 'signed_contrast', 'choice']], on='Bin')
    return all_df


def broader_label(df):
    
    df['broader_label'] = df['label']
    df.loc[df['broader_label']=='Stimulus right', 'broader_label'] = 'Stimulus'
    df.loc[df['broader_label']=='Stimulus left', 'broader_label'] = 'Stimulus'
    df.loc[df['broader_label']=='Quiescence', 'broader_label'] = 'Quiescence'
    df.loc[df['broader_label']=='Left choice', 'broader_label'] = 'Choice'
    df.loc[df['broader_label']=='Right choice', 'broader_label'] = 'Choice'
    df.loc[df['broader_label']=='Correct feedback', 'broader_label'] = 'ITI'
    df.loc[df['broader_label']=='Incorrect feedback', 'broader_label'] = 'ITI'
    df.loc[df['broader_label']=='ITI_correct', 'broader_label'] = 'ITI'
    df.loc[df['broader_label']=='ITI_incorrect', 'broader_label'] = 'ITI'
    
    return df


def plot_states_aligned_trial(trial_init, empirical_data, session_trials, bin_size, trials_to_plot, num_states):
    
    # PLOT
    fig, axs = plt.subplots(nrows=trials_to_plot, ncols=1, sharex=True, sharey=True, figsize=[8, 6])

    plt.rc('font', size=12)
    use_data = empirical_data.dropna()
    use_data['new_bin'] = use_data['new_bin'] * bin_size
    multiplier = 1/bin_size
    
    trials = empirical_data.loc[empirical_data['new_bin']==0]
    bins = list(trials['Bin'])
    
    for t, trial in enumerate(range(trials_to_plot)):
        
        trial_bin = bins[trial_init + t]
        bin_data = use_data.loc[(use_data['Bin']<trial_bin + 1.4*multiplier) & (use_data['Bin']> trial_bin - 1*multiplier)]
        trial_data = session_trials.loc[(session_trials['goCueTrigger_times']< trial_bin/(multiplier*1)+2) & 
                                        (session_trials['goCueTrigger_times']> trial_bin/(multiplier*1)-2)]
        
        # # Plot trial
        # Hacky solution to make sure color palette is used properly
        attach_array1 = np.arange(0, len(use_data['most_likely_states'].unique()), 1)
        attach_array = np.concatenate([np.arange(0, 10, 1)*np.nan, attach_array1])
        axs[t].imshow(
            np.concatenate([bin_data['most_likely_states'], attach_array])[None,:], 
            extent=(0, len(np.concatenate([bin_data['most_likely_states'], attach_array])), -1, 1),
            aspect="auto",
            cmap='viridis',
            alpha=0.3) 


        axs[t].vlines(.9*multiplier,-1, 1, label='Stim On', color='Black', linewidth=2)
        axs[t].vlines(np.array(trial_data.loc[trial_data['feedbackType']==1, 'feedback_times'] * multiplier) - 
                      trial_bin + 1.0*multiplier, -1, 1, label='Correct', color='Green', linewidth=2)
        axs[t].vlines(np.array(trial_data.loc[trial_data['feedbackType']==-1, 'feedback_times'] * multiplier) - 
                      trial_bin + 1.0*multiplier, -1, 1, label='Incorrect', color='Red', linewidth=2)
        axs[t].vlines(np.array(trial_data['firstMovement_times'] * multiplier) - trial_bin + 1*multiplier, -1, 1, 
                      label='First movement', color='Blue')
        axs[t].vlines(np.array((trial_data['goCueTrigger_times'] - trial_data['quiescencePeriod']) * multiplier) - 
                      trial_bin + 1.0*multiplier, -1, 1, label='Quiescence start', color='Purple')

    axs[t].set_yticks([] ,[])
    axs[t].set_xticks([0, .9*multiplier, 1.9*multiplier] ,[-0.9, 0, 1])
    axs[t].set_xlabel(str('Time from go cue (s)'))
    axs[t].set_xlim([0, 2.4*multiplier])

    axs[t].legend(loc='upper left', bbox_to_anchor=(1, -0.5))
    plt.show()


# def traces_over_sates (init, interval, design_matrix, session_trials):
    
#     # Compute the most likely states
#     # design matrix arg should be empirical_data
#     end = init + interval

#     fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(15, 8))

#     df_normalized = design_matrix.copy()
#     df_normalized['Bin'] = design_matrix['Bin']
#     use_normalized = df_normalized.loc[(df_normalized['Bin']>init) & (df_normalized['Bin']<end)]

#     axs[0].imshow(use_normalized['most_likely_states'][None,:], 
#             extent=(0, len(use_normalized['most_likely_states']), -1, 1),
#             aspect="auto",
#             cmap='viridis',
#             alpha=0.3) 
#     axs[0].vlines(np.array(session_trials['goCueTrigger_times'] * 10)-init,-1, 1, label='Stim On', color='Black', linewidth=2)
#     axs[0].vlines(np.array(session_trials.loc[session_trials['feedbackType']==1, 'feedback_times'] * 10)-init, -1, 1, label='Correct', color='Green', linewidth=2)
#     axs[0].vlines(np.array(session_trials.loc[session_trials['feedbackType']==-1, 'feedback_times'] * 10)-init, -1, 1, label='Incorrect', color='Red', linewidth=2)
#     axs[0].vlines(np.array(session_trials['firstMovement_times'] * 10)-init, -1, 1, label='First movement', color='Blue')
#     axs[0].vlines(np.array(session_trials['intervals_0'] * 10)-init, -1, 1, label='Trial end', color='Grey', linewidth=2)
#     axs[0].vlines(np.array((session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']) * 10)-init, -1, 1, label='Quiescence start', color='Pink', linewidth=2)

#     axs[0].hlines(0, init, end, color='Black', linestyles='dashed', linewidth=2)

#     # Plot original values
#     axs[0].plot(df_normalized['Bin']-init, df_normalized['avg_wheel_vel'], label='Wheel velocity', linewidth=2)
#     axs[0].plot(df_normalized['Bin']-init, df_normalized['l_paw_speed'], label='Paw speed', linewidth=2)

#     axs[1].imshow(use_normalized['most_likely_states'][None,:], 
#             extent=(0, len(use_normalized['most_likely_states']), -1, 1),
#             aspect="auto",
#             cmap='viridis',
#             alpha=0.3) 

#     axs[1].vlines(np.array(session_trials['goCueTrigger_times'] * 10)-init,-1, 1, color='Black', linewidth=2)
#     axs[1].vlines(np.array(session_trials.loc[session_trials['feedbackType']==1, 'feedback_times'] * 10)-init, -1, 1, color='Green', linewidth=2)
#     axs[1].vlines(np.array(session_trials.loc[session_trials['feedbackType']==-1, 'feedback_times'] * 10)-init, -1, 1, color='Red', linewidth=2)
#     axs[1].vlines(np.array(session_trials['firstMovement_times'] * 10)-init, -1, 1, color='Blue')
#     axs[1].vlines(np.array(session_trials['intervals_0'] * 10)-init, -1, 1, color='Grey', linewidth=2)
#     axs[1].vlines(np.array((session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']) * 10)-init, -1, 1, color='Pink', linewidth=2)
#     axs[1].hlines(0, init, end, color='Black', linestyles='dashed', linewidth=2)

#     # Plot original values
#     axs[1].plot(df_normalized['Bin']-init, df_normalized['whisker_me'], label='Whisker ME', linewidth=2)
#     axs[1].plot(df_normalized['Bin']-init, df_normalized['nose_speed'], label='Nose speed', linewidth=2)
#     axs[1].plot(df_normalized['Bin']-init, df_normalized['Lick count'], label='Licks', linewidth=2)

#     axs[2].imshow(use_normalized['most_likely_states'][None,:], 
#             extent=(0, len(use_normalized['most_likely_states']), -1, 1),
#             aspect="auto",
#             cmap='viridis',
#             alpha=0.3) 

#     axs[2].vlines(np.array(session_trials['goCueTrigger_times'] * 10)-init,-1, 1, color='Black', linewidth=2)
#     axs[2].vlines(np.array(session_trials.loc[session_trials['feedbackType']==1, 'feedback_times'] * 10)-init, -1, 1, color='Green', linewidth=2)
#     axs[2].vlines(np.array(session_trials.loc[session_trials['feedbackType']==-1, 'feedback_times'] * 10)-init, -1, 1, color='Red', linewidth=2)
#     axs[2].vlines(np.array(session_trials['firstMovement_times'] * 10)-init, -1, 1, color='Blue')
#     axs[2].vlines(np.array(session_trials['intervals_0'] * 10)-init, -1, 1, color='Grey', linewidth=2)
#     axs[2].vlines(np.array((session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']) * 10)-init, -1, 1, color='Pink', linewidth=2)
#     axs[2].hlines(0, init, end, color='Black', linestyles='dashed', linewidth=2)

#     # Plot original values
#     axs[2].plot(df_normalized['Bin']-init, df_normalized['pupil_diameter'], label='Pupil diameter', linewidth=2)
#     axs[2].plot(df_normalized['Bin']-init, df_normalized['pupil_speed'], label='Pupil speed', linewidth=2)


#     axs[0].set_ylim(-1, 1)

#     axs[0].set_ylabel("emissions")
#     axs[1].set_ylabel("emissions")
#     axs[2].set_ylabel("emissions")
#     axs[2].set_xlabel("time (s)")
#     axs[0].set_xlim(0, end-init)
#     axs[0].set_xticks(np.arange(0, end-init+50, 50),np.arange(init/10, end/10+5, 5))
#     axs[0].set_title("inferred states")
#     axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
#     axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
#     axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

#     plt.tight_layout()
#     plt.show()
    

def traces_over_sates (init, interval, design_matrix, session_trials, multiplier):
    
    # Compute the most likely states
    # design matrix arg should be empirical_data
    end = init + interval

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(15, 8))

    df_normalized = design_matrix.copy()
    df_normalized['Bin'] = design_matrix['Bin']
    use_normalized = df_normalized.loc[(df_normalized['Bin']>init) & (df_normalized['Bin']<end)]
    
    # To make sure color code is used correctly
    number_of_states = len(use_normalized['most_likely_states'].unique()) - np.sum(np.isnan(use_normalized['most_likely_states'].unique()))
    states_to_append = np.arange(0, number_of_states, 1)
    
    
    plot_max = np.max([use_normalized['avg_wheel_vel'], use_normalized['whisker_me'],
                       use_normalized['nose_X'], use_normalized['nose_Y'],
                       use_normalized['Lick count']])
    plot_min = np.min([use_normalized['avg_wheel_vel'], use_normalized['whisker_me'],
                       use_normalized['nose_X'], use_normalized['nose_Y'],
                       use_normalized['Lick count']])
    
    axs[0].imshow(np.concatenate([use_normalized['most_likely_states'], states_to_append])[None,:],  
            extent=(0, len(np.concatenate([use_normalized['most_likely_states'], states_to_append])), 
                    plot_min, plot_max),
            aspect="auto",
            cmap='viridis',
            alpha=0.3) 
    axs[0].vlines(np.array(session_trials['goCueTrigger_times'] * 1*multiplier)-init, plot_min, 
                  plot_max, label='Stim On', color='Black', linewidth=2)
    axs[0].vlines(np.array(session_trials.loc[session_trials['feedbackType']==1, 'feedback_times'] * 1*multiplier)-init, 
                  plot_min, plot_max, label='Correct', color='Green', linewidth=2)
    axs[0].vlines(np.array(session_trials.loc[session_trials['feedbackType']==-1, 'feedback_times'] * 1*multiplier)-init, plot_min, 
                  plot_max, label='Incorrect', color='Red', linewidth=2)
    axs[0].vlines(np.array(session_trials['firstMovement_times'] * 1*multiplier)-init, plot_min, plot_max, label='First movement', color='Blue')
    axs[0].vlines(np.array(session_trials['intervals_0'] * 10)-init, plot_min, plot_max, label='Trial end', color='Grey', linewidth=2)
    axs[0].vlines(np.array((session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']) * 1*multiplier)-init, 
                  plot_min, plot_max, label='Quiescence start', color='Pink', linewidth=2)

    axs[0].hlines(0, init, end, color='Black', linestyles='dashed', linewidth=2)

    # Plot original values
    axs[0].plot(df_normalized['Bin']-init, df_normalized['avg_wheel_vel'], label='Wheel velocity', linewidth=2)
    # axs[0].plot(df_normalized['Bin']-init, df_normalized['l_paw_speed'], label='Paw speed', linewidth=2)

    axs[1].imshow(np.concatenate([use_normalized['most_likely_states'], states_to_append])[None,:], 
            extent=(0, len(np.concatenate([use_normalized['most_likely_states'], states_to_append])), plot_min, plot_max),
            aspect="auto",
            cmap='viridis',
            alpha=0.3) 

    axs[1].vlines(np.array(session_trials['goCueTrigger_times'] * 1*multiplier)-init, plot_min, plot_max, color='Black', linewidth=2)
    axs[1].vlines(np.array(session_trials.loc[session_trials['feedbackType']==1, 'feedback_times'] * 1*multiplier)-init, 
                  plot_min, plot_max, color='Green', linewidth=2)
    axs[1].vlines(np.array(session_trials.loc[session_trials['feedbackType']==-1, 'feedback_times'] * 1*multiplier)-init, 
                  plot_min, plot_max, color='Red', linewidth=2)
    axs[1].vlines(np.array(session_trials['firstMovement_times'] * 1*multiplier)-init, plot_min, plot_max, color='Blue')
    axs[1].vlines(np.array(session_trials['intervals_0'] * 1*multiplier)-init, plot_min, plot_max, color='Grey', linewidth=2)
    axs[1].vlines(np.array((session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']) * 1*multiplier)-init, 
                  plot_min, plot_max, color='Pink', linewidth=2)
    axs[1].hlines(0, init, end, color='Black', linestyles='dashed', linewidth=2)

    # Plot original values
    axs[1].plot(df_normalized['Bin']-init, df_normalized['whisker_me'], label='Whisker ME', linewidth=2)
    axs[1].plot(df_normalized['Bin']-init, df_normalized['nose_X'], label='Nose X', linewidth=2)
    axs[1].plot(df_normalized['Bin']-init, df_normalized['nose_Y'], label='Nose Y', linewidth=2)

    axs[2].imshow(np.concatenate([use_normalized['most_likely_states'], states_to_append])[None,:], 
            extent=(0, len(np.concatenate([use_normalized['most_likely_states'], states_to_append])), plot_min, plot_max),
            aspect="auto",
            cmap='viridis',
            alpha=0.3) 

    axs[2].vlines(np.array(session_trials['goCueTrigger_times'] * 1*multiplier)-init, plot_min, plot_max, color='Black', linewidth=2)
    axs[2].vlines(np.array(session_trials.loc[session_trials['feedbackType']==1, 'feedback_times'] * 1*multiplier)-init, 
                  plot_min, plot_max, color='Green', linewidth=2)
    axs[2].vlines(np.array(session_trials.loc[session_trials['feedbackType']==-1, 'feedback_times'] * 1*multiplier)-init, 
                  plot_min, plot_max, color='Red', linewidth=2)
    axs[2].vlines(np.array(session_trials['firstMovement_times'] * 1*multiplier)-init, plot_min, plot_max, color='Blue')
    axs[2].vlines(np.array(session_trials['intervals_0'] * 1*multiplier)-init, plot_min, plot_max, color='Grey', linewidth=2)
    axs[2].vlines(np.array((session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']) * 1*multiplier)-init, 
                  plot_min, plot_max, color='Pink', linewidth=2)
    axs[2].hlines(0, init, end, color='Black', linestyles='dashed', linewidth=2)

    # Plot original values
    axs[2].plot(df_normalized['Bin']-init, df_normalized['Lick count'], label='Lick count', linewidth=2)


    axs[0].set_ylim(plot_min, plot_max)

    axs[0].set_ylabel("emissions")
    axs[1].set_ylabel("emissions")
    axs[2].set_ylabel("emissions")
    axs[2].set_xlabel("time (s)")
    # axs[0].set_xlim(0, end-init)
    # axs[0].set_xticks(np.arange(0, end-init+50, 50),np.arange(init/10, end/10+5, 5))
    # axs[0].set_xticks(np.arange(0, end-init+50, 50),np.arange(init/10, end/10+5, 5))

    axs[0].set_title("inferred states")
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()
    
    
def traces_over_few_sates (init, inter, design_matrix, session_trials, columns_to_standardize, multiplier):
    # Compute the most likely states
    
    end = init + inter

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(12, 4))

    df_normalized = design_matrix
    df_normalized['Bin'] = design_matrix['Bin']
    
    use_normalized = df_normalized.loc[(df_normalized['Bin']>init) & (df_normalized['Bin']<end)]
    
    # To make sure color code is used correctly
    number_of_states = len(use_normalized['most_likely_states'].unique()) - np.sum(np.isnan(use_normalized['most_likely_states'].unique()))
    states_to_append = np.arange(0, number_of_states, 1)
    
    # Plot original values
    if len(columns_to_standardize) == 2:
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[0]], label=columns_to_standardize[0], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[1]], label=columns_to_standardize[1], linewidth=2)
        plot_max = np.max([use_normalized[columns_to_standardize[0]], 
                           use_normalized[columns_to_standardize[1]]])
        plot_min = np.min([use_normalized[columns_to_standardize[0]], 
                           use_normalized[columns_to_standardize[1]]])
    elif len(columns_to_standardize) == 1:
        use_index0 = ~np.isnan(use_normalized[columns_to_standardize[0]])
        use_time = np.arange(0, len(use_index0), 1)  # NOTE! If there are NaNs, x axis label will have the time wrong # NOTE!!
        # axs.plot(use_normalized['Bin'][use_index0]-init, use_normalized[columns_to_standardize[0]][use_index0], label=columns_to_standardize[0], linewidth=2)
        axs.plot(use_time, use_normalized[columns_to_standardize[0]][use_index0], label=columns_to_standardize[0], linewidth=2)
        plot_max = np.max(use_normalized[columns_to_standardize[0]])
        plot_min = np.min(use_normalized[columns_to_standardize[0]])
    elif len(columns_to_standardize) == 3:
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[0]], label=columns_to_standardize[0], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[1]], label=columns_to_standardize[1], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[2]], label=columns_to_standardize[2], linewidth=2)
        plot_max = np.max([use_normalized[columns_to_standardize[0]], 
                           use_normalized[columns_to_standardize[1]],
                           use_normalized[columns_to_standardize[2]]])
        plot_min = np.min([use_normalized[columns_to_standardize[0]], 
                           use_normalized[columns_to_standardize[1]],
                           use_normalized[columns_to_standardize[2]]])
    elif len(columns_to_standardize) == 4:
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[0]], label=columns_to_standardize[0], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[1]], label=columns_to_standardize[1], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[2]], label=columns_to_standardize[2], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[3]], label=columns_to_standardize[3], linewidth=2)
        plot_max = np.max([use_normalized[columns_to_standardize[0]], 
                           use_normalized[columns_to_standardize[1]],
                           use_normalized[columns_to_standardize[2]],
                           use_normalized[columns_to_standardize[3]]])
        plot_min = np.min([use_normalized[columns_to_standardize[0]], 
                           use_normalized[columns_to_standardize[1]],
                           use_normalized[columns_to_standardize[2]],
                           use_normalized[columns_to_standardize[3]]])
    axs.imshow(np.concatenate([use_normalized['most_likely_states'], states_to_append])[None,:], 
            extent=(0, len(np.concatenate([use_normalized['most_likely_states'], states_to_append])), plot_min, plot_max),
            aspect="auto",
            cmap='viridis',
            alpha=0.3) 

    axs.hlines(0, init, end, color='Black', linestyles='dashed', linewidth=2)
    axs.vlines(np.array(session_trials['goCueTrigger_times'] * 1*multiplier)-init, plot_min, plot_max, label='Stim On', 
               color='Black', linewidth=2)
    axs.vlines(np.array(session_trials.loc[session_trials['feedbackType']==1, 'feedback_times'] * 1*multiplier)-init, 
               plot_min, plot_max, label='Correct', color='Green', linewidth=2)
    axs.vlines(np.array(session_trials.loc[session_trials['feedbackType']==-1, 'feedback_times'] * 1*multiplier)-init, 
               plot_min, plot_max, label='Incorrect', color='Red', linewidth=2)
    axs.vlines(np.array(session_trials['firstMovement_times'] * 1*multiplier)-init, plot_min, plot_max, label='First movement', color='Blue')
    axs.vlines(np.array(session_trials['intervals_0'] * 1*multiplier)-init, plot_min, plot_max, label='Trial end', color='Grey', linewidth=2)
    axs.vlines(np.array((session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']) * 1*multiplier)-init, 
               plot_min, plot_max, label='Quiescence start', color='Pink', linewidth=2)

    axs.set_ylim(plot_min, plot_max)
    axs.set_ylabel("emissions")
    axs.set_xlabel("time (s)")
    axs.set_xlim(0, end-init)
    axs.set_xticks(np.arange(0, inter, inter/5),np.arange(init/multiplier, 
                                                          end/multiplier, (inter/multiplier)/5))
    axs.set_title("inferred states")
    axs.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()
    
    
def plot_avg_state(unique_states, empirical_data, inverted_mapping):
    use_vars = ['avg_wheel_vel', 'Lick count', 'whisker_me',
        'Bin', 'most_likely_states']
    
    use_data = empirical_data[use_vars].copy()
    use_data['avg_wheel_vel'] = np.abs(use_data['avg_wheel_vel'])

    melted = pd.melt(use_data, id_vars=['Bin', 'most_likely_states'], value_vars=use_vars)

    melted.loc[melted['variable']=='avg_wheel_vel', 'variable'] = 'Signed wheel speed'
    melted.loc[melted['variable']=='whisker_me', 'variable'] = 'Whisker ME'
    states = unique_states
            
    fig, ax = plt.subplots(ncols=int(np.ceil(len(states)/2)) , nrows=2, sharex=True, sharey=True, figsize=[20, 8])
    plt.rc('font', size=12)

    for s, state in enumerate(states):
        use_data = melted.loc[melted['most_likely_states']==s]
        state_label = inverted_mapping[state]
        if s < len(states)/2:  
            sns.barplot(y='variable', x='value', data=use_data, ax=ax[0,s], palette='plasma')
            ax[0,s].vlines(0, -.50, 3, color='Gray', linestyles='--', linewidth=1)
            ax[0,s].set_xlabel('Mean')
            ax[0,s].set_title(state_label)
        else:
            sns.barplot(y='variable', x='value', data=use_data, ax=ax[1,s-3], palette='plasma')
            ax[1,s-3].vlines(0, -.50, 3, color='Gray', linestyles='--', linewidth=1)
            ax[1,s-3].set_xlabel('Mean')
            ax[1,s-3].set_title(state_label)

    plt.tight_layout()
    plt.show()