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
from mpl_toolkits.axes_grid1 import make_axes_locatable

# # Custom functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Functions/'
# functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Functions/'
os.chdir(functions_path)
from one_functions_generic import prepro


################################################### INES's FUNCTIONS ############################################################

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

        feedback = session_trials['feedbackType']
        choice = session_trials['choice']
        contrast = np.abs(prepro(session_trials)['signed_contrast'])
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
        
        # Pre-quiescence
        use_data.loc[(use_data['Bin']+0.5 <= pre_qui_end[t]*multiplier) & 
                     (use_data['Bin']+0.5 > pre_qui_init[t]*multiplier), 'label'] = 'Pre-quiescence'

        # Quiescence
        use_data.loc[(use_data['Bin']+0.5 <= qui_end[t]*multiplier) &
                     (use_data['Bin']+0.5 > qui_init[t]*multiplier), 'label'] = 'Quiescence'
        
        # ITI
        if session_trials['feedbackType'][t] == -1.:
            use_data.loc[(use_data['Bin']+0.5 <= iti_end_incorrect[t]*multiplier) & 
                            (use_data['Bin']+0.5 > iti_init[t]*multiplier), 'label'] = 'ITI'
        elif session_trials['feedbackType'][t] == 1.:
            use_data.loc[(use_data['Bin']+0.5 <= iti_end_correct[t]*multiplier) & 
                            (use_data['Bin']+0.5 > iti_init[t]*multiplier), 'label'] = 'ITI'
        # Move
        if session_trials['choice'][t] == -1:
            use_data.loc[(use_data['Bin']+0.5 <= move_end[t]*multiplier) & 
                         (use_data['Bin']+0.5 > move_init[t]*multiplier), 'label'] = 'Left choice'
        elif session_trials['choice'][t] == 1.:
            use_data.loc[(use_data['Bin']+0.5 <= move_end[t]*multiplier) & 
                         (use_data['Bin']+0.5 > move_init[t]*multiplier), 'label'] = 'Right choice'
            
        # React        
        if prepro(session_trials)['signed_contrast'][t] < 0:
            use_data.loc[(use_data['Bin']+0.5 <= rt_end[t]*multiplier) & 
                         (use_data['Bin']+0.5 > rt_init[t]*multiplier), 'label'] = 'Stimulus left'
        elif prepro(session_trials)['signed_contrast'][t] > 0:
            use_data.loc[(use_data['Bin']+0.5 <= rt_end[t]*multiplier) & 
                         (use_data['Bin']+0.5 > rt_init[t]*multiplier), 'label'] = 'Stimulus right'
            
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


def plot_states_aligned_trial(trial_init, empirical_data, session_trials, bin_size, trials_to_plot, inverted_mapping):
    
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
        cax = axs[t].imshow(
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

    # divider = make_axes_locatable(axs[t])
    # cax_colorbar = divider.append_axes("right", size="5%", pad=0.5)
    cbar = fig.colorbar(cax, ax=axs)
    cbar.set_label('State')
    if len(inverted_mapping) > 0:
        # Set the ticks and labels based on the dictionary
        cbar.set_ticks(list(inverted_mapping.keys()))
        cbar.set_ticklabels(list(inverted_mapping.values()))
        
    axs[t].set_yticks([] ,[])
    axs[t].set_xticks([0, .9*multiplier, 1.9*multiplier] ,[-0.9, 0, 1])
    axs[t].set_xlabel(str('Time from go cue (s)'))
    axs[t].set_xlim([0, 2.4*multiplier])

    axs[t].legend(loc='upper left', bbox_to_anchor=(1.2, 0))
    plt.show()


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
    
    
def traces_over_few_sates (init, inter, design_matrix, session_trials, columns_to_standardize, multiplier, inverted_mapping):
    # Compute the most likely states
    
    end = init + inter

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(20, 3))

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
        use_index0 = ~np.isnan(use_normalized[columns_to_standardize[0]])
        use_time0 = np.arange(0, len(use_index0), 1)  # NOTE! If there are NaNs, x axis label will have the time wrong # NOTE!!
        use_index1 = ~np.isnan(use_normalized[columns_to_standardize[1]])
        use_time1 = np.arange(0, len(use_index1), 1)  # NOTE! If there are NaNs, x axis label will have the time wrong # NOTE!!
        use_index2 = ~np.isnan(use_normalized[columns_to_standardize[2]])
        use_time2 = np.arange(0, len(use_index2), 1)  # NOTE! If there are NaNs, x axis label will have the time wrong # NOTE!!

        axs.plot(use_time0, use_normalized[columns_to_standardize[0]], label=columns_to_standardize[0], linewidth=2)
        axs.plot(use_time1, use_normalized[columns_to_standardize[1]], label=columns_to_standardize[1], linewidth=2)
        axs.plot(use_time2, use_normalized[columns_to_standardize[2]], label=columns_to_standardize[2], linewidth=2)
        
        # axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[0]], label=columns_to_standardize[0], linewidth=2)
        # axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[1]], label=columns_to_standardize[1], linewidth=2)
        # axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[2]], label=columns_to_standardize[2], linewidth=2)
        plot_max = np.max([use_normalized[columns_to_standardize[0]], 
                           use_normalized[columns_to_standardize[1]],
                           use_normalized[columns_to_standardize[2]]])
        plot_min = np.min([use_normalized[columns_to_standardize[0]], 
                           use_normalized[columns_to_standardize[1]],
                           use_normalized[columns_to_standardize[2]]])
    elif len(columns_to_standardize) == 7:
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[0]], label=columns_to_standardize[0], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[1]], label=columns_to_standardize[1], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[2]], label=columns_to_standardize[2], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[3]], label=columns_to_standardize[3], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[4]], label=columns_to_standardize[4], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[5]], label=columns_to_standardize[5], linewidth=2)
        axs.plot(use_normalized['Bin']-init, use_normalized[columns_to_standardize[6]], label=columns_to_standardize[6], linewidth=2)
        plot_max = np.max([use_normalized[columns_to_standardize[0]], 
                           use_normalized[columns_to_standardize[1]],
                           use_normalized[columns_to_standardize[2]],
                           use_normalized[columns_to_standardize[3]],
                           use_normalized[columns_to_standardize[4]],
                           use_normalized[columns_to_standardize[5]],
                           use_normalized[columns_to_standardize[6]]])
        plot_min = np.min([use_normalized[columns_to_standardize[0]], 
                           use_normalized[columns_to_standardize[1]],
                           use_normalized[columns_to_standardize[2]],
                           use_normalized[columns_to_standardize[3]],
                           use_normalized[columns_to_standardize[4]],
                           use_normalized[columns_to_standardize[5]],
                           use_normalized[columns_to_standardize[6]]])
    cax = axs.imshow(np.concatenate([use_normalized['most_likely_states'], states_to_append])[None,:], 
            extent=(0, len(np.concatenate([use_normalized['most_likely_states'], states_to_append])), plot_min, plot_max),
            aspect="auto",
            cmap='viridis',
            alpha=0.3) 

    divider = make_axes_locatable(axs)
    cax_colorbar = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(cax, cax=cax_colorbar, ax=axs)

    cbar.set_label('State')
    if len(inverted_mapping) > 0:
        # Set the ticks and labels based on the dictionary
        cbar.set_ticks(list(inverted_mapping.keys()))
        cbar.set_ticklabels(list(inverted_mapping.values()))

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
    axs.legend(loc='upper left', bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    plt.show()
    
    
def plot_avg_state(unique_states, empirical_data, inverted_mapping):
    use_vars = ['avg_wheel_vel', 'Lick count', 'whisker_me',
        'Bin', 'most_likely_states']
    
    use_data = empirical_data[use_vars].copy()
    use_data['avg_wheel_vel'] = np.abs(use_data['avg_wheel_vel'])

    melted = pd.melt(use_data, id_vars=['Bin', 'most_likely_states'], value_vars=use_vars)

    melted.loc[melted['variable']=='avg_wheel_vel', 'variable'] = 'Absolute wheel speed'
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


def plot_states_aligned(init, end, reduced_design_matrix, event_type_name, bin_size, inverted_mapping):

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
                        multiple="stack", binwidth=bin_size, binrange=(bin_size*init+0.01, bin_size*end), legend=True, ax = ax[0, 0], palette='viridis')
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
        
        if len(inverted_mapping) > 0:
            ordered_labels = [inverted_mapping[hue] for hue in sorted(use_data.loc[(use_data['correct']==1) 
                                                                                    & (use_data['choice']=='left'), 'most_likely_states'].unique())]
            # Get current handles and labels
            handles, _ = ax[0, 0].get_legend_handles_labels()
            print(handles)

            # Set custom labels
            ax[0, 0].legend(handles=handles, labels=ordered_labels, loc='upper left', bbox_to_anchor=(1, 1))
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
            
            
def plot_x_y_dynamics(x_var, y_var, mouse_dynamics, mouse_name, new_states, design_matrix_heading, inverted_mapping, grid_density, trajectory_num, plot_traj=True):
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    unique_states = np.array(list(inverted_mapping.keys()))
    unique_states = unique_states[~np.isnan(unique_states)]
    # mouse_dynamics = dynamics[mouse_name]
    
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