"""
Wheel functions
Jun 2023
Inês Laranjeira
"""

# %%
import pandas as pd
import numpy as np
from one.api import ONE
import matplotlib.pyplot as plt
import os
import brainbox.behavior.wheel as wh


# Get my functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Functions/'
# functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Functions/'
os.chdir(functions_path)
from one_functions_generic import prepro
from design_functions import wheel_velocity


# Define some constants
ENC_RES = 1024 * 4  # Rotary encoder resolution, assumes X4 encoding
WHEEL_DIAMETER = 3.1 * 2  # Wheel diameter in cm

# %%

# This function is redundant, use timeseries_PSTH from one_functions_generic
"""
def stack_trials(time, position, trials, event, t_init, t_end):

    wheel_df = pd.DataFrame({'time':time, 'position':position})
    
    time_step = np.median(np.diff(time))
    interval_length = int((t_end+t_init)/time_step + .25 * (t_end+t_init)/time_step) # This serves as an estimation for how large the data might be?
    
    onset_times = trials[event]
    # Start a matrix with #trials x # wheel bins
    wheel_stack = np.zeros((len(onset_times), interval_length)) * np.nan

    for t, trial_onset in enumerate(onset_times):
        if np.isnan(trial_onset) == False:
            if len(wheel_df.loc[wheel_df['time'] > trial_onset, 'time']) > 0:
                trial_onset_index = wheel_df.loc[wheel_df['time'] > trial_onset, 'time'].reset_index()['index'][0]
                onset_position = wheel_df['position'][trial_onset_index]

                # Populate dataframe with useful trial-aligned information
                window_values = wheel_df.loc[(wheel_df['time']> trial_onset-t_init) & (wheel_df['time'] <= trial_onset+t_end), 'position'] - onset_position 
                wheel_stack[t, :len(window_values)] = window_values
    return wheel_stack
"""


def wheel_interval(one, t_init, t_end, interval_length, sessions):
    # one = ONE()
    all_wheel = pd.DataFrame()
    # Loop through sessions (could put this inside an animal loop)
    for s, session in enumerate(sessions):
        # Get session data
        eid = session
        wheel = one.load_object(eid, 'wheel', collection='alf')
        wheelMoves = one.load_object(eid, 'wheelMoves', collection='alf')
        trials_data = data.loc[data['session']==eid][0:50]
        processed_data = prepro(trials_data)

        # Get wheel data for that session
        pos, time = wh.interpolate_position(wheel.timestamps, wheel.position)

        wheel_stack = pd.DataFrame(stack_trials(time=time, position=pos, trials=processed_data, event='stimOn_times', t_init=t_init, t_end=t_end, interval_length=interval_length))
        
        if s == 0:
            all_wheel = wheel_stack.copy()
            #all_wheel['mouse_name'] = 'KS014'  #TODO: make this not hard-coded
            #all_wheel['session_number'] = s + 1
            all_wheel['feedback'] = list(processed_data['correct'])
            all_wheel['choice'] = list(processed_data['choice'])
            all_wheel['contrast'] = list(processed_data['contrast'])
            all_wheel['side'] = list(np.sign(processed_data['signed_contrast']))
        else:
            #wheel_stack['mouse_name'] = 'KS014'  #TODO: make this not hard-coded
            #wheel_stack['session_number'] = s + 1
            wheel_stack['feedback'] = list(processed_data['correct'])
            wheel_stack['choice'] = list(processed_data['choice'])
            wheel_stack['contrast'] = list(processed_data['contrast'])
            wheel_stack['side'] = list(np.sign(processed_data['signed_contrast']))
            all_wheel = all_wheel.append(wheel_stack)
        
        return all_wheel


# From chatGPT
def find_periods_below_threshold(velocity, threshold, min_period):
    below_threshold = np.abs(velocity) < threshold
    start_idx = np.flatnonzero(np.diff(below_threshold, prepend=0, append=0) == 1)
    end_idx = np.flatnonzero(np.diff(below_threshold, prepend=0, append=0) == -1)
    durations = end_idx - start_idx
    periods = np.stack((start_idx, end_idx), axis=1)[durations >= min_period]
    return periods

# From chatGPT
def create_movement_array(velocity, periods_below_threshold):
    movement_array = np.ones_like(velocity)
    for start, end in periods_below_threshold:
        movement_array[start:end] = 0
    return movement_array


def stack_trial_events(one, session_trials, trials_to_plot, session_eid, time_max):

    # PLOT
    fig, axs = plt.subplots(nrows=trials_to_plot, ncols=1, sharex=True, sharey=False, figsize=[18, 14])
    plt.rc('font', size=12)
    y = [-10, 10]
    bin_size = 0.05
    time_min = -0.1
    threshold = 0.25 # Need to check if this makes sense
    min_period = 200 # This is approximately 200 ms

    use_data = session_trials.reset_index()
    
    # Account for naming differences in how data was loaded
    if 'quiescencePeriod' in use_data.keys():
        quiescence_name = 'quiescencePeriod'
    elif 'quiescence' in use_data.keys():
        quiescence_name = 'quiescence'
    
    for t, trial in enumerate(range(trials_to_plot)):

        trial_start = use_data.loc[use_data['index']==t, 'intervals_0']
        next_trial = use_data.loc[use_data['index']==t+1]
        trial_data = use_data.loc[use_data['index']==t]
        trial_feedback = trial_data['feedbackType']

        axs[t].vlines(np.array(trial_data['stimOn_times']) - trial_start,
                        -10, 10, label='Stim On', color='Black')
        if list(trial_feedback)[0] == 1:
            axs[t].vlines(np.array(trial_data['feedback_times']) - trial_start,
                        -10, 10, label='Correct', color='Green', linewidth=2)
            axs[t].fill_betweenx(y, list(trial_data['firstMovement_times'])[0]-list(trial_start)[0], 
                          list(trial_data['feedback_times'])[0]-list(trial_start)[0], color='green', alpha=0.3)
            axs[t].fill_betweenx(y, list(trial_data['feedback_times'])[0]-list(trial_start)[0], 
                          list(trial_data['feedback_times'])[0]-list(trial_start)[0], color='green', alpha=0.6)
        else:
            axs[t].vlines(np.array(trial_data['feedback_times']) - trial_start,
                        -10, 10, label='Incorrect', color='Red', linewidth=2)
            axs[t].fill_betweenx(y, list(trial_data['firstMovement_times'])[0]-list(trial_start)[0], 
                          list(trial_data['feedback_times'])[0]-list(trial_start)[0], color='red', alpha=0.3)
            axs[t].fill_betweenx(y, list(trial_data['feedback_times'])[0]-list(trial_start)[0], 
                          list(trial_data['feedback_times'])[0]-list(trial_start)[0], color='red', alpha=0.6)
            
        axs[t].vlines(np.array(trial_data['firstMovement_times']) - trial_start, -10, 10, 
                        label='First movement', color='Blue')
        axs[t].vlines(np.array((trial_data['goCue_times'] - trial_data[quiescence_name])) - trial_start,
                        -10, 10, label='Quiescence start', color='Purple')
        axs[t].vlines(np.array(trial_data['feedback_times']) - trial_start,
                        -10, 10, label='Stim Off', color='Brown')
        axs[t].vlines(np.array(trial_data['intervals_1']) - trial_start,
                        -10, 10, label='Trial end', color='Orange')
        axs[t].vlines(np.array(next_trial['intervals_0']) - trial_start,
                        -10, 10, label='Next trial start', color='Grey')  
        axs[t].vlines(np.array((next_trial['goCue_times'] - next_trial[quiescence_name])) - trial_start,
                        -10, 10, label='Quiescence start', color='Purple')

        axs[t].fill_betweenx(y, 0, list(trial_data['goCue_times'] - 
                          trial_data[quiescence_name])[0]-list(trial_start)[0], color='purple', alpha=0.6)
        axs[t].fill_betweenx(y, list(trial_data['goCue_times'] - 
                          trial_data[quiescence_name])[0]-list(trial_start)[0], 
                          list(trial_data['stimOn_times'])[0]-list(trial_start)[0], color='purple', alpha=0.3)
        axs[t].fill_betweenx(y, list(trial_data['stimOn_times'])[0]-list(trial_start)[0],
                             list(trial_data['firstMovement_times'])[0]-list(trial_start)[0], color='blue', alpha=0.3)
        axs[t].fill_betweenx(y, list(trial_data['feedback_times'])[0]-list(trial_start)[0],
                             list(trial_data['intervals_1'])[0]-list(trial_start)[0], color='orange', alpha=0.3)
        axs[t].fill_betweenx(y, list(next_trial['intervals_0'])[0] - list(trial_start)[0], 
                             list(next_trial['goCue_times'] - 
                          next_trial[quiescence_name])[0]-list(trial_start)[0], color='purple', alpha=0.6)
        
        # Wheel
        wheel_data = one.load_object(session_eid, 'wheel', collection='alf')
        pos, wheel_times = wh.interpolate_position(wheel_data.timestamps, wheel_data.position)
        # Calculate wheel velocity
        wheel_vel = wheel_velocity(bin_size, wheel_times, pos, use_data)
        wheel_trace = np.array(wheel_vel['avg_wheel_vel'])

        xx = wheel_times - list(trial_start)[0]
        yy = wheel_trace

        trial_time_max = list(np.array(next_trial['goCue_times'] - 
                                        next_trial[quiescence_name]) - trial_start)[0]
        mask = np.where((xx <trial_time_max) & (xx> time_min))
        wheel_max = np.max(wheel_trace[mask])
        wheel_min = np.min(wheel_trace[mask])

        # Plot wheel
        axs[t].plot(xx[mask], yy[mask], color='Black')

        # Compute stillness
        periods_below_threshold = find_periods_below_threshold(wheel_trace, threshold, min_period)
        stillness_array = create_movement_array(wheel_trace, periods_below_threshold)
        movement_array = stillness_array.copy()
        movement_array[movement_array >0] = np.nan 
        stillness_array[stillness_array==0] = np.nan 
        stillness_array[stillness_array==1] = 0

        # Plot movement and stilness
        axs[t].plot(xx[mask], stillness_array[mask], color='Orange')
        axs[t].plot(xx[mask], movement_array[mask], color='Blue')
        
        axs[t].set_ylim([wheel_min, wheel_max])
        axs[t].set_yticks([] ,[])
    axs[t].set_xlabel(str('Time from trial start (s)'))
    axs[t].set_xlim([time_min, time_max])
    axs[t].legend(loc='upper left', bbox_to_anchor=(1, -0.5))
    plt.show()


def wheel_trial_epoch(one, session_trials, session_eid, bin_size, threshold, min_period):
    
    #TODO need to accout for this commented out line in code outside function
    # use_data = prepro(session_trials.reset_index())

    use_data = session_trials
    # Account for differences in how data was loaded
    if 'quiescencePeriod' in use_data.keys():
        quiescence_name = 'quiescencePeriod'
    elif 'quiescence' in use_data.keys():
        quiescence_name = 'quiescence'

    df = pd.DataFrame(columns=['trial', 'time', 'wheel', 'movement', 'trial_epoch', 
                               'feedback', 'next_feedback', 'signed_contrast', 
                               'response', 'reaction', 'choice', 'probabilityLeft'])

    # Wheel
    wheel_data = one.load_object(session_eid, 'wheel', collection='alf')
    pos, wheel_times = wh.interpolate_position(wheel_data.timestamps, wheel_data.position)
    # Calculate wheel velocity
    wheel_vel = wheel_velocity(bin_size, wheel_times, pos, use_data)
    wheel_trace = np.array(wheel_vel['avg_wheel_vel'])
    
    # Compute stillness
    periods_below_threshold = find_periods_below_threshold(wheel_trace, threshold, min_period)
    movement_array = create_movement_array(wheel_trace, periods_below_threshold)

    # Save data on dataframe
    df['time'] = wheel_times
    df['wheel'] = wheel_trace
    df['movement'] = movement_array

    for t, trial in enumerate(range(len(use_data)-1)):

        
        trial_data = use_data.loc[use_data['index']==t]
        next_trial = use_data.loc[use_data['index']==t+1]
            
        # Compute timings
        trial_start = list(trial_data['intervals_0'])[0]
        quiescence_start = list(trial_data['goCue_times'] - trial_data[quiescence_name])[0]
        stim_on = list(trial_data['stimOn_times'])[0]
        first_movement = list(trial_data['firstMovement_times'])[0]
        response_time = list(trial_data['response_times'])[0]
        # stim_off = list(trial_data['feedback_times'])[0]
        # trial_end = list(trial_data['intervals_1'])[0]
        next_trial_start = list(next_trial['intervals_0'])[0]

        # Compute intervals
        df.loc[(df['time'] >= trial_start) & (df['time'] < quiescence_start), 'trial_epoch'] = 'trial_start'
        df.loc[(df['time'] >= quiescence_start) & (df['time'] < stim_on), 'trial_epoch'] = 'quiescence'
        df.loc[(df['time'] >= stim_on) & (df['time'] < first_movement), 'trial_epoch'] = 'stim_on'
        df.loc[(df['time'] >= first_movement) & (df['time'] < response_time), 'trial_epoch'] = 'movement'
        df.loc[(df['time'] >= response_time) & (df['time'] < next_trial_start), 'trial_epoch'] = 'post_choice'

        
        df.loc[(df['time'] >= trial_start) & (df['time'] < next_trial_start), 'feedback'] = list(trial_data['feedbackType'])[0]
        df.loc[(df['time'] >= trial_start) & (df['time'] < next_trial_start), 'next_feedback'] = list(next_trial['feedbackType'])[0]
        df.loc[(df['time'] >= trial_start) & (df['time'] < next_trial_start), 'signed_contrast'] =  list(trial_data['signed_contrast'])[0]
        df.loc[(df['time'] >= trial_start) & (df['time'] < next_trial_start), 'trial'] =  trial
        df.loc[(df['time'] >= trial_start) & (df['time'] < next_trial_start), 'response'] =  list(trial_data['response'])[0]
        df.loc[(df['time'] >= trial_start) & (df['time'] < next_trial_start), 'reaction'] =  list(trial_data['reaction'])[0]
        df.loc[(df['time'] >= trial_start) & (df['time'] < next_trial_start), 'choice'] =  list(trial_data['choice'])[0]
        df.loc[(df['time'] >= trial_start) & (df['time'] < next_trial_start), 'probabilityLeft'] =  list(trial_data['probabilityLeft'])[0]

    
    return df


