"""
Wheel functions
Jun 2023
Inês Laranjeira
"""

# %%
import pandas as pd
import numpy as np
from one.api import ONE



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


def wheel_interval(t_init, t_end, interval_length, sessions):
    one = ONE()
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
