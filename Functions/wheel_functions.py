"""
Wheel functions
Jun 2023
Inês Laranjeira
"""

# %%
import pandas as pd
import numpy as np


# %%

def stack_trials(time, position, trials, event, t_init, t_end, interval_length):

    wheel_df = pd.DataFrame({'time':time, 'position':position})
    onset_times = trials[event]
    # Start a matrix with #trials x # wheel bins
    wheel_stack = np.zeros((len(onset_times), interval_length)) * np.nan

    for t, trial_onset in enumerate(onset_times):
        if np.isnan(trial_onset) == False:
            if len(wheel_df.loc[wheel_df['time'] > trial_onset, 'time']) > 0:
                trial_onset_index = wheel_df.loc[wheel_df['time'] > trial_onset, 'time'].reset_index()['index'][0]
                onset_position = wheel_df['position'][trial_onset_index]
                #trial_feedback = list(preprocessed_data['correct'])[t]

                # Populate dataframe with useful trial-aligned information
                wheel_stack[t, :] = wheel_df.loc[(wheel_df['time']> trial_onset-t_init) & (wheel_df['time'] <= trial_onset+t_end), 'position'] - onset_position 
    return wheel_stack
# %%
