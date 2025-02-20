""" 
IMPORTS
"""
import os
import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Custom functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Functions/'
# functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability/Functions/'
os.chdir(functions_path)
from one_functions_generic import prepro



def plot_kde(X_embedded, kernel):
    xmin = -150
    xmax = 150
    ymin=-150
    ymax=150
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
            extent=[xmin, xmax, ymin, ymax])
    ax.plot(X_embedded[:, 0], X_embedded[:, 1], 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.show()
    

def plot_mean_cluster(cluster_features, plt_vars):
    plt.rcParams.update({'font.size': 16})

    # Get viridis color palette
    unique_clusters = cluster_features['most_likely_states'].unique()
    colors = sns.color_palette("viridis", len(unique_clusters)).as_hex()  # Get hex colors

    # Create the figure
    fig = go.Figure()

    for i, cluster in enumerate(unique_clusters):
        fig.add_trace(go.Scatterpolar(
            r=np.array(cluster_features.loc[cluster_features['most_likely_states'] == cluster, plt_vars])[0],
            theta=plt_vars,
            fill='toself',
            name=f'Wavelet transform cluster {cluster}',
            line=dict(color=colors[i])  # Assign Viridis color
        ))

    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                tickfont=dict(size=16)  # Set the font size of the theta labels
            ),
            radialaxis=dict(
                visible=True,
                tickfont=dict(size=16)
            )
        ),
        showlegend=True
    )

    fig.show()

def timeseries_PSTH(time, positions, trials, event, t_init, t_end, subtract_baseline):
    """
    Compute peri-stimulus time histograms (PSTH) for multiple position signals.

    Parameters:
    - time: Array of time values.
    - positions: Dictionary of position arrays (keys are names of signals).
    - trials: DataFrame containing trial information.
    - event: Event name for alignment.
    - t_init: Time before event onset.
    - t_end: Time after event onset.
    - subtract_baseline: Whether to subtract baseline.

    Returns:
    - df_melted: Long-format DataFrame with aligned position data.
    """
    series_df = pd.DataFrame({'time': time})
    for key, pos in positions.items():
        series_df[key] = pos  # Store multiple position signals
    
    onset_times = trials[event].dropna().values  # Drop NaNs early
    
    time_step = np.median(np.diff(time))
    interval_length = int((t_end + t_init) / time_step * 1.25)  # Preallocate buffer
    
    series_stack = {key: np.full((len(onset_times), interval_length), np.nan) for key in positions}
    
    time_arr = series_df['time'].values
    pos_arrays = {key: series_df[key].values for key in positions}

    for t in range(len(onset_times)):
        trial_onset = onset_times[t]
        next_onset = onset_times[t+1] if t < len(onset_times) - 1 else trial_onset + t_end + 1

        trial_onset_idx = np.searchsorted(time_arr, trial_onset, side='right')
        next_onset_idx = np.searchsorted(time_arr, next_onset, side='right')

        mask = (time_arr > trial_onset - t_init) & (time_arr <= trial_onset + t_end)
        time_window = time_arr[mask] - trial_onset  

        for key, pos_arr in pos_arrays.items():
            window_values = pos_arr[mask]

            if subtract_baseline:
                window_values -= pos_arr[trial_onset_idx]

            window_values[time_window > (time_arr[next_onset_idx] - trial_onset)] = np.nan
            series_stack[key][t, :len(window_values)] = window_values

    preprocessed_trials = prepro(trials)
    preprocessed_trials = preprocessed_trials.dropna(subset=[event])
    df_stack = pd.DataFrame()

    for key, data in series_stack.items():
        df = pd.DataFrame(data)
        df['signal'] = key
        df_stack = pd.concat([df_stack, df], axis=0, ignore_index=True)

    trial_vars = ['feedbackType', 'choice', 'contrast']
    for var in trial_vars:
        df_stack[var] = np.tile(preprocessed_trials[var].values, len(positions))

    df_melted = df_stack.melt(id_vars=['feedbackType', 'choice', 'contrast', 'signal'], 
                              var_name='time', value_name='position')

    df_melted['time'] = df_melted['time'].astype(int)
    df_melted['time'] = df_melted['time'].map(lambda x: time_window[x] if x < len(time_window) else np.nan)

    return df_melted


