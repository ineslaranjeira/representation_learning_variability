""" 
IMPORTS
"""
import autograd.numpy as np
import pandas as pd


""" DESIGN MATRIX FUNCTIONS """

def lick_rate(bin_size, lick_times, trials):
    
    session_length = list(trials['stimOff_times'][-1:])[0]

    # Create a sample DataFrame with values
    data = {'Lick times': lick_times}
    df = pd.DataFrame(data)

    # Define the number of bins and create the bins
    # Define the time bins
    bins = np.arange(0, np.floor(session_length), bin_size)
    bins = pd.cut(df['Lick times'], bins=bins, labels=False)

    # Create a new column with the bin number
    df['Bin'] = bins

    # Count events within each time bin
    count_values = df.groupby('Bin')['Lick times'].count()
    count_values = count_values.rename('Lick count')
    merged_df = df.merge(count_values, left_index=True, right_index=True, how='left')
    
    return merged_df


def wheel_velocity(bin_size, wheel_times, wheel_pos, trials):
    
    session_length = list(trials['stimOff_times'][-1:])[0]

    # Create a sample DataFrame with values
    data = {'Wheel times': wheel_times,
            'Wheel position': wheel_pos}

    df = pd.DataFrame(data)

    # Define the number of bins and create the bins
    bins = np.arange(0, np.floor(session_length), bin_size)
    bins = pd.cut(df['Wheel times'], bins=bins, labels=False)

    # Create a new column with the bin number
    df['Bin'] = bins
    
    # Count events within each time bin
    change_values = df.groupby('Bin')['Wheel position'].agg(lambda x: x.iloc[-1] - x.iloc[0]) # This should be weighed by amount of time on each location?
    change_values = change_values.rename('avg_wheel_vel')
    vel_values = change_values/bin_size
    
    merged_df = df.merge(vel_values, on='Bin', how='left')

    return merged_df


def wheel_displacement(wheel_times, wheel_positions, trials, bin_size, onset_subtraction=True):

    # Create a sample DataFrame with values
    data = {'Wheel times': wheel_times,
            'Wheel position': wheel_positions}
    df = pd.DataFrame(data)


    session_length = list(trials['stimOff_times'][-1:])[0]
    #session_length = len(trials['stimOff_times'])

    # Trim wheel data after end of session
    df = df.loc[df['Wheel times'] < session_length]
    
    # Define the number of bins and create the bins
    bins = np.arange(0, np.floor(session_length), bin_size)
    bins = pd.cut(df['Wheel times'], bins=bins, labels=False)

    # Create a new column with the bin number
    df['Bin'] = bins

    # Define the bin edges array
    trial_edges = list(trials['stimOn_times'])
    df['Trial'] = pd.cut(df['Wheel times'], bins=trial_edges, labels=False)

    onsets = pd.DataFrame({'Onset times': trial_edges,
                          'Trial': np.arange(0, len(trial_edges), 1)})
    # Subtract onset value
    if onset_subtraction == True:

        # Baseline is exact value at stimulus onset
        onsets['baseline'] = onsets['Onset times'].agg([lambda x: np.nanmean(wheel_positions[np.where(wheel_times >= x)][0])])
                
        # Merge dataframes
        df = df.merge(onsets, on='Trial')
        df['wheel_final'] = df['Wheel position'] - df['baseline']

        wheel_displacement = df.groupby(['Bin', 'Onset times'])['wheel_final'].mean()
        wheel_displacement = wheel_displacement.reset_index(level=[0, 1])
        
    else:
            
        # Merge dataframes
        df = df.merge(onsets, on='Trial')
        df['wheel_final'] = df['Wheel position'] 

        wheel_displacement = df.groupby(['Bin', 'Onset times'])['wheel_final'].mean()
        wheel_displacement = wheel_displacement.reset_index(level=[0, 1])
        
    return wheel_displacement


def pupil_diam(pupil_times, pupil_dia_smooth, trials, bin_size, onset_subtraction=True):

    # Bins pupil diameter and subtracts stimulus onset_value
    
    # Create a sample DataFrame with values
    data = {'Pupil times': pupil_times,
            'Pupil diam': pupil_dia_smooth}
    
    df = pd.DataFrame(data)

    session_length = list(trials['stimOff_times'][-1:])[0]

    # Define the number of bins and create the bins
    bins = np.arange(0, np.floor(session_length), bin_size)
    bins = pd.cut(df['Pupil times'], bins=bins, labels=False)

    # Create a new column with the bin number
    df['Bin'] = bins

    # Define the bin edges array
    trial_edges = list(trials['stimOn_times'])
    df['Trial'] = pd.cut(df['Pupil times'], bins=trial_edges, labels=False)
    
    # Subtract onset value
    if onset_subtraction == True:
        onsets = pd.DataFrame({'Onset times': trial_edges,
                'Trial': np.arange(0, len(trial_edges), 1)})
        # Baseline is 500 ms before stimulus onset
        onsets['baseline'] = onsets['Onset times'].agg([lambda x: 
                                                        np.nanmean(pupil_dia_smooth[np.where((pupil_times <= x) &
                                                                                        (pupil_times > x - 0.5))])])
        # Merge dataframes
        df = df.merge(onsets, on='Trial')

        df['pupil_final'] = df['Pupil diam'] - df['baseline']

        pupil_df = df.groupby('Bin')['pupil_final'].mean()
        pupil_df = pupil_df.reset_index(level=[0])

    else:
        df['pupil_final'] = df['Pupil diam'] 

        pupil_df = df.groupby('Bin')['pupil_final'].mean()
        pupil_df = pupil_df.reset_index(level=[0])


    return pupil_df


def cont_bin(times, metric, trials, bin_size):

        # Bins continuous data types
    
        # Create a sample DataFrame with values
        data = {'Times': times,
                'Values': metric}

        df = pd.DataFrame(data)
        session_length = list(trials['stimOff_times'][-1:])[0]

        # Define the number of bins and create the bins
        bins = np.arange(0, np.floor(session_length), bin_size)
        bins = pd.cut(df['Times'], bins=bins, labels=False)

        # Create a new column with the bin number
        df['Bin'] = bins

        # Define the bin edges array
        trial_edges = list(trials['stimOn_times'])
        df['Trial'] = pd.cut(df['Times'], bins=trial_edges, labels=False)

        onsets = pd.DataFrame({'Onset times': trial_edges,
                'Trial': np.arange(0, len(trial_edges), 1)})

        # Merge dataframes
        df = df.merge(onsets, on='Trial')

        df_binned = df.groupby(['Bin', 'Trial'])['Values'].mean()
        df_binned = df_binned.reset_index(level=[0, 1])

        return df_binned


def align_stimOn(df, trials):

    # Define the bin edges array
    trial_edges = list(trials['stimOn_times'])

    onsets = pd.DataFrame({'onset_times': trial_edges,
                          'Trial': np.arange(0, len(trial_edges), 1)})
    
    # Merge dataframes
    df = df.merge(onsets, on='Trial')
        
    return df