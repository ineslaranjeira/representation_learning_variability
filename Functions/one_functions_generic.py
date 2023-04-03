"""
Generic functions to get learning data using ONE
Jan 2023
Inês Laranjeira
"""

# %%
import pandas as pd
import numpy as np
import datetime
import pickle 
import os

from brainbox.task.trials import find_trial_ids
from brainbox.behavior.training import get_sessions, get_training_status

from one.api import ONE
# one = ONE(base_url='https://openalyx.internationalbrainlab.org')  # public database
one = ONE(base_url='https://alyx.internationalbrainlab.org')

# %%
"""
GET TRIALS INFO INTO ONE TABLE
"""

def get_trials(training_protocol='training', mouse_project='ibl_neuropixel_brainwide_01'):

    """ Download session data """
    # Search sessions of interest
    sessions = one.search(task_protocol=training_protocol, project=mouse_project, details=True)
    session_eids = sessions[0]
    session_details = sessions[1]

    # Initialize dataframe to collect all data
    all_trials = pd.DataFrame()
    # Loop through sessions
    for s, sess in enumerate(session_eids):

        # Try to donwload data or print a warning
        try:
            # Download trial data
            trials = one.load_object(sess, obj='trials', namespace='ibl')
            trials_df = trials.to_df()

            # Compute trial id information
            trials_df['trial_id'] = find_trial_ids(trials)[0]
            # Save session details
            trials_df['subject_nickname'] = session_details[s]['subject']
            trials_df['session_date'] = session_details[s]['date']
            trials_df['session_number'] = session_details[s]['number']
            trials_df['task_protocol'] = session_details[s]['task_protocol']
            trials_df['session_uuid'] = sess

            # Save session data to big dataframe
            if s == 0:
                all_trials = trials_df.copy()
            else:
                all_trials = all_trials.append(trials_df)

        # Throw warning in case data can't be downloaded
        except:
            print(str('Problems with session:' + str(sess)))

    """ Compute training day """
    # Initialize dataframe for training days of all subejcts
    all_subjects = pd.DataFrame()

    # Loop through subjects data
    subjects = all_trials['subject_nickname'].unique()
    for sub, subject in enumerate(subjects):

        # Get subject data
        subject_data = all_trials.loc[all_trials['subject_nickname'] == subject]
        # Group by date
        subject_sessions = pd.DataFrame(subject_data.groupby(['subject_nickname',
                                                              'session_uuid'])
                                        ['session_date'].max())
        subject_sessions = subject_sessions.reset_index(level=[0, 1])
        # Get rid of repeated dates (sessions of the same day have the same training day)
        subject_sessions = subject_sessions[['subject_nickname', 'session_date']].drop_duplicates()
        # TODO: this is not very good code; shoulds find a better way
        # Sort dates and write corresponding training days
        subject_sessions = subject_sessions.sort_values(by='session_date')
        subject_sessions['training_day'] = np.arange(1, len(subject_sessions) + 1)

        # Save data in big dataframe
        if sub == 0:
            all_subjects = subject_sessions.copy()
        else:
            all_subjects = all_subjects.append(subject_sessions)

    # Merge training days dataframe with trials dataframe
    df = all_trials.merge(all_subjects, on=['subject_nickname', 'session_date'], how='outer')

    return df


def get_first_session(trials):
    subjects = trials['subject_nickname'].unique()
    first_session = pd.DataFrame(columns=['subject_nickname', 'session_date'],
                                 index=range(len(subjects)))
    for s, sess in enumerate(subjects):

        mouse_data = trials.loc[trials['subject_nickname'] == sess]
        first_session['subject_nickname'][s] = sess
        first_session['session_date'][s] = np.min(mouse_data.loc[mouse_data['training_day'] <= 1,
                                                                 'session_date'])

    return first_session


"""
CALCULATE SOME USEFUL GENERIC METRICS
"""

def performance_metrics(trials, session):

    """
    Build design matrix with performance metrics

    Parameters
    trials:             All training trials for the mice of interest

    """

    mice = trials['subject_nickname'].unique()
    d_matrix = pd.DataFrame(columns=['subject_nickname', 'perf_init', 'RT_init', 'trials_init',
                                     'delta_variance', 'trials_sum', 'perf_delta1', 'RT_delta1',
                                     'trials_delta1'], index=range(len(mice)))

    # --Pre-processing
    trials['contrastLeft'][trials['contrastLeft'].isnull()] = 0
    trials['contrastRight'][trials['contrastRight'].isnull()] = 0

    trials['contrast'] = trials['contrastLeft'] + trials['contrastRight']
    trials['RT'] = trials['response_times'] - trials['stimOn_times']
    trials['correct'] = (trials['feedbackType'] + 1) / 2
    # trials['performance_easy'] = trials['correct']

    for m, mouse in enumerate(mice):

        mouse_data = trials.loc[trials.subject_nickname == mouse]

        sess_perf = pd.DataFrame(mouse_data.loc[mouse_data['contrast'] >=
                                                0.5].groupby(['subject_nickname', 
                                                              'training_day'])['correct'].mean())
        sess_perf = sess_perf.reset_index(level=[0, 1])
        sess_perf = sess_perf.rename(columns={'correct': 'performance_easy'})
        mouse_data = mouse_data.merge(sess_perf, on=['subject_nickname', 'training_day'])
        
        #mouse_data.loc[mouse_data['subject_nickname'] == mouse, 'performance_easy'] = perf_data['performance_easy']

        #trials.loc[trials['subject_nickname'] == mouse, 'performance_easy'] = mouse_data['performance_easy']
        # Get first training session for this mouse (is it 1 or zero)
        first_session = np.min(mouse_data['training_day'])

        if first_session > 1:
            print('Mouse', mouse, 'missing first session')
        else:

            d_matrix['subject_nickname'][m] = mouse

            # Task performance on the first session
            perf_init = mouse_data.loc[mouse_data['training_day'] == first_session,
                                       'performance_easy']
            if len(perf_init) > 0:
                d_matrix['perf_init'][m] = np.nanmean(perf_init)

            RT_init = list(mouse_data.loc[(mouse_data['training_day'] == first_session) &
                                          (mouse_data['contrast'] >= 0.5), 'RT'])
            if len(RT_init) > 0:
                d_matrix['RT_init'][m] = np.nanmedian(RT_init)

            trials_init = len(mouse_data.loc[mouse_data['training_day'] == first_session])
            if trials_init > 0:
                d_matrix['trials_init'][m] = trials_init

            # Change in task performance across the first sessions
            perf_last = mouse_data.loc[mouse_data['training_day'] == (session + first_session - 1),
                                       'performance_easy']
            if len(perf_last) == 0 or len(perf_init) == 0:
                d_matrix['perf_delta1'][m] = np.nan
            else:
                d_matrix['perf_delta1'][m] = np.nanmean(perf_last) - np.nanmean(perf_init)

            RT_last = list(mouse_data.loc[(mouse_data['training_day'] ==
                                          (session + first_session - 1)) &
                                          (mouse_data['contrast'] >= 0.5), 'RT'])

            if len(RT_last) == 0 or len(RT_init) == 0:
                d_matrix['RT_delta1'][m] = np.nan
            else:
                d_matrix['RT_delta1'][m] = np.nanmedian(RT_last) - np.nanmedian(RT_init)

            trials_last = len(mouse_data.loc[mouse_data['training_day'] ==
                                             (session + first_session - 1)])

            if trials_last == 0 or trials_init == 0:
                d_matrix['trials_delta1'][m] = np.nan
            else:
                d_matrix['trials_delta1'][m] = trials_last - trials_init

            if first_session == 0:
                d_matrix['trials_sum'][m] = len(mouse_data.loc[mouse_data['training_day'] == 0]) + \
                    len(mouse_data.loc[mouse_data['training_day'] == 1]) + \
                    len(mouse_data.loc[mouse_data['training_day'] == 2]) + \
                    len(mouse_data.loc[mouse_data['training_day'] == 3]) + \
                    len(mouse_data.loc[mouse_data['training_day'] == 4])
            elif first_session == 1:
                d_matrix['trials_sum'][m] = len(mouse_data.loc[mouse_data['training_day'] == 1]) + \
                    len(mouse_data.loc[mouse_data['training_day'] == 2]) + \
                    len(mouse_data.loc[mouse_data['training_day'] == 3]) + \
                    len(mouse_data.loc[mouse_data['training_day'] == 4]) + \
                    len(mouse_data.loc[mouse_data['training_day'] == 5])

            restricted_mouse_data = mouse_data.loc[(mouse_data.subject_nickname == mouse) &
                                           (mouse_data.training_day <= session + first_session - 1)]

            mouse_perf = pd.DataFrame(restricted_mouse_data.groupby(['subject_nickname', 'training_day'])
                                  ['performance_easy'].mean())
            mouse_perf = mouse_perf.reset_index(level=[0, 1])
            delta = np.sign(np.diff(mouse_perf['performance_easy']))
            d_matrix['delta_variance'][m] = np.sum(delta) / len(delta)

    return d_matrix


def training_time(trials):

    # Create new dataframe with training status and training time columns
    training_time_df = pd.DataFrame(trials.groupby(['subject_nickname', 'session_date',
                                                    'session_uuid'])['training_day'].mean())
    training_time_df = training_time_df.reset_index(level=[0, 1, 2])
    training_time_df['training_status'] = np.zeros(len(training_time_df)) * np.nan
    training_time_df['training_time'] = np.zeros(len(training_time_df)) * np.nan

    """ Get training status for each session """
    # Loop through each subject
    subjects = training_time_df['subject_nickname'].unique()
    for nickname in subjects:

        # Get all sessions for that mouse after the third
        mouse_sessions = training_time_df.loc[(training_time_df['subject_nickname'] == nickname) &
                                              (training_time_df['training_day'] >= 3),
                                              'session_uuid'].unique()
        # Loop through sessions of the mouse starting after the third
        for session_uuid in mouse_sessions:

            try:
                # Get details to download data
                date = training_time_df.loc[(training_time_df['subject_nickname'] == nickname) &
                                            (training_time_df['session_uuid'] == session_uuid),
                                            'session_date']

                # Get three last sessions and compute training status
                sessions_three, task_protocol, ephys_sess, n_delay = get_sessions(nickname,
                                                                                  date=str(list(
                                                                                           date)[0]
                                                                                           ),
                                                                                  one=one)
                training_status = get_training_status(sessions_three, task_protocol, [], n_delay)

                # Add training status to dataframe
                training_time_df.loc[(training_time_df['subject_nickname'] == nickname) &
                                     (training_time_df['session_date'] == list(date)[0]),
                                     'training_status'] = training_status[0]
            except:
                print(str('Problems with session:' + str(session_uuid)))

        """ Calculate training time for that mouse """
        # Note: 'untrainable' is not a training_status anymore
        training_days = np.max(training_time_df.loc[(training_time_df['subject_nickname'] ==
                                                     nickname) &
                                                    (training_time_df['training_status'] ==
                                                    'in training'), 'training_day'])
        # Add training time to dataframe
        training_time_df.loc[training_time_df['subject_nickname'] == nickname,
                             'training_time'] = training_days

    return training_time_df


def quartile(trials, criterion='training_time'):

    # crit can be 'training_time' or 'learning_onset'
    trials_grouped = pd.DataFrame(trials.groupby(['subject_nickname'])[criterion].mean())
    trials_grouped = trials_grouped.reset_index(level=[0])
    trials_grouped[criterion].unique()

    quantile_df = trials_grouped.copy()
    quantile_df = quantile_df.dropna()
    quantile_df = quantile_df.drop_duplicates()
    quantile_df['quantile'] = quantile_df[criterion]
    crit = quantile_df[criterion].dropna()
    quantiles = crit.quantile([.25, .5, .75])
    quantile_df.loc[quantile_df[criterion] <= quantiles[0.25], 'quantile'] = 1
    quantile_df.loc[(quantile_df[criterion] > quantiles[0.25]) &
                    (quantile_df[criterion] <= quantiles[0.5]), 'quantile'] = 2
    quantile_df.loc[(quantile_df[criterion] > quantiles[0.5]) &
                    (quantile_df[criterion] <= quantiles[0.75]), 'quantile'] = 3
    quantile_df.loc[quantile_df[criterion] > quantiles[0.75], 'quantile'] = 4

    return quantile_df


def prepro(trials):

    """ Performance """
    # Some preprocessing
    trials['contrastLeft'] = trials['contrastLeft'].fillna(0)
    trials['contrastRight'] = trials['contrastRight'].fillna(0)
    trials['signed_contrast'] = - trials['contrastLeft'] + trials['contrastRight']
    trials['contrast'] = trials['contrastLeft'] + trials['contrastRight']
    trials['correct_easy'] = trials['feedbackType']
    trials.loc[trials['correct_easy']==-1, 'correct_easy'] = 0
    trials['correct'] = trials['feedbackType']
    trials.loc[trials['contrast']<.5, 'correct_easy'] = np.nan
    trials.loc[trials['correct']==-1, 'correct'] = 0

    """ Response/ reaction times """
    trials['response'] = trials['response_times'] - trials['stimOn_times']
    trials['reaction'] = trials['firstMovement_times'] - trials['stimOn_times']
    #TODO : trials['days_to_trained'] = trials['training_time']
    
    return trials


"""
DIFFERENT WAYS OF BINNING DATA
"""

def bin_frac(trials, bin_num):

    subjects = trials.subject_nickname.unique()

    # Create new empty dataframe
    new_df = pd.DataFrame()

    # Loop through subjects
    for s, subject in enumerate(subjects):
        # Get subject data
        subject_data = trials.loc[trials['subject_nickname']==subject]
        mouse_training_day = int(subject_data['training_time'].unique()[0]) + 1
        subject_data = subject_data.loc[subject_data['training_day']<mouse_training_day]
        subject_data = subject_data.sort_values(by=['training_day', 'trial_id'])
        # Caltulate bin specifics
        total_trials = len(subject_data)
        bin_size = int(np.round(total_trials / bin_num))
        bin_index = np.array([])
        # Design bin number array
        for n in range(bin_num):
            this_bin_index = np.ones(bin_size) * (n+1)
            bin_index = np.concatenate((bin_index, this_bin_index), axis=None)
        # Add buffer to the end in case array is shorter than dataframe
        bin_index = np.concatenate((bin_index, np.ones(bin_size) * 15), axis=None)
        subject_data['bin_frac'] = bin_index[0:len(subject_data)]
        # Append subject data to big dataframe
        new_df = new_df.append(subject_data)
        
    return new_df


# TODO

def wheel(trials):

    return wheel_df


def learning_onset(trials):

    return learning_onset

