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
import uuid
from pathlib import Path

from brainbox.task.trials import find_trial_ids
from brainbox.behavior.training import get_sessions, get_training_status

from one.api import ONE
from one.alf.files import add_uuid_string
from one.remote import aws
# one = ONE(base_url='https://openalyx.internationalbrainlab.org')  # public database
one = ONE(base_url='https://alyx.internationalbrainlab.org')

# %%
"""
GET TRIALS INFO INTO ONE TABLE
"""

# Function written by Julia 
def download_subjectTables(one, subject=None, trials=True, training=True,
                           target_path=None, tag=None, overwrite=False, check_updates=True):
    """
    Function to download the aggregated clusters information associated with the given data release tag from AWS.
    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to database.
    trials: bool
        Whether to download the subjectTrials.table.pqt, default is True
    training: bool
        Whether to donwnload the subjectTraining.table.pqt, defaults is True
    subject: str, uuid or None
        Nickname or UUID of the subject to download all trials from. If None, download all available trials tables
        (associated with 'tag' if one is given)
    target_path: str or pathlib.Path
        Directory to which files should be downloaded. If None, downloads to one.cache_dir/aggregates
    tag: str
        Data release tag to download _ibl_subjectTrials.table datasets from. Default is None.
    overwrite : bool
        If True, will re-download files even if file exists locally and file sizes match.
    check_updates : bool
        If True, will check if file sizes match and skip download if they do. If False, will just return the paths
        and not check if the data was updated on AWS.
    Returns
    -------
    trials_tables: list of pathlib.Path
        Paths to the downloaded subjectTrials files
    training_tables: list of pathlib.Path
        Paths to the downloaded subjectTraining files
    """

    if target_path is None:
        target_path = Path(one.cache_dir).joinpath('aggregates')
        target_path.mkdir(exist_ok=True)
    else:
        assert target_path.exists(), 'The target_path you passed does not exist.'

    # Get the datasets
    trials_ds = []
    training_ds = []
    if subject:
        try:
            subject_id = uuid.UUID(subject)
        except ValueError:
            subject_id = one.alyx.rest('subjects', 'list', nickname=subject)[0]['id']
        if trials:
            trials_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTrials.table.pqt',
                                           django=f'object_id,{subject_id}'))
        if training:
            training_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTraining.table.pqt',
                                             django=f'object_id,{subject_id}'))
    else:
        if tag:
            if trials:
                trials_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTrials.table.pqt', tag=tag))
            if training:
                training_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTraining.table.pqt', tag=tag))
        else:
            if trials:
                trials_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTrials.table.pqt'))
            if training:
                training_ds.extend(one.alyx.rest('datasets', 'list', name='_ibl_subjectTraining.table.pqt'))

    # Set up the bucket
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)

    all_out = []
    for ds_list in [trials_ds, training_ds]:
        out_paths = []
        for ds in ds_list:
            relative_path = add_uuid_string(ds['file_records'][0]['relative_path'], ds['url'][-36:])
            src_path = 'aggregates/' + str(relative_path)
            dst_path = target_path.joinpath(relative_path)
            if check_updates:
                out = aws.s3_download_file(src_path, dst_path, s3=s3, bucket_name=bucket_name, overwrite=overwrite)
            else:
                out = dst_path

            if out and out.exists():
                out_paths.append(out)
            else:
                print(f'Downloading of {src_path} table failed.')
        all_out.append(out_paths)

    return all_out[0], all_out[1]


def query_subjects_interest(protocol='training', ibl_project='ibl_neuropixel_brainwide_01'):
    
    # Function to query subjects of interest based on task protocol and project
    """ Download session data """
    # Search sessions of interest
    sessions = one.search(task_protocol=protocol, project=ibl_project, details=True)
    session_details = sessions[1]

    """ List animals of interest"""
    subjects_interest = []
    for s, ses in enumerate(session_details):
        nickname = session_details[s]['subject']
        subjects_interest = np.append(subjects_interest, nickname)

    subjects_interest = np.unique(subjects_interest)
    return subjects_interest


def subjects_interest_data(subjects_interest, phase, protocol):
    
    # Parameters
    # phase can be 'learning' or 'profficient'

    all_data = pd.DataFrame()
    # Loop through subjects and get data and training status for each
    for s, subject in enumerate(subjects_interest):

        subject_trials, subject_training = download_subjectTables(one, subject=subject, trials=True, training=True,
                            target_path=None, tag=None, overwrite=False, check_updates=True)

        # Check if there is data for this mouse
        if (len(subject_trials) > 0) & (len(subject_training) > 0):
            dsets = [subject_trials[0], subject_training[0]]
            files = [one.cache_dir.joinpath(x) for x in dsets]
            trials, training = [pd.read_parquet(file) for file in files]
            trials['subject_nickname'] = subject
            
            # Check if animal ever got trained
            if 'trained 1a' in training['training_status'].unique():
                training_date = list(training.loc[training['training_status']=='trained 1a'].reset_index()['date'])[0]
            elif 'trained 1b' in training['training_status'].unique():
                training_date = list(training.loc[training['training_status']=='trained 1b'].reset_index()['date'])[0]
            else:
                training_date = []

            # If animal got trained, include
            if len(training_date) > 0:
                # Check phase of interest
                if phase == 'learning':
                    # If learning keep all sessions until trained
                    subject_data = trials.loc[trials['session_start_time'] <= pd.to_datetime(training_date)]
                if phase == 'proficient':
                    # If proficient, take the date of trained_1b:
                    # Check if animal ever got trained
                    if 'trained 1b' in training['training_status'].unique():
                        training_1b = list(training.loc[training['training_status']=='trained 1b'].reset_index()['date'])[0]
                    else:
                        training_1b = []
                        
                    # Select protocol
                    if protocol == 'biased':
                        # If profficient keep all biased sessions after 1b
                        subject_data = trials.loc[(trials['session_start_time'] > pd.to_datetime(training_1b)) 
                                                & (trials['task_protocol'].apply(lambda x: x[14:18])=='bias')]
                    elif protocol == 'ephys':
                        # If profficient keep all biased sessions after 1b
                        subject_data = trials.loc[(trials['session_start_time'] > pd.to_datetime(training_1b)) 
                                                & (trials['task_protocol'].apply(lambda x: x[14:18])=='ephy')]
                    else:
                        print('Protocol not contemplated yet')
                        

                # Save to main dataframe
                if len(all_data) == 0:
                    all_data = subject_data
                else:
                    all_data = all_data.append(subject_data)
        else:
            print(subject)

    return all_data

    
def get_trials(training_protocol='training', mouse_project='ibl_neuropixel_brainwide_01'):

    # GETS DATA PER SESSION
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
        # Need to first check if animal ever got trained
        mouse_training_status = training_time_df.loc[(training_time_df['subject_nickname'] == nickname),
                                     'training_status'] 
        if 'trained 1a' in mouse_training_status.unique() or 'trained 1b' in mouse_training_status.unique():
            training_days = np.max(training_time_df.loc[(training_time_df['subject_nickname'] ==
                                                        nickname) &
                                                        (training_time_df['training_status'] ==
                                                        'in training'), 'training_day'])
            # Add training time to dataframe
            training_time_df.loc[training_time_df['subject_nickname'] == nickname,
                                'training_time'] = training_days
        else:
            # Add training time to dataframe
            training_time_df.loc[training_time_df['subject_nickname'] == nickname,
                                'training_time'] = np.nan


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


"""
STACK TIME SERIES DATA FOR psth
"""
def timeseries_PSTH(time, position, trials, event, t_init, t_end, subtract_baseline):
    
    # subtract_baseline can be True, False, or pupil
    
    series_df = pd.DataFrame({'time':time, 'position':position})
    onset_times = trials[event]

    # Start a matrix with #trials x # wheel bins
    time_step = np.median(np.diff(time))
    interval_length = int((t_end+t_init)/time_step + .25 * 
                          (t_end+t_init)/time_step) # This serves as an estimation for size of data
    series_stack = np.zeros((len(onset_times), interval_length)) * np.nan

    # Loop through trials
    for t, trial_onset in enumerate(onset_times):
        if np.isnan(trial_onset) == False:
            if len(series_df.loc[series_df['time'] > trial_onset, 'time']) > 0:
                trial_onset_index = series_df.loc[series_df['time'] > trial_onset, 
                                                  'time'].reset_index()['index'][0]
                # Get time from first trial (only once to avoid the last trial)
                if t == 1:
                    onset_time = series_df['time'][trial_onset_index]
                    time_window = series_df.loc[(series_df['time']> trial_onset-t_init) & 
                                                (series_df['time'] <= trial_onset+t_end), 'time'] - onset_time
                
                # Subtract baseline if requested
                if subtract_baseline == True:
                    onset_position = series_df['position'][trial_onset_index]
                    # Populate dataframe with useful trial-aligned information
                    window_values = series_df.loc[(series_df['time']> trial_onset-t_init) & 
                                                (series_df['time'] <= trial_onset+t_end), 
                                                'position'] - onset_position 
                elif subtract_baseline == False:
                    window_values = series_df.loc[(series_df['time']> trial_onset-t_init) & 
                                                  (series_df['time'] <= trial_onset+t_end), 'position']
                    
                elif subtract_baseline == 'pupil':
                    max_pupil = np.max(series_df['position'])
                    min_pupil = np.min(series_df['position'])
                    series_df['norm_position'] = (series_df['position']) * 100 / (max_pupil - min_pupil)  #  (series_df['position'] - min_pupil) * 100 / (max_pupil - min_pupil)
                    baseline = np.mean(series_df.loc[(series_df['time'] > trial_onset-t_init) & 
                                                     (series_df['time'] < trial_onset), 'norm_position'])
                    window_values = series_df.loc[(series_df['time']> trial_onset-t_init) & 
                                                (series_df['time'] <= trial_onset+t_end), 
                                                'norm_position'] - baseline 

                series_stack[t, :len(window_values)] = window_values
                
    # Build data frame with extra info
    preprocessed_trials = prepro(trials)
    df_stack = pd.DataFrame(series_stack[:, :len(window_values)])
    df_stack['feedback'] = preprocessed_trials['feedbackType']
    df_stack['choice'] = preprocessed_trials['choice']
    df_stack['contrast'] = preprocessed_trials['contrast']
    df_stack['response_time'] = preprocessed_trials['response_times'] - preprocessed_trials['stimOn_times']
    df_stack['feedback_time'] = preprocessed_trials['feedback_times'] - preprocessed_trials['stimOn_times']

    df_melted = pd.melt(df_stack, id_vars=['feedback', 'choice', 'contrast', 
                                           'response_time', 'feedback_time'], 
                        value_vars=np.array(df_stack.keys()[1:-5]))
    
    # Rename variable to reflect event-aligned time
    df_melted['variable'] = df_melted['variable'].replace(np.arange(1, int(np.max(df_melted['variable'])+1)), 
                                                          np.array(list(time_window)[:int(np.max(df_melted['variable']))]))

    return df_melted

