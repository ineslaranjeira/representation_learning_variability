"""
Functions to query mice/sessions/trials using DJ-exclusive functions
Jan 2023
Inês Laranjeira
"""

# %%
import pandas as pd
import numpy as np
import datetime
import pickle 
import os

import datajoint as dj
from ibl_pipeline import subject, acquisition, behavior
from ibl_pipeline.analyses import behavior as behavioral_analyses

# %%

"""
QUERY DATA
"""
CUTOFF_DATE = '2020-03-23'  # Date after which sessions are excluded, previously 30th Nov

def query_subjects(as_dataframe=False, criterion='trained'):
    """
    Query all mice for analysis of behavioral data
    Parameters
    ----------
    as_dataframe:    boolean if true returns a pandas dataframe (default is False)
    criterion:       what criterion by the 30th of November - trained (a and b), biased, ephys
                     (includes ready4ephysrig, ready4delay and ready4recording).  If None,
                     all mice that completed a training session are returned, with date_trained
                     being the date of their first training session.
    """
    from ibl_pipeline import subject, acquisition, reference
    from ibl_pipeline.analyses import behavior as behavior_analysis

    # Query all subjects with project ibl_neuropixel_brainwide_01 and get the date at which
    # they reached a given training status
    all_subjects = (subject.Subject * subject.SubjectLab * reference.Lab * subject.SubjectProject &
                    'subject_project = "ibl_neuropixel_brainwide_01"')
    sessions = acquisition.Session * behavior_analysis.SessionTrainingStatus()
    fields = ('subject_nickname', 'sex', 'subject_birth_date', 'institution_short')

    if criterion is None:
        # Find first session of all mice; date_trained = date of first training session
        subj_query = all_subjects.aggr(
            sessions, * fields, date_trained='min(date(session_start_time))')
    else:  # date_trained = date of first session when criterion was reached
        if criterion == 'trained':
            restriction = 'training_status="trained_1a" OR training_status="trained_1b"'
        elif criterion == 'biased':
            restriction = 'task_protocol LIKE "%biased%"'
        elif criterion == 'ephys':
            restriction = 'training_status LIKE "ready%"'
        else:
            raise ValueError('criterion must be "trained", "biased" or "ephys"')
        subj_query = all_subjects.aggr(
            sessions & restriction, * fields, date_trained='min(date(session_start_time))')

    # Select subjects that reached criterion before cutoff date
    subjects = (subj_query & 'date_trained <= "%s"' % CUTOFF_DATE)
    if as_dataframe is True:
        subjects = subjects.fetch(format='frame')
        subjects = subjects.sort_values(by=['lab_name']).reset_index()

    return subjects


def training_time(subjects):

    """
    Days until 'trained' criterion

    Parameters
    subjects:           DJ table of subjects of interest

    """
    # -- Query data
    sessions = (acquisition.Session * subjects * behavioral_analyses.SessionTrainingStatus).proj(
        'subject_uuid', 'training_status', 'session_start_time', 'session_uuid',
        session_date='DATE(session_start_time)')
    sessions = pd.DataFrame.from_dict(sessions.fetch(as_dict=True))

    df = pd.DataFrame(columns=['subject_uuid', 'training_time', 'date'], index=range(len(
        sessions['subject_uuid'].unique())))

    for i, mouse in enumerate(sessions['subject_uuid'].unique()):
        subj_sess = sessions.loc[sessions['subject_uuid'] == mouse].copy()
        subj_sess = subj_sess.drop_duplicates(subset=['session_uuid'])

        df['subject_uuid'][i] = mouse
        if (np.sum(subj_sess['training_status'] == "trained_1a") + np.sum(
                subj_sess['training_status'] == "trained_1b")) > 0:

            subj_sess = subj_sess.drop_duplicates(subset=['session_date'])
            trained_sessions = subj_sess.loc[subj_sess
                                             ['training_status'] ==
                                             'trained_1a'].append(subj_sess.loc[
                                                                  subj_sess['training_status'] ==
                                                                  'trained_1b']).sort_index()
            trained_session = trained_sessions.reset_index()[0:1]['session_start_time']
            df['training_time'][i] = len(subj_sess.loc[subj_sess['session_start_time'] <
                                                       trained_session[0]])
            df['date'][i] = list(trained_session)[0]
        else:
            df['training_time'][i] = np.nan
            df['date'][i] = np.nan

    return df


def training_trials(subjects):

    """
    Query all training trials for animals that got trained

    Parameters
    ----------
    subjects:         Subjects to query trials of

    """

    # --Find subjects that got trained
    trained = (acquisition.Session * subjects * behavioral_analyses.SessionTrainingStatus &
               'training_status in ("trained_1a", "trained_1b")').proj('training_status')
    trained = pd.DataFrame.from_dict(trained.fetch(as_dict=True))
    trained_subjects = trained['subject_uuid'].unique()

    # --Query training trials from animals that got trained

    training_sessions = (acquisition.Session * subject.Subject *
                         behavioral_analyses.SessionTrainingStatus &
                         [{'subject_uuid': eid} for eid in trained_subjects] &
                         'training_status in ("in_training", "untrainable")').proj(
        'session_uuid', 'subject_nickname', 'task_protocol', 'training_status',
        'session_start_time', 'session_lab', session_date='DATE(session_start_time)')
    #trials = (training_sessions * behavior.TrialSet.Trial *
    #          behavioral_analyses.BehavioralSummaryByDate & 'training_day <= 5')
    trials = (training_sessions * behavior.TrialSet.Trial *
              behavioral_analyses.BehavioralSummaryByDate)

    # -- Warning, this step takes up a lot of memory and some time
    trials_df = pd.DataFrame.from_dict(trials.fetch(as_dict=True))

    return trials_df


def training_sessions(subjects):

    """
    Query all training trials for animals that got trained

    Parameters
    ----------
    subjects:         Subjects to query trials of

    """

    # --Find subjects that got trained
    trained = (acquisition.Session * subjects * behavioral_analyses.SessionTrainingStatus &
               'training_status in ("trained_1a", "trained_1b")').proj('training_status')
    trained = pd.DataFrame.from_dict(trained.fetch(as_dict=True))
    trained_subjects = trained['subject_uuid'].unique()

    # --Query training trials from animals that got trained

    training_sessions = (acquisition.Session * subject.Subject *
                         behavioral_analyses.SessionTrainingStatus &
                         [{'subject_uuid': eid} for eid in trained_subjects] &
                         'training_status in ("in_training", "untrainable")').proj(
        'session_uuid', 'subject_nickname', 'task_protocol', 'training_status',
        'session_start_time', 'session_lab', 'session_number',
        session_date='DATE(session_start_time)')

    sessions = (training_sessions * 
                behavioral_analyses.BehavioralSummaryByDate).proj('session_uuid',
                                                                  'subject_nickname',
                                                                  'session_date',
                                                                  'session_number', 'training_day')

    # -- Warning, this step takes up a lot of memory and some time
    sessions_df = pd.DataFrame.from_dict(sessions.fetch(as_dict=True))

    return sessions_df