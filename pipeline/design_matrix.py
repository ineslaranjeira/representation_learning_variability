"""
Pre-processing pipeline for brain-wide map (BWM) behavioral data and design matrices.
Parameterized functions for:
1. Querying and QC filtering sessions
2. Loading and preparing BWM data
3. Inspecting design matrices
@author: Ines
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import scipy.interpolate as interpolate
import gc
import concurrent.futures
import gzip
from dateutil import parser
from datetime import datetime
from brainbox.io.one import SessionLoader
from one.api import ONE
import glob

# Import helper utilities from a consolidated module
pipeline_dir = str(Path(__file__).resolve().parent)
if pipeline_dir not in sys.path:
    sys.path.append(pipeline_dir)

from design_matrix_utils import (
    merge_licks,
    resample_common_time,
    interpolate_nans,
    low_pass,
    extended_qc,
    lick_psth,
    get_speed,
    event_locked_signal,
    plot_licks_PSTH,
    plot_whisker_psth,
    plot_paw_hist,
    plot_choice_psth,
    plot_paw_feedback_psth,
    plot_paw_left_feedbackpsth,
    plot_paw_right_feedbackpsth,
    plot_wheel_choicepsth,
)

# ============================================================================
# 1. SESSIONS QUERY AND QC
# ============================================================================

def query_and_filter_bwm_sessions(
    one_instance=None,
    base_query=None,
    qc_task=None,
    marked_pass=None
):
    """
    Query brainwide map sessions and filter based on QC criteria.
    
    Parameters
    ----------
    one_instance : ONE instance, optional
        ONE instance to use. If None, creates a new one in remote mode.
    base_query : str, optional
        Base query string for session filtering. If None, uses default BWM queries.
    qc_task : str, optional
        Task QC query string. If None, uses default task QC filters.
    marked_pass : str, optional
        Query for manually marked pass sessions. If None, uses default.

    Returns
    -------
    bwm_df : pd.DataFrame
        DataFrame with session information including eid, pid, probe_name, etc.
    """
    
    if one_instance is None:
        one_instance = ONE(mode='remote')
    
    # Default queries if not provided
    if base_query is None:
        base_query = (
            'session__projects__name__icontains,ibl_neuropixel_brainwide_01,'
            '~session__json__IS_MOCK,True,'
            'session__qc__lt,50,'
            'session__extended_qc__behavior,1,'
            '~json__qc,CRITICAL,'
            'json__extended_qc__alignment_count__gt,0,'
        )
    
    if qc_task is None:
        qc_task = (
            '~session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
            '~session__extended_qc___task_response_feedback_delays__lt,0.9,'
            '~session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
            '~session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
            '~session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
            '~session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
            '~session__extended_qc___task_reward_volumes__lt,0.9,'
            '~session__extended_qc___task_reward_volume_set__lt,0.9,'
            '~session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
            '~session__extended_qc___task_audio_pre_trial__lt,0.9'
        )
    
    if marked_pass is None:
        marked_pass = 'session__extended_qc___experimenter_task,PASS'
    
    # Query insertions
    insertions = list(one_instance.alyx.rest('insertions', 'list', django=base_query + qc_task))
    insertions.extend(list(one_instance.alyx.rest('insertions', 'list', django=base_query + marked_pass)))
    
    print(f"Found {len(insertions)} insertions")
    
    # Create DataFrame
    bwm_df = pd.DataFrame({
        'pid': np.array([i['id'] for i in insertions]),
        'eid': np.array([i['session'] for i in insertions]),
        'probe_name': np.array([i['name'] for i in insertions]),
        'session_number': np.array([i['session_info']['number'] for i in insertions]),
        'date': np.array([parser.parse(i['session_info']['start_time']).date() for i in insertions]),
        'subject': np.array([i['session_info']['subject'] for i in insertions]),
        'lab': np.array([i['session_info']['lab'] for i in insertions]),
    }).sort_values(by=['lab', 'subject', 'date', 'eid'])
    
    bwm_df.drop_duplicates(inplace=True)
    bwm_df.reset_index(inplace=True, drop=True)
    
    return bwm_df


def filter_video_qc(
    one_instance,
    session_eids,
    left_qc_criteria=None,
    right_qc_criteria=None,
):
    """
    Filter sessions based on video QC criteria.
    
    Parameters
    ----------
    one_instance : ONE instance
        ONE instance for loading data.
    session_eids : array-like
        Unique session EIDs to filter.
    left_qc_criteria : dict, optional
        Left camera QC criteria. If None, uses default criteria.
    right_qc_criteria : dict, optional
        Right camera QC criteria. If None, uses default criteria.
    extended_qc_func : callable, optional
        Function to compute extended QC. If None, assumes ext_qc is available.
    
    Returns
    -------
    final_qc : pd.DataFrame
        Filtered DataFrame containing sessions that pass all QC criteria.
    """
    
    # Default criteria
    if left_qc_criteria is None:
        left_qc_criteria = {
            '_lightningPoseLeft_lick_detection': ['PASS'],
            '_lightningPoseLeft_time_trace_length_match': ['PASS'],
            '_lightningPoseLeft_trace_all_nan': ['PASS'],
            '_videoLeft_pin_state': 'pass_or_list',
            '_videoLeft_camera_times': 'pass_or_list',
            '_videoLeft_dropped_frames': 'pass_or_list_or_none',
            '_videoLeft_timestamps': [True, 'PASS'],
        }
    
    if right_qc_criteria is None:
        right_qc_criteria = {
            '_lightningPoseRight_lick_detection': ['PASS'],
            '_lightningPoseRight_time_trace_length_match': ['PASS'],
            '_lightningPoseRight_trace_all_nan': ['PASS'],
            '_videoRight_pin_state': 'pass_or_list',
            '_videoRight_camera_times': 'pass_or_list',
            '_videoRight_dropped_frames': 'pass_or_list_or_none',
            '_videoRight_timestamps': [True, 'PASS'],
        }
    
    # Get extended QC
    ext_qc = extended_qc(one_instance, session_eids)

    # Apply left camera filters
    final_qc = ext_qc.loc[(ext_qc['_lightningPoseLeft_lick_detection'].isin(['PASS'])) &
                          (ext_qc['_lightningPoseLeft_time_trace_length_match'].isin(['PASS'])) &   
                          (ext_qc['_videoLeft_pin_state'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &
                          (ext_qc['_lightningPoseLeft_trace_all_nan'].isin(['PASS'])) & 
                          (ext_qc['_videoLeft_camera_times'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &   
                          (ext_qc['_videoLeft_dropped_frames'].apply(lambda x: (isinstance(x, list) and True in x) or x == None or x == 'PASS')) &
                          (ext_qc['_videoLeft_timestamps'].isin([True, 'PASS']))]
    
    # Apply right camera filters
    final_qc = final_qc.loc[(final_qc['_lightningPoseRight_lick_detection'].isin(['PASS'])) &
                            (final_qc['_lightningPoseRight_time_trace_length_match'].isin(['PASS'])) &   
                            (final_qc['_videoRight_pin_state'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &
                            (final_qc['_lightningPoseRight_trace_all_nan'].isin(['PASS'])) & 
                            (final_qc['_videoRight_camera_times'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &   
                            (final_qc['_videoRight_dropped_frames'].apply(lambda x: (isinstance(x, list) and True in x) or x == None or x == 'PASS')) &
                            (final_qc['_videoRight_timestamps'].isin([True, 'PASS']))]
    
    return final_qc


def save_qc_data(
    qc_df,
    qc_results_path,
    filename_prefix='bwm_qc_',
    format='parquet'
):
    """
    Save QC-filtered data to disk.
    
    Parameters
    ----------
    qc_df : pd.DataFrame
        DataFrame to save.
    qc_results_path : str
        Path to save directory.
    filename_prefix : str, default 'bwm_qc_'
        Prefix for the filename.
    format : str, default 'parquet'
        Format to save in ('parquet' or 'pickle').
    
    Returns
    -------
    full_path : str
        Full path to saved file.
    """
    
    os.makedirs(qc_results_path, exist_ok=True)
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y")
    base_name = filename_prefix + date_time

    # Ensure files have an explicit extension so they can be discovered reliably
    if format == 'parquet':
        filename = base_name + '.parquet'
        full_path = os.path.join(qc_results_path, filename)
        qc_df.to_parquet(full_path, compression='gzip')
    elif format == 'pickle':
        filename = base_name + '.pkl'
        full_path = os.path.join(qc_results_path, filename)
        qc_df.to_pickle(full_path, compression='gzip')
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Saved to {full_path}")
    return full_path


def load_qc_data(
    qc_results_path,
    filename=None
):
    """
    Load QC-filtered data from disk.

    Parameters
    ----------
    qc_results_path : str
        Directory containing saved QC files.
    filename : str, optional
        Exact filename to load. If None, loads the most recent QC file.

    Returns
    -------
    qc_df : pd.DataFrame
        QC DataFrame loaded from disk.
    full_path : str
        Path of the file actually loaded.
    """
    if filename is None:
        # look for parquet or pickle by wildcard
        pattern = os.path.join(qc_results_path, 'bwm_qc_*')
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No saved QC files found in {qc_results_path}")
        full_path = max(matches, key=os.path.getmtime)
    else:
        full_path = os.path.join(qc_results_path, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"QC file not found: {full_path}")

    load_errs = {}
    try:
        qc_df = pd.read_parquet(full_path)
        print(f"Loaded QC results from {full_path} (parquet)")
        return qc_df, full_path
    except Exception as e_parquet:
        load_errs['parquet'] = str(e_parquet)

    try:
        qc_df = pd.read_pickle(full_path)
        print(f"Loaded QC results from {full_path} (pickle)")
        return qc_df, full_path
    except Exception as e_pickle:
        load_errs['pickle'] = str(e_pickle)

    try:
        with gzip.open(full_path, 'rb') as f:
            qc_df = pickle.load(f)
        print(f"Loaded QC results from {full_path} (gzip+pickle)")
        return qc_df, full_path
    except Exception as e_gz:
        load_errs['gzip_pickle'] = str(e_gz)

    raise RuntimeError(f"Failed to load QC file {full_path}. Attempts: {load_errs}")


# ============================================================================
# 3. SESSION FILTERING FOR DESIGN MATRIX PROCESSING
# ============================================================================

def filter_sessions_for_processing(
    sessions,
    one_instance=None,
    exclude_sessions=None,
    data_path=None,
    prefix_filter='design_matrix_'
):
    """
    Filter sessions to process by checking for existing design matrix files.
    
    Parameters
    ----------
    sessions : array-like
        Session EIDs to check.
    one_instance : ONE instance, optional
        ONE instance for session path lookup. If None, creates a new one.
    exclude_sessions : list, optional
        Session EIDs to exclude explicitly.
    data_path : str, optional
        Directory where processed design matrix files are saved.
    prefix_filter : str, default 'design_matrix_'
        Filename prefix for design matrix files.
    
    Returns
    -------
    sessions_to_process : list
        Sessions that should be processed.
    """
    if one_instance is None:
        one_instance = ONE(mode='remote')
    
    if exclude_sessions is None:
        exclude_sessions = []
    
    sessions_to_process = []
    
    files = []
    if data_path is not None and os.path.isdir(data_path):
        files = os.listdir(data_path)
    
    for sess in sessions:
        if sess in exclude_sessions:
            continue
        
        filename = None
        if data_path is not None:
            try:
                # check for any file starting with the expected prefix+session id
                file_path = one_instance.eid2path(sess)
                if prefix_filter == 'design_matrix_':
                    if '/home/ines/repositories/' in str(file_path):
                        mouse_name = file_path.parts[8]
                    else:
                        mouse_name = file_path.parts[7]
                    expected_prefix = f"{prefix_filter}{sess}_"
                else:
                    expected_prefix = f"{prefix_filter}{sess}"

                # if any file in the data_path starts with the expected prefix, assume processed
                if any(f.startswith(expected_prefix) for f in files):
                    continue
            except Exception:
                pass
        
        sessions_to_process.append(sess)
    
    return sessions_to_process


# ============================================================================
# 4. DESIGN MATRIX INSPECTION AND VISUALIZATION
# ============================================================================

def get_mouse_name_from_path(file_path):
    """
    Extract mouse name/subject name from an ONE file path.
    Typically paths look like /path/to/Subjects/subject_name/date/session_num/...
    Or relative parts.
    """
    parts = list(Path(file_path).parts)
    try:
        # Search for 'Subjects' folder
        if 'Subjects' in parts:
            idx = parts.index('Subjects')
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception:
        pass
    
    # Fallback to the parent of the parent of the parent or something similar
    # Typically session paths end in Subjects/subject_name/date/number
    # If not found, look at elements in order: Subjects -> subject_name.
    # If file_path has at least 3 parts, try to take parts[7] or parts[8] based on standard folders
    if len(parts) > 8:
        return parts[8] if 'repositories' in str(file_path) else parts[7]
    elif len(parts) > 7:
        return parts[7]
    return "unknown_subject"


def get_design_matrix_filename(
    session,
    one_instance=None,
    data_path=None,
    prefix='/Users/ineslaranjeira/Documents/Repositories/'
):
    """
    Build the saved design matrix filename for a given session.
    """
    if one_instance is None:
        one_instance = ONE(mode='remote')
    
    file_path = one_instance.eid2path(session)
    mouse_name = get_mouse_name_from_path(file_path)
    
    # use explicit parquet extension
    return os.path.join(data_path, f"design_matrix_{session}_{mouse_name}.parquet")


def inspect_design_matrices(
    sessions,
    data_path,
    results_path,
    one_instance=None,
    prefix='/Users/ineslaranjeira/Documents/Repositories/'
):
    if one_instance is None:
        one_instance = ONE(mode='remote')
    
    os.makedirs(results_path, exist_ok=True)
    inspected = []
    failed = []
    for mat in sessions:
        try:
            filename = get_design_matrix_filename(mat, one_instance=one_instance, data_path=data_path, prefix=prefix)
            design_matrix = pd.read_parquet(filename)
            trials = one_instance.load_object(mat, obj='trials', namespace='ibl')
            trial_df = trials.to_df()
            lick_times = np.array(design_matrix.loc[design_matrix['Lick count'] == 1, 'Bin'])
            
            # Run each plot in a separate try-except block to make generation resilient
            plots = [
                ("licks_PSTH", lambda: plot_licks_PSTH(mat, trial_df, lick_times, results_path)),
                ("whisker_psth", lambda: plot_whisker_psth(mat, design_matrix, trial_df, 'stimOn_times', results_path)),
                ("paw_hist", lambda: plot_paw_hist(mat, design_matrix, results_path)),
                ("choice_psth", lambda: plot_choice_psth(mat, design_matrix, trial_df, results_path, event_key='choice', feature='paw', camera_split=True, title='Paw choice PSTH', suffix='paw_choice_psth')),
                ("wheel_choicepsth", lambda: plot_wheel_choicepsth(mat, design_matrix, trial_df, results_path)),
                ("paw_left_feedbackpsth", lambda: plot_paw_left_feedbackpsth(mat, design_matrix, trial_df, results_path)),
                ("paw_right_feedbackpsth", lambda: plot_paw_right_feedbackpsth(mat, design_matrix, trial_df, results_path))
            ]
            
            for plot_name, plot_func in plots:
                try:
                    plot_func()
                except Exception as e_plot:
                    print(f"Warning: Plot '{plot_name}' failed for session {mat}: {e_plot}")
            
            inspected.append(mat)
            print(f'Plotted inspection for session {mat}')
        except Exception as e:
            failed.append((mat, str(e)))
            print(f'Failed inspection/loading for session {mat}: {e}')
    return {
        'inspected': inspected,
        'failed': failed,
        'total': len(sessions)
    }


# ============================================================================
# 5. DESIGN MATRIX PROCESSING AND CREATION
# ============================================================================

def process_design_matrix(
    session,
    one_instance=None,
    data_path=None,
    prefix='/Users/ineslaranjeira/Documents/Repositories/',
    paw_states=False
):
    """
    Process a single session and create design matrix with behavioral variables.
    
    Parameters
    ----------
    session : str
        Session EID to process.
    one_instance : ONE instance, optional
        ONE instance for loading data. If None, creates a new one.
    data_path : str, optional
        Path to save design matrix. If None, uses default.
    prefix : str, default
        System prefix for path extraction.
    paw_states : bool, default False
        If True, includes paw state variables in design matrix.
    
    Returns
    -------
    success : bool
        Whether processing was successful.
    """
    
    if one_instance is None:
        one_instance = ONE(mode='remote')
    
    try:
        file_path = one_instance.eid2path(session)
        
        # Extract mouse name from path
        mouse_name = get_mouse_name_from_path(file_path)
        
        # Load session data
        sl = SessionLoader(eid=session, one=one_instance)
        sl.load_pose(views=['left', 'right'], tracker='lightningPose')
        sl.load_session_data(trials=True, wheel=True, motion_energy=True)
        
        # Check if all data is available
        if np.sum(sl.data_info['is_loaded']) < 4:
            print(f'Data missing for session {session}')
            return False
        
        # Extract pose data
        poses = sl.pose
        lc_t = np.asarray(poses['leftCamera']['times'])
        rc_t = np.asarray(poses['rightCamera']['times'])
        left_fr = int(1 / np.nanmean(np.diff(lc_t)))
        right_fr = int(1 / np.nanmean(np.diff(rc_t)))
        
        # Check frame rates are sufficient
        if not (left_fr > 55 and right_fr > 60):
            print(f'Frame rates too low for session {session}: left={left_fr}, right={right_fr}')
            return False
        
        # Process motion energy
        me = sl.motion_energy
        if left_fr > 60:
            motion_energy_l = low_pass(
                interpolate_nans(me['leftCamera']['whiskerMotionEnergy'], left_fr),
                cutoff=30, sf=left_fr
            )
        else:
            motion_energy_l = interpolate_nans(me['leftCamera']['whiskerMotionEnergy'], left_fr)
        
        motion_energy_r = low_pass(
            interpolate_nans(me['rightCamera']['whiskerMotionEnergy'], right_fr),
            cutoff=30, sf=right_fr
        )
        
        # Extract licks
        features = ['tongue_end_l_x', 'tongue_end_l_y', 'tongue_end_r_x', 'tongue_end_r_y']
        lick_t, licks = merge_licks(poses, features, common_fs=150)
        
        # Extract paw coordinates
        if left_fr > 60:
            l_paw_x = low_pass(
                interpolate_nans(poses['leftCamera']['paw_r_x'], left_fr),
                cutoff=30, sf=left_fr
            )
            l_paw_y = low_pass(
                interpolate_nans(poses['leftCamera']['paw_r_y'], left_fr),
                cutoff=30, sf=left_fr
            )
        else:
            l_paw_x = interpolate_nans(poses['leftCamera']['paw_r_x'], left_fr)
            l_paw_y = interpolate_nans(poses['leftCamera']['paw_r_y'], left_fr)
        
        r_paw_x = low_pass(
            interpolate_nans(poses['rightCamera']['paw_r_x'], right_fr),
            cutoff=30, sf=right_fr
        )
        r_paw_y = low_pass(
            interpolate_nans(poses['rightCamera']['paw_r_y'], right_fr),
            cutoff=30, sf=right_fr
        )
        
        l_paw_t = lc_t
        r_paw_t = rc_t
        
        # Extract wheel data
        wheel = sl.wheel
        wheel_t = np.asarray(wheel['times'], dtype=np.float64)
        wheel_vel = wheel['velocity'].astype(np.float32)
        
        # Load paw states if requested
        paws_states_leftCamera = None
        paws_states_rightCamera = None
        if paw_states:
            try:
                times_l = one_instance.load_dataset(session, '_ibl_leftCamera.times.npy')
                times_r = one_instance.load_dataset(session, '_ibl_rightCamera.times.npy')
                assert np.all(lc_t == times_l)
                assert np.all(rc_t == times_r)
                paws_states_leftCamera = one_instance.load_dataset(session, '_ibl_leftCamera.pawstates.pqt')
                paws_states_rightCamera = one_instance.load_dataset(session, '_ibl_rightCamera.pawstates.pqt')
                paws_states_leftCamera['times'] = times_l
                paws_states_rightCamera['times'] = times_r
            except:
                print(f"Warning: Could not load paw states for session {session}")
                paw_states = False
        
        # Find common resampling window
        onset = max(lc_t.min(), rc_t.min(), wheel_t.min(), lick_t.min())
        offset = min(lc_t.max(), rc_t.max(), wheel_t.max(), lick_t.max())
        fs = 60
        ref_t = np.arange(onset, offset, 1 / fs, dtype=np.float64)
        
        # Restrict data to common time window
        def restrict(t, x):
            mask = (t >= onset) & (t <= offset)
            return t[mask], x[mask]
        
        mel_t, motion_energy_l = restrict(lc_t, motion_energy_l)
        mer_t, motion_energy_r = restrict(rc_t, motion_energy_r)
        wheel_t, wheel_vel = restrict(wheel_t, wheel_vel)
        l_paw_t_x, l_paw_x = restrict(l_paw_t, l_paw_x)
        l_paw_t_y, l_paw_y = restrict(l_paw_t, l_paw_y)
        r_paw_t_x, r_paw_x = restrict(r_paw_t, r_paw_x)
        r_paw_t_y, r_paw_y = restrict(r_paw_t, r_paw_y)
        lick_t, licks = restrict(lick_t, licks)
        
        # Resample all data to common timebase at 60 Hz
        mel_down, rt = resample_common_time(ref_t, mel_t, motion_energy_l, kind='linear')
        mer_down, _ = resample_common_time(ref_t, mer_t, motion_energy_r, kind='linear')
        wh_down, _ = resample_common_time(ref_t, wheel_t, wheel_vel, kind='linear')
        lk_down, _ = resample_common_time(ref_t, lick_t, licks, kind='nearest')
        lpx_down, _ = resample_common_time(ref_t, l_paw_t_x, l_paw_x, kind='linear')
        lpy_down, _ = resample_common_time(ref_t, l_paw_t_y, l_paw_y, kind='linear')
        rpx_down, _ = resample_common_time(ref_t, r_paw_t_x, r_paw_x, kind='linear')
        rpy_down, _ = resample_common_time(ref_t, r_paw_t_y, r_paw_y, kind='linear')
        
        # Create design matrix
        design_matrix = pd.DataFrame({
            'Bin': rt,
            'Lick count': lk_down.astype(np.int8),
            'avg_wheel_vel': wh_down,
            'whisker_me': mel_down,
            'avg_whisker_me': np.nanmean([mel_down, mer_down], axis=0),
            'l_paw_x': lpx_down,
            'l_paw_y': lpy_down,
            'r_paw_x': rpx_down,
            'r_paw_y': rpy_down,
        })
        
        # Append paw states if available
        if paw_states and paws_states_leftCamera is not None:
            paw_vars = [
                'paw_r_still', 'paw_r_move', 'paw_r_wheel_turn', 'paw_r_groom',
                'paw_r_still_ens_var', 'paw_r_move_ens_var', 'paw_r_wheel_turn_ens_var', 'paw_r_groom_ens_var',
                'paw_l_still', 'paw_l_move', 'paw_l_wheel_turn', 'paw_l_groom',
                'paw_l_still_ens_var', 'paw_l_move_ens_var', 'paw_l_wheel_turn_ens_var', 'paw_l_groom_ens_var'
            ]
            for state in paw_vars:
                if state in paws_states_leftCamera:
                    resampled, _ = resample_common_time(ref_t, paws_states_leftCamera['times'],
                                                       paws_states_leftCamera[state], kind='linear')
                    design_matrix[f'leftCam_{state}'] = resampled
                
                if state in paws_states_rightCamera:
                    resampled, _ = resample_common_time(ref_t, paws_states_rightCamera['times'],
                                                       paws_states_rightCamera[state], kind='linear')
                    design_matrix[f'rightCam_{state}'] = resampled
        
        # Load trial data and restrict design matrix
        session_trials = sl.trials
        session_start = list(session_trials['goCueTrigger_times'])[0]
        design_matrix = design_matrix.loc[(design_matrix['Bin'] > session_start)].reset_index(drop=True)
        
        # Save design matrix and trials
        if data_path is not None:
            os.makedirs(data_path, exist_ok=True)

            dm_fname = os.path.join(data_path, f"design_matrix_{session}_{mouse_name}.parquet")
            design_matrix.to_parquet(dm_fname, compression='gzip')

            trials_fname = os.path.join(data_path, f"session_trials_{session}_{mouse_name}.parquet")
            session_trials.to_parquet(trials_fname, compression='gzip')
            
            print(f"Saved design matrix for session {session}")
        
        # Cleanup
        del design_matrix, session_trials, sl, poses, me, wheel
        gc.collect()
        
        return True
    
    except Exception as e:
        print(f"Error processing session {session}: {e}")
        return False


def parallel_process_sessions(
    sessions,
    one_instance=None,
    data_path=None,
    prefix='/Users/ineslaranjeira/Documents/Repositories/',
    paw_states=False,
    n_workers=4
):
    """
    Process multiple sessions in parallel.
    
    Parameters
    ----------
    sessions : list
        List of session EIDs to process.
    one_instance : ONE instance, optional
        ONE instance for loading data.
    data_path : str, optional
        Path to save design matrices.
    prefix : str, default
        System prefix for path extraction.
    paw_states : bool, default False
        If True, includes paw state variables.
    n_workers : int, default 4
        Number of parallel workers.
    
    Returns
    -------
    results : dict
        Dictionary with processing results.
    """
    
    successful = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_design_matrix, session, one_instance, data_path, prefix, paw_states)
            for session in sessions
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                failed += 1
    
    results = {
        'successful': successful,
        'failed': failed,
        'total': len(sessions)
    }
    
    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_preprocessing_pipeline(
    config=None,
    steps=['qc', 'process_design_matrix', 'inspect_matrices']
):
    """
    Run the complete pre-processing pipeline.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with paths and parameters.
        Keys: 'prefix', 'qc_results_path', 'gdrive_prefix',
              'design_matrices_path', 'inspection_results_path',
              'exclude_sessions', 'paw_states', 'n_workers', 'steps'
    steps : list, default ['qc', 'process_design_matrix', 'inspect_matrices']
        Steps to execute in the pipeline.
        Available steps:
        - 'qc': query sessions, apply video QC, and save QC data
        - 'load_qc': load previously saved QC results without rerunning QC
        - 'process_design_matrix': build design matrices from QCed sessions
        - 'inspect_matrices': inspect saved design matrices and generate plots
    
    Returns
    -------
    results : dict
        Dictionary containing results from each step.
    """
    
    if config is None:
        config = {}
    steps = config.get('steps', steps)
    normalized_steps = set(steps)
    if any(step in normalized_steps for step in ['query_qc', 'filter_video', 'filter_qc', 'save_qc']):
        normalized_steps.add('qc')
    
    results = {}
    one_instance = ONE(mode='remote')
    
    config.setdefault('prefix', '/Users/ineslaranjeira/.gemini/antigravity/scratch/')
    config.setdefault('qc_results_path', None)
    # More explicit, per-user config keys
    config.setdefault('inspection_results_path', None)
    config.setdefault('design_matrices_path', None)
    config.setdefault('load_qc_path', None)
    # Optional per-computer Google Drive prefix (e.g. '/Users/you/Google Drive/My Drive/')
    config.setdefault('gdrive_prefix', None)
    config.setdefault('exclude_sessions', [])
    config.setdefault('paw_states', False)
    config.setdefault('n_workers', 4)
    
    # Auto-detect Google Drive prefix on macOS if not specified
    if config.get('gdrive_prefix') is None:
        # Check standard user home directory location
        gdrive_symlink = os.path.expanduser('~/Google Drive')
        if os.path.exists(gdrive_symlink):
            config['gdrive_prefix'] = gdrive_symlink
        else:
            # Fallback to the CloudStorage mount directly
            cloud_storage_root = os.path.expanduser('~/Library/CloudStorage')
            if os.path.exists(cloud_storage_root):
                for folder in os.listdir(cloud_storage_root):
                    if folder.startswith('GoogleDrive-'):
                        config['gdrive_prefix'] = os.path.join(cloud_storage_root, folder)
                        break

    if config.get('qc_results_path') is None:
        # default QC results: prefer a Google Drive prefix when provided
        gprefix = config.get('gdrive_prefix')
        if gprefix:
            # Check for the correct drive sub-path structures (My Drive vs O meu disco)
            possible_subdirs = ['My Drive', 'O meu disco']
            resolved_gprefix = gprefix
            for sd in possible_subdirs:
                test_path = os.path.join(gprefix, sd)
                if os.path.exists(test_path):
                    resolved_gprefix = test_path
                    break
            
            config['qc_results_path'] = os.path.join(
                resolved_gprefix,
                'CCU/PhD Project/paper-individuality/data/pipeline/segmentation/'
            )
        else:
            config['qc_results_path'] = os.path.join(
                config['prefix'],
                'representation_learning_variability/pipeline/'
            )
    
    # legacy `save_path` replaced by explicit `inspection_results_path`
    if config.get('inspection_results_path') is None:
        config['inspection_results_path'] = os.path.join(
            config['qc_results_path'],
            'design_matrix_inspection'
        )
    
    if config.get('design_matrices_path') is None:
        # store design matrices inside the segmentation folder under 'design_matrices'
        config['design_matrices_path'] = os.path.join(config['qc_results_path'], 'design_matrices')
        os.makedirs(config['design_matrices_path'], exist_ok=True)

    # ensure inspection results folder exists
    os.makedirs(config['inspection_results_path'], exist_ok=True)
    
    # Step 1: QC (query, filter, save)
    if 'qc' in normalized_steps:
        print("Step 1: Running QC pipeline (query, video filter, save)...")
        bwm_df = query_and_filter_bwm_sessions(one_instance=one_instance)
        results['bwm_df'] = bwm_df
        
        session_eids = results['bwm_df']['eid'].unique()
        final_qc = filter_video_qc(one_instance, session_eids)
        results['final_qc'] = final_qc
        
        save_path = config['qc_results_path']
        saved_path = save_qc_data(results['final_qc'], save_path)
        results['saved_path'] = saved_path
    
    # Step 2: Process design matrices
    # Optional step: load previously saved QC results without re-running QC
    if 'load_qc' in normalized_steps:
        print("Loading saved QC results (load_qc step)...")
        load_path = config.get('load_qc_path') or config['qc_results_path']
        final_qc, loaded_path = load_qc_data(load_path)
        results['final_qc'] = final_qc
        results['loaded_qc_path'] = loaded_path

    if 'process_design_matrix' in normalized_steps:
        print("Step 2: Processing design matrices...")
        
        if 'final_qc' not in results:
            raise ValueError("Must run 'qc' or 'load_qc' step first")
        
        # Get all sessions from final_qc
        sessions = list(results['final_qc']['eid'].unique())
        exclude_sessions = config.get('exclude_sessions', [])
        
        # Filter sessions by existing design matrix files and explicit exclusions
        sessions_to_process = filter_sessions_for_processing(
            sessions,
            one_instance=one_instance,
            exclude_sessions=exclude_sessions,
            data_path=config['design_matrices_path'],
            prefix_filter='design_matrix_'
        )
        
        print(f"Processing {len(sessions_to_process)} sessions...")
        
        # Process in parallel
        proc_results = parallel_process_sessions(
            sessions_to_process,
            one_instance=one_instance,
            data_path=config['design_matrices_path'],
            prefix=config['prefix'],
            paw_states=config.get('paw_states', False),
            n_workers=config.get('n_workers', 4)
        )
        
        results['process_results'] = proc_results
        print(f"Completed: {proc_results['successful']} successful, {proc_results['failed']} failed")
    
    # Step 3: Inspect design matrices
    if 'inspect_matrices' in normalized_steps:
        print("Step 3: Inspecting design matrices...")
        
        if 'final_qc' not in results:
            raise ValueError("Must run 'qc' or 'load_qc' step before inspecting matrices")
        
        sessions = list(results['final_qc']['eid'].unique())
        inspections = inspect_design_matrices(
            sessions,
            data_path=config['design_matrices_path'],
            results_path=os.path.join(config['inspection_results_path'], '0_pre-processing/inspection_plots'),
            one_instance=one_instance,
            prefix=config['prefix']
        )
        results['inspection_results'] = inspections
        results['inspected_sessions'] = sessions
        print(f"Completed inspection for {len(sessions)} sessions")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Design Matrix Processing Pipeline")
    print("=" * 50)
    
    # Run full pipeline saving results to a Google Drive mount
    print("\nExample: Run full pipeline with Google Drive prefix and explicit paths")
    config = {
        'steps': ['load_qc'],
        'gdrive_prefix': '/Users/ineslaranjeira/Google Drive/O meu disco/',
        'paw_states': False,
        'n_workers': 4,
    }
    results = run_preprocessing_pipeline(config=config)
