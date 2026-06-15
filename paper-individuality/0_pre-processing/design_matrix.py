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

# Import helper utilities from a consolidated module
try:
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
except ImportError:
    # Fall back to older module paths if needed
    from functions import merge_licks, resample_common_time, interpolate_nans, low_pass
    from Functions.video_functions import lick_psth
    from Functions.video_functions import extended_qc
    from design_matrix_utils import (
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
    marked_pass=None,
    return_extended_info=True
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
    return_extended_info : bool, default True
        If True, returns extended session information.
    
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
    extended_qc_func=None
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
    if extended_qc_func is None:
        from functions import extended_qc
        ext_qc = extended_qc(one_instance, session_eids)
    else:
        ext_qc = extended_qc_func(one_instance, session_eids)
    
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
    save_path,
    filename_prefix='1_bwm_qc_',
    format='parquet'
):
    """
    Save QC-filtered data to disk.
    
    Parameters
    ----------
    qc_df : pd.DataFrame
        DataFrame to save.
    save_path : str
        Path to save directory.
    filename_prefix : str, default '1_bwm_qc_'
        Prefix for the filename.
    format : str, default 'parquet'
        Format to save in ('parquet' or 'pickle').
    
    Returns
    -------
    full_path : str
        Full path to saved file.
    """
    
    os.makedirs(save_path, exist_ok=True)
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y")
    filename = filename_prefix + date_time
    
    if format == 'parquet':
        full_path = os.path.join(save_path, filename)
        qc_df.to_parquet(full_path, compression='gzip')
    elif format == 'pickle':
        full_path = os.path.join(save_path, filename)
        qc_df.to_pickle(full_path, compression='gzip')
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Saved to {full_path}")
    return full_path


# ============================================================================
# 2. LOAD BWM QCed DATA
# ============================================================================

def load_bwm_data(
    data_path,
    filename,
    from_gdrive=False
):
    """
    Load pre-processed BWM data (post-QC).
    
    Parameters
    ----------
    data_path : str
        Path to data directory.
    filename : str
        Filename (without extension).
    from_gdrive : bool, default False
        If True, assumes path is on Google Drive.
    
    Returns
    -------
    bwm_query : pd.DataFrame or similar
        Loaded data.
    """
    
    full_path = os.path.join(data_path, filename)
    
    try:
        bwm_query = pickle.load(gzip.open(full_path, "rb"))
        print(f"Loaded data from {full_path}")
        return bwm_query
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


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
                file_path = one_instance.eid2path(sess)
                if prefix_filter == 'design_matrix_':
                    if '/home/ines/repositories/' in str(file_path):
                        mouse_name = file_path.parts[8]
                    else:
                        mouse_name = file_path.parts[7]
                    filename = f"{prefix_filter}{sess}_{mouse_name}"
                else:
                    filename = f"{prefix_filter}{sess}"
                
                if filename in files:
                    continue
            except Exception:
                pass
        
        sessions_to_process.append(sess)
    
    return sessions_to_process


# ============================================================================
# 4. DESIGN MATRIX INSPECTION AND VISUALIZATION
# ============================================================================

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
    if '/home/ines/repositories/' in str(file_path):
        mouse_name = file_path.parts[8]
    else:
        mouse_name = file_path.parts[7]
    
    return os.path.join(data_path, f"design_matrix_{session}_{mouse_name}")


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
            plot_licks_PSTH(mat, trial_df, lick_times, results_path)
            plot_whisker_psth(mat, design_matrix, trial_df, 'stimOn_times', results_path)
            plot_paw_hist(mat, design_matrix, results_path)
            plot_choice_psth(mat, design_matrix, trial_df, results_path, event_key='choice', feature='paw', camera_split=True, title='Paw choice PSTH', suffix='paw_choice_psth')
            plot_wheel_choicepsth(mat, design_matrix, trial_df, results_path)
            plot_paw_left_feedbackpsth(mat, design_matrix, trial_df, results_path)
            plot_paw_right_feedbackpsth(mat, design_matrix, trial_df, results_path)
            inspected.append(mat)
            print(f'Plotted inspection for session {mat}')
        except Exception as e:
            failed.append((mat, str(e)))
            print(f'Failed inspection for session {mat}: {e}')
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
        if prefix == '/home/ines/repositories/':
            mouse_name = file_path.parts[8]
        else:
            mouse_name = file_path.parts[7]
        
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
            
            filename = os.path.join(data_path, f"design_matrix_{session}_{mouse_name}")
            design_matrix.to_parquet(filename, compression='gzip')
            
            filename = os.path.join(data_path, f"session_trials_{session}_{mouse_name}")
            session_trials.to_parquet(filename, compression='gzip')
            
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
    steps=['query_qc', 'filter_video', 'save_qc', 'process_design_matrix', 'inspect_matrices']
):
    """
    Run the complete pre-processing pipeline.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with paths and parameters.
        Keys: 'prefix', 'save_path', 'data_path', 'exclude_sessions', 'paw_states', 'n_workers'
    steps : list, default ['query_qc', 'filter_video', 'save_qc', 'process_design_matrix', 'inspect_matrices']
        Steps to execute in the pipeline.
    
    Returns
    -------
    results : dict
        Dictionary containing results from each step.
    """
    
    results = {}
    one_instance = ONE(mode='remote')
    
    if config is None:
        config = {
            'prefix': '/Users/ineslaranjeira/Documents/Repositories/',
            'save_path': None,
            'data_path': None,
            'exclude_sessions': [],
            'paw_states': False,
            'n_workers': 4,
        }
    
    if config.get('save_path') is None:
        config['save_path'] = os.path.join(
            config['prefix'],
            'representation_learning_variability/paper-individuality/'
        )
    
    if config.get('data_path') is None:
        config['data_path'] = os.path.join(
            config['prefix'],
            'representation_learning_variability/paper-individuality/data/design_matrices/'
        )
    
    # Step 1: Query and QC
    if 'query_qc' in steps:
        print("Step 1: Querying sessions and applying QC...")
        bwm_df = query_and_filter_bwm_sessions(one_instance=one_instance)
        results['bwm_df'] = bwm_df
    
    # Step 2: Filter video QC
    if 'filter_video' in steps:
        print("Step 2: Filtering based on video QC...")
        if 'bwm_df' not in results:
            raise ValueError("Must run 'query_qc' step first")
        
        session_eids = results['bwm_df']['eid'].unique()
        final_qc = filter_video_qc(one_instance, session_eids)
        results['final_qc'] = final_qc
    
    # Step 3: Save QC data
    if 'save_qc' in steps:
        print("Step 3: Saving QC-filtered data...")
        if 'final_qc' not in results:
            raise ValueError("Must run 'filter_video' step first")
        
        save_path = os.path.join(config['save_path'], '0_pre-processing/')
        saved_path = save_qc_data(results['final_qc'], save_path)
        results['saved_path'] = saved_path
    
    # Step 4: Process design matrices
    if 'process_design_matrix' in steps:
        print("Step 4: Processing design matrices...")
        
        if 'final_qc' not in results:
            raise ValueError("Must run 'filter_video' step first")
        
        # Get all sessions from final_qc
        sessions = list(results['final_qc']['eid'].unique())
        exclude_sessions = config.get('exclude_sessions', [])
        
        # Filter sessions by existing design matrix files and explicit exclusions
        sessions_to_process = filter_sessions_for_processing(
            sessions,
            one_instance=one_instance,
            exclude_sessions=exclude_sessions,
            data_path=config['data_path'],
            prefix_filter='design_matrix_'
        )
        
        print(f"Processing {len(sessions_to_process)} sessions...")
        
        # Process in parallel
        proc_results = parallel_process_sessions(
            sessions_to_process,
            one_instance=one_instance,
            data_path=config['data_path'],
            prefix=config['prefix'],
            paw_states=config.get('paw_states', False),
            n_workers=config.get('n_workers', 4)
        )
        
        results['process_results'] = proc_results
        print(f"Completed: {proc_results['successful']} successful, {proc_results['failed']} failed")
    
    # Step 5: Inspect design matrices
    if 'inspect_matrices' in steps:
        print("Step 5: Inspecting design matrices...")
        
        if 'final_qc' not in results:
            raise ValueError("Must run 'filter_video' step before inspecting matrices")
        
        sessions = list(results['final_qc']['eid'].unique())
        inspections = inspect_design_matrices(
            sessions,
            data_path=config['data_path'],
            results_path=config.get('results_path', os.path.join(config['save_path'], '0_pre-processing/inspection_plots')),
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
    
    # Full pipeline example
    print("\nExample 1: Run full pipeline")
    config = {
        'prefix': '/Users/ineslaranjeira/.gemini/antigravity/scratch/',
        'paw_states': False,
        'n_workers': 4,
    }
    # '/Users/ineslaranjeira/Documents/Repositories/'
    # results = run_preprocessing_pipeline(config=config)
    
    # Process only design matrices for existing sessions
    print("\nExample 2: Process design matrices only")
    from functions import extended_qc
    one = ONE(mode='remote')
    # sessions = [...]  # Your session EIDs
    # results = parallel_process_sessions(sessions, one_instance=one, data_path='...', n_workers=4)
    
    # Process a single session
    print("\nExample 3: Process single session")
    # success = process_design_matrix(
    #     session='your-session-eid',
    #     one_instance=one,
    #     data_path='/path/to/save',
    #     paw_states=False
    # )
    
    print("\nUse the functions directly for more control:")
    print("- query_and_filter_bwm_sessions()")
    print("- filter_video_qc()")
    print("- process_design_matrix()")
    print("- parallel_process_sessions()")
    print("- run_preprocessing_pipeline()")
