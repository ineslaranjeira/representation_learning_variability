import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import mode
import scipy.interpolate as interpolate
import pandas as pd

"""
SCRIPT: Syllables per trial epoch
"""
def idxs_from_files(design_matrices):
    
    idxs = []
    mouse_names = []
    for m, mat in enumerate(design_matrices):
        mouse_name = design_matrices[m][51:]
        eid = design_matrices[m][14:50]
        idx = str(eid + '_' + mouse_name)

        if len(idxs) == 0:
            idxs = idx
            mouse_names = mouse_name
        else:
            idxs = np.hstack((idxs, idx))
            mouse_names = np.hstack((mouse_names, mouse_name))
            
    return idxs, mouse_names

# This function uses get_XYs, not smoothing, is closer to brainbox function: https://github.com/int-brain-lab/ibllib/blob/78e82df8a51de0be880ee4076d2bb093bbc1d2c1/brainbox/behavior/dlc.py#L63
def get_speed(poses, times, camera, sampling_rate, split, feature):
    """
    FIXME Document and add unit test!

    :param dlc: dlc pqt table
    :param dlc_t: dlc time points
    :param camera: camera type e.g 'left', 'right', 'body'
    :param feature: dlc feature to compute speed over
    :return:
    """

    RESOLUTION = {'left': 2,
                  'right': 1,
                  'body': 1}
    sampling_rate = 60

    speeds = {}
    times = np.array(times)
    x = poses[f'{feature}_x'] / RESOLUTION[camera]
    y = poses[f'{feature}_y'] / RESOLUTION[camera]

    dt = np.diff(times)
    tv = times[:-1] + dt / 2


    # Calculate velocity for x and y separately if split is true
    if split == True:
        s_x = np.diff(x) * sampling_rate
        s_y = np.diff(y) * sampling_rate
        speeds = [times, s_x, s_y]
        # interpolate over original time scale
        if tv.size > 1:
            ifcn_x = interpolate.interp1d(tv, s_x, fill_value="extrapolate")
            ifcn_y = interpolate.interp1d(tv, s_y, fill_value="extrapolate")
            speeds = [times, ifcn_x(times), ifcn_y(times)]
    else:
        # Speed vector is given by the Pitagorean theorem
        s = ((np.diff(x)**2 + np.diff(y)**2)**.5) * sampling_rate
        speeds = [times, s]
        # interpolate over original time scale
        if tv.size > 1:
            ifcn = interpolate.interp1d(tv, s, fill_value="extrapolate")
            speeds = [times, ifcn(times)]

    return speeds  


""" State post-processing """
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
    trials['response'] = trials['response_times'] - trials['goCue_times']
    trials['reaction'] = trials['firstMovement_times'] - trials['goCue_times']
    """ Quiescence elongation """
    trials['elongation'] = trials['goCue_times'] - trials['quiescencePeriod'] - trials['intervals_0']
    """ Win stay lose shift """
    trials['prev_choice'] = trials['choice'] * np.nan
    trials['prev_choice'][1:] = trials['choice'][:-1]
    trials['prev_feedback'] = trials['feedbackType'] * np.nan
    trials['prev_feedback'][1:] = trials['feedbackType'][:-1]
    trials['wsls'] = trials['choice'] * np.nan
    trials.loc[(trials['prev_feedback']==1.) & (trials['choice']==trials['prev_choice']), 'wsls'] = 'wst'
    trials.loc[(trials['prev_feedback']==1.) & (trials['choice']!=trials['prev_choice']), 'wsls'] = 'wsh'
    trials.loc[(trials['prev_feedback']==-1.) & (trials['choice']!=trials['prev_choice']), 'wsls'] = 'lsh'
    trials.loc[(trials['prev_feedback']==-1.) & (trials['choice']==trials['prev_choice']), 'wsls'] = 'lst'
    #TODO : trials['days_to_trained'] = trials['training_time']

    return trials


def state_identifiability_old(combined_states, design_matrix_heading, use_sets):
    
    unique_states = np.unique(combined_states)
    new_states = unique_states.copy()

    # Create new mapping depending on empirical data for each state
    for v, var in enumerate(use_sets):
        zeros = [s[v] == '0' if s != 'nan' else False for s in combined_states]
        ones = [s[v] == '1' if s != 'nan' else False for s in combined_states]
        
        # For an empty variable, do not make changes (wavelet)
        if len(var) == 0:
            var_0 = np.nan
            var_1 = np.nan
        elif var == ['avg_wheel_vel']:
            var_0 = np.array(np.abs(design_matrix_heading[var]))[zeros]
            var_1 = np.array(np.abs(design_matrix_heading[var]))[ones]
        elif var == ['left_X', 'left_Y', 'right_X', 'right_Y']:
            var_0 = np.array(np.abs(np.diff(design_matrix_heading[var], axis=0)))[zeros[1:]]
            var_1 = np.array(np.abs(np.diff(design_matrix_heading[var], axis=0)))[ones[1:]]
        elif var == ['nose_x', 'nose_Y']:
            print('Not implemented yet')
        else:
            var_0 = np.array(design_matrix_heading[var])[zeros]
            var_1 = np.array(design_matrix_heading[var])[ones]
        
        if np.nanmean(var_0)> np.nanmean(var_1):
            var_state_0 = [s[v] == '0' if s != 'nan' else False for s in unique_states]
            new_states[var_state_0] = np.array([s[:v] + '1' + s[v+1:] for s in new_states[var_state_0]])
            var_state_1 = [s[v] == '1' if s != 'nan' else False for s in unique_states]
            new_states[var_state_1] = np.array([s[:v] + '0' + s[v+1:] for s in new_states[var_state_1]])

    identifiable_mapping = {unique: key for unique, key in zip(unique_states, new_states)}

    # Use np.vectorize to apply the mapping
    replace_func = np.vectorize(identifiable_mapping.get)
    identifiable_states = replace_func(combined_states)
    
    return identifiable_states


def state_identifiability(session_states, use_sets):
    # Create new mapping depending on empirical data for each state
    for v, var in enumerate(use_sets):
        var_states = var+'_states'
        
        # For an empty variable, do not make changes (wavelet)
        if len(var) == 0:
            var_0 = np.nan
            var_1 = np.nan
        elif var == ['avg_wheel_vel']:
            var_0 = np.nanmean(np.abs(session_states.loc[session_states[var_states]==0, var]))
            var_1 = np.nanmean(np.abs(session_states.loc[session_states[var_states]==1, var]))
        elif var == ['left_X', 'left_Y', 'right_X', 'right_Y']:
            var_0 = np.array(np.abs(np.diff(session_states.loc[session_states[var_states]==0, var])))
            var_1 = np.array(np.abs(np.diff(session_states.loc[session_states[var_states]==0, var])))
        elif var == ['nose_x', 'nose_Y']:
            print('Not implemented yet')
        else:
            var_0 = session_states.loc[session_states[var_states]==0, var]
            var_1 = session_states.loc[session_states[var_states]==1, var]
        
        if np.nanmean(var_0)> np.nanmean(var_1):
            session_states[var_states] = session_states[var_states] * -1 + 1
    return session_states


def align_bin_design_matrix (init, end, event_type_list, session_trials, design_matrix, most_likely_states, multiplier):
    
    for e, this_event in enumerate(event_type_list):
        
        # Initialize variables
        # Before there was a function for keeping validation set apart, now deprecated
        reduced_design_matrix = design_matrix.copy()
        reduced_design_matrix['most_likely_states'] = most_likely_states
        reduced_design_matrix['new_bin'] = reduced_design_matrix['Bin'] * np.nan
        reduced_design_matrix['correct'] = reduced_design_matrix['Bin'] * np.nan
        reduced_design_matrix['choice'] = reduced_design_matrix['Bin'] * np.nan
        reduced_design_matrix['contrast'] = reduced_design_matrix['Bin'] * np.nan        
        reduced_design_matrix['block'] = reduced_design_matrix['Bin'] * np.nan        

        feedback = session_trials['feedbackType']
        choice = session_trials['choice']
        contrast = np.abs(prepro(session_trials)['signed_contrast'])
        block = session_trials['probabilityLeft']
        reaction = prepro(session_trials)['reaction']
        response = prepro(session_trials)['response']
        elongation = prepro(session_trials)['elongation']
        wsls = prepro(session_trials)['wsls']
        trial_id = session_trials['index'] 

        events = session_trials[this_event]
                
        for t, trial in enumerate(events[:-1]):
            event = events[t]
            trial_start = session_trials['intervals_0'][t]
            trial_end = session_trials['intervals_0'][t+1]
            
            # Check feedback
            if feedback[t] ==1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'correct'] = 1
            elif feedback[t] == -1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'correct'] = 0
            # Check choice
            if choice[t] ==1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'choice'] = 'right'
            elif choice[t] == -1:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'choice'] = 'left'
            
            # Check reaction
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'reaction'] = reaction[t]
            # Check response
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'response'] = response[t]
            # Check elongation
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']>trial_start*multiplier), 
                                            'elongation'] = elongation[t]

            # Check contrast
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'contrast'] = contrast[t]

            # Check block
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'block'] = block[t]
            
            # Check wsls
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'wsls'] = wsls[t]

            # Check trial id
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            'trial_id'] = trial_id[t]
            
            # Add reliable timestamp to identify trial
            reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= trial_end*multiplier) &
                                            (reduced_design_matrix['Bin']> trial_start*multiplier), 
                                            this_event] = event
            
            # Rename bins so that they are aligned on stimulus onset
            if event > 0:
                event_window = reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) &
                                                         (reduced_design_matrix['Bin']> event*multiplier + init)]
                onset_bin = reduced_design_matrix.loc[reduced_design_matrix['Bin']>= event*multiplier, 'Bin']
                if (len(event_window)>0) & len(onset_bin)>0:
                    bin = list(onset_bin)[0]
                    reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) &
                                            (reduced_design_matrix['Bin']> event*multiplier + init), 
                                            'new_bin'] = reduced_design_matrix.loc[(reduced_design_matrix['Bin']< event*multiplier + end) & 
                                            (reduced_design_matrix['Bin']>= event*multiplier + init), 'Bin'] - bin
                else:
                    reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) & 
                                              (reduced_design_matrix['Bin']> event*multiplier + init), 'new_bin'] = np.nan
            else:
                reduced_design_matrix.loc[(reduced_design_matrix['Bin']<= event*multiplier + end) & 
                                          (reduced_design_matrix['Bin']> event*multiplier + init), 'new_bin'] = np.nan
                
    return reduced_design_matrix


def states_per_trial_phase(reduced_design_matrix, session_trials, multiplier):
    
    # Split session into trial phases and gather most likely states of those trial phases
    # use_data = reduced_design_matrix.dropna()
    use_data = reduced_design_matrix.copy()
    use_data['label'] = use_data['Bin'] * np.nan
    trial_num = len(session_trials)

    # Pre-quiescence 
    pre_qui_init = session_trials['intervals_0']
    pre_qui_end = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']

    # Quiescence
    qui_init = session_trials['goCueTrigger_times'] - session_trials['quiescencePeriod']
    qui_end = session_trials['goCueTrigger_times']
    
    # ITI
    iti_init = session_trials['feedback_times']
    iti_end_correct = session_trials['intervals_1']
    iti_end_incorrect = session_trials['intervals_1'] - 1
    
    # Reaction time 
    rt_init = session_trials['goCueTrigger_times']
    rt_end = session_trials['firstMovement_times']

    # Movement time 
    move_init = session_trials['firstMovement_times']
    move_end = session_trials['feedback_times']
    

    for t in range(trial_num):
          
        # Pre-quiescence
        use_data.loc[(use_data['Bin'] <= pre_qui_end[t]*multiplier) & 
                     (use_data['Bin'] > pre_qui_init[t]*multiplier), 'label'] = 'Pre-quiescence'

        # Quiescence
        use_data.loc[(use_data['Bin'] <= qui_end[t]*multiplier) &
                     (use_data['Bin'] > qui_init[t]*multiplier), 'label'] = 'Quiescence'
        
        # ITI
        if session_trials['feedbackType'][t] == -1.:
            use_data.loc[(use_data['Bin'] <= iti_end_incorrect[t]*multiplier) & 
                            (use_data['Bin'] > iti_init[t]*multiplier), 'label'] = 'ITI'
        elif session_trials['feedbackType'][t] == 1.:
            use_data.loc[(use_data['Bin'] <= iti_end_correct[t]*multiplier) & 
                            (use_data['Bin'] > iti_init[t]*multiplier), 'label'] = 'ITI'
        # Move
        if session_trials['choice'][t] == -1:
            use_data.loc[(use_data['Bin'] <= move_end[t]*multiplier) & 
                         (use_data['Bin'] > move_init[t]*multiplier), 'label'] = 'Left choice'
        elif session_trials['choice'][t] == 1.:
            use_data.loc[(use_data['Bin'] <= move_end[t]*multiplier) & 
                         (use_data['Bin'] > move_init[t]*multiplier), 'label'] = 'Right choice'
            
        # React        
        if prepro(session_trials)['signed_contrast'][t] < 0:
            use_data.loc[(use_data['Bin'] <= rt_end[t]*multiplier) & 
                         (use_data['Bin'] > rt_init[t]*multiplier), 'label'] = 'Stimulus left'
        elif prepro(session_trials)['signed_contrast'][t] > 0:
            use_data.loc[(use_data['Bin'] <= rt_end[t]*multiplier) & 
                         (use_data['Bin'] > rt_init[t]*multiplier), 'label'] = 'Stimulus right'
    return use_data


def broader_label(df):
    
    df['broader_label'] = df['label']
    # df.loc[df['broader_label']=='Stimulus right', 'broader_label'] = 'Stimulus'
    # df.loc[df['broader_label']=='Stimulus left', 'broader_label'] = 'Stimulus'
    df.loc[df['broader_label']=='Stimulus right', 'broader_label'] = 'Choice'
    df.loc[df['broader_label']=='Stimulus left', 'broader_label'] = 'Choice'
    df.loc[df['broader_label']=='Quiescence', 'broader_label'] = 'Quiescence'
    df.loc[df['broader_label']=='Pre-quiescence', 'broader_label'] = 'Pre-quiescence'
    df.loc[df['broader_label']=='Left choice', 'broader_label'] = 'Choice'
    df.loc[df['broader_label']=='Right choice', 'broader_label'] = 'Choice'
    df.loc[df['broader_label']=='Correct feedback', 'broader_label'] = 'ITI'
    df.loc[df['broader_label']=='Incorrect feedback', 'broader_label'] = 'ITI'
    df.loc[df['broader_label']=='ITI_correct', 'broader_label'] = 'ITI'
    df.loc[df['broader_label']=='ITI_incorrect', 'broader_label'] = 'ITI'
    
    return df


def define_trial_types(states_trial_type, trial_type_agg):
    
    """ Define trial types"""
    states_trial_type['correct_str'] = states_trial_type['correct']
    states_trial_type.loc[states_trial_type['correct_str']==1., 'correct_str'] = 'correct'
    states_trial_type.loc[states_trial_type['correct_str']==0., 'correct_str'] = 'incorrect'
    states_trial_type['contrast_str'] = states_trial_type['contrast'].astype(str)
    states_trial_type['block_str'] = states_trial_type['block'].astype(str)
    states_trial_type['perseverence'] = states_trial_type['wsls'].copy()
    states_trial_type.loc[states_trial_type['wsls'].isin(['wst', 'lst']), 'perseverence']  = 'stay'
    states_trial_type.loc[states_trial_type['wsls'].isin(['wsh', 'lsh']), 'perseverence']  = 'shift'
    states_trial_type['trial_type'] = states_trial_type[trial_type_agg].fillna('unknown').agg(' '.join, axis=1)
    # states_trial_type['trial_type'] = states_trial_type[trial_type_agg].agg(' '.join, axis=1)
    states_trial_type['trial_str'] = states_trial_type['trial_id'].astype(str)
    states_trial_type['sample'] = states_trial_type[['session', 'trial_str']].agg(' '.join, axis=1)
    if 'ballistic' in states_trial_type.columns:
        states_trial_type.loc[states_trial_type['ballistic']==True, 'ballistic'] = 1
        states_trial_type.loc[states_trial_type['ballistic']==False, 'ballistic'] = 0
    return states_trial_type


def rescale_sequence(seq, target_length):
    """
    Rescales a categorical sequence to a fixed target length.
    
    - If `target_length` is smaller than the original length, it takes the mode of each bin.
    - If `target_length` is larger, it repeats values evenly.
    
    Parameters:
        seq (array-like): The original categorical sequence.
        target_length (int): The desired length of the output sequence.
    
    Returns:
        np.ndarray: The transformed sequence with the specified target length.
    """
    original_length = len(seq)

    if original_length == target_length:
        return np.array(seq)  # No change needed

    if target_length < original_length:
        # Compression: Split into bins and take mode of each bin
        bins = np.array_split(seq, target_length)
        # return np.array([mode(b)[0][0] for b in bins])  # Extract mode from result
        return np.array([mode(b)[0] for b in bins])  # Extract mode from result

    else:
        # Stretching: Repeat values to fit new size
        stretched_indices = np.floor(np.linspace(0, original_length - 1, target_length)).astype(int)
        return np.array(seq)[stretched_indices]  # Map stretched indices to original values


def plot_binned_sequence(df_grouped, index, states_to_append, palette):
        title = df_grouped['broader_label'][index]
        fig, axs = plt.subplots(2, 1, sharex=False, sharey=True, figsize=(5, 2))
        axs[0].imshow(np.concatenate([df_grouped['sequence'][index], states_to_append])[None,:],  
                extent=(0, len(np.concatenate([df_grouped['sequence'][index], states_to_append])), 
                        0, 1),
                aspect="auto",
                cmap=palette,
                alpha=0.7) 
        axs[0].set_xlim([0, len(df_grouped['sequence'][index])])

        axs[1].imshow(np.concatenate([df_grouped['binned_sequence'][index], states_to_append])[None,:],  
                extent=(0, len(np.concatenate([df_grouped['binned_sequence'][index], states_to_append])), 
                        0, 1),
                aspect="auto",
                cmap=palette,
                alpha=0.7) 
        axs[1].set_xlim([0, len(df_grouped['binned_sequence'][index])])
        axs[0].set_title(title)
        plt.tight_layout()


def get_metadata(one, sessions):
    metadata = pd.DataFrame(columns=['session', 'lab'], index=range(len(sessions)))
    for s, session in enumerate(sessions):
        session_details = one.get_details(session, full=False)
        metadata['session'][s] = session
        metadata['lab'][s] = session_details['lab']
    return metadata


"""
SCRIPT: LDA pipeline -- shared by LDA_analyses_pipeline_ALLSESSIONS.ipynb and
        segmentation/6_lda_score_sweep.py
========================================================================
These five functions used to be copy-pasted between the notebook and the sweep script,
and had drifted apart in four places (session exclusions, the balanced-subsampling
guard, the trial-feature branch, the chance level). They live here so there is one
definition of each and the two callers cannot disagree again.
"""
import os
import pathlib
import sys

from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def qc_exclusions(timepoint='Proficient', strictness='filtered_out', verbose=True):
    """Sessions to drop, read from the curated QC sheet rather than a hardcoded list.

    `session_filters` reads individuality-paper_data_4Sep26.csv and returns the eids whose
    `Used in paper` column says to drop them, so editing the sheet changes every analysis at
    once and the folders cannot drift apart.

      strictness 'filtered_out'            -> drop Used in paper == 'filtered out'
                 'filtered_out_and_revise' -> drop those PLUS 'need to revise'

    `timepoint` picks which column's exclusions apply; the LDA pipelines read the
    proficient (recording) files, so they want 'Proficient'.
    """
    # session_filters lives in paper-individuality/learning_individuality. Find the
    # paper-individuality root from this file rather than from the caller's cwd, which
    # the notebooks os.chdir away from.
    root = pathlib.Path(__file__).resolve().parent.parent
    for cand in (root / 'learning_individuality', root):
        if (cand / 'session_filters.py').exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))

    # a long-lived kernel caches imported modules, so edits to session_filters -- or the
    # QC sheet moving folders -- stay invisible until a restart. Reload explicitly.
    import importlib
    import session_filters
    importlib.reload(session_filters)

    excl = session_filters.exclusions_by_timepoint(strictness)
    out = sorted(excl[timepoint])
    if verbose:
        print(f'QC sheet: {session_filters.find_csv()}')
        print(f'exclusions ({strictness}): '
              + ', '.join(f'{k} {len(v)}' for k, v in excl.items())
              + f'  ->  {len(out)} {timepoint.lower()} sessions dropped')
    return out


def filter_sequences(all_sequences, prob_sessions, min_sessions=3, verbose=True):
    """Drop QC-failed sessions, then drop mice with fewer than `min_sessions` sessions.

    The mouse filter is what makes the leave-one-session-out LDA answerable: a mouse needs
    at least one training session left after its own session is held out.
    """
    if verbose:
        print(f'{all_sequences["mouse_name"].nunique()} mice in total')
        print(f'{all_sequences["session"].nunique()} sessions in total')

    # drop=True: reset_index would otherwise add a stray 'index' column that then rides
    # along into the pivot and shifts every positional column lookup downstream
    all_sequences = all_sequences.loc[
        ~all_sequences['session'].isin(prob_sessions)].reset_index(drop=True)
    if verbose:
        print(f'{all_sequences["session"].nunique()} sessions after removing bad sessions')

    session_count = (all_sequences[['mouse_name', 'session']].drop_duplicates()
                     .groupby(['mouse_name'])['session'].count().reset_index())
    multi_sess_mice = session_count.loc[session_count['session'] >= min_sessions, 'mouse_name']
    all_sequences = all_sequences.loc[
        all_sequences['mouse_name'].isin(multi_sess_mice)].reset_index(drop=True)
    if verbose:
        print(f'{len(multi_sess_mice)} mice with at least {min_sessions} sessions')
        print(f'{all_sequences["session"].nunique()} remaining sessions')

    assert 'index' not in all_sequences.columns, "ERROR: 'index' column created by reset_index!"
    return all_sequences


def binarize(n_features_per_step, use_sequences, n_paw_states=8):
    """One-hot the syllable sequence: paw state (n_paw_states cols) + whisk + lick, per bin.

    A syllable code is `lick * 2 * n_paw_states + whisk * n_paw_states + paw`, so the three
    factors come back out by arithmetic. Paw state index 1 is dropped at every timestep as
    the reference level, which is why the caller passes `n_features_per_step` (the count
    BEFORE that drop, i.e. n_paw_states + 2).

    Parameters
    ----------
    n_features_per_step : int
        Features per bin before dropping the reference column (n_paw_states + 2).
    use_sequences : np.ndarray, (n_trials, timesteps)
        Encoded integer syllables, NaN where a bin has no syllable.
    n_paw_states : int
        Number of paw clusters in this segmentation.
    """
    n_trials, timesteps = use_sequences.shape
    binarized = np.zeros((n_trials, timesteps * n_features_per_step))

    for t in range(timesteps):
        current_vals = use_sequences[:, t]
        nan_mask = np.isnan(current_vals)
        valid_mask = ~nan_mask
        labels_0idx = current_vals[valid_mask].astype(int)
        start_col = t * n_features_per_step

        if len(labels_0idx) > 0:
            valid_row_idx = np.arange(n_trials)[valid_mask]
            # paw states: cols 0 .. n_paw_states-1
            binarized[valid_row_idx, start_col + labels_0idx % n_paw_states] = 1
            # whisking: col n_paw_states
            binarized[valid_mask, start_col + n_paw_states] = (
                (labels_0idx // n_paw_states) % 2).astype(int)
            # licking: col n_paw_states + 1
            binarized[valid_mask, start_col + n_paw_states + 1] = (
                labels_0idx // (n_paw_states * 2)).astype(int)

        # a bin with no syllable must stay NaN across all its features, not read as zeros
        if np.any(nan_mask):
            binarized[nan_mask, start_col:start_col + n_features_per_step] = np.nan

    # reference level: paw state index 1, dropped at every timestep
    cols_to_delete = [t * n_features_per_step + 1 for t in range(timesteps)]
    return np.delete(binarized, cols_to_delete, axis=1)


def build_design_matrix(filename, n_paw_states=8, prob_sessions=None, min_sessions=3,
                        verbose=True):
    """Load one data-type file and return its per-session-averaged (features, design_df).

    The data type is read off the filename -- 'syllables'/'sequences', 'raw', or 'trial' --
    so the three encodings stay in one place instead of being selected by commenting a
    variable in and out.

    `prob_sessions=None` reads the QC sheet via qc_exclusions(); pass a list to override.
    """
    if prob_sessions is None:
        prob_sessions = qc_exclusions(verbose=verbose)

    all_sequences = pd.read_parquet(filename)
    if 'syllables' in filename or 'sequences' in filename or 'raw' in filename:
        all_sequences['session'] = all_sequences['sample'].str[:36]

    all_sequences = filter_sequences(all_sequences, prob_sessions, min_sessions, verbose)

    if 'syllables' in filename or 'sequences' in filename:
        design_df = all_sequences.pivot(
            index=['mouse_name', 'session', 'sample', 'trial_type'],
            columns=['broader_label'], values='binned_sequence').reset_index().dropna()
        if 'index' in design_df.columns:
            design_df = design_df.drop(columns=['index'])
        design_df = design_df.sort_values(by='session')
        assert len(design_df) > 0, 'ERROR: design_df is empty after filtering!'
        if verbose:
            print(f"design_df: {len(design_df)} rows, {design_df['mouse_name'].nunique()} mice, "
                  f"{design_df['session'].nunique()} sessions")

        epoch_to_analyse = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
        use_sequences = np.vstack(
            design_df[epoch_to_analyse].apply(lambda row: np.hstack(row), axis=1))
        assert len(use_sequences) == len(design_df), (
            f'ERROR: use_sequences length ({len(use_sequences)}) != '
            f'design_df length ({len(design_df)})')

        n_features_per_step = n_paw_states + 2      # paw states + whisk + lick
        use_format = binarize(n_features_per_step, use_sequences, n_paw_states)

    elif 'raw' in filename:
        binned_vars = ['Lick count_binned_sequence', 'whisker_me_binned_sequence',
                       'l_paw_x_vel_binned_sequence', 'l_paw_y_vel_binned_sequence',
                       'r_paw_x_vel_binned_sequence', 'r_paw_y_vel_binned_sequence']
        design_df = all_sequences.pivot(
            index=['mouse_name', 'session', 'sample', 'trial_type'],
            columns=['broader_label'], values=binned_vars).reset_index().dropna()
        if 'index' in design_df.columns:
            design_df = design_df.drop(columns=['index'])
        design_df = design_df.sort_values(by='session')
        epoch_to_analyse = design_df.keys()[4:]
        use_sequences = np.vstack(
            design_df[epoch_to_analyse].apply(lambda row: np.hstack(row), axis=1))
        assert len(use_sequences) == len(design_df), (
            'ERROR: use_sequences length != design_df length')
        use_format = zscore(use_sequences, axis=0, nan_policy='omit')

    elif 'trial' in filename:
        design_df = all_sequences.dropna().copy()      # .copy(): we assign columns below
        design_df['choice'] = design_df['choice'].map({'left': 0, 'right': 1}).astype(float)
        # some trial files store the outcome as a string 'feedback', others as numeric
        # 'correct' -- normalise to one numeric 'feedback'
        if 'feedback' in design_df.columns:
            design_df['feedback'] = design_df['feedback'].map(
                {'incorrect': 0, 'correct': 1}).astype(float)
        else:
            design_df['feedback'] = design_df['correct'].astype(float)

        bias_df = (design_df[design_df['contrast'] == 0]
                   .groupby(['session', 'block'])['choice'].mean()
                   .unstack(level='block'))
        bias_df['bias'] = bias_df[0.8] - bias_df[0.2]

        agg = {'trial_id': 'count', 'reaction': 'median', 'elongation': 'median',
               'feedback': 'mean', 'choice': 'mean'}
        features = ['trial_id', 'feedback', 'choice', 'log_reaction', 'log_elongation', 'bias']
        # p_state1 (GLM-HMM engagement) is only in some trial files, e.g. session_trial_meta_*
        if 'p_state1' in design_df.columns:
            agg['p_state1'] = 'mean'
            features.insert(3, 'p_state1')

        merged = (design_df.groupby(['session', 'mouse_name']).agg(agg)
                  .merge(bias_df['bias'], on='session', how='left'))
        # log1p, not log: session-median reaction time can be slightly NEGATIVE (first
        # movement before the go cue), and np.log of a negative is NaN, which the dropna
        # below then removed -- silently discarding the session rather than deciding to.
        # Measured on session_trial_meta_19-08-2026: 1 of 249 sessions, median -0.0052 s.
        # log1p(x) = log(1 + x) is finite and monotone for every x > -1, which covers it.
        for _src, _dst in (('reaction', 'log_reaction'), ('elongation', 'log_elongation')):
            if (merged[_src] <= -1).any():
                raise ValueError(
                    f'{_src} has values <= -1, outside log1p\'s domain: '
                    f'min = {merged[_src].min():.4g}')
            merged[_dst] = np.log1p(merged[_src])

        clean_df = merged[features].dropna()
        use_format = zscore(clean_df.to_numpy(), axis=0)
        # drop_duplicates BEFORE the merge: without it a session with many trial rows
        # multiplies out and the design_df no longer has one row per session
        design_df = (clean_df.reset_index()
                     .merge(design_df[['mouse_name', 'session']].drop_duplicates(),
                            on='session')
                     .drop_duplicates())
    else:
        raise ValueError(f"cannot tell the data type from the filename: {filename}")

    """ SESSION AVERAGE """
    session_mouse_mapping = (design_df[['session', 'mouse_name']].drop_duplicates()
                             .set_index('session')['mouse_name'].to_dict())
    assert len(session_mouse_mapping) == len(
        design_df[['session', 'mouse_name']].drop_duplicates()), \
        'ERROR: A session maps to multiple mice!'

    session_syllables = pd.DataFrame(use_format)
    session_syllables['session'] = design_df['session'].values
    session_syllables = session_syllables.groupby('session', sort=False)[
        np.arange(0, np.shape(use_format)[1], 1)].mean()

    if verbose:
        print(f'session aggregation complete: {len(session_syllables)} sessions, '
              f'{session_syllables.shape[1]} features')
    return session_syllables, design_df


def dim_red(all_features):
    """PCA on the session x feature matrix. Returns the full score matrix; slice columns
    to pick a dimensionality.

    NOTE the features are NOT standardised before PCA -- deliberately, and identically in
    both callers. Change it here if you change it at all, so the two cannot diverge.
    """
    n_components = np.min(np.shape(np.array(all_features)))
    pca = PCA(n_components)
    return pca.fit_transform(np.array(all_features))


def bootstrap_over_mice(per_session_hit, mouse_of_session, n_boot=2000, seed=0):
    """95% CI for a mean accuracy, resampling MICE rather than sessions.

    `per_session_hit` is (n_sessions,) or (n_sessions, n_variants) -- each entry the
    probability that session was classified correctly, averaged over repeats first.

    WHY MICE. Sessions from one animal are not independent draws: 3 to 16 of them share
    an animal, a rig and a tracking pipeline, so resampling SESSIONS would treat
    correlated observations as independent and return an interval that is too narrow. It
    would also answer the wrong question -- "what if I had more sessions from these mice"
    rather than "what if I had a different sample of mice", which is the population every
    claim here is about. Resampling animals and carrying all of a drawn animal's sessions
    with it respects the clustering and matches the inferential unit.

    Not to be confused with the std across training subsamples, which measures sensitivity
    to the random draw and shrinks as the balanced cap rises (a mouse with exactly
    n_per_mouse sessions has no choice left to make).
    """
    per_session_hit = np.asarray(per_session_hit, dtype=float)
    flat = per_session_hit.ndim == 1
    H = per_session_hit[:, None] if flat else per_session_hit
    unique_mice = np.unique(mouse_of_session)
    sessions_of_mouse = {m: np.flatnonzero(mouse_of_session == m) for m in unique_mice}
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, H.shape[1]))
    for b in range(n_boot):
        drawn = rng.choice(unique_mice, len(unique_mice), replace=True)
        boot[b] = H[np.concatenate([sessions_of_mouse[m] for m in drawn])].mean(axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return (float(lo[0]), float(hi[0])) if flat else (lo, hi)


def run_lda(design_df, session_syllables, n_component, norm_pop, n_per_mouse=3,
            n_repeats=10, seed=0, n_boot=2000, return_ci=False, verbose=True):
    """Leave-one-session-out mouse discriminability, with a shuffled-label control.

    THE BALANCED SUBSAMPLING CAP IS A MAXIMUM, NOT A MINIMUM REQUIREMENT.
    The rule this replaces was

        if len(m_idx) >= cap: rng.choice(m_idx, cap)

    which dropped a mouse with fewer sessions than the cap out of the TRAINING set while
    still TESTING it -- against a classifier that had never seen that class, so those folds
    were guaranteed misses. Because the guard is evaluated AFTER leave-one-out, a mouse fell
    below it precisely in the folds where it was the test item. With 20 of 58 mice having
    exactly 3 sessions, n_per_mouse=4 turned ~22% of folds into automatic errors and the
    score fell for that reason alone: measured 0.770 / 0.651 / 0.480 at n_per_mouse 3/4/5
    under the old rule, versus 0.770 / 0.814 / 0.848 under `take = min(cap, len(m_idx))` --
    monotone, as it should be. At n_per_mouse=3 the two rules agree exactly, which is why
    this went unnoticed.

    The assertion makes the failure loud rather than silent if it ever recurs.

    UNCERTAINTY: `return_ci=True` adds a 95% interval from bootstrapping MICE, a different
    quantity from the std of `true_scores` and the one that belongs on a figure.
      * std over repeats  = how much the score moves when the TRAINING DRAW changes. A
        robustness diagnostic, not uncertainty -- and it shrinks artificially as the cap
        rises, because a mouse with exactly n_per_mouse sessions has no choice left to make
        (20 of 58 mice here have exactly 3).
      * bootstrap over SESSIONS would treat a mouse's 3-16 sessions as independent draws.
        They are not -- same animal, rig, tracking -- so it understates the spread.
      * bootstrap over MICE resamples animals, carrying all of a drawn animal's sessions.
        That respects the clustering and matches the population the claim is about: other
        mice, not more sessions from these ones. Same construction as lda_sweep_timepoints.
    """
    mapping = pd.DataFrame(np.array(design_df[['mouse_name', 'session']].drop_duplicates()),
                           columns=['mouse_name', 'session'])
    df_with_sessions = session_syllables.reset_index()
    mouse_names = df_with_sessions.merge(mapping, on=['session'])['mouse_name']

    lda_components = np.min([n_component, len(mouse_names.unique()) - 1])
    # The cap IS n_per_mouse. The old `n_per_mouse - 1` dated from the broken guard,
    # where 2 was the largest cap every mouse could always meet (filter_sequences keeps
    # >= 3 sessions, leave-one-out removes at most one) so nobody was ever dropped.
    # `take = min(...)` handles short mice directly, so the -1 is redundant -- and it had
    # left this function disagreeing with the notebook cells that cap at n_per_mouse.
    cap = n_per_mouse

    X = np.array(norm_pop).copy()
    y = pd.factorize(mouse_names)[0]
    n_samples = X.shape[0]
    # TWO independent streams, not one. With a single generator the label shuffle
    # consumed draws between folds, so the balanced subsamples here differed from the
    # ones lda_dimension_sweep drew with the same seed -- and the sweep's full-rank point
    # then sat ~0.003 ABOVE this score for no reason a reader could see. Separate streams
    # mean the same seed gives the same subsample in both, so the curve lands on the line.
    sub_rng = np.random.default_rng(seed)
    shuf_rng = np.random.default_rng(seed + 1_000_000)

    true_scores_all, shuffle_scores_all = [], []
    # per-repeat, per-session hit/miss, kept so the interval can resample MICE
    correct_per_session = np.zeros((n_repeats, n_samples))
    for rep in range(n_repeats):
        scores_true, scores_shuff = [], []

        for test_idx in range(n_samples):
            X_test, y_test = X[test_idx:test_idx + 1], y[test_idx:test_idx + 1]
            train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
            X_train_full, y_train_full = X[train_idx], y[train_idx]

            # --- balanced subsampling for training ---
            # every mouse stays a class; sparse ones contribute what they have
            balanced_idx = []
            for m in np.unique(y_train_full):
                m_idx = np.where(y_train_full == m)[0]
                take = min(cap, len(m_idx))
                if take > 0:
                    balanced_idx.extend(sub_rng.choice(m_idx, take, replace=False))
            balanced_idx = np.array(balanced_idx)
            assert y_test[0] in y_train_full[balanced_idx], (
                'the held-out mouse is not a training class -- this fold is unanswerable')
            X_train, y_train = X_train_full[balanced_idx], y_train_full[balanced_idx]

            # --- true labels ---
            lda = LinearDiscriminantAnalysis(
                priors=np.ones(len(np.unique(y_train))) / len(np.unique(y_train)),
                n_components=lda_components)
            lda.fit(X_train, y_train)
            hit = lda.score(X_test, y_test)
            scores_true.append(hit)
            correct_per_session[rep, test_idx] = hit

            # --- shuffled labels ---
            y_train_shuff = y_train.copy()
            shuf_rng.shuffle(y_train_shuff)
            lda_shuff = LinearDiscriminantAnalysis(
                priors=np.ones(len(np.unique(y_train_shuff))) / len(np.unique(y_train_shuff)),
                n_components=lda_components)
            lda_shuff.fit(X_train, y_train_shuff)
            scores_shuff.append(lda_shuff.score(X_test, y_test))

        true_scores_all.append(np.mean(scores_true))
        shuffle_scores_all.append(np.mean(scores_shuff))

    # --- 95% CI by resampling MICE (see the docstring) ---
    # average over repeats first, so each session carries its probability of being
    # classified correctly; then resample animals, taking all of a drawn animal's sessions
    mouse_of_session = mouse_names.to_numpy()
    unique_mice = np.unique(mouse_of_session)
    ci_low, ci_high = bootstrap_over_mice(correct_per_session.mean(axis=0),
                                          mouse_of_session, n_boot=n_boot, seed=seed)

    if verbose:
        print(f'True labels     mean {np.mean(true_scores_all):.4f}  '
              f'[95% CI over {len(unique_mice)} mice: {ci_low:.4f}, {ci_high:.4f}]')
        print(f'   (spread across training subsamples, NOT a CI: '
              f'+/-{np.std(true_scores_all):.4f})')
        print(f'Shuffled labels mean +/- std: {np.mean(shuffle_scores_all):.4f} '
              f'{np.std(shuffle_scores_all):.4f}')
    if return_ci:
        return (np.array(true_scores_all), np.array(shuffle_scores_all),
                float(ci_low), float(ci_high))
    return np.array(true_scores_all), np.array(shuffle_scores_all)


def weighted_lda_fit(X, y, n_components=None, shrinkage=0.0, weights='per_class'):
    """LDA whose scatter matrices give every CLASS equal weight, using all the data.

    WHY THIS EXISTS. sklearn's LinearDiscriminantAnalysis (solver='svd') builds

        xbar_ = priors_ @ means_                                  <- priors-weighted
        S_B   from sqrt(n_samples * priors_) * (means_ - xbar_)   <- priors-weighted
        S_W   from concat(Xg - means_[g]) for every sample        <- RAW, one row per sample

    so passing uniform `priors` (as the pipelines do) already balances the grand mean and
    the between-class scatter -- but NOT the within-class scatter. With 3 to 16 sessions per
    mouse, the whitening metric is therefore set mostly by the session-rich animals, and the
    discriminant directions are "maximise balanced between-mouse separation after whitening
    by the rich mice's within-mouse covariance".

    Subsampling to a fixed cap fixes that by throwing data away, and it reintroduces
    draw-dependence: the leading eigenvalues here are near-degenerate (0.267 vs 0.253), so
    each balanced draw picks a different rotation inside the same subspace and two draws can
    give LD1 axes correlating anywhere from 0.05 to 0.85. Weighting instead keeps every
    session AND is deterministic.

    WHAT IT DOES NOT FIX. Weighting equalises each class's INFLUENCE, not the PRECISION of
    its mean: a 3-session mouse still has a noisier -- and so more outlying -- class mean
    than a 16-session one, which inflates apparent separation. Only more sessions, or
    `shrinkage` toward the grand mean, addresses that.

    Parameters
    ----------
    X : (n_samples, n_features)
    y : (n_samples,) class labels
    n_components : int or None
        Discriminant directions to keep. Default min(n_classes - 1, n_features).
    shrinkage : float in [0, 1]
        Ledoit-Wolf-style shrinkage of S_W toward a scaled identity. 0 disables it. Useful
        when n_features approaches n_samples, where S_W is near-singular.
    weights : 'per_class' | 'per_sample'
        'per_class' gives every class total weight 1 (the point of this function);
        'per_sample' reproduces sklearn's unbalanced S_W, for comparison.

    Returns
    -------
    dict with 'scalings' (n_features, n_components) projection, 'means' (class means),
    'grand_mean', 'classes', 'eigenvalues' (separation carried by each direction) and
    'explained_variance_ratio'.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    n_classes, n_features = len(classes), X.shape[1]
    if n_components is None:
        n_components = min(n_classes - 1, n_features)

    # per-sample weights: every class sums to 1, so a 16-session mouse and a 3-session
    # mouse count the same no matter how unequal their session counts are
    w = np.ones(len(y), dtype=float)
    if weights == 'per_class':
        for c in classes:
            sel = y == c
            w[sel] = 1.0 / sel.sum()
    elif weights != 'per_sample':
        raise ValueError("weights must be 'per_class' or 'per_sample'")
    w = w / w.sum()

    means = np.vstack([np.average(X[y == c], axis=0, weights=w[y == c]) for c in classes])
    grand_mean = np.average(means, axis=0)      # unweighted: classes already count equally

    # within-class scatter, weighted
    S_W = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        sel = y == c
        Xc = X[sel] - means[i]
        S_W += (Xc * w[sel][:, None]).T @ Xc
    if shrinkage > 0:
        S_W = (1 - shrinkage) * S_W + shrinkage * np.trace(S_W) / n_features * np.eye(n_features)

    # between-class scatter, every class weight 1 / n_classes
    M = means - grand_mean
    S_B = (M.T @ M) / n_classes

    # solve S_W^-1 S_B by whitening S_W first -- symmetric eigendecomposition, so the
    # result is real and orthogonal in the whitened space rather than whatever a general
    # eig() returns
    evals_w, evecs_w = np.linalg.eigh(S_W)
    keep = evals_w > max(evals_w.max(), 0) * 1e-10
    if not keep.any():
        raise np.linalg.LinAlgError('within-class scatter is singular; raise `shrinkage`')
    whiten = evecs_w[:, keep] / np.sqrt(evals_w[keep])
    evals, evecs = np.linalg.eigh(whiten.T @ S_B @ whiten)

    order = np.argsort(evals)[::-1][:n_components]
    scalings = whiten @ evecs[:, order]
    evals = evals[order]

    # sign convention: largest-magnitude loading positive, so refits are comparable
    flip = np.sign(scalings[np.argmax(np.abs(scalings), axis=0), np.arange(scalings.shape[1])])
    flip[flip == 0] = 1.0
    scalings = scalings * flip

    total = np.sum(np.maximum(evals, 0))
    return {'scalings': scalings, 'means': means, 'grand_mean': grand_mean,
            'classes': classes, 'eigenvalues': evals,
            'explained_variance_ratio': evals / total if total > 0 else evals * np.nan}


def weighted_lda_transform(fit, X):
    """Project samples onto the discriminant axes from weighted_lda_fit."""
    return (np.asarray(X, dtype=float) - fit['grand_mean']) @ fit['scalings']


def weighted_lda_predict(fit, X):
    """Nearest class mean in the discriminant space -- the equal-prior decision rule.

    Equivalent to sklearn's LDA with uniform priors, but on the weighted subspace.
    """
    Z = weighted_lda_transform(fit, X)
    Zc = (fit['means'] - fit['grand_mean']) @ fit['scalings']
    d = ((Z[:, None, :] - Zc[None, :, :]) ** 2).sum(axis=2)
    return fit['classes'][np.argmin(d, axis=1)]


def run_weighted_lda(design_df, session_syllables, n_component, norm_pop, n_repeats=1,
                     shrinkage=0.0, seed=0, verbose=True):
    """Leave-one-session-out score for weighted_lda_fit, with a shuffled-label control.

    Same protocol as run_lda so the two are directly comparable, with one difference that is
    the whole point: there is NO balanced subsampling and therefore no `n_per_mouse`. Every
    training session is used and the weighting does the balancing, so the result is
    deterministic and `n_repeats` only matters for the label shuffle.
    """
    mapping = pd.DataFrame(np.array(design_df[['mouse_name', 'session']].drop_duplicates()),
                           columns=['mouse_name', 'session'])
    df_with_sessions = session_syllables.reset_index()
    mouse_names = df_with_sessions.merge(mapping, on=['session'])['mouse_name']

    X = np.array(norm_pop).copy()
    y = pd.factorize(mouse_names)[0]
    n_samples = X.shape[0]
    n_comp = min(n_component, len(np.unique(y)) - 1)
    rng = np.random.default_rng(seed)

    true_scores_all, shuffle_scores_all = [], []
    for _ in range(n_repeats):
        scores_true, scores_shuff = [], []
        for test_idx in range(n_samples):
            X_test, y_test = X[test_idx:test_idx + 1], y[test_idx:test_idx + 1]
            train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
            X_train, y_train = X[train_idx], y[train_idx]

            fit = weighted_lda_fit(X_train, y_train, n_components=n_comp, shrinkage=shrinkage)
            scores_true.append(float(weighted_lda_predict(fit, X_test)[0] == y_test[0]))

            y_shuff = y_train.copy()
            rng.shuffle(y_shuff)
            fit_s = weighted_lda_fit(X_train, y_shuff, n_components=n_comp, shrinkage=shrinkage)
            scores_shuff.append(float(weighted_lda_predict(fit_s, X_test)[0] == y_test[0]))

        true_scores_all.append(np.mean(scores_true))
        shuffle_scores_all.append(np.mean(scores_shuff))

    if verbose:
        print(f'True labels     mean +/- std: {np.mean(true_scores_all):.4f} '
              f'{np.std(true_scores_all):.4f}')
        print(f'Shuffled labels mean +/- std: {np.mean(shuffle_scores_all):.4f} '
              f'{np.std(shuffle_scores_all):.4f}')
    return np.array(true_scores_all), np.array(shuffle_scores_all)


def lda_dimension_sweep(design_df, session_syllables, n_component, norm_pop, n_per_mouse=3,
                        n_repeats=10, seed=0, dims=None, n_boot=2000, return_ci=False,
                        verbose=True):
    """Mouse discriminability as a function of how many LDA DIMENSIONS are used.

    Distinct from the PCA sweep in segmentation/6_lda_score_sweep.py: there the x axis is
    how many PCA components go INTO the LDA; here the LDA input is fixed and the x axis is
    how many discriminant axes come OUT of it. It answers "is the mouse identity carried by
    LD1, or spread over many axes?", which matters because the paper reads LD1 as a single
    interpretable individuality axis.

    Note `LinearDiscriminantAnalysis(n_components=k)` cannot be used for this: n_components
    only affects `transform`, never `predict`/`score`. So the classification is done
    explicitly as nearest class mean in the first k discriminant dimensions, which with
    equal priors is exactly sklearn's rule -- at full rank this reproduces `lda.score` to
    the digit, so the curve lands on the number you already know.

    Protocol is identical to run_lda (same leave-one-session-out, same balanced-subsampling
    cap, same shuffled control is omitted here), so the two are directly comparable.

    Returns (dims, mean_scores, std_scores), each of length len(dims), or
    (dims, mean_scores, std_scores, ci_low, ci_high) with `return_ci=True`. The CI comes
    from bootstrap_over_mice and is the band to plot; std_scores is the across-subsample
    spread, which is a robustness diagnostic and not uncertainty -- see that function.
    """
    mapping = pd.DataFrame(np.array(design_df[['mouse_name', 'session']].drop_duplicates()),
                           columns=['mouse_name', 'session'])
    df_with_sessions = session_syllables.reset_index()
    mouse_names = df_with_sessions.merge(mapping, on=['session'])['mouse_name']

    X = np.array(norm_pop).copy()
    y = pd.factorize(mouse_names)[0]
    n_samples = X.shape[0]
    cap = n_per_mouse        # same definition as run_lda -- see the note there
    max_dim = min(n_component, len(np.unique(y)) - 1, X.shape[1])
    if dims is None:
        dims = np.arange(1, max_dim + 1)
    dims = np.asarray([d for d in dims if d <= max_dim])
    sub_rng = np.random.default_rng(seed)      # same stream as run_lda's subsample

    per_repeat = np.zeros((n_repeats, len(dims)))
    # per-repeat, per-session, per-dimension hits, so the CI can resample MICE at every
    # dimension rather than only at the full-rank point
    correct_per_session = np.zeros((n_repeats, n_samples, len(dims)))
    for r in range(n_repeats):
        hits = np.zeros((n_samples, len(dims)))
        for test_idx in range(n_samples):
            X_test, y_test = X[test_idx:test_idx + 1], y[test_idx:test_idx + 1]
            train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
            X_train_full, y_train_full = X[train_idx], y[train_idx]

            balanced_idx = []
            for m in np.unique(y_train_full):
                m_idx = np.where(y_train_full == m)[0]
                take = min(cap, len(m_idx))
                if take > 0:
                    balanced_idx.extend(sub_rng.choice(m_idx, take, replace=False))
            balanced_idx = np.array(balanced_idx)
            X_train, y_train = X_train_full[balanced_idx], y_train_full[balanced_idx]

            lda = LinearDiscriminantAnalysis(
                priors=np.ones(len(np.unique(y_train))) / len(np.unique(y_train)))
            lda.fit(X_train, y_train)

            Z = lda.transform(X_test)[0]                 # (n_ld,)
            Zc = lda.transform(lda.means_)               # (n_classes, n_ld)
            # squared distance accumulated dimension by dimension, so every k is scored
            # from one fit instead of refitting per k
            cum = np.cumsum((Zc - Z) ** 2, axis=1)
            for j, d in enumerate(dims):
                k = min(d, cum.shape[1]) - 1
                hits[test_idx, j] = lda.classes_[np.argmin(cum[:, k])] == y_test[0]
        correct_per_session[r] = hits
        per_repeat[r] = hits.mean(axis=0)
        if verbose:
            print(f'  repeat {r + 1}/{n_repeats}: max {per_repeat[r].max():.3f} '
                  f'at {dims[per_repeat[r].argmax()]} LDA dims')

    if return_ci:
        ci_low, ci_high = bootstrap_over_mice(correct_per_session.mean(axis=0),
                                              mouse_names.to_numpy(), n_boot=n_boot,
                                              seed=seed)
        return dims, per_repeat.mean(axis=0), per_repeat.std(axis=0), ci_low, ci_high
    return dims, per_repeat.mean(axis=0), per_repeat.std(axis=0)


def leave_mice_out_verification(design_df, session_syllables, n_component, norm_pop,
                                n_folds=5, seed=0, weighted=False, shrinkage=0.0,
                                verbose=True):
    """Does the embedding separate mice it has NEVER SEEN?

    Leave-one-session-out asks "which of these 58 mice is this?" -- the classifier has seen
    every mouse, so a high score is partly memorisation of these particular animals. Holding
    out whole MICE removes that, but then identification is unanswerable: you cannot name a
    class that was not in training. The fix is the standard move from face and speaker
    recognition -- switch from IDENTIFICATION to VERIFICATION:

        given two held-out sessions, are they the same mouse?

    scored by distance in the embedding. That question is well posed for unseen identities,
    and its chance level is exactly 0.5 (AUC), which makes it directly interpretable.

    WHOLE MICE ARE HELD OUT IN GROUPS, not one at a time: with a single held-out mouse every
    pair is a same-mouse pair and there is nothing to discriminate against. `n_folds` splits
    the mice, so each fold yields both same- and different-mouse pairs.

    THE BASELINE IS THE POINT. The same AUC is computed in the input space (`norm_pop`, i.e.
    the PCA scores) with no LDA at all. If the LDA does not beat it on held-out mice, the
    discriminant axes are specific to the training animals and carry no general "individuality
    geometry" -- which is the thing the paper claims. Comparing to 0.5 alone cannot show that;
    comparing to the baseline can.

    Returns a DataFrame with one row per fold: AUC and silhouette for the LDA embedding and
    for the input-space baseline, plus the fold's mouse and session counts.
    """
    from sklearn.metrics import roc_auc_score, silhouette_score

    mapping = pd.DataFrame(np.array(design_df[['mouse_name', 'session']].drop_duplicates()),
                           columns=['mouse_name', 'session'])
    df_with_sessions = session_syllables.reset_index()
    mouse_names = df_with_sessions.merge(mapping, on=['session'])['mouse_name']

    X = np.array(norm_pop).copy()
    y = pd.factorize(mouse_names)[0]
    mice = np.unique(y)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(mice), n_folds)

    def pair_auc(Z, labels):
        """AUC of -distance predicting 'same mouse', over every pair of held-out sessions."""
        d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(Z), k=1)
        same = (labels[:, None] == labels[None, :])[iu].astype(int)
        if same.sum() == 0 or same.sum() == len(same):
            return np.nan
        return roc_auc_score(same, -d[iu])

    rows = []
    for f, held in enumerate(folds):
        te = np.isin(y, held)
        tr = ~te
        if len(np.unique(y[te])) < 2:
            continue        # need at least two held-out mice to have different-mouse pairs

        n_comp = min(n_component, len(np.unique(y[tr])) - 1, X.shape[1])
        if weighted:
            fit = weighted_lda_fit(X[tr], y[tr], n_components=n_comp, shrinkage=shrinkage)
            Z_te = weighted_lda_transform(fit, X[te])
        else:
            ncls = len(np.unique(y[tr]))
            lda = LinearDiscriminantAnalysis(priors=np.ones(ncls) / ncls,
                                             n_components=n_comp)
            lda.fit(X[tr], y[tr])
            Z_te = lda.transform(X[te])[:, :n_comp]

        lab = y[te]
        rows.append({
            'fold': f, 'n_mice_held_out': len(np.unique(lab)), 'n_sessions': int(te.sum()),
            'auc_lda': pair_auc(Z_te, lab),
            'auc_input': pair_auc(X[te], lab),
            'silhouette_lda': silhouette_score(Z_te, lab),
            'silhouette_input': silhouette_score(X[te], lab),
        })

    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(f'held-out mice per fold: {out["n_mice_held_out"].tolist()}')
        print(f'  verification AUC   LDA {out["auc_lda"].mean():.3f} +/- {out["auc_lda"].std():.3f}'
              f'   |  input space {out["auc_input"].mean():.3f} +/- {out["auc_input"].std():.3f}'
              f'   (chance 0.500)')
        print(f'  silhouette         LDA {out["silhouette_lda"].mean():.3f}'
              f'   |  input space {out["silhouette_input"].mean():.3f}'
              f'   (0 = no structure)')
    return out


# =============================================================================
# LAB VARIANCE
# =============================================================================
# Labs differ in rig, camera placement, handling and water schedule, and those
# differences land in the behavioural features. They are NESTED INSIDE MOUSE here --
# every mouse belongs to exactly one lab -- so any lab correction necessarily removes
# part of what the LDA is asked to classify. Read remove_lab_variance's docstring before
# using it: the amount removed is 1/(mice in that lab), which is not the same for every
# lab, and that alone changes the geometry.

def lab_labels(sessions, mouse_names=None, csv_path=None, verbose=True):
    """Session -> lab, parsed OFFLINE from the QC sheet's `rig_name` column.

    No ONE call: the sheet already carries a rig per eid, and rig names have the form
    `_iblrig_<lab>_<ephys|behavior>_<n>`. Two traps, both handled here:
      * `angelaki` and `angelakilab` are the same lab under two rig spellings;
      * `churchlandlab` (CSHL) and `churchlandlab_ucla` are DIFFERENT labs, and
        `hofer` and `mrsicflogel` are different labs at the same institute -- so the
        names are not collapsed by institute.

    A session missing from the sheet is filled from the mouse's other sessions when
    `mouse_names` is given; no mouse in this dataset spans two labs, so that is safe.
    """
    import re
    import pandas as pd
    import numpy as np
    from session_filters import find_csv

    sheet = pd.read_csv(csv_path or find_csv(), header=1)

    def _lab_of(rig):
        if not isinstance(rig, str):
            return np.nan
        m = re.match(r'_iblrig_(.+?)_(ephys|behavior)', rig)
        if not m:
            return np.nan
        return {'angelaki': 'angelakilab'}.get(m.group(1), m.group(1))

    lut = sheet.assign(lab=sheet['rig_name'].map(_lab_of)).drop_duplicates('eid').set_index('eid')['lab']
    labs = pd.Series(list(sessions), index=list(sessions)).map(lut)

    if mouse_names is not None:
        mouse_names = pd.Series(list(mouse_names), index=labs.index)
        by_mouse = labs.groupby(mouse_names).agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        n_filled = int(labs.isna().sum())
        labs = labs.fillna(mouse_names.map(by_mouse))
        if verbose and n_filled:
            print(f'  {n_filled} session(s) had no rig in the sheet; lab taken from the mouse')
        spanning = labs.groupby(mouse_names).nunique()
        assert (spanning <= 1).all(), \
            f'mice mapped to more than one lab: {list(spanning[spanning > 1].index)}'

    if verbose:
        n_missing = int(labs.isna().sum())
        print(f'  labs: {labs.nunique()} for {len(labs)} sessions'
              + (f' ({n_missing} still unknown)' if n_missing else ''))
    return labs


def lab_variance_explained(features, labs):
    """Mean eta^2 -- the share of each feature's variance that sits BETWEEN labs,
    averaged over features. A blunt but honest size-of-the-problem number: it says how
    much of the feature variance a lab label alone accounts for, with no claim that the
    cause is the lab rather than the mice in it."""
    import numpy as np
    import pandas as pd
    X = np.asarray(features, float)
    labs = pd.Series(list(labs)).to_numpy()
    ok = pd.notna(labs)
    X, labs = X[ok], labs[ok]
    grand = X.mean(axis=0)
    ss_tot = ((X - grand) ** 2).sum(axis=0)
    ss_bet = np.zeros(X.shape[1])
    for l in np.unique(labs):
        m = labs == l
        ss_bet += m.sum() * (X[m].mean(axis=0) - grand) ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        eta2 = np.where(ss_tot > 0, ss_bet / ss_tot, np.nan)
    return float(np.nanmean(eta2))


def remove_lab_variance(features, labs, mode='center', mouse_names=None,
                        min_mice_per_lab=2, verbose=True):
    """Remove the lab component from a (sessions x features) matrix.

    mode
      'none'    return the features unchanged.
      'center'  subtract each lab's mean feature vector (lab-mean subtraction).
      'zscore'  subtract the lab mean and divide by the lab's SD across its sessions
                (within-lab z-scoring). Also removes per-lab SCALE, which 'center' leaves
                alone -- use it when labs differ in the spread of a feature and not only
                in its level. The SD is over SESSIONS, so a lab with few sessions gets a
                noisy divisor; a feature with no spread inside a lab is left centred
                rather than divided by ~0.

    The lab mean is the MEAN OF THE LAB'S PER-MOUSE MEANS, not the mean over its sessions.
    Session counts inside a lab run 3 to 16, so a session-weighted lab mean is dominated by
    its busiest animal and centring would then subtract mostly that one mouse from all of
    its lab-mates.

    WHAT THIS COSTS, AND IT IS NOT OPTIONAL. Lab is nested inside mouse -- every mouse
    belongs to exactly one lab -- so after centring, the corrected mouse means within a lab
    sum to zero. The mouse-identity signal therefore loses one dimension per lab (10 of the
    57 discriminant dimensions in this dataset), and it loses proportionally more in small
    labs: a 3-mouse lab gives up 1 of its 2 between-mouse dimensions, a 9-mouse lab 1 of 8.
    A lab correction is a statement that between-lab differences are nuisance; here that
    statement cannot be made without also discarding part of what identifies the mice.

    Labs with fewer than `min_mice_per_lab` mice are LEFT UNCORRECTED and named: a one-mouse
    lab's mean IS that mouse's mean, and centring it would zero the animal out entirely.

    (A leave-one-mouse-out lab mean was tried and dropped. Within a lab it returns exactly
    n/(n-1) times what 'center' returns, so it is plain centring with a per-lab scale factor
    that inflates the smallest labs most -- it recovers nothing that centring removes.)
    """
    import numpy as np
    import pandas as pd

    if mode in (None, 'none'):
        return features

    is_df = isinstance(features, pd.DataFrame)
    X = np.asarray(features, float).copy()
    labs = pd.Series(list(labs)).to_numpy(dtype=object)
    # positional arrays, so row indices line up with X whatever index the caller's Series
    # carried -- session_syllables is indexed by session, not by position
    mice = None if mouse_names is None else pd.Series(list(mouse_names)).to_numpy(dtype=object)

    skipped, corrected = [], []
    for lab in pd.unique(labs[pd.notna(labs)]):
        rows = np.where(labs == lab)[0]
        n_mice = len(set(mice[rows])) if mice is not None else None
        if n_mice is not None and n_mice < min_mice_per_lab:
            skipped.append(f'{lab} ({n_mice} mouse)')
            continue
        corrected.append(lab)
        if mice is not None:
            lab_mice = mice[rows]
            mu = np.mean([X[rows[lab_mice == m]].mean(axis=0) for m in set(lab_mice)], axis=0)
        else:
            mu = X[rows].mean(axis=0)
        if mode == 'center':
            X[rows] -= mu
        elif mode == 'zscore':
            sd = X[rows].std(axis=0)
            X[rows] = (X[rows] - mu) / np.where(sd > 0, sd, 1.0)
        else:
            raise ValueError(f'unknown mode {mode!r}')

    unknown = int(pd.isna(labs).sum())
    if verbose:
        print(f'  lab correction {mode!r}: {len(corrected)} labs corrected'
              + (f', left alone: {", ".join(skipped)}' if skipped else '')
              + (f', {unknown} sessions with unknown lab left alone' if unknown else ''))
    return pd.DataFrame(X, index=features.index, columns=features.columns) if is_df else X


def rank_inverse_normal(features, groups=None, offset='blom', min_per_group=8,
                        verbose=True):
    """Rank-based inverse-normal ('probit') transform of a sessions x features matrix.

    Each FEATURE is replaced by the normal scores of its ranks, so its marginal becomes
    standard normal by construction. Ties get the same value. This is what Forkosh et al.
    2019 call quantile normalisation:

        "We quantile-normalized the data to have a normal distribution by computing the
         quantile of each sample and then computing the inverse of the normal cumulative
         distribution function (also known as the 'probit' function). If two or more
         samples were identical prior to the normalization they were all assigned the
         same value."

    WHY IT IS NOT JUST ANOTHER z-SCORE, AND WHY THAT MATTERS HERE. The LDA objective is
    invariant under any invertible AFFINE map of the features, so `StandardScaler` -- on the
    raw features or on the PCA scores -- cannot change the discriminants, the eigenvalues or
    the LOO score by anything beyond numerical conditioning. This transform is monotone but
    NONLINEAR, so it does change them. It is the only feature-level normalisation in this
    file that the LDA can actually see (the other one being remove_lab_variance, which is
    affine per lab rather than globally).

    WHAT IT BUYS. The features are session means of binary indicators, i.e. proportions in
    [0, 1]. A syllable that is rare at a given timebin sits near 0 in most sessions with a
    handful well above, so its marginal is strongly skewed, and because the PCA runs on raw
    variance (see dim_red) one aberrant session can plant a component that the LDA then
    rides. Replacing values by normal scores bounds how far any single session can sit from
    its neighbours, and makes the marginals match the Gaussian that LDA is optimal under.

    WHAT IT COSTS. Two things, both real:

      1. It forces EVERY feature to unit variance, which reverses dim_red's deliberate
         choice not to standardise before PCA. For a proportion, Var ~ p(1-p)/n, so the
         un-normalised PCA is implicitly downweighting rare syllables; afterwards a rare
         syllable's ordering counts as much as a common one's. Re-read the scree plot and
         `min_components` after turning this on -- the spectrum WILL move.
      2. Only the ordering survives. Two sessions three units apart and two sessions a
         thousand units apart become equally far apart if no other session lies between
         them. If a feature's absolute scale is part of what distinguishes mice, it is gone.

    Parameters
    ----------
    features : DataFrame or array, (n_sessions, n_features)
    groups : array-like of length n_sessions, or None
        None (default) = one global ranking per feature. Pass `lab_of_session` to rank
        WITHIN LAB instead, which is Forkosh's per-batch-per-day version and removes the
        lab's location, scale AND distribution shape at once -- strictly more than
        remove_lab_variance(mode='zscore'), which only removes the first two.

        READ remove_lab_variance's docstring BEFORE USING THIS. Lab is nested inside mouse,
        so ranking within lab pays exactly the same price: the mouse-identity signal loses
        one dimension per lab (10 of the 57 here). Forkosh's batches are nested the same way
        and the paper does not mention it.
    offset : 'blom' | 'vdw'
        The plotting position that turns a rank r of n into a quantile.
        'blom' (default) uses (r - 3/8) / (n + 1/4); 'vdw' (van der Waerden) uses
        r / (n + 1). Both keep the extreme ranks finite -- a plain r/n would send the
        largest value to Phi^-1(1) = inf.
    min_per_group : int
        Groups smaller than this are transformed anyway but NAMED in the printout. A rank
        map estimated from 12 sessions is coarse and noisy, and the noise differs per group.
        Note this is the opposite of remove_lab_variance's rule, which SKIPS small labs:
        there, skipping leaves them on the same raw scale as everyone else, which is
        harmless; here, skipping would leave them on a different scale from the transformed
        groups, which is worse than a noisy map.

    Returns
    -------
    Same type as `features`, same shape, same index/columns. NaNs are left in place and are
    excluded from the ranking of their own column.
    """
    from scipy.stats import rankdata, norm

    if offset not in ('blom', 'vdw'):
        raise ValueError(f"offset must be 'blom' or 'vdw', got {offset!r}")

    is_df = isinstance(features, pd.DataFrame)
    X = np.asarray(features, dtype=float).copy()

    def _score_block(block):
        """Normal scores of one (rows x features) block, column by column, NaN-aware."""
        out = np.full_like(block, np.nan)
        for j in range(block.shape[1]):
            col = block[:, j]
            ok = ~np.isnan(col)
            n = int(ok.sum())
            if n == 0:
                continue
            if n == 1:
                out[ok, j] = 0.0        # a single value has no ordering; put it at the median
                continue
            # 'average' so that tied values all receive the same score, as Forkosh specify
            r = rankdata(col[ok], method='average')
            q = (r - 0.375) / (n + 0.25) if offset == 'blom' else r / (n + 1.0)
            out[ok, j] = norm.ppf(q)
        return out

    if groups is None:
        X = _score_block(X)
        if verbose:
            print(f'  rank-inverse-normal ({offset}): global, {X.shape[1]} features '
                  f'over {X.shape[0]} sessions')
    else:
        g = pd.Series(list(groups)).to_numpy(dtype=object)
        if len(g) != X.shape[0]:
            raise ValueError(f'groups has length {len(g)}, features has {X.shape[0]} rows')
        small, n_groups = [], 0
        for lab in pd.unique(g[pd.notna(g)]):
            rows = np.where(g == lab)[0]
            n_groups += 1
            if len(rows) < min_per_group:
                small.append(f'{lab} ({len(rows)} sessions)')
            X[rows] = _score_block(X[rows])
        unknown = int(pd.isna(g).sum())
        if unknown:
            # ranked together rather than left raw: a raw block among transformed ones
            # would be the only part of the matrix still on the original scale
            X[pd.isna(g)] = _score_block(X[pd.isna(g)])
        if verbose:
            print(f'  rank-inverse-normal ({offset}): within group, {n_groups} groups'
                  + (f', {unknown} sessions with no group ranked together' if unknown else '')
                  + (f'; SMALL (coarse, noisy map): {", ".join(small)}' if small else ''))

    return (pd.DataFrame(X, index=features.index, columns=features.columns)
            if is_df else X)
