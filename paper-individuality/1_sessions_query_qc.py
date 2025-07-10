"""
1. Query brainwide map sessions and filter based on qc
@author: Ines
"""
#%%

import numpy as np
import pandas as pd
from dateutil import parser

from functions import extended_qc

from one.api import ONE
one = ONE(mode='remote')

#%%
# THIS CELL USES A QUERY VERY SIMILAR TO THE BWM PAPER BUT WITHOUT A DATE CUTOFF OR REQUIREMENT FOR PROBE ALIGNMENT QC
# https://github.com/int-brain-lab/paper-brain-wide-map/blob/4b9d47f4444c5f4b91026588218e4d5869aff5a9/brainwidemap/bwm_loading.py#L21 

base_query = (
    'session__projects__name__icontains,ibl_neuropixel_brainwide_01,'
    '~session__json__IS_MOCK,True,'
    'session__qc__lt,50,'
    'session__extended_qc__behavior,1,'
    '~json__qc,CRITICAL,'  # Should clarify these
    'json__extended_qc__alignment_count__gt,0,'  # No need for alignment resolved
)
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
    '~session__extended_qc___task_audio_pre_trial__lt,0.9')

marked_pass = (
    'session__extended_qc___experimenter_task,PASS')  # What is this?

insertions = list(one.alyx.rest('insertions', 'list', django=base_query + qc_task))
insertions.extend(list(one.alyx.rest('insertions', 'list', django=base_query + marked_pass)))
print(len(insertions))

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

#%%
# TODO there must be smarter way to do this
# ADD VIDEO QC FILTER

ext_qc = extended_qc(one, bwm_df)

final_qc = ext_qc.loc[(ext_qc['_lightningPoseLeft_lick_detection'].isin(['PASS'])) &
                             (ext_qc['_lightningPoseLeft_time_trace_length_match'].isin(['PASS'])) &   
                             (ext_qc['_videoLeft_pin_state'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &
                             (ext_qc['_lightningPoseLeft_trace_all_nan'].isin(['PASS'])) &
                             (ext_qc['_videoLeft_framerate'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &   
                             (ext_qc['_videoLeft_camera_times'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &   
                             (ext_qc['_videoLeft_dropped_frames'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &  # can make more conservative by removing or  x == None
                             (ext_qc['_videoLeft_timestamps'].isin([True, 'PASS']))]

#%%