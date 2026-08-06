"""
1. Query brainwide map behavioral sessions and filter based on qc - no need for neural QC or available insertions
@author: Ines
"""
#%%

import numpy as np
import pandas as pd
from dateutil import parser
from datetime import datetime
import os
from segmentation_functions import extended_qc

from one.api import ONE
one = ONE(mode='remote')

#%%
# THIS CELL USES A QUERY VERY SIMILAR TO THE BWM PAPER BUT WITHOUT A DATE CUTOFF OR REQUIREMENT FOR PROBE ALIGNMENT QC
# https://github.com/int-brain-lab/paper-brain-wide-map/blob/4b9d47f4444c5f4b91026588218e4d5869aff5a9/brainwidemap/bwm_loading.py#L21
# Unlike the original BWM query, this queries the 'sessions' endpoint directly instead of 'insertions',
# so sessions are kept even if they have no (fully QC'd) probe insertion.

base_query = (
    'projects__name__icontains,ibl_neuropixel_brainwide_01,'
    '~json__IS_MOCK,True,'
    'qc__lt,50,'
    'extended_qc__behavior,1,'
)
qc_task = (
    '~extended_qc___task_stimOn_goCue_delays__lt,0.9,'
    '~extended_qc___task_response_feedback_delays__lt,0.9,'
    '~extended_qc___task_wheel_move_before_feedback__lt,0.9,'
    '~extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
    '~extended_qc___task_error_trial_event_sequence__lt,0.9,'
    '~extended_qc___task_correct_trial_event_sequence__lt,0.9,'
    '~extended_qc___task_reward_volumes__lt,0.9,'
    '~extended_qc___task_reward_volume_set__lt,0.9,'
    '~extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
    '~extended_qc___task_audio_pre_trial__lt,0.9')

marked_pass = (
    'extended_qc___experimenter_task,PASS')  # What is this?

sessions = list(one.alyx.rest('sessions', 'list', django=base_query + qc_task))
sessions.extend(list(one.alyx.rest('sessions', 'list', django=base_query + marked_pass)))
print(len(sessions))

bwm_df = pd.DataFrame({
    'eid': np.array([s['id'] for s in sessions]),
    'session_number': np.array([s['number'] for s in sessions]),
    'date': np.array([parser.parse(s['start_time']).date() for s in sessions]),
    'subject': np.array([s['subject'] for s in sessions]),
    'lab': np.array([s['lab'] for s in sessions]),
}).sort_values(by=['lab', 'subject', 'date', 'eid'])
bwm_df.drop_duplicates(inplace=True)
bwm_df.reset_index(inplace=True, drop=True)

#%%
# TODO there must be smarter way to do this
# ADD VIDEO QC FILTER

ext_qc = extended_qc(one, bwm_df['eid'].unique())  #TODO: THIS CODE GIVES AN ERROR ON THE PC AND NOT ON MAC... 

# Includes right camera; no need to confirm frame rate, run this on the 12Mar2026
final_qc = ext_qc.loc[(ext_qc['_lightningPoseLeft_lick_detection'].isin(['PASS'])) &
                      (ext_qc['_lightningPoseLeft_time_trace_length_match'].isin(['PASS'])) &   
                      (ext_qc['_videoLeft_pin_state'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &
                      (ext_qc['_lightningPoseLeft_trace_all_nan'].isin(['PASS'])) & 
                      (ext_qc['_videoLeft_camera_times'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &   
                      (ext_qc['_videoLeft_dropped_frames'].apply(lambda x: (isinstance(x, list) and True in x) or  x == None or x == 'PASS')) &  # can make more conservative by removing or  x == None
                      (ext_qc['_videoLeft_timestamps'].isin([True, 'PASS']))&
                      (ext_qc['_lightningPoseRight_lick_detection'].isin(['PASS'])) &
                      (ext_qc['_lightningPoseRight_time_trace_length_match'].isin(['PASS'])) &   
                      (ext_qc['_videoRight_pin_state'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &
                      (ext_qc['_lightningPoseRight_trace_all_nan'].isin(['PASS'])) & 
                      (ext_qc['_videoRight_camera_times'].apply(lambda x: (isinstance(x, list) and True in x) or x == 'PASS')) &   
                      (ext_qc['_videoRight_dropped_frames'].apply(lambda x: (isinstance(x, list) and True in x) or  x == None or x == 'PASS')) &  # can make more conservative by removing or  x == None
                      (ext_qc['_videoRight_timestamps'].isin([True, 'PASS']))]

# #%%
# Save to google drive
gdrive_path = "/Users/ineslaranjeira/Google Drive/O meu disco/CCU/PhD Project/paper-individuality/data/segmentation/"
gdrive_path = "/home/ines/repositories/representation_learning_variability/paper-individuality/segmentation/data_query/"
# Ensure the directory exists
os.makedirs(gdrive_path, exist_ok=True)
# Save your file
filename = 'bwm_qc_new_'
now = datetime.now() # current date and time
date_time = now.strftime("%m-%d-%Y")
file_path = os.path.join(gdrive_path, filename + date_time)
final_qc.to_pickle(file_path, compression='gzip')  

# %%
