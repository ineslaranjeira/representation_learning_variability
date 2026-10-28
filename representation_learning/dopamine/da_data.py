"""
Loaders for the dopamine-during-training data inventory.
========================================================
* `load_inventory()`: the Alyx inventory written by da_inventory.py: one row per session of every
  DAT-Cre mainenlab mouse, with photometry / wheel / left-camera / Lightning Pose flags, and one
  row per mouse.
* `load_lp_sheet()`: the "LightningPoseSessions" sheet (neuromodulators/, exported from the
  collaborators' Google sheet). It has two header rows: section names, then column names; the
  second is the real header. One row per session (eid) with Lightning Pose status, video QC
  (timing, quality), the LP wheel-x-paw cross-correlation QC, event-triggered movement QC and
  manual QC.
* `session_table()`: the inventory joined to the sheet on eid (left join: sessions the sheet
  doesn't list keep NaN QC).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LP_SHEET = REPO / 'neuromodulators' / 'LightningPoseSessions - LightningPoseSessions.csv'
VIDEO_QC = ['qc_videoLeft_timestamps', 'qc_videoLeft_dropped_frames', 'qc_videoLeft_pin_state',
            'qc_videoLeft_focus', 'qc_videoLeft_position', 'qc_videoLeft_brightness',
            'qc_videoLeft_resolution', 'qc_videoLeft_wheel_alignment']
QC_ORDER = ['PASS', 'WARNING', 'FAIL', 'CRITICAL', 'NOT_SET']


def load_inventory():
    sessions = pd.read_csv(HERE / 'da_inventory_sessions.csv')
    mice = pd.read_csv(HERE / 'da_inventory_mice.csv')
    return sessions, mice


def load_lp_sheet(path=LP_SHEET):
    sheet = pd.read_csv(path, header=1).drop(columns=['index'])
    sheet = sheet[sheet['eid'].notna()]
    num = ['fraction_correct', 'mean_rt', 'framerate_from_tpts', 'length_discrepancy', 'video_qc_score',
           'drift', 'peak_lag_early', 'peak_lag_mid', 'peak_lag_late', 'peak_val_early', 'peak_val_mid',
           'peak_val_late', 'motion_energy', 'nose', 'paw', 'tongue_likelihood', 'tongue_speed']
    sheet[num] = sheet[num].apply(pd.to_numeric, errors='coerce')
    sheet['video_qc_score'] = sheet['video_qc_score'].where(sheet['video_qc_score'] >= 0)   # -1 = not computed
    sheet['min_peak_val'] = sheet[['peak_val_early', 'peak_val_mid', 'peak_val_late']].min(axis=1)
    return sheet


def session_table():
    """Inventory sessions with the sheet's columns; `in_sheet` marks sessions the sheet lists, and
    `training_ordinal` numbers each mouse's training sessions from 1 in date order."""
    sessions, mice = load_inventory()
    sheet = load_lp_sheet()
    t = sessions.merge(sheet.rename(columns={'session_type': 'sheet_session_type'}), on='eid', how='left')
    t['in_sheet'] = t['LP status'].notna()
    t = t.sort_values(['subject', 'date', 'number']).reset_index(drop=True)
    is_tr = t['session_class'] == 'training'
    t.loc[is_tr, 'training_ordinal'] = t[is_tr].groupby('subject').cumcount() + 1
    return t.merge(mice[['subject', 'target', 'reached_biased']], on='subject', how='left')
