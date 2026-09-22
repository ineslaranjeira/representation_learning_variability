"""
4. Habituation sessions for the LDA cohort: do they exist, do they have video, and what
   video QC was run on them.
=========================================================================================
The LDA pipeline (paper-individuality/4_mice/lda_shrinkage.ipynb) ends up with a specific
set of mice -- those that survive the QC sheet, the missing-bin screen and the
`>= MIN_SESSIONS_PER_MOUSE` rule. This script takes exactly that set and asks, per mouse,
whether Alyx has habituationChoiceWorld sessions for it and whether those sessions could
ever feed the same segmentation pipeline.

Three things are worth knowing before reading the output:

  * Habituation is LEFT CAMERA ONLY. Across the cohort there is not a single rightCamera
    or bodyCamera dataset on a habituation session, so the two-camera lick detection the
    proficient pipeline uses (`merge_licks` in segmentation_functions) has no counterpart
    here. Anything built on habituation video is a one-camera analysis.

  * There is NO pose estimation. No DLC, no lightningPose, on any habituation session.
    Pose would have to be run from the raw mp4, which also means the lightningPose QC
    checks that `1_sessions_query_qc.py` filters proficient sessions on simply do not
    exist to filter on.

  * The QC values are stored in several encodings. Alyx has written these fields as bare
    booleans, as PASS/FAIL/WARNING strings, as `[outcome, value, ...]` lists, and in the
    oldest sessions as `[value, value]` lists with no outcome recorded at all. Reading any
    of them with `== 'PASS'` silently scores the boolean-era sessions as failures, so
    everything goes through `normalise_qc` below.

@author: Ines
"""
#%%
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- LDA COHORT DEFINITION --------------------------------------------------------
# These MUST match paper-individuality/4_mice/lda_shrinkage.ipynb. They are duplicated
# rather than imported because the filter lives inside the notebook, not in functions.py.
# If the notebook's config cell changes, change these too or the cohort silently drifts.
SYLLABLE_FILE = '8_k_10_bin_syllables_19-08-2026'
QC_STRICTNESS = 'filtered_out'
MAX_SESSION_MISSING = 0.1
MIN_SESSIONS_PER_MOUSE = 3
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']

# ---- QUERY ------------------------------------------------------------------------
REFRESH = False
# False reuses CACHE_FILE if it exists. The query is ~1 call per mouse plus 1 per
# habituation session (about 210 REST calls, a few minutes). Set True after new
# sessions are registered or after the QC sheet changes the cohort.
CACHE_FILE = 'habituation_alyx_cache.pkl'

# ---- VIDEO QC GROUPING ------------------------------------------------------------
# Split the same way `bwm_habituation_sessions_video.csv` splits it: whether the video is
# usable as PIXELS, and whether it is usable as a TIME SERIES aligned to the task. The
# second is the one that matters for trial-epoch binning, and it is the one that is most
# often missing.
QUALITY_CHECKS = ['_videoLeft_focus', '_videoLeft_position', '_videoLeft_brightness',
                  '_videoLeft_resolution', '_videoLeft_file_headers']
TIMING_CHECKS = ['_videoLeft_camera_times', '_videoLeft_timestamps', '_videoLeft_pin_state',
                 '_videoLeft_dropped_frames', '_videoLeft_framerate']
# `_videoLeft_wheel_alignment` is deliberately in neither: it is None or NOT_SET on every
# habituation session in the cohort, so including it would make every tier empty.
ALL_CHECKS = QUALITY_CHECKS + TIMING_CHECKS

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
PAPER_ROOT = HERE.parent.parent                     # paper-individuality/


#%%
# ---- STEP 1: reproduce the LDA cohort ---------------------------------------------
def lda_cohort():
    """The mice the LDA pipeline actually fits, with their surviving session counts.

    A mirror of `filter_sequences` + the pivot in `build_design_matrix`, minus the
    binarisation -- we only need which mice and sessions come out the far end. The pivot
    is kept because its `.dropna()` can in principle remove a session, and the cohort has
    to be the one the LDA sees, not the one the screens alone imply.
    """
    sys.path.insert(0, str(PAPER_ROOT / 'learning_individuality'))
    sys.path.insert(0, str(PAPER_ROOT / '4_mice'))
    from session_filters import exclusions_by_timepoint, find_csv
    from functions import lab_labels

    seq = pd.read_parquet(str(PAPER_ROOT / 'data' / SYLLABLE_FILE))
    seq['session'] = seq['sample'].str[:36]
    print(f'{seq.mouse_name.nunique()} mice, {seq.session.nunique()} sessions in total')

    prob_sessions = sorted(exclusions_by_timepoint(QC_STRICTNESS)['Proficient'])
    print(f'QC sheet: {find_csv().name} -> {len(prob_sessions)} proficient sessions dropped')
    seq = seq.loc[~seq['session'].isin(prob_sessions)].reset_index(drop=True)
    print(f'{seq["session"].nunique()} sessions after the QC sheet ({QC_STRICTNESS})')

    if MAX_SESSION_MISSING is not None:
        nan_frac = np.isnan(np.stack(seq['binned_sequence'].to_numpy())).mean(axis=1)
        by_session = (pd.Series(nan_frac, index=seq['session'].to_numpy())
                      .groupby(level=0).mean())
        drop = by_session[by_session > MAX_SESSION_MISSING]
        seq = seq.loc[~seq['session'].isin(drop.index)].reset_index(drop=True)
        print(f'{seq["session"].nunique()} sessions after dropping {len(drop)} with > '
              f'{MAX_SESSION_MISSING:.1%} missing bins')

    counts = (seq[['mouse_name', 'session']].drop_duplicates()
              .groupby('mouse_name')['session'].count())
    keep = counts[counts >= MIN_SESSIONS_PER_MOUSE].index
    seq = seq.loc[seq['mouse_name'].isin(keep)].reset_index(drop=True)

    trials = (seq.pivot(index=['mouse_name', 'session', 'sample', 'trial_type'],
                        columns=['broader_label'], values='binned_sequence')
              .reset_index().dropna())
    pairs = trials[['mouse_name', 'session']].drop_duplicates()
    cohort = (pairs.groupby('mouse_name')['session'].count()
              .rename('n_lda_sessions').reset_index())

    # Lab from the QC sheet's rig names, not from Alyx. It is offline, it is what every
    # other figure in the paper uses, and -- the reason it matters here -- it covers the
    # mice that have no habituation session at all, which an Alyx habituation query by
    # construction cannot. It is also wrong for SWC; see the cross-check in step 4.
    labs = lab_labels(pairs['session'], mouse_names=pairs['mouse_name'], verbose=False)
    cohort['lab'] = cohort['mouse_name'].map(
        pd.Series(labs.to_numpy(), index=pairs['mouse_name'].to_numpy()).groupby(level=0).first())

    print(f'-> LDA cohort: {len(cohort)} mice with >= {MIN_SESSIONS_PER_MOUSE} sessions, '
          f'{trials["session"].nunique()} proficient sessions, '
          f'{cohort["lab"].nunique()} labs')
    return cohort


cohort = lda_cohort()


#%%
# ---- STEP 2: pull every habituation session for those mice ------------------------
def query_habituation(subjects, one):
    """{subject: [full session record, ...]} for habituationChoiceWorld sessions.

    `sessions/read` is used rather than `list_datasets` because the read already carries
    `data_dataset_session_related` AND `extended_qc`, so it is one call per session
    instead of two. `task_protocol='habituation'` is an icontains match on Alyx and
    catches every _iblrig_tasks_habituationChoiceWorld* version (14 of them here).
    """
    out = {}
    for i, subj in enumerate(subjects):
        sess = one.alyx.rest('sessions', 'list', subject=subj,
                             task_protocol='habituation')
        out[subj] = [one.alyx.rest('sessions', 'read', id=s['id']) for s in sess]
        print(f'  [{i + 1}/{len(subjects)}] {subj}: {len(sess)} habituation sessions',
              flush=True)
    return out


cache_path = HERE / CACHE_FILE
if cache_path.exists() and not REFRESH:
    records = pickle.load(open(cache_path, 'rb'))
    missing = [m for m in cohort['mouse_name'] if m not in records]
    print(f'loaded {cache_path.name} ({len(records)} mice)'
          + (f'  !! {len(missing)} cohort mice not in cache: {missing} -- set REFRESH=True'
             if missing else ''))
else:
    from one.api import ONE
    one = ONE(mode='remote')
    records = query_habituation(cohort['mouse_name'].tolist(), one)
    pickle.dump(records, open(cache_path, 'wb'))
    print(f'wrote {cache_path.name}')


#%%
# ---- STEP 3: normalise the QC encodings -------------------------------------------
OUTCOMES = ('PASS', 'WARNING', 'FAIL', 'CRITICAL', 'NOT_SET')


def normalise_qc(value):
    """Any of Alyx's QC encodings -> one of PASS / WARNING / FAIL / CRITICAL / NOT_SET /
    NO_OUTCOME / ABSENT.

    The encodings actually present on habituation sessions in this cohort:
        True / False                 boolean era          -> PASS / FAIL
        'PASS' / 'FAIL' / 'WARNING'  string era           -> itself
        ['PASS', 0]  [True, 0, 0]    outcome + values     -> first element, recursed
        [202, 0]  [0, 0]  0  32.895  values, NO outcome   -> NO_OUTCOME
        None                                              -> NOT_SET

    NO_OUTCOME is kept distinct from FAIL and from NOT_SET on purpose. It means the check
    ran and stored its measurement but no verdict was ever written, which is a different
    claim from "the check failed" and from "the check never ran". Collapsing it into
    either one would misreport the oldest sessions, which are a third of the cohort.
    """
    if isinstance(value, (list, tuple)):
        if not len(value):
            return 'NOT_SET'
        return normalise_qc(value[0])
    if isinstance(value, bool):
        return 'PASS' if value else 'FAIL'
    if value is None:
        return 'NOT_SET'
    if isinstance(value, str):
        return value.upper() if value.upper() in OUTCOMES else 'NO_OUTCOME'
    return 'NO_OUTCOME'          # a bare number: a measurement with no verdict


def session_row(subject, det):
    """One habituation session -> a flat row of presence flags and normalised QC."""
    names = {d['name'] for d in det['data_dataset_session_related']}
    eqc = det.get('extended_qc') or {}

    row = {
        # mouse_name/eid/date first and named the way `*_eids.csv` names them, so this file
        # IS the eid list -- `df[df.has_raw_leftCam]` is the LightningPose job list, and
        # `& df.has_leftCam_times` is the subset that can already be binned into trials.
        # There is deliberately no separate eids CSV to drift out of sync with this one.
        'mouse_name': subject,
        'eid': det['id'],
        'date': det['start_time'][:10],
        'alyx_lab': det['lab'],
        'number': det['number'],
        'task_protocol': det['task_protocol'],
        'session_qc': det.get('qc'),
        # -- what video data exists --
        'has_raw_leftCam': '_iblrig_leftCamera.raw.mp4' in names,
        'has_raw_rightCam': any('rightCamera.raw' in n for n in names),
        'has_raw_bodyCam': any('bodyCamera.raw' in n for n in names),
        'has_rig_timestamps': '_iblrig_leftCamera.timestamps.ssv' in names,
        'has_leftCam_times': '_ibl_leftCamera.times.npy' in names,   # ALF, task-aligned
        'has_dlc': any('dlc' in n.lower() for n in names),
        'has_lightningPose': any('lightningPose' in n or 'lightning_pose' in n
                                 for n in names),
        # -- what video QC exists --
        'video_qc_run': any(k.startswith('_videoLeft_') for k in eqc),
        'videoLeft_qc': eqc.get('videoLeft', 'ABSENT'),
    }
    for check in ALL_CHECKS:
        row[check] = normalise_qc(eqc[check]) if check in eqc else 'ABSENT'

    quality = [row[c] for c in QUALITY_CHECKS]
    timing = [row[c] for c in TIMING_CHECKS]
    row['no_quality_FAIL'] = not any(v in ('FAIL', 'CRITICAL') for v in quality)
    row['no_timing_FAIL'] = not any(v in ('FAIL', 'CRITICAL') for v in timing)
    row['n_checks_resolved'] = sum(v in ('PASS', 'WARNING', 'FAIL', 'CRITICAL')
                                   for v in quality + timing)

    # Nested tiers, each adding one requirement to the one above. There is no `tier1`
    # column because it would be a verbatim copy of `has_raw_leftCam`.
    row['tier2_timed'] = row['has_raw_leftCam'] and row['has_leftCam_times']
    row['tier3_good'] = (row['tier2_timed'] and row['no_quality_FAIL']
                         and row['no_timing_FAIL'])
    row['tier4_strict'] = (row['tier3_good']
                           and row['n_checks_resolved'] == len(ALL_CHECKS)
                           and all(v == 'PASS' for v in quality + timing))
    return row


sessions = pd.DataFrame([session_row(subj, det)
                         for subj, dets in records.items() for det in dets])
if len(sessions):
    sessions = (sessions.sort_values(['alyx_lab', 'mouse_name', 'date'])
                .reset_index(drop=True))
# Carry the cohort columns onto every session row so this one file answers "which mouse,
# which lab, how many proficient sessions does it have" without a second lookup.
sessions = sessions.merge(cohort.rename(columns={'lab': 'sheet_lab'}), on='mouse_name',
                          how='left')
print(f'{len(sessions)} habituation sessions across '
      f'{sessions["mouse_name"].nunique()} mice')


#%%
# ---- STEP 4: per-mouse summary ----------------------------------------------------
agg = (sessions.groupby('mouse_name')
       .agg(alyx_lab=('alyx_lab', 'first'),
            n_habituation=('eid', 'size'),
            n_with_video=('has_raw_leftCam', 'sum'),
            n_with_times=('has_leftCam_times', 'sum'),
            n_video_qc_run=('video_qc_run', 'sum'),
            n_good_video=('tier3_good', 'sum'),
            n_strict=('tier4_strict', 'sum'),
            first_hab=('date', 'min'),
            last_hab=('date', 'max')))

per_mouse = cohort.set_index('mouse_name').join(agg).reset_index()
count_cols = ['n_habituation', 'n_with_video', 'n_with_times', 'n_video_qc_run',
              'n_good_video', 'n_strict']
per_mouse[count_cols] = per_mouse[count_cols].fillna(0).astype(int)
per_mouse['has_habituation'] = per_mouse['n_habituation'] > 0

# `lab` came with the cohort (QC sheet rig names, every mouse); `alyx_lab` came with the
# habituation sessions (only the mice that have any). BOTH are kept, because at SWC they
# disagree and the sheet is the one that is wrong.
#
# `lab_labels` reads the lab out of `_iblrig_<lab>_<ephys|behavior>_<n>`. That works when a
# rig belongs to one lab, and at SWC it does not: every SWC mouse in this cohort ran its
# PROFICIENT sessions on the shared `_iblrig_mrsicflogel_ephys_0`, so all 7 come back as
# 'mrsicflogel' even though Alyx puts 3 of them in hoferlab. (The sheet does carry
# `_iblrig_hofer_behavior_*` rigs for their training sessions, and SWC_060 appears on a
# hofer rig and a mrsicflogel rig in different weeks -- so rig name does not identify the
# lab at that institute in either direction.) This is the same merge that `lab_labels`'
# own docstring says must not happen, and it is inherited by anything that calls it on
# ephys sessions, including the LD1 lab-confound analysis.
#
# The by-lab table below still uses the sheet lab so it lines up with every other figure in
# the paper. Use `alyx_lab` if you want hoferlab and mrsicflogellab actually separated.
clash = per_mouse.dropna(subset=['alyx_lab']).query('lab != alyx_lab')
if len(clash):
    pairs = clash.groupby(['lab', 'alyx_lab'])['mouse_name'].agg(list)
    print('!! sheet lab != Alyx lab (both columns kept in the CSV):')
    for (sheet_lab, alyx_lab), subs in pairs.items():
        print(f'     sheet {sheet_lab!r} -> Alyx {alyx_lab!r}: {len(subs)} mice '
              f'({", ".join(subs)})')
per_mouse = per_mouse.sort_values(['lab', 'mouse_name']).reset_index(drop=True)


#%%
# ---- STEP 5: the report -----------------------------------------------------------
n_mice = len(per_mouse)
line = '=' * 78


def pct(n, d=None):
    d = n_mice if d is None else d
    return f'{n}/{d} ({n / d:.0%})' if d else f'{n}/0'


print(f'\n{line}\nHABITUATION SESSIONS FOR THE LDA COHORT\n{line}')
print(f'Cohort: {n_mice} mice with >= {MIN_SESSIONS_PER_MOUSE} proficient sessions '
      f'in {SYLLABLE_FILE}')

print(f'\n-- 1. DO THEY HAVE HABITUATION SESSIONS --')
print(f'mice with >= 1 habituation session : {pct(per_mouse["has_habituation"].sum())}')
print(f'total habituation sessions         : {int(per_mouse["n_habituation"].sum())}')
none = per_mouse.loc[~per_mouse['has_habituation'], 'mouse_name'].tolist()
if none:
    print(f'mice with NONE registered on Alyx  : {", ".join(none)}')
dist = per_mouse['n_habituation'].value_counts().sort_index()
print('sessions per mouse                 : '
      + ', '.join(f'{k}->{v} mice' for k, v in dist.items()))

print(f'\n-- 2. DO THOSE SESSIONS HAVE VIDEO --')
ns = len(sessions)
for label, col in [('raw leftCamera mp4', 'has_raw_leftCam'),
                   ('raw rightCamera mp4', 'has_raw_rightCam'),
                   ('raw bodyCamera mp4', 'has_raw_bodyCam'),
                   ('rig .timestamps.ssv', 'has_rig_timestamps'),
                   ('ALF leftCamera.times (task-aligned)', 'has_leftCam_times'),
                   ('DLC', 'has_dlc'),
                   ('lightningPose', 'has_lightningPose')]:
    print(f'  sessions with {label:<37s}: {pct(int(sessions[col].sum()), ns)}')
print(f'  mice with >= 1 habituation session WITH video   : '
      f'{pct((per_mouse["n_with_video"] > 0).sum())}')
print(f'  mice with >= 1 habituation session WITH ALF times: '
      f'{pct((per_mouse["n_with_times"] > 0).sum())}')

print(f'\n-- 3. WHAT VIDEO QC EXISTS --')
print(f'  sessions with any _videoLeft_* QC : '
      f'{pct(int(sessions["video_qc_run"].sum()), ns)}')
print(f'  `videoLeft` aggregate outcome     : '
      + ', '.join(f'{k} {v}' for k, v in
                  sessions['videoLeft_qc'].astype(str).value_counts().items()))
print('\n  per-check outcomes (ABSENT = the check is not on this session at all;')
print('   NO_OUTCOME = a measurement was stored but no verdict was ever written):\n')
qc_table = pd.DataFrame({c: sessions[c].value_counts() for c in ALL_CHECKS}).T
qc_table = qc_table.reindex(columns=[c for c in
                                     ['PASS', 'WARNING', 'FAIL', 'CRITICAL', 'NOT_SET',
                                      'NO_OUTCOME', 'ABSENT'] if c in qc_table.columns])
qc_table.insert(0, 'group', ['quality' if c in QUALITY_CHECKS else 'timing'
                             for c in qc_table.index])
print(qc_table.fillna(0).astype({c: int for c in qc_table.columns if c != 'group'})
      .to_string())

print(f'\n-- 4. HOW MANY SESSIONS SURVIVE EACH TIER --')
for tier, desc in [('has_raw_leftCam', 'raw leftCamera mp4 registered'),
                   ('tier2_timed', '+ ALF leftCamera.times (alignable to trials)'),
                   ('tier3_good', '+ no FAIL/CRITICAL on any quality or timing check'),
                   ('tier4_strict', '+ every check resolved and PASS')]:
    n_sess = int(sessions[tier].sum())
    n_mouse = sessions.loc[sessions[tier], 'mouse_name'].nunique()
    print(f'  {tier:<13s} {desc:<45s} {n_sess:>4d} sessions, {n_mouse:>3d} mice')

usable = per_mouse[per_mouse['n_good_video'] >= MIN_SESSIONS_PER_MOUSE]
print(f'\n  mice with >= {MIN_SESSIONS_PER_MOUSE} tier3 habituation sessions '
      f'(a within-habituation LDA of the same shape): {len(usable)}')

print(f'\n-- 5. BY LAB --')
by_lab = per_mouse.groupby('lab').agg(
    mice=('mouse_name', 'size'), with_hab=('has_habituation', 'sum'),
    hab_sessions=('n_habituation', 'sum'), with_video=('n_with_video', 'sum'),
    with_times=('n_with_times', 'sum'), qc_run=('n_video_qc_run', 'sum'),
    tier3=('n_good_video', 'sum'))
print(by_lab.to_string())


#%%
# ---- STEP 6: save -----------------------------------------------------------------
# TWO files and no more. The LightningPose job list and its time-aligned subset are
# `sessions[sessions.has_raw_leftCam]` and `& sessions.has_leftCam_times` -- writing those
# out separately would only create copies that go stale when this script is re-run.
# `per_mouse` is NOT derivable from `sessions`: it carries the cohort mice that have no
# habituation session at all, which a one-row-per-session file structurally cannot.
sess_out = HERE / 'habituation_sessions.csv'
mouse_out = HERE / 'habituation_per_mouse.csv'
sessions.to_csv(sess_out, index=False)
per_mouse.to_csv(mouse_out, index=False)
print(f'\nwrote {sess_out.name} ({len(sessions)} rows)')
print(f'wrote {mouse_out.name} ({len(per_mouse)} rows)')

# %%
