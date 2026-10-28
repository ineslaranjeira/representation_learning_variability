"""
Per-session task performance of the DA mice (training, biased, ephys sessions).
==============================================================================
Loads each session's trials table from Alyx (brainbox SessionLoader, which finds the task
collection) for every session in da_inventory_sessions.csv of the mice with DA photometry, and
writes one row per session:

    n_trials, n_contrasts           how many trials, how many distinct |contrast| levels shown
                                    (the training stages add contrasts: 2 -> 3 -> ... -> 6 incl. 0)
    p_correct                       fraction correct over go trials
    p_correct_easy                  fraction correct at |contrast| >= 0.5 (the IBL "easy" criterion)
    p_nogo                          fraction of trials with no response (choice == 0)
    median_rt                       median response_times - goCue_times (s), go trials only
    session_minutes                 last trial end - first trial start

Habituation sessions are skipped (no choices). Resumable: sessions already in the output are skipped.

Usage (iblenv, Alyx access):  python da_performance.py
Output: da_performance.csv (next to this file)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('TQDM_DISABLE', '1')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'da_performance.csv'
CLASSES = ['training', 'biased', 'ephys']
COLUMNS = ['eid', 'n_trials', 'n_contrasts', 'p_correct', 'p_correct_easy', 'p_nogo', 'median_rt',
           'session_minutes', 'error']   # fixed, so rows that failed don't shift the columns
N_WORKERS = 6             # parallel downloads (each worker has its own ONE)


def performance(trials):
    c = np.nan_to_num(trials['contrastLeft'].to_numpy(float)) + np.nan_to_num(trials['contrastRight'].to_numpy(float))
    go = trials['choice'].to_numpy() != 0
    correct = trials['feedbackType'].to_numpy() == 1
    rt = (trials['response_times'] - trials['goCue_times']).to_numpy()
    easy = go & (c >= 0.5)
    return dict(n_trials=len(trials), n_contrasts=len(np.unique(c)),
                p_correct=correct[go].mean() if go.any() else np.nan,
                p_correct_easy=correct[easy].mean() if easy.any() else np.nan,
                p_nogo=1 - go.mean(),
                median_rt=np.nanmedian(rt[go]) if go.any() else np.nan,
                session_minutes=(trials['intervals_1'].max() - trials['intervals_0'].min()) / 60)


_ONE = None


def _init_worker():
    global _ONE
    from one.api import ONE
    _ONE = ONE(base_url='https://alyx.internationalbrainlab.org', silent=True)


def session_row(eid):
    from brainbox.io.one import SessionLoader
    row = dict(eid=eid)
    try:
        sl = SessionLoader(eid=eid, one=_ONE)
        sl.load_trials()
        row.update(performance(sl.trials))
    except Exception as e:
        row['error'] = ' '.join(f'{type(e).__name__}: {e}'.split())[:200]
    return row


def main(n_workers=N_WORKERS):
    from multiprocessing import Pool
    s = pd.read_csv(HERE / 'da_inventory_sessions.csv')
    da_mice = s.loc[s['signal'], 'subject'].unique()
    s = s[s['subject'].isin(da_mice) & s['session_class'].isin(CLASSES)]
    done = set(pd.read_csv(OUT)['eid']) if OUT.exists() else set()
    todo = list(s.loc[~s['eid'].isin(done), 'eid'])
    print(f'{len(s)} sessions, {len(todo)} to do', flush=True)
    with Pool(n_workers, initializer=_init_worker) as pool:
        for i, row in enumerate(pool.imap_unordered(session_row, todo)):
            pd.DataFrame([row], columns=COLUMNS).to_csv(OUT, mode='a', header=not OUT.exists(), index=False)
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{len(todo)}', flush=True)


if __name__ == '__main__':
    main()
