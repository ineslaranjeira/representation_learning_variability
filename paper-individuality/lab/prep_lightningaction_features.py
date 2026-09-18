"""
lightningAction PAW STATES IN THE SAME 10-BIN-PER-EPOCH FORMAT AS THE SYLLABLES
===============================================================================
Same skeleton as prep_wheel_features.py, and therefore as
learning_individuality/wheel/wheel_velocity_timepoints.ipynb -- the only difference is that
the per-bin value is a CATEGORICAL action state, so the sequences are rescaled with
`rescale_sequence(..., 'mode')` exactly as the syllables are, not 'mean'.

THE ONE EXTRA STEP is getting camera frames onto the design-matrix bin grid:

  frame -> absolute seconds        `_ibl_{cam}Camera.times.npy`
  seconds -> design-matrix bin     nearest `Bin` in the states file, BY TIMESTAMP
  bin -> one state                 mode of the frames that fall in that bin
  bins -> 10 per epoch             rescale_sequence(..., 'mode'), as for the syllables

Matching by timestamp rather than by index is not fussiness: camera frames drop, and the
states-file grid starts at the first design-matrix bin rather than at t=0, so frame i and row
i are different moments. Frames landing more than half a bin from any bin are dropped rather
than snapped across a gap. Measured frame match on a test session: 99.7%.

Taking the mode WITHIN a bin first, before rescaling, is what makes these features parallel
to the syllables: the syllable file already holds one state per 1/60 s bin, so both feature
sets are "one categorical state per bin, 10 bins per epoch" by the time they are encoded.

SANITY CHECK worth re-reading in the output: Quiescence should come out almost entirely
`still` with no `wheel_turn`, and Choice should be `wheel_turn`-dominated. The classifier is
never told about the task, so that is independent evidence the alignment is right.

PAWS: both, from the LEFT camera. Note the naming is mirrored between the two camera files --
left-camera `paw_r` is the animal's LEFT forepaw (the near one).
"""
import os
import sys
import glob
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / '4_mice'), str(ROOT / 'segmentation'),
           str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import functions
from segmentation_functions import define_trial_types, rescale_sequence

CACHE = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
STATES_DIR = ROOT / 'data' / 'states_files'
SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')
OUT = HERE / 'lightningaction_10bin_sequences.pqt'
STATES = ['background', 'still', 'move', 'wheel_turn', 'groom']
PAWS = ['paw_r', 'paw_l']
TRIAL_TYPE_AGG = ['correct_str', 'contrast_str', 'block_str', 'choice']
TARGET_LENGTH = 10
BIN = 1 / 60.0
CAM = 'left'


def _load_states(d):
    """argmax state per frame per paw; prefers the reduced npz, falls back to the parquet."""
    npz = os.path.join(d, f'pawstates_argmax_{CAM}.npz')
    if os.path.exists(npz):
        z = np.load(npz)
        return {p: z[p] for p in PAWS}
    pq = os.path.join(d, f'_ibl_{CAM}Camera.pawstates.pqt')
    if not os.path.exists(pq):
        return None
    df = pd.read_parquet(pq, columns=[f'{p}_{s}' for p in PAWS for s in STATES])
    return {p: np.argmax(df[[f'{p}_{s}' for s in STATES]].to_numpy(), axis=1).astype(np.uint8)
            for p in PAWS}


def session_sequences(eid, mouse, d):
    sf = glob.glob(str(STATES_DIR / f'8_states_file_{eid}_{mouse}'))
    times_f = os.path.join(os.path.dirname(d), f'_ibl_{CAM}Camera.times.npy')
    if not sf or not os.path.exists(times_f):
        return None, np.nan
    states = _load_states(d)
    if states is None:
        return None, np.nan

    st = pd.read_parquet(sf[0])
    st = define_trial_types(st, TRIAL_TYPE_AGG)
    bins = st['Bin'].to_numpy()

    t = np.load(times_f)
    n = min(len(t), len(states[PAWS[0]]))
    t = t[:n]
    pos = np.clip(np.searchsorted(bins, t), 1, len(bins) - 1)
    take = np.where(np.abs(t - bins[pos - 1]) < np.abs(t - bins[pos]), pos - 1, pos)
    ok = np.isfinite(t) & (np.abs(t - bins[take]) <= BIN / 2)
    coverage = float(ok.mean())

    rows = []
    for p in PAWS:
        s = states[p][:n]
        # One state per BIN (mode of that bin's frames), then the usual 10-bin rescale.
        # VECTORISED. The obvious `groupby('bin').agg(lambda v: bincount(v).argmax())` costs
        # a Python call per bin, and with 1/60 s bins against a ~60 Hz camera there is about
        # ONE FRAME PER BIN -- so that is ~200k calls per paw per session, 12 s a session and
        # 53 min over the set. Counting (bin, state) pairs in a single bincount does the same
        # thing in one pass; ties break to the lowest state index either way.
        key = take[ok].astype(np.int64) * len(STATES) + s[ok].astype(np.int64)
        counts = np.bincount(key, minlength=len(bins) * len(STATES)).reshape(len(bins), len(STATES))
        present = counts.sum(axis=1) > 0
        sub = st.iloc[np.flatnonzero(present)].copy()
        sub['state'] = counts[present].argmax(axis=1)
        sub = sub.dropna(subset=['broader_label', 'trial_id'])
        if len(sub) == 0:
            continue
        g = (sub.groupby(['sample', 'trial_type', 'broader_label', 'mouse_name'])['state']
             .apply(list).reset_index().rename(columns={'state': 'sequence'}))
        g['binned_sequence'] = g['sequence'].apply(
            lambda q: rescale_sequence(q, TARGET_LENGTH, 'mode'))
        g['paw'] = p
        g['session'] = eid
        rows.append(g.drop(columns=['sequence']))
    return (pd.concat(rows, ignore_index=True) if rows else None), coverage


def main():
    ss, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    mouse = dd[['mouse_name', 'session']].drop_duplicates().set_index('session')['mouse_name']
    from one.api import ONE
    one = ONE(mode='local')

    out, cov, absent = [], [], 0
    for i, eid in enumerate(ss.index, 1):
        p = one.eid2path(eid)
        if p is None:
            absent += 1
            continue
        q = str(p).split('/')
        d = os.path.join(CACHE, q[-5], 'Subjects', q[-3], q[-2], q[-1], 'alf', 'lightningaction')
        r, c = session_sequences(eid, mouse[eid], d)
        if r is None:
            absent += 1
            continue
        out.append(r)
        cov.append(c)
        if i % 25 == 0:
            print(f'  {i}/{len(ss)}  {len(out)} sessions done', flush=True)

    allseq = pd.concat(out, ignore_index=True)
    allseq.to_parquet(OUT)
    print(f'\n{len(allseq)} trial-epoch-paw rows, {allseq.session.nunique()} sessions, '
          f'{allseq.mouse_name.nunique()} mice -> {OUT}')
    print(f'{absent} sessions without lightningAction / camera times')
    print(f'frames matched to a bin: median {np.median(cov):.1%}, min {np.min(cov):.1%}')

    # the alignment check: the classifier is never told about the task
    x = allseq.copy()
    x['occ'] = x['binned_sequence'].apply(
        lambda a: np.bincount(np.asarray(a, int), minlength=len(STATES)) / len(a))
    occ = pd.DataFrame(np.stack(x['occ']), columns=STATES)
    occ['epoch'] = x['broader_label'].to_numpy()
    print('\nmean state occupancy per epoch (should be still-dominated in Quiescence,')
    print('wheel_turn-dominated in Choice):')
    print(occ.groupby('epoch')[STATES].mean().round(3).to_string())


if __name__ == '__main__':
    main()
