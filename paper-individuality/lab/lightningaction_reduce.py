"""
REDUCE lightningAction TO WHAT IS ACTUALLY NEEDED, AND ALIGN IT TO THE TRIALS
=============================================================================
The downloaded pawstates files are 45-110 MB each because they carry a five-way posterior
AND a five-way ensemble variance per paw per frame. The analysis needs one integer per paw
per frame -- the argmax state. Reducing first turns ~12 GB into ~150 MB, which matters on a
disk that is already 93% full.

Everything else comes from the files we already have, as you said:
  * `data/states_files/<k>_states_file_<eid>_<mouse>` carries `Bin` (absolute seconds on the
    same 1/60 s grid the design matrices use), `trial_id`, `label` and `broader_label`;
  * `_ibl_leftCamera.times.npy` (already cached for >1000 sessions) maps camera FRAME ->
    absolute seconds.
So frame -> time -> Bin -> trial and epoch, with no further downloads.

ALIGN BY TIMESTAMP, NOT BY INDEX. Camera frames drop, and the states_file grid starts at the
first design-matrix bin rather than at t=0, so frame i and row i are not the same moment. The
two are matched with searchsorted on the timestamps and frames more than half a bin from any
bin are discarded rather than snapped to a neighbour.

STATES: 0 background, 1 still, 2 move, 3 wheel_turn, 4 groom (order as stored).
"""
import os
import sys
import glob
import pathlib
import argparse
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / '4_mice'), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
STATES_DIR = ROOT / 'data' / 'states_files'
STATES = ['background', 'still', 'move', 'wheel_turn', 'groom']
PAWS = ['paw_r', 'paw_l']
BIN = 1 / 60.0


def reduce_session(d, cam='left', delete=False):
    """45 MB of posteriors -> one uint8 array per paw, saved next to them as .npz."""
    src = os.path.join(d, f'_ibl_{cam}Camera.pawstates.pqt')
    dst = os.path.join(d, f'pawstates_argmax_{cam}.npz')
    if os.path.exists(dst) and not delete:
        return dst, 0
    if not os.path.exists(src):
        return None, 0
    df = pd.read_parquet(src, columns=[f'{p}_{s}' for p in PAWS for s in STATES])
    out = {p: np.argmax(df[[f'{p}_{s}' for s in STATES]].to_numpy(), axis=1).astype(np.uint8)
           for p in PAWS}
    np.savez_compressed(dst, **out)
    freed = os.path.getsize(src)
    if delete:
        os.remove(src)
    return dst, freed


def align_to_trials(eid, mouse, d, cam='left', n_bins=10,
                    epochs=('Pre-quiescence', 'Quiescence', 'Choice', 'ITI')):
    """Per-trial, per-epoch 10-bin state sequences, same shape as the syllable file."""
    npz = os.path.join(d, f'pawstates_argmax_{cam}.npz')
    times_f = os.path.join(d, os.pardir, f'_ibl_{cam}Camera.times.npy')
    sf = glob.glob(str(STATES_DIR / f'*_states_file_{eid}_{mouse}'))
    if not (os.path.exists(npz) and os.path.exists(times_f) and sf):
        return None
    z = np.load(npz)
    t = np.load(times_f)
    st = pd.read_parquet(sf[0], columns=['Bin', 'trial_id', 'broader_label'])
    n = min(len(t), len(z[PAWS[0]]))
    t = t[:n]

    bins = st['Bin'].to_numpy()
    pos = np.searchsorted(bins, t)
    pos = np.clip(pos, 1, len(bins) - 1)
    left, right = bins[pos - 1], bins[pos]
    take = np.where(np.abs(t - left) < np.abs(t - right), pos - 1, pos)
    ok = np.isfinite(t) & (np.abs(t - bins[take]) <= BIN / 2)   # no snapping across a gap

    rows = []
    for p in PAWS:
        s = z[p][:n]
        df = pd.DataFrame({'state': s[ok], 'trial_id': st['trial_id'].to_numpy()[take[ok]],
                           'epoch': st['broader_label'].to_numpy()[take[ok]]}).dropna()
        for (tid, ep), g in df.groupby(['trial_id', 'epoch']):
            if ep not in epochs or len(g) < n_bins:
                continue
            chunks = np.array_split(g['state'].to_numpy(), n_bins)
            seq = [np.bincount(c, minlength=len(STATES)).argmax() for c in chunks]
            rows.append(dict(session=eid, mouse_name=mouse, paw=p, trial_id=tid,
                             broader_label=ep, binned_sequence=np.array(seq, dtype=np.uint8)))
    return pd.DataFrame(rows), float(ok.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delete', action='store_true',
                    help='remove the big pawstates parquet after reducing it')
    ap.add_argument('--align', action='store_true', help='also write trial-aligned sequences')
    args = ap.parse_args()

    dirs = sorted(glob.glob(CACHE + '*/Subjects/*/*/*/alf/lightningaction'))
    done = freed = 0
    for d in dirs:
        dst, f = reduce_session(d, 'left', delete=args.delete)
        if dst:
            done += 1
            freed += f
    print(f'reduced {done} sessions; {freed/1e9:.2f} GB '
          f'{"freed" if args.delete else "would be freed with --delete"}')

    if args.align:
        out = []
        for d in dirs:
            q = d.split('/')
            eid_dir = os.path.join(*q[:-1])
            r = align_to_trials(_eid_from_path(d), q[-5], d)
            if r is not None:
                out.append(r[0])
        if out:
            allseq = pd.concat(out, ignore_index=True)
            allseq.to_parquet(HERE / 'lightningaction_binned_sequences.pqt')
            print(f'{len(allseq)} trial-epoch rows -> lightningaction_binned_sequences.pqt')


def _eid_from_path(d):
    from one.api import ONE
    global _ONE
    try:
        _ONE
    except NameError:
        _ONE = ONE(mode='local')
    q = d.split('/')
    return _ONE.path2eid('/'.join(q[:-2]))


if __name__ == '__main__':
    main()
