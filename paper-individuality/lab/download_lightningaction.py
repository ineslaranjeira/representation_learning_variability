"""
FETCH lightningAction pawstates FOR THE LDA SESSION SET
=======================================================
LEFT CAMERA ONLY by default: it is what the lab-bias analysis uses, the right-camera files
are ~2.4x larger, and the machine had 63 GB free when this was written. Pass --both to take
the right camera too (needed only for the cross-camera paw check, which is already done on
45 sessions).

Resumable: a session whose file is already in the ONE cache is skipped, so re-running after
an interruption costs nothing. Measured rate on this machine: ~7 MB/s sustained, ~50 MB per
left-camera file, so the full set is roughly 12 GB / 30 minutes.
"""
import os
import sys
import time
import pathlib
import argparse

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / '4_mice'), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import functions
import lightningaction_reduce as R

CACHE = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--both', action='store_true', help='also fetch the right camera')
    ap.add_argument('--reduce', action='store_true',
                    help='reduce each file to argmax uint8 and DELETE the parquet, so the '
                         'peak disk cost is one file rather than the whole set')
    args = ap.parse_args()
    cams = ['left'] + (['right'] if args.both else [])

    ss, _ = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    from one.api import ONE
    one = ONE()
    print(f'{len(ss)} LDA sessions, cameras: {cams}', flush=True)

    got = skipped = absent = failed = reduced = 0
    reclaimed = 0
    t0 = time.time()
    total_bytes = 0
    for i, eid in enumerate(ss.index, 1):
        p = one.eid2path(eid)
        if p is None:
            absent += 1
            continue
        q = str(p).split('/')
        d = os.path.join(CACHE, q[-5], 'Subjects', q[-3], q[-2], q[-1], 'alf', 'lightningaction')
        for cam in cams:
            fn = f'_ibl_{cam}Camera.pawstates.pqt'
            if os.path.exists(os.path.join(d, fn)):
                skipped += 1
                if args.reduce:      # reduce files that were already on disk too
                    _, _f = R.reduce_session(d, cam, delete=True)
                    reduced += 1
                    reclaimed += _f
                continue
            try:
                fp = one.load_dataset(eid, fn, collection='alf/lightningaction',
                                      download_only=True)
                total_bytes += os.path.getsize(fp)
                got += 1
                if args.reduce:
                    _, _f = R.reduce_session(d, cam, delete=True)
                    reduced += 1
                    reclaimed += _f
            except Exception as e:
                # most of these are "dataset does not exist for this session", not errors
                # ALFObjectNotFound = this session simply has no file for that camera
                # (plenty have only the right one); that is absence, not a failure
                if type(e).__name__ == 'ALFObjectNotFound' or 'not found' in str(e).lower():
                    absent += 1
                else:
                    failed += 1
                    print(f'  !! {eid[:8]} {cam}: {type(e).__name__}: {str(e)[:90]}', flush=True)
        if i % 10 == 0:
            el = time.time() - t0
            rate = total_bytes / 1e6 / el if el else 0
            print(f'  {i}/{len(ss)}  downloaded {got} ({total_bytes/1e9:.1f} GB) '
                  f'skipped {skipped} absent {absent} failed {failed}  '
                  f'{rate:.1f} MB/s  elapsed {el/60:.1f} min'
                  + (f'  | reduced {reduced}, {reclaimed/1e9:.1f} GB reclaimed' if args.reduce else ''),
                  flush=True)

    print(f'\nDONE: {got} files ({total_bytes/1e9:.2f} GB) in {(time.time()-t0)/60:.1f} min; '
          f'{skipped} already present, {absent} not available, {failed} failed'
          + (f'; reduced {reduced} files, {reclaimed/1e9:.2f} GB reclaimed' if args.reduce else ''))


if __name__ == '__main__':
    main()
