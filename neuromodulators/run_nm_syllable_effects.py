"""
Every fiber of the export through Functions/nm_syllables.py, cached for 7_nm_syllable_effects.ipynb.

    python neuromodulators/run_nm_syllable_effects.py [n_jobs] [comma-separated analyses, default all]

Writes to paper-individuality/data/neuromodulators/syllables_photometry/ (pickles: the cells hold arrays):
    effects_task_<RUN>.pkl, effects_extended_<RUN>.pkl   ns.effect_maps
    joint_<RUN>.pkl                                      ns.joint_maps
    variance_<RUN>.pkl                                   ns.variance_partition
    task_vs_vigor_<RUN>.pkl                              ns.task_vs_movement (vigor only, outcomes pooled)
    shapes_<RUN>.pkl                                     ns.shape_analyses (CCA + RRR, unmasked trial shapes)
"""
import pathlib
import sys
import warnings

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from Functions import nm_syllables as ns  # noqa: E402

CACHE = ns.DATA / 'syllables_photometry'
ANALYSES = {'effects_task': lambda f: ns.effect_maps(f, covariates='task'),
            'effects_extended': lambda f: ns.effect_maps(f, covariates='extended'),
            'joint': ns.joint_maps,
            'variance': ns.variance_partition,
            'task_vs_vigor': ns.task_vs_movement}


def one_session(eid, only):
    warnings.filterwarnings('ignore')
    out = {k: [] for k in only if k != 'shapes'}
    try:
        if out:
            for fiber in ns.load_session(eid):
                for k in out:
                    out[k].append(ANALYSES[k](fiber))
        if 'shapes' in only:
            out['shapes'] = [r for f in ns.load_session(eid, mask=False) if (r := ns.shape_analyses(f)) is not None]
    except Exception as e:  # report, keep going
        print(f'{eid}: {type(e).__name__}: {e}', flush=True)
    return out


if __name__ == '__main__':
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else list(ANALYSES) + ['shapes']
    rec = ns.load_recordings()
    eids = rec['eid'].unique()
    print(f'{len(rec)} fibers, {len(eids)} sessions', flush=True)
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one_session)(e, only) for e in eids)
    if 'shapes' in only:
        shapes = [s for r in res for s in r.get('shapes', [])]
        table = pd.DataFrame([row for row, _ in shapes])
        table.attrs.update(y_labels=shapes[0][1][0], x_labels=shapes[0][1][1], windows=ns.SHAPE_WINDOWS,
                           bin=ns.SHAPE_BIN)
        table.to_pickle(CACHE / f'shapes_{ns.RUN}.pkl')
        print('shapes', table.shape, flush=True)
    for k in [k for k in only if k != 'shapes']:
        parts = [t for r in res for t in r[k]]
        table = pd.concat(parts, ignore_index=True)
        table.attrs = parts[0].attrs
        table.to_pickle(CACHE / f'{k}_{ns.RUN}.pkl')
        print(k, table.shape, flush=True)
