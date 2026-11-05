"""
Decompose every movement of the given sessions into minimum-jerk primitives and cache the tables:
    paper-individuality/data/training/wheel_primitives/{movements,primitives}_{eid}_{mouse}
Usage: python cache_primitives.py [--fml] MOUSE [MOUSE ...]
    all training sessions of each mouse, or with --fml only its first, middle and last
"""
import sys
import pandas as pd
from joblib import Parallel, delayed
import wheel_states as ws
import primitives as pr

OUT = ws.DM_DIR.parent / 'wheel_primitives'


def run(eid, mouse):
    f = OUT / f'primitives_{eid}_{mouse}'
    if f.exists():
        return
    from one.api import ONE
    one = ONE(mode='remote')
    dm, trials = ws.load_session(eid, mouse)
    iv = one.load_object(eid, 'wheelMoves', collection='alf')['intervals']
    mv, p = pr.session_primitives(dm.times.values, dm.velocity.values.astype(float), trials, iv)
    mv.to_parquet(OUT / f'movements_{eid}_{mouse}')
    p.to_parquet(f)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    man = ws.manifest()
    mice = [a for a in sys.argv[1:] if not a.startswith('--')]
    man = man[man.mouse.isin(mice)].sort_values(['mouse', 'session_index'])
    if '--fml' in sys.argv:
        man = man.groupby('mouse', group_keys=False).apply(lambda m: m.iloc[sorted({0, len(m) // 2, len(m) - 1})])
    Parallel(n_jobs=8, verbose=5)(delayed(run)(e, m) for e, m in zip(man.eid, man.mouse))
