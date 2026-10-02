"""
IS A NEAR-EMPTY STATE A PROPERTY OF THE SESSION, THE MOUSE, THE RIG, OR THE METHOD?
For each variant, take the state that is rarest overall and ask of its per-session occupancy:
  * how many sessions have it below 1%              (the method's floor)
  * ICC(1) by mouse                                  (is it repeatable within a mouse = individual)
  * lab eta^2, and its excess over a null that shuffles lab across mice   (is it the rig)
Same cohort as compare_pipelines.py section 2 (LDA cohort, 260 sessions, 56 mice).
Reads compare_pipelines_features.npz.
"""
import sys, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lda'))
import make_embedding as me                                 # noqa: E402
from functions import lab_labels                            # noqa: E402

occ = np.load(HERE / 'compare_pipelines_features.npz', allow_pickle=True)['occ'].item()
ref, mouse_of = me.build_design_matrix(me.SYLLABLE_FILE)
idx = [s for s in ref.index if all(s in occ[v] for v in occ)]
mice = mouse_of.loc[idx].to_numpy()
labs = np.array(list(lab_labels(pd.Index(idx), mouse_names=mouse_of.loc[idx], verbose=False)))
rng = np.random.default_rng(0)


def icc1(v, g):
    s = pd.Series(v).groupby(g)
    k = s.size().mean()
    msb, msw = s.mean().var(ddof=1) * k, s.var(ddof=1).mean()
    return (msb - msw) / (msb + (k - 1) * msw)


def eta(v, g):
    s = pd.Series(v).groupby(g)
    return ((s.mean() - v.mean()) ** 2 * s.size()).sum() / ((v - v.mean()) ** 2).sum()


m_lab = pd.Series(labs, index=mice).groupby(level=0).first()
print(f'{len(idx)} sessions, {len(set(mice))} mice\n')
print(f'{"variant":44s} {"state":>5s} {"share":>6s} {"<1%":>5s} {"ICC mouse":>9s} '
      f'{"lab eta2":>8s} {"null":>6s} {"p":>6s}')
for v in occ:
    O = np.vstack([occ[v][s] for s in idx])
    for st in np.argsort(O.mean(0))[:2]:                 # the two rarest states
        x = O[:, st]
        obs = eta(x, labs)
        null = np.array([eta(x, pd.Series(rng.permutation(m_lab.to_numpy()),
                                          index=m_lab.index).reindex(mice).to_numpy())
                         for _ in range(2000)])
        p = (1 + (null >= obs).sum()) / 2001
        print(f'{v:44s} {st:5d} {x.mean():6.3f} {(x < .01).mean():5.2f} {icc1(x, mice):9.2f} '
              f'{obs:8.3f} {null.mean():6.3f} {p:6.3f}')
    print()
