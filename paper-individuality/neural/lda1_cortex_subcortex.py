"""
DOES LD1 RELATE DIFFERENTLY TO CORTICAL AND SUBCORTICAL ACTIVITY?
================================================================
Steinmetz et al. (2019, Nature) found engagement-related pre-stimulus activity with opposite signs in neocortex
(lower in the task context) and outside it (higher). If LD1 is an engagement-like trait, its relationship with
firing could follow the same split -- and a brain-wide null for firing rate could hide two opposite effects.

For every test of fig_neural (firing rate, Fano factor, noise correlation x baseline / post-stimulus / change):

  cortex, subcortex   the fig_neural test (lda1_neural_metrics.age_effect_test: partial r of LD1 with region and
                      trial-count [+ neuron-count] covariates, LD1 permuted across sessions) on the neocortical rows
                      only, and on all other rows
  interaction         ONE model on all rows: y ~ LD1 + LD1 x cortex + region + covariates. The cortex main effect
                      is absorbed by the region dummies. Statistic: the interaction slope (does LD1's slope differ
                      between neocortex and the rest?); p from the same session permutation of LD1, applied to both
                      the LD1 and the LD1 x cortex columns at once.

"Cortex" = Cosmos 'Isocortex' (neocortex), as in Steinmetz et al.; hippocampal formation, olfactory areas and
cortical subplate count as non-neocortical. Same tables, cohort and LD1 as fig_neural (the newest
lda1_neural_pr90_fr10_sess_<date> build, picked by the date in its name).

Writes results/lda1_cortex_subcortex_<date>.csv.
"""
import re
import sys
import pathlib
from datetime import date

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent
for p in [str(PI), str(HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    import numpy._core.numeric  # noqa: F401
except ModuleNotFoundError:     # the LDA file was pickled under numpy 2
    import numpy.core.numeric, numpy.core.multiarray
    sys.modules['numpy._core.numeric'], sys.modules['numpy._core.multiarray'] = np.core.numeric, np.core.multiarray
import lda1_neural_metrics as nm
import run_behavior_axes as rba
from iblatlas.regions import BrainRegions

N_PERM = 2000
SEED = 0
CORTEX = {'Isocortex'}


def load():
    cfg = rba.config()
    want = nm.cache_tag(cfg)
    builds = sorted({f.name.split('__')[0] for f in nm.CACHE.glob(f'{want}_*__ff.pqt')
                     if re.fullmatch(r'\d{2}-\d{2}-\d{4}', f.name.split('__')[0][len(want) + 1:])},
                    key=lambda b: pd.to_datetime(b[-10:], dayfirst=True))
    T = nm.apply_cohort(nm.load_tables(tag=builds[-1], verbose=False), cfg, 'common', verbose=False)
    br = BrainRegions()
    regs = pd.unique(pd.concat([T[k]['region'] for k in ('fr', 'ff', 'rsc')]).dropna())
    cosmos = dict(zip(regs, br.acronym2acronym(regs, mapping='Cosmos')))
    for k in ('fr', 'ff', 'rsc'):
        T[k]['cortex'] = T[k]['region'].map(cosmos).isin(CORTEX)
    print(f'{builds[-1]}: ' + ', '.join(f"{k} {len(T[k])} rows ({T[k]['cortex'].mean():.0%} cortical)"
                                        for k in ('fr', 'ff', 'rsc')))
    return T, cfg


def interaction_test(df, y_col, covars, rng):
    """Slope of LD1 x cortex in y ~ LD1 + LD1 x cortex + region + covars; LD1 permuted across sessions."""
    d = df.dropna(subset=[y_col, 'lda_1', 'region'] + list(covars)).reset_index(drop=True)
    y = d[y_col].to_numpy(float)
    cx = d['cortex'].to_numpy(float)
    Z = np.column_stack([np.ones(len(d)), pd.get_dummies(d['region'], drop_first=True).to_numpy(float),
                         d[list(covars)].to_numpy(float)])
    sess = d.groupby('session')['lda_1'].first()
    row = d['session'].map({s: i for i, s in enumerate(sess.index)}).to_numpy()

    # Frisch-Waugh-Lovell in two steps, so a permutation costs two projections instead of a new pseudo-inverse:
    # residualise everything on Z once, then take the LD1 main effect (a nuisance here) out of the interaction
    Zp = np.linalg.pinv(Z)
    yz = y - Z @ (Zp @ y)

    def slope(x):
        inter = x * cx
        xz, iz = x - Z @ (Zp @ x), inter - Z @ (Zp @ inter)
        ri = iz - xz * (xz @ iz) / (xz @ xz)
        ry = yz - xz * (xz @ yz) / (xz @ xz)
        return (ri @ ry) / (ri @ ri), np.corrcoef(ri, ry)[0, 1]

    x = sess.to_numpy()[row]
    obs, r_obs = slope(x)
    null = np.array([slope(rng.permutation(sess.to_numpy())[row])[0] for _ in range(N_PERM)])
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + N_PERM)
    return dict(inter_slope=obs, inter_r=r_obs, inter_p=p, n_sessions=len(sess),
                n_sessions_cortex=d.loc[d['cortex'], 'session'].nunique(),
                n_sessions_sub=d.loc[~d['cortex'], 'session'].nunique())


def main():
    T, cfg = load()
    rng = np.random.default_rng(SEED)
    rows = []
    for (metric, test), spec in nm.GRID.items():
        if spec is None or test == 'Response onset':      # onset is not in fig_neural
            continue
        tbl, y_col, _, _, region_col, covars = spec
        df = T[tbl]
        out = dict(metric=metric, test=test)
        for name, part in [('cortex', df[df['cortex']]), ('subcortex', df[~df['cortex']])]:
            r = nm.age_effect_test(part, y_col, predictor='lda_1', region_col=region_col, extra_covars=covars,
                                   n_perm=N_PERM, seed=SEED)
            out.update({f'{name}_r': r['r_partial'], f'{name}_p': r['p_perm'], f'{name}_n': r['n'],
                        f'{name}_sessions': r['n_sessions']})
        out.update(interaction_test(df, y_col, covars, rng))
        rows.append(out)
        print(f"  {metric:<18} {test:<19} cortex r = {out['cortex_r']:+.3f} (p {out['cortex_p']:.3f}, "
              f"{out['cortex_sessions']} sess) | subcortex r = {out['subcortex_r']:+.3f} (p {out['subcortex_p']:.3f}, "
              f"{out['subcortex_sessions']} sess) | difference p = {out['inter_p']:.3f}")
    t = pd.DataFrame(rows)
    for c in ['cortex_p', 'subcortex_p', 'inter_p']:
        t[c.replace('_p', '_q')] = t.groupby('metric')[c].transform(lambda s: multipletests(s, method='fdr_bh')[1])
    out = PI / 'results' / f'lda1_cortex_subcortex_{date.today().strftime("%d-%m-%Y")}.csv'
    t.to_csv(out, index=False)
    print(f'\nwrote {out.relative_to(PI)}  (q = BH within metric, as fig_neural)')
    return t


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
