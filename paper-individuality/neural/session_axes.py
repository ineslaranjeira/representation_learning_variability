"""
SESSION-LEVEL BEHAVIOURAL AXES -- alternatives to LDA 1 as the neural predictor
==============================================================================
`lda1_neural_metrics.run_tests(predictor=...)` will test any per-session column against the
three neural metrics. This module builds the behavioural columns worth testing, so the
question "is the LDA-1 effect really reaction time, or engagement?" can be asked with the
same statistics, the same cohort and the same cached tables as the LDA-1 grid.

WHY THESE AXES. The LDA-1 result is that high LDA 1 goes with lower Fano-factor quench and
lower noise correlation. Two ordinary explanations are that such sessions are faster
(reaction time) or less prone to switching out of the engaged state (GLM-HMM). Both are
measurable per session from the trial table, so both can be put on the x axis in place of
LDA 1 and the neural grid re-run.

SCREENING FIRST, AND IT MATTERS. An axis that does not correlate with LDA 1 cannot be what
LDA 1 was measuring, whatever it does to the neural metrics on its own. Measured against
`mouse_LDA_5_bins_raw_25_20-09-2026` over 237 shared sessions:

    switch_rate   -0.281        rt_median   -0.240        rt_mean   -0.158
    p_engaged     +0.135        frac_engaged +0.021       rt_sd     +0.016

so switch_rate and rt_median are the live candidates and the last two are already ruled out
as mediators. `axis_vs_lda()` reprints that table for whatever embedding is current.

REACTION TIME IS log1p, NOT log. Session reaction times in this file run from -0.623 s (a
first movement before the go cue) to 60 s. `np.log` of a negative is NaN, which would
silently drop the session; log1p is finite and monotone for everything above -1. This
matches build_design_matrix in 4_mice/functions.py, which takes the MEDIAN first and then
log1p -- done the same way here so the two files cannot disagree.

THE ENGAGEMENT AXES ARE NEAR CEILING. p_state1 has median 0.925 and frac_engaged reaches
0.96 by the 75th percentile with many sessions at exactly 1.0. The compressed range costs
power, so a null on engagement is weak evidence of absence -- say so rather than reporting
it as "engagement does not matter".

    import session_axes as sa
    ax = sa.session_axes()              # one row per session
    tables = sa.attach_axes(tables, ax) # adds the columns to every metric table
    nm.run_tests(tables, cfg, predictor='switch_rate')
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
PAPER = ROOT.parent
TRIAL_META = [PAPER / 'data/session_trial_meta_10-07-2026',
              PAPER / 'data/session_trial_meta_19-08-2026']
CACHE = ROOT / 'lda1_tables' / 'session_axes.pqt'

AXES = ['rt_median', 'rt_mean', 'rt_sd', 'p_engaged', 'frac_engaged', 'switch_rate',
        'performance', 'n_trials_beh']


def _first_existing(paths):
    for p in paths:
        if pathlib.Path(p).exists():
            return pathlib.Path(p)
    raise FileNotFoundError(f'none of these exist: {[str(p) for p in paths]}')


def session_axes(path=None, min_trials=40, cache=True, verbose=True):
    """One row per session, indexed by `session`, carrying every column in AXES.

    `min_trials` drops sessions too short for a stable median -- the same floor
    lda1_neural_metrics uses for a window estimate, so the two agree on what a usable
    session is.
    """
    if cache and CACHE.exists():
        ax = pd.read_parquet(CACHE)
        if verbose:
            print(f'session axes: {len(ax)} sessions from cache {CACHE.name}')
        return ax

    f = _first_existing(path and [path] or TRIAL_META)
    tm = pd.read_parquet(f)
    if verbose:
        print(f'session axes from {f.name}: {len(tm)} trials, {tm.session.nunique()} sessions')

    # feedback is a string in some files and numeric in others -- normalise before averaging
    fb = tm['feedback']
    tm = tm.assign(_correct=(fb == 'correct').astype(float) if fb.dtype == object
                   else fb.astype(float))
    # the GLM-HMM here has two states; 'engaged' is state1, the high-p_state1 one
    tm = tm.assign(_engaged=(tm['dominant_state'] == 'state1').astype(float))
    tm = tm.sort_values(['session', 'trial_id'])

    def _switch(s):
        """State transitions per trial: how often the animal changes engagement state."""
        v = s.to_numpy(dtype=float)
        return np.abs(np.diff(v)).sum() / max(len(v) - 1, 1)

    g = tm.groupby('session')
    ax = pd.DataFrame({
        'rt_median':    np.log1p(g['reaction'].median()),
        'rt_mean':      np.log1p(g['reaction'].mean()),
        'rt_sd':        g['reaction'].std(),
        'p_engaged':    g['p_state1'].mean(),
        'frac_engaged': g['_engaged'].mean(),
        'switch_rate':  g['_engaged'].apply(_switch),
        'performance':  g['_correct'].mean(),
        'n_trials_beh': g.size().astype(float),
    })
    short = ax['n_trials_beh'] < min_trials
    if short.any() and verbose:
        print(f'  dropped {short.sum()} session(s) with < {min_trials} trials')
    ax = ax.loc[~short]
    if cache:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        ax.reset_index().to_parquet(CACHE, index=False)
        if verbose:
            print(f'  wrote {CACHE.name}: {len(ax)} sessions')
    return ax.reset_index()


def attach_axes(tables, ax=None, cols=None, verbose=True):
    """Add the axis columns to every metric table, by session. Existing columns are kept.

    Mirrors lda1_neural_metrics.attach_lda: it adds columns only and never overwrites one
    that is already there, so a cached predictor cannot be silently replaced.
    """
    ax = session_axes(verbose=False) if ax is None else ax
    cols = cols or [c for c in AXES if c in ax.columns]
    maps = {c: dict(zip(ax['session'], ax[c])) for c in cols}
    out, added = dict(tables), set()
    for k, t in out.items():
        if not isinstance(t, pd.DataFrame) or 'session' not in t.columns:
            continue
        t = t.copy()
        for c, m in maps.items():
            if c not in t.columns:
                t[c] = t['session'].map(m)
                added.add(c)
        out[k] = t
    if verbose and added:
        print(f'  attached {", ".join(sorted(added))}')
    return out


def axis_vs_lda(lda_file, ax=None, cols=None):
    """Correlation of each axis with LDA 1 -- the screen for whether it could be the cause."""
    from scipy.stats import pearsonr
    ax = session_axes(verbose=False) if ax is None else ax
    lda = pd.read_pickle(lda_file).rename(columns={0: 'LD1'})
    j = ax.merge(lda[['session', 'LD1']], on='session')
    rows = []
    for c in (cols or [x for x in AXES if x in ax.columns]):
        d = j[[c, 'LD1']].dropna()
        r, p = pearsonr(d[c], d['LD1'])
        rows.append({'axis': c, 'r_with_LD1': r, 'p': p, 'n': len(d)})
    return pd.DataFrame(rows).sort_values('r_with_LD1', key=abs, ascending=False)
