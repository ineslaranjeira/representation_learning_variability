"""Fano factor vs session accuracy, with an explicit test for an inverted-U relationship.

Companion to ff_psth_ldabin.ipynb (FF vs LDA-1) and ff_vs_engagement_bouts.ipynb (FF vs engagement
bout structure). Same single-neuron window Fano factor and the same session-level permutation
machinery, with session accuracy as the predictor.

WHY THE INVERTED-U NEEDS ITS OWN MACHINERY. A significant quadratic term is weak evidence for a
U-shape and is the single most over-read statistic in this literature. Three things produce one
without any underlying non-monotonicity:

  1. BOUNDED-SCALE COMPRESSION. Accuracy lives in [0, 1] and these sessions sit at a median of 0.82,
     so the top of the range is compressed. If FF is monotone in the LATENT discriminability that
     accuracy reports, the relationship on the probability scale must curve. Every test below is
     therefore run on both accuracy and logit(accuracy); a quadratic that survives on the raw scale
     but dies on the logit scale is a scale artefact, not a finding. This is the first thing to look
     at in the output.
  2. A VERTEX OUTSIDE THE DATA. A fitted parabola whose turning point lies beyond the observed
     accuracy range is a monotone curve over the data, however significant its quadratic term. The
     vertex location and whether it falls inside the range are reported for every fit
     (the Lind & Mehlum / Sasabuchi condition).
  3. OUTLIERS AND UNEVEN DENSITY. A handful of low-accuracy sessions can bend a parabola on their
     own. Hence the two-lines test (Simonsohn 2018) -- separate slopes either side of the breakpoint,
     which requires BOTH to be significant with OPPOSITE signs -- and a non-parametric quintile
     profile that shows the shape without assuming one.

WHY ENGAGEMENT IS THE CONFOUND THAT MATTERS. Session accuracy correlates r = +0.54 with the fraction
of trials in the engaged GLM-HMM state, and ff_vs_engagement_bouts.ipynb shows engagement dynamics
track FF. So an accuracy effect may be an engagement effect wearing a different label. Every test is
therefore reported with and without frac_engaged as a covariate, and repeated on engaged trials only.

Run: python ff_vs_accuracy.py            (figures land in ./ff_vs_accuracy_figs/)
"""
import os, pickle, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, t as tdist
import pingouin as pg
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid'); plt.rcParams['figure.facecolor'] = 'white'

# ------------------------------------------------------------------ config
PREFIX = '/home/ines/repositories/representation_learning_variability/paper-individuality/'
FR_DIR = PREFIX + 'data/firing_rates/'
CLU_DIR = PREFIX + 'clustering/data_files/'
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ff_vs_accuracy_figs')

DROP = ['root', 'void']
MIN_NEURONS = 15
PRE_WINDOW, POST_WINDOW = (-0.2, 0.0), (0.1, 0.3)
MIN_TRIALS = 40
MIN_WINDOW_COUNT = 0.5
REMOVE_CONDITION = True
N_PERM = 2000            # session-level permutation iterations
MOUSE_PERM = 600         # within-mouse permutation iterations
N_BINS_PROFILE = 5       # accuracy quantile bins for the non-parametric profile
SEED = 0

# RT matching, identical to ff_psth_ldabin.ipynb's current defaults
RT_MATCH_BINS, RT_MATCH_MAX = 8, 1.0
ENGAGED_P_MIN = 0.5
MIN_TRIALS_ENGAGED = 40

# ------------------------------------------------------------------ behaviour
lda = pd.read_pickle(CLU_DIR + 'mouse_LDA_5_bins_cut19-08-2026').rename(columns={0: 'lda_1'})
lda1_map = dict(zip(lda['session'], lda['lda_1']))
trials_df = pd.read_parquet(PREFIX + '4_mice/session_trial_meta_19-08-2026')
trials_df = trials_df[trials_df['session'].isin(set(lda['session']))].copy()
trials_df['correct'] = (trials_df['feedback'] == 'correct').astype(float)
trials_df['abs_contrast'] = trials_df['signed_contrast'].abs()
trials_df['engaged'] = trials_df['p_state1'] > ENGAGED_P_MIN

# Zero-contrast trials are excluded from every accuracy measure: at 0% there is no stimulus-defined
# correct side, so `feedback` there reports the task's random reward assignment rather than the
# animal's discrimination, and its prevalence differs between sessions.
_nz = trials_df[trials_df['abs_contrast'] > 0]
_easy = _nz[_nz['abs_contrast'] >= 0.5]      # the contrast set is {0, .0625, .125, .25, 1}: easy == 1.0
_hard = _nz[_nz['abs_contrast'] <= 0.125]

# Contrast-standardised accuracy: mean of the per-contrast accuracies reweighted by ONE global
# contrast distribution, so sessions are not separated by their own contrast mix. (That mix barely
# varies here -- the easy fraction spans 0.19-0.25 -- so this should track raw accuracy closely; it
# is computed anyway because "closely" is an empirical claim, printed below.)
_w = _nz.groupby('abs_contrast').size() / len(_nz)
_per = _nz.groupby(['session', 'abs_contrast'])['correct'].mean().unstack()
acc_std = (_per * _w).sum(axis=1, min_count=1) / _per.notna().mul(_w, axis=1).sum(axis=1)

g = _nz.groupby('session')
sess = pd.DataFrame({
    'n_trials_beh': g.size(),
    'acc': g['correct'].mean(),
    'acc_std': acc_std,
    'acc_easy': _easy.groupby('session')['correct'].mean(),
    'acc_hard': _hard.groupby('session')['correct'].mean(),
    'acc_engaged': _nz[_nz['engaged']].groupby('session')['correct'].mean(),
    'frac_engaged': trials_df.groupby('session')['engaged'].mean(),
    'frac_easy': _nz.assign(e=_nz['abs_contrast'] >= 0.5).groupby('session')['e'].mean(),
    'mouse_name': g['mouse_name'].first(),
}).reset_index()
sess['lda_1'] = sess['session'].map(lda1_map)
# logit is the scale on which a bounded proportion behaves linearly; the clip only guards the 0/1
# edge and touches nothing at these accuracies (min 0.63, max 0.95 for `acc`).
for c in ['acc', 'acc_std', 'acc_easy', 'acc_hard', 'acc_engaged']:
    sess['logit_' + c] = np.log(sess[c].clip(1e-3, 1 - 1e-3) / (1 - sess[c].clip(1e-3, 1 - 1e-3)))

# ------------------------------------------------------------------ RT matching (as in ff_psth_ldabin)
_rt = trials_df[['session', 'trial_id', 'reaction']].copy()
_pos = (_rt['reaction'] > 0) & (_rt['reaction'] < RT_MATCH_MAX)
_edges = np.quantile(np.log(_rt.loc[_pos, 'reaction']), np.linspace(0, 1, RT_MATCH_BINS + 1))
_edges[0], _edges[-1] = -np.inf, np.inf
_rt['rt_bin'] = -1
_rt.loc[_pos, 'rt_bin'] = np.searchsorted(_edges, np.log(_rt.loc[_pos, 'reaction']), side='right') - 1
rt_bin_by_session = {s: gg.set_index('trial_id')['rt_bin'] for s, gg in _rt.groupby('session')}
eng_by_session = {s: gg.set_index('trial_id')['engaged'] for s, gg in trials_df.groupby('session')}


def selections(session, trials):
    """Three trial selections per session, as boolean masks over `trials`.

    'all'  - every trial.
    'rt'   - equal trials per global log-RT quantile bin: identical RT distribution in every session.
    'rand' - same eligible pool and same trial COUNT as 'rt', but the session's own RT distribution.
             The power control: 'rt' vs 'rand' separates "the effect was RT" from "the effect was
             the trials matching discarded". See ff_psth_ldabin.ipynb.
    """
    out = {'all': np.ones(len(trials), dtype=bool)}
    rb = rt_bin_by_session.get(session)
    if rb is None:
        out['rt'] = out['rand'] = np.zeros(len(trials), dtype=bool)
        return out
    b = pd.to_numeric(rb.reindex(trials), errors='coerce').to_numpy()
    b = np.where(np.isfinite(b), b, -1).astype(int)
    per = [np.where(b == k)[0] for k in range(RT_MATCH_BINS)]
    n_per = min(len(p) for p in per)
    key = np.array([int(hashlib.md5(f"{session}|{t}|{SEED}".encode()).hexdigest()[:8], 16)
                    for t in trials])
    m = np.zeros(len(trials), dtype=bool)
    for p in per:
        if n_per:
            m[p[np.argsort(key[p])[:n_per]]] = True
    out['rt'] = m
    elig = np.concatenate(per) if per else np.array([], dtype=int)
    n_take = min(n_per * RT_MATCH_BINS, len(elig))
    q = np.zeros(len(trials), dtype=bool)
    if n_take:
        q[elig[np.argsort(key[elig])[:n_take]]] = True
    out['rand'] = q
    return out


def window_ff(counts, cond):
    """Per-neuron condition-adjusted Fano factor -- identical to ff_psth_ldabin.ipynb's window_ff.

    NOTE on `condition`: in the firing-rate files the condition label encodes the side the animal
    CHOSE, not the side the stimulus appeared on (they diverge on error trials). Removing its mean
    therefore strips choice-locked variance. That is benign for an accuracy analysis in one respect
    -- choice-related activity is not left to masquerade as variability -- but it does mean a
    low-accuracy session has a different stimulus/choice mixture inside each condition cell than a
    high-accuracy one. The correct-trials-only variant below is the check on that.
    """
    mean = np.nanmean(counts, axis=1)
    if REMOVE_CONDITION:
        resid = counts.astype(float).copy()
        for cc in pd.unique(cond):
            ci = np.where(cond == cc)[0]
            resid[:, ci] -= np.nanmean(counts[:, ci], axis=1, keepdims=True)
        var = np.nanvar(resid, axis=1, ddof=1)
    else:
        var = np.nanvar(counts, axis=1, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ff = var / (mean + 1e-6)
    ff[(mean <= MIN_WINDOW_COUNT) | ~np.isfinite(ff)] = np.nan
    return ff


# ------------------------------------------------------------------ one pass over the firing rates
def compute_ff():
    files = sorted(f for f in os.listdir(FR_DIR) if f.startswith('firing_rate_'))
    recs = []
    for i, fn in enumerate(files):
        try:
            with open(os.path.join(FR_DIR, fn), 'rb') as f:
                d = pickle.load(f)
            d = d[~d['area'].isin(DROP)]
            if len(d) == 0:
                continue
            session = d['session'].iloc[0]
            if session not in lda1_map or session not in rt_bin_by_session:
                continue
            tcols = sorted([c for c in d.columns if c.startswith('t_')],
                           key=lambda x: float(x.split('_')[1]))
            tsec = np.array([float(c.split('_')[1]) for c in tcols])
            bw = float(np.median(np.diff(tsec)))
            pre_m = (tsec >= PRE_WINDOW[0]) & (tsec < PRE_WINDOW[1])
            post_m = (tsec >= POST_WINDOW[0]) & (tsec < POST_WINDOW[1])
            d = d.copy(); d['nuid'] = d['pid'].astype(str) + '__' + d['neuron_id'].astype(str)
            neurons = sorted(d['nuid'].unique()); trials = sorted(d['trial_id'].unique())
            ni = {n: k for k, n in enumerate(neurons)}; ti = {t: k for k, t in enumerate(trials)}
            A = np.full((len(neurons), len(trials), len(tcols)), np.nan)
            A[d['nuid'].map(ni).values, d['trial_id'].map(ti).values, :] = d[tcols].values * bw
            area = d.groupby('nuid')['area'].first().reindex(neurons).values
            cond = d.drop_duplicates('trial_id').set_index('trial_id')['condition'].reindex(trials).values
            c_pre = np.nansum(A[:, :, pre_m], axis=2)
            c_post = np.nansum(A[:, :, post_m], axis=2)
            eng = eng_by_session[session].reindex(trials).fillna(False).to_numpy().astype(bool)
            sels = selections(session, trials)
            for tag, mask in sels.items():
                for eng_only in (False, True):
                    kk = np.where(mask & eng if eng_only else mask)[0]
                    need = MIN_TRIALS_ENGAGED if eng_only else MIN_TRIALS
                    if len(kk) < need:
                        continue
                    for reg in pd.unique(area):
                        if not isinstance(reg, str):
                            continue
                        idx = np.where(area == reg)[0]
                        if len(idx) < MIN_NEURONS:
                            continue
                        fp = window_ff(c_pre[idx][:, kk], cond[kk])
                        fo = window_ff(c_post[idx][:, kk], cond[kk])
                        ok = np.isfinite(fp) & np.isfinite(fo)
                        for j in np.where(ok)[0]:
                            recs.append((tag, eng_only, session, reg, len(kk), fp[j], fo[j]))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(files)} files...", flush=True)
        except Exception as e:
            print(f"  Error {fn}: {e}")
    df = pd.DataFrame(recs, columns=['sel', 'engaged_only', 'session', 'region', 'n_trials',
                                     'ff_pre', 'ff_post'])
    df['log_ff_pre'] = np.log(df['ff_pre'].clip(lower=1e-6))
    df['log_ff_post'] = np.log(df['ff_post'].clip(lower=1e-6))
    df['ff_quench'] = df['ff_pre'] - df['ff_post']
    return df.merge(sess, on='session', how='left')


# ------------------------------------------------------------------ tests
def icc1(df, col, cluster='mouse_name'):
    """Between-cluster share of variance, ICC(1), for unbalanced cluster sizes.

    The obvious shortcut -- var(cluster means) / var(all values) -- is badly biased upward, because
    each cluster mean carries its own sampling noise: at ~4 sessions per mouse it returned 0.98 for
    session accuracy where the correct answer is 0.58. This uses the one-way random-effects
    decomposition with the unbalanced average cluster size, so the within-cluster mean square is
    subtracted off before the between-cluster variance is formed.
    """
    d = df.dropna(subset=[col, cluster])
    g = d.groupby(cluster)[col]
    k, m, n = g.size(), g.ngroups, len(d)
    if m < 2 or n <= m:
        return np.nan
    ms_between = (k * (g.mean() - d[col].mean()) ** 2).sum() / (m - 1)
    ms_within = ((d[col] - d[cluster].map(g.mean())) ** 2).sum() / (n - m)
    k0 = (n - (k ** 2).sum() / n) / (m - 1)
    var_between = max((ms_between - ms_within) / k0, 0.0)
    return var_between / (var_between + ms_within)


def _design(d, extra):
    dummies = pd.get_dummies(d['region'], drop_first=True)
    Z = pd.concat([pd.Series(1.0, index=d.index, name='const'), dummies,
                   d[list(extra)].astype(float)], axis=1)
    return Z.values.astype(float)


def quad_test(df, y, pred, extra=('n_trials',), n_perm=N_PERM, mouse_perm=MOUSE_PERM, seed=SEED):
    """Linear and quadratic coefficients of `pred` on `y`, each with a session-level permutation p.

    The predictor is CENTRED before squaring. Without that, `pred` and `pred**2` correlate ~0.99 at
    these accuracies and the two coefficients are not separately interpretable -- a real curvature
    can then show up entirely in the linear term or vice versa.

    Both nulls relabel whole sessions (and, for p_mouse, permute among the sessions of one mouse,
    since accuracy varies within a mouse). Under permutation the linear and quadratic columns are
    rebuilt from the shuffled predictor, so the null respects the fact that they are functions of
    the same quantity.
    """
    d = df.dropna(subset=[y, pred, 'region', 'mouse_name'] + list(extra)).reset_index(drop=True)
    if len(d) < 200:
        return None
    yv = d[y].values.astype(float)
    Z = _design(d, extra)
    Zp = np.linalg.pinv(Z)
    y_r = yv - Z @ (Zp @ yv)

    sess_x = d.groupby('session')[pred].first()
    mu = sess_x.mean()

    def coefs(x):
        X = np.column_stack([x - mu, (x - mu) ** 2])
        Xr = X - Z @ (Zp @ X)
        beta, *_ = np.linalg.lstsq(Xr, y_r, rcond=None)
        return beta

    obs = coefs(d[pred].values.astype(float))
    # partial correlation of the quadratic term with y, after the linear term and covariates
    X = np.column_stack([d[pred].values - mu, (d[pred].values - mu) ** 2])
    Zl = np.column_stack([Z, X[:, 0]])
    Zlp = np.linalg.pinv(Zl)
    r_quad = pearsonr(X[:, 1] - Zl @ (Zlp @ X[:, 1]), yv - Zl @ (Zlp @ yv))[0]
    r_lin = pearsonr(X[:, 0] - Z @ (Zp @ X[:, 0]), y_r)[0]

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, 2))
    si = d['session'].values
    for i in range(n_perm):
        sh = pd.Series(rng.permutation(sess_x.values), index=sess_x.index)
        null[i] = coefs(sh.reindex(si).values.astype(float))
    p_lin = float((1 + np.sum(np.abs(null[:, 0]) >= abs(obs[0]))) / (1 + n_perm))
    p_quad = float((1 + np.sum(np.abs(null[:, 1]) >= abs(obs[1]))) / (1 + n_perm))

    m2x = d.groupby('mouse_name')[pred].first()
    sess_mouse = d.groupby('session')['mouse_name'].first()
    rng = np.random.default_rng(seed)
    nullm = np.empty((mouse_perm, 2))
    for i in range(mouse_perm):
        shuffled = sess_x.groupby(sess_mouse).transform(
            lambda s: rng.permutation(s.values) if len(s) > 1 else s.values)
        nullm[i] = coefs(shuffled.reindex(si).values.astype(float))
    pm_quad = float((1 + np.sum(np.abs(nullm[:, 1]) >= abs(obs[1]))) / (1 + mouse_perm))
    pm_lin = float((1 + np.sum(np.abs(nullm[:, 0]) >= abs(obs[0]))) / (1 + mouse_perm))

    # vertex of the fitted parabola, and whether it lies inside the observed predictor range
    vertex = mu - obs[0] / (2 * obs[1]) if obs[1] != 0 else np.nan
    lo, hi = sess_x.min(), sess_x.max()
    return dict(n=len(d), n_sessions=d['session'].nunique(), b_lin=obs[0], b_quad=obs[1],
                r_lin=r_lin, r_quad=r_quad, p_lin=p_lin, p_quad=p_quad,
                pm_lin=pm_lin, pm_quad=pm_quad, vertex=vertex, lo=lo, hi=hi,
                vertex_inside=bool(np.isfinite(vertex) and lo < vertex < hi),
                shape=('inverted-U' if obs[1] < 0 else 'U'))


def two_lines(df, y, pred, extra=('n_trials',), n_break=None):
    """Simonsohn's two-lines test, at the session level.

    A significant quadratic term is not evidence of a U-shape; two significant slopes of OPPOSITE
    sign are. Sessions are collapsed to one FF value each (region- and covariate-adjusted, so the
    adjustment matches the GLM above), split at the breakpoint, and a slope is fitted either side.
    The breakpoint is the interior quantile that maximises the smaller |t| of the two slopes -- a
    robust stand-in for Simonsohn's Robin Hood procedure, which is what protects the test from the
    arbitrary-split problem.
    """
    d = df.dropna(subset=[y, pred, 'region'] + list(extra)).reset_index(drop=True)
    if len(d) < 200:
        return None
    Z = _design(d, extra)
    Zp = np.linalg.pinv(Z)
    resid = d[y].values.astype(float) - Z @ (Zp @ d[y].values.astype(float))
    s = pd.DataFrame({'session': d['session'], 'x': d[pred].values, 'r': resid}) \
        .groupby('session').agg(x=('x', 'first'), y=('r', 'mean')).reset_index()
    s = s.sort_values('x').reset_index(drop=True)

    def slope_t(sub):
        if len(sub) < 15 or sub['x'].std() == 0:
            return np.nan, np.nan
        b, a = np.polyfit(sub['x'], sub['y'], 1)
        yh = a + b * sub['x']
        sse = np.sum((sub['y'] - yh) ** 2)
        se = np.sqrt(sse / (len(sub) - 2) / np.sum((sub['x'] - sub['x'].mean()) ** 2))
        return b, b / se

    cands = np.quantile(s['x'], np.linspace(0.2, 0.8, 25)) if n_break is None else [n_break]
    best = None
    for xb in cands:
        lo_, hi_ = s[s['x'] <= xb], s[s['x'] > xb]
        b1, t1 = slope_t(lo_); b2, t2 = slope_t(hi_)
        if not (np.isfinite(t1) and np.isfinite(t2)):
            continue
        score = min(abs(t1), abs(t2))
        if best is None or score > best['score']:
            best = dict(score=score, xb=xb, b1=b1, t1=t1, n1=len(lo_), b2=b2, t2=t2, n2=len(hi_))
    if best is None:
        return None
    best['p1'] = 2 * tdist.sf(abs(best['t1']), best['n1'] - 2)
    best['p2'] = 2 * tdist.sf(abs(best['t2']), best['n2'] - 2)
    best['u_shaped'] = bool(best['p1'] < .05 and best['p2'] < .05 and
                            np.sign(best['b1']) != np.sign(best['b2']))
    return best


def profile(df, y, pred, k=N_BINS_PROFILE, extra=('n_trials',)):
    """Non-parametric shape: covariate-adjusted session means in `k` accuracy quantile bins."""
    d = df.dropna(subset=[y, pred, 'region'] + list(extra)).reset_index(drop=True)
    Z = _design(d, extra); Zp = np.linalg.pinv(Z)
    resid = d[y].values.astype(float) - Z @ (Zp @ d[y].values.astype(float))
    s = pd.DataFrame({'session': d['session'], 'x': d[pred].values, 'r': resid}) \
        .groupby('session').agg(x=('x', 'first'), y=('r', 'mean')).reset_index()
    s['bin'] = pd.qcut(s['x'], k, labels=False, duplicates='drop')
    return s.groupby('bin').agg(x=('x', 'mean'), mean=('y', 'mean'),
                                sem=('y', lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                n=('y', 'size')).reset_index(), s


def _robust_ylim(ax, v, k=3.5):
    """Clip the y-axis to the bulk of the data (median +/- k robust sds).

    FF_quench has a handful of sessions at +/-3 against a bulk inside +/-1; on a shared axis those
    few points compress everything else into a band and the shape of the relationship -- the thing
    the panel exists to show -- becomes unreadable. Points outside the limit are still plotted and
    still in every test; only the view is clipped, and the count is annotated so the clipping is
    never silent.
    """
    v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
    if len(v) < 10:
        return
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    if mad <= 0:
        return
    lo, hi = med - k * mad, med + k * mad
    n_out = int(np.sum((v < lo) | (v > hi)))
    if n_out and (v.min() < lo or v.max() > hi):
        ax.set_ylim(lo, hi)
        ax.text(0.02, 0.02, f'{n_out} session(s) outside view', transform=ax.transAxes,
                fontsize=7, color='0.35', va='bottom')


# ------------------------------------------------------------------ report
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 100)
    print("SESSION ACCURACY: definitions and confounds")
    print("=" * 100)
    print(f"{len(sess)} sessions, {sess['mouse_name'].nunique()} mice "
          f"(zero-contrast trials excluded from every accuracy measure)")
    print(f"\n{'measure':<14}{'min':>8}{'p25':>8}{'median':>9}{'p75':>8}{'max':>8}{'at 1.0':>8}")
    for c in ['acc', 'acc_std', 'acc_easy', 'acc_hard', 'acc_engaged', 'frac_engaged']:
        v = sess[c].dropna()
        print(f"{c:<14}{v.min():>8.3f}{v.quantile(.25):>8.3f}{v.median():>9.3f}"
              f"{v.quantile(.75):>8.3f}{v.max():>8.3f}{int((v >= 0.999).sum()):>8}")
    print("\n`acc_easy` saturates: sessions sitting exactly at 1.0 cannot express a further increase,")
    print("so any curvature it shows at the top is a ceiling, which is why `acc` is the primary measure.")

    print("\nhow much accuracy range there actually is -- the limit on detecting ANY curvature:")
    _b = [(0, .75), (.75, .80), (.80, .85), (.85, .90), (.90, .95), (.95, 1.01)]
    print("   " + "".join(f"{f'{lo:.2f}-{hi:.2f}':>12}" for lo, hi in _b))
    print("   " + "".join(f"{int(((sess['acc'] >= lo) & (sess['acc'] < hi)).sum()):>12}" for lo, hi in _b))
    print(f"  middle 80% of sessions span {sess['acc'].quantile(.1):.3f}-{sess['acc'].quantile(.9):.3f}, "
          f"a range of {sess['acc'].quantile(.9) - sess['acc'].quantile(.1):.3f}.")
    print("  Both tails are thin. A U-shape has to be carried by the arms, so with ~10 sessions below")
    print("  0.75 and ~8 above 0.95 this design has little power for curvature regardless of the test")
    print("  used -- read a negative U-shape result as 'not detectable here', not 'absent'.")

    print("\nwhat accuracy is entangled with:")
    for c in ['frac_engaged', 'frac_easy', 'lda_1', 'n_trials_beh']:
        d = sess[['acc', c]].dropna()
        print(f"  r(acc, {c:<14}) = {pearsonr(d['acc'], d[c])[0]:+.3f}")
    d = sess[['acc', 'acc_std']].dropna()
    print(f"  r(acc, acc_std)        = {pearsonr(d['acc'], d['acc_std'])[0]:+.3f}   "
          f"(contrast mix barely varies, so standardising changes little)")
    gm = sess.groupby('mouse_name')['acc']
    print(f"\nbetween-mouse share of variance, ICC(1): accuracy {icc1(sess, 'acc'):.3f}, "
          f"lda_1 {icc1(sess, 'lda_1'):.3f}, frac_engaged {icc1(sess, 'frac_engaged'):.3f}")
    print(f"  ({gm.ngroups} mice, {(gm.size() > 1).sum()} with >1 session; median within-mouse sd of")
    print(f"  accuracy {sess.groupby('mouse_name').filter(lambda g: len(g) >= 3).groupby('mouse_name')['acc'].std().median():.3f} "
          f"against a total sd of {sess['acc'].std():.3f})")
    print("Accuracy has ~42% of its variance WITHIN a mouse against ~9% for lda_1, so unlike the")
    print("LDA notebooks the within-mouse null here has real leverage: p_mouse below permutes among")
    print("the sessions of one mouse rather than relabelling whole mice.")

    print("\n" + "=" * 100)
    print("Computing single-neuron Fano factor (one pass, 3 trial selections x all/engaged trials)")
    print("=" * 100)
    df = compute_ff()
    base = df[(df['sel'] == 'all') & (~df['engaged_only'])]
    print(f"neurons: {len(base)} | sessions: {base['session'].nunique()} | "
          f"mice: {base['mouse_name'].nunique()} | regions: {base['region'].nunique()}")

    METRICS = [('log_ff_pre', 'FF_pre (log)'), ('log_ff_post', 'FF_post (log)'),
               ('ff_quench', 'FF_quench')]

    # ---------------- 1. the headline: is there curvature, and does it survive the logit scale ----
    print("\n" + "=" * 100)
    print("1. LINEAR AND QUADRATIC TERMS, RAW vs LOGIT ACCURACY  (selection: all trials, no RT control)")
    print("   The quadratic column is the inverted-U claim. Compare `acc` against `logit_acc`:")
    print("   curvature that appears only on the bounded scale is compression, not a U-shape.")
    print("=" * 100)
    print(f"{'metric':<15}{'predictor':<12}{'b_lin':>10}{'p_lin':>8}{'b_quad':>11}{'p_quad':>8}"
          f"{'pm_quad':>9}{'shape':>12}{'vertex':>9}{'in range?':>11}")
    head = []
    for y, ylab in METRICS:
        for pred in ['acc', 'logit_acc']:
            r = quad_test(base, y, pred)
            if r is None:
                continue
            head.append(dict(metric=ylab, pred=pred, **r))
            print(f"{ylab:<15}{pred:<12}{r['b_lin']:>10.3f}{r['p_lin']:>8.3f}{r['b_quad']:>11.3f}"
                  f"{r['p_quad']:>8.3f}{r['pm_quad']:>9.3f}{r['shape']:>12}{r['vertex']:>9.3f}"
                  f"{str(r['vertex_inside']):>11}")
    print(f"\nobserved range: acc {sess['acc'].min():.3f}-{sess['acc'].max():.3f}, "
          f"logit_acc {sess['logit_acc'].min():.3f}-{sess['logit_acc'].max():.3f}")
    print("A vertex outside that range means the parabola is monotone over the data, however small")
    print("p_quad is -- that row is NOT evidence of a U-shape (Lind & Mehlum / Sasabuchi condition).")

    # ---------------- 2. two-lines test ----------------
    print("\n" + "=" * 100)
    print("2. TWO-LINES TEST -- the test a U-shape actually has to pass")
    print("   Both slopes significant AND of opposite sign. A significant quadratic without this is")
    print("   almost always one tail bending the parabola.")
    print("=" * 100)
    print(f"{'metric':<15}{'predictor':<12}{'break':>8}{'slope_lo':>10}{'p_lo':>8}{'n_lo':>6}"
          f"{'slope_hi':>10}{'p_hi':>8}{'n_hi':>6}{'U-shaped?':>11}")
    for y, ylab in METRICS:
        for pred in ['acc', 'logit_acc']:
            r = two_lines(base, y, pred)
            if r is None:
                continue
            print(f"{ylab:<15}{pred:<12}{r['xb']:>8.3f}{r['b1']:>10.3f}{r['p1']:>8.3f}{r['n1']:>6}"
                  f"{r['b2']:>10.3f}{r['p2']:>8.3f}{r['n2']:>6}{str(r['u_shaped']):>11}")

    # ---------------- 3. non-parametric profile ----------------
    print("\n" + "=" * 100)
    print(f"3. NON-PARAMETRIC PROFILE -- covariate-adjusted session means in {N_BINS_PROFILE} accuracy bins")
    print("   A real inverted U has to be visible here, without a parabola being fitted for it.")
    print("=" * 100)
    prof_store = {}
    for y, ylab in METRICS:
        p, s = profile(base, y, 'acc')
        prof_store[y] = (p, s)
        print(f"\n{ylab}  (adjusted session means, arbitrary offset)")
        print("   " + "".join(f"{f'bin{int(b)}':>12}" for b in p['bin']))
        print(f"   {'acc':<3}" + "".join(f"{v:>12.3f}" for v in p['x']))
        print(f"   {'FF':<3}" + "".join(f"{v:>12.3f}" for v in p['mean']))
        print(f"   {'sem':<3}" + "".join(f"{v:>12.3f}" for v in p['sem']))
        print(f"   {'n':<3}" + "".join(f"{int(v):>12}" for v in p['n']))
        peak = int(p['mean'].idxmax()); trough = int(p['mean'].idxmin())
        interior = 0 < peak < len(p) - 1
        print(f"   -> highest bin {peak}, lowest bin {trough}; peak is "
              f"{'INTERIOR (consistent with an inverted U)' if interior else 'at an edge (monotone)'}")

    # ---------------- 4. the engagement confound ----------------
    print("\n" + "=" * 100)
    print("4. IS IT ACCURACY, OR ENGAGEMENT WEARING ITS LABEL?")
    print("   acc correlates r = +0.54 with frac_engaged, and ff_vs_engagement_bouts.ipynb shows")
    print("   engagement dynamics track FF. Three views: raw, frac_engaged partialled out, and")
    print("   engaged trials only (where accuracy can no longer act through time-off-task).")
    print("=" * 100)
    eng_only = df[(df['sel'] == 'all') & (df['engaged_only'])]
    print(f"{'metric':<15}{'view':<22}{'r_lin':>9}{'p_lin':>8}{'r_quad':>9}{'p_quad':>8}{'pm_lin':>9}")
    for y, ylab in METRICS:
        for view, frame, extra, pred in [
                ('raw', base, ('n_trials',), 'logit_acc'),
                ('| frac_engaged', base, ('n_trials', 'frac_engaged'), 'logit_acc'),
                ('engaged trials only', eng_only, ('n_trials',), 'logit_acc_engaged')]:
            r = quad_test(frame, y, pred, extra=extra)
            if r is None:
                continue
            print(f"{ylab:<15}{view:<22}{r['r_lin']:>9.4f}{r['p_lin']:>8.3f}"
                  f"{r['r_quad']:>9.4f}{r['p_quad']:>8.3f}{r['pm_lin']:>9.3f}")

    # ---------------- 5. RT control ----------------
    print("\n" + "=" * 100)
    print("5. RT CONTROL  (cap 1.0 s, 8 bins; 'rand' = same trial count, RT distribution untouched)")
    print("   Read 'rt' against 'rand', never against 'all': matching costs trials, and lost trials")
    print("   shrink a slope on their own.")
    print("=" * 100)
    print(f"{'metric':<15}{'selection':<10}{'n':>7}{'n_sess':>7}{'r_lin':>9}{'p_lin':>8}"
          f"{'r_quad':>9}{'p_quad':>8}")
    for y, ylab in METRICS:
        for tag in ['all', 'rt', 'rand']:
            sub = df[(df['sel'] == tag) & (~df['engaged_only'])]
            r = quad_test(sub, y, 'logit_acc')
            if r is None:
                continue
            print(f"{ylab:<15}{tag:<10}{r['n']:>7}{r['n_sessions']:>7}{r['r_lin']:>9.4f}"
                  f"{r['p_lin']:>8.3f}{r['r_quad']:>9.4f}{r['p_quad']:>8.3f}")

    # ---------------- 6. other accuracy definitions ----------------
    print("\n" + "=" * 100)
    print("6. ROBUSTNESS TO THE ACCURACY DEFINITION  (all trials, no RT control)")
    print("=" * 100)
    print(f"{'metric':<15}{'predictor':<16}{'r_lin':>9}{'p_lin':>8}{'r_quad':>9}{'p_quad':>8}")
    for y, ylab in METRICS:
        for pred in ['logit_acc', 'logit_acc_std', 'logit_acc_easy', 'logit_acc_hard']:
            r = quad_test(base, y, pred)
            if r is None:
                continue
            print(f"{ylab:<15}{pred:<16}{r['r_lin']:>9.4f}{r['p_lin']:>8.3f}"
                  f"{r['r_quad']:>9.4f}{r['p_quad']:>8.3f}")

    # ---------------- figures ----------------
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    for ci, (y, ylab) in enumerate(METRICS):
        p, s = prof_store[y]
        ax = axes[0][ci]
        ax.scatter(s['x'], s['y'], c=s['x'], cmap='coolwarm', s=42, alpha=0.75,
                   edgecolors='black', linewidth=0.3)
        ax.errorbar(p['x'], p['mean'], yerr=p['sem'], color='black', lw=2.2, marker='o',
                    ms=7, capsize=4, zorder=5, label=f'{N_BINS_PROFILE} accuracy bins (+/- SEM)')
        _x = np.linspace(s['x'].min(), s['x'].max(), 100)
        _c = np.polyfit(s['x'], s['y'], 2)
        ax.plot(_x, np.polyval(_c, _x), color='#762a83', ls='--', lw=1.8, label='quadratic fit')
        _robust_ylim(ax, s['y'])
        ax.set_xlabel('session accuracy'); ax.set_ylabel(f'{ylab}\n(region/n_trials adjusted)')
        ax.set_title(f'{ylab} vs accuracy', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); sns.despine(ax=ax, offset=6)

        ax = axes[1][ci]
        s2 = s.merge(sess[['session', 'logit_acc']], on='session', how='left')
        ax.scatter(s2['logit_acc'], s2['y'], c=s2['logit_acc'], cmap='coolwarm', s=42, alpha=0.75,
                   edgecolors='black', linewidth=0.3)
        _x = np.linspace(s2['logit_acc'].min(), s2['logit_acc'].max(), 100)
        _c = np.polyfit(s2['logit_acc'], s2['y'], 2)
        ax.plot(_x, np.polyval(_c, _x), color='#762a83', ls='--', lw=1.8, label='quadratic fit')
        _c1 = np.polyfit(s2['logit_acc'], s2['y'], 1)
        ax.plot(_x, np.polyval(_c1, _x), color='black', lw=2, label='linear fit')
        _robust_ylim(ax, s2['y'])
        ax.set_xlabel('logit(session accuracy)'); ax.set_ylabel(f'{ylab} (adjusted)')
        ax.set_title(f'{ylab} vs logit accuracy\n(the scale a bounded proportion is linear on)',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); sns.despine(ax=ax, offset=6)
    fig.suptitle('Fano factor vs session accuracy: raw scale (top) and logit scale (bottom)',
                 y=1.01, fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'ff_vs_accuracy.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"figure -> {out}")
    tbl = os.path.join(FIG_DIR, 'ff_vs_accuracy_neurons.parquet')
    df.to_parquet(tbl)
    print(f"per-neuron FF table -> {tbl}  ({len(df)} rows; reload it to try other predictors or")
    print("covariates without another pass over the firing-rate files)")
    return df, pd.DataFrame(head)


if __name__ == '__main__':
    _df, _head = main()
