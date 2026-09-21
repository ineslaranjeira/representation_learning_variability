"""
IS THIS CORRELATION INDIVIDUALITY, OR IS IT THE LAB?
=====================================================
Each mouse is recorded in one lab, so `lab` is nested inside `mouse` and any correlation
between an LD axis and a behavioural measure can be produced two ways that look identical
in a scatter plot:

  BETWEEN labs   mice from lab A sit high on the axis AND high on the measure, for
                 reasons that may be rig, cohort, handler or genuine cohort biology.
                 One lab is one observation here, so ten labs is ten points -- and the
                 raw correlation across 57 mice reports it as if it were 57.
  WITHIN labs    among mice recorded on the same rig by the same people, the ones higher
                 on the axis are higher on the measure. This is the individuality claim.

The same number cannot answer both, so every measure gets three:

  r_raw        Pearson across mice. Includes both routes. The number usually quoted.
  r_within     both variables centred on their own lab's mean first, so only the second
               route survives. Its p uses df = n_mice - n_labs - 1, because the ten lab
               means were estimated from the same data.
  slope_mixed  lab as a RANDOM effect instead of a subtracted constant, fit on the
               sessions rather than the mouse means. Partial pooling is the point: with
               3-mouse labs in this cohort, hard centring subtracts a mean that is mostly
               noise and over-corrects, while a random effect shrinks a small lab toward
               the grand mean in proportion to how badly it is measured. Both variables
               are z-scored first, so the slope is in the same units as r.

HOW TO READ THE THREE TOGETHER
  raw significant, within not      the effect lives between labs. It may still be real
                                   biology, but it is a claim about COHORTS on ~10
                                   independent points, not about individuals.
  both significant, similar size   survives the control; the effect exists among labmates.
  within LARGER than raw           lab structure was masking it -- possible, and worth
                                   saying out loud rather than quietly reporting the
                                   bigger one.

WHAT NONE OF THIS FIXES. Centring or modelling lab cannot remove within-lab rig drift,
and it cannot tell rig from cohort biology, because nothing in the data separates them.
`eta2_x` / `eta2_y` say how much of each variable lab explains at all -- read them against
`eta2_null_95`, since with 10 labs and ~57 mice the chance floor is around 0.28, not 0.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.regression.mixed_linear_model import MixedLM
    _HAS_SM = True
except ImportError:                                  # the mixed column is simply skipped
    _HAS_SM = False


# ----------------------------------------------------------------- building blocks
def to_mouse_level(df, cols, mouse='mouse_name', lab='lab'):
    """One row per mouse: the mean of `cols`, plus its lab.

    The unit of analysis, and not a detail: sessions of one mouse are not independent,
    so a correlation across 269 sessions counts the same animal three to sixteen times
    and its p-value is meaningless. Everything here aggregates first.
    """
    cols = [c for c in cols if c in df.columns]
    out = df.groupby(mouse)[cols].mean(numeric_only=True)
    out[lab] = df.groupby(mouse)[lab].first()
    out['n_sessions'] = df.groupby(mouse).size()
    return out.reset_index()


def lab_center(frame, cols, lab='lab'):
    """Subtract each lab's own mean from `cols`. Expects MOUSE-level rows: centring on a
    lab mean computed over sessions would weight a 16-session mouse four times a
    4-session one when forming that mean."""
    out = frame.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(lab)[c].transform('mean')
    return out


def eta2_lab(values, labs):
    """Share of the variance across mice that a lab label explains."""
    v = np.asarray(values, float)
    l = np.asarray(labs)
    ok = np.isfinite(v)
    v, l = v[ok], l[ok]
    tot = ((v - v.mean()) ** 2).sum()
    if tot <= 0:
        return np.nan
    within = sum(((v[l == g] - v[l == g].mean()) ** 2).sum() for g in np.unique(l))
    return float(1 - within / tot)


def eta2_null(labs, n_perm=2000, seed=0):
    """Chance level for eta2_lab in THIS design. Ten lab means estimated from ~57 mice
    absorb noise, so eta2 is biased upwards -- around 0.15 median and 0.28 at the 95th
    percentile here, not 0. An eta2 of 0.2 is therefore not evidence of a lab effect.
    (lab/variance_partition.py makes the same point and gives unbiased components.)"""
    l = np.asarray(labs)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_perm, len(l)))
    return np.array([eta2_lab(x[i], rng.permutation(l)) for i in range(n_perm)])


def _p_from_r(r, df):
    if not np.isfinite(r) or df <= 0 or abs(r) >= 1:
        return np.nan
    t = abs(r) * np.sqrt(df / max(1 - r ** 2, 1e-12))
    return float(2 * stats.t.sf(t, df))


def _z(s):
    s = np.asarray(s, float)
    sd = np.nanstd(s)
    return (s - np.nanmean(s)) / sd if sd > 0 else s * np.nan


# ----------------------------------------------------------------- the three numbers
def three_ways(session_df, x, y, lab='lab', mouse='mouse_name', mixed=True,
               mixed_on='session'):
    """`x` vs `y` three ways: raw, lab-centred, and with lab as a random effect.

    `session_df` is session-level; the raw and centred correlations are computed on mouse
    means taken from it. The mixed model is fit on the SESSIONS by default, with lab as
    the grouping factor and mouse as a nested variance component, so it uses every
    session without pretending they are independent. `mixed_on='mouse'` fits it on the
    mouse means instead, which is simpler and converges more reliably.
    """
    M = to_mouse_level(session_df, [x, y], mouse=mouse, lab=lab)
    M = M.dropna(subset=[x, y])
    n, k = len(M), M[lab].nunique()
    out = dict(x=x, y=y, n_mice=n, n_labs=k,
               eta2_x=eta2_lab(M[x], M[lab]), eta2_y=eta2_lab(M[y], M[lab]))
    if n < 5 or k < 2:
        return {**out, 'r_raw': np.nan, 'p_raw': np.nan, 'r_within': np.nan,
                'p_within': np.nan, 'slope_mixed': np.nan, 'p_mixed': np.nan,
                'sd_lab': np.nan, 'icc_lab': np.nan}

    r_raw, p_raw = stats.pearsonr(M[x], M[y])
    # NOT named `C`: patsy resolves the `C(...)` in vc_formula below from this function's
    # namespace, so a local DataFrame called C shadows patsy's categorical helper and the
    # mixed model dies with an unhelpful PatsyError.
    cen = lab_center(M, [x, y], lab=lab)
    r_w = stats.pearsonr(cen[x], cen[y])[0]
    # df spent: one grand mean plus (k - 1) lab means, then the slope
    out.update(r_raw=r_raw, p_raw=p_raw, r_within=r_w,
               p_within=_p_from_r(r_w, n - k - 1), df_within=n - k - 1)

    out.update(slope_mixed=np.nan, p_mixed=np.nan, sd_lab=np.nan, icc_lab=np.nan,
               sd_mouse=np.nan, icc_mouse=np.nan, mixed_note='', mixed_unit='')
    if mixed and _HAS_SM:
        unit = mixed_on
        note = ''
        if unit == 'session':
            # A PREDICTOR THAT IS CONSTANT WITHIN A MOUSE CANNOT BE FIT ALONGSIDE A MOUSE
            # RANDOM EFFECT: the two are collinear and the slope is unidentified. It does
            # not error -- it silently returns 0.000 with p = 1, which reads as 'no
            # effect'. log_training is exactly this case, one value per mouse. So detect
            # it and fall back to the mouse-level fit instead of reporting the zero.
            for col in (x, y):
                v = session_df[[col, mouse]].dropna()
                if len(v) and v[col].std() > 0:
                    wv = v.groupby(mouse)[col].transform(lambda s: s - s.mean()).var()
                    if not np.isfinite(wv) or wv / max(v[col].var(), 1e-12) < 0.01:
                        unit, note = 'mouse', f'{col} is constant within mouse'
                        break
        if unit == 'session':
            d = session_df[[x, y, lab, mouse]].dropna().copy()
        else:
            d = M[[x, y, lab, mouse]].copy() if mouse in M else M[[x, y, lab]].copy()
        d['_x'], d['_y'] = _z(d[x]), _z(d[y])
        out['mixed_unit'] = unit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                kw = {}
                if unit == 'session' and d[mouse].nunique() < len(d):
                    # mouse nested in lab, so sessions are not treated as independent.
                    # re_formula='1' IS REQUIRED: passing vc_formula on its own makes
                    # statsmodels drop the random intercept for `groups` entirely -- the
                    # fit succeeds, cov_re comes back 0x0, and the model silently has no
                    # LAB effect at all, which is the one thing it exists to have.
                    kw['vc_formula'] = {'mouse': f'0 + C({mouse})'}
                    kw['re_formula'] = '1'
                res = MixedLM.from_formula('_y ~ _x', groups=lab, data=d, **kw).fit()
            out.update(slope_mixed=float(res.params['_x']),
                       p_mixed=float(res.pvalues['_x']))
            if not getattr(res, 'converged', True):
                note = (note + '; ' if note else '') + 'did not converge'
        except Exception as e:                       # singular fits are common and fine
            note = (note + '; ' if note else '') + type(e).__name__
            res = None
        # variance components, each guarded on its own: a failure here must not discard
        # the slope, which is the number the table is actually built on
        if res is not None:
            try:
                cr = np.atleast_2d(np.asarray(res.cov_re, dtype=float))
                v_lab = float(cr[0, 0]) if cr.size else np.nan
                sc = float(res.scale)
                vc = np.asarray(getattr(res, 'vcomp', []), dtype=float)
                v_mouse = float(vc[0]) if vc.size else 0.0
                out['sd_lab'] = np.sqrt(max(v_lab, 0)) if np.isfinite(v_lab) else np.nan
                out['sd_mouse'] = np.sqrt(max(v_mouse, 0))
                tot = v_lab + v_mouse + sc
                out['icc_lab'] = v_lab / tot if np.isfinite(v_lab) and tot > 0 else np.nan
                out['icc_mouse'] = v_mouse / tot if tot > 0 else np.nan
            except Exception:
                pass
        out['mixed_note'] = note
    elif mixed:
        out['mixed_note'] = 'statsmodels not installed'
    return out


def verdict(row, alpha=0.05, p_raw='p_raw', p_within='p_within'):
    """One line per measure, naming which of the two routes carries the effect."""
    sr, sw = row[p_raw] < alpha, row[p_within] < alpha
    if sr and sw:
        shrink = abs(row['r_within']) / max(abs(row['r_raw']), 1e-9)
        return 'survives lab-centring' + (f' ({shrink:.0%} of raw)' if shrink < 0.8 else '')
    if sr and not sw:
        return 'BETWEEN labs only -- a claim about cohorts, on ~%d points' % row['n_labs']
    if sw and not sr:
        return 'within labs only -- lab structure was masking it'
    return 'no relation either way'


def screen(session_df, lda_col, variables, lab='lab', mouse='mouse_name',
           alpha=0.05, method='fdr_bh', mixed=True, mixed_on='session'):
    """`three_ways` for a list of measures, with FDR applied to each p column separately.

    Correcting the three columns separately is deliberate: they are three different
    questions asked of the same measures, not one family of tests, and pooling them would
    make the control harder to pass the more controls you run.
    """
    from statsmodels.stats.multitest import multipletests
    rows = [three_ways(session_df, lda_col, v, lab=lab, mouse=mouse,
                       mixed=mixed, mixed_on=mixed_on) for v in variables]
    t = pd.DataFrame(rows)
    for col, out in [('p_raw', 'q_raw'), ('p_within', 'q_within'), ('p_mixed', 'q_mixed')]:
        ok = t[col].notna()
        t[out] = np.nan
        if ok.sum():
            t.loc[ok, out] = multipletests(t.loc[ok, col], alpha=alpha, method=method)[1]
    t['verdict'] = [verdict(r, alpha, 'q_raw', 'q_within') for _, r in t.iterrows()]
    return t


def print_screen(t, alpha=0.05):
    """The table, as text, with the columns a reader needs in one place."""
    print(f'{"measure":22s}{"n":>4s}{"r_raw":>8s}{"q":>8s}{"r_within":>10s}{"q":>8s}'
          f'{"mixed":>8s}{"q":>8s}{"eta2 lab":>10s}   verdict')
    for _, r in t.iterrows():
        e = f'{r.eta2_x:.2f}/{r.eta2_y:.2f}'
        print(f'{str(r.y)[:22]:22s}{int(r.n_mice):4d}{r.r_raw:+8.3f}{r.q_raw:8.3f}'
              f'{r.r_within:+10.3f}{r.q_within:8.3f}{r.slope_mixed:+8.3f}'
              f'{r.q_mixed:8.3f}{e:>10s}   {r.verdict}')
    print(f'\neta2 lab = share of the LD axis / of the measure that a lab label explains.')
    if 'mixed_note' in t and (t.mixed_note.astype(str).str.len() > 0).any():
        for _, r in t[t.mixed_note.astype(str).str.len() > 0].iterrows():
            print(f'  mixed model for {r.y}: {r.mixed_note}')


def plot_screen(t, lda_col, ps=None, plt=None, save=False):
    """raw vs lab-centred r for every measure, side by side. A measure whose bar shrinks
    to nothing when centred is one that lived between labs."""
    if plt is None:
        import matplotlib.pyplot as plt
    y = np.arange(len(t))
    neutral = getattr(ps, 'NEUTRAL', '#7A7A7A') if ps else '#7A7A7A'
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(t) + 1.4))
    ax.barh(y - 0.2, t.r_raw, height=0.36, color='0.35', label='raw')
    ax.barh(y + 0.2, t.r_within, height=0.36, color=neutral, alpha=0.85,
            label='lab-centred')
    for i, r in enumerate(t.itertuples()):
        for off, q, val in [(-0.2, r.q_raw, r.r_raw), (0.2, r.q_within, r.r_within)]:
            if np.isfinite(q) and q < 0.05:
                ax.text(val + np.sign(val) * 0.02, i + off, '*', va='center',
                        ha='left' if val > 0 else 'right', fontsize=9)
    ax.axvline(0, color='0.6', lw=0.8)
    ax.set_yticks(y, [str(v)[:24] for v in t.y])
    ax.invert_yaxis()
    ax.set_xlabel(f'correlation with {lda_col} (mouse level)')
    ax.set_title('Does the relation survive lab-centring?  (* = q < 0.05)')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if save and ps is not None:
        ps.savefig(fig, f'lab_controls_{lda_col}')
    plt.show()
    return fig
