"""
LAB / MOUSE / SESSION: THE SESSION-LEVEL VARIANCE PARTITION
===========================================================
The simple version of nested_variance.ipynb. The unit is the SESSION: its mean value of a feature, averaged over
its trials. Per feature, a nested random-effects model

    y_session = mu + lab + mouse(lab) + session

estimated with nested_variance's unbiased method-of-moments estimator (the same one, with sessions as the units).
Each lab trains on one rig and every mouse ran all of its sessions on that rig, so the mouse component is the
variance between mice OF THE SAME LAB (AND RIG): rig changes cannot inflate the session component, and lab or rig
differences cannot inflate the mouse component.

The session residual contains a little trial-sampling noise (a session mean averages ~560 trials). It is
estimated per session as (within-session variance / n trials) and split off, so
    shares = lab + mouse + session + sampling noise = 1
    R = mouse / (mouse + session + noise)   repeatability of ONE session within its lab

FEATURES. Interpretable summaries, per epoch: whisking, licking, the stillest paw state (Paw 0) and mean paw-state
vigour (sum over states of rank x occupancy; uniform numbering, 0 = stillest). Plus "all features": the whole
400-feature syllable profile (40 bins x 10 indicators), components summed over features.

UNCERTAINTY. Leave-one-mouse-out jackknife (see jackknife()): SE over the 56 deletions, t-based 95% CI and p.
A bootstrap over mice was tried first and is biased against the mouse component with this few mice per lab.

PER LAB. For each lab with >= 3 mice, a one-way version (mouse vs session) on the whole profile, jackknife over
that lab's mice: does mouse > session hold inside each lab, not only on average?

Writes figures/session_variance_partition.{png,pdf,csv}.
"""
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent
for p in (PI, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import nested_variance as nv
import paper_style as ps

FIGDIR = HERE / 'figures'
EPOCHS = ps.EPOCH_SHORT                      # Pre-Q, Quiesc., Choice, ITI
COLORS = {'Lab': '#22344A', 'Mouse': '#3E8E54', 'Session': '#8FCB8B', 'Noise': '#D9D9D9'}


SYLLABLE_SETS = {
    # name: (file, remap the paw digit to the uniform vigor numbering?)
    'production': (nv.SYLLABLE_FILE, True),
    # left paw + wheel states (both standardised per session, like the paw states); its paw digits are relabelled
    # and NOT in vigor order, and the map is not on this machine -- so only channel totals are meaningful
    'paw_wheel': (PI / 'data' / 'paw_wheel_8_k_10_bin_syllables_28-08-2026', False),
}


def build(syllables='production'):
    ps.use_paw_states('uniform')
    path, remap = SYLLABLE_SETS[syllables]
    codes, meta = nv.load_trials(syllable_file=path)
    OH = nv.one_hot(ps.paw_codes(codes) if remap else codes)      # trials x 40 x 10
    n_tr, B, S = OH.shape

    # per-trial summary features, then the whole profile
    vig = (OH[..., :8] * np.arange(8)).sum(-1)                    # vigour rank of each bin's paw state
    feats = {}
    for e, name in enumerate(EPOCHS):
        sl = slice(e * 10, (e + 1) * 10)
        feats[f'Whisking, {name}'] = OH[:, sl, 8].mean(1)
        feats[f'Licking, {name}'] = OH[:, sl, 9].mean(1)
        feats[f'Still paws (Paw 0), {name}'] = OH[:, sl, 0].mean(1)
        feats[f'Paw vigour, {name}'] = vig[:, sl].mean(1)
    F = pd.DataFrame(feats)
    blocks = {'summary': F.to_numpy(float), 'profile': OH.reshape(n_tr, -1).astype(float)}

    s_codes, s_ids = pd.factorize(meta['session'], sort=True)
    A, _ = nv._indicator(s_codes, n_tr)
    n = np.bincount(s_codes).astype(float)
    out = {}
    for key, Y in blocks.items():
        mean = np.asarray(A @ Y) / n[:, None]
        sq = np.asarray(A @ (Y ** 2)) / n[:, None]
        within = (sq - mean ** 2) * n[:, None] / (n[:, None] - 1)  # within-session variance
        out[key] = (mean, within / n[:, None])                    # session means, sampling noise of each mean
    ses = meta.groupby(s_codes)[['session', 'mouse_name', 'lab']].first().reset_index(drop=True)
    return ses, out, list(F.columns)


def partition(ses, mean, noise, idx=None):
    """Variance components per feature: lab, mouse, session (noise removed), noise. idx: rows (a resample)."""
    if idx is not None:
        ses, mean, noise = ses.iloc[idx].reset_index(drop=True), mean[idx], noise[idx]
    groups, K = nv.design(ses, levels=['lab', 'mouse'])
    comp = nv.components(mean, groups, K)                         # lab, mouse, residual(session) x features
    nz = noise.mean(0)
    return np.vstack([comp[0], comp[1], comp[2] - nz, nz])


def one_way(ses, mean, noise, idx=None):
    """Within one lab: mouse vs session (noise removed), summed over features."""
    if idx is not None:
        ses, mean, noise = ses.iloc[idx].reset_index(drop=True), mean[idx], noise[idx]
    groups, K = nv.design(ses, levels=['mouse'])
    comp = nv.components(mean, groups, K).sum(1)
    nz = noise.mean(0).sum()
    return np.array([comp[0], comp[1] - nz, nz])


def jackknife(ses, fn):
    """Leave-one-mouse-out jackknife of fn(ses_subset, rows) -> array. Returns (estimate, SE, n_mice).

    Not a bootstrap: resampling mice with replacement leaves fewer DISTINCT mice per lab (3-8 here), which biases the
    mouse component down in every resample (measured: mouse - session share shifted by about -0.10, the estimate
    sitting at the top edge of its percentile CI). Deleting one mouse at a time has no such bias."""
    est = fn(ses, np.arange(len(ses)))
    mice = ses['mouse_name'].unique()
    reps = []
    for m in mice:
        keep = np.flatnonzero(ses['mouse_name'].to_numpy() != m)
        reps.append(fn(ses.iloc[keep].reset_index(drop=True), keep))
    reps = np.array(reps)
    n = len(mice)
    se = np.sqrt((n - 1) / n * ((reps - reps.mean(0)) ** 2).sum(0))
    return est, se, n


def t_ci(est, se, n):
    t = stats.t.ppf(0.975, n - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = 2 * stats.t.sf(np.abs(est / se), n - 1)
    return est - t * se, est + t * se, p


def run():
    ses, blocks, labels = build()
    print(f'{len(ses)} sessions, {ses.mouse_name.nunique()} mice, {ses.lab.nunique()} labs')

    # ---- shares, mouse - session and one-session repeatability, summary features + the whole profile
    def stats_of(ses_, rows):
        comp_s = partition(ses_, blocks['summary'][0][rows], blocks['summary'][1][rows])          # 4 x 16
        comp_p = partition(ses_, blocks['profile'][0][rows], blocks['profile'][1][rows]).sum(1)   # 4
        c = np.column_stack([comp_p[:, None], comp_s])                                           # 4 x 17
        sh = c / c.sum(0)
        return np.vstack([sh, sh[1] - sh[2], c[1] / c[1:].sum(0)])                               # 6 x 17
    names = ['All features (400)'] + labels
    est, se, n = jackknife(ses, stats_of)
    lo, hi, p = t_ci(est, se, n)
    tab = pd.DataFrame({'feature': names,
                        'lab': est[0], 'mouse': est[1], 'session': est[2], 'noise': est[3],
                        'mouse_lo': lo[1], 'mouse_hi': hi[1], 'session_lo': lo[2], 'session_hi': hi[2],
                        'mouse_minus_session_share': est[4], 'diff_lo': lo[4], 'diff_hi': hi[4], 'p_diff': p[4],
                        'R_one_session': est[5], 'R_lo': lo[5], 'R_hi': hi[5]})

    # ---- per lab, whole profile (jackknife over that lab's mice; with 3-4 mice the CIs are very wide)
    lab_rows = []
    for lab, g in ses.groupby('lab'):
        if g['mouse_name'].nunique() < 3:
            continue
        sel = g.index.to_numpy()
        s_r = ses.iloc[sel].reset_index(drop=True)
        mean, noise = blocks['profile'][0][sel], blocks['profile'][1][sel]

        def lab_stats(ses_, rows):
            e = one_way(ses_, mean[rows], noise[rows])
            sh = e / e.sum()
            return np.r_[sh, sh[0] - sh[1]]
        e, se_l, n_l = jackknife(s_r, lab_stats)
        lo_l, hi_l, p_l = t_ci(e, se_l, n_l)
        lab_rows.append(dict(lab=lab, n_mice=n_l, n_sessions=len(g), mouse=e[0], session=e[1], noise=e[2],
                             mouse_lo=lo_l[0], mouse_hi=hi_l[0], session_lo=lo_l[1], session_hi=hi_l[1],
                             diff=e[3], diff_lo=lo_l[3], diff_hi=hi_l[3], p_diff=p_l[3]))
    LAB = pd.DataFrame(lab_rows).sort_values('mouse', ascending=False).reset_index(drop=True)

    FIGDIR.mkdir(exist_ok=True)
    tab.to_csv(FIGDIR / 'session_variance_partition.csv', index=False)
    LAB.to_csv(FIGDIR / 'session_variance_partition_per_lab.csv', index=False)
    with pd.option_context('display.width', 220):
        print(tab[['feature', 'lab', 'mouse', 'session', 'noise', 'mouse_minus_session_share', 'diff_lo', 'diff_hi',
                   'p_diff', 'R_one_session']].round(3).to_string(index=False))
        print('\nper lab (whole profile):\n', LAB.round(3).to_string(index=False))
    figure(tab, LAB)


def figure(tab, LAB):
    ps.use('paper')
    fig = plt.figure(figsize=(7.09, 4.6))
    G = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.9, 0.95], wspace=0.55, left=0.2, right=0.98, top=0.86, bottom=0.12)

    # a: stacked shares per feature
    ax = fig.add_subplot(G[0])
    y = np.arange(len(tab))
    left = np.zeros(len(tab))
    for key, col in (('lab', 'Lab'), ('mouse', 'Mouse'), ('session', 'Session'), ('noise', 'Noise')):
        v = tab[key].clip(lower=0).to_numpy()
        ax.barh(y, v, left=left, color=COLORS[col], height=0.72, label=col if key != 'noise' else 'Sampling noise',
                edgecolor='white', lw=0.3)
        left += v
    ax.set_yticks(y); ax.set_yticklabels(tab['feature'], fontsize=5.6)
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.set_ylim(len(tab) - 0.5, -0.5)
    ax.set_xlim(0, 1); ax.set_xlabel('Share of session-level variance', fontsize=6.5)
    ax.tick_params(axis='x', labelsize=6)
    ax.axhline(0.5, color='0.3', lw=0.6)
    ax.legend(frameon=False, fontsize=5.8, ncol=4, loc='lower left', bbox_to_anchor=(-0.02, 1.0), handlelength=1,
              columnspacing=0.8)

    # b: mouse - session, with CI
    ax = fig.add_subplot(G[1], sharey=ax)
    d, lo, hi = tab['mouse_minus_session_share'], tab['diff_lo'], tab['diff_hi']
    sig = (lo > 0) | (hi < 0)
    ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt='none', ecolor='0.4', elinewidth=0.7)
    ax.scatter(d[sig], y[sig], s=12, color=COLORS['Mouse'], zorder=3)
    ax.scatter(d[~sig], y[~sig], s=12, facecolor='white', edgecolor=COLORS['Mouse'], zorder=3)
    ax.axvline(0, color='0.5', lw=0.6)
    ax.axhline(0.5, color='0.3', lw=0.6)
    ax.tick_params(labelleft=False, labelsize=6)
    ax.set_xlabel('Mouse − session share\n(filled: 95% CI excludes 0)', fontsize=6.5)
    ax.set_xlim(-0.8, 0.8)

    # c: per lab, whole profile
    ax = fig.add_subplot(G[2])
    yr = np.arange(len(LAB))
    for off, key, col in ((-0.18, 'mouse', 'Mouse'), (0.18, 'session', 'Session')):
        v = LAB[key]
        ax.errorbar(v, yr + off, xerr=[v - LAB[f'{key}_lo'], LAB[f'{key}_hi'] - v], fmt='o', ms=3, color=COLORS[col],
                    elinewidth=0.7, label=col)
    ax.set_yticks(yr)
    ax.set_yticklabels([f"{r.lab} ({r.n_mice} mice)" for r in LAB.itertuples()], fontsize=5.6)
    ax.set_ylim(len(LAB) - 0.5, -0.5)
    ax.set_xlim(0, 1); ax.set_xlabel('Share of variance within the lab\n(whole profile)', fontsize=6.5)
    ax.tick_params(axis='x', labelsize=6)
    ax.legend(frameon=False, fontsize=5.8, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)

    for a_, letter in zip(fig.axes, 'abc'):
        b = a_.get_position()
        fig.text(b.x0 - (0.17 if letter == 'a' else 0.03 if letter == 'b' else 0.12), 0.93, letter,
                 fontsize=9, fontweight='bold')
    fig.text(0.2, 0.015, 'Unit: the session. Mouse = variance between mice of the same lab (and rig). '
                         'CIs: jackknife over mice.', fontsize=5.6, color='0.35')
    for path in (FIGDIR / 'session_variance_partition.png', FIGDIR / 'session_variance_partition.pdf'):
        fig.savefig(path, dpi=300)
        print('wrote', path.relative_to(PI))


if __name__ == '__main__':
    run()
