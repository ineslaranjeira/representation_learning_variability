"""
The session-level variance partition as one figure (the session-level counterpart of nested_variance.ipynb's figure).
Unit: the session -- its mean syllable profile (40 bins x 10 indicators). Model and estimator as in
session_variance_partition.py: y_session = mu + lab + mouse(lab) + session, sampling noise of each session mean split
off. Lab = rig (one rig per lab, every mouse on one rig), so "mouse" is the variance between mice of the same lab.

MAIN (session_partition):
  a  variance budget of the whole profile (pie); the mouse-vs-session difference with its CI and p underneath:
     leave-one-mouse-out jackknife (svp.jackknife)
  b  distances between session profiles: same mouse, other mouse of the same lab, other lab
  c  stability across days: same-mouse distance against days apart (bins with >= 10 pairs)
  d  consistency as a trait: each mouse's own day-to-day variance, sorted, against a null in which every mouse is
     equally consistent (session deviations shuffled across mice); p for the spread across mice
SUPPLEMENT (session_partition_supp):
  a  reliability of a mouse's profile from n sessions, R = mouse / (mouse + (session + noise) / n)
  b  where in the trial: the shares bin by bin, per channel
  (the per-channel budget is printed and in session_variance_partition.csv)
Lab stays a level of the model (so "mouse" is within lab); the per-lab breakdown is not drawn -- with 3-8 mice per
lab its CIs are uninformative (session_variance_partition.py still writes it to csv).

Writes figures/session_partition.{png,pdf}; `python fig_session_partition.py paw_wheel` uses the left paw + wheel
syllables instead and writes figures/session_partition_paw_wheel.{png,pdf}.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session_variance_partition as svp      # noqa: E402  (paths, estimator, jackknife)
import paper_style as ps                      # noqa: E402
from session_filters import find_csv          # noqa: E402

C = svp.COLORS                                 # Lab, Mouse, Session, Noise
SYLLABLES = sys.argv[1] if len(sys.argv) > 1 else 'production'   # or 'paw_wheel'
N_PERM_CONS = 1000
CHANNELS = {'Paw states' if SYLLABLES == 'production' else 'Left paw + wheel': slice(0, 8), 'Whisking': slice(8, 9), 'Licking': slice(9, 10)}
B, S = 40, 10


def main():
    ses, blocks, _ = svp.build(SYLLABLES)
    mean, noise = blocks['profile']                                   # sessions x 400
    ns = np.arange(1, 11)

    def by_channel(comp):                                             # 4 x 400 -> 4 x (all + 3 channels)
        c3 = comp.reshape(4, B, S)
        return np.column_stack([comp.sum(1)] + [c3[:, :, sl].sum((1, 2)) for sl in CHANNELS.values()])

    def stats_of(ses_, rows):
        c = by_channel(svp.partition(ses_, mean[rows], noise[rows]))  # 4 x 4
        sh = c / c.sum(0)
        R = c[1][None] / (c[1][None] + (c[2] + c[3])[None] / ns[:, None])   # 10 x 4: reliability from n sessions
        return np.vstack([sh, (sh[1] - sh[2])[None], (sh[0] + sh[1])[None], R])   # 16 x 4

    est, se, n = svp.jackknife(ses, stats_of)
    lo, hi, p = svp.t_ci(est, se, n)
    share = est[:4]
    names = ['Whole profile'] + list(CHANNELS)
    for j, nm in enumerate(names):
        print(f'{nm:14s} lab {share[0, j]:.2f}  mouse {share[1, j]:.2f}  session {share[2, j]:.2f}  noise {share[3, j]:.2f}'
              f'   mouse - session {est[4, j]:+.2f} [{lo[4, j]:+.2f}, {hi[4, j]:+.2f}], p = {p[4, j]:.3f}'
              f'   R(1 session) {est[6, j]:.2f}, R(3) {est[8, j]:.2f}')
    est_full = svp.partition(ses, mean, noise)
    global P_DIFF
    P_DIFF = p[4]


    # distances between session profiles, and days apart
    sheet = pd.read_csv(find_csv(), header=1).drop_duplicates('eid').set_index('eid')
    dates = pd.to_datetime(ses['session'].map(sheet['date']), errors='coerce').to_numpy()
    D = np.sqrt(((mean[:, None] - mean[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(mean), 1)
    m, l = ses['mouse_name'].to_numpy(), ses['lab'].to_numpy()
    same_m = (m[:, None] == m[None])[iu]
    same_l = (l[:, None] == l[None])[iu]
    DIST = {'Same mouse': D[iu][same_m], 'Other mouse,\nsame lab': D[iu][same_l & ~same_m],
            'Other lab': D[iu][~same_l]}
    days = np.abs((dates[:, None] - dates[None]).astype('timedelta64[D]').astype(float))[iu][same_m]
    DAY = pd.DataFrame({'days': days, 'dist': DIST['Same mouse']}).dropna()
    DAY['bin'] = pd.cut(DAY['days'], [0, 1, 2, 4, 8, 16, 32, 64], right=True)
    DAYS = DAY.groupby('bin', observed=True)['dist'].agg(['mean', 'sem', 'size'])
    DAYS = DAYS[DAYS['size'] >= 10]
    print(DAYS.round(3))

    CONS = consistency(ses, mean, noise)
    if SYLLABLES == 'production':
        # what fig_lda draws (it only reads caches): the whole-profile budget and the distances
        cache = svp.PI / 'figure_generation' / 'cache' / 'session_partition.pkl'
        pd.to_pickle({'share': dict(zip(['lab', 'mouse', 'session', 'noise'], share[:, 0])),
                      'mouse_minus_session': (est[4, 0], lo[4, 0], hi[4, 0], p[4, 0]),
                      'distances': {k.replace('\n', ' '): v for k, v in DIST.items()},
                      'meta': dict(n_sessions=len(ses), n_mice=ses.mouse_name.nunique(), n_labs=ses.lab.nunique(),
                                   ci='leave-one-mouse-out jackknife', written=pd.Timestamp.now().strftime('%d-%m-%Y'))},
                     cache)
        print('wrote', cache.relative_to(svp.PI))
    figure(share, names, est_full, DIST, DAYS, est, lo, hi, ns, CONS)


def consistency(ses, mean, noise, n_perm=N_PERM_CONS, seed=0):
    """Each mouse's own session-to-session variance of the whole profile (summed over features, sampling noise
    removed): the spread of its session profiles around its own mean. Null: the within-mouse deviation vectors
    shuffled across mice (each mouse keeps its number of sessions), re-centred, recomputed. If mice were equally
    consistent, their sorted values would fall inside the null band."""
    m = ses['mouse_name'].to_numpy()
    mice = pd.unique(m)
    groups = [np.flatnonzero(m == x) for x in mice]
    dev = mean - pd.DataFrame(mean).groupby(m).transform('mean').to_numpy()
    nz = noise.sum(1)

    def per_mouse(D, N):
        out = []
        for g in groups:
            Dg = D[g] - D[g].mean(0)
            out.append((Dg ** 2).sum() / (len(g) - 1) - N[g].mean())
        return np.array(out)
    obs = per_mouse(dev, nz)
    rng = np.random.default_rng(seed)
    null_sorted, null_spread = [], []
    for _ in range(n_perm):
        pi = rng.permutation(len(dev))
        v = per_mouse(dev[pi], nz[pi])
        null_sorted.append(np.sort(v)); null_spread.append(v.var())
    null_sorted, null_spread = np.array(null_sorted), np.array(null_spread)
    p = (1 + (null_spread >= obs.var()).sum()) / (1 + n_perm)
    print(f'day-to-day variability per mouse: range {obs.min():.2f}-{obs.max():.2f} '
          f'({obs.max() / max(obs.min(), 1e-9):.0f}-fold); heterogeneity p = {p:.4f}')
    per = pd.DataFrame({'mouse_name': mice, 'lab': [ses['lab'].to_numpy()[g[0]] for g in groups],
                        'n_sessions': [len(g) for g in groups], 'day_to_day_variance': obs})
    if SYLLABLES == 'production':
        per.to_csv(svp.FIGDIR / 'mouse_consistency.csv', index=False)
    return dict(obs=np.sort(obs), lo=np.percentile(null_sorted, 2.5, 0), hi=np.percentile(null_sorted, 97.5, 0), p=p,
                per_mouse=per)


def draw_consistency(ax, CONS, comp, fs):
    """comp: raw components summed over features (lab, mouse, session, noise)."""
    x = np.arange(1, len(CONS['obs']) + 1)
    ax.fill_between(x, CONS['lo'], CONS['hi'], color=ps.NULL_BAND, alpha=0.35, lw=0, label='Equal consistency (null)')
    ax.plot(x, CONS['obs'], 'o', ms=2.6, color=C['Session'], mec='0.3', mew=0.3, label='Mice')
    ax.axhline(comp[2], color=C['Session'], lw=0.8, ls='--', label='Pooled session variance')
    ax.axhline(comp[1], color=C['Mouse'], lw=0.8, label='Between-mouse variance')
    ax.set_xlim(0, len(x) + 1); ax.set_ylim(bottom=min(0, CONS['obs'].min() * 1.1))
    ax.set_xlabel('Mice, sorted'); ax.set_xticks([1, len(x)])
    ax.set_ylabel("A mouse's day-to-day\nvariance")
    ax.legend(frameon=False, fontsize=fs * 0.62, loc='upper left', handlelength=1.4)
    ax.text(0.97, 0.04, f'p = {CONS["p"]:.3f}', transform=ax.transAxes, ha='right', fontsize=fs * 0.7, color='0.3')


def draw_pie(ax, share, lo, hi, p, fs):
    """The whole-profile budget as a pie; the mouse-vs-session test written underneath."""
    keys = [('Lab', 'Lab'), ('Mouse', 'Mouse'), ('Session', 'Session'), ('Noise', 'Noise')]
    v = share[:, 0].clip(0)
    wedges, _ = ax.pie(v, colors=[C[k] for k, _ in keys], startangle=90, counterclock=False,
                       wedgeprops=dict(edgecolor='white', linewidth=0.8))
    for w, (k, lab), val in zip(wedges, keys, v):
        if val < 0.05:
            continue
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        ax.text(0.62 * np.cos(ang), 0.62 * np.sin(ang), f'{lab}\n{100 * val:.0f}%', ha='center', va='center',
                fontsize=fs * 0.75, color='white' if k in ('Lab', 'Mouse') else '0.15')
    ax.set_aspect('equal')
    ax.text(0, -1.25, f'mouse − session = {share[1, 0] - share[2, 0]:+.2f}\n[{lo[4, 0]:+.2f}, {hi[4, 0]:+.2f}], '
                      f'p = {p:.3f}', ha='center', va='top', fontsize=fs * 0.65, color='0.3')
    ax.set_title('Whole syllable profile', fontsize=fs * 0.85, pad=2)


def draw_budget(ax, share, names, lo, hi, fs):
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for k, (key, lab) in enumerate((('Lab', 'Lab'), ('Mouse', 'Mouse'), ('Session', 'Session'), ('Noise', 'Sampling noise'))):
        v = share[k].clip(0)
        ax.barh(y, v, left=left, color=C[key], height=0.7, edgecolor='white', lw=0.4, label=lab)
        for i in range(len(names)):
            if v[i] > 0.09:
                ax.text(left[i] + v[i] / 2, y[i], f'{100 * v[i]:.0f}', ha='center', va='center', fontsize=fs * 0.7,
                        color='white' if key in ('Lab', 'Mouse') else '0.15')
        left += v
    # CI on the mouse | session boundary, and the mouse-vs-session test
    edge = share[0] + share[1]
    ax.errorbar(edge, y, xerr=[edge - lo[5], hi[5] - edge], fmt='none', ecolor='0.1', elinewidth=0.7, capsize=1.5)
    for i in range(len(names)):
        star = ps.stars(P_DIFF[i]) if P_DIFF[i] < 0.05 else 'n.s.'
        ax.text(1.02, y[i], star, transform=ax.get_yaxis_transform(), va='center', fontsize=fs * 0.75, color='0.25')
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.set_ylim(len(names) - 0.5, -0.5); ax.set_xlim(0, 1)
    ax.set_xlabel('Share of session-level variance')
    ax.legend(frameon=False, ncol=2, fontsize=fs * 0.75, loc='lower left', bbox_to_anchor=(-0.05, 1.0),
              handlelength=1, columnspacing=0.8)


def draw_distances(ax, DIST, fs):
    keys = list(DIST)
    parts = ax.violinplot([DIST[k] for k in keys], positions=range(3), showextrema=False, widths=0.75)
    for pc, col in zip(parts['bodies'], [C['Mouse'], C['Session'], C['Lab']]):
        pc.set_facecolor(col); pc.set_alpha(0.35); pc.set_edgecolor('none')
    for i, k in enumerate(keys):
        ax.plot(i, np.median(DIST[k]), 'o', color='0.1', ms=3)
        ax.text(i, np.percentile(DIST[k], 99.5), f'{np.median(DIST[k]):.2f}', ha='center', va='bottom',
                fontsize=fs * 0.7, color='0.3')
    ax.set_xticks(range(3)); ax.set_xticklabels(['Same\nmouse', 'Other mouse,\nsame lab', 'Other\nlab'], fontsize=fs * 0.8)
    ax.set_ylabel('Euclidean distance between\nsession syllable profiles')


def draw_days(ax, DAYS, DIST, fs):
    x = np.arange(len(DAYS))
    ax.errorbar(x, DAYS['mean'], yerr=DAYS['sem'], fmt='o-', ms=3, color=C['Mouse'], label='Same mouse')
    for k, ls in (('Other mouse,\nsame lab', '--'), ('Other lab', ':')):
        ax.axhline(np.mean(DIST[k]), color=C['Lab'], ls=ls, lw=0.8, label=k.replace('\n', ' '))
    ax.set_xticks(x); ax.set_xticklabels([f'≤{int(b.right)}' for b in DAYS.index], fontsize=fs * 0.85)
    ax.set_xlabel('Days between sessions')
    ax.legend(frameon=False, fontsize=fs * 0.65, loc='upper left', handlelength=1.6)
    for i, n in enumerate(DAYS['size']):
        ax.text(i, DAYS['mean'].iloc[i] - DAYS['sem'].iloc[i] - 0.05, f'{n}', ha='center', va='top',
                fontsize=fs * 0.55, color='0.45')


def draw_reliability(ax, names, est, lo, hi, ns, fs):
    styles = {'Whole profile': ('0.15', '-'), list(CHANNELS)[0]: ('#7A5C9E', '--'), 'Whisking': (C['Mouse'], '-'),
              'Licking': ('#C9A13B', ':')}
    for j, n in enumerate(names):
        col, ls = styles[n]
        ax.fill_between(ns, np.clip(lo[6:, j], 0, 1), np.clip(hi[6:, j], 0, 1), color=col, alpha=0.12, lw=0)
        ax.plot(ns, est[6:, j], color=col, ls=ls, label=n)
    ax.set_ylim(0, 1); ax.set_xlim(1, 10); ax.set_xticks([1, 4, 7, 10])
    ax.set_xlabel('Sessions per mouse'); ax.set_ylabel("Reliability of a\nmouse's profile")
    ax.legend(frameon=False, fontsize=fs * 0.65, loc='lower right', handlelength=1.6)


def letters(fig, panels, fs, dy=0.04):
    for letter, (ax, dx) in panels.items():
        b = ax.get_position()
        fig.text(b.x0 - dx, b.y1 + dy, letter, fontsize=fs + 3, fontweight='bold')


def save(fig, name):
    tag = '' if SYLLABLES == 'production' else f'_{SYLLABLES}'
    for path in (svp.FIGDIR / f'{name}{tag}.png', svp.FIGDIR / f'{name}{tag}.pdf'):
        fig.savefig(path, dpi=300)
        print('wrote', path.relative_to(svp.PI))


def figure(share, names, est_full, DIST, DAYS, est, lo, hi, ns, CONS):
    ps.use('paper')
    fs = plt.rcParams['font.size']
    note = ('Unit: the session (mean over its trials). Mouse = variance between mice of the same lab (one rig per lab). '
            'CIs and stars: jackknife over mice.')

    # MAIN, one row: budget (pie), distances, stability across days, consistency as a trait
    fig = plt.figure(figsize=(7.09, 2.35))
    G = fig.add_gridspec(1, 4, width_ratios=[0.85, 1, 1, 1], wspace=0.6, left=0.03, right=0.99, top=0.85, bottom=0.25)
    a = fig.add_subplot(G[0])
    d = fig.add_subplot(G[1])
    e = fig.add_subplot(G[2], sharey=d)
    f = fig.add_subplot(G[3])
    draw_pie(a, share, lo, hi, P_DIFF[0], fs)
    draw_distances(d, DIST, fs)
    draw_days(e, DAYS, DIST, fs)
    draw_consistency(f, CONS, est_full.sum(1), fs)
    letters(fig, {'a': (a, 0.02), 'b': (d, 0.075), 'c': (e, 0.04), 'd': (f, 0.075)}, fs, dy=0.06)
    fig.text(0.03, 0.012, note, fontsize=fs * 0.6, color='0.35')
    save(fig, 'session_partition')

    # SUPPLEMENT: reliability, and the components bin by bin per channel
    fig = plt.figure(figsize=(7.09, 2.3))
    G = fig.add_gridspec(1, 2, width_ratios=[1, 3.2], wspace=0.3, left=0.09, right=0.98, top=0.78, bottom=0.2)
    r = fig.add_subplot(G[0])
    draw_reliability(r, names, est, lo, hi, ns, fs)
    g = G[1].subgridspec(1, 3, wspace=0.1)
    comp = est_full.reshape(4, B, S)
    first = None
    for k, (ch, sl) in enumerate(CHANNELS.items()):
        ax = fig.add_subplot(g[k], sharey=first)
        first = first or ax
        c = comp[:, :, sl].sum(-1).clip(0)
        c = c / c.sum(0)
        ax.stackplot(np.arange(B), c, colors=[C['Lab'], C['Mouse'], C['Session'], C['Noise']], lw=0,
                     labels=['Lab', 'Mouse', 'Session', 'Sampling noise'])
        ps.epoch_lines(ax, B, label=True, names=ps.EPOCH_SHORT, color='white')
        ax.set_xlim(0, B - 1); ax.set_ylim(0, 1); ax.set_xticks([])
        ax.set_title(ch, fontsize=fs * 0.9, pad=12)
        if k:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel('Share of variance\n(per trial bin)')
    first.legend(frameon=False, ncol=4, fontsize=fs * 0.7, loc='upper left', bbox_to_anchor=(0, -0.05),
                 handlelength=1, columnspacing=0.8)
    letters(fig, {'a': (r, 0.07), 'b': (first, 0.06)}, fs, dy=0.1)
    fig.text(0.09, 0.015, note, fontsize=fs * 0.6, color='0.35')
    save(fig, 'session_partition_supp')


if __name__ == '__main__':
    main()
