"""
CARTOON (talk version): how trials become the LDA design matrix, in ONE visual language
========================================================================================
Mock data only. A syllable is always a cell in its own colour (paw states ps.PAW_STATE_COLORS, whisking /
licking as the paper's textures: '/' hatch and dark dots, both on light grey), and a probability is always a bar HEIGHT on one scale (P = 1 is U units tall). So:

  1. one trial in real time: one coloured cell per 60 Hz frame (paw strip, whisking row, licking row)
  2. time-warped to 40 bins (each epoch -> 10 chunks, mode syllable per chunk)
  3. many trials, each its own 40-bin strip (a raster)
  4. a session: per bin, how often each syllable occurs -> stacked histogram (paw states stack to 1)
  5. the stack pulled apart, one row per syllable: the session image (same bars, same heights)
  6. flattened bin by bin into one session vector
  7. sessions stacked into the design matrix, labelled by mouse
  8. shrinkage LDA: each mouse a cluster of sessions

Each `draw_*` takes an Axes and draws one step, so the notebook can save them as separate slides.
The mock generator follows cartoon_time_warping.py (that script runs at import, so it cannot be imported).
The paper's reference paw state (dropped before the LDA) is not shown: all 10 rows are drawn.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paper_style as ps

FPS = 60
N_BINS = 10                                  # per epoch
NB = 4 * N_BINS
EPOCHS = ps.EPOCH_SHORT
PAW_P = {'Pre-Q':   [.20, .25, .10, .12, .15, .08, .07, .03],
         'Quiesc.': [.70, .26, .02, .01, .01, 0, 0, 0],
         'Choice':  [.04, .08, .10, .18, .18, .20, .17, .05],
         'ITI':     [.35, .30, .10, .08, .10, .03, .03, .01]}
WHISK_ON = {'Pre-Q': .45, 'Quiesc.': .30, 'Choice': .75, 'ITI': .80}
LICK_ON = {'Pre-Q': .05, 'Quiesc.': .02, 'Choice': .15, 'ITI': .70}
EPOCH_TINT = ['#F2F2F2', '#E4E4E4', '#F2F2F2', '#E4E4E4']
GRID = '0.85'
LINE = '0.3'


def colours():
    """Row order (top -> bottom) and colour of every syllable row: paw states by vigor, then whisk, lick."""
    rows = [('paw', s) for s in ps.PAW_VIGOR_ORDER[::-1]] + [('whisk', None), ('lick', None)]
    cols = [ps.PAW_STATE_COLORS[s] for _, s in rows[:8]] + [ps.SYLLABLE['Whisk'], ps.SYLLABLE['Lick']]
    return rows, cols


def edge(c, f=0.75):
    """A darker edge of the same hue, so pale syllables (paw 0) still read as filled."""
    return tuple(np.array(to_rgb(c)) * f)


# ---------------------------------------------------------------------------------------------------------------
# mock data
# ---------------------------------------------------------------------------------------------------------------
def _sticky(rng, n, dist, stay):
    dist = np.atleast_1d(dist).astype(float)
    if dist.size == 1:
        dist = np.array([1 - dist[0], dist[0]])
    dist /= dist.sum()
    s = np.empty(n, int); s[0] = rng.choice(len(dist), p=dist)
    for i in range(1, n):
        s[i] = s[i - 1] if rng.random() < stay else rng.choice(len(dist), p=dist)
    return s


def make_trial(rng, durations):
    """Per-frame paw, whisk, lick and epoch index."""
    paw, wh, li, ep = [], [], [], []
    for k, (name, d) in enumerate(zip(EPOCHS, durations)):
        n = max(1, int(round(d * FPS)))
        paw.append(_sticky(rng, n, PAW_P[name], 0.93)); wh.append(_sticky(rng, n, WHISK_ON[name], 0.96))
        li.append(_sticky(rng, n, LICK_ON[name], 0.9)); ep.append(np.full(n, k))
    return [np.concatenate(x) for x in (paw, wh, li, ep)]


def _mode(v):
    vals, counts = np.unique(v, return_counts=True)
    return vals[np.argmax(counts)]


def warp(paw, wh, li, ep):
    """40 bins; mode of the COMBINED syllable per chunk (as syllable_pipeline.rescale_sequence)."""
    code = paw * 4 + wh * 2 + li
    out = []
    for k in range(4):
        seq = code[ep == k]; n = len(seq)
        if n >= N_BINS:
            q, r = divmod(n, N_BINS)
            e = np.r_[0, np.cumsum([q + 1] * r + [q] * (N_BINS - r))]
            out.append([_mode(seq[e[i]:e[i + 1]]) for i in range(N_BINS)])
        else:
            out.append(seq[np.floor(np.linspace(0, n - 1, N_BINS)).astype(int)])
    b = np.concatenate(out)
    return b // 4, (b // 2) % 2, b % 2


def onehot(wp, ww, wl):
    """10 x 40 image (rows as colours()): 1 where that syllable occupies the bin."""
    rows, _ = colours()
    M = np.zeros((len(rows), NB))
    for i, (kind, s) in enumerate(rows):
        M[i] = (wp == s) if kind == 'paw' else (ww if kind == 'whisk' else wl)
    return M


def random_durations(rng):
    return [rng.uniform(.2, 2.5), rng.uniform(.4, .75), rng.lognormal(np.log(.35), .6), rng.uniform(1, 2.2)]


def session(rng, n_trials=120):
    """n_trials mock trials warped to 40 bins, and their mean one-hot image (= P(syllable) per bin)."""
    trials = [warp(*make_trial(rng, random_durations(rng))) for _ in range(n_trials)]
    return np.mean([onehot(*t) for t in trials], axis=0), trials


class Mock:
    """Everything the steps share, drawn once from one seed so the slides show the SAME trial / session."""

    def __init__(self, seed=7, durations=(0.9, 0.55, 0.30, 1.6), n_mice=(5, 3, 4, 6), n_trials=120):
        rng = np.random.default_rng(seed)
        self.dur = list(durations)
        self.paw, self.wh, self.li, self.ep = make_trial(rng, self.dur)
        self.wp, self.ww, self.wl = warp(self.paw, self.wh, self.li, self.ep)
        self.M = onehot(self.wp, self.ww, self.wl)
        P, trials = session(rng, n_trials - 1)
        self.trials = [(self.wp, self.ww, self.wl)] + trials   # the example trial is trial 1
        self.n_trials = n_trials
        self.P = (P * (n_trials - 1) + self.M) / n_trials
        self.vec = self.P.T.ravel()                          # bin-major: 10 syllables of bin 1, then bin 2, ...
        rows, of = [], []
        for m, n in enumerate(n_mice):                       # each mouse its own perturbed version of the profile
            proto = np.clip(self.P + rng.normal(0, 0.10, self.P.shape) * (self.P > 0.02), 0, 1)
            for _ in range(n):
                S = np.clip(proto + rng.normal(0, 0.05, self.P.shape) * (proto > 0.02), 0, 1)
                rows.append(S.T.ravel()); of.append(m)
        self.X, self.mouse = np.vstack(rows), np.array(of)
        self.n_mice = list(n_mice)
        self.lda_rng = np.random.default_rng(seed + 1)


class Real:
    """The same attributes as Mock, from REAL data (the paper's syllables and states files). Their paw codes are
    relabelled to the active state set with ps.paw_codes, so call ps.use_paw_states('uniform') before building
    AND drawing (colours, vigor order and side labels are then the paper's):

      * the session: the Proficient session whose 40-bin profile is closest (L1) to the mean profile of all
        sessions -- a typical one -- unless `eid` is given;
      * its example trial: per-frame states from its states file (steps 1-2), the trial with every epoch >= 10
        frames whose epoch durations are closest to the session's medians, unless `trial` is given; its 40 bins
        are the syllable file's own `binned_sequence` (what the pipeline really produced);
      * the raster: the session's trials (trials with a missing bin are left out);
      * the histogram / session image / vector: `profile='all'` (default) the mean over all sessions, each session
        weighted once; `profile='session'` the typical session's own profile;
      * the design matrix: `n_mice` mice with 3-6 sessions each, the typical session's mouse first (mice with
        biased paw dropouts, `avoid_mice`, are not picked).
    """
    EPOCH_LABELS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']

    def __init__(self, syllable_file, states_dir, eid=None, trial=None, n_mice=4, max_sessions=6, seed=0,
                 avoid_mice=('NYU-30', 'CSH_ZAD_026'), profile='all'):
        import pandas as pd
        rng = np.random.default_rng(seed)
        s = pd.read_parquet(syllable_file, columns=['mouse_name', 'broader_label', 'binned_sequence', 'sample'])
        s['eid'] = s['sample'].str[:36]
        s['trial'] = s['sample'].str[37:].astype(float)
        s['e'] = s['broader_label'].map({l: i for i, l in enumerate(self.EPOCH_LABELS)})
        # the stored codes are in the syllable file's own paw numbering (the 19Ago2026 relabel); paw_codes maps
        # them to the ACTIVE state set's numbering -- call ps.use_paw_states('uniform') first, as the paper does
        seq = ps.paw_codes(np.stack(s['binned_sequence'].to_numpy()).astype(float))   # rows x 10 bins
        # one row per (session, trial): 40 bins in epoch order
        s['row'] = np.arange(len(s))
        wide = s.pivot_table(index=['eid', 'trial'], columns='e', values='row', aggfunc='first').dropna()
        idx = wide.to_numpy().astype(int)                                       # trials x 4 epochs
        codes = seq[idx].reshape(len(idx), 4 * N_BINS)                          # trials x 40
        ok = ~np.isnan(codes).any(1)
        codes, keys = codes[ok], wide.index[ok]
        paw, wh, li = ps.decode_syllables(codes)
        eids = keys.get_level_values('eid').to_numpy()
        mouse_of = s.drop_duplicates('eid').set_index('eid')['mouse_name']
        # per-session P(syllable) per bin
        onehots = np.stack([onehot(*t) for t in zip(paw.astype(int), wh.astype(int), li.astype(int))])
        ses = pd.Index(np.unique(eids))
        sid = ses.get_indexer(eids)
        counts = np.bincount(sid)
        Psum = np.zeros((len(ses),) + onehots.shape[1:])
        np.add.at(Psum, sid, onehots)
        Pses = Psum / counts[:, None, None]
        if eid is None:                                                         # typical: closest to the mean
            eid = ses[np.argmin(np.abs(Pses - Pses.mean(0)).sum((1, 2)))]
        self.eid, self.mouse_name = eid, mouse_of[eid]
        k = ses.get_loc(eid)
        sel = np.flatnonzero(eids == eid)
        # the profile drawn in the histogram / session image / vector: 'all' = the mean over ALL sessions (each
        # session weighted once), 'session' = the typical session alone
        self.n_sessions = len(ses)
        self.P = Pses.mean(0) if profile == 'all' else Pses[k]
        self.vec = self.P.T.ravel()
        # example trial (per-frame states from the states file)
        st = pd.read_parquet(f'{states_dir}/8_states_file_{eid}_{self.mouse_name}',
                             columns=['trial_id', 'most_likely_states', 'broader_label'])
        st = st[st['broader_label'].isin(self.EPOCH_LABELS)].dropna(subset=['most_likely_states'])
        n = st.groupby(['trial_id', 'broader_label']).size().unstack().reindex(columns=self.EPOCH_LABELS)
        trials_here = keys[sel].get_level_values('trial').to_numpy()
        n = n.loc[n.index.isin(trials_here)].dropna()
        n = n[(n >= N_BINS).all(1)]
        if trial is None:
            trial = (np.abs(np.log(n) - np.log(n.median()))).sum(1).idxmin()
        self.trial = trial
        t = st[st['trial_id'] == trial]
        fr = [t[t['broader_label'] == l]['most_likely_states'].to_numpy() for l in self.EPOCH_LABELS]
        p_, w_, l_ = ps.decode_syllables(ps.paw_codes(np.concatenate(fr)))
        self.paw, self.wh, self.li = p_.astype(int), w_.astype(int), l_.astype(int)
        self.ep = np.concatenate([np.full(len(f), i) for i, f in enumerate(fr)])
        self.dur = [len(f) / FPS for f in fr]
        j = sel[np.flatnonzero(trials_here == trial)[0]]
        self.wp, self.ww, self.wl = paw[j].astype(int), wh[j].astype(int), li[j].astype(int)
        self.M = onehot(self.wp, self.ww, self.wl)
        others = [i for i in sel if i != j]
        self.trials = [(self.wp, self.ww, self.wl)] + [(paw[i].astype(int), wh[i].astype(int), li[i].astype(int))
                                                       for i in others]
        self.n_trials = len(sel)
        # design matrix: real sessions of real mice
        n_per = pd.Series(mouse_of.reindex(ses).to_numpy()).value_counts()
        cand = [m for m in n_per.index if m != self.mouse_name and 3 <= n_per[m] and m not in avoid_mice]
        mice = [self.mouse_name] + list(rng.choice(cand, n_mice - 1, replace=False))
        rows, of = [], []
        ses_mouse = mouse_of.reindex(ses).to_numpy()
        for mi, m in enumerate(mice):
            ks = np.flatnonzero(ses_mouse == m)[:max_sessions]
            rows += [Pses[kk].T.ravel() for kk in ks]; of += [mi] * len(ks)
        self.X, self.mouse = np.vstack(rows), np.array(of)
        self.n_mice = [int(np.sum(self.mouse == i)) for i in range(n_mice)]
        self.mice = mice


MOUSE_COLS = ['#E9C46A', '#E76F51', '#577590', '#3D3D3D']   # as fig_lda panel A


# ---------------------------------------------------------------------------------------------------------------
# primitives -- ONE scale everywhere: a probability of 1 is U units tall
# ---------------------------------------------------------------------------------------------------------------
U = 5.0          # height of P = 1 (the paw stack's full height)
ROW_GAP = 0.35


# whisking and licking carry a TEXTURE on top of their grey, so they stay recognisable next to the (also greyish)
# paw states, and their rows can be followed from the histogram into the session image:
# whisking = dashed lines, licking = dots. Hatch density is per inch, so it reads the same in every panel.
# whisking and licking carry the paper's TEXTURES (ps.PAW_HATCH, as cartoon_segmentation_wheel): whisking = '/'
# diagonal hatch on light grey, licking = dark dots on the same grey, each bar outlined. They stay recognisable next to the
# greyish paw states, and their rows can be followed from the session summary into the session image.
WHISK_HATCH, LICK_HATCH = ps.PAW_HATCH[(1, 0)] * 3, '.....'                          # '//////', 5 dots per unit (between the paper's and sparser)
HATCH_INK = '0.2'
WL_BG = '#E4E4E4'                                # one light grey under both textures
TEXTURE = {8: (WL_BG, WHISK_HATCH, 0.4), 9: (WL_BG, LICK_HATCH, 0.6)}   # colours() row -> (fill, hatch, hatch lw)


def _textured(ax, x, y, w, h, row, fc, z=2, ec=None, lw=0.25):
    """One cell: the syllable's colour; whisking / licking (rows 8 / 9) as their outlined textures instead."""
    if row in TEXTURE:
        fill, hatch, hlw = TEXTURE[row]                 # a thicker hatch line = bigger dots
        import matplotlib as mpl
        with mpl.rc_context({'hatch.linewidth': hlw}):
            ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=HATCH_INK, hatch=hatch, lw=0.4,
                                   zorder=z))
        return
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or 'none', lw=lw if ec else 0, zorder=z))


def strip(ax, x0, widths, paw, wh, li, y, h_paw=0.5, h_tick=0.11, gap=0.07):
    """Real-time / 40-bin strip: paw cells, whisking (hatched) and licking (dotted) rows underneath.
    Consecutive equal cells are merged so the texture runs continuously."""
    xs = x0 + np.r_[0, np.cumsum(widths)]
    yw, yl = y + h_tick + gap, y
    yp = yw + h_tick + gap
    for i in range(len(paw)):
        ax.add_patch(Rectangle((xs[i], yp), widths[i], h_paw, color=ps.PAW_STATE_COLORS[paw[i]], lw=0))
    for on, yy, row, c in ((wh, yw, 8, ps.SYLLABLE['Whisk']), (li, yl, 9, ps.SYLLABLE['Lick'])):
        on = np.asarray(on).astype(bool)
        d = np.diff(np.r_[0, on.astype(int), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            _textured(ax, xs[a], yy, xs[b] - xs[a], h_tick, row, c)
    return yp + h_paw


def _cell(ax, x, y, w, h, c, z=2, row=None):
    if h > 1e-3:
        _textured(ax, x, y, w, h, row, c, z=z, ec=edge(c))


def hist_layout(P):
    """y of the whisk and lick baselines and of the paw stack's base (lick at the bottom)."""
    h_l, h_w = U * P[9].max(), U * P[8].max()
    y_l = 0.0
    y_w = y_l + h_l + ROW_GAP
    y_p = y_w + h_w + ROW_GAP
    return y_l, y_w, y_p, y_p + U


def draw_stack(ax, P, x0=0.0, cw=1.0, highlight=None):
    """Per bin: P(paw state) stacked to 1 (still at the bottom, vigorous on top), P(whisk), P(lick) underneath."""
    _, cols = colours()
    y_l, y_w, y_p, top = hist_layout(P)
    for j in range(P.shape[1]):
        x = x0 + j * cw + 0.05 * cw
        y = y_p
        for r in range(7, -1, -1):                   # colours() rows 7..0 = vigor order, still first
            _cell(ax, x, y, 0.9 * cw, U * P[r, j], cols[r]); y += U * P[r, j]
        _cell(ax, x, y_w, 0.9 * cw, U * P[8, j], cols[8], row=8)
        _cell(ax, x, y_l, 0.9 * cw, U * P[9, j], cols[9], row=9)
    for yy in (y_l, y_w, y_p):
        ax.plot([x0, x0 + P.shape[1] * cw], [yy, yy], color=GRID, lw=0.4, zorder=1)
    for k in range(5):
        ax.plot([x0 + k * N_BINS * cw] * 2, [-0.15, top], color=LINE, lw=0.6, zorder=3)
    if highlight is not None:
        ax.add_patch(Rectangle((x0 + highlight * cw, -0.25), cw, top + 0.5, fill=False, ec='0.1', lw=0.8, zorder=6))
    return y_l, y_w, y_p, top


def explode_layout(P):
    """The stack pulled apart: each syllable its own row, as tall as its largest P (same U scale), rows in the
    stack's order (vigorous paw on top ... still paw, then whisk, lick). Returns each row's baseline and top."""
    h = U * P.max(axis=1)
    base = np.zeros(len(h)); y = 0.0
    for r in range(len(h) - 1, -1, -1):              # bottom-up: lick, whisk, paw still ... vigorous
        base[r] = y; y += h[r] + ROW_GAP
    return base, base + h


def draw_exploded(ax, P, x0=0.0, cw=1.0, highlight=None, n_first=0):
    """Session image: the same bars as draw_stack, one row per syllable."""
    _, cols = colours()
    base, tops = explode_layout(P)
    for r in range(P.shape[0]):
        ax.plot([x0, x0 + P.shape[1] * cw], [base[r]] * 2, color=GRID, lw=0.4, zorder=1)
        for j in range(P.shape[1]):
            _cell(ax, x0 + j * cw + 0.05 * cw, base[r], 0.9 * cw, U * P[r, j], cols[r], row=r)
    top = tops.max()
    for k in range(5):
        ax.plot([x0 + k * N_BINS * cw] * 2, [-0.15, top], color=LINE, lw=0.6, zorder=3)
    if highlight is not None:
        ax.add_patch(Rectangle((x0 + highlight * cw, -0.25), cw, top + 0.5, fill=False, ec='0.1', lw=0.8, zorder=6))
    for j in range(n_first):
        ax.add_patch(Rectangle((x0 + j * cw, -0.25), cw, top + 0.5, fill=False, ec='0.1', lw=0.7, zorder=6))
    return base, top


def epoch_labels(ax, x0, cw, y, fs):
    for k, name in enumerate(EPOCHS):
        ax.text(x0 + (k + .5) * N_BINS * cw, y, name, ha='center', va='bottom', fontsize=fs)


def exploded_labels(ax, P, x, fs, side='left'):
    """'paw states' bracket over the 8 paw rows, then whisk and lick, next to draw_exploded."""
    base, tops = explode_layout(P)
    ha = 'left' if side == 'left' else 'right'
    ax.text(x, (base[9] + tops[9]) / 2, 'lick', ha=ha, va='center', fontsize=fs, color='0.25')
    ax.text(x, (base[8] + tops[8]) / 2, 'whisk', ha=ha, va='center', fontsize=fs, color='0.25')
    ax.text(x, (base[7] + tops[0]) / 2, 'paw\nstates', ha=ha, va='center', fontsize=fs, color='0.25',
            linespacing=0.95)



def _clean(ax):
    ax.set_axis_off()


# ---------------------------------------------------------------------------------------------------------------
# the steps
# ---------------------------------------------------------------------------------------------------------------
def draw_trial(ax, m, fs=7, scalebar=True):
    """1. one trial in real time: one cell per 60 Hz frame."""
    T = len(m.paw) / FPS
    top = strip(ax, 0, np.full(len(m.paw), 1 / FPS), m.paw, m.wh, m.li, 0)
    bounds = np.r_[0, np.cumsum([np.sum(m.ep == k) for k in range(4)])] / FPS
    for b in bounds:
        ax.plot([b, b], [-0.02, top + 0.02], color=LINE, lw=0.6)
    for k, name in enumerate(EPOCHS):
        ax.text((bounds[k] + bounds[k + 1]) / 2, top + 0.08, name, ha='center', va='bottom', fontsize=fs)
    ax.text(-0.05, 0.36 + 0.25, 'paw', ha='right', va='center', fontsize=fs * 0.85, color='0.25')
    ax.text(-0.05, 0.18 + 0.055, 'whisk', ha='right', va='center', fontsize=fs * 0.85, color='0.25')
    ax.text(-0.05, 0.055, 'lick', ha='right', va='center', fontsize=fs * 0.85, color='0.25')
    if scalebar:
        ax.plot([T - 1, T], [-0.15] * 2, color='0.2', lw=0.8)
        ax.text(T - 0.5, -0.2, '1 s', ha='center', va='top', fontsize=fs * 0.85)
    ax.set_xlim(-0.6, T + 0.05); ax.set_ylim(-0.45, top + 0.35)
    _clean(ax)
    return bounds, top


def draw_warp(ax, m, fs=7, W=None):
    """2. real time -> 40 equal bins (each epoch 10 chunks, mode syllable)."""
    T = len(m.paw) / FPS
    W = W or T
    yr = 1.6
    top = strip(ax, 0, np.full(len(m.paw), 1 / FPS), m.paw, m.wh, m.li, yr)
    bounds = np.r_[0, np.cumsum([np.sum(m.ep == k) for k in range(4)])] / FPS
    bw = W / NB
    x0 = (T - W) / 2
    strip(ax, x0, np.full(NB, bw), m.wp, m.ww, m.wl, 0)
    for k in range(4):
        tx = x0 + np.array([k, k + 1]) * N_BINS * bw
        ax.add_patch(Polygon([(bounds[k], yr - .03), (bounds[k + 1], yr - .03), (tx[1], .9), (tx[0], .9)],
                             closed=True, color=EPOCH_TINT[k], lw=0, zorder=0))
    for b in bounds:
        ax.plot([b, b], [yr - .02, top + .02], color=LINE, lw=0.6)
    for k in range(NB + 1):
        ax.plot([x0 + k * bw] * 2, [0.36, 0.86], color='white', lw=0.3)
    for k in range(5):
        ax.plot([x0 + k * N_BINS * bw] * 2, [-0.02, 0.88], color=LINE, lw=0.6)
    for k, name in enumerate(EPOCHS):
        ax.text((bounds[k] + bounds[k + 1]) / 2, top + 0.08, name, ha='center', va='bottom', fontsize=fs)
    ax.text(-0.05, yr + 0.5, 'real time', ha='right', va='center', fontsize=fs * 0.85, color='0.25')
    ax.text(x0 - 0.05, 0.5, '40 bins', ha='right', va='center', fontsize=fs * 0.85, color='0.25')
    ax.set_xlim(-0.75, T + 0.05); ax.set_ylim(-0.15, top + 0.35)
    _clean(ax)


def draw_raster(ax, m, fs=7, n_show=12):
    """3. many trials, each warped to the same 40 bins, stacked: one row per trial (the example trial on top)."""
    cw = 1.0
    hp, ht, g = 0.55, 0.2, 0.04
    rowh = hp + 2 * ht + 3 * g + 0.25
    for i in range(n_show):
        wp, ww, wl = m.trials[i]
        strip(ax, 0, np.full(NB, cw), wp, ww, wl, (n_show - 1 - i) * rowh, h_paw=hp, h_tick=ht, gap=g)
    H = n_show * rowh
    for k in range(5):
        ax.plot([k * N_BINS * cw] * 2, [-0.1, H], color=LINE, lw=0.6)
    epoch_labels(ax, 0, cw, H + 0.15, fs)
    for y, t in [(H - rowh / 2, 'trial 1'), (H - 1.5 * rowh, 'trial 2'), (rowh / 2, f'trial {m.n_trials}')]:
        ax.text(-0.8, y, t, ha='right', va='center', fontsize=fs * 0.85, color='0.25')
    ax.text(-0.8, H / 2 - rowh, '⋮', ha='right', va='center', fontsize=fs, color='0.25')
    ax.set_xlim(-7, NB + 0.5); ax.set_ylim(-0.3, H + 1.3)
    _clean(ax)


def draw_hist(ax, m, fs=7):
    """4. count each bin's syllables over the trials: a stacked histogram per bin = the session."""
    y_l, y_w, y_p, top = draw_stack(ax, m.P)
    epoch_labels(ax, 0, 1.0, top + 0.3, fs)
    h_w, h_l = U * m.P[8].max(), U * m.P[9].max()
    for y, t in [((y_p + top) / 2, 'P(paw state)'), (y_w + h_w / 2, 'P(whisk)'), (y_l + h_l / 2, 'P(lick)')]:
        ax.text(-0.8, y, t, ha='right', va='center', fontsize=fs * 0.85, color='0.25')
    ax.set_xlim(-9, NB + 0.5); ax.set_ylim(-0.5, top + 1.5)
    _clean(ax)


def draw_unstack(ax, m, fs=7, j=24):
    """5. the same bars pulled apart, one row per syllable: the histogram becomes the session image."""
    _, _, _, top_s = draw_stack(ax, m.P, highlight=j)
    X1 = NB + 9
    base, top_e = draw_exploded(ax, m.P, x0=X1, highlight=j)
    epoch_labels(ax, 0, 1.0, top_s + 0.3, fs)
    epoch_labels(ax, X1, 1.0, top_e + 0.3, fs)
    exploded_labels(ax, m.P, X1 + NB + 0.6, fs * 0.85)
    ym = top_s / 2
    ax.annotate('', xy=(X1 - 1.0, ym), xytext=(NB + 1.0, ym),
                arrowprops=dict(arrowstyle='-|>', color='0.4', lw=0.8, mutation_scale=7))
    ax.text(NB + 4.5, ym + 0.5, 'one row\nper syllable', ha='center', va='bottom', fontsize=fs * 0.85, color='0.25')
    ax.set_xlim(-0.5, X1 + NB + 6); ax.set_ylim(-0.6, max(top_s, top_e) + 1.6)
    _clean(ax)


def draw_flatten(ax, m, fs=7, n_show=3):
    """6. image -> vector: bin 1's column of syllables, then bin 2's, ... (the first bins outlined)."""
    _, cols = colours()
    base, top = draw_exploded(ax, m.P, n_first=n_show)
    epoch_labels(ax, 0, 1.0, top + 0.3, fs)
    exploded_labels(ax, m.P, -0.8, fs * 0.85, side='right')
    hv = 0.5 * U                                         # the vector: same colours, heights at half scale
    yv = -hv - 6.0
    vw = NB / m.vec.size                                 # spans the image's width
    for i, v in enumerate(m.vec):
        if v > 0.005:
            ax.add_patch(Rectangle((i * vw, yv), vw, hv * v, facecolor=cols[i % 10], lw=0))
    ax.plot([0, NB], [yv, yv], color=GRID, lw=0.4)
    for k in range(5):
        ax.plot([k * N_BINS] * 2, [yv - 0.2, yv + hv], color=LINE, lw=0.6)
    for j in range(n_show):
        ax.add_patch(Rectangle((j * 10 * vw, yv - 0.1), 10 * vw, hv + 0.2, fill=False, ec='0.1', lw=0.5, zorder=5))
    ax.annotate('', xy=(n_show * 10 * vw / 2, yv + hv + 0.4), xytext=(n_show / 2, -0.5),
                arrowprops=dict(arrowstyle='-|>', color='0.4', lw=0.8, mutation_scale=7))
    ax.text(n_show + 1.0, (yv + hv) / 2 - 0.3, 'flatten, bin by bin', ha='left', va='center', fontsize=fs * 0.85,
            color='0.25')
    ax.text(NB / 2, yv - 0.6, 'session vector (40 bins × 10 syllables)', ha='center', va='top', fontsize=fs * 0.85,
            color='0.25')
    ax.set_xlim(-7, NB + 0.5); ax.set_ylim(yv - 2.4, top + 1.6)
    _clean(ax)



def draw_design(ax, m, fs=7):
    """6. one session vector per row, labelled by mouse."""
    _, cols = colours()
    n, F = m.X.shape
    rh, gap = 1.0, 0.15
    vw = 40.0 / F
    for r in range(n):
        y = (n - 1 - r) * (rh + gap)
        for i in np.flatnonzero(m.X[r] > 0.005):
            ax.add_patch(Rectangle((i * vw, y), vw, rh * m.X[r, i], facecolor=cols[i % 10], lw=0))
        ax.plot([0, 40], [y, y], color=GRID, lw=0.3)
        ax.add_patch(Rectangle((-1.6, y + 0.05), 1.0, rh - 0.1, color=MOUSE_COLS[m.mouse[r]], lw=0))
        if r and m.mouse[r] != m.mouse[r - 1]:
            ax.plot([-1.6, 40], [y + rh + gap / 2] * 2, color='0.5', lw=0.5)
    H = n * (rh + gap)
    for k in range(5):
        ax.plot([k * 10] * 2, [-0.1, H], color=LINE, lw=0.6)
    for k, name in enumerate(EPOCHS):
        ax.text(k * 10 + 5, H + 0.3, name, ha='center', va='bottom', fontsize=fs)
    ax.text(-2.2, H / 2, 'sessions', ha='right', va='center', rotation=90, fontsize=fs * 0.85, color='0.25')
    ax.text(20, -0.8, 'features', ha='center', va='top', fontsize=fs * 0.85, color='0.25')
    ax.set_xlim(-3.5, 40.3); ax.set_ylim(-2.2, H + 1.6)
    _clean(ax)


def draw_lda(ax, m, fs=7):
    """7. shrinkage LDA: each mouse a cluster of sessions (mock)."""
    rng = np.random.default_rng(12)
    cent = np.array([[-2.2, 0.8], [-0.6, -1.6], [1.0, 1.6], [2.3, -0.4]])
    for k, (cx, cy) in enumerate(cent):
        pts = rng.normal(0, 0.38, (m.n_mice[k], 2)) + (cx, cy)
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color=MOUSE_COLS[k], lw=0, alpha=0.9)
        ax.plot(cx, cy, '+', ms=6, mew=1.2, color=MOUSE_COLS[k])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('LD1', fontsize=fs, labelpad=2); ax.set_ylabel('LD2', fontsize=fs, labelpad=2)
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.8, 2.9); ax.set_box_aspect(1)


def legend(ax, fs=6, ncol=5):
    """Syllable colour key in vigor order (paw states), then whisking and licking."""
    items = [(ps.paw_label(s), ps.PAW_STATE_COLORS[s]) for s in ps.PAW_VIGOR_ORDER] + \
            [('whisking', ps.SYLLABLE['Whisk']), ('licking', ps.SYLLABLE['Lick'])]
    for i, (name, c) in enumerate(items):
        x, y = (i % ncol), -(i // ncol)
        _textured(ax, x, y, 0.18, 0.5, {8: 8, 9: 9}.get(i), c, ec=edge(c), lw=0.3)
        ax.text(x + 0.25, y + 0.25, name, va='center', fontsize=fs)
    ax.set_xlim(-0.1, ncol); ax.set_ylim(-((len(items) - 1) // ncol) - 0.2, 0.7)
    _clean(ax)
