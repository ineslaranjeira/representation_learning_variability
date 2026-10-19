"""
CARTOON: how a trial's syllables are time-warped onto the 40-bin template
=========================================================================
A mock (synthetic syllables, realistic durations) of what `segmentation/syllable_pipeline.py` does:

  * each video frame (60 Hz) carries one syllable = paw state x whisking state x licking state;
  * each trial is cut into 4 epochs (pre-quiescence, quiescence, choice = go cue -> feedback, ITI);
  * every epoch, whatever its duration, is split into 10 contiguous, near-equal chunks of frames and each chunk
    is replaced by its MODE syllable (`rescale_sequence(..., 10, 'mode')`); an epoch shorter than 10 frames is
    stretched instead, by repeating frames (np.floor(np.linspace(0, n - 1, 10)));
  * so every trial becomes 4 x 10 = 40 bins, aligned on epochs, and a session is the probability of each syllable
    in each of the 40 bins across its trials -- the LDA's session vector.

Writes figure_generation/figures/cartoon_time_warping.{svg,pdf,png} (paper) and, with --talk,
cartoon_time_warping_talk.{svg,pdf,png}: two trials, larger type, for slides.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paper_style as ps

ps.use('paper')
TALK = '--talk' in sys.argv
FS = 1.75 if TALK else 1.0           # type scale: between paper and poster for slides
FPS = 60
N_BINS = 10
EPOCHS = ps.EPOCH_SHORT                       # Pre-Q, Quiesc., Choice, ITI
PAW_CMAP = ListedColormap(ps.PAW_STATE_COLORS)
WHISK_C, LICK_C = ps.SYLLABLE['Whisk'], ps.SYLLABLE['Lick']
EPOCH_TINT = ['#F2F2F2', '#E4E4E4', '#F2F2F2', '#E4E4E4']
rng = np.random.default_rng(7)

# paw-state preferences per epoch (fidgety before quiescence, still during it, vigorous at the choice, mixed after)
PAW_P = {'Pre-Q':   [.20, .25, .10, .12, .15, .08, .07, .03],
         'Quiesc.': [.70, .26, .02, .01, .01, 0, 0, 0],
         'Choice':  [.04, .08, .10, .18, .18, .20, .17, .05],
         'ITI':     [.35, .30, .10, .08, .10, .03, .03, .01]}
WHISK_ON = {'Pre-Q': .45, 'Quiesc.': .30, 'Choice': .75, 'ITI': .80}
LICK_ON = {'Pre-Q': .05, 'Quiesc.': .02, 'Choice': .15, 'ITI': .70}


def sticky(n, p_on_or_dist, stay=0.93):
    """A sticky Markov sequence of n frames: binary (p_on given) or categorical (distribution given)."""
    dist = np.atleast_1d(p_on_or_dist)
    if dist.size == 1:
        dist = np.array([1 - dist[0], dist[0]])
    dist = dist / dist.sum()
    s = np.empty(n, int); s[0] = rng.choice(len(dist), p=dist)
    for i in range(1, n):
        s[i] = s[i - 1] if rng.random() < stay else rng.choice(len(dist), p=dist)
    return s


def make_trial(durations):
    """Per-frame paw, whisk and lick states and the epoch index of each frame."""
    paw, wh, li, ep = [], [], [], []
    for k, (name, d) in enumerate(zip(EPOCHS, durations)):
        n = max(1, int(round(d * FPS)))
        paw.append(sticky(n, PAW_P[name])); wh.append(sticky(n, WHISK_ON[name], 0.96))
        li.append(sticky(n, LICK_ON[name], 0.9)); ep.append(np.full(n, k))
    return [np.concatenate(x) for x in (paw, wh, li, ep)]


def mode(v):
    vals, counts = np.unique(v, return_counts=True)
    return vals[np.argmax(counts)]


def rescale(seq, target=N_BINS):
    """syllable_pipeline.rescale_sequence(seq, target, 'mode'), returning the chunk edges too (for drawing)."""
    n = len(seq)
    if n >= target:
        q, r = divmod(n, target)
        edges = np.r_[0, np.cumsum([q + 1] * r + [q] * (target - r))]
        return np.array([mode(seq[edges[i]:edges[i + 1]]) for i in range(target)]), edges
    idx = np.floor(np.linspace(0, n - 1, target)).astype(int)
    return np.asarray(seq)[idx], None


def warp(paw, wh, li, ep):
    """The trial's 40 bins. The mode is taken on the COMBINED syllable (paw x whisk x lick), as in the pipeline."""
    code = paw * 4 + wh * 2 + li
    bins, edges = [], []
    for k in range(4):
        b, e = rescale(code[ep == k]); bins.append(b); edges.append(e)
    b = np.concatenate(bins)
    return b // 4, (b // 2) % 2, b % 2, edges


# three example trials (s): a fast one, a slow one, and one whose choice epoch is shorter than 10 frames (0.12 s)
TRIALS = [('Trial 1  (fast)', [0.9, 0.55, 0.30, 1.6]),
          ('Trial 2  (slow)', [2.6, 0.70, 1.10, 1.25]),
          ('Trial 3  (choice < 10 frames)', [0.30, 0.45, 0.12, 2.2])]
if TALK:
    TRIALS = [('Fast trial', [0.9, 0.55, 0.30, 1.6]), ('Slow trial', [2.6, 0.70, 1.10, 1.25])]

T_MAX = max(sum(d) for _, d in TRIALS)
W = 4.4                                       # display width of the 40-bin template, in the real-time axis' units
X0 = (T_MAX - W) / 2                          # centre the template under the real-time rows
BW = W / (4 * N_BINS)

fig = plt.figure(figsize=(10, 6.4) if TALK else (7.2, 6.6))
gs = fig.add_gridspec(len(TRIALS) + 1, 1, height_ratios=[1] * len(TRIALS) + [1.15], hspace=0.35,
                      left=0.13 if TALK else 0.13, right=0.985, top=0.93 if TALK else 0.95, bottom=0.1 if TALK else 0.07)


def strip(ax, x0, widths, paw, wh, li, y):
    """Paw strip (height .55) with whisking and licking ticks underneath, starting at x0, one cell per width."""
    xs = x0 + np.r_[0, np.cumsum(widths)]
    for i in range(len(paw)):
        ax.add_patch(Rectangle((xs[i], y + .38), widths[i], .50, color=ps.PAW_STATE_COLORS[paw[i]], lw=0))
        if wh[i]:
            ax.add_patch(Rectangle((xs[i], y + .20), widths[i], .11, color=WHISK_C, lw=0))
        if li[i]:
            ax.add_patch(Rectangle((xs[i], y + .03), widths[i], .11, color=LICK_C, lw=0))
    return xs


for row, (label, dur) in enumerate(TRIALS):
    ax = fig.add_subplot(gs[row]); ax.set_axis_off()
    paw, wh, li, ep = make_trial(dur)
    wp, ww, wl, edges = warp(paw, wh, li, ep)
    # real time (top): one cell per frame
    yr, yw = 2.0, 0.0
    strip(ax, 0, np.full(len(paw), 1 / FPS), paw, wh, li, yr)
    bounds = np.r_[0, np.cumsum([np.sum(ep == k) for k in range(4)])] / FPS
    for k in range(4):
        # chunk edges inside each epoch: where the frames are pooled into one bin
        if edges[k] is not None:
            for e in edges[k][1:-1]:
                ax.plot([bounds[k] + e / FPS] * 2, [yr + .38, yr + .88], color='white', lw=0.35)
        # the warp: a band from the epoch's real-time span to its 10 bins
        tx = X0 + np.array([k, k + 1]) * N_BINS * BW
        ax.add_patch(Polygon([(bounds[k], yr - .02), (bounds[k + 1], yr - .02), (tx[1], yw + .92), (tx[0], yw + .92)],
                             closed=True, color=EPOCH_TINT[k], lw=0, zorder=0))
    for b in bounds:
        ax.plot([b, b], [yr - .02, yr + .9], color='0.25', lw=0.6)
    # template (bottom): 40 equal bins
    strip(ax, X0, np.full(4 * N_BINS, BW), wp, ww, wl, yw)
    for k in range(5):
        ax.plot([X0 + k * N_BINS * BW] * 2, [yw - .02, yw + .9], color='0.25', lw=0.6)
    for k in range(4 * N_BINS + 1):
        ax.plot([X0 + k * BW] * 2, [yw + .38, yw + .88], color='white', lw=0.3)
    ax.text(-0.08, yr + .63, label, ha='right', va='center', fontsize=7 * FS)
    if not TALK:     # at talk size the row labels collide; the legend names the whisking / licking rows
        ax.text(-0.08, yr + .255, 'whisk', ha='right', va='center', fontsize=4.5 * FS, color='0.4')
        ax.text(-0.08, yr + .085, 'lick', ha='right', va='center', fontsize=4.5 * FS, color='0.4')
    ax.text(X0 - 0.08, yw + .63, '40 bins', ha='right', va='center', fontsize=6 * FS, color='0.3')
    ax.text(bounds[-1] + 0.06, yr + .63, f'{bounds[-1]:.1f} s', ha='left', va='center', fontsize=6 * FS, color='0.3')
    if row == 0:
        for k, name in enumerate(EPOCHS):
            ax.text((bounds[k] + bounds[k + 1]) / 2, yr + 1.02, name, ha='center', va='bottom', fontsize=6 * FS)
            ax.text(X0 + (k + .5) * N_BINS * BW, yw - 0.12, '10 bins', ha='center', va='top', fontsize=5 * FS, color='0.4')
    if dur[2] < 10 / FPS:
        k = 2
        ax.annotate('< 10 frames: frames repeated', xy=((bounds[k] + bounds[k + 1]) / 2, yr + .88),
                    xytext=((bounds[k] + bounds[k + 1]) / 2 + 0.45, yr + 1.25), fontsize=5.5 * FS, color='0.3',
                    arrowprops=dict(arrowstyle='-', color='0.4', lw=0.5))
    ax.set_xlim(-0.05, T_MAX + 0.4); ax.set_ylim(-0.25, 3.3)
    if row == (0 if TALK else len(TRIALS) - 1):     # where the row leaves room on the right
        ax.plot([T_MAX - 1, T_MAX], [yr + .45] * 2, color='0.2', lw=0.8)
        ax.text(T_MAX - 0.5, yr + .5, '1 s', ha='center', va='bottom', fontsize=6 * FS)

# session representation: syllable probability per bin over many trials
ax = fig.add_subplot(gs[-1]); ax.set_axis_off()
N_TR = 120
P = np.zeros((4 * N_BINS, 8)); PW = np.zeros(4 * N_BINS); PL = np.zeros(4 * N_BINS)
for _ in range(N_TR):
    d = [rng.uniform(.2, 2.5), rng.uniform(.4, .75), rng.lognormal(np.log(.35), .6), rng.uniform(1, 2.2)]
    wp, ww, wl, _ = warp(*make_trial(d))
    P[np.arange(4 * N_BINS), wp] += 1; PW += ww; PL += wl
P /= N_TR; PW /= N_TR; PL /= N_TR
yb = 0.9
for j in range(4 * N_BINS):
    y = yb
    for s in ps.PAW_VIGOR_ORDER:                # pale-still at the bottom, dark-vigorous on top
        ax.add_patch(Rectangle((X0 + j * BW, y), BW * 0.92, P[j, s] * 1.6, color=ps.PAW_STATE_COLORS[s], lw=0))
        y += P[j, s] * 1.6
    ax.add_patch(Rectangle((X0 + j * BW, 0.45), BW * 0.92, 0.3, color=WHISK_C, alpha=float(PW[j]), lw=0))
    ax.add_patch(Rectangle((X0 + j * BW, 0.05), BW * 0.92, 0.3, color=LICK_C, alpha=float(PL[j]), lw=0))
for k in range(5):
    ax.plot([X0 + k * N_BINS * BW] * 2, [0.0, yb + 1.65], color='0.25', lw=0.6)
for k, name in enumerate(EPOCHS):
    ax.text(X0 + (k + .5) * N_BINS * BW, yb + 1.7, name, ha='center', va='bottom', fontsize=6 * FS)
ax.text(X0 - 0.08, yb + .8, 'P(paw state)', ha='right', va='center', fontsize=6 * FS)
ax.text(X0 - 0.08, 0.6, 'P(whisk)', ha='right', va='center', fontsize=6 * FS)
ax.text(X0 - 0.08, 0.2, 'P(lick)', ha='right', va='center', fontsize=6 * FS)
ax.text(X0 + W + 0.12, yb + .8, 'session vector:\n40 bins x syllable\nprobabilities\n-> LDA', ha='left', va='center',
        fontsize=6 * FS, color='0.25')
ax.text(-0.08, yb + 1.95, 'Session: probability over all trials' if TALK else f'Session (all trials, here {N_TR} mock)', ha='left', va='bottom', fontsize=7 * FS)
ax.set_xlim(-0.05, T_MAX + 0.4); ax.set_ylim(-0.1, 2.9)

# legend: the paw states in vigor order, then whisking and licking
lx, ly, dx = (0.06, 0.045, 0.19) if TALK else (0.06, 0.025, 0.093)
PER_ROW = 5 if TALK else 10
for i, s in enumerate(ps.PAW_VIGOR_ORDER):
    lxi, lyi = lx + (i % PER_ROW) * dx, ly - (i // PER_ROW) * 0.03
    fig.patches.append(Rectangle((lxi, lyi), 0.018, 0.012, color=ps.PAW_STATE_COLORS[s],
                                 transform=fig.transFigure, figure=fig, lw=0.3, ec='0.6'))
    fig.text(lxi + 0.022, lyi + 0.006, ps.paw_label(s), fontsize=5 * FS, va='center')
for i, (n, c) in enumerate([('whisking', WHISK_C), ('licking', LICK_C)]):
    j = 8 + i; lxi, lyi = lx + (j % PER_ROW) * dx, ly - (j // PER_ROW) * 0.03
    fig.patches.append(Rectangle((lxi, lyi), 0.018, 0.012, color=c, transform=fig.transFigure,
                                 figure=fig, lw=0))
    fig.text(lxi + 0.022, lyi + 0.006, n, fontsize=5 * FS, va='center')
fig.text(0.13, 0.975, 'Each epoch: 10 equal chunks of frames  ->  mode syllable per chunk  ->  40-bin template' if TALK else
         'Real time (60 Hz frames)  ->  each epoch split into 10 chunks, mode syllable per chunk  ->  40-bin template',
         fontsize=7 * FS, va='center')

OUT = HERE / 'figures'
for ext in ('svg', 'pdf', 'png'):
    fig.savefig(OUT / f"cartoon_time_warping{'_talk' if TALK else ''}.{ext}", dpi=300)
print('wrote', OUT / f"cartoon_time_warping{'_talk' if TALK else ''}.svg")
