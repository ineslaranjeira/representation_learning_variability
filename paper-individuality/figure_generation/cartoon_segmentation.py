"""
CARTOON: the segmentation pipeline, from pose-estimated signals to per-frame syllables
=====================================================================================
A mock (synthetic signals, nothing in it is a measurement) of what segmentation/ does:

  * paw: x and y velocity of each paw (Lightning Pose, 60 Hz) -> Morlet wavelet power at 0.5-8 Hz
    (3.1.1_paw_wavelets) -> session z-score -> PCA (95% variance) -> k-means, k = 8, numbered stillest to most
    vigorous, every frame labelled by its nearest centroid (3.3_wavelet_clusters_uniform);
  * whisking: whisker motion energy -> 2-state Gaussian HMM (high / low; 4.0_hmm_fit);
  * licking: lick count -> 2-state Poisson HMM (licking / not);
  * syllable = paw state x whisking x licking, 8 x 2 x 2 = 32 per frame (syllable_pipeline).

`draw(fig, spec)` is what fig_segmentation.ipynb calls for panel b. Run as a script it writes
figure_generation/figures/cartoon_segmentation.{svg,pdf,png} at the panel's size.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paper_style as ps

T = 150                                        # frames in the mock snippet (2.5 s at 60 Hz)
FREQS = [0.5, 1, 2, 4, 8]
WHISK_C, LICK_C = '#B8B8B8', '#484949'         # as fig_segmentation's whisking / licking states
ARROW = dict(arrowstyle='-|>', color='0.35', lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0)

# rows (y centre, half height) and columns (x span) in the panel's own 0-100 x 0-32 frame
ROW = {'paw': (21.5, 5.2), 'whisk': (10.2, 2.6), 'lick': (3.0, 2.6)}
COL = {'raw': (12, 30), 'wav': (35, 51), 'km': (55, 69), 'state': (73, 85), 'syl': (89, 100)}


def sticky(rng, n, dist, stay):
    dist = np.asarray(dist, float) / np.sum(dist)
    s = np.empty(n, int); s[0] = rng.choice(len(dist), p=dist)
    for i in range(1, n):
        s[i] = s[i - 1] if rng.random() < stay else rng.choice(len(dist), p=dist)
    return s


def mock(seed=3):
    """Per-frame paw state, whisking and licking, and signals that look like what they were fit to."""
    rng = np.random.default_rng(seed)
    order = list(ps.PAW_VIGOR_ORDER)
    vig = np.empty(ps.N_PAW_STATES); vig[order] = np.linspace(0, 1, len(order))   # 0 = stillest, 1 = most vigorous
    paw = sticky(rng, T, [3, 3, 2, 2, 2, 2, 2, 1], 0.93)
    whisk = np.repeat([0, 1, 0, 1], [30, 50, 25, 45])            # two whisking bouts
    lick = np.repeat([0, 1, 0, 1, 0], [45, 25, 30, 30, 20])         # two licking bouts
    t = np.arange(T) / 60
    v = vig[paw]
    side = np.array([{'left': 1, 'right': -1}.get(ps.PAW_SIDE.get(s, ''), 0) for s in paw])
    # velocity: amplitude and frequency both rise with vigor; the leading side moves more
    f = 1 + 6 * v
    vel_l = (0.15 + v * (1 + 0.6 * side)) * np.sin(2 * np.pi * np.cumsum(f) / 60 + 0.4) + 0.08 * rng.normal(size=T)
    vel_r = (0.15 + v * (1 - 0.6 * side)) * np.sin(2 * np.pi * np.cumsum(f * 0.9) / 60 + 1.7) + 0.08 * rng.normal(size=T)
    # wavelet power: each paw's power sits at the band nearest its movement frequency, smoothed in time
    k = np.exp(-0.5 * (np.arange(-6, 7) / 3) ** 2); k /= k.sum()
    def spec(amp):
        P = np.array([amp * np.exp(-0.5 * ((np.log2(f) - np.log2(b)) / 0.9) ** 2) for b in FREQS])
        return np.array([np.convolve(r, k, mode='same') for r in P])
    pow_l, pow_r = spec(0.15 + v * (1 + 0.6 * side)), spec(0.15 + v * (1 - 0.6 * side))
    me = 0.25 + 0.55 * whisk + 0.07 * rng.normal(size=T) + 0.06 * whisk * np.sin(2 * np.pi * 9 * t)
    licks = (rng.random(T) < np.where(lick == 1, 0.16, 0.0)).astype(float)
    return dict(t=t, paw=paw, whisk=whisk, lick=lick, vel=(vel_l, vel_r), pow=(pow_l, pow_r), me=me, licks=licks,
                vig=vig)


def _y(row, frac):
    """A y inside a row: frac 0 = bottom edge, 1 = top edge."""
    c, h = ROW[row]
    return c - h + 2 * h * frac


def _trace(ax, row, col, y, color, lo=None, hi=None, lw=0.6, y0=0.0, y1=1.0):
    x = np.linspace(*COL[col], len(y))
    lo = np.min(y) if lo is None else lo; hi = np.max(y) if hi is None else hi
    ax.plot(x, _y(row, y0 + (y1 - y0) * (y - lo) / (hi - lo)), color=color, lw=lw, solid_capstyle='butt')


def _strip(ax, row, col, values, cmap, vmin, vmax, y0=0.15, y1=0.85):
    ax.imshow(np.asarray(values, float)[None, :], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation='nearest', extent=[*COL[col], _y(row, y0), _y(row, y1)], zorder=2)
    ax.add_patch(Rectangle((COL[col][0], _y(row, y0)), np.diff(COL[col])[0], _y(row, y1) - _y(row, y0),
                           fill=False, ec='0.45', lw=0.3, zorder=3))


def _arrow(ax, x0, x1, y):
    ax.annotate('', xy=(x1, y), xytext=(x0, y), arrowprops=ARROW, zorder=4)


def _box(ax, col_from, col_to, row, text, fs):
    x0, x1 = COL[col_from][0], COL[col_to][1]
    c, h = ROW[row]
    ax.add_patch(FancyBboxPatch((x0, c - h * 0.72), x1 - x0, 1.44 * h, boxstyle='round,pad=0,rounding_size=1.0',
                                fc='#F4F4F4', ec='0.55', lw=0.4, zorder=2))
    ax.text((x0 + x1) / 2, c, text, ha='center', va='center', fontsize=fs, color='0.2', zorder=3)


def draw(fig, spec, seed=3):
    ax = fig.add_subplot(spec)
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 32); ax.set_axis_off()
    fs = plt.rcParams['font.size'] * 0.72
    small = fs * 0.88
    m = mock(seed)
    paw_cmap = ListedColormap(ps.PAW_STATE_COLORS)

    # column headers
    for col, name in [('raw', 'Pose / video'), ('wav', 'Wavelets'), ('km', 'Clustering'), ('state', 'States'),
                      ('syl', 'Syllables')]:
        ax.text(np.mean(COL[col]), 31.5, name, ha='center', va='top', fontsize=fs, fontweight='bold', color='0.15')
    # row labels
    for row, name in [('paw', 'Paws'), ('whisk', 'Whisker'), ('lick', 'Tongue')]:
        ax.text(0, ROW[row][0], name, ha='left', va='center', fontsize=fs, color='0.15')

    # --- raw signals
    vl, vr = m['vel']
    lim = np.abs(np.r_[vl, vr]).max()
    _trace(ax, 'paw', 'raw', vl, ps.PAW_LEFT, -lim, lim, y0=0.52, y1=1.0)
    _trace(ax, 'paw', 'raw', vr, ps.PAW_RIGHT, -lim, lim, y0=0.0, y1=0.48)
    ax.text(COL['raw'][0], _y('paw', 1.0) + 0.3, 'velocity x, y (L, R)', ha='left', va='bottom', fontsize=small,
            color='0.35')
    _trace(ax, 'whisk', 'raw', m['me'], '0.45', y0=0.1, y1=0.9)
    ax.text(COL['raw'][0], _y('whisk', 1.0) + 0.1, 'motion energy', ha='left', va='bottom', fontsize=small, color='0.35')
    for x in np.linspace(*COL['raw'], T)[m['licks'] > 0]:
        ax.plot([x, x], [_y('lick', 0.15), _y('lick', 0.85)], color='0.1', lw=0.45)
    ax.text(COL['raw'][0], _y('lick', 1.0) + 0.1, 'licks', ha='left', va='bottom', fontsize=small, color='0.35')

    # --- paw: wavelet power (one tint ladder per paw, slow band at the bottom)
    pl, pr = m['pow']
    vmax = max(pl.max(), pr.max())
    for P, col_, (y0, y1) in [(pl, ps.PAW_LEFT, (0.52, 1.0)), (pr, ps.PAW_RIGHT, (0.0, 0.48))]:
        cmap = LinearSegmentedColormap.from_list('', ['white', col_])
        ax.imshow(P, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, origin='lower', interpolation='bilinear',
                  extent=[*COL['wav'], _y('paw', y0), _y('paw', y1)], zorder=2)
        ax.add_patch(Rectangle((COL['wav'][0], _y('paw', y0)), np.diff(COL['wav'])[0], _y('paw', y1) - _y('paw', y0),
                               fill=False, ec='0.45', lw=0.3, zorder=3))
    ax.text(np.mean(COL['wav']), _y('paw', 0.0) - 0.3, 'Morlet, 0.5–8 Hz', ha='center', va='top', fontsize=small,
            color='0.35')

    # --- paw: PCA + k-means, clusters laid out along vigor
    rng = np.random.default_rng(seed + 1)
    (x0, x1), (yb, yt) = COL['km'], (_y('paw', 0.22), _y('paw', 1.0))
    for s in range(ps.N_PAW_STATES):
        v = m['vig'][s]
        side = {'left': 1, 'right': -1}.get(ps.PAW_SIDE.get(s, ''), 0)
        cx = x0 + (0.12 + 0.76 * v) * (x1 - x0)
        cy = yb + (0.5 + 0.36 * side + 0.06 * np.sin(5 * v)) * (yt - yb)
        n = 14
        ax.scatter(cx + rng.normal(0, 0.55, n), cy + rng.normal(0, 0.45, n), s=1.6, color=ps.PAW_STATE_COLORS[s],
                   edgecolors='0.35', linewidths=0.1, zorder=3)
    ax.annotate('', xy=(x1, _y('paw', 0.08)), xytext=(x0, _y('paw', 0.08)),
                arrowprops=dict(arrowstyle='-|>', color='0.55', lw=0.4, mutation_scale=3))
    ax.text(np.mean(COL['km']), _y('paw', 0.0) - 0.3, 'PCA + k-means, k = 8', ha='center', va='top', fontsize=small,
            color='0.35')
    ax.text(x0, _y('paw', 0.08) + 0.15, 'vigor', ha='left', va='bottom', fontsize=small * 0.85, color='0.5')

    # --- whisking and licking: a two-state HMM each, spanning the paw's two middle columns
    _box(ax, 'wav', 'km', 'whisk', '2-state Gaussian HMM', small)
    _box(ax, 'wav', 'km', 'lick', '2-state Poisson HMM', small)

    # --- states
    _strip(ax, 'paw', 'state', m['paw'], paw_cmap, -0.5, ps.N_PAW_STATES - 0.5, 0.25, 0.75)
    ax.text(np.mean(COL['state']), _y('paw', 0.25) - 0.3, '8 paw states', ha='center', va='top', fontsize=small,
            color='0.35')
    _strip(ax, 'whisk', 'state', m['whisk'], ListedColormap(['white', WHISK_C]), -0.5, 1.5)
    _strip(ax, 'lick', 'state', m['lick'], ListedColormap(['white', LICK_C]), -0.5, 1.5)

    # --- arrows along each row
    for row in ROW:
        y = ROW[row][0]
        _arrow(ax, COL['raw'][1] + 0.6, COL['wav'][0] - 0.6, y)
        _arrow(ax, COL['km'][1] + 0.6, COL['state'][0] - 0.6, y)
    _arrow(ax, COL['wav'][1] + 0.6, COL['km'][0] - 0.3, ROW['paw'][0])

    # --- syllable: the three strips merge into one, paw colour with whisking / licking hatched over it
    yc = (ROW['paw'][0] + ROW['lick'][0]) / 2
    xm = COL['state'][1] + 1.3
    for row in ROW:
        ax.plot([COL['state'][1] + 0.3, xm], [ROW[row][0], ROW[row][0]], color='0.35', lw=0.5)
    ax.plot([xm, xm], [ROW['lick'][0], ROW['paw'][0]], color='0.35', lw=0.5)
    _arrow(ax, xm, COL['syl'][0] - 0.4, yc)
    h = 2.6
    sx = np.linspace(*COL['syl'], T + 1)
    ax.imshow(m['paw'][None, :].astype(float), aspect='auto', cmap=paw_cmap, vmin=-0.5, vmax=ps.N_PAW_STATES - 0.5,
              interpolation='nearest', extent=[*COL['syl'], yc - h, yc + h], zorder=2)
    code = m['whisk'] + 2 * m['lick']
    cuts = np.r_[0, np.flatnonzero(np.diff(code)) + 1, T]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if code[a]:
            ax.add_patch(Rectangle((sx[a], yc - h), sx[b] - sx[a], 2 * h, fill=False, ec='0.25', lw=0,
                                   hatch=ps.PAW_HATCH[(code[a] % 2, code[a] // 2)] * 2, zorder=3))   # doubled: the strip is narrow
    ax.add_patch(Rectangle((COL['syl'][0], yc - h), np.diff(COL['syl'])[0], 2 * h, fill=False, ec='0.45', lw=0.3,
                           zorder=4))
    ax.text(np.mean(COL['syl']), yc + h + 0.4, 'paw\n× whisk (/)\n× lick (·)', ha='center', va='bottom', fontsize=small,
            color='0.35', linespacing=1.0)
    ax.text(np.mean(COL['syl']), yc - h - 0.4, '32 per frame', ha='center', va='top', fontsize=small, color='0.35')
    return [ax]


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    ps.use('paper')
    ps.use_paw_states('uniform')
    plt.rcParams['hatch.linewidth'] = 0.4
    fig = plt.figure(figsize=(3.6, 1.25))
    draw(fig, fig.add_gridspec(1, 1, left=0.01, right=0.99, top=0.98, bottom=0.02)[0])
    OUT = HERE / 'figures'
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(OUT / f'cartoon_segmentation.{ext}', dpi=600)
    print('wrote', OUT / 'cartoon_segmentation.svg')
