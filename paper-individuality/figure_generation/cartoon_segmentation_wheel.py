"""
CARTOON: the segmentation pipeline with the wheel -- cartoon_segmentation.py plus a wheel row under the paws
====================================================================================================
A mock (synthetic signals, nothing in it is a measurement), as cartoon_segmentation, of what segmentation/ does,
now including 3.4's wheel MOVEMENT segmentation:

  * wheel: wheel velocity (rotary encoder, 60 Hz) -> IBL movements (between stillness), each one's signed
    velocity time-normalised -> PCA + Gaussian mixture, split by direction first (CW / balanced / CCW; 20 clusters) -> 8 states
    (still, small, CW / balanced / CCW x mid / high; wheel_movement_clusters.py)
  * paw, whisking, licking: unchanged (cartoon_segmentation's mock and drawing helpers are reused)
  * whisking and licking states are textures, not greys: whisking = '/' hatch, licking = '.' hatch (ps.PAW_HATCH)
  * syllables as they are now: the paw, wheel, whisking and licking states joined frame by frame (8 x 8 x 2 x 2):
    paw = base colour, wheel = horizontal stripes in the wheel colour while it moves, whisking '/', licking '.'
  * no bold column headers (cs.draw's are removed)

`draw(fig, spec)` is what 1_segmentation/fig_segmentation_wheel.ipynb calls for panel b; `draw(fig, spec, variant='6')`
draws the alternative wheel row of fig_segmentation_wheel6.ipynb: ONE Gaussian mixture on all movements (no direction
split, no merging; K = 6, representation_learning/wheel_movements/nosplit_states.py), states coloured by vigor only
(one hue: they carry no direction). Run as a script it
writes figure_generation/figures/cartoon_segmentation_wheel.{svg,pdf,png}. cartoon_segmentation.py is untouched.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'segmentation'))
import paper_style as ps
import cartoon_segmentation as cs

WHEEL_STILL = '#ECECEC'
# paw on top (a paw state always means something; a wheel state only exists while the wheel moves), then the
# wheel, then cartoon_segmentation's whisker and tongue rows; cs.draw is pointed at the moved paw row
ROW = {'paw': (35.5, 5.2), 'wheel': (21.2, 3.0), 'whisk': cs.ROW['whisk'], 'lick': cs.ROW['lick']}
CS_ROW = {k: ROW[k] for k in ('paw', 'whisk', 'lick')}
YTOP = 43.0
WHISK_HATCH, LICK_HATCH = ps.PAW_HATCH[(1, 0)], ps.PAW_HATCH[(0, 1)]   # '//' and '..'
HATCH_INK = '0.2'
WL_BG = '#E4E4E4'                                # light grey under the whisking / licking textures


def _runs(on):
    on = np.asarray(on).astype(int)
    cuts = np.r_[0, np.flatnonzero(np.diff(on)) + 1, len(on)]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if on[a]]


def hatch_runs(ax, x_edges, on, y0, y1, hatch, ink=HATCH_INK, zorder=3, fc=WL_BG):
    """Hatch every run of `on` (x_edges has len(on) + 1 entries) between y0 and y1; no fill, no outline."""
    for a, b in _runs(on):
        ax.add_patch(Rectangle((x_edges[a], y0), x_edges[b] - x_edges[a], y1 - y0, fc=fc, ec=ink, lw=0,
                               hatch=hatch, zorder=zorder))


WHEEL6_HUE = 120          # olive: one hue for the direction-less states (not teal / orange = CW / CCW, not blue / red = LD1)


def wheel6_colours():
    """The 6 no-split states (U1..U6, by median peak): lightness = vigor rank, one hue, constant chroma."""
    try:
        import pickle
        import wheel_movement_clusters as wmc
        with open(wmc.states_dir('proficient') + 'model_k7_nosplit.pkl', 'rb') as f:
            peaks = pickle.load(f)['desc'].peak_rad_s.values
    except Exception:
        peaks = np.array([0.29, 0.64, 1.8, 2.75, 2.97, 3.97])
    # lighter than the paw ramp (OKLCH L 0.90 -> 0.48, not 0.93 -> 0.30): the darkest state stays clearly olive
    L = np.linspace(0.90, 0.48, len(peaks))[np.argsort(np.argsort(peaks))]
    return [ps.oklch_to_hex(l, 0.075, WHEEL6_HUE) for l in L]


def wheel_colours(variant='8'):
    """still + the state colours: the paper's 8-state set (wheel_movement_clusters) or the 6 no-split states."""
    if variant == '6':
        return [WHEEL_STILL] + list(wheel6_colours())
    try:
        import wheel_movement_clusters as wmc
        return [WHEEL_STILL] + wmc.colours(wmc.load_model('proficient', 21), 'small_side_vigor')[1:]
    except Exception:
        return [WHEEL_STILL, '#F4E9E2', '#2BB5B5', '#0E5E5E', '#9BA8A8', '#5D6B6B', '#D9874E', '#8E3F12']


MOCK6 = {1: 1, 6: 3, 5: 5, 3: 6}   # the mock movements in the 6-state set: twitch U1, push U3, back-and-forth U5, fast turn U6


def mock_wheel(seed=3, variant='8'):
    """Per-frame wheel state and a velocity trace of movements between stillness (state 0)."""
    rng = np.random.default_rng(seed + 7)
    T = cs.T
    v = 0.03 * rng.normal(size=T)
    state = np.zeros(T, int)
    # (start, length, state, sign, amplitude): a small twitch, a right push, a balanced bout, a fast left turn
    for a, n, s, sign, amp in [(8, 10, 1, 1, 0.25), (30, 22, 6, 1, 1.0), (62, 34, 5, 1, 0.9), (108, 24, 3, -1, 1.4)]:
        x = np.linspace(0, 1, n)
        if s == 5:                                     # balanced: back and forth
            v[a:a + n] += amp * np.sin(2 * np.pi * 3 * x) * np.sin(np.pi * x)
        else:                                          # one-sided: a bell-shaped push
            v[a:a + n] += sign * amp * np.sin(np.pi * x) ** 2
        state[a:a + n] = MOCK6[s] if variant == '6' else s
    return v, state


def _yr(row, frac):
    return cs._y(row, frac)


def _y(row, frac):
    c, h = ROW[row]
    return c - h + 2 * h * frac


def draw(fig, spec, seed=3, variant='8'):
    """cartoon_segmentation.draw with the paw row moved up and a wheel row under it."""
    grey, rows = (cs.WHISK_C, cs.LICK_C), dict(cs.ROW)
    cs.WHISK_C = cs.LICK_C = 'white'                   # whisking / licking state strips: textures, drawn below
    cs.ROW.clear(); cs.ROW.update(CS_ROW)              # in place: cs's helpers read this dict
    try:
        ax_list = cs.draw(fig, spec, seed=seed)
    finally:
        cs.WHISK_C, cs.LICK_C = grey
        cs.ROW.clear(); cs.ROW.update(rows)
    ax = ax_list[0]
    m = cs.mock(seed)
    COL = cs.COL
    # cs.draw's syllable column (everything right of the state strips) is redrawn below with the wheel in it
    x_cut = COL['state'][1] + 0.1
    for art in list(ax.images) + list(ax.patches) + list(ax.lines):
        try:
            x0 = art.get_extent()[0] if hasattr(art, 'get_extent') else (
                art.get_x() if hasattr(art, 'get_x') else np.min(art.get_xdata()))
        except Exception:
            continue
        if x0 >= x_cut:
            art.remove()
    for t in list(ax.texts):
        if t.get_position()[0] >= x_cut:
            t.remove()
    # whisking / licking states as textures
    xe = np.linspace(*COL['state'], cs.T + 1)
    hatch_runs(ax, xe, m['whisk'], _yr('whisk', 0.15), _yr('whisk', 0.85), WHISK_HATCH * 2)
    hatch_runs(ax, xe, m['lick'], _yr('lick', 0.15), _yr('lick', 0.85), LICK_HATCH * 2)
    ax.set_ylim(-0.5, YTOP)
    fs = plt.rcParams['font.size'] * 0.72
    small = fs * 0.88
    # no column headers: cs.draw's bold titles (at y = 31.5) are removed
    for t in [t for t in ax.texts if t.get_position()[1] == 31.5]:
        t.remove()
    ax.text(0, ROW['wheel'][0], 'Wheel', ha='left', va='center', fontsize=fs, color='0.15')

    v, state = mock_wheel(seed, variant)
    wcol = wheel_colours(variant)
    x = np.linspace(*COL['raw'], len(v))
    lim = np.abs(v).max()
    # raw: velocity, movements shaded by their state
    edges = np.r_[0, np.flatnonzero(np.diff(state)) + 1, len(state)]
    for a, b in zip(edges[:-1], edges[1:]):
        if state[a]:
            ax.fill_between([x[a], x[min(b, len(x) - 1)]], _y('wheel', 0.0), _y('wheel', 1.0), color=wcol[state[a]],
                            alpha=0.55, lw=0, zorder=1)
    ax.plot(x, _y('wheel', 0.5 + 0.48 * v / lim), color='0.1', lw=0.6, zorder=2)
    ax.text(COL['raw'][0], _y('wheel', 1.0) + 0.3, 'velocity, movements', ha='left', va='bottom',
            fontsize=small, color='0.35')
    # features: each movement stretched to the same length, overlaid
    xs = np.linspace(*COL['wav'], 60)
    for a, b in zip(edges[:-1], edges[1:]):
        if state[a]:
            seg = v[a:b]
            y = np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, len(seg)), seg)
            ax.plot(xs, _y('wheel', 0.5 + 0.45 * y / lim), color=wcol[state[a]] if (state[a] != 1 or variant == '6') else '0.6',
                    lw=0.8, zorder=3)
    ax.plot(COL['wav'], [_y('wheel', 0.5)] * 2, color='0.7', lw=0.3, zorder=2)
    ax.text(np.mean(COL['wav']), _y('wheel', 0.0) - 0.3, 'time-normalised', ha='center', va='top',
            fontsize=small, color='0.35')
    # clustering: split by direction (CW / balanced / CCW), then a mixture per direction (blobs in teal / grey / orange)
    rng = np.random.default_rng(seed + 11)
    (x0, x1) = COL['km']
    if variant == '6':                             # one mixture, blobs ordered by vigor, no direction groups
        blobs = list(zip(range(1, 7), [(0.15, 0.3), (0.3, 0.7), (0.47, 0.3), (0.62, 0.72), (0.78, 0.3), (0.9, 0.75)]))
        label = 'one GMM\n(no split)'
    else:
        blobs = list(zip(range(1, 8), [(0.5, 0.15), (0.12, 0.55), (0.25, 0.85), (0.5, 0.55), (0.5, 0.88),
                                       (0.88, 0.55), (0.75, 0.85)]))
        label = 'CW / both / CCW,\nGMM in each'
    for s, (fx, fy) in blobs:
        ax.scatter(x0 + fx * (x1 - x0) + rng.normal(0, 0.5, 10), _y('wheel', fy) + rng.normal(0, 0.35, 10), s=1.6,
                   color=wcol[s], edgecolors='0.35', linewidths=0.1, zorder=3)
    ax.text(np.mean(COL['km']), _y('wheel', 0.0) - 0.3, label, ha='center', va='top',
            fontsize=small, color='0.35')
    # states
    rows = dict(cs.ROW)
    cs.ROW.clear(); cs.ROW.update(ROW)                 # cs helpers read their own ROW: point them at this one
    try:
        cs._strip(ax, 'wheel', 'state', state, ListedColormap(wcol), -0.5, len(wcol) - 0.5, 0.25, 0.75)
        for x_a, x_b in [(COL['raw'][1] + 0.6, COL['wav'][0] - 0.6), (COL['wav'][1] + 0.6, COL['km'][0] - 0.3),
                         (COL['km'][1] + 0.6, COL['state'][0] - 0.6)]:
            cs._arrow(ax, x_a, x_b, ROW['wheel'][0])
    finally:
        cs.ROW.clear(); cs.ROW.update(rows)
    ax.text(np.mean(COL['state']), _y('wheel', 0.25) - 0.3, f'{len(wcol) - 1} wheel states' if variant == '6' else '8 wheel states',
            ha='center', va='top', fontsize=small,
            color='0.35')
    _syllables(ax, m, state, wcol, small)
    return [ax]


def _syllables(ax, m, wheel, wcol, small, h=3.4, n_px=36, stripe=6):
    """The four state strips merge into one: paw colour, wheel stripes, whisking '/' and licking '.' over it."""
    COL = cs.COL
    rows = ['paw', 'wheel', 'whisk', 'lick']
    yc = (ROW['paw'][0] + ROW['lick'][0]) / 2
    xm = COL['state'][1] + 1.3
    for r in rows:
        ax.plot([COL['state'][1] + 0.3, xm], [ROW[r][0]] * 2, color='0.35', lw=0.5)
    ax.plot([xm, xm], [ROW['lick'][0], ROW['paw'][0]], color='0.35', lw=0.5)
    cs._arrow(ax, xm, COL['syl'][0] - 0.4, yc)
    # paw base with the wheel as horizontal stripes (its own colour) wherever the wheel moves
    paw_rgb = np.array([to_rgb(c) for c in ps.PAW_STATE_COLORS])[m['paw']]
    whl_rgb = np.array([to_rgb(c) for c in wcol])[wheel]
    on = (np.arange(n_px) // (stripe // 2)) % 2 == 1
    img = np.where((on[:, None] & (wheel > 0)[None, :])[..., None], whl_rgb[None], paw_rgb[None])
    ax.imshow(img, aspect='auto', interpolation='nearest', extent=[*COL['syl'], yc - h, yc + h], zorder=2)
    xe = np.linspace(*COL['syl'], len(wheel) + 1)
    hatch_runs(ax, xe, m['whisk'], yc - h, yc + h, WHISK_HATCH * 2, fc='none')   # over the paw / wheel colours
    hatch_runs(ax, xe, m['lick'], yc - h, yc + h, LICK_HATCH * 2, fc='none')
    ax.add_patch(Rectangle((COL['syl'][0], yc - h), np.diff(COL['syl'])[0], 2 * h, fill=False, ec='0.45', lw=0.3,
                           zorder=4))
    ax.text(np.mean(COL['syl']), yc + h + 0.4, 'paw\n× wheel (═)\n× whisk (/)\n× lick (·)', ha='center', va='bottom',
            fontsize=small, color='0.35', linespacing=1.0)
    ax.text(np.mean(COL['syl']), yc - h - 0.4, f'8 × {len(wcol)} × 2 × 2\nper frame', ha='center', va='top', fontsize=small,
            color='0.35', linespacing=1.0)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    ps.use('paper')
    ps.use_paw_states('uniform')
    plt.rcParams['hatch.linewidth'] = 0.4
    fig = plt.figure(figsize=(3.6, 1.65))
    draw(fig, fig.add_gridspec(1, 1, left=0.01, right=0.99, top=0.98, bottom=0.02)[0])
    OUT = HERE / 'figures'
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(OUT / f'cartoon_segmentation_wheel.{ext}', dpi=600)
    print('wrote', OUT / 'cartoon_segmentation_wheel.svg')
