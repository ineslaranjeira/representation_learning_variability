"""
ONE PLACE FOR EVERY FIGURE DECISION IN THE PAPER
================================================
What goes in here: anything that should look the same in two different figures --
what colour a timepoint is, which colormap means "LD1", how big a one-column panel
is, where figures get written. What does NOT go in here: anything specific to a
single analysis.

USE IT (paste at the top of a notebook, works from any subfolder):

    import sys, pathlib
    _p = pathlib.Path.cwd().resolve()
    while not (_p / 'paper_style.py').exists() and _p != _p.parent:
        _p = _p.parent
    sys.path.insert(0, str(_p))
    import paper_style as ps
    ps.use('poster')      # or ps.use('paper'); 'poster' is bigger type, same look

Then refer to things by MEANING, never by literal value:

    ax.plot(x, y, color=ps.TIMEPOINT['Early'])          # not '#ECA307'
    ax.imshow(R, cmap=ps.CORR_CMAP, norm=ps.corr_norm())
    ps.epoch_lines(ax, n_bins=40)
    ps.savefig(fig, 'fig3_syllable_correlations')            # PNG, dated
    ps.savefig(fig, 'fig3_syllable_correlations', svg=True)   # PNG + vector master

Figures land in figures/ as <name>[_poster]_<DD-MM-YYYY>.<ext>.

NOTHING HERE RESCALES OR TRANSFORMS DATA. The module holds colours, sizes and
rcParams; every plotting call still receives your numbers untouched. The one
optional exception is opt-in and off by default -- see set_ld1_scale().
"""
from datetime import date as _date
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / 'figures'
DATE_FMT = '%d-%m-%Y'      # same convention as the repo's data files, e.g. 08-09-2026

# Two output modes over one set of decisions. paper_base.mplstyle holds everything
# that must NOT differ between them (colours come from this module, spine treatment,
# vector-text export); the mode file on top changes only type size, stroke weight and
# panel size. So a poster figure becomes the paper figure by switching modes.
BASE_STYLE = ROOT / 'paper_base.mplstyle'
MODE_STYLE = {'paper': ROOT / 'paper.mplstyle', 'poster': ROOT / 'poster.mplstyle'}
MODE = 'paper'

# =============================================================================
# COLOURS -- by meaning
# =============================================================================

# =============================================================================
# LEARNING TIMEPOINTS -- ONE RAMP, DARKENING WITH TRAINING
# =============================================================================
# The timepoints are ORDERED (Early -> Late -> Pre-rec -> Proficient), so they get an
# ordered colour: one ramp through magma, coral -> crimson -> magenta -> deep purple,
# darkening the nearer the mouse is to proficient. That replaces the four unrelated
# hues (orange / purple / green / blue) used up to 08-09-2026, which the reader had to
# learn from a key -- with a ramp, the darkest line IS the most trained one. It is the
# same encoding ps.shade() and syllable_correlations' tp_weight() already use for
# closeness to proficient.
#
# Why magma rather than a single-hue ramp: its LIGHTNESS falls monotonically along the
# ramp, so the order survives greyscale printing AND every stop is dark enough to see
# on white -- the palest end of a one-hue ramp (a pale lavender, say) disappears in
# thin lines and small alpha-blended markers. Keeping Early warm also carries over
# from the previous palette, where Early was orange.
#
# The magenta-purple family is free: LD1 is blue-red (LD1_CMAP), correlations
# brown-teal (CORR_CMAP), syllables Set3 pastels with a grey whisk and a near-black
# lick -- so nothing else in the paper reads as a timepoint.
#
# Used by syllable_correlations, lda_sweep_timepoints, lda_trajectories,
# lda_all_timepoints, lda_pred, relational_structure.
TIMEPOINT_ORDER = ['Early', 'Late', 'Pre-rec', 'Proficient']
TIMEPOINT_LABEL = {
    'Early': 'Early learning', 'Late': 'Late learning',
    'Pre-rec': 'Pre-recording', 'Proficient': 'Proficient',
}

# The stretch of the source colormap the ramp spans, as (Early end, Proficient end).
# magma runs dark -> light, so the pair DESCENDS. To make Early stronger, move the
# FIRST number toward 0 (0.62 gives a deeper coral); to keep Proficient off black,
# move the second up (0.28). The four stops and the continuous ramp move together, so
# that one edit is the whole adjustment.
TIMEPOINT_RAMP = 'magma'
TIMEPOINT_SPAN = (0.72, 0.20)
TIMEPOINT_CMAP = LinearSegmentedColormap.from_list(
    'timepoint', plt.get_cmap(TIMEPOINT_RAMP)(np.linspace(*TIMEPOINT_SPAN, 256)))

# Where each timepoint sits on the ramp: 0 = the start of training, 1 = proficient.
TIMEPOINT_POS = {n: i / (len(TIMEPOINT_ORDER) - 1)
                 for i, n in enumerate(TIMEPOINT_ORDER)}
TIMEPOINT = {n: mpl.colors.to_hex(TIMEPOINT_CMAP(p)) for n, p in TIMEPOINT_POS.items()}


def timepoint_color(t):
    """Colour for a timepoint.

    `t` is either one of the four labels, or a POSITION on the training ramp in
    [0, 1] (0 = start of training, 1 = proficient) -- which is what to use for
    something the four labels do not cover, e.g. one line per session with the
    sessions spread across training.
    """
    if isinstance(t, str):
        return TIMEPOINT[t]
    return TIMEPOINT_CMAP(float(np.clip(t, 0.0, 1.0)))


def timepoint_colors(values, vmin=None, vmax=None):
    """RGBA per value on the training ramp -- for scatter points or line collections
    coloured by a continuous measure of training (session number, days trained, ...).
    Scale comes from the values unless vmin/vmax are given, which is how two figures
    are made to share one scale."""
    v = np.asarray(values, float)
    lo = np.nanmin(v) if vmin is None else float(vmin)
    hi = np.nanmax(v) if vmax is None else float(vmax)
    span = (hi - lo) or 1.0
    return TIMEPOINT_CMAP(np.clip((v - lo) / span, 0.0, 1.0))


def timepoint_colorbar(ax, label='Training stage', named_ticks=True, **kw):
    """A colorbar for the training ramp. With named_ticks it is ticked at the four
    timepoints instead of 0-1, so the bar doubles as the legend."""
    sm = plt.cm.ScalarMappable(cmap=TIMEPOINT_CMAP,
                               norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, label=label, **kw)
    if named_ticks:
        cb.set_ticks([TIMEPOINT_POS[n] for n in TIMEPOINT_ORDER])
        cb.set_ticklabels([TIMEPOINT_LABEL[n] for n in TIMEPOINT_ORDER])
    return cb

# Behavioural syllables: 8 paw states (Set3, as in lda_trajectories) + whisk + lick.
N_PAW_STATES = 8
SYLLABLE_NAMES = [f'Paw {i}' for i in range(N_PAW_STATES)] + ['Whisk', 'Lick']
SYLLABLE_COLORS = ([mpl.colors.to_hex(c) for c in plt.get_cmap('Set3').colors[:N_PAW_STATES]]
                   + ['#b8b8b8', '#484949'])
SYLLABLE = dict(zip(SYLLABLE_NAMES, SYLLABLE_COLORS))
MODALITY = {'Paw': list(range(N_PAW_STATES)), 'Whisk': [8], 'Lick': [9]}

# Trial structure. The epochs are equal blocks of the binned trial, so the edges
# follow from however many bins the file has.
EPOCH_NAMES = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
# The same four epochs, abbreviated. Four full names do not fit above a panel much
# narrower than a full-width one -- at poster type they run into each other -- so a
# small-multiple passes these to epoch_lines(names=...) instead.
EPOCH_SHORT = ['Pre-Q', 'Quiesc.', 'Choice', 'ITI']

# =============================================================================
# GLM-HMM ENGAGEMENT STATES, AND THE TASK'S BIAS BLOCKS
# =============================================================================
# Used by GLM-HMM/load_states, engagement_trial_modes, k_state_model_comparison.
#
# THE STATES. The fits label them by index (state1 / state2) and the index is only
# meaningful within one animal's own fit, so the paper refers to them by what they
# ARE -- engaged / disengaged -- and load_states checks per mouse that state1 is the
# more accurate one.
#
# THE TWO STATES SHARE ONE INK AND ARE TOLD APART BY FORM (approved 08-09-2026):
# engaged is solid with filled markers, disengaged is dashed with open markers. The
# poster already spends five palettes -- LD1 (blue-white-red), the learning-stage ramp
# (coral to deep purple), correlations (brown-grey-teal), syllables (Set3 plus a grey
# whisk and a near-black lick) and the bias blocks -- and every remaining hue sits in
# one of those families, so a sixth would be read as one of them. A binary distinction
# does not need a hue: it survives greyscale, photocopying and every colour-vision
# type, and it leaves the blocks free to keep colour in the one figure where blocks and
# states appear together.
#
# CONSEQUENCE FOR CALL SITES: `color=ps.state_color(s)` alone no longer distinguishes
# anything. Ask for the whole style instead -- ps.state_kw(s, 'line' | 'marker' |
# 'scatter' | 'hist' | 'patch') returns the matplotlib kwargs, so one edit here
# restyles every panel. Give the pair real hues by editing STATE and STATE_DASH.
STATE_INK = '#2B2B2B'
STATE = {'engaged': STATE_INK, 'disengaged': STATE_INK}
STATE_DASH = {'engaged': '-', 'disengaged': (0, (5, 2))}
STATE_FILLED = {'engaged': True, 'disengaged': False}
STATE_LABEL = {'engaged': 'Engaged', 'disengaged': 'Disengaged'}
STATE_ORDER = ['engaged', 'disengaged']
# how the GLM-HMM's own labels map onto those names
STATE_FROM_INDEX = {'state1': 'engaged', 'state2': 'disengaged',
                    1: 'engaged', 2: 'disengaged'}


def state_name(s):
    """'state1' / 'state2' (or 1 / 2) -> 'engaged' / 'disengaged'. Already-named
    states pass through, so a caller can hand this either convention."""
    return STATE_FROM_INDEX.get(s, s)


def state_color(s):
    """The ink. On its own it does NOT tell the states apart -- see state_kw."""
    return STATE[state_name(s)]


def state_kw(s, kind='line', label=True, **over):
    """Matplotlib kwargs that distinguish the two states by FORM, not hue.

    kind='line'     plot()      solid vs dashed
    kind='marker'   errorbar()  solid vs dashed, filled vs open marker
    kind='scatter'  scatter()   filled vs open marker
    kind='hist'     hist()      filled step vs dashed step outline
    kind='patch'    bar/axvspan solid fill vs open, hatched fill

    label=True adds the state's name, so a legend needs no second argument; pass a
    string to label it something else, or False for no legend entry (the second call
    for the same state in one panel, say). Anything passed as **over wins, for the one
    panel that needs an exception.
    """
    nm = state_name(s)
    ink, dash, filled = STATE[nm], STATE_DASH[nm], STATE_FILLED[nm]
    lw = plt.rcParams['lines.linewidth']
    if kind == 'line':
        kw = dict(color=ink, ls=dash)
    elif kind == 'marker':
        kw = dict(color=ink, ls=dash, marker='o', markeredgecolor=ink,
                  markerfacecolor=ink if filled else 'white',
                  markeredgewidth=lw * 0.5)
    elif kind == 'scatter':
        kw = dict(facecolors=ink if filled else 'none', edgecolors=ink,
                  linewidths=lw * 0.5)
    elif kind == 'hist':
        kw = (dict(color=ink, histtype='stepfilled', alpha=0.30, lw=lw, edgecolor=ink)
              if filled else
              dict(color=ink, histtype='step', lw=lw, ls=dash))
    elif kind == 'patch':
        kw = (dict(facecolor=ink, alpha=0.30, edgecolor=ink, lw=lw * 0.5) if filled else
              dict(facecolor='white', edgecolor=ink, lw=lw * 0.5, hatch='///'))
    else:
        raise ValueError("kind must be one of 'line', 'marker', 'scatter', 'hist', 'patch'")
    if label is True:
        kw['label'] = STATE_LABEL[nm]
    elif isinstance(label, str):
        kw['label'] = label
    kw.update(over)
    return kw


def state_label(s, with_index=False):
    """'Engaged', or 'Engaged (state1)' with with_index -- worth showing wherever the
    figure is about whether the index really means what the name says."""
    nm = state_name(s)
    return f'{STATE_LABEL[nm]} ({s})' if with_index else STATE_LABEL[nm]


# THE BIAS BLOCKS. p(left) = 0.2 / 0.5 / 0.8 is a SIGNED condition -- which side the
# block favours -- with a neutral middle, so it gets a diverging green <-> pink pair
# around grey rather than three arbitrary hues (the tab:blue/orange/green it replaces
# read as unordered, and its orange collided with the disengaged state).
BLOCK = {0.2: '#1B7837', 0.5: '#808080', 0.8: '#C51B7D'}
BLOCK_LABEL = {0.2: 'p(left) = 0.2', 0.5: 'unbiased', 0.8: 'p(left) = 0.8'}

# For a single distribution, a nuisance covariate, a pooled cloud -- anything whose
# colour carries NO meaning. One grey, so "no colour meaning" looks the same
# everywhere instead of being a different tab: colour in each figure.
NEUTRAL = '#7A7A7A'

# =============================================================================
# COLORMAPS -- also by meaning
# =============================================================================

# THE LD1 GRADIENT. Blue = low LD1, red = high, white at zero. 'coolwarm' (approved
# 08-09-2026) rather than the 'bwr' in lda_trajectories: same blue-red reading, less
# saturated poles, so detail at the extremes is not clipped.
LD1_CMAP = plt.get_cmap('coolwarm')

# OPT-IN, OFF BY DEFAULT. With LD1_VMAX = None the colour scale is taken from each
# figure's own data, which is matplotlib's normal behaviour. Setting it fixes one
# scale across figures so a given colour means the same LD1 value everywhere; that
# is a presentation choice with a real trade-off (a figure whose LD1 range is small
# will look washed out), so it is yours to make, not the module's.
LD1_VMAX = None


def set_ld1_scale(vmax):
    """Optional: fix the LD1 colour scale so +/- vmax maps to the extremes in every
    figure from then on. Not called anywhere by default. set_ld1_scale(None) undoes
    it and returns to per-figure autoscaling."""
    global LD1_VMAX
    LD1_VMAX = None if vmax is None else float(abs(vmax))
    return LD1_VMAX


def ld1_norm(values=None):
    """Diverging normaliser centred at 0 for LD1: symmetric, so zero sits at white.
    Scale comes from `values` unless a fixed one was set with set_ld1_scale()."""
    if LD1_VMAX is not None:
        v = LD1_VMAX
    else:
        if values is None:
            raise ValueError("pass the values to scale to, or fix a scale with "
                             "ps.set_ld1_scale(vmax)")
        v = float(np.nanmax(np.abs(np.asarray(values, float))))
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)


def ld1_colors(values):
    """RGBA per value on the LD1 gradient -- for scatter points or line collections."""
    return LD1_CMAP(ld1_norm(values)(np.asarray(values, float)))


def ld1_colorbar(ax, label='LD1', values=None, **kw):
    sm = plt.cm.ScalarMappable(cmap=LD1_CMAP, norm=ld1_norm(values))
    sm.set_array([])
    return plt.colorbar(sm, ax=ax, label=label, **kw)


def _mix(a, b, w):
    A, B = np.array(to_rgb(a)), np.array(to_rgb(b))
    return tuple(A * (1 - w) + B * w)


def grey_centered(neg='#8C5A1A', pos='#11736B', grey='#CCCCCC', name='grey_centered'):
    """Diverging map with GREY at zero. Grey rather than white because a near-zero
    cell should read as "nothing here" instead of dissolving into the page, and
    because it leaves white free to mean "not estimated" (cmap.set_bad)."""
    cm = LinearSegmentedColormap.from_list(
        name, [neg, _mix(neg, grey, 0.55), grey, _mix(pos, grey, 0.55), pos])
    cm.set_bad('white')
    return cm


# Correlation maps (r across mice, RDMs, ...). Brown <-> teal deliberately avoids
# blue-red, which is spoken for by LD1 -- a red/blue correlation map reads as an LD1
# map at a glance.
CORR_CMAP = grey_centered()
CORR_VMAX = 1.0


def corr_norm(vmax=None):
    v = CORR_VMAX if vmax is None else abs(vmax)
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)


# THE SAME MAP under the name to use when the quantity is not a correlation: any
# signed measure with a meaningful zero -- a difference of probabilities, a
# high-minus-low contrast, a residual. One diverging map for the whole paper means a
# grey cell always reads as "no difference", and it keeps the blue-red for LD1.
# corr_norm's vmax argument does the scaling in exactly the same way.
DIVERGING_CMAP = CORR_CMAP
diverging_norm = corr_norm


def truncate_cmap(cmap, lo=0.0, hi=1.0, n=256, name=None):
    """A colormap restricted to the [lo, hi] slice of another one.

    Sequential maps that run all the way to their ends are awkward on a white page:
    the dark end is near-black, so cells there swallow any annotation drawn on top
    and read as "missing", and the light end fades into the background, so the top of
    the scale disappears. Trimming keeps the perceptually uniform middle and leaves
    black free for text and white free for NaN.
    """
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return LinearSegmentedColormap.from_list(
        name or f'{cmap.name}_{lo:g}_{hi:g}', cmap(np.linspace(lo, hi, n)))


# Magnitudes with no meaningful zero (occupancy maps, RDMs, densities). Magma trimmed
# at both ends: 0.15 drops the near-black, 0.92 drops the pale yellow that vanishes on
# white. Swap the slice or the base map here and every panel follows.
SEQUENTIAL_CMAP = truncate_cmap('magma', 0.15, 0.92)
SEQUENTIAL_CMAP.set_bad('white')

# =============================================================================
# SIZES -- in inches. Paper sizes are journal column widths; poster sizes are the
# same panels enlarged, so relative proportions survive a mode switch.
# =============================================================================
SIZES = {
    'paper': {'single': (3.40, 2.40),    # 1 column,   ~86 mm
              'wide':   (4.72, 2.80),    # 1.5 column, ~120 mm
              'double': (7.09, 3.20),    # 2 columns,  ~180 mm
              'square': (3.40, 3.40)},
    'poster': {'single': (7.0, 5.0),
               'wide':   (9.5, 5.6),
               'double': (14.0, 6.4),
               'square': (7.0, 7.0)},
}
# The pristine sizes, never mutated. use(scale=...) multiplies THESE, and the factor
# is remembered per mode -- so use('poster', scale=1.5) twice gives 1.5x, not 2.25x,
# which is what mutating SIZES in place used to do.
BASE_SIZES = {m: dict(d) for m, d in SIZES.items()}
SCALE = {m: 1.0 for m in SIZES}
# rcParams that use() was asked to override on top of the style files (the dpi
# arguments). plt.style.use() resets everything it does not set, so reapply() would
# otherwise silently undo a save_dpi you asked for; they are remembered here and
# re-applied by both.
OVERRIDES = {}
FIG = SIZES[MODE]


def figure(kind='single', scale=1.0, **kw):
    """A figure at the current mode's size for that panel kind."""
    w, h = SIZES[MODE][kind]
    return plt.subplots(figsize=(w * scale, h * scale), **kw)


# =============================================================================
# HELPERS that keep repeated panel furniture identical
# =============================================================================

def use(mode='paper', spines='lb', dpi=None, save_dpi=None, scale=None):
    """Apply the shared rcParams. Call once per notebook, after the imports.

    mode    'paper'  -> journal column widths, 7 pt type, thin strokes
            'poster' -> the same figures with 16 pt type, thick strokes, big panels
    spines  'lb'   left and bottom only (no box) -- the default
            'none' no spines at all; the ticks and labels carry the scale
            'box'  all four, for the rare panel that needs a frame (an image, say)
    dpi       on-screen preview resolution only; does not affect saved files
    save_dpi  resolution of saved RASTER files (PNG). Vector output (SVG, PDF)
              ignores it -- it has no pixels to define.
    scale     multiply every panel size in this mode by a factor, e.g. 1.5 for
              panels half again as large. It is applied to the mode's ORIGINAL sizes
              and remembered for that mode, so calling use() again without `scale`
              keeps the factor and calling it again WITH the same one is a no-op.
              TYPE SIZE IS UNCHANGED -- points are
              absolute, so a 16 pt label is 16 pt whether the panel is 7 in or 14 in
              wide. That is deliberate: every figure on the poster then carries the
              same size text regardless of how large the panel is. The corollary is
              that figures must be PLACED AT 100% in the poster layout; resizing an
              exported file there scales its text along with it and breaks the match.

    Switching modes changes nothing but size, so a poster figure becomes the paper
    figure with `ps.use('paper')` and a re-run -- no per-figure edits.
    """
    global MODE, FIG
    if mode not in MODE_STYLE:
        raise ValueError(f"mode must be one of {list(MODE_STYLE)}")
    if mode != MODE:
        # a dpi override belongs to the mode it was asked for -- switching modes goes
        # back to that mode's own dpi (400 for paper, 300 for poster) unless this same
        # call passes a new one
        OVERRIDES.clear()
    MODE = mode
    if scale is not None:
        SCALE[mode] = float(scale)
    SIZES[mode] = {k: (w * SCALE[mode], h * SCALE[mode])
                   for k, (w, h) in BASE_SIZES[mode].items()}
    FIG = SIZES[mode]
    plt.style.use([str(BASE_STYLE), str(MODE_STYLE[mode])])
    if dpi is not None:
        OVERRIDES['figure.dpi'] = dpi
    if save_dpi is not None:
        OVERRIDES['savefig.dpi'] = save_dpi
    plt.rcParams.update(OVERRIDES)
    on = {'lb': (True, True, False, False),
          'none': (False, False, False, False),
          'box': (True, True, True, True)}[spines]
    for side, flag in zip(('left', 'bottom', 'top', 'right'), on):
        plt.rcParams[f'axes.spines.{side}'] = flag
    print(f"paper_style: {mode} mode, spines={spines}, "
          f"font {plt.rcParams['font.size']:.0f} pt, "
          f"panels {SIZES[mode]['single'][0]:.1f}x{SIZES[mode]['single'][1]:.1f} in, "
          f"save {plt.rcParams['savefig.dpi']:.0f} dpi")
    return MODE


def epoch_lines(ax, n_bins, label=False, names=None, color='k', **kw):
    """The dashed trial-epoch boundaries, drawn the same way in every panel.

    label=True also names the four epochs above the axes; `names` chooses the wording
    (EPOCH_NAMES by default, EPOCH_SHORT for a panel too narrow to hold them)."""
    edges = [n_bins // 4, n_bins // 2, 3 * n_bins // 4]
    lw = 0.5 if MODE == 'paper' else 1.2
    for e in edges:
        ax.axvline(e, color=color, ls='--', lw=lw, alpha=0.45, **kw)
    if label:
        bounds = [0] + edges + [n_bins]
        for k, nm in enumerate(EPOCH_NAMES if names is None else names):
            ax.text((bounds[k] + bounds[k + 1]) / 2, 1.01, nm,
                    fontsize=plt.rcParams['font.size'] * 0.8,
                    ha='center', va='bottom', color='0.35',
                    transform=ax.get_xaxis_transform())
    return edges


def zero_line(ax, **kw):
    ax.axhline(0, color='k', lw=0.5 if MODE == 'paper' else 1.2, alpha=0.6,
               zorder=0, **kw)


def stars(p):
    """p-value -> the paper's asterisk convention."""
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


def shade(color, w, l_floor=0.42, s_boost=0.55):
    """Same hue, deeper the larger w. Used to encode an ordered variable (distance in
    time from the proficient state, say) without spending a second hue on it."""
    import colorsys
    h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
    l = l * (1 - float(w)) + (l * l_floor) * float(w)
    return colorsys.hls_to_rgb(h, l, min(1.0, s * (1 + s_boost * float(w))))


def regline(ax, x, y, significant, color='k', lw=None, ls='-', n_points=100, **kw):
    """Least-squares line, drawn ONLY when the correlation is significant.

    THE CONVENTION: a line through a null result is read as a trend whatever the
    caption says, so a non-significant correlation gets its points and nothing else.
    `significant` is passed in rather than computed here, because whether a p-value
    counts as significant depends on the correction the analysis applied (raw, FDR,
    Bonferroni) -- that decision belongs to the analysis, not to the plotting layer.

    Returns the Line2D, or None when nothing was drawn.
    """
    if not significant:
        return None
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    m, b = np.polyfit(x[ok], y[ok], 1)
    xs = np.linspace(x[ok].min(), x[ok].max(), n_points)
    lw = plt.rcParams['lines.linewidth'] if lw is None else lw
    return ax.plot(xs, m * xs + b, color=color, lw=lw, ls=ls, zorder=3, **kw)[0]


def corr_title(name, r, p, significant=None, n=None):
    """One phrasing for every correlation panel in the paper. `significant` should be
    the analysis's own verdict (post-correction if it corrected); without it the
    uncorrected p is used."""
    sig = (p < 0.05) if significant is None else bool(significant)
    mark = stars(p) if sig else 'n.s.'
    tail = f", n = {n}" if n is not None else ''
    return f"{name}\nr = {r:.2f}, p = {p:.1e} {mark}{tail}"


def reapply():
    """Re-assert the current mode's rcParams, quietly.

    Worth calling at the top of a figure cell. rcParams are global and sticky: some
    helpers in this repo (Models/Sub-trial/3_postprocess_results/plotting_functions.py,
    for instance) call plt.rc('font', size=12) inside their bodies, so ONE call to
    those silently reverts type size for every figure drawn afterwards in that kernel.
    """
    spines = {s: plt.rcParams[f'axes.spines.{s}'] for s in
              ('left', 'bottom', 'top', 'right')}
    plt.style.use([str(BASE_STYLE), str(MODE_STYLE[MODE])])
    plt.rcParams.update(OVERRIDES)
    for k, v in spines.items():
        plt.rcParams[f'axes.spines.{k}'] = v
    return plt.rcParams['font.size']


def corr_label(r, p, significant=None, n=None):
    """The stats block for a correlation panel, as text to place INSIDE the axes."""
    sig = (p < 0.05) if significant is None else bool(significant)
    mark = stars(p) if sig else 'n.s.'
    out = f"r = {r:.2f}\np = {p:.1e} {mark}"
    return out + (f"\nn = {n}" if n is not None else '')


def annotate_corr(ax, r, p, significant=None, n=None, loc='upper left', **kw):
    """Place corr_label() in a corner of the axes, same corner and same size in every
    panel. Inside the axes rather than in the title, so the title carries only what
    the panel IS and the numbers sit with the data."""
    x, ha = (0.04, 'left') if 'left' in loc else (0.96, 'right')
    y, va = (0.97, 'top') if 'upper' in loc else (0.04, 'bottom')
    kw.setdefault('fontsize', plt.rcParams['font.size'] * 0.85)
    kw.setdefault('color', '0.2')
    kw.setdefault('linespacing', 1.25)
    return ax.text(x, y, corr_label(r, p, significant, n), transform=ax.transAxes,
                   ha=ha, va=va, zorder=5, **kw)


def corr_panel(ax, x, y, color=None, method='pearson', log_y=False, xlabel=None,
               ylabel=None, annotate='upper left', s_scale=1.0, colorbar=False,
               cb_label='LD1', label_rho=None):
    """THE PAPER'S CORRELATION PANEL, in one call: one point per unit, a least-squares
    line drawn ONLY where the correlation is significant, and the statistics inside the
    axes rather than in the title.

    It exists because this panel is drawn a few dozen times across load_states,
    k_state_model_comparison, engagement_trial_modes and LDA_behavior, and every copy
    of it was choosing its own marker size, its own fit rule and its own way of
    reporting r -- which is exactly the kind of decision that belongs here.

    color   None colours the points on the LD1 gradient by their x value, which is
            what to use when x IS an LD1 value. Pass ps.NEUTRAL (or any colour) when
            it is not -- a residual, a trial count -- so the gradient never implies an
            LD1 scale that the axis does not carry.
    method  which correlation gates the line and is reported ('pearson' or 'spearman').
            Both are always returned.
    log_y   for a positive, right-skewed y (a ratio): fits linear in log(y), draws the
            curve back on a log axis.
    label_rho  None follows `method` (rho for Spearman, r for Pearson); force it with
            True/False when the caption says otherwise.

    Returns the stats dict, so the cell can still print or collect the numbers.
    """
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    r, p = (r_p, p_p) if method == 'pearson' else (r_s, p_s)

    c = ld1_colors(x) if color is None else color
    ax.scatter(x, y, c=c, s=(plt.rcParams['lines.markersize'] * s_scale) ** 2,
               alpha=0.85, edgecolors='none')
    if log_y:
        if p < 0.05:
            a, b = np.polyfit(x, np.log(y), 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, np.exp(a * xs + b), color='k',
                    lw=plt.rcParams['lines.linewidth'], zorder=3)
        ax.set_yscale('log')
    else:
        regline(ax, x, y, p < 0.05)
    if annotate:
        use_rho = (method == 'spearman') if label_rho is None else bool(label_rho)
        txt = corr_label(r, p, n=len(x))
        if use_rho:
            txt = txt.replace('r =', 'ρ =')
        xa, ha = (0.04, 'left') if 'left' in annotate else (0.96, 'right')
        ya, va = (0.97, 'top') if 'upper' in annotate else (0.04, 'bottom')
        ax.text(xa, ya, txt, transform=ax.transAxes, ha=ha, va=va, zorder=5,
                fontsize=plt.rcParams['font.size'] * 0.85, color='0.2', linespacing=1.25)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if colorbar:
        ld1_colorbar(ax, label=cb_label, values=x, fraction=0.046, pad=0.02)
    return dict(r_pearson=r_p, p_pearson=p_p, r_spearman=r_s, p_spearman=p_s, n=len(x))


def savefig(fig, name, svg=False, formats=None, subdir=None, dated=True,
            tag_mode=True, **kw):
    """Write a figure to figures/ under the paper's naming convention.

        <name>[_<mode>]_<DD-MM-YYYY>.png

    PNG by default because that is what goes into slides and drafts; pass svg=True to also write the vector master (text stays text --
    see paper.mplstyle -- so labels remain editable in Illustrator). `formats` takes
    an explicit tuple if you want something else entirely, e.g. ('pdf',).

    The date is the RUN date, so re-running never silently overwrites yesterday's
    figure: you get a new file and the old one stays for comparison. Pass dated=False
    for a stable filename when a manuscript references one.
    """
    if formats is None:
        formats = ('png', 'svg') if svg else ('png',)
    stem = name
    # poster and paper renders of the same figure must not overwrite each other
    if tag_mode and MODE != 'paper':
        stem = f'{stem}_{MODE}'
    if dated:
        stem = f'{stem}_{_date.today().strftime(DATE_FMT)}'
    out = FIGDIR if subdir is None else FIGDIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in formats:
        p = out / f'{stem}.{f}'
        fig.savefig(p, format=f, **kw)
        paths.append(p)
    print('wrote ' + ', '.join(str(p.relative_to(ROOT)) for p in paths))
    return paths
