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
    ps.saving(True)                                          # writing is OPT-IN
    ps.savefig(fig, 'fig3_syllable_correlations')            # PNG, dated
    ps.savefig(fig, 'fig3_syllable_correlations', svg=True)   # PNG + vector master

Figures land in figures/ as <name>[_poster]_<DD-MM-YYYY>.<ext>, but ONLY after
ps.saving(True) -- by default savefig prints the name it would write and writes nothing.

NOTHING HERE RESCALES OR TRANSFORMS DATA. The module holds colours, sizes and
rcParams; every plotting call still receives your numbers untouched. The one
optional exception is opt-in and off by default -- see set_ld1_scale().
"""
from datetime import date as _date
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import (LinearSegmentedColormap, ListedColormap, TwoSlopeNorm,
                               to_rgb)

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / 'figures'
DATE_FMT = '%d-%m-%Y'      # same convention as the repo's data files, e.g. 08-09-2026

# FIGURE WRITING IS OFF BY DEFAULT -- savefig() reports the name it would write and
# returns []. Notebooks here get re-run constantly while a knob is being tuned, and every
# run used to leave another dated PNG/SVG pair in figures/, so the directory filled with
# near-identical files and the one that belongs in the paper was whichever ran last.
# Call ps.saving(True) for the run that counts. A reload resets this to False, which is
# the safe direction.
SAVE_FIGURES = False

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
# brown-teal (CORR_CMAP), paw syllables blue-orange by laterality with a grey whisk and a
# near-black lick (PAW_STATE_COLORS) -- so nothing else in the paper reads as a timepoint.
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

# Behavioural syllables: 8 paw states + whisk + lick.
N_PAW_STATES = 8
SYLLABLE_NAMES = [f'Paw {i}' for i in range(N_PAW_STATES)] + ['Whisk', 'Lick']
MODALITY = {'Paw': list(range(N_PAW_STATES)), 'Whisk': [8], 'Lick': [9]}

# ---------------------------------------------------------------------------
# PAW STATES COLOURED BY WHAT THEY ARE, not by an arbitrary categorical order
# ---------------------------------------------------------------------------
# HUE says which forepaw leads -- VIOLET LEFT, AMBER RIGHT -- and LIGHTNESS says how
# vigorous the state is. Both come
# from segmentation/paw_bias/state_profiles_19Ago2026.csv, reduced the way
# 4_mice/laterality/laterality_features.state_laterality() reduces it:
#   vigor = mean session-z-scored wavelet power over all 20 paw channels
#   LI    = (left - right) / (|left| + |right|),  so POSITIVE = LEFT paw
# The eight states sit on that plane already -- 3 and 4 mirror each other at moderate
# vigor, 5 and 6 mirror each other higher up, and 0/1/2/7 are a symmetric spine from
# still to very fast -- so this palette shows the structure rather than hiding it.
#
# CHROMA IS THE SIZE OF THE LEFT-RIGHT GAP, |left - right|, NOT the laterality index.
# LI is a RATIO, (L-R)/(|L|+|R|), and its denominator collapses for the quiet states, so it
# manufactures bias out of nothing: state 1's paws differ by 0.045 -- no movement either
# side -- but over a denominator of 0.349 that reads as LI +0.13 and came out visibly
# tinted. State 4 is the same trick inverted, a 0.386 gap over 0.472 giving LI -0.82.
# The two rankings disagree almost everywhere:
#     by |LI|            4 > 3 > 6 > 5 > 1 > 2 > 7 > 0
#     by |left - right|  5 > 6 > 3 > 7 > 4 > 2 > 1 > 0
# The gap is the honest one. It also fixes, for free, what two earlier attempts could not:
# states 5 and 6 are the FAST lateralised pair the LD3 result rests on and they hold the two
# largest real asymmetries (0.956, 0.952), so they finally come out the boldest marks on the
# panel instead of the palest. A compressive transform (sqrt|LI|) was tried and goes the
# WRONG way -- it lifts the small values, doubling state 1's chroma.
#
# WHAT IT COSTS: state 4 falls from C 0.20 to 0.10. Its LI is the most extreme in the set but
# its actual gap is modest, because both its paws barely move; it is also the single strongest
# correlate of LD3 (r = +0.61), so the state doing most of the work on that axis is no longer
# the most saturated. Measured: normal 12.6 / CVD 12.4, against |LI|'s 14.6 / 13.4.
# VIOLET LEFT (310), AMBER RIGHT (90) -- chosen to stay off LD1_CMAP, which runs blue (271)
# to red (22) and so owns both ends of the obvious diverging pair. Blue/orange separated
# better (CVD 12.4 against 9.6) because blue-yellow is the one chromatic axis dichromats
# keep, but it put the paw palette in the same hues as the individuality axis the paper is
# about. 9.6 still clears the 8 the validator asks for.
#
# 310 rather than the CVD-optimal 295: the sweep's best violet is only 24 deg off LD1's blue
# pole, i.e. it still reads blue-purple and does not solve the problem it exists for. 310 is
# 39 deg clear and genuinely purple. Past 320 the CVD floor goes.
#     violet  295 -> CVD 11.4, LD1 gap 24     310 -> CVD 9.6, gap 39     325 -> CVD 7.5, gap 54
# The amber hue is free: CVD is flat across 80-100, so it was picked on appearance.
PAW_LEFT, PAW_RIGHT = '#7B09AE', '#985800'
#                                                     gap    vigor   frames   C
PAW_STATE_COLORS = ['#EDE8D8',   # 0  still, neutral   -0.018  -0.61   35.7%  0.022
                    '#BFB5C7',   # 1  slow, neutral    +0.045  -0.17   27.3%  0.027
                    '#857691',   # 2  moderate, faint  +0.123  +0.70    7.8%  0.044
                    '#A96BD3',   # 3  moderate, LEFT   +0.670  +0.53    7.9%  0.161
                    '#B79C50',   # 4  moderate, right  -0.386  +0.24   11.5%  0.101
                    '#7B09AE',   # 5  fast, LEFT       +0.956  +1.54    3.5%  0.223
                    '#985800',   # 6  fast, RIGHT      -0.952  +1.16    5.7%  0.119
                    '#462600']   # 7  very fast, right -0.507  +3.19    0.7%  0.069

# THE HMM's STATE NUMBERS ARE ARBITRARY, so index order is not vigor order -- state 2 is more
# vigorous than 3 and 4 and is therefore darker, which reads as a mistake in a legend laid out
# 0..7. This is the order to DISPLAY them in: stacks and legends built on it run monotonically
# from pale-still to dark-vigorous, which is what lightness already encodes.
#     lightness along PAW_VIGOR_ORDER   0.93 0.79 0.70 0.64 0.59 0.52 0.47 0.30   monotone
#     lightness in index order          0.93 0.79 0.59 0.64 0.70 0.47 0.52 0.30   not
# The COLOUR keying is untouched: PAW_STATE_COLORS[k] is still state k.
PAW_VIGOR_ORDER = [0, 1, 4, 3, 2, 6, 5, 7]

PAW_LATERAL = {'left': [3, 5], 'right': [4, 6], 'symmetric': [0, 1, 2, 7]}
PAW = dict(zip([f'Paw {i}' for i in range(N_PAW_STATES)], PAW_STATE_COLORS))

PAW_SIDE = {s: side for side, states in PAW_LATERAL.items() for s in states}


def paw_label(state, with_side=True):
    """'Paw 3  (left)' -- so a legend states the encoding instead of asking the reader to
    remember it. Symmetric states get no tag, because they have no side to name."""
    side = PAW_SIDE.get(state, 'symmetric')
    return f'Paw {state}' + ('' if side == 'symmetric' or not with_side else f'  ({side})')


# ---------------------------------------------------------------------------
# THE SYLLABLE RASTER -- trials x bins, one convention for every figure using it
# ---------------------------------------------------------------------------
# A syllable is (paw state, whisking on/off, licking on/off), packed as
#     code = paw + n_paw*whisk + 2*n_paw*lick          paw is the FAST index
# which is 32 values for 8 paw states. Drawn as ONE image, that needs 32 distinguishable
# colours, and the old figure got them as 8 hues x 4 lightness steps.
#
# WHY THAT NO LONGER WORKS. PAW_STATE_COLORS spends BOTH hue and lightness on the paw
# dimension -- hue is which forepaw leads, lightness is vigor -- so there is no free
# visual channel left inside one image. Any lightness step for whisk/lick now collides
# with vigor, and at one pixel per bin a 4-way lightness step was not resolvable anyway:
# it read as texture rather than as data.
#
# SO THE RASTER SPLITS. Whisking and licking are BINARY -- one bit each -- and a slim
# band carries one bit better than a shade does. Three aligned images: the paw raster,
# then whisk, then lick, sharing the x axis and the trial ordering. Nothing is lost, the
# collision is gone, and it matches how the structural-coefficient panel already splits
# paw from whisk/lick, so the two figures stop disagreeing about what colour means.
#
# MISSING BINS ARE HATCHED, NOT WHITE. Paw state 0 is '#EFE6E1' -- nearly white, and 36%
# of all frames -- so a NaN drawn as white would be indistinguishable from the commonest
# state, in figures whose whole point is sometimes that a session is badly tracked. The
# axes carry a hatched patch underneath and NaNs are left transparent, so a gap reads as
# a gap.
NODATA_HATCH = '////'
NODATA_EDGE = '#C8C8C8'


def decode_syllables(codes, n_paw_states=N_PAW_STATES):
    """Unpack `code = paw + n*whisk + 2n*lick` into (paw, whisk, lick).

    NaN in, NaN out -- a missing bin stays missing in all three, rather than decoding to
    paw 0 (which is a real and very common state, so the mistake would be invisible).
    """
    c = np.asarray(codes, dtype=float)
    bad = np.isnan(c)
    ci = np.where(bad, 0, c).astype(int)
    paw = np.where(bad, np.nan, ci % n_paw_states)
    whisk = np.where(bad, np.nan, (ci // n_paw_states) % 2)
    lick = np.where(bad, np.nan, ci // (2 * n_paw_states))
    return paw, whisk, lick


def _binary_cmap(color):
    return ListedColormap(['#FFFFFF', color])


def syllable_raster(codes, fig=None, subplot_spec=None, sort='paw',
                    n_paw_states=N_PAW_STATES, band=0.09, epochs=True, labels=True):
    """Draw one session as three aligned bands: paw state, whisking, licking.

    `codes` is (trials, bins) of PACKED syllable codes -- the raw `binned_sequence`
    values, not a renumbering. Unpacking happens here so that every figure agrees about
    which index is which.

    sort : 'paw'  order trials by mean paw state, which is what makes the block
                  structure legible; 'none' keeps the true trial order (a raster);
                  or pass an explicit index array.
    subplot_spec : a SubplotSpec to draw into, for embedding beside other panels.
                   Without one the current or a new figure is used.

    Returns {'paw': ax, 'whisk': ax, 'lick': ax}.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from matplotlib.patches import Rectangle

    codes = np.asarray(codes, dtype=float)
    paw, whisk, lick = decode_syllables(codes, n_paw_states)

    if isinstance(sort, str):
        if sort == 'paw':
            order = np.argsort(np.nanmean(np.where(np.isnan(paw), np.nan, paw), axis=1))
        elif sort == 'none':
            order = np.arange(len(codes))
        else:
            raise ValueError("sort must be 'paw', 'none', or an index array")
    else:
        order = np.asarray(sort)
    paw, whisk, lick = paw[order], whisk[order], lick[order]

    fig = fig or plt.gcf()
    heights = [1 - 2 * band, band, band]
    if subplot_spec is None:
        gs = fig.add_gridspec(3, 1, height_ratios=heights, hspace=0.06)
    else:
        gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=subplot_spec,
                                     height_ratios=heights, hspace=0.06)
    axes = {}
    for row, (key, data, cmap) in enumerate((
            ('paw', paw, ListedColormap(PAW_STATE_COLORS)),
            ('whisk', whisk, _binary_cmap(SYLLABLE['Whisk'])),
            ('lick', lick, _binary_cmap(SYLLABLE['Lick'])))):
        ax = fig.add_subplot(gs[row])
        # the hatch sits UNDER the image; NaNs are transparent, so a missing bin shows it
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, zorder=0,
                               facecolor='white', edgecolor=NODATA_EDGE,
                               hatch=NODATA_HATCH, linewidth=0))
        cm = cmap.copy()
        cm.set_bad(alpha=0.0)
        vmax = n_paw_states - 1 if key == 'paw' else 1
        ax.imshow(np.ma.masked_invalid(data), aspect='auto', cmap=cm,
                  interpolation='none', vmin=0, vmax=vmax, zorder=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if epochs:
            epoch_lines(ax, codes.shape[1])
        if labels and key != 'paw':
            ax.set_ylabel(key, rotation=0, ha='right', va='center',
                          fontsize=plt.rcParams['font.size'] * 0.7)
        axes[key] = ax
    if labels:
        axes['paw'].set_ylabel('Trials')
    return axes


# Whisk and lick are one BIT each, so when a mark has room they can ride on top of the
# paw colour as texture instead of taking a colour channel. Compositional on purpose:
# "both" is the two marks together, so there is nothing extra to learn for it. Line vs dot
# rather than '/' vs '\\', which are mirror images and the hardest pair in the hatch
# vocabulary to tell apart; the sparse punctate mark goes to licking, which is on in ~11%
# of bins against whisking's ~48%.
PAW_HATCH = {(0, 0): '', (1, 0): '//', (0, 1): '..', (1, 1): '//..'}

# ONE BLACK INK by default. Note what it costs: the palette spans OKLCH L 0.93 to 0.30, and
# on the three darkest states -- 5, 6 and 7, which are the vigorous ones and so the likeliest
# to be whisking -- black hatch on a dark fill has little contrast and largely disappears.
# hatch_ink() below switches to white ink there and syllable_histogram(edge='auto') uses it,
# which is more legible but puts two inks in one panel. The split is in relative luminance
# rather than OKLCH so it needs no colour-space maths; 0.17 sits between state 2 (0.200) and
# state 6 (0.137), which is where OKLCH L 0.55 falls for this palette.
HATCH_INK_DARK = '#000000CC'
HATCH_INK_LIGHT = '#FFFFFFDD'
HATCH_INK_SPLIT = 0.17


def hatch_ink(fill):
    """The hatch colour to use over `fill` -- dark ink on pale fills, light on dark."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
               for c in to_rgb(fill))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return HATCH_INK_LIGHT if lum < HATCH_INK_SPLIT else HATCH_INK_DARK
HATCH_MIN_PT = 12.0     # a hatch needs roughly this much cell width to read; measured by
                        # stepping one patch down from 40pt to 3pt -- below ~10 the four
                        # textures stop separating, and below ~6 they vanish entirely


def syllable_histogram(codes, ax=None, texture=False, n_paw_states=N_PAW_STATES,
                       lowest_on_top=True, edge=None, warn=True):
    """Trial-aligned syllable histogram: one stacked column per time bin.

    Each column is the composition of syllables ACROSS TRIALS at that bin -- trial identity
    is deliberately gone, which is what separates this from syllable_raster. Equivalent to
    fig 2A's `imshow(np.sort(seq, axis=0))`, with two differences:

      * the sort key is the decoded (paw, whisk, lick), so the stack is ordered by paw
        state directly. fig 2A had to renumber the raw codes to control that order.
      * the y axis is a FRACTION of trials. fig 2A labelled it "Syllable count", but every
        column holds the same number of trials, so the height carried no information.

    texture : lay ps.PAW_HATCH over each block, so one panel carries paw side, paw vigor,
        whisking and licking together. Only worth it when bins are wide -- the blocks are
        drawn as patches so that a hatch is possible at all, and `warn` reports the bin
        width when it falls under HATCH_MIN_PT. It also costs something: whisking splits
        most paw blocks in two, so the panel gains edges that are not paw transitions,
        and the laterality contrast gets quieter. Prefer it when whisk/lick ARE the
        question, and colour alone when paw side is.

    lowest_on_top : keep fig 2A's orientation, where paw state 0 sits at the top of the
        stack. That is what imshow's default origin='upper' produced there, so matching it
        keeps old and new panels comparable. False stacks upward from state 0 instead.

    edge : the hatch colour. None means HATCH_INK_DARK -- one black ink everywhere. Pass
        'auto' for hatch_ink(), which switches to white ink on the dark fills; that reads
        better on states 5, 6 and 7, at the cost of two inks in one panel.

    Missing bins shrink the stack rather than being filled in, so a poorly tracked bin is
    a short column, not a fabricated one.
    """
    from matplotlib.patches import Rectangle
    ax = ax or plt.gca()
    codes = np.asarray(codes, dtype=float)
    paw, whisk, lick = decode_syllables(codes, n_paw_states)
    # RANK, not the raw state number. The HMM's numbering is arbitrary, so stacking on it
    # puts state 2 (vigor +0.70) above states 3 and 4 (+0.53, +0.24) and the stack's
    # lightness jumps about. Stacking on PAW_VIGOR_ORDER makes the column run
    # pale-still -> dark-vigorous, which is what lightness already means.
    rank = np.empty(n_paw_states, dtype=float)
    rank[np.asarray(PAW_VIGOR_ORDER)] = np.arange(n_paw_states)
    paw_rank = np.where(np.isnan(paw), np.nan, rank[np.where(np.isnan(paw), 0, paw).astype(int)])
    key = paw_rank * 4 + whisk + 2 * lick      # vigor sets block order, whisk/lick divide it
    unrank = np.asarray(PAW_VIGOR_ORDER)       # rank -> state, to get the colour back
    n_bins = codes.shape[1]

    if texture and warn:
        w_pt = ax.get_window_extent().width / max(n_bins, 1) * 72 / ax.figure.dpi
        if w_pt < HATCH_MIN_PT - 0.5:      # tolerance: sizing FOR the threshold lands on
                                           # 11.99 and a warning there is just noise
            print(f'  syllable_histogram: ~{w_pt:.0f} pt per bin, under the {HATCH_MIN_PT:.0f} '
                  f'a hatch needs -- widen the panel, show fewer bins, or use texture=False')

    for b in range(n_bins):
        col = key[:, b]
        col = np.sort(col[~np.isnan(col)])
        n = len(col)
        if not n:
            continue
        cuts = np.flatnonzero(np.diff(col)) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, n]):
            k = int(col[lo])
            p, wl = unrank[k // 4], k % 4      # back from vigor rank to the state index
            w, l = wl % 2, wl // 2
            y0, y1 = lo / n, hi / n
            if lowest_on_top:
                y0, y1 = 1 - y1, 1 - y0
            fill = PAW_STATE_COLORS[p]
            ink = (hatch_ink(fill) if edge == 'auto'
                   else (HATCH_INK_DARK if edge is None else edge)) if texture else 'none'
            ax.add_patch(Rectangle((b, y0), 1, y1 - y0, facecolor=fill,
                                   hatch=PAW_HATCH[(w, l)] if texture else '',
                                   edgecolor=ink, linewidth=0))
    ax.set_xlim(0, n_bins)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction of trials')
    return ax


def whisk_lick_panel(codes, ax=None, n_paw_states=N_PAW_STATES):
    """P(whisking) as a filled band and P(licking) as a line, per time bin.

    The companion to syllable_histogram(texture=False): the two binary channels as the
    proportions they are, rather than squeezed into the stack. Licking largely rides on
    whisking, which this shows directly -- the lick curve sits inside the whisk envelope.
    """
    ax = ax or plt.gca()
    _, whisk, lick = decode_syllables(np.asarray(codes, dtype=float), n_paw_states)
    t = np.arange(whisk.shape[1])
    ax.fill_between(t, np.nanmean(whisk, axis=0), color=SYLLABLE['Whisk'], lw=0,
                    label='whisk')
    ax.plot(t, np.nanmean(lick, axis=0), color=SYLLABLE['Lick'], lw=2, label='lick')
    ax.set_xlim(0, whisk.shape[1] - 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('P(on)')
    return ax


def paw_legend(target=None, ncol=4, title='Paw state', missing=True,
               hatch=False, **kw):
    """Legend for the paw raster, labelled with the side each state leads with.

    `target` is a Figure or an Axes. A FIGURE is usually what you want for a row of
    rasters -- one legend under the whole row rather than one hanging off a panel --
    and it defaults to the current figure. With `missing`, the hatch used for absent
    bins gets a key too, because an unexplained hatch reads as a rendering artifact.
    With `hatch`, the whisk/lick/both marks are keyed as well, on a neutral fill so the
    mark is what reads rather than the colour under it -- pass it whenever the panel was
    drawn with texture=True.
    """
    from matplotlib.patches import Patch
    # listed in PAW_VIGOR_ORDER, so the swatches form a smooth pale-to-dark ramp instead of
    # jumping around with the HMM's arbitrary numbering
    handles = [Patch(facecolor=PAW_STATE_COLORS[s], label=paw_label(s))
               for s in PAW_VIGOR_ORDER]
    if missing:
        handles.append(Patch(facecolor='white', edgecolor=NODATA_EDGE,
                             hatch=NODATA_HATCH, label='no data'))
    if hatch:
        # keyed on a neutral fill so the mark itself is what reads, not the colour under it
        # keyed on a neutral fill, so the mark reads rather than the colour under it; the
        # ink matches what hatch_ink() would choose for that fill
        handles += [Patch(facecolor='white', edgecolor=hatch_ink('white'),
                          hatch=PAW_HATCH[k], label=lab)
                    for k, lab in (((1, 0), 'whisk'), ((0, 1), 'lick'), ((1, 1), 'both'))]
    target = target if target is not None else plt.gcf()
    kw.setdefault('loc', 'upper center')
    kw.setdefault('bbox_to_anchor', (0.5, 0.02))
    if hasattr(target, 'add_subplot'):          # a Figure
        return target.legend(handles=handles, title=title, ncol=ncol,
                             frameon=False, **kw)
    return target.legend(handles=handles, title=title, ncol=ncol, frameon=False, **kw)


# SYLLABLE IS DERIVED FROM PAW_STATE_COLORS, never listed separately. It used to be its own
# Set3 slice, and leaving it that way while PAW_STATE_COLORS existed would have meant
# ps.SYLLABLE['Paw 3'] and ps.PAW_STATE_COLORS[3] returning DIFFERENT colours for the same
# state -- two figures in the same paper drawing state 3 in two colours, with nothing
# saying which was current. Whisk and lick keep their greys: they have no side and no vigor.
SYLLABLE_COLORS = PAW_STATE_COLORS + ['#b8b8b8', '#484949']
SYLLABLE = dict(zip(SYLLABLE_NAMES, SYLLABLE_COLORS))

# One neutral null band, not one per state. A permutation null's width depends on the
# sample size and the feature's own variance, which across these eight states differ by
# 4.6% -- so eight near-identical bands in eight colours stack into a dark block that
# carries no information and swallows the traces crossing it.
NULL_BAND = '#9AA0A6'

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
    # RETINA PREVIEW. The inline backend rasterises at figure.dpi and the notebook then
    # displays that PNG at 1 logical pixel per rendered pixel, so on a Retina screen every
    # figure is upscaled 2x and looks soft. 'retina' renders at 2x and tags the image as 2x,
    # which the browser displays at the right physical size and full sharpness. It changes
    # only the preview -- savefig is unaffected, and SVG has no pixels to begin with.
    try:
        from IPython import get_ipython
        _ip = get_ipython()
        if _ip is not None:
            _ip.run_line_magic('config', "InlineBackend.figure_format = 'retina'")
    except Exception:
        pass        # not in IPython, or the magic is unavailable -- the dpi bump still helps

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
            tag_mode=True, save=None, **kw):
    """Write a figure to figures/ under the paper's naming convention.

        <name>[_<mode>]_<DD-MM-YYYY>.png

    PNG by default because that is what goes into slides and drafts; pass svg=True to also write the vector master (text stays text --
    see paper.mplstyle -- so labels remain editable in Illustrator). `formats` takes
    an explicit tuple if you want something else entirely, e.g. ('pdf',).

    The date is the RUN date, so re-running never silently overwrites yesterday's
    figure: you get a new file and the old one stays for comparison. Pass dated=False
    for a stable filename when a manuscript references one.

    WRITING IS OPT-IN. By default this only reports the filename it WOULD write and
    returns []. Every exploratory re-run of a notebook otherwise drops another dated
    pair into figures/, so the directory fills with near-identical files and the one
    that belongs in the paper is whichever happened to run last. Turn it on for the
    run that matters, either globally or for one call:

        ps.saving(True)                 # this session writes figures
        ps.savefig(fig, 'name')         # ...as usual
        ps.savefig(fig, 'name', save=True)   # or just this one, whatever the default

    `save` overrides the module default in both directions, so a notebook that must
    always write can pass save=True and ignore the switch.
    """
    if save is None:
        save = SAVE_FIGURES
    if formats is None:
        formats = ('png', 'svg') if svg else ('png',)
    stem = name
    # poster and paper renders of the same figure must not overwrite each other
    if tag_mode and MODE != 'paper':
        stem = f'{stem}_{MODE}'
    if dated:
        stem = f'{stem}_{_date.today().strftime(DATE_FMT)}'
    out = FIGDIR if subdir is None else FIGDIR / subdir
    if not save:
        # Say what WOULD be written, and how to write it. A silent no-op is worse than
        # the old always-write: you go looking in figures/ for something that is not there.
        rel = ', '.join(str((out / f'{stem}.{f}').relative_to(ROOT)) for f in formats)
        print(f'not saved (ps.SAVE_FIGURES is False): {rel}'
              '   -- ps.saving(True) to write, or pass save=True')
        return []
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in formats:
        p = out / f'{stem}.{f}'
        fig.savefig(p, format=f, **kw)
        paths.append(p)
    print('wrote ' + ', '.join(str(p.relative_to(ROOT)) for p in paths))
    return paths


def saving(on=True):
    """Turn figure writing on or off for this session. Returns the new state, so
    `ps.saving(True)` reads as a statement in a notebook."""
    global SAVE_FIGURES
    SAVE_FIGURES = bool(on)
    print(f'figure writing {"ON -- figures/ will be updated" if SAVE_FIGURES else "OFF"}')
    return SAVE_FIGURES
