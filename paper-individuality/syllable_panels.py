"""
SYLLABLE-COMPOSITION PANELS -- one image per session/mouse, trial epochs on x
============================================================================
The panel this module draws is fig 2A: for one session, the syllable used in every
10-bin epoch of every trial, as an image. It lives here rather than in a notebook
because two notebooks now draw it -- 2_pre-trial/fig2.ipynb (the figure itself) and
4_mice/LDA_analyses_pipeline_ALLSESSIONS.ipynb (example sessions at the extremes of
the LD1/LD2 plane) -- and two copies would drift in palette, syllable ordering or
sort mode, which is exactly what makes two panels uncomparable.

Import it the same way as paper_style, from any subfolder:

    import sys, pathlib
    _p = pathlib.Path.cwd().resolve()
    while not (_p / 'paper_style.py').exists() and _p != _p.parent:
        _p = _p.parent
    sys.path.insert(0, str(_p))
    import syllable_panels as sp

NOTHING HERE RESHAPES YOUR ANALYSIS. The one transform it applies is a RELABELLING:
syllable codes are stored whisk/lick-major and the palette is paw-major, so the codes
are renumbered for display only (see to_paw_major).
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb, to_hex

import paper_style as ps

N_BINS = 10          # bins per trial epoch
N_SYLL = 32          # 8 paw states x whisk x lick
N_PAW_STATES = 8


# =============================================================================
# SYLLABLE CODES
# =============================================================================
# A syllable is a triple (paw state 0-7, whisking 0/1, licking 0/1), written as the
# 3-character string 'pwl'. It appears in the data under ONE numbering and is drawn
# under ANOTHER:
#
#   STORED     code = paw + 8*whisk + 16*lick   -- whisk/lick-major. This is what
#              `binned_sequence` holds, and what functions.binarize decodes with
#              `paw = code % 8`, `whisk = (code // 8) % 2`, `lick = code // 16`.
#   PAW-MAJOR  code = 4*paw + 2*lick + whisk    -- paw-major, so the 8 paw states are
#              4-wide blocks. The palette is 8 hues x 4 shades, so a block of 4
#              consecutive codes shares a hue and the whisk/lick combination reads as
#              a shade WITHIN that hue. Under the stored numbering the same palette
#              would cut across paw states and mean nothing.
#
# NOTE THE WEIGHTS IN THE PAW-MAJOR CODE: lick carries 2 and whisk carries 1, so within
# a paw hue the shades run (whisk0 lick0, whisk1 lick0, whisk0 lick1, whisk1 lick1) --
# shade changes fastest with WHISKING and the hue's light/dark halves are lick off/on.
# That is not the obvious ordering, but it is the one fig2 was drawn with, and matching
# it is the whole point of this module: swapping the two weights recolours 16 of the 32
# syllables and silently makes the LDA-extremes panels uncomparable with fig 2A.
STRING_TO_STORED = {f'{p}{w}{l}': float(p + 8 * w + 16 * l)
                    for p in range(N_PAW_STATES) for w in (0, 1) for l in (0, 1)}
STRING_TO_PAW_MAJOR = {f'{p}{w}{l}': float(4 * p + 2 * l + w)
                       for p in range(N_PAW_STATES) for w in (0, 1) for l in (0, 1)}
STORED_TO_STRING = {v: k for k, v in STRING_TO_STORED.items()}


def to_paw_major(codes):
    """Stored syllable codes -> paw-major codes, for plotting only.

    NaN-SAFE AND WITHOUT A DICT ROUND-TRIP. The notebook version did
    `np.vectorize(dict.get)` twice, code -> 'pwl' -> paw-major code. Two problems with
    that: a dict cannot be looked up by NaN (since Python 3.10 nan hashes by identity,
    so a nan that is not the very `np.nan` object used as the key silently misses) and
    `.get` returns None on a miss rather than raising -- so an unmapped or missing bin
    became None inside an array that is then assigned into a float image. Arithmetic
    on the codes is exact, total, and propagates NaN as NaN.
    """
    c = np.asarray(codes, dtype=float)
    paw = np.mod(c, N_PAW_STATES)
    whisk = np.mod(np.floor_divide(c, N_PAW_STATES), 2)
    lick = np.floor_divide(c, 2 * N_PAW_STATES)
    return 4 * paw + 2 * lick + whisk


# the arithmetic above must agree with the tables it replaces -- cheap enough to check
# on import, and it is the only thing standing between a palette and a wrong palette
_codes = np.array(sorted(STORED_TO_STRING))
assert np.array_equal(to_paw_major(_codes),
                      np.array([STRING_TO_PAW_MAJOR[STORED_TO_STRING[c]] for c in _codes])), \
    'to_paw_major disagrees with STRING_TO_PAW_MAJOR'
assert np.isnan(to_paw_major([np.nan]))[0]
del _codes


# =============================================================================
# PALETTE -- 8 hues (paw state) x 4 shades (whisk/lick)
# =============================================================================
def _base_hues():
    """Set3 with yellow and the greys dropped, so no paw state is drawn in a colour
    that reads as 'missing' -- indices 1, 8, 9, 10 out, then the first and last of what
    is left swapped (the original ordering choice in fig2)."""
    set3 = sns.color_palette('Set3', n_colors=20)
    keep = [c for i, c in enumerate(set3) if i not in (1, 8, 9, 10)][:N_PAW_STATES]
    return [keep[0], keep[7]] + keep[1:7]


def syllable_cmap(n_groups=N_PAW_STATES, shades_per_group=4, base_palette=None):
    """ListedColormap of 32 colours: shade 0 (lightest) .. shade 3 (darkest) within
    each paw hue. `np.clip` guards the ends -- factor 1.5 on a dark base colour would
    otherwise send a channel negative and to_hex would raise."""
    base_colors = _base_hues() if base_palette is None else list(base_palette)[:n_groups]
    full = []
    for color in base_colors:
        rgb = np.array(to_rgb(color))
        for f in np.linspace(0.5, 1.5, shades_per_group):
            full.append(to_hex(np.clip(rgb * f + (1 - f), 0, 1)))
    return ListedColormap(full)


def syllable_norm(n_syll=N_SYLL):
    """One colour per integer code. imshow MUST be given this: left to autoscale, a
    session that never uses syllable 31 would recolour every other syllable and two
    panels would no longer be comparable."""
    return BoundaryNorm(np.arange(-0.5, n_syll), n_syll)


# =============================================================================
# THE IMAGE
# =============================================================================
def session_image(all_sequences, session, epochs=None, n_bins=N_BINS,
                  sort_mode='within_bin', syllable_order='descending',
                  session_col='session'):
    """(image, n_trials) for one session -- rows trials, columns epoch x bin.

    sort_mode
      'within_bin'  each COLUMN is sorted independently, so a row is a different trial
                    in each bin and rows do not track trials at all. What stays readable
                    is the vertical extent of each colour in a column: sweeping up a
                    column accumulates trials, syllable by syllable, until every trial
                    in that bin is accounted for. Hence 'Cumulative trials' as the y
                    label -- a raster's 'Trial' would misdescribe it.
      'none'        rows in file order, y labelled 'Trial'. USE WITH CARE: the four
                    epochs are stacked POSITIONALLY, not keyed on `sample`, and in this
                    file they neither hold the same trials (e.g. 554 Pre-quiescence rows
                    against 527 Choice rows in one session) nor the same order (rows are
                    sorted lexicographically, so '0.0' leads in some epochs and not in
                    others). Row k is therefore a DIFFERENT trial in each epoch block --
                    checked: true of all 332 sessions in the syllable file. Only the
                    within-bin composition is meaningful, which is what 'within_bin'
                    shows; fixing the raster means pivoting on `sample` first, as
                    functions.build_design_matrix does.

    syllable_order  'descending' puts the highest code at the bottom of the stack.
                    Negate-sort-negate, not `np.sort(...)[::-1]`: the latter drags the
                    empty (NaN) rows to the bottom and opens a blank gap under the stack.
    """
    epochs = ps.EPOCH_NAMES if epochs is None else epochs
    n_cols = len(epochs) * n_bins

    data = all_sequences.loc[all_sequences[session_col] == session]
    if len(data) == 0:
        raise KeyError(f'no rows for session {session!r} in column {session_col!r}')
    n_trials = int(len(data) / len(epochs))

    img = np.full((n_trials, n_cols), np.nan)
    for e, epoch in enumerate(epochs):
        rows = data.loc[data['broader_label'] == epoch, 'binned_sequence'].values
        if len(rows) == 0:
            continue
        epoch_data = to_paw_major(np.vstack(rows)[:n_trials, :])
        img[:epoch_data.shape[0], n_bins * e:n_bins * (e + 1)] = epoch_data

    if sort_mode == 'within_bin':
        img = (-np.sort(-img, axis=0) if syllable_order == 'descending'
               else np.sort(img, axis=0))
    return img, n_trials


def missing_fraction(all_sequences, session, **kw):
    """Fraction of the panel that is missing data -- bins with no syllable. Worth
    printing next to any panel that is being read as behaviour: a session can be short
    of tracking for a whole epoch and the sorted image just looks like a smaller stack."""
    img, _ = session_image(all_sequences, session, **kw)
    return float(np.isnan(img).mean())


def session_missing_fractions(all_sequences, session_col='session'):
    """Fraction of NaN bins per session, for EVERY session at once -- a Series indexed
    by session id. Use this to screen candidates before choosing which sessions to draw;
    calling missing_fraction() in a loop would build one image per session for a number
    that needs no image.

    Slightly smaller than missing_fraction() for the same session: this counts only bins
    that are NaN in the stored sequences, while the drawn image also carries NaN padding
    rows where an epoch holds fewer trials than the session's mean trial count. Across
    this file the two agree to within ~0.002 (0.0097 vs 0.0115 on average).
    """
    bins = np.isnan(np.stack(all_sequences['binned_sequence'].to_numpy())).mean(axis=1)
    return pd.Series(bins, index=all_sequences[session_col].to_numpy()).groupby(level=0).mean()


def plot_session(ax, all_sequences, session, epochs=None, n_bins=N_BINS,
                 sort_mode='within_bin', syllable_order='descending',
                 cmap=None, norm=None, session_col='session',
                 xticklabels=True, ylabel=None, title=None,
                 nan_color=None, annotate_missing=False):
    """Draw one session's panel on `ax`; returns the AxesImage.

    origin='lower' in sorted mode so the count builds upward from 0 and NaN (trials
    with no data in that bin) lands at the TOP, above the stack, where it reads as
    absent rather than as a state.

    nan_color         missing bins are transparent by default, which is what fig 2A
                      was drawn with -- they read as page. Pass a colour ('0.9', say)
                      to paint them instead, so a session missing a third of its ITI
                      cannot be mistaken for one with fewer trials.
    annotate_missing  append the missing fraction to the title. Off by default; on
                      wherever panels of different sessions are compared.
    """
    epochs = ps.EPOCH_NAMES if epochs is None else epochs
    n_cols = len(epochs) * n_bins
    cmap = syllable_cmap() if cmap is None else cmap
    norm = syllable_norm() if norm is None else norm
    if nan_color is not None:
        cmap = cmap.copy()
        cmap.set_bad(nan_color)

    img, n_trials = session_image(all_sequences, session, epochs=epochs, n_bins=n_bins,
                                  sort_mode=sort_mode, syllable_order=syllable_order,
                                  session_col=session_col)
    if sort_mode == 'within_bin':
        im = ax.imshow(img, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                       origin='lower', extent=(0, n_cols, 0, n_trials))
        ax.set_ylim(0, n_trials)
        default_ylabel = 'Cumulative trials'
    else:
        im = ax.imshow(img, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                       extent=(0, n_cols, n_trials, 0))
        default_ylabel = 'Trial'

    ps.epoch_lines(ax, n_bins=n_cols)
    ax.set_xticks(np.arange(n_bins, 3 * n_bins + 1, n_bins))
    ax.set_xticklabels(['Quiescence', 'Stimulus', 'Response'] if xticklabels else [],
                       rotation=30, ha='right')
    ax.set_xlabel('')
    if ylabel is not None:
        ax.set_ylabel(default_ylabel if ylabel is True else ylabel)
    if annotate_missing:
        _miss = np.isnan(img).mean()
        title = f'{title}\n{_miss:.1%} missing' if title else f'{_miss:.1%} missing'
    if title:
        ax.set_title(title)
    return im


def session_panels(all_sequences, sessions, titles=None, kind=None, scale=.6,
                   sharey=False, widen=None, **kw):
    """A row of panels, one per session. Returns (fig, axs, im).

    Panel sizes come from paper_style, so ps.use('paper') re-renders the row at journal
    size. Only the leftmost panel is given a y label; `sharey=False` by default because
    sessions differ in trial count and forcing a common y axis would make a short
    session look like a truncated long one.

    widen  a 'double' figure is two columns wide whatever `ncols` says, so a row of 4
           sessions would be squeezed into the width of 2. By default the width is
           scaled by n/2 once there are more than two panels; widen=False keeps the
           fixed journal width (and the 1- and 2-panel cases are untouched either way).
    """
    sessions = list(sessions)
    n = max(len(sessions), 1)
    kind = ('square' if n == 1 else 'double') if kind is None else kind
    fig, axs = ps.figure(kind, ncols=n, sharex=True, sharey=sharey,
                         squeeze=False, scale=scale)
    if (widen is None and n > 2) or widen:
        _w, _h = fig.get_size_inches()
        fig.set_size_inches(_w * n / 2, _h)
    axs = axs[0]
    cmap, norm = syllable_cmap(), syllable_norm()
    im = None
    for i, s in enumerate(sessions):
        im = plot_session(axs[i], all_sequences, s, cmap=cmap, norm=norm,
                          title=None if titles is None else titles[i],
                          ylabel=True if i == 0 else None, **kw)
    return fig, axs, im


def palette_strip(ax=None, cmap=None):
    """The 32-colour key as a strip, for a legend panel."""
    cmap = syllable_cmap() if cmap is None else cmap
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 1))
    ax.imshow(np.linspace(0, 1, cmap.N).reshape(1, -1), aspect='auto', cmap=cmap)
    ax.set_axis_off()
    return ax
