"""
Plotting for the 5.x HMM fits — inspection only, never fits.

Reads the pickles `hmm_functions.run_session` wrote, which already contain the decoded
states, so a whole cohort reviews in seconds.

Panels are variable-aware: a continuous signal (whisker ME, wheel velocity) is drawn as a
trace, whereas lick counts are drawn as event ticks, because a count series plotted as a
line is unreadable at 60 Hz.

Colours are slots 1-3 of a CVD-validated categorical palette (blue / orange / aqua) plus
text greys. Single-series panels use one hue; panels with two categories carry a legend, so
identity is never colour-alone.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hmm_functions import (dwell_times, load_fit_variable, prepare_batches, MODELS,
                           coarsen, BIN_AGG)

SURFACE = '#fcfcfb'
INK, INK_2, INK_MUTED = '#0b0b0b', '#52514e', '#87867f'
SERIES_1, SERIES_2, SERIES_3 = '#2a78d6', '#eb6834', '#1baf7a'
BAND = '#cde2fb'

# variables whose values are counts of events rather than a continuous level
COUNT_VARS = {'Lick count'}


def _style(ax, grid=True):
    """Recessive chrome: hairline SOLID grid and axes (dashes would read as a threshold)."""
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
    if grid:
        ax.grid(color=INK_MUTED, lw=0.4, alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
    return ax


def runs_of(mask):
    """Contiguous True runs as [(start, stop), ...]."""
    e = np.flatnonzero(np.diff(np.concatenate(([0], np.asarray(mask).astype(int), [0]))))
    return list(zip(e[::2], e[1::2]))


# ============================================================================
# LOADING
# ============================================================================

def available_sessions(save_path, var):
    """[(mouse_name, session), ...] for every finished fit. Reads the save directory, so
    it works while a run is still going."""
    pre = f'best_results_{var}_'
    out = []
    for name in sorted(os.listdir(save_path)):
        if name.startswith(pre) and not name.endswith('.tmp'):
            rest = name[len(pre):]
            out.append((rest[:-36], rest[-36:]))
    return out


def resolve_session(save_path, var, eid=None, mouse_name=None, session=None):
    """Accepts a full eid, an eid prefix (the 8 characters shown in titles), or an
    explicit mouse/session pair, and returns (mouse_name, session)."""
    if mouse_name is not None and session is not None:
        return mouse_name, session
    key = eid if eid is not None else session
    if key is None:
        raise ValueError('give eid=... or mouse_name=... and session=...')
    hits = [(m, s) for m, s in available_sessions(save_path, var) if s.startswith(key)]
    if not hits:
        raise FileNotFoundError(f'no finished fit whose eid starts with {key!r}')
    if len(hits) > 1:
        raise ValueError(f'{key!r} matches {len(hits)}: {hits} — give more characters')
    return hits[0]


def load_result(save_path, var, mouse_name=None, session=None, eid=None):
    """The saved record for one session. `eid` alone is enough."""
    mouse_name, session = resolve_session(save_path, var, eid=eid,
                                          mouse_name=mouse_name, session=session)
    with open(os.path.join(save_path, f'best_results_{var}_{mouse_name}{session}'),
              'rb') as f:
        return pickle.load(f)


def load_signal(data_path, session, mouse_name, var, zsc, num_train_batches, model,
                n_states=None, bin_frames=1):
    """The signal exactly as the fit saw it, so it is aligned sample for sample.

    `bin_frames` must match the fit: a fit run on 100 ms bins stores its states at BIN
    resolution, so the signal has to be coarsened the same way or the two are neither
    the same length nor the same time base. Read it from the fit's own config rather
    than passing it by hand -- see plot_session.
    """
    # load_fit_variable returns (array_matrix, bins) -- it must be unpacked, or the
    # whole tuple reaches prepare_batches and np.shape() raises an inhomogeneous-shape
    # ValueError. run_session already unpacks it; this did not.
    dm, bins = load_fit_variable(data_path, session, mouse_name, [var], zsc,
                                 binarise=MODELS[model]['binarise'])
    dm, _bins, _n = coarsen(dm, bins, bin_frames, how=BIN_AGG[model])
    sa, _, _ = prepare_batches(dm, num_train_batches)
    sig = np.asarray(sa)[:, 0]
    return sig if n_states is None else sig[:n_states]


# ============================================================================
# PER-SESSION
# ============================================================================

def pick_window(states, fps, win_s=15, target_transitions=7):
    """Start frame of an informative window: balanced occupancy, a few transitions.
    A window chosen at random is usually all one state and shows nothing."""
    win = int(win_s * fps)
    best, best_score = 0, -np.inf
    for c in range(0, max(len(states) - win, 1), int(fps * 2)):
        seg = states[c:c + win]
        if len(seg) < win:
            break
        score = (-abs(float(seg.mean()) - 0.5) * 12
                 - abs(int(np.sum(np.diff(seg) != 0)) - target_transitions) * 0.4)
        if score > best_score:
            best_score, best = score, c
    return best


def plot_states(d, signal, var, fps, win_s=15, ax=None, start=None, start_s=None,
                compare_states=None, labels=('fit', 'compare')):
    """The segmentation on the data. Pale band + ribbon = the high state.

    Window placement: `start_s` (seconds into the session) wins, then `start` (frame),
    otherwise an informative window is picked automatically. Pass `compare_states` to
    overlay a second segmentation with the disagreeing frames shaded.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 2.8))
    states = np.asarray(d['most_likely_states'])
    comparing = compare_states is not None
    win = int(win_s * fps)
    if start_s is not None:
        st = int(round(float(start_s) * fps))
    elif start is not None:
        st = int(start)
    elif comparing:
        rr = runs_of(np.asarray(compare_states) != states)
        st = ((sum(max(rr, key=lambda t: t[1] - t[0])) // 2) - win // 2) if rr \
            else pick_window(states, fps, win_s)
    else:
        st = pick_window(states, fps, win_s)
    st = int(np.clip(st, 0, max(len(states) - win, 0)))

    t = np.arange(win) / fps
    sig, seg = signal[st:st + win], states[st:st + win]
    is_count = var in COUNT_VARS
    _style(ax, grid=False)

    if is_count:
        lo, hi = 0., 1.
        ribbon_y, ribbon_h, gap = 0.06, 0.10, 0.20
    else:
        pad = 0.75 if comparing else 0.35
        lo, hi = float(np.nanmin(sig)) - pad, float(np.nanmax(sig)) + 0.5
        ribbon_y, ribbon_h, gap = lo + 0.05, 0.15, 0.22

    for p, q in runs_of(seg == 1):
        ax.axvspan(t[p], t[min(q, win - 1)], color=BAND, lw=0, zorder=0)
    if comparing:
        cmp_seg = np.asarray(compare_states)[st:st + win]
        for p, q in runs_of(cmp_seg != seg):
            ax.axvspan(t[p], t[min(q, win - 1)], color=SERIES_2, alpha=0.40, lw=0, zorder=1)
        rows = [(seg, labels[0]), (cmp_seg, labels[1])]
    else:
        rows = [(seg, labels[0])]

    for k, (row, lab) in enumerate(rows):
        y = ribbon_y + k * gap
        for p, q in runs_of(row == 1):
            ax.fill_between([t[p], t[min(q, win - 1)]], y, y + ribbon_h,
                            color=SERIES_1, lw=0, zorder=4)
        if comparing:
            ax.annotate(lab, xy=(-0.006, y + ribbon_h / 2),
                        xycoords=('axes fraction', 'data'), fontsize=7.5,
                        color=INK_2, ha='right', va='center')

    if is_count:
        ev = np.flatnonzero(sig > 0)
        ax.vlines(t[ev], 0.45, 0.92, color=INK, lw=0.8, zorder=3)
        ax.set_yticks([])
        ax.set_ylabel(f'{var}', fontsize=8.5, color=INK_2)
    else:
        ax.plot(t, sig, color=INK, lw=0.9, zorder=2)
        ax.set_ylabel(f'{var}' + (' (z)' if d['config']['zsc'] else ''),
                      fontsize=8.5, color=INK_2)
    ax.set_xlim(0, t[-1]); ax.set_ylim(lo, hi)
    ax.set_xlabel('time (s)', fontsize=8.5, color=INK_2)

    a = d['assessment']
    flags = [k for k in ('collapsed', 'degenerate_occupancy', 'flickering') if a.get(k)]
    head = f"dwell {a['median_dwell_ms']:.0f} ms · {a['n_segments']:,} segments · " \
           f"occupancy {a['occupancy_state1']:.2f}"
    if comparing:
        head = f"{np.mean(np.asarray(compare_states) == states):.1%} of frames agree · " + head
    ax.set_title(head + ('' if not flags else '   ·   FLAGGED: ' + ', '.join(flags)),
                 fontsize=9, color=INK if not flags else SERIES_2, loc='left', pad=17)
    ax.annotate(f'start_s={st / fps:.0f} of {len(states) / fps:.0f} · '
                f'shaded = high state' + (' · orange = disagreement' if comparing else ''),
                xy=(0, 1.0), xycoords='axes fraction', xytext=(0, 2),
                textcoords='offset points', fontsize=7.5, color=INK_MUTED)
    return ax


def plot_session(save_path, data_path, var, mouse_name=None, session=None, eid=None,
                 num_train_batches=5, fps=60., win_s=15, start=None, start_s=None,
                 compare_states=None, labels=('fit', 'compare'), figsize=(12, 2.9)):
    """One session's segmentation. Returns (fig, record)."""
    mouse_name, session = resolve_session(save_path, var, eid=eid,
                                          mouse_name=mouse_name, session=session)
    d = load_result(save_path, var, mouse_name, session)
    states = np.asarray(d['most_likely_states'])
    # bin width the fit actually used; older pickles predate the option, hence the default
    bin_frames = int(d['config'].get('bin_frames', 1))
    signal = load_signal(data_path, session, mouse_name, var, d['config']['zsc'],
                         num_train_batches, d['config']['model'], n_states=len(states),
                         bin_frames=bin_frames)
    # the series being plotted advances one sample per BIN, so the time axis rate is
    # fps / bin_frames -- otherwise every duration on the plot is wrong by that factor
    fps_plot = fps / bin_frames
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    plot_states(d, signal, var, fps_plot, win_s=win_s, ax=ax, start=start, start_s=start_s,
                compare_states=compare_states, labels=labels)
    binlab = '' if bin_frames == 1 else f'  ·  {bin_frames * 1000.0 / fps:.0f} ms bins'
    ax.set_title(f'{mouse_name}  {session[:8]}  ·  {var} ({d["config"]["model"]}){binlab}  ·  '
                 + ax.get_title(), fontsize=9, color=ax.title.get_color(),
                 loc='left', pad=17)
    fig.subplots_adjust(top=0.76, bottom=0.19, left=0.075, right=0.99)
    return fig, d


# ============================================================================
# COHORT SUMMARY
# ============================================================================

def plot_summary(assessments, var, fps=60., target_dwell_ms=None, min_dwell_ms=167.,
                 figsize=(14, 8)):
    """Summary statistics of the fits: six panels, each answering one question.

    `target_dwell_ms` draws an external reference (for whisker ME the model-free
    changepoint duration is ~450 ms); leave it None for licking, where no such anchor
    exists.
    """
    df = assessments.copy()
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    axA, axB, axC, axD, axE, axF = axes.ravel()

    # A -- syllable duration
    _style(axA)
    dw = df.median_dwell_ms.dropna()
    if len(dw):
        axA.hist(dw, bins=30, color=SERIES_1, zorder=3)
        axA.axvline(min_dwell_ms, color=SERIES_2, lw=1.2, ls=(0, (4, 3)), zorder=4)
        axA.annotate(f'flicker screen\n{min_dwell_ms:.0f} ms', xy=(min_dwell_ms, 0.97),
                     xycoords=('data', 'axes fraction'), xytext=(4, 0),
                     textcoords='offset points', fontsize=7.5, color=SERIES_2, va='top')
        if target_dwell_ms:
            axA.axvline(target_dwell_ms, color=SERIES_3, lw=1.2, ls=(0, (4, 3)), zorder=4)
            axA.annotate(f'changepoint\n{target_dwell_ms:.0f} ms',
                         xy=(target_dwell_ms, 0.97), xycoords=('data', 'axes fraction'),
                         xytext=(4, 0), textcoords='offset points', fontsize=7.5,
                         color=SERIES_3, va='top')
    axA.set_xlabel('median dwell (ms)', fontsize=8.5, color=INK_2)
    axA.set_ylabel('sessions', fontsize=8.5, color=INK_2)
    axA.set_title('A · syllable duration', fontsize=10, color=INK, loc='left', pad=16)

    # B -- occupancy
    _style(axB)
    occ = df.occupancy_state1.dropna()
    if len(occ):
        axB.hist(occ, bins=np.linspace(0, 1, 41), color=SERIES_1, zorder=3)
        for edge in (0.02, 0.98):
            axB.axvline(edge, color=SERIES_2, lw=1.1, ls=(0, (4, 3)), zorder=4)
        axB.set_xlim(0, 1)
    axB.set_xlabel('fraction of frames in the high state', fontsize=8.5, color=INK_2)
    axB.set_ylabel('sessions', fontsize=8.5, color=INK_2)
    axB.set_title('B · is either state vestigial?', fontsize=10, color=INK, loc='left', pad=16)
    axB.annotate('dashed = the 2% / 98% degeneracy screen', xy=(0, 1.015),
                 xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # C -- internal consistency
    _style(axC)
    ok = df.fit_ok.fillna(False) if 'fit_ok' in df else pd.Series(True, index=df.index)
    if {'median_dwell_ms', 'n_segments'} <= set(df):
        axC.scatter(df.median_dwell_ms[ok], df.n_segments[ok], s=28, color=SERIES_1,
                    lw=0, alpha=0.85, zorder=3, label='fit_ok')
        axC.scatter(df.median_dwell_ms[~ok], df.n_segments[~ok], s=40, facecolor=SURFACE,
                    edgecolor=SERIES_2, lw=1.5, zorder=4, label='flagged')
        axC.set_xscale('log'); axC.set_yscale('log')
        if (~ok).any():
            axC.legend(frameon=False, fontsize=7.5, labelcolor=INK_2)
    axC.set_xlabel('median dwell (ms)', fontsize=8.5, color=INK_2)
    axC.set_ylabel('segments', fontsize=8.5, color=INK_2)
    axC.set_title('C · internal consistency', fontsize=10, color=INK, loc='left', pad=16)
    axC.annotate('a clean inverse relation; outliers are degenerate fits',
                 xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # D -- the model's high/low claim
    _style(axD)
    if {'level_low', 'level_high'} <= set(df):
        axD.scatter(df.level_low, df.level_high, s=28, color=SERIES_1, lw=0, alpha=0.85,
                    zorder=3)
        lim = [min(df.level_low.min(), df.level_high.min()),
               max(df.level_low.max(), df.level_high.max())]
        axD.plot(lim, lim, color=INK_MUTED, lw=0.9, zorder=1)
        axD.annotate('equal levels = no contrast', xy=(lim[1], lim[1]), xytext=(-4, -12),
                     textcoords='offset points', ha='right', fontsize=7.5, color=INK_MUTED)
    axD.set_xlabel('low-state level', fontsize=8.5, color=INK_2)
    axD.set_ylabel('high-state level', fontsize=8.5, color=INK_2)
    axD.set_title('D · what the two states are', fontsize=10, color=INK, loc='left', pad=16)
    axD.annotate('distance from the line is the separation the model found',
                 xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # E -- how much was learned
    _style(axE)
    if 'bits_LL' in df and df.bits_LL.notna().any():
        axE.hist(df.bits_LL.dropna(), bins=30, color=SERIES_1, zorder=3)
    axE.set_xlabel('bits_LL (held-out, vs the unfitted baseline)', fontsize=8.5, color=INK_2)
    axE.set_ylabel('sessions', fontsize=8.5, color=INK_2)
    axE.set_title('E · how much the model learned', fontsize=10, color=INK, loc='left', pad=16)
    axE.annotate('low bits does NOT imply a bad segmentation — screens are independent',
                 xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # F -- failure screens
    _style(axF)
    names, counts = [], []
    for c in ('collapsed', 'degenerate_occupancy', 'flickering'):
        if c in df:
            names.append(c.replace('_', ' ')); counts.append(int(df[c].fillna(False).sum()))
    if 'n_folds_failed' in df:
        names.append('fold NaNs'); counts.append(int((df.n_folds_failed.fillna(0) > 0).sum()))
    if 'error' in df:
        names.append('errored'); counts.append(int((df.error.fillna('') != '').sum()))
    if 'fit_ok' in df:
        names.append('NOT fit_ok'); counts.append(int((~df.fit_ok.fillna(False)).sum()))
    y = np.arange(len(names))
    axF.barh(y, counts, color=SERIES_1, height=0.6, zorder=3)
    for yi, c in zip(y, counts):
        axF.annotate(f'{c}', xy=(c, yi), xytext=(4, 0), textcoords='offset points',
                     va='center', fontsize=8, color=INK_2)
    axF.set_yticks(y); axF.set_yticklabels(names, fontsize=8); axF.invert_yaxis()
    axF.set_xlim(0, max(max(counts) * 1.18, 1) if counts else 1)
    axF.set_xlabel(f'sessions (of {len(df)})', fontsize=8.5, color=INK_2)
    axF.set_title('F · failure screens', fontsize=10, color=INK, loc='left', pad=16)
    axF.annotate('three independent screens — a fit can pass two and fail the third',
                 xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    model = df.model.iloc[0] if 'model' in df and len(df) else '?'
    fig.suptitle(f'{var} — {model} HMM, {len(df)} sessions', fontsize=12.5, color=INK,
                 x=0.006, ha='left', y=0.985)
    fig.subplots_adjust(top=0.88, bottom=0.075, left=0.06, right=0.99,
                        wspace=0.27, hspace=0.46)
    return fig


def per_mouse_summary(assessments):
    """One row per mouse: how consistent is a mouse across its own sessions?"""
    df = assessments
    return (df.groupby('mouse')
            .agg(n=('eid', 'count'),
                 dwell_ms=('median_dwell_ms', 'median'),
                 dwell_spread=('median_dwell_ms', lambda s: float(s.max() - s.min())),
                 occupancy=('occupancy_state1', 'median'),
                 bits=('bits_LL', 'median'),
                 flagged=('fit_ok', lambda s: int((~s.fillna(False)).sum())))
            .sort_values('n', ascending=False))
