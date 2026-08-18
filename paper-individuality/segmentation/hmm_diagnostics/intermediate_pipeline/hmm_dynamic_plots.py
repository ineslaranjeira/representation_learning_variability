"""
Visualisation for the dynamic HMM search — inspection only, never refits.

Why this is a separate module from `hmm_dynamic_functions`: that one fits, this one
only looks. It never fits or cross-validates, but it CAN re-decode a session at any
lag that was in its grid (`decode_at_lag`), because every grid cell's fitted parameters
are in the pickle. That is what makes the lag-1 comparison free.

Everything is read back out of the pickles `run_session` already wrote, which store
`most_likely_states`, the per-fold LL table, the lag profile and every selection step.
So a whole cohort reviews in seconds. 4.2 could not do this — it re-initialised a model
and re-ran `most_likely_states` for every session just to look at one, which is why
looking at the fits used to cost as much as making them.

The panels, and what each is for:

  plot_states        the actual segmentation on the actual trace — the only panel that
                     can reveal a fit that is numerically fine and behaviourally absurd
  plot_lag_profile   held-out LL vs lag: context, i.e. what a naive argmax would have done
  plot_paired_gains  the quantity the selection rule ACTUALLY tests — the paired per-fold
                     gain against the currently adopted lag, with its 95% CI
  plot_cohort        the whole set at once: lag distribution, dwell, bits, flag counts

Colours are slots 1–3 of the validated categorical palette (blue / orange / aqua) plus
text greys. Single-series panels use one hue; the one two-category panel (adopted vs
rejected lag) carries a legend, so identity is never colour-alone.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

# reuse the loading/prep logic from the fitting module rather than restating it, so a
# figure can never disagree with the fit about what the data was
from hmm_dynamic_functions import (load_fit_variable, prepare_batches, dwell_times,
                                   compute_inputs, decode_ar_states, orient_states,
                                   pick_fold)


# ============================================================================
# STYLE
# ============================================================================

SURFACE = '#fcfcfb'
INK, INK_2, INK_MUTED = '#0b0b0b', '#52514e', '#87867f'
SERIES_1, SERIES_2, SERIES_3 = '#2a78d6', '#eb6834', '#1baf7a'   # blue, orange, aqua
BAND = '#cde2fb'      # pale blue fill for the high state


def _style(ax, grid=True):
    """Recessive chrome: hairline solid grid and axes, no top/right spines."""
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
    if grid:
        # solid, one shade off the surface -- dashes would read as a threshold
        ax.grid(color=INK_MUTED, lw=0.4, alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
    return ax


# ============================================================================
# FINDING AND LOADING WHAT IS ON DISK
# ============================================================================

def available_sessions(save_path, var_interest, sticky=False):
    """Which sessions have a finished fit. Same idea as 4.2's 'sessions available'
    cell, but it reads the save directory instead of re-deriving the list from the
    design matrices — so it also catches fits whose design matrix has since moved.

    Returns [(mouse_name, session), ...] sorted by mouse.
    """
    pre = f"{'best_sticky' if sticky else 'best'}_results_{var_interest[0]}_"
    out = []
    for name in sorted(os.listdir(save_path)):
        if name.startswith(pre):
            rest = name[len(pre):]
            out.append((rest[:-36], rest[-36:]))     # eid is the trailing 36 chars
    return out


def resolve_session(save_path, var_interest, eid=None, mouse_name=None, session=None,
                    sticky=False):
    """Turn whatever identifier you have into (mouse_name, session).

    Accepts a full eid, an eid prefix (the 8 characters printed in the titles), or an
    explicit mouse/session pair. Mouse name is looked up from the filename, so you never
    have to remember which mouse an eid belongs to.
    """
    if mouse_name is not None and session is not None:
        return mouse_name, session
    key = eid if eid is not None else session
    if key is None:
        raise ValueError('give either eid=... or mouse_name=... and session=...')
    hits = [(m, s) for m, s in available_sessions(save_path, var_interest, sticky=sticky)
            if s.startswith(key) or s == key]
    if not hits:
        raise FileNotFoundError(f'no finished fit whose eid starts with {key!r} in {save_path}')
    if len(hits) > 1:
        raise ValueError(f'{key!r} matches {len(hits)} sessions: {hits} — give more characters')
    return hits[0]


def load_result(save_path, var_interest, mouse_name=None, session=None, sticky=False,
                eid=None):
    """Load one session's rich pickle (the dict `run_session` saved).

    `eid` alone is enough — a full eid or just its first 8 characters.
    """
    mouse_name, session = resolve_session(save_path, var_interest, eid=eid,
                                          mouse_name=mouse_name, session=session,
                                          sticky=sticky)
    fit_id = str(mouse_name + session)
    fn = os.path.join(
        save_path, f"{'best_sticky' if sticky else 'best'}_results_{var_interest[0]}_{fit_id}")
    with open(fn, "rb") as f:
        return pickle.load(f)


def load_prepared(data_path, session, mouse_name, var_interest, zsc, num_train_batches):
    """The `shortened_array` the fit actually saw, shape (T, emission_dim).

    Same `load_fit_variable` + `prepare_batches` as `run_session`, so it is aligned with
    `most_likely_states` frame for frame -- and it is what `decode_at_lag` needs.
    """
    design_matrix = load_fit_variable(data_path, session, mouse_name, var_interest, zsc)
    shortened_array, _, _ = prepare_batches(design_matrix, num_train_batches)
    return np.asarray(shortened_array)


def load_signal(data_path, session, mouse_name, var_interest, zsc, num_train_batches,
                n_states=None):
    """The fitted signal as a 1-D trace, truncated to n_states if given (belt and
    braces, they should already match)."""
    sig = load_prepared(data_path, session, mouse_name, var_interest, zsc,
                        num_train_batches)[:, 0]
    return sig if n_states is None else sig[:n_states]


def legacy_result(legacy_path, var_interest, mouse_name, session):
    """The previous pipeline's output for this session, in 4.2's format.

    4.2 wrote `(most_likely_states, use_fold, (num_states, lag, kappa))` to
    <legacy_path>/<var>_<mouse><eid>. Returns None when the session was not in that run
    (the old whisker set had 319 sessions, the new one has 342).
    """
    fn = os.path.join(legacy_path, var_interest[0] + '_' + str(mouse_name) + str(session))
    if not os.path.exists(fn) or os.path.getsize(fn) == 0:
        return None
    with open(fn, 'rb') as f:
        return pickle.load(f)


def legacy_lag(legacy_path, var_interest, mouse_name, session):
    """Just the lag the previous code selected, or None if it never ran this session.

    NOTE the old grid was [1, 10, 20, 30, 40], so the value need not be in the new
    doubling grid -- `nearest_fitted_lag` handles that.
    """
    r = legacy_result(legacy_path, var_interest, mouse_name, session)
    if r is None:
        return None
    return int(r[2][1])          # best_parameters = (num_states, lag, kappa)


def nearest_fitted_lag(d, lag):
    """The lag in THIS session's grid closest to `lag`, in log2 space.

    Needed because the old grid ([1, 10, 20, 30, 40]) and the new one (powers of 2) only
    share the value 1. Only lags that were actually fitted have stored parameters, so a
    legacy lag of 10 is shown at the nearest fitted lag (8), and the label says so.
    """
    grid = np.array(sorted(int(l) for l in d['all_fit_params']))
    return int(grid[np.argmin(np.abs(np.log2(grid) - np.log2(max(int(lag), 1))))])



def decode_at_lag(d, shortened_array, lag, num_states=2, method='prior', kappa=0.0):
    """Re-decode a session at ANY lag that was in its grid, from the stored parameters.

    No refitting: `all_fit_params[lag]` holds the cross-validated parameters for that
    grid cell, so this costs one Viterbi pass. It is what makes the lag-1 comparison
    cheap, and it also answers "would a longer lag have changed anything?" per session.

    Oriented the same way as the saved states (state 1 = higher-mean state), so the two
    sequences are comparable rather than differing by an arbitrary label swap.
    """
    lag = int(lag)
    if lag not in d['all_fit_params']:
        raise KeyError(f'lag {lag} not in this session grid: {sorted(d["all_fit_params"])}')
    fold = pick_fold(d['all_lls'][lag])
    if fold is None:
        raise ValueError(f'every fold is NaN at lag {lag}')
    ed = shortened_array.shape[1]
    my_inputs = compute_inputs(shortened_array, lag, ed)
    states, _ = decode_ar_states(shortened_array, my_inputs, lag, num_states, ed,
                                 d['all_fit_params'][lag], fold, method, kappa=kappa)
    return orient_states(states, shortened_array)



def runs_of(mask):
    """Contiguous True runs as [(start, stop), ...]."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def pick_window(states, fps, win_s=15, target_transitions=7):
    """Start frame of a window that is actually informative: balanced occupancy and a
    handful of transitions. A window picked at random is usually all one state and
    shows nothing.
    """
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


def paired_gains(d, alpha=0.05):
    """The selection record as a table: for each tested lag, the PAIRED per-fold gain
    against the lag that was adopted at the time, and its 95% CI.

    This is the quantity the rule tests. The across-fold SD of the absolute LL is a
    different and much larger number (13x on some sessions) because it includes
    between-fold differences that cancel in the pairing — which is exactly the bug
    that made `find_2_best_param` return the grid minimum.
    """
    rows = []
    for stp in d.get('selection_steps', []):
        lo = np.asarray(d['all_lls'][int(stp['lag_from'])], dtype=float)
        hi = np.asarray(d['all_lls'][int(stp['lag_to'])], dtype=float)
        diff = hi - lo
        ok = np.isfinite(diff)
        n = int(ok.sum())
        mu = diff[ok].mean() if n else np.nan
        se = diff[ok].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        ci = tdist.ppf(1 - alpha / 2, n - 1) * se if n > 1 else np.nan
        rows.append(dict(lag_from=int(stp['lag_from']), lag_to=int(stp['lag_to']),
                         gain=mu, se=se, ci95=ci, n_folds=n, adopted=bool(stp['pays'])))
    return pd.DataFrame(rows)


# ============================================================================
# PER-SESSION PANELS
# ============================================================================

def plot_states(d, signal, fps, win_s=15, ax=None, start=None, start_s=None,
                states=None, compare_states=None, labels=('adopted', 'lag 1')):
    """The segmentation on the trace. Pale band + ribbon = the high state.

    Window placement, in order of precedence:
      start_s   start time in SECONDS into the session (start_s=600 -> 10 min in)
      start     start FRAME
      neither   `pick_window` picks an informative one automatically

    Pass `compare_states` (e.g. from `decode_at_lag(d, sa, 1)`) to overlay a second
    segmentation: both get a ribbon, and the frames where they DISAGREE are shaded
    orange. That turns "the lag barely matters" from a claim into something visible --
    the disagreements sit at state boundaries, not in the interior of bouts.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.6))
    states = np.asarray(d['most_likely_states'] if states is None else states)
    comparing = compare_states is not None
    win = int(win_s * fps)
    if start_s is not None:
        st = int(round(float(start_s) * fps))
    elif start is not None:
        st = int(start)
    elif comparing:
        # centre on the biggest disagreement -- that is the point of the comparison
        diff = np.asarray(compare_states) != states
        runs = runs_of(diff)
        if runs:
            lo_, hi_ = max(runs, key=lambda t: t[1] - t[0])
            st = (lo_ + hi_) // 2 - win // 2
        else:
            st = pick_window(states, fps, win_s)
    else:
        st = pick_window(states, fps, win_s)
    # keep the window inside the session however it was specified
    st = int(np.clip(st, 0, max(len(states) - win, 0)))

    t = np.arange(win) / fps
    sig, seg = signal[st:st + win], states[st:st + win]
    # two ribbons need more headroom under the trace than one
    pad = 0.75 if comparing else 0.35
    lo, hi = np.nanmin(sig) - pad, np.nanmax(sig) + 0.5

    _style(ax, grid=False)
    for p, q in runs_of(seg == 1):
        q = min(q, win - 1)
        ax.axvspan(t[p], t[q], color=BAND, lw=0, zorder=0)
    if comparing:
        cmp_seg = np.asarray(compare_states)[st:st + win]
        for p, q in runs_of(cmp_seg != seg):
            ax.axvspan(t[p], t[min(q, win - 1)], color=SERIES_2, alpha=0.40, lw=0, zorder=1)
        for k, (row, lab) in enumerate(zip((seg, cmp_seg), labels)):
            y = lo + 0.05 + k * 0.22
            for p, q in runs_of(row == 1):
                ax.fill_between([t[p], t[min(q, win - 1)]], y, y + 0.15,
                                color=SERIES_1, lw=0, zorder=4)
            ax.annotate(lab, xy=(-0.006, y + 0.075), xycoords=('axes fraction', 'data'),
                        fontsize=7.5, color=INK_2, ha='right', va='center')
    else:
        for p, q in runs_of(seg == 1):
            ax.fill_between([t[p], t[min(q, win - 1)]], lo + 0.02, lo + 0.16,
                            color=SERIES_1, lw=0, zorder=3)
    ax.plot(t, sig, color=INK, lw=0.9, zorder=2)

    ax.set_xlim(0, t[-1])
    ax.set_ylim(lo, hi)
    ax.set_xlabel('time (s)', fontsize=8.5, color=INK_2)
    ax.set_ylabel('signal', fontsize=8.5, color=INK_2)
    a = d['assessment']
    if comparing:
        agree = float(np.mean(np.asarray(compare_states) == states))
        title = (f"{labels[0]} vs {labels[1]} · {agree:.2%} of frames agree · "
                 f"dwell {a['median_dwell_ms']:.0f} ms")
        note = (f'orange = disagreement · window centred on the largest one · '
                f'start_s={st / fps:.0f} of {len(states) / fps:.0f}')
    else:
        title = (f"segmentation · dwell {a['median_dwell_ms']:.0f} ms · "
                 f"{a['n_segments']:,} segments · occupancy {a['occupancy_state1']:.2f}")
        note = (f'start_s={st / fps:.0f}  ({st / fps / 60:.1f} min of '
                f'{len(states) / fps / 60:.0f}) · shaded = high state')
    ax.set_title(title, fontsize=9, color=INK, loc='left', pad=17)
    ax.annotate(note, xy=(0, 1.0), xycoords='axes fraction', xytext=(0, 2),
                textcoords='offset points', fontsize=7.5, color=INK_MUTED)
    return ax



def plot_lag_profile(d, ax=None):
    """Absolute mean held-out LL per lag. CONTEXT ONLY — this is what an argmax would
    follow, and on several sessions it rises smoothly to a lag the paired test rejects.
    One series, so no legend; the adopted lag is ringed and the cap marked.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    _style(ax)
    prof = d.get('lag_profile') or {}
    if not prof:
        ax.text(0.5, 0.5, 'no lag grid\n(Poisson model)', ha='center', va='center',
                fontsize=9, color=INK_MUTED, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    lags = sorted(prof)
    y = [prof[l] for l in lags]
    ax.plot(lags, y, color=INK_MUTED, lw=1.4, zorder=2)
    ax.scatter(lags, y, s=18, color=INK_MUTED, zorder=3)
    ax.scatter([d['best_lag']], [prof[d['best_lag']]], s=95, facecolor='none',
               edgecolor=SERIES_2, lw=2, zorder=4)
    # dashed here is deliberate and semantic: a threshold, not a gridline
    ax.axvline(d['cap'], color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.annotate(f"cap {d['cap']}", xy=(d['cap'], 0.02), xycoords=('data', 'axes fraction'),
                fontsize=7.5, color=INK_MUTED, ha='right', rotation=90, va='bottom')
    ax.set_xscale('log', base=2)
    ax.set_xticks(lags)
    ax.set_xticklabels([str(l) for l in lags], fontsize=7)
    ax.set_xlabel('lag (frames)', fontsize=8.5, color=INK_2)
    ax.set_ylabel('mean held-out LL / frame', fontsize=8.5, color=INK_2)
    ax.set_title(f"profile · adopted lag {d['best_lag']}"
                 f"{' (AT CAP)' if d['at_cap'] else ''}",
                 fontsize=9, color=INK, loc='left', pad=6)
    return ax


def plot_paired_gains(d, ax=None, alpha=0.05):
    """The decision itself: paired gain vs the adopted lag, with 95% CI. A lag is
    adopted exactly when its CI clears zero. Two categories, so a legend is present.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    _style(ax)
    g = paired_gains(d, alpha=alpha)
    if not len(g):
        ax.text(0.5, 0.5, 'no lag grid\n(Poisson model)', ha='center', va='center',
                fontsize=9, color=INK_MUTED, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    ax.axhline(0, color=INK, lw=0.9, zorder=1)
    ax.errorbar(g.lag_to, g.gain, yerr=g.ci95, fmt='none', ecolor=INK_2,
                elinewidth=1.1, capsize=3, zorder=2)
    ok, no = g[g.adopted], g[~g.adopted]
    ax.scatter(ok.lag_to, ok.gain, s=46, color=SERIES_1, zorder=3, label='adopted')
    # hollow with a surface-coloured face = a 2px surface ring where markers overlap
    ax.scatter(no.lag_to, no.gain, s=54, facecolor=SURFACE, edgecolor=SERIES_2, lw=1.6,
               zorder=3, label='rejected')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(d['lag_profile']))
    ax.set_xticklabels([str(l) for l in sorted(d['lag_profile'])], fontsize=7)
    ax.set_xlabel('lag being tested', fontsize=8.5, color=INK_2)
    ax.set_ylabel('paired gain vs adopted lag', fontsize=8.5, color=INK_2)
    ax.set_title('the tested quantity · 95% CI', fontsize=9, color=INK, loc='left', pad=6)
    ax.legend(frameon=False, fontsize=7.5, loc='best', labelcolor=INK_2)
    return ax


def plot_session(save_path, data_path, var_interest, mouse_name=None, session=None,
                 zsc=True, num_train_batches=5, fps=60., win_s=15, sticky=False,
                 alpha=0.05, eid=None, start=None, start_s=None, compare_lag=None,
                 legacy_path=None, panels=('states', 'profile'), figsize=None):
    """Inspect one session: the segmentation, and by default the lag profile beside it.

    Identify the session however is convenient:
        plot_session(..., eid='63f3dbc1')            # eid prefix is enough
        plot_session(..., mouse_name='PL017', session='1b61b7f2-...')

    Move the window with `start_s` (seconds into the session) to look somewhere else:
        plot_session(..., eid='63f3dbc1', start_s=1200)      # 20 min in

    Compare against another lag. Both segmentations get a ribbon and the disagreeing
    frames are shaded; with no `start_s` the window centres on the biggest disagreement,
    which is the honest place to look.

        plot_session(..., eid='63f3dbc1', compare_lag=1)          # the no-filter baseline
        plot_session(..., eid='63f3dbc1', compare_lag='legacy',   # what 4.1/4.2 chose
                     legacy_path=.../most_likely_states/5_prior_em_zsc_True/)

    `compare_lag='legacy'` reads the lag out of the OLD pipeline's own output file and
    re-decodes at it on the CURRENT data. It deliberately does not reuse the old state
    array: the NaN-handling fix changed which rows survive, so for roughly half the cohort
    the old sequence is a different length and would silently misalign. Re-decoding keeps
    the comparison frame-exact and isolates the thing being compared -- the selection
    rule -- from the data fix. (Use `legacy_result` directly if you do want the raw old
    output, and check the lengths yourself.)

    `panels` picks what to draw — 'states', 'profile', 'gains' — so the paired-gain panel
    is still available (`panels=('states', 'profile', 'gains')`) without being in the way
    by default.

    Returns (fig, result_dict) so the caller can keep inspecting the same pickle.
    """
    mouse_name, session = resolve_session(save_path, var_interest, eid=eid,
                                          mouse_name=mouse_name, session=session,
                                          sticky=sticky)
    d = load_result(save_path, var_interest, mouse_name, session, sticky=sticky)
    states = np.asarray(d['most_likely_states'])
    shortened_array = load_prepared(data_path, session, mouse_name, var_interest, zsc,
                                    num_train_batches)
    signal = shortened_array[:len(states), 0]

    # the comparison costs one Viterbi pass, because the parameters for every grid cell
    # were saved -- no refitting
    compare_states, compare_label, requested = None, None, compare_lag
    if isinstance(compare_lag, str):
        if compare_lag != 'legacy':
            raise ValueError(f"compare_lag must be a lag or 'legacy', got {compare_lag!r}")
        if legacy_path is None:
            raise ValueError("compare_lag='legacy' needs legacy_path=<old most_likely_states dir>")
        requested = legacy_lag(legacy_path, var_interest, mouse_name, session)
        if requested is None:
            print(f'{mouse_name} {session[:8]}: not in the previous run, nothing to compare')
            compare_lag = None
        else:
            compare_lag = nearest_fitted_lag(d, requested)
            compare_label = (f'old lag {requested}' if compare_lag == requested
                             else f'old {requested}->{compare_lag}')   # old grid was not powers of 2
            if int(compare_lag) == int(d['best_lag']):
                # the two pipelines landed on the same lag -- say so, rather than
                # silently drawing a single ribbon as though the comparison failed
                same = ('the same lag' if compare_lag == requested
                        else f'the same lag after snapping ({requested} -> {compare_lag})')
                print(f"{mouse_name} {session[:8]}: old and new chose {same}, nothing to overlay")
    if compare_lag is not None and int(compare_lag) != int(d['best_lag']):
        compare_states = decode_at_lag(d, shortened_array, compare_lag,
                                       num_states=d['config']['num_states'],
                                       method=d['config']['method'],
                                       kappa=d['config']['kappa'])[:len(states)]

    panels = tuple(panels)
    widths = {'states': 2.05, 'profile': 1.0, 'gains': 1.0}
    ratios = [widths[p] for p in panels]
    if figsize is None:
        figsize = (3.05 + 5.2 * sum(ratios) / 3.05, 3.1)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1, len(panels), width_ratios=ratios, wspace=0.30)
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        if p == 'states':
            plot_states(d, signal, fps, win_s=win_s, ax=ax, start=start, start_s=start_s,
                        compare_states=compare_states,
                        labels=(f"lag {d['best_lag']}",
                                compare_label or f'lag {compare_lag}'))
        elif p == 'profile':
            plot_lag_profile(d, ax=ax)
        elif p == 'gains':
            plot_paired_gains(d, ax=ax, alpha=alpha)
        else:
            raise ValueError(f'unknown panel {p!r}; use states / profile / gains')

    a = d['assessment']
    flags = [k for k in ('collapsed', 'degenerate_occupancy', 'flickering') if a.get(k)]
    head = f"{mouse_name}   {session[:8]}   ·   {var_interest[0]}"
    tail = ('OK' if a['fit_ok'] else 'FLAGGED: ' + ', '.join(flags or ['fold NaNs']))
    fig.suptitle(f"{head}   ·   {tail}", fontsize=11,
                 color=INK if a['fit_ok'] else SERIES_2, x=0.006, ha='left', y=0.995)
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.055, right=0.99)
    if compare_states is not None:
        # attach the comparison so the caller can report it without decoding again
        d = dict(d, compare_lag=int(compare_lag), compare_states=compare_states,
                 compare_requested_lag=requested, compare_label=compare_label,
                 compare_agreement=float(np.mean(compare_states == states)))
    return fig, d


# ============================================================================
# COHORT OVERVIEW
# ============================================================================

def plot_cohort(assessments, fps=60.0, target_dwell_ms=450., bits_screen=0.35,
                figsize=(13.6, 6.4)):
    """The whole set at once. Takes the assessment table (the CSV, or
    `hmm_dynamic_functions.assessments_from_pickles`).

    Each panel is one series, so one hue; the reference lines are thresholds, which is
    what earns them a dash.
    """
    df = assessments.copy()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax_lag, ax_dwell, ax_bits, ax_flag = axes.ravel()

    # --- A: selected lag, split by whether it hit the cap -------------------------
    _style(ax_lag)
    if 'best_lag' in df and df.best_lag.notna().any():
        at_cap = df.at_cap.fillna(False) if 'at_cap' in df else pd.Series(False, df.index)
        lags = sorted(df.best_lag.dropna().unique())
        x = np.arange(len(lags))
        n_free = [int(((df.best_lag == l) & ~at_cap).sum()) for l in lags]
        n_cap = [int(((df.best_lag == l) & at_cap).sum()) for l in lags]
        # edgecolor = the surface, i.e. a gap between segments rather than a border
        ax_lag.bar(x, n_free, color=SERIES_1, width=0.68, label='below cap',
                   edgecolor=SURFACE, lw=1.2, zorder=3)
        ax_lag.bar(x, n_cap, bottom=n_free, color=SERIES_2, width=0.68, label='at cap',
                   edgecolor=SURFACE, lw=1.2, zorder=3)
        for xi, tot in zip(x, np.add(n_free, n_cap)):
            if tot:
                ax_lag.annotate(f'{tot}', xy=(xi, tot), xytext=(0, 3),
                                textcoords='offset points', ha='center',
                                fontsize=7.5, color=INK_2)
        ax_lag.set_xticks(x)
        ax_lag.set_xticklabels([f'{int(l)}' for l in lags], fontsize=8)
        ax_lag.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    ax_lag.set_xlabel('selected lag (frames)', fontsize=8.5, color=INK_2)
    ax_lag.set_ylabel('sessions', fontsize=8.5, color=INK_2)
    ax_lag.set_title('A · selected lag', fontsize=10, color=INK, loc='left', pad=16)
    ax_lag.annotate('at cap = the grid, not the data, stopped the search',
                    xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # --- B: dwell -----------------------------------------------------------------
    _style(ax_dwell)
    dw = df.median_dwell_ms.dropna()
    if len(dw):
        ax_dwell.hist(dw, bins=30, color=SERIES_1, zorder=3)
        ax_dwell.axvline(target_dwell_ms, color=SERIES_2, lw=1.2, ls=(0, (4, 3)), zorder=4)
        ax_dwell.annotate(f'model-free\nchangepoints\n{target_dwell_ms:.0f} ms',
                          xy=(target_dwell_ms, 0.97), xycoords=('data', 'axes fraction'),
                          xytext=(4, 0), textcoords='offset points', fontsize=7.5,
                          color=SERIES_2, va='top')
        ax_dwell.axvline(1000 / fps * 10, color=INK_MUTED, lw=1, ls=(0, (2, 2)), zorder=4)
        ax_dwell.annotate('flicker screen\n(10 frames)',
                          xy=(1000 / fps * 10, 0.97), xycoords=('data', 'axes fraction'),
                          xytext=(-4, 0), textcoords='offset points', fontsize=7.5,
                          color=INK_MUTED, va='top', ha='right')
    ax_dwell.set_xlabel('median dwell (ms)', fontsize=8.5, color=INK_2)
    ax_dwell.set_ylabel('sessions', fontsize=8.5, color=INK_2)
    ax_dwell.set_title('B · syllable duration', fontsize=10, color=INK, loc='left', pad=16)
    ax_dwell.annotate('should sit near the changepoint duration, not at the screen',
                      xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # --- C: bits ------------------------------------------------------------------
    _style(ax_bits)
    bits = df.bits_LL.dropna()
    if len(bits):
        ax_bits.hist(bits, bins=30, color=SERIES_1, zorder=3)
        ax_bits.axvline(bits_screen, color=SERIES_2, lw=1.2, ls=(0, (4, 3)), zorder=4)
        ax_bits.annotate(f'screen {bits_screen}', xy=(bits_screen, 0.97),
                         xycoords=('data', 'axes fraction'), xytext=(4, 0),
                         textcoords='offset points', fontsize=7.5, color=SERIES_2, va='top')
    ax_bits.set_xlabel('bits_LL (held-out, vs the prior-sampled baseline)',
                       fontsize=8.5, color=INK_2)
    ax_bits.set_ylabel('sessions', fontsize=8.5, color=INK_2)
    ax_bits.set_title('C · how much the model learned', fontsize=10, color=INK,
                      loc='left', pad=16)
    ax_bits.annotate('low bits does NOT imply a bad segmentation — screens are independent',
                     xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # --- D: flags -----------------------------------------------------------------
    _style(ax_flag)
    names = ['collapsed', 'degenerate_occupancy', 'flickering']
    counts = [int(df[c].fillna(False).sum()) for c in names if c in df]
    labels = [c.replace('_', ' ') for c in names if c in df]
    if 'n_folds_failed' in df:
        labels.append('fold NaNs'); counts.append(int((df.n_folds_failed.fillna(0) > 0).sum()))
    if 'error' in df:
        labels.append('errored'); counts.append(int((df.error.fillna('') != '').sum()))
    if 'fit_ok' in df:
        labels.append('NOT fit_ok'); counts.append(int((~df.fit_ok.fillna(False)).sum()))
    y = np.arange(len(labels))
    ax_flag.barh(y, counts, color=SERIES_1, height=0.6, zorder=3)
    for yi, c in zip(y, counts):
        ax_flag.annotate(f'{c}', xy=(c, yi), xytext=(4, 0), textcoords='offset points',
                         va='center', fontsize=8, color=INK_2)
    ax_flag.set_yticks(y)
    ax_flag.set_yticklabels(labels, fontsize=8)
    ax_flag.invert_yaxis()
    ax_flag.set_xlabel(f'sessions (of {len(df)})', fontsize=8.5, color=INK_2)
    ax_flag.set_xlim(0, max(max(counts) * 1.18, 1))
    ax_flag.set_title('D · failure screens', fontsize=10, color=INK, loc='left', pad=16)
    ax_flag.annotate('three independent screens — a fit can pass two and fail the third',
                     xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.065, right=0.985,
                        wspace=0.24, hspace=0.42)
    return fig


def plot_fit_statistics(assessments, fps=60.0, target_dwell_ms=450., bits_screen=0.35,
                        min_dwell_frames=10, figsize=(14.2, 8.2)):
    """Relationships in the assessment table, as opposed to `plot_cohort`'s distributions.

    Reads the CSV `run_all` writes (or `assessments_from_pickles`). Each panel exists to
    answer one question that a distribution cannot:

      A  does the CAP bind, or does the data choose the lag?      tau vs selected lag
      B  is the segmentation self-consistent?                     dwell vs n_segments
      C  are the failure screens really independent?              bits vs dwell
      D  is either state vestigial?                               occupancy
      E  is the lag a property of the MOUSE or of the session?    per-mouse spread
      F  is anything just a function of recording length?         n_frames vs bits

    E is the one worth arguing about for an individuality paper: if the selected lag
    clusters by mouse, it is a trait and picking it per session is right; if it scatters
    within mouse as much as between, it is fold noise and the cohort would be better
    served by one pooled value.
    """
    df = assessments.copy()
    has_lag = 'best_lag' in df and df.best_lag.notna().any() and df.best_lag.max() > 0
    at_cap = (df.at_cap.fillna(False) if 'at_cap' in df
              else pd.Series(False, index=df.index))
    ok = df.fit_ok.fillna(False) if 'fit_ok' in df else pd.Series(True, index=df.index)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    axA, axB, axC, axD, axE, axF = axes.ravel()

    # --- A: tau vs selected lag ---------------------------------------------------
    _style(axA)
    if has_lag and 'tau' in df:
        sub = df[df.tau.notna() & df.best_lag.notna()]
        lim_hi = max(float(sub.tau.max()) * 1.3, float(sub.best_lag.max()) * 1.3, 4)
        axA.plot([1, lim_hi], [1, lim_hi], color=INK_MUTED, lw=0.9, zorder=1)
        axA.annotate('lag = τ', xy=(lim_hi, lim_hi), xytext=(-4, -10),
                     textcoords='offset points', fontsize=7.5, color=INK_MUTED, ha='right')
        free, capd = sub[~at_cap.reindex(sub.index).fillna(False)], sub[at_cap.reindex(sub.index).fillna(False)]
        axA.scatter(free.tau, free.best_lag, s=30, color=SERIES_1, lw=0, alpha=0.85,
                    zorder=3, label='below cap')
        axA.scatter(capd.tau, capd.best_lag, s=42, facecolor=SURFACE, edgecolor=SERIES_2,
                    lw=1.5, zorder=4, label='at cap')
        axA.set_xscale('log', base=2); axA.set_yscale('log', base=2)
        axA.legend(frameon=False, fontsize=7.5, loc='upper left', labelcolor=INK_2)
    axA.set_xlabel('τ, ACF 1/e crossing (frames)', fontsize=8.5, color=INK_2)
    axA.set_ylabel('selected lag (frames)', fontsize=8.5, color=INK_2)
    axA.set_title('A · does the cap bind?', fontsize=10, color=INK, loc='left', pad=16)
    axA.annotate('points on the line were stopped by the cap, not the likelihood',
                 xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # --- B: dwell vs n_segments ---------------------------------------------------
    _style(axB)
    if {'median_dwell_ms', 'n_segments'} <= set(df):
        axB.scatter(df.median_dwell_ms[ok], df.n_segments[ok], s=30, color=SERIES_1,
                    lw=0, alpha=0.85, zorder=3, label='fit_ok')
        axB.scatter(df.median_dwell_ms[~ok], df.n_segments[~ok], s=42, facecolor=SURFACE,
                    edgecolor=SERIES_2, lw=1.5, zorder=4, label='flagged')
        axB.set_xscale('log'); axB.set_yscale('log')
        if (~ok).any():
            axB.legend(frameon=False, fontsize=7.5, loc='upper right', labelcolor=INK_2)
    axB.set_xlabel('median dwell (ms)', fontsize=8.5, color=INK_2)
    axB.set_ylabel('segments', fontsize=8.5, color=INK_2)
    axB.set_title('B · internal consistency', fontsize=10, color=INK, loc='left', pad=16)
    axB.annotate('a clean inverse relation; anything off it is a degenerate fit',
                 xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # --- C: bits vs dwell — the independence claim --------------------------------
    _style(axC)
    if {'bits_LL', 'median_dwell_ms'} <= set(df):
        axC.scatter(df.bits_LL, df.median_dwell_ms, s=30, color=SERIES_1, lw=0,
                    alpha=0.85, zorder=3)
        axC.axvline(bits_screen, color=SERIES_2, lw=1.1, ls=(0, (4, 3)), zorder=2)
        axC.axhline(min_dwell_frames * 1000 / fps, color=SERIES_2, lw=1.1,
                    ls=(0, (4, 3)), zorder=2)
        axC.set_yscale('log')
        axC.annotate('flicker screen', xy=(0.02, min_dwell_frames * 1000 / fps),
                     xycoords=('axes fraction', 'data'), xytext=(0, 3),
                     textcoords='offset points', fontsize=7.5, color=SERIES_2)
        axC.annotate('bits screen', xy=(bits_screen, 0.98),
                     xycoords=('data', 'axes fraction'), xytext=(3, 0),
                     textcoords='offset points', fontsize=7.5, color=SERIES_2, va='top')
        r = df[['bits_LL', 'median_dwell_ms']].dropna()
        if len(r) > 2:
            rho = r.bits_LL.corr(r.median_dwell_ms, method='spearman')
            axC.annotate(f'Spearman ρ = {rho:+.2f}', xy=(0.97, 0.04),
                         xycoords='axes fraction', ha='right', fontsize=8, color=INK_2)
    axC.set_xlabel('bits_LL', fontsize=8.5, color=INK_2)
    axC.set_ylabel('median dwell (ms)', fontsize=8.5, color=INK_2)
    axC.set_title('C · are the screens independent?', fontsize=10, color=INK,
                  loc='left', pad=16)
    axC.annotate('a weak ρ is the point — likelihood does not predict a degenerate fit',
                 xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # --- D: occupancy -------------------------------------------------------------
    _style(axD)
    if 'occupancy_state1' in df:
        occ = df.occupancy_state1.dropna()
        axD.hist(occ, bins=np.linspace(0, 1, 41), color=SERIES_1, zorder=3)
        for edge in (0.02, 0.98):
            axD.axvline(edge, color=SERIES_2, lw=1.1, ls=(0, (4, 3)), zorder=4)
        axD.set_xlim(0, 1)
        axD.annotate('degenerate\n< 2% or > 98%', xy=(0.5, 0.97), xycoords='axes fraction',
                     ha='center', va='top', fontsize=7.5, color=SERIES_2)
    axD.set_xlabel('fraction of frames in the high state', fontsize=8.5, color=INK_2)
    axD.set_ylabel('sessions', fontsize=8.5, color=INK_2)
    axD.set_title('D · is either state vestigial?', fontsize=10, color=INK, loc='left', pad=16)
    axD.annotate('both states should carry real occupancy',
                 xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # --- E: is the lag a mouse trait? ---------------------------------------------
    _style(axE)
    if has_lag and 'mouse' in df:
        g = df[df.best_lag.notna()].groupby('mouse').best_lag
        multi = g.count()[g.count() >= 2].index
        sub = df[df.mouse.isin(multi) & df.best_lag.notna()]
        if len(sub):
            order = sub.groupby('mouse').best_lag.median().sort_values().index
            pos = {m: i for i, m in enumerate(order)}
            # deterministic jitter (not random) so a session lands in the same spot
            # on every rerun -- Math.random-style jitter makes figures uncomparable
            jit = ((np.arange(len(sub)) % 5) - 2) * 0.11
            axE.scatter([pos[m] + j for m, j in zip(sub.mouse, jit)], sub.best_lag,
                        s=26, color=SERIES_1, lw=0, alpha=0.8, zorder=3)
            med = sub.groupby('mouse').best_lag.median().reindex(order)
            axE.plot(range(len(order)), med.values, color=SERIES_2, lw=1.2, zorder=4)
            axE.set_yscale('log', base=2)
            # past ~30 mice the names collide into an unreadable smear; the identity of
            # each mouse is not the point of this panel, the spread is
            if len(order) <= 30:
                axE.set_xticks(range(len(order)))
                axE.set_xticklabels(order, rotation=90, fontsize=6)
            else:
                axE.set_xticks([])
                axE.set_xlabel(f'{len(order)} mice, ordered by median lag',
                               fontsize=8.5, color=INK_2)
            # within- vs between-mouse spread: the actual answer to the question
            within = sub.groupby('mouse').best_lag.apply(
                lambda s: np.log2(s).std(ddof=1)).mean()
            between = np.log2(med.dropna()).std(ddof=1)
            axE.annotate(f'SD of log2(lag):  within mouse {within:.2f}   '
                         f'between mice {between:.2f}',
                         xy=(0.5, 0.965), xycoords='axes fraction', ha='center',
                         fontsize=7.5, color=INK_2)
        else:
            axE.text(0.5, 0.5, 'no mouse has ≥2 finished sessions yet', ha='center',
                     va='center', fontsize=9, color=INK_MUTED, transform=axE.transAxes)
    axE.set_ylabel('selected lag (frames)', fontsize=8.5, color=INK_2)
    axE.set_title('E · trait or session noise?', fontsize=10, color=INK, loc='left', pad=16)
    axE.annotate('mice with ≥2 sessions, ordered by median (orange)',
                 xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    # --- F: recording length ------------------------------------------------------
    _style(axF)
    if {'n_frames', 'bits_LL'} <= set(df):
        mins = df.n_frames / fps / 60
        axF.scatter(mins, df.bits_LL, s=30, color=SERIES_1, lw=0, alpha=0.85, zorder=3)
        r = pd.DataFrame({'m': mins, 'b': df.bits_LL}).dropna()
        if len(r) > 2:
            rho = r.m.corr(r.b, method='spearman')
            axF.annotate(f'Spearman ρ = {rho:+.2f}', xy=(0.97, 0.04),
                         xycoords='axes fraction', ha='right', fontsize=8, color=INK_2)
    axF.set_xlabel('usable recording (min)', fontsize=8.5, color=INK_2)
    axF.set_ylabel('bits_LL', fontsize=8.5, color=INK_2)
    axF.set_title('F · a length artefact?', fontsize=10, color=INK, loc='left', pad=16)
    axF.annotate('bits is per frame, so it should NOT track duration',
                 xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

    fig.subplots_adjust(top=0.92, bottom=0.13, left=0.055, right=0.99,
                        wspace=0.28, hspace=0.58)
    return fig
