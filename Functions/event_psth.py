"""
EVENT PSTHs ON TWO WINDOWS: stimulus-aligned and feedback-aligned, split axis
==============================================================================
Shared by neuromodulators/4_ddm_hmm_photometry.ipynb (photometry) and 5_ddm_hmm_syllable_psth.ipynb
(syllables), so both are cut on the same time grid with the same per-trial masks and drawn in the same
layout.

The photometry export (neuromodulators/pose_qc_hmm_k4) holds -1..+1 s around stimOnTrigger_times and
feedback_times at 30 Hz; the syllable states files are binned at 30 Hz on the same session clock
(goCueTrigger_times agree to the sample). TIME is that grid.

Per-trial masks: the stimulus window drops samples after the trial's feedback and before its start
(intervals_0); the feedback window drops samples before stimulus onset and after the trial's end
(intervals_1). No sample is shown twice and none belongs to a neighbouring trial.

    from Functions import event_psth as ep
    STIM, FB, meta = ep.photometry_windows(rec, trials)      # trials: dh.load_all go trials
    SYL_S, SYL_F = ep.syllable_windows(meta, states_dir)     # syllable codes on the same rows
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

TIME = np.arange(-30, 30) / 30          # s, the photometry export's grid
EVENTS = {'stim': 'stimOnTrigger_times', 'fb': 'feedback_times'}
EPOCH_SHADE = {'Pre-quiescence': '0.95', 'Quiescence': '0.88', 'Choice': '0.95', 'ITI': '0.88'}


def window_masks(t):
    """Boolean (n_trials, len(TIME)) masks of samples to DROP, for the stim and fb windows."""
    on, fb = t['stimOnTrigger_times'].to_numpy()[:, None], t['feedback_times'].to_numpy()[:, None]
    start, end = t['intervals_0'].to_numpy()[:, None], t['intervals_1'].to_numpy()[:, None]
    drop_s = (TIME[None] >= fb - on) | (TIME[None] < start - on)
    drop_f = (TIME[None] < on - fb) | (TIME[None] > end - fb)
    return drop_s, drop_f


def photometry_windows(rec, trials, baseline=False):
    """Masked photometry windows for every trial of `trials` that has a response row.

    rec: recordings.csv rows to use (one fiber per row); trials: per-trial table with eid, trial and the
    IBL time columns. Returns STIM, FB (float32, n x len(TIME)) and meta (rec columns + the trials row),
    row-aligned. baseline=True subtracts each trial's mean over stimulus -0.1..0 s from both windows."""
    from Functions import ddm_hmm as dh
    L = dh._loaders()
    S, F, keys = [], [], []
    for _, r in rec.iterrows():
        resp = L.load_responses(r['eid'])[r['brain_region']]
        assert np.allclose(resp['time'].values, TIME)
        t = trials[trials['eid'] == r['eid']].set_index('trial')
        common = np.intersect1d(resp['trial'].values, t.index.values)
        t = t.loc[common]
        s = resp.sel(event=EVENTS['stim'], trial=common).values.copy()
        f = resp.sel(event=EVENTS['fb'], trial=common).values.copy()
        if baseline:
            b = s[:, (TIME >= -0.1) & (TIME <= 0)].mean(axis=1)
            s -= b[:, None]; f -= b[:, None]
        drop_s, drop_f = window_masks(t)
        s[drop_s] = np.nan; f[drop_f] = np.nan
        S.append(s.astype(np.float32)); F.append(f.astype(np.float32))
        keys.append(t.reset_index().assign(target_NM=r['target_NM'], region=r['brain_region'], subject=r['subject']))
    return np.vstack(S), np.vstack(F), pd.concat(keys, ignore_index=True)


def syllable_windows(meta, states_dir, column='most_likely_states'):
    """Syllable codes (code = paw + n_paw * whisk + 2 * n_paw * lick) on the TIME grid around each trial's
    stimulus and feedback, rows aligned with `meta`, masked like the photometry. NaN where the states file
    has no state (tracking dropout) or the sample is masked."""
    S = np.full((len(meta), len(TIME)), np.nan, dtype=np.float32)
    F = np.full_like(S, np.nan)
    for eid, idx in meta.groupby('eid').indices.items():
        files = glob.glob(str(Path(states_dir) / f'*_states_file_{eid}_*'))
        if not files:
            continue
        st = pd.read_parquet(files[0], columns=['Bin', column]).sort_values('Bin')
        bins, codes = st['Bin'].to_numpy(), st[column].to_numpy()
        dt = np.median(np.diff(bins))
        t = meta.iloc[idx]
        for out, ev in [(S, EVENTS['stim']), (F, EVENTS['fb'])]:
            when = t[ev].to_numpy()[:, None] + TIME[None]
            j = np.clip(np.searchsorted(bins, when), 1, len(bins) - 1)
            j = np.where(np.abs(bins[j - 1] - when) < np.abs(bins[j] - when), j - 1, j)
            vals = codes[j].astype(np.float32)
            vals[np.abs(bins[j] - when) > dt / 2] = np.nan
            out[idx] = vals
        drop_s, drop_f = window_masks(t)
        S[idx] = np.where(drop_s, np.nan, S[idx]); F[idx] = np.where(drop_f, np.nan, F[idx])
    return S, F


def decode(codes, n_paw=8):
    """paw, whisk, lick arrays from syllable codes (NaN stays NaN)."""
    paw = codes % n_paw
    whisk = (codes // n_paw) % 2
    lick = codes // (2 * n_paw)
    return paw, whisk, lick


def split_panels(fig, spec, title=None, epochs=True, qp=0.5, xlabels=True):
    """One row: stimulus-aligned | feedback-aligned, equal widths, break marks between. Returns (a, b)."""
    gs = spec.subgridspec(1, 2, wspace=0.12)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax, lab in [(a, 'from stimulus (s)'), (b, 'from feedback (s)')]:
        ax.set_xlim(-1, 1); ax.axvline(0, color='0.6', lw=0.6, zorder=1)
        if xlabels:
            ax.set_xlabel(lab, fontsize=6)
    a.set_xticks([-1, -0.5, 0, 0.5]); b.set_xticks([-0.5, 0, 0.5, 1])
    if not xlabels:
        a.set_xticklabels([]); b.set_xticklabels([])
    if epochs:
        for ax, bands in [(a, [('Pre-quiescence', -1, -qp), ('Quiescence', -qp, 0), ('Choice', 0, 1)]),
                          (b, [('Choice', -1, 0), ('ITI', 0, 1)])]:
            for name, x0, x1 in bands:
                ax.axvspan(x0, x1, ymin=0.94, ymax=1, color=EPOCH_SHADE[name], lw=0, zorder=0)
                short = {'Pre-quiescence': 'Pre-Q', 'Quiescence': 'Q'}.get(name, name)
                ax.text((x0 + x1) / 2, 0.97, short, transform=ax.get_xaxis_transform(),
                        ha='center', va='center', fontsize=4.5, color='0.35')
    b.spines['left'].set_visible(False); b.tick_params(axis='y', left=False, labelleft=False)
    for ax, x in [(a, 1), (b, 0)]:
        ax.plot([x, x], [-0.02, 0.02], transform=ax.transAxes, color='k', lw=0, marker=[(-1, -1.5), (1, 1.5)],
                ms=5, mew=0.8, clip_on=False)
    if title:
        a.set_title(title, fontsize=7, loc='left')
    return a, b


def share_ylim(a, b, pad=0.12, lim=None):
    lo = min(ax.get_ylim()[0] for ax in (a, b)) if lim is None else lim[0]
    hi = max(ax.get_ylim()[1] for ax in (a, b)) if lim is None else lim[1]
    for ax in (a, b):
        ax.set_ylim(lo, hi + pad * (hi - lo))
