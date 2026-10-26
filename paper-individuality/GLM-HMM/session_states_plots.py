"""
Per-session view of the BWM GLM-HMM: the state posterior on top, the behaviour it was fitted to underneath.
Functions only; the figures are made in session_states.ipynb.

Data: merged_behavioral_and_states.pqt (engaged.py's per-mouse K=2 fits, one row per trial in trial order).
It has no choice column, so choice is rebuilt from stimulus side and reward. This is exact, also at 0%
contrast, except on no-go trials (~0.4%), which come out as "chose the non-stimulus side".

Which state is engaged is decided per mouse: the state whose posterior-weighted accuracy on easy trials
(|contrast| >= EASY) is highest. For K=2 this is state1 in the paper's convention; it is checked here, not assumed.
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paper_style as ps

HERE = pathlib.Path(__file__).resolve().parent
STATES_FILE = HERE / 'merged_behavioral_and_states.pqt'
EASY = 0.25          # |contrast| counted as easy
WIN_ALL = 11         # rolling accuracy window, trials
WIN_EASY = 7         # rolling accuracy window, easy trials


def load_states(path=STATES_FILE):
    """Trials with trial (0-based, per session), choice_right, correct, contrast (|signed|), easy, the p_state*
    columns, p_engaged and map_state ('engaged' / 'disengaged' / 'state<k>')."""
    d = pd.read_parquet(path)
    d['trial'] = d.groupby('eid', sort=False).cumcount()
    stim_right = d['contrastRight'].notna()
    d['correct'] = (d['rewarded'] == 1).astype(float)
    d['choice_right'] = (stim_right == (d['correct'] == 1)).astype(float)
    d['contrast'] = d['signed_contrast'].abs()
    d['easy'] = d['contrast'] >= EASY
    P = sorted(c for c in d.columns if c.startswith('p_state'))
    names = engaged_states(d, P)
    eng = d['animal'].map(names)
    d['p_engaged'] = d[P].to_numpy()[np.arange(len(d)), eng.map(P.index).to_numpy()]
    best = np.array(P)[d[P].to_numpy().argmax(1)]
    d['map_state'] = np.where(best == eng, 'engaged', 'disengaged' if len(P) == 2 else best)
    d.attrs['engaged_state'] = names
    return d


def engaged_states(d, P):
    """{animal: name of its engaged p_state column}: highest posterior-weighted easy-trial accuracy."""
    e = d[d['easy']]
    out = {}
    for a, g in e.groupby('animal'):
        acc = [(g[p] * g['correct']).sum() / g[p].sum() for p in P]
        out[a] = P[int(np.nanargmax(acc))]
    return out


def pick_sessions(d, n_sessions, animals=None, per_animal=None, seed=0):
    """eids to plot: n_sessions at random (seeded), optionally only these animals and at most per_animal each."""
    s = d.drop_duplicates('eid')[['animal', 'eid']]
    if animals is not None:
        s = s[s['animal'].isin(animals)]
    s = s.sample(frac=1, random_state=seed)
    if per_animal is not None:
        s = s.groupby('animal', sort=False).head(per_animal)
    return s.head(n_sessions).sort_values(['animal', 'eid'])


def plot_session(g, fig=None):
    """Two rows for one session (rows of load_states for that eid).
    Top: P(engaged) and P(disengaged) (the whole posterior for K=2), MAP-disengaged trials shaded.
    Bottom: block (background), choices (top = right, bottom = left; filled = correct, open = error,
    size = contrast) and rolling accuracy over all trials and over easy trials."""
    g = g.sort_values('trial')
    x = g['trial'].to_numpy()
    if fig is None:
        fig = plt.figure(figsize=(12, 3.6))
    ax = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1.3], 'hspace': 0.12})

    a = ax[0]
    dis = g['map_state'].to_numpy() != 'engaged'
    a.fill_between(x, 0, 1, where=dis, step='mid', color=ps.NEUTRAL, alpha=0.18, lw=0)
    a.plot(x, g['p_engaged'], **ps.state_kw('engaged', lw=0.9))
    a.plot(x, 1 - g['p_engaged'], **ps.state_kw('disengaged', lw=0.7))
    a.axhline(0.5, color=ps.NEUTRAL, lw=0.4, ls=':')
    a.set_ylim(-0.02, 1.02)
    a.set_ylabel('posterior')
    a.legend(frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize='small')

    a = ax[1]
    pl = g['probabilityLeft'].to_numpy()
    edges = np.r_[0, np.nonzero(np.diff(pl))[0] + 1, len(pl)]
    for i0, i1 in zip(edges[:-1], edges[1:]):
        a.axvspan(x[i0] - 0.5, x[i1 - 1] + 0.5, color=ps.BLOCK.get(pl[i0], ps.NEUTRAL), alpha=0.12, lw=0)
    y = np.where(g['choice_right'] == 1, 1.12, -0.12)
    size = 2 + 14 * g['contrast'].to_numpy()
    ok = g['correct'].to_numpy() == 1
    a.scatter(x[ok], y[ok], s=size[ok], color=ps.STATE_INK, lw=0)
    a.scatter(x[~ok], y[~ok], s=size[~ok], facecolors='none', edgecolors=ps.STATE_INK, lw=0.5)
    a.plot(x, g['correct'].rolling(WIN_ALL, center=True, min_periods=3).mean(), color=ps.STATE_INK, lw=0.9,
           label=f'accuracy, all ({WIN_ALL} trials)')
    e = g[g['easy']]
    a.plot(e['trial'], e['correct'].rolling(WIN_EASY, center=True, min_periods=3).mean(), color=ps.NEUTRAL,
           lw=0.9, ls=(0, (3, 1.5)), label=f'accuracy, easy ({WIN_EASY} easy trials)')
    a.axhline(0.5, color=ps.NEUTRAL, lw=0.4, ls=':')
    a.set_ylim(-0.25, 1.25)
    a.set_yticks([-0.12, 0, 0.5, 1, 1.12], ['L', '0', '0.5', '1', 'R'])
    a.set_ylabel('choice / accuracy')
    a.set_xlabel('trial')
    a.set_xlim(x.min() - 1, x.max() + 1)
    handles, labels = a.get_legend_handles_labels()
    handles += [plt.Rectangle((0, 0), 1, 1, color=ps.BLOCK[b], alpha=0.3) for b in (0.2, 0.5, 0.8)]
    labels += [ps.BLOCK_LABEL[b] for b in (0.2, 0.5, 0.8)]
    a.legend(handles, labels, frameon=False, ncol=5, loc='upper left', bbox_to_anchor=(0, -0.28),
             fontsize='small')

    acc_eng = g.loc[g['easy'] & ~dis, 'correct'].mean()
    acc_dis = g.loc[g['easy'] & dis, 'correct'].mean()
    fig.suptitle(f"{g['animal'].iat[0]} · {g['eid'].iat[0][:8]} · {len(g)} trials · disengaged {dis.mean():.0%} · "
                 f"easy accuracy engaged {acc_eng:.2f} / disengaged {acc_dis:.2f}", fontsize='medium', y=0.99)
    return fig, ax


def save_sessions(d, sessions, out_dir, show=0, dpi=200):
    """One PNG per session in out_dir/<animal>/<animal>_<eid>.png; shows the first `show` inline."""
    out_dir = pathlib.Path(out_dir)
    paths = []
    for i, (animal, eid) in enumerate(sessions[['animal', 'eid']].itertuples(index=False)):
        fig, _ = plot_session(d[d['eid'] == eid])
        p = out_dir / animal / f'{animal}_{eid}.png'
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        paths.append(p)
        if i < show:
            plt.show()
        else:
            plt.close(fig)
    return paths
