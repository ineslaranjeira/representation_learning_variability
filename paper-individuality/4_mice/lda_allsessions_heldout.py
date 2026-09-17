"""
LDA FIT ON THE >=3-SESSION MICE, EVERY OTHER MOUSE PROJECTED IN
===============================================================
Same data and same pipeline as LDA_analyses_pipeline_ALLSESSIONS.ipynb. The one
change is that the >=3-session rule stops being a rule about WHICH MICE EXIST and
becomes a rule about WHICH MICE DEFINE THE SPACE.

In the notebook, `filter_sequences` does

    multi_sess_mice = session_count.loc[session_count['session'] > 2, 'mouse_name']
    all_sequences   = all_sequences.loc[all_sequences.mouse_name.isin(multi_sess_mice)]

so the 43 mice with one or two sessions are deleted before anything is fit, and
they never appear in the LD1/LD2 figure. Here they are kept, but held out of every
fitted object:

    PCA          fit on the >=3-session sessions, applied to all
    StandardScaler   "          "          "
    LDA (58 mouse classes)     "          "

and then drawn on top of the notebook's figure. Nothing the held-out mice contain
influences where they land.

WHAT THIS IS A TEST OF, and how to read it
------------------------------------------
Two separate questions, and the figure answers them in different panels:

  1. Do held-out mice land anywhere in particular, or do they pile up at the
     origin? LDA maximises between-class scatter for the classes it was TRAINED on.
     A mouse that was never a class has no reason to be pushed outwards, so
     shrinkage toward the centroid is the expected null. `spread ratio` in the
     printout is the number to look at: held-out mouse-mean SD divided by fit
     mouse-mean SD, per LD. Near 1 = the axes describe behaviour generally;
     well below 1 = the axes are partly a fit to the 58 training mice.

  2. For the 18 held-out mice that have exactly 2 sessions, do the two sessions
     land near each other? That asks whether the space is stable for a mouse it
     has never seen, and it is tested against a label shuffle.

The 25 one-session mice can only answer question 1 -- they have no within-mouse
distance, and their point carries no error bar.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats


# ------------------------------------------------------------------ paper_style
def _find_paper_style():
    here = pathlib.Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / 'paper_style.py').exists():
            return cand
        if (cand / 'paper-individuality' / 'paper_style.py').exists():
            return cand / 'paper-individuality'
    raise FileNotFoundError('paper_style.py not found')


ROOT = _find_paper_style()
for _p in (str(ROOT), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import paper_style as ps
from session_filters import exclusions_by_timepoint, find_csv

ps.use('poster')

# ---------------------------------------------------------------------- config
base_path = str(ROOT / 'data') + '/'
n_paw_states = 8
MIN_SESSIONS = 3          # the notebook's `session_count['session'] > 2`
MIN_COMPONENTS = 30       # notebook: min_components = 30
LDA_COMPONENTS = 30       # notebook: lda_components = min_components
QC_STRICTNESS = 'filtered_out'
SHOW_LABELS = False       # names on the held-out points; 43 of them is crowded

data_files = {
    'syllables': base_path + '8_k_10_bin_syllables_19-08-2026',
}

_excl = exclusions_by_timepoint(QC_STRICTNESS)
prob_sessions = sorted(_excl['Proficient'])


# --------------------------------------------- notebook functions (split in two)
def filter_sequences_qc_only(all_sequences):
    """The notebook's filter_sequences MINUS the >=3-session cut.

    The QC-sheet exclusion still applies to everybody; the session-count rule moves
    downstream, where it selects the fitting cohort instead of deleting mice.
    """
    print(f'{all_sequences.mouse_name.nunique()} mice in total')
    print(f'{all_sequences.session.nunique()} sessions in total')
    all_sequences = all_sequences.loc[~all_sequences['session'].isin(prob_sessions)].reset_index(drop=True)
    print(f"{all_sequences['session'].nunique()} sessions after removing bad sessions")
    cnt = (all_sequences[['mouse_name', 'session']].drop_duplicates()
           .groupby(['mouse_name'])['session'].count())
    print(f'  {(cnt >= MIN_SESSIONS).sum()} mice with at least {MIN_SESSIONS} sessions '
          f'({cnt[cnt >= MIN_SESSIONS].sum()} sessions)  <- the notebook keeps only these')
    print(f'  {(cnt < MIN_SESSIONS).sum()} mice with fewer '
          f'({cnt[cnt < MIN_SESSIONS].sum()} sessions)  <- the notebook deletes these; kept here')
    assert 'index' not in all_sequences.columns
    return all_sequences


def binarize(n_features_per_step, use_sequences, n_paw_states=8):
    """Verbatim from the notebook."""
    n_trials, timesteps = use_sequences.shape
    binarized = np.zeros((n_trials, timesteps * n_features_per_step))
    for t in range(timesteps):
        current_vals = use_sequences[:, t]
        nan_mask = np.isnan(current_vals)
        valid_mask = ~nan_mask
        labels_0idx = current_vals[valid_mask].astype(int)
        start_col = t * n_features_per_step
        if len(labels_0idx) > 0:
            valid_row_idx = np.arange(n_trials)[valid_mask]
            binarized[valid_row_idx, start_col + (labels_0idx % n_paw_states)] = 1
            binarized[valid_mask, start_col + n_paw_states] = ((labels_0idx // n_paw_states) % 2).astype(int)
            binarized[valid_mask, start_col + n_paw_states + 1] = (labels_0idx // (n_paw_states * 2)).astype(int)
        if np.any(nan_mask):
            binarized[nan_mask, start_col:start_col + n_features_per_step] = np.nan
    cols_to_delete = [t * n_features_per_step + 1 for t in range(timesteps)]
    return np.delete(binarized, cols_to_delete, axis=1)


def build_design_matrix(filename, n_paw_states=8):
    """Notebook's syllables branch, with filter_sequences_qc_only in place of
    filter_sequences. Per-session averaging is independent across sessions, so
    keeping the extra mice here does not alter any row that the notebook also has."""
    all_sequences = pd.read_parquet(filename)
    all_sequences['session'] = all_sequences['sample'].str[:36]
    all_sequences = filter_sequences_qc_only(all_sequences)

    design_df = (all_sequences.pivot(index=['mouse_name', 'session', 'sample', 'trial_type'],
                                     columns=['broader_label'], values='binned_sequence')
                 .reset_index().dropna())
    if 'index' in design_df.columns:
        design_df = design_df.drop(columns=['index'])
    design_df = design_df.sort_values(by='session')
    assert len(design_df) > 0
    print(f"✓ design_df: {len(design_df)} rows, {design_df['mouse_name'].nunique()} mice, "
          f"{design_df['session'].nunique()} sessions")

    epoch_to_analyse = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
    use_sequences = np.vstack(design_df[epoch_to_analyse].apply(lambda row: np.hstack(row), axis=1))
    assert len(use_sequences) == len(design_df)
    use_format = binarize(n_paw_states + 2, use_sequences, n_paw_states)

    session_mouse = design_df[['session', 'mouse_name']].drop_duplicates().set_index('session')['mouse_name'].to_dict()
    session_syllables = pd.DataFrame(use_format)
    session_syllables['session'] = design_df['session'].values
    session_syllables = session_syllables.groupby('session', sort=False)[
        np.arange(0, use_format.shape[1], 1)].mean()
    mouse_names_list = np.array([session_mouse[s] for s in session_syllables.index])
    print(f"✓ session aggregation: {len(session_syllables)} sessions")
    return session_syllables, mouse_names_list


# ------------------------------------------------------------------- statistics
def spread_table(coords, mice, fit_mask, n_lds=3):
    """SD of the MOUSE MEANS on each LD, fit cohort vs held-out cohort."""
    df = pd.DataFrame(coords[:, :n_lds], columns=[f'LD{i+1}' for i in range(n_lds)])
    df['mouse'] = mice
    df['cohort'] = np.where(fit_mask, 'fit', 'held-out')
    means = df.groupby(['cohort', 'mouse']).mean(numeric_only=True)
    rows = []
    for i in range(n_lds):
        ld = f'LD{i+1}'
        a = means.loc['fit', ld].std()
        b = means.loc['held-out', ld].std()
        rows.append(dict(dim=ld, fit_sd=a, held_sd=b, ratio=b / a))
    return pd.DataFrame(rows)


def cohort_offset(coords, mice, fit_mask, n_lds=3):
    """Is the held-out cohort sitting somewhere else entirely?

    LDA centres on the FIT data's mean, so the fit cohort's mouse means average to
    zero by construction and the held-out cohort's do not have to. A large offset
    would mean the two cohorts differ in behaviour, not just in how many sessions
    they have -- worth knowing before the held-out points are read as "more of the
    same". One mouse = one observation.
    """
    df = pd.DataFrame(coords[:, :n_lds], columns=[f'LD{i+1}' for i in range(n_lds)])
    df['mouse'] = mice
    df['cohort'] = np.where(fit_mask, 'fit', 'held-out')
    means = df.groupby(['cohort', 'mouse']).mean(numeric_only=True)
    rows = []
    for i in range(n_lds):
        ld = f'LD{i+1}'
        a, b = means.loc['fit', ld].values, means.loc['held-out', ld].values
        rows.append(dict(dim=ld, fit_mean=a.mean(), held_mean=b.mean(),
                         diff=b.mean() - a.mean(),
                         welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue,
                         mwu_p=stats.mannwhitneyu(a, b).pvalue))
    return pd.DataFrame(rows)


def matched_spread_control(coords, mice, fit_mask, n_boot=2000, seed=0, n_lds=3):
    """The control the raw spread comparison needs.

    `spread_table` compares the SD of 58 fit mouse means (3-16 sessions each) with
    the SD of 43 held-out mouse means (1-2 sessions each), and those two numbers are
    pushed in OPPOSITE directions by things that have nothing to do with the
    embedding generalising:

      * in-sample optimism inflates the FIT SD. The LDA axes were chosen to spread
        out those 58 mouse means specifically, sampling noise included -- 269
        sessions, 58 classes, 30 dimensions is ~4.6 samples per class, so a good
        part of that spread is fit to noise. New mice get no such boost.
      * few sessions per mouse inflates the HELD-OUT SD. A mean of one session
        carries the full single-session noise; a mean of eight carries an eighth of
        it, and that noise variance adds straight into the SD across mice.

    So a ratio near 1 can mean "generalises well" or "two biases cancelling". This
    resamples each FIT mouse down to the session counts the held-out mice actually
    have, which removes the second effect and leaves the first.
    """
    held_counts = pd.Series(mice[~fit_mask]).value_counts().values
    rng = np.random.default_rng(seed)
    fit_mice = np.unique(mice[fit_mask])
    idx_by_mouse = {m: np.where(mice == m)[0] for m in fit_mice}
    out = {f'LD{i+1}': [] for i in range(n_lds)}
    for _ in range(n_boot):
        means = []
        for m in fit_mice:
            k = int(rng.choice(held_counts))
            pool = idx_by_mouse[m]
            take = rng.choice(pool, min(k, len(pool)), replace=False)
            means.append(coords[take, :n_lds].mean(0))
        means = np.array(means)
        for i in range(n_lds):
            out[f'LD{i+1}'].append(means[:, i].std())
    rows = []
    for i in range(n_lds):
        ld = f'LD{i+1}'
        b = np.array(out[ld])
        held_sd = pd.Series(coords[~fit_mask, i], index=mice[~fit_mask]).groupby(level=0).mean().std()
        rows.append(dict(dim=ld, held_sd=held_sd,
                         fit_sd_matched=b.mean(), matched_lo=np.percentile(b, 2.5),
                         matched_hi=np.percentile(b, 97.5),
                         ratio=held_sd / b.mean()))
    return pd.DataFrame(rows)


def within_mouse_test(coords, mice, keep, n_perm=5000, seed=0):
    """Mean within-mouse pairwise distance in LD1-LD2 against a label shuffle,
    using only mice in `keep` that have >= 2 sessions."""
    sel = np.isin(mice, keep)
    C, m = coords[sel, :2], np.asarray(mice)[sel]
    ok = [x for x in np.unique(m) if (m == x).sum() >= 2]
    if len(ok) < 2:
        return None
    sel2 = np.isin(m, ok)
    C, m = C[sel2], m[sel2]

    def stat(labels):
        d = []
        for x in np.unique(labels):
            P = C[labels == x]
            if len(P) < 2:
                continue
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
            d.append(D[np.triu_indices(len(P), 1)].mean())
        return float(np.mean(d))

    obs = stat(m)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(m)) for _ in range(n_perm)])
    return dict(n_mice=len(ok), n_sessions=len(m), observed=obs,
                null_mean=float(null.mean()), null_sd=float(null.std()),
                p=float((np.sum(null <= obs) + 1) / (n_perm + 1)))


# ------------------------------------------------------------------------- main
def main():
    print(f"QC sheet: {find_csv()}")
    print(f"exclusions ({QC_STRICTNESS}): "
          + ", ".join(f"{k} {len(v)}" for k, v in _excl.items())
          + f"  ->  {len(prob_sessions)} proficient sessions dropped\n")

    parts, names = [], None
    for data_type, filename in data_files.items():
        print(f'--- {data_type}: {pathlib.Path(filename).name} ---')
        ss, mn = build_design_matrix(filename, n_paw_states=n_paw_states)
        parts.append(ss.add_prefix(f'{data_type}_'))
        names = mn if names is None else names
    session_syllables = pd.concat(parts, axis=1, join='inner')
    mouse_names = pd.Series(names, index=session_syllables.index, name='mouse_name')
    print(f"✓ combined: {len(session_syllables)} sessions, {session_syllables.shape[1]} features")

    # ---- the split: who defines the space
    counts = mouse_names.value_counts()
    fit_mice = np.array(sorted(counts[counts >= MIN_SESSIONS].index))
    held_mice = np.array(sorted(counts[counts < MIN_SESSIONS].index))
    mice = mouse_names.values
    fit_mask = np.isin(mice, fit_mice)
    print(f"\ncohort split at MIN_SESSIONS={MIN_SESSIONS}")
    print(f"  fit      {len(fit_mice):3d} mice  {fit_mask.sum():4d} sessions")
    print(f"  held out {len(held_mice):3d} mice  {(~fit_mask).sum():4d} sessions "
          f"({(counts[held_mice] == 1).sum()} with 1 session, "
          f"{(counts[held_mice] == 2).sum()} with 2)")

    X_all = np.array(session_syllables)

    # ---- PCA -> scaler -> LDA, every one of them fit on the fit cohort only
    pca = PCA(n_components=min(fit_mask.sum(), X_all.shape[1])).fit(X_all[fit_mask])
    X_pca = pca.transform(X_all)[:, :MIN_COMPONENTS]
    scaler = StandardScaler().fit(X_pca[fit_mask])
    norm_pop = scaler.transform(X_pca)

    y = pd.factorize(mice)[0]
    n_cls = len(fit_mice)
    lda = LinearDiscriminantAnalysis(priors=np.ones(n_cls) / n_cls,
                                     n_components=min(LDA_COMPONENTS, n_cls - 1))
    lda.fit(norm_pop[fit_mask], mice[fit_mask])
    coords = lda.transform(norm_pop)
    print(f"\nLDA fit on {fit_mask.sum()} sessions / {n_cls} mice; "
          f"projected all {len(coords)} sessions")
    print("LDA explained_variance_ratio_ (top 6):",
          np.round(lda.explained_variance_ratio_[:6], 3))

    # ---- question 1: does the held-out cohort occupy the same range?
    spread = spread_table(coords, mice, fit_mask)
    print("\nspread of MOUSE MEANS, held-out vs fit (ratio ~1 = same range, "
          "<<1 = held-out shrink to the centroid):")
    print(spread.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    matched = matched_spread_control(coords, mice, fit_mask)
    print("\nsame comparison with the FIT mice resampled to the held-out session counts\n"
          "(1-2 sessions each), which is the only version of it that is interpretable:")
    print(matched.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    print("\noffset of the held-out cohort (one mouse = one observation; the fit "
          "cohort averages to 0 by construction):")
    print(cohort_offset(coords, mice, fit_mask).to_string(
        index=False, float_format=lambda v: f'{v:.3f}'))

    # ---- question 2: are held-out mice placed consistently across their sessions?
    print("\nwithin-mouse distance in LD1-LD2 vs shuffled labels:")
    for name, keep in [('fit mice', fit_mice), ('held-out (2-session)', held_mice)]:
        r = within_mouse_test(coords, mice, keep)
        print(f"  {name:22s} " + ("n/a" if r is None else
              f"{r['n_mice']:3d} mice / {r['n_sessions']:3d} sessions   "
              f"observed {r['observed']:.3f}  null {r['null_mean']:.3f}±{r['null_sd']:.3f}   "
              f"p = {r['p']:.4f}"))

    # --------------------------------------------------------------------- figure
    clustered = pd.DataFrame(coords)
    clustered['mouse_name'] = mice
    clustered['cohort'] = np.where(fit_mask, 'fit', 'held-out')
    clustered['session'] = session_syllables.index

    grouped = clustered.groupby(['cohort', 'mouse_name'])
    mouse_means = grouped[[0, 1, 2]].mean()
    mouse_sems = grouped[[0, 1, 2]].std().div(np.sqrt(grouped.size()), axis=0)

    fit_means = mouse_means.loc['fit']
    fit_sems = mouse_sems.loc['fit']
    held_means = mouse_means.loc['held-out']
    held_sems = mouse_sems.loc['held-out']
    held_n = counts[held_means.index]

    # LD1 colour scale is set by the fit cohort, so the colours mean the same thing
    # they do in the notebook's figure; held-out mice are then coloured on that scale.
    ps.set_ld1_scale(np.nanmax(np.abs(fit_means[0].values)))

    fig, ax = ps.figure('single', scale=1.6)

    # the notebook's figure, faded, as the background
    for mouse, cvec in zip(fit_means.index, ps.ld1_colors(fit_means[0].values)):
        ax.errorbar(fit_means.loc[mouse, 0], fit_means.loc[mouse, 1],
                    xerr=fit_sems.loc[mouse, 0], yerr=fit_sems.loc[mouse, 1],
                    fmt='o', color=cvec, ecolor=cvec, alpha=0.45,
                    markeredgecolor='0.55', markeredgewidth=0.3, zorder=2)

    # the mice the notebook throws away, projected in, on top
    for mouse, cvec in zip(held_means.index, ps.ld1_colors(held_means[0].values)):
        n = int(held_n[mouse])
        ax.errorbar(held_means.loc[mouse, 0], held_means.loc[mouse, 1],
                    xerr=(0 if np.isnan(held_sems.loc[mouse, 0]) else held_sems.loc[mouse, 0]),
                    yerr=(0 if np.isnan(held_sems.loc[mouse, 1]) else held_sems.loc[mouse, 1]),
                    fmt='D' if n == 1 else 's', color=cvec, ecolor='0.3',
                    markersize=plt.rcParams['lines.markersize'] * 1.15,
                    markeredgecolor='k', markeredgewidth=0.9, alpha=1.0, zorder=5)
        if SHOW_LABELS:
            ax.text(held_means.loc[mouse, 0] + 0.06, held_means.loc[mouse, 1] + 0.06,
                    str(mouse), fontsize=plt.rcParams['font.size'] * 0.5,
                    color='0.2', zorder=6)

    h_fit = plt.Line2D([], [], marker='o', ls='', color='0.55', alpha=0.6,
                       label=f'fit on ({len(fit_mice)} mice, {fit_mask.sum()} sessions)')
    h_h2 = plt.Line2D([], [], marker='s', ls='', color='0.25', markeredgecolor='k',
                      label=f'held out, 2 sessions ({(held_n == 2).sum()})')
    h_h1 = plt.Line2D([], [], marker='D', ls='', color='0.25', markeredgecolor='k',
                      label=f'held out, 1 session ({(held_n == 1).sum()})')
    ax.legend(handles=[h_fit, h_h2, h_h1], fontsize=plt.rcParams['font.size'] * 0.6,
              loc='best', frameon=False)

    ax.axvline(0, color='0.6', ls='--', lw=plt.rcParams['axes.linewidth'], zorder=0)
    ax.axhline(0, color='0.6', ls='--', lw=plt.rcParams['axes.linewidth'], zorder=0)
    ax.set_box_aspect(1)
    ps.ld1_colorbar(ax, label='LD1', fraction=0.046, pad=0.02)
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_title(f'LDA fit on mice with ≥{MIN_SESSIONS} sessions;\n'
                 'mice with fewer projected in',
                 fontsize=plt.rcParams['font.size'] * 0.8)
    fig.tight_layout()
    ps.savefig(fig, 'lda12_mice_heldout', svg=True)

    # session-level version of the same thing, no averaging
    fig2, ax2 = ps.figure('single', scale=1.6)
    ax2.scatter(coords[fit_mask, 0], coords[fit_mask, 1], s=plt.rcParams['lines.markersize'] ** 2 * 0.5,
                color='0.70', alpha=0.45, linewidths=0, zorder=1,
                label=f'fit ({fit_mask.sum()} sessions)')
    ax2.scatter(coords[~fit_mask, 0], coords[~fit_mask, 1],
                s=plt.rcParams['lines.markersize'] ** 2 * 0.9,
                color='#1f6fb4', edgecolor='k', linewidths=0.6, zorder=3,
                label=f'held out ({(~fit_mask).sum()} sessions)')
    ax2.axvline(0, color='0.6', ls='--', lw=plt.rcParams['axes.linewidth'], zorder=0)
    ax2.axhline(0, color='0.6', ls='--', lw=plt.rcParams['axes.linewidth'], zorder=0)
    ax2.set_xlabel('LD1'); ax2.set_ylabel('LD2'); ax2.set_box_aspect(1)
    ax2.legend(fontsize=plt.rcParams['font.size'] * 0.6, frameon=False)
    ax2.set_title('every session, not mouse means', fontsize=plt.rcParams['font.size'] * 0.8)
    fig2.tight_layout()
    ps.savefig(fig2, 'lda12_sessions_heldout', svg=True)

    # ---------------------------------------------------------------- figure 3
    # Sessions behind, mouse averages in front, both on the paper's LD1 gradient,
    # mice with SEM across their own sessions -- i.e. the notebook's figure, with
    # the sessions that go into each average made visible underneath.
    #
    # Cohort is deliberately NOT a colour or a size: fit and held-out share the
    # ramp and the marker, and the only cohort cue is the ring. A second hue for
    # cohort would fight the LD1 ramp, which is the thing the figure is about.
    def _lighten(rgba, w=0.45):
        c = np.array(rgba, float).copy()
        c[..., :3] = 1.0 - (1.0 - c[..., :3]) * (1.0 - w)
        return c

    # The scale is set by every mouse IN THIS FIGURE, so the held-out mouse at
    # LD1 ~ 10 does not saturate. That makes the ramp slightly wider than the
    # 58-mouse figure, where the fit cohort alone sets it.
    ps.set_ld1_scale(np.nanmax(np.abs(np.r_[fit_means[0].values, held_means[0].values])))

    fig3, ax3 = ps.figure('single', scale=1.6)
    ms = plt.rcParams['lines.markersize']
    elw = plt.rcParams['lines.linewidth'] * 0.7
    RING = '#111111'

    # --- sessions, same ramp blended toward white
    sess_c = _lighten(ps.ld1_colors(coords[:, 0]))
    # a faint grey outline, because the LD1 ramp is WHITE at zero: lightened
    # near-zero sessions are invisible against the page without it
    ax3.scatter(coords[fit_mask, 0], coords[fit_mask, 1], s=ms ** 2 * 0.5,
                c=sess_c[fit_mask], edgecolors='0.78', linewidths=0.45, zorder=1)
    ax3.scatter(coords[~fit_mask, 0], coords[~fit_mask, 1], s=ms ** 2 * 0.5,
                c=sess_c[~fit_mask], edgecolors=RING, linewidths=0.8, zorder=2)

    # --- mouse averages, SEM across that mouse's sessions (1-session mice get none)
    for means, sems, is_held in ((fit_means, fit_sems, False), (held_means, held_sems, True)):
        cols = ps.ld1_colors(means[0].values)
        for mouse, cvec in zip(means.index, cols):
            xe, ye = sems.loc[mouse, 0], sems.loc[mouse, 1]
            ax3.errorbar(means.loc[mouse, 0], means.loc[mouse, 1],
                         xerr=0.0 if np.isnan(xe) else xe,
                         yerr=0.0 if np.isnan(ye) else ye,
                         fmt='none', ecolor=cvec, elinewidth=elw, alpha=0.9,
                         zorder=4 if is_held else 3)
        ax3.scatter(means[0].values, means[1].values, s=ms ** 2 * 1.4, c=cols,
                    edgecolors=RING if is_held else '0.55',
                    linewidths=1.8 if is_held else 0.5,
                    zorder=6 if is_held else 5)

    handles = [
        plt.Line2D([], [], marker='o', ls='', markerfacecolor=_lighten(np.array([0.45, 0.45, 0.45, 1.0])),
                   markeredgecolor='0.78', markeredgewidth=0.45, markersize=ms * 0.7,
                   label=f'session ({len(coords)})'),
        plt.Line2D([], [], marker='o', ls='', markerfacecolor='0.45',
                   markeredgecolor='0.55', markeredgewidth=0.5, markersize=ms * 1.2,
                   label=f'mouse average \u00b1 SEM ({len(fit_means) + len(held_means)})'),
        plt.Line2D([], [], marker='o', ls='', markerfacecolor='none',
                   markeredgecolor=RING, markeredgewidth=1.8, markersize=ms * 1.2,
                   label=f'held out ({len(held_mice)} mice, {(~fit_mask).sum()} sessions)'),
    ]
    ax3.legend(handles=handles, fontsize=plt.rcParams['font.size'] * 0.6,
               loc='best', frameon=False)
    ax3.axvline(0, color='0.6', ls='--', lw=plt.rcParams['axes.linewidth'], zorder=0)
    ax3.axhline(0, color='0.6', ls='--', lw=plt.rcParams['axes.linewidth'], zorder=0)
    ax3.set_box_aspect(1)
    ps.ld1_colorbar(ax3, label='LD1', fraction=0.046, pad=0.02)
    ax3.set_xlabel('LD1')
    ax3.set_ylabel('LD2')
    fig3.tight_layout()
    ps.savefig(fig3, 'lda12_sessions_and_means_heldout', svg=True)

    ps.set_ld1_scale(None)
    out = ROOT / 'figures'
    print(f"\nfigures written to {out}")
    return clustered


if __name__ == '__main__':
    main()
