#%%
"""
LDA SCORE SWEEP -- mouse discriminability vs number of dimensions
=================================================================
How well the LDA tells mice apart, as a function of how many PCA dimensions it is given,
for several behavioural encodings (syllables at different k, raw binned signals, trial
summaries).

SHARED CODE WITH THE NOTEBOOK. The encoding, filtering and LDA all live in
4_mice/functions.py and are imported below, so this script and
4_mice/LDA_analyses_pipeline_ALLSESSIONS.ipynb cannot drift apart again. They previously
disagreed in four places -- session exclusions, the balanced-subsampling guard, the trial
feature branch and the chance level. Two consequences worth knowing:

  * SESSION EXCLUSIONS come from the curated QC sheet (individuality-paper_data_4Sep26.csv,
    via learning_individuality/session_filters), not the hardcoded list this script used
    to carry. Editing the sheet now changes this sweep too.
  * THE SUBSAMPLING CAP is a maximum contribution, not a minimum requirement, so no mouse
    is ever tested against a classifier that never saw it. See run_lda's docstring for the
    measured effect -- it is invisible at n_per_mouse=3 and large above it.

The notebook's operating point is 30 PCA dimensions, which is in the sweep grid below, so
x = 30 is the directly comparable value.
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# the shared pipeline lives in paper-individuality/4_mice/functions.py
_root = pathlib.Path(__file__).resolve().parent.parent      # paper-individuality/
sys.path.insert(0, str(_root / '4_mice'))
sys.path.insert(0, str(_root))
from functions import build_design_matrix, dim_red, qc_exclusions, run_lda

# paper_style holds every cross-figure decision -- panel sizes, colours, where figures are
# written. 'poster' is big type for a poster; ps.use('paper') gives the journal version of
# the same figures with no other edits.
import paper_style as ps
ps.use('poster')

#%%
""" IMPORTANT PATHS """
base_path = str(_root / 'data') + '/'

# One QC read for the whole sweep, so every dataset is filtered identically and the sheet
# is not re-parsed per dataset.
PROB_SESSIONS = qc_exclusions(timepoint='Proficient', strictness='filtered_out')

N_PER_MOUSE = 3
N_REPEATS = 10


def uses_dim_red(filename):
    """Trial-level data is a handful of interpretable session summaries, so it is fed to the
    LDA as-is (no PCA). Everything else goes through dimensionality reduction."""
    return not ('trial' in filename and 'syllables' not in filename and 'raw' not in filename)


def load_and_prepare(dataset, n_paw_states, verbose=False):
    """Load one dataset file, filter, encode and average per session."""
    return build_design_matrix(base_path + dataset, n_paw_states=n_paw_states,
                               prob_sessions=PROB_SESSIONS, verbose=verbose)


def score_dataset(dataset, n_paw_states, n_dim, verbose=True):
    """LDA score of one dataset at `n_dim` PCA dimensions (or on its raw features, if the
    dataset gets no dimensionality reduction).

    Returns (true_scores, shuffle_scores, n_used, n_mice)."""
    session_syllables, design_df = load_and_prepare(dataset, n_paw_states, verbose)

    if uses_dim_red(base_path + dataset):
        mat = np.array(dim_red(session_syllables)[:, :n_dim])
    else:
        mat = np.array(session_syllables)      # no dimensionality reduction
    n_used = mat.shape[1]

    norm_pop = StandardScaler().fit_transform(mat.copy())
    true_scores, shuffle_scores = run_lda(design_df, session_syllables, n_used, norm_pop,
                                          n_per_mouse=N_PER_MOUSE, n_repeats=N_REPEATS,
                                          verbose=verbose)
    return true_scores, shuffle_scores, n_used, design_df['mouse_name'].nunique()

#%%
#### LOOP ###

# ONE ENTRY PER DATASET -- a dict, not four parallel lists.
# The parallel lists silently desynced the last time this was edited: `datasets` was cut to
# three entries while `paw_states`, `labels` and `colors` kept six, so each dataset was
# handed another dataset's k -- and the first got NaN, which made n_features_per_step NaN
# and crashed binarize. Keyed by filename, that failure mode is gone.
# n_paw is np.nan for encodings with no paw states (raw signals, trial summaries).
DATASETS = {
    '10_k_10_bin_syllables_10-09-2026': dict(n_paw=10, label='12-state syllables',
                                             color='#990000'),
    '8_k_10_bin_syllables_19-08-2026':  dict(n_paw=8,  label='10-state syllables',
                                             color='#E46D6F'),
    '6_k_10_bin_syllables_21-08-2026':  dict(n_paw=6,  label='8-state syllables',
                                             color='#F28F8F'),
    # a syllable label counts k paw states + whisk + lick, hence k + 2
    '10_bin_raw_10-09-2026':  dict(n_paw=np.nan, label='Raw data',   color='#1A1A1A'),
    'all_trials_10-09-2026':  dict(n_paw=np.nan, label='Trial data', color='#3B6EA5')
}

n_components = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 100, 200]

# All results keyed by dataset name, so nothing depends on positional alignment.
# Trial-level data gets no dimensionality reduction and therefore no sweep -- its score is
# a single number, plotted as one dot at x = number of trial features.
scores, score_std = {}, {}
# 95% CI from resampling MICE -- the band that goes on the figure. `score_std` is the
# spread across training subsamples, which is a robustness diagnostic and NOT uncertainty:
# it shrinks as n_per_mouse rises simply because a mouse with exactly n_per_mouse sessions
# has no draw left to make. Kept in the printout, off the plot.
score_ci_low, score_ci_high = {}, {}
feature_dims = {}     # dataset -> n features used, for the datasets with no PCA
mouse_counts = {}     # dataset -> n mice, because chance is 1/n_mice and it differs per dataset

for dataset, meta in DATASETS.items():
    filename = base_path + dataset
    if not os.path.exists(filename):
        print(f'!! skipping missing dataset: {dataset}')
        continue
    print(f'\n=== {dataset}  (k = {meta["n_paw"]}) ===')

    session_syllables, design_df = load_and_prepare(dataset, meta['n_paw'], verbose=True)
    mouse_counts[dataset] = design_df['mouse_name'].nunique()
    scaler = StandardScaler()

    if not uses_dim_red(filename):
        """ NO DIMENSIONALITY REDUCTION -- use the trial features directly """
        n_feat = session_syllables.shape[1]
        norm_pop = scaler.fit_transform(np.array(session_syllables))
        print(f'{dataset}: {n_feat} trial features, no dimensionality reduction')

        true_scores, shuffle_scores, lo, hi = run_lda(
            design_df, session_syllables, n_feat, norm_pop, n_per_mouse=N_PER_MOUSE,
            n_repeats=N_REPEATS, return_ci=True)
        for _d in (scores, score_std, score_ci_low, score_ci_high):
            _d[dataset] = np.full(len(n_components), np.nan)
        scores[dataset][0] = np.mean(true_scores)
        score_std[dataset][0] = np.std(true_scores)
        score_ci_low[dataset][0], score_ci_high[dataset][0] = lo, hi
        feature_dims[dataset] = n_feat
        continue

    """ DIMENSIONALITY REDUCTION """
    X_pca = dim_red(session_syllables)
    for _d in (scores, score_std, score_ci_low, score_ci_high):
        _d[dataset] = np.full(len(n_components), np.nan)

    for b, n_component in enumerate(n_components):
        print(f'  {dataset}  {n_component}D')
        mat = np.array(X_pca[:, :n_component])
        norm_pop = scaler.fit_transform(mat.copy())

        """ RUN LDA """
        true_scores, shuffle_scores, lo, hi = run_lda(
            design_df, session_syllables, n_component, norm_pop, n_per_mouse=N_PER_MOUSE,
            n_repeats=N_REPEATS, return_ci=True, verbose=False)
        scores[dataset][b] = np.mean(true_scores)
        score_std[dataset][b] = np.std(true_scores)
        score_ci_low[dataset][b], score_ci_high[dataset][b] = lo, hi
        print(f'    true {scores[dataset][b]:.3f}  [95% CI over mice {lo:.3f}, {hi:.3f}]'
              f'  (subsample spread +/-{score_std[dataset][b]:.3f})')

# %%

fig, ax = ps.figure('square', scale=.6)

for dataset, meta in DATASETS.items():
    if dataset not in scores or np.all(np.isnan(scores[dataset])):
        continue

    # Trial data has no dimensionality sweep -> single dot at x = number of trial features
    if dataset in feature_dims:
        ax.errorbar(feature_dims[dataset], scores[dataset][0],
                    yerr=[[scores[dataset][0] - score_ci_low[dataset][0]],
                          [score_ci_high[dataset][0] - scores[dataset][0]]],
                    fmt='o', color=meta['color'], capsize=3, label=meta['label'])
        continue

    ax.plot(n_components, scores[dataset], color=meta['color'], label=meta['label'])
    # band = 95% CI over MICE (asymmetric), not +/- the across-subsample spread
    ax.fill_between(n_components, score_ci_low[dataset], score_ci_high[dataset],
                    color=meta['color'], alpha=0.15, lw=0)

ax.set_xticks(np.array(n_components).astype(int))
ax.set_xticks([1, 10, 20, 30, 50, 100])
ax.set_ylim([0, 1])
ax.set_xlim([0, 100])
# ax.set_xticks(n_components)
ax.set_xlabel('# PCA dimensions')
# the score depends on the balance cap as much as on the encoding (0.76 at n_per_mouse=3
# vs 0.85 at 4), so the setting travels with the figure rather than living only in the code
ax.set_ylabel(f'LDA score')

# Chance is 1/n_mice, and the mouse count depends on which sessions survive QC for each
# dataset -- so it is read from the data rather than hardcoded. One line per distinct value,
# but a SINGLE label: 1/55 and 1/58 are visually identical and two texts overprint.
chance_counts = sorted(set(mouse_counts.values()))
for n_mice in chance_counts:
    ax.axhline(1 / n_mice, color=ps.NEUTRAL, ls='--', lw=1.0)
chance_label = ('chance (1/%d)' % chance_counts[0] if len(chance_counts) == 1
                else 'chance (1/%d to 1/%d)' % (chance_counts[-1], chance_counts[0]))
chance_label = ('chance ')
ax.text(np.min(n_components), np.mean([1 / n for n in chance_counts]), chance_label,
        ha='left', va='bottom', color=ps.NEUTRAL,
        fontsize=plt.rcParams['font.size'] * 0.8)

ax.legend(frameon=False)
fig.tight_layout()
ps.savefig(fig, 'lda_score_sweep', svg=True)
plt.show()

# %%
#### BAR PLOT AT A FIXED NUMBER OF DIMENSIONS ####
# Standalone: only needs the imports and helpers above, not the sweep loop.

n_dim = 6

# label: (dataset file, n_paw_states)
bar_datasets = {
    'Trial data': ('all_trials_06-07-2026', np.nan),   # no dimensionality reduction
    'Syllables': ('8_k_10_bin_syllables_06-07-2026', 8),
    'Raw data': ('10_bin_raw_16-06-2026', np.nan),
}
bar_colors = {'Trial data': '#3B6EA5', 'Syllables': '#990000', 'Raw data': '#1A1A1A'}

bar_results = {}
for label, (dataset, n_paw) in bar_datasets.items():
    if not os.path.exists(base_path + dataset):
        print(f'!! skipping missing dataset: {dataset}')
        continue
    print(f'--- {label}: {dataset} ---')
    true_scores, shuffle_scores, n_used, n_mice = score_dataset(dataset, n_paw, n_dim)
    bar_results[label] = {'true': true_scores, 'shuffle': shuffle_scores,
                          'n_used': n_used, 'n_mice': n_mice}
    print(f'{label}: {n_used} dimensions, {n_mice} mice, '
          f'true {np.mean(true_scores):.3f} vs shuffled {np.mean(shuffle_scores):.3f}')

# %%
plot_labels = list(bar_results.keys())
x = np.arange(len(plot_labels))
width = 0.38

fig, ax = ps.figure('single' if len(plot_labels) <= 3 else 'wide')

for i, label in enumerate(plot_labels):
    r = bar_results[label]
    # True labels: one bar per dataset, in the dataset's own colour
    ax.bar(x[i] - width/2, np.mean(r['true']), width, yerr=np.std(r['true']),
           color=bar_colors.get(label, 'grey'), capsize=3,
           label='True labels' if i == 0 else None)
    # Shuffled labels: neutral grey + hatch, so the control never reads as a dataset
    ax.bar(x[i] + width/2, np.mean(r['shuffle']), width, yerr=np.std(r['shuffle']),
           color='white', edgecolor=ps.NEUTRAL, hatch='///', capsize=3,
           label='Shuffled labels' if i == 0 else None)

ax.set_xticks(x)
ax.set_xticklabels([f"{lab}\n({bar_results[lab]['n_used']}D)" for lab in plot_labels])
ax.set_ylabel(f'Mouse discriminability score\n(n_per_mouse = {N_PER_MOUSE})')
# band is the CI over mice -- say so, since the previous version's band was a different thing
ax.set_ylim([0, 1])
ax.legend(frameon=False)      # spine treatment comes from paper_style, not set per figure
fig.tight_layout()
ps.savefig(fig, f'lda_score_bars_{n_dim}D', svg=True)
plt.show()
# %%
