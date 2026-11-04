"""
WHERE DOES BEHAVIOURAL VARIABILITY LIVE?  lab > mouse > session > trial > moment
=================================================================================
Agnostic counterpart to the LDA: no labels are used to choose a projection, every feature
counts, and each level of the hierarchy is given only the variance its own sampling noise
cannot explain.

The unit is one TRIAL: its syllable sequence (40 time-warped bins, 4 epochs x 10) one-hot
coded as 10 indicators per bin (8 paw states, whisking, licking) -> 400 binary features.
Same sessions, mice and filters as 4_mice/lda_shrinkage.ipynb (260 sessions, 56 mice),
minus the 0.7% of trials with a missing bin (DROP_INCOMPLETE_TRIALS there, LD1 |r| 0.994).

THE ESTIMATOR (nested random effects, method of moments / ANOVA, unbalanced)
---------------------------------------------------------------------------
    y = mu + lab + mouse(lab) + session(mouse) + trial(session)

For level l with groups g nested in parents p,  SS_l = sum_g n_g |ybar_g - ybar_p|^2, and
for a random effect at a level c at or below l, with groups h,

    E[SS_l] / sigma^2_c  =  sum_g S_g / n_g  -  sum_p S_p / n_p,     S_g = sum_{h in g} n_h^2

(0 for c above l). Solving the triangular system K sigma^2 = SS gives unbiased components
for any unbalanced nesting -- raw sums of squares do not: a mouse mean still carries its
sessions' noise, a session mean its trials' noise. Everything is linear in the features,
so per-feature components sum to the multivariate (trace) decomposition.

Two further splits put the sub-trial scale on the same axis:
  * TASK TIME -- the variance of the population's mean syllable use ACROSS bins: how much
    of "what a mouse is doing at a random moment" is set by where in the trial it is.
  * TRIAL-WIDE vs MOMENT -- within an epoch, the part of a trial's deviation shared by all
    its bins (a whole trial spent whisking) vs bin-to-bin fluctuation. Balanced one-way
    random model, trials as groups, the epoch's 10 bins as replicates. Bins are smoothed
    by time-warping, so neighbouring bins are correlated and trial-wide is an upper bound.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy import sparse

PI = pathlib.Path(__file__).resolve().parents[1]
for _p in (PI, PI / 'learning_individuality', PI / '4_mice'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SYLLABLE_FILE = PI / 'clustering' / 'data_files' / '8_k_10_bin_syllables_02-10-2026'   # the LDA's file
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
N_PAW_STATES = 8
N_SYLL = N_PAW_STATES + 2            # paw one-hot + whisk + lick, per bin
LEVELS = ['lab', 'mouse', 'session']  # coarse -> fine; the trial is the residual


# ------------------------------------------------------------------------- data
def load_trials(qc_strictness='filtered_out', max_session_missing=0.1, min_sessions=3, syllable_file=None):
    """-> (codes [trials x 40], meta DataFrame). lda_shrinkage's filters, in its order:
    QC sheet, sessions with > 10% missing bins, mice with < 3 sessions; then incomplete trials.
    syllable_file: another file with the same layout (default SYLLABLE_FILE, the LDA's)."""
    from session_filters import exclusions_by_timepoint, find_csv
    from functions import lab_labels

    seq = pd.read_parquet(syllable_file or SYLLABLE_FILE)
    seq['session'] = seq['sample'].str[:36]
    seq = seq[~seq['session'].isin(exclusions_by_timepoint(qc_strictness)['Proficient'])]
    nan_frac = np.isnan(np.stack(seq['binned_sequence'].to_numpy())).mean(axis=1)
    by_session = pd.Series(nan_frac, index=seq['session'].to_numpy()).groupby(level=0).mean()
    seq = seq[~seq['session'].isin(by_session[by_session > max_session_missing].index)]
    counts = seq[['mouse_name', 'session']].drop_duplicates().groupby('mouse_name')['session'].count()
    seq = seq[seq['mouse_name'].isin(counts[counts >= min_sessions].index)]

    t = (seq.pivot(index=['mouse_name', 'session', 'sample', 'trial_type'], columns='broader_label',
                   values='binned_sequence').reset_index().dropna(subset=EPOCHS))
    codes = np.vstack([np.hstack(r) for r in t[EPOCHS].to_numpy()]).astype(float)
    complete = ~np.isnan(codes).any(axis=1)
    print(f'{len(codes)} trials, {t["session"].nunique()} sessions, {t["mouse_name"].nunique()} mice; '
          f'dropping {1 - complete.mean():.1%} with a missing bin')
    codes, t = codes[complete], t[complete].reset_index(drop=True)

    meta = t[['mouse_name', 'session', 'trial_type']].copy()
    meta['trial'] = t['sample'].str[37:].astype(float).astype(int)
    tt = meta['trial_type'].str.split(expand=True)
    meta['outcome'], meta['contrast'], meta['block'], meta['side'] = tt[0], tt[1].astype(float), tt[2], tt[3]
    labs = lab_labels(pd.Index(meta['session'].unique()),
                      mouse_names=meta.drop_duplicates('session').set_index('session')['mouse_name'],
                      verbose=False)
    meta['lab'] = meta['session'].map(labs)
    sheet = pd.read_csv(find_csv(), header=1).drop_duplicates('eid').set_index('eid')
    meta['date'] = pd.to_datetime(meta['session'].map(sheet['date']), errors='coerce')
    meta = meta.sort_values(['lab', 'mouse_name', 'session', 'trial'])
    codes = codes[meta.index.to_numpy()]
    meta = meta.reset_index(drop=True)
    print(f'kept {len(meta)} trials, {meta.session.nunique()} sessions, {meta.mouse_name.nunique()} mice, '
          f'{meta.lab.nunique()} labs')
    return codes, meta


def one_hot(codes, n_paw=N_PAW_STATES):
    """[trials x bins] packed codes -> [trials x bins x 10] float32 indicators."""
    c = codes.astype(int)
    out = np.zeros(c.shape + (n_paw + 2,), dtype=np.float32)
    np.put_along_axis(out, (c % n_paw)[..., None], 1, axis=-1)
    out[..., n_paw] = (c // n_paw) % 2
    out[..., n_paw + 1] = c // (2 * n_paw)
    return out


# -------------------------------------------------------------- the estimator
def _indicator(labels, n):
    codes, uniq = pd.factorize(labels, sort=True)
    return sparse.csr_matrix((np.ones(n, np.float32), (codes, np.arange(n))), shape=(len(uniq), n)), codes


def design(meta, levels=LEVELS):
    """Group codes per level (coarse -> fine, then the trial), and the K matrix of the estimator."""
    n = len(meta)
    groups = [pd.factorize(meta[l] if l != 'mouse' else meta['mouse_name'], sort=True)[0] for l in levels]
    groups.append(np.arange(n))                                   # the trial: the residual level
    return groups, k_matrix(groups)


def k_matrix(groups):
    """K[l, c] = E[SS_l] / sigma^2_c for nested groups (coarse -> fine, last = units)."""
    L = len(groups)
    n_at = [np.bincount(g).astype(float) for g in groups]
    K = np.zeros((L, L))
    for l in range(L):
        parent = groups[l - 1] if l else np.zeros_like(groups[0])
        for c in range(l, L):
            nh = n_at[c]
            h_of_g = pd.Series(groups[l]).groupby(groups[c]).first().to_numpy()   # each c-group's l-group
            h_of_p = pd.Series(parent).groupby(groups[c]).first().to_numpy()
            S_g = np.bincount(h_of_g, weights=nh ** 2, minlength=len(n_at[l]))
            S_p = np.bincount(h_of_p, weights=nh ** 2)
            n_p = np.bincount(parent).astype(float)
            K[l, c] = (S_g / n_at[l]).sum() - (S_p[S_p > 0] / n_p[S_p > 0]).sum()
    return K


def sums_of_squares(Y, groups):
    """SS per level and feature: [levels x features]. Y: [trials x features]."""
    n = len(Y)
    means, ss = [Y.mean(axis=0, keepdims=True)], []
    parent_of_unit = np.zeros(n, int)
    for g in groups[:-1]:
        A, _ = _indicator(g, n)
        cnt = np.asarray(A.sum(axis=1)).ravel()
        m = np.asarray(A @ Y) / cnt[:, None]
        p = means[-1][parent_of_unit]                           # parent mean, unit-aligned
        ss.append(((m[g] - p) ** 2).sum(axis=0))                # sum over units = sum_g n_g |.|^2
        means.append(m)
        parent_of_unit = g
    ss.append(((Y - means[-1][parent_of_unit]) ** 2).sum(axis=0))
    return np.stack(ss)


def components(Y, groups, K=None):
    """Variance components [levels(+trial) x features]. Can be negative (sampling); not truncated."""
    K = k_matrix(groups) if K is None else K
    return np.linalg.solve(K, sums_of_squares(Y, groups))


def trial_wide_split(R, epoch_bins=10):
    """Within-session residuals R [trials x bins x syllables] -> (trial-wide, moment) variance per
    (epoch, syllable): balanced one-way random model, trials as groups, the epoch's bins as replicates."""
    n, B, S = R.shape
    tw, mo = np.zeros((B // epoch_bins, S)), np.zeros((B // epoch_bins, S))
    for e in range(B // epoch_bins):
        Re = R[:, e * epoch_bins:(e + 1) * epoch_bins]
        m = Re.mean(axis=1, keepdims=True)
        ms_w = ((Re - m) ** 2).sum(axis=(0, 1)) / (n * (epoch_bins - 1))
        ms_b = epoch_bins * ((m[:, 0] - m[:, 0].mean(axis=0)) ** 2).sum(axis=0) / (n - 1)
        tw[e], mo[e] = (ms_b - ms_w) / epoch_bins, ms_w
    return tw, mo


def condition_component(R, conditions):
    """Within-session residuals R [trials x features] -> variance explained by a crossed task factor,
    df-corrected: (SS_between conditions - (k - 1) * MS_within) / n. A fixed effect, so it is a share of
    the trial level, not a separate level of the nesting."""
    n = len(R)
    A, codes = _indicator(conditions, n)
    cnt = np.asarray(A.sum(axis=1)).ravel()
    m = np.asarray(A @ R) / cnt[:, None]
    ss_b = (cnt[:, None] * (m - R.mean(axis=0)) ** 2).sum(axis=0)
    ss_w = ((R - m[codes]) ** 2).sum(axis=0)
    k = len(cnt)
    return (ss_b - (k - 1) * ss_w / (n - k)) / n


# ------------------------------------------------------------ permutation nulls
def permute_within(labels_to_shuffle, block, rng):
    """Shuffle the unique values of `labels_to_shuffle` among themselves within each `block`,
    keeping every unit with its group -- e.g. sessions moved between mice of one lab."""
    df = pd.DataFrame({'u': labels_to_shuffle, 'b': block}).drop_duplicates('u')
    df['new'] = df.groupby('b')['u'].transform(lambda s: rng.permutation(s.to_numpy()))
    return df.set_index('u')['new']


# ------------------------------------------------------------- distances
def hamming_pairs(codes, i, j):
    """Fraction of the 40 bins whose syllable differs, for trial pairs (i, j)."""
    return (codes[i] != codes[j]).mean(axis=1)


def sample_pairs(meta, kind, n_pairs, rng):
    """Random trial pairs of one relatedness kind: 'same session' (any lag), 'same mouse'
    (different session), 'same lab' (different mouse), 'other lab'."""
    n = len(meta)
    sess, mouse, lab = (pd.factorize(meta[c])[0] for c in ('session', 'mouse_name', 'lab'))
    out_i, out_j = [], []
    need = n_pairs
    while need > 0:
        i, j = rng.integers(0, n, 4 * need), rng.integers(0, n, 4 * need)
        if kind == 'same session':
            # draw j inside i's session
            starts = pd.Series(np.arange(n)).groupby(sess).min().to_numpy()
            sizes = np.bincount(sess)
            j = starts[sess[i]] + rng.integers(0, sizes[sess[i]])
            ok = i != j
        elif kind == 'same mouse':
            ok = (mouse[i] == mouse[j]) & (sess[i] != sess[j])
        elif kind == 'same lab':
            ok = (lab[i] == lab[j]) & (mouse[i] != mouse[j])
        else:
            ok = lab[i] != lab[j]
        if kind in ('same mouse', 'same lab'):
            # uniform draws rarely land in the same mouse: draw j from i's mouse / lab directly
            grp = mouse if kind == 'same mouse' else lab
            order = np.argsort(grp, kind='stable')
            starts = np.searchsorted(grp[order], np.arange(grp.max() + 1))
            sizes = np.bincount(grp)
            j = order[starts[grp[i]] + rng.integers(0, sizes[grp[i]])]
            ok = (sess[i] != sess[j]) if kind == 'same mouse' else (mouse[i] != mouse[j])
        out_i.append(i[ok][:need]); out_j.append(j[ok][:need])
        need -= ok.sum()
    return np.concatenate(out_i)[:n_pairs], np.concatenate(out_j)[:n_pairs]
