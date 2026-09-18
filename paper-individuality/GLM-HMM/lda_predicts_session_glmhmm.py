"""
Do LDA coordinates predict the GLM-HMM parameters fit to choice behavior?

Two stages, at either of two levels (--level session | mouse).

STAGE 1 (--fit): fit K=2 GLM-HMMs and extract their parameters.
    --level session (default): one INDEPENDENT fit per session. This is the
        level the question is really about, and no existing file has it.
        engaged.py's model_single_mouse (which produced p_state1/p_state2 in
        merged_behavioral_and_states.pqt) and fit_and_save_all_k.py both pool
        all of a mouse's sessions into ONE joint EM fit with shared weights
        and a shared transition matrix, so their parameters differ between
        mice but are identical for every session of the same mouse - useless
        as a session-level target. Only compare_k2_k3_pilot.py fits sessions
        independently, and only 4 hand-picked ones.
    --level mouse: all of a mouse's sessions pooled into one fit (exactly
        engaged.py's design, with the stimulus z-scored across the mouse's
        pooled trials as process_bwm_mouse does), paired with that mouse's
        MEAN LDA coordinates. Use this if the per-session fits turn out too
        unstable to be worth regressing on: many more trials per fit, at the
        cost of ~4x fewer data points and no within-mouse variation at all.

    Both levels use the same covariates, hyperparameters and paper
    initialization as the rest of this directory, fit by session_glmhmm_em.py
    rather than `ssm`, which is not installed on this machine (see that
    module's docstring; --validate checks the two against each other).

STAGE 2 (--predict): regress each fitted parameter on the first n LDA
    components, n = 1, 2, 3, ..., with cross-validated R2.

    At session level two things make the naive version misleading, so both
    are reported side by side:
      - The LDA was fit to discriminate MICE, so a session's LDA coordinates
        are nearly constant within an animal. A random CV split therefore
        lets the model recognize the mouse and recall its other sessions'
        parameters. The headline number is GroupKFold with whole mice held
        out; the random split is printed next to it as the inflated
        comparison.
      - Sessions are not independent, so the permutation null shuffles whole
        mouse blocks, preserving within-mouse clumping of the target while
        breaking its link to the LDA.
    A within-mouse-centered variant asks whether the residual session-to-
    session LDA variation predicts anything once each mouse's mean is
    removed. The LDA has little variance left there by construction, so a
    null result in that variant is weak evidence while a positive one is
    strong. At mouse level none of this applies: units are independent, the
    CV is a plain KFold and the null is a plain shuffle.

--validate: cross-checks session_glmhmm_em.py against the stored ssm output
    by refitting whole mice the way engaged.py did and correlating the
    posteriors with the p_state1 column those ssm fits wrote.

Usage (iblenv):
    python lda_predicts_session_glmhmm.py --validate
    python lda_predicts_session_glmhmm.py --fit --predict                # session level
    python lda_predicts_session_glmhmm.py --fit --predict --level mouse
"""
import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold

from session_glmhmm_em import fit_glmhmm_multistart, INPUT_NAMES

GLM_HMM_DIR = Path(__file__).resolve().parent
PREFIX = GLM_HMM_DIR.parent
STATES_PATH = GLM_HMM_DIR / 'merged_behavioral_and_states.pqt'
LDA_PATH = PREFIX / 'clustering' / 'data_files' / 'mouse_LDA_5_bins_cut19-08-2026'
OUT_DIR = GLM_HMM_DIR / 'lda_vs_session_params'
OUT_DIR.mkdir(exist_ok=True)

NUM_STATES = 2
N_RESTARTS = 3          # restart 0 is the deterministic paper init
MIN_TRIALS = 200
RIDGE_ALPHAS = np.logspace(-3, 4, 25)
N_FOLDS = 5
N_PERM = 200
MAX_COMPONENTS = 12     # the LDA file has 24; sweep the leading ones

# Parameters extracted in stage 1 and predicted in stage 2.
TARGETS = (
    [f'w_engaged_{n}' for n in INPUT_NAMES]
    + [f'w_disengaged_{n}' for n in INPUT_NAMES]
    + ['p_stay_engaged', 'p_stay_disengaged',
       'frac_engaged', 'n_transitions_per_trial', 'mean_dwell_engaged',
       'mean_dwell_disengaged', 'mean_max_posterior', 'bits_per_trial']
)


def paths_for(level):
    return (OUT_DIR / f'{level}_glmhmm_k2_params.csv',
            OUT_DIR / f'{level}_glmhmm_k2_posteriors.parquet',
            OUT_DIR / f'{level}_lda_prediction_results.csv',
            OUT_DIR / f'{level}_lda_univariate_correlations.csv')


# ----------------------------------------------------------------- data ----

def load_data():
    """Behavior + LDA merged on session id, restricted to sessions present in
    both. Returns (states_df, lda_df, lda_component_names)."""
    states_df = pd.read_parquet(STATES_PATH)
    lda = pd.read_pickle(LDA_PATH)
    comp_cols = sorted([c for c in lda.columns if isinstance(c, (int, np.integer))])
    lda = lda.rename(columns={c: f'lda_{c + 1}' for c in comp_cols})
    lda_names = [f'lda_{c + 1}' for c in comp_cols]
    lda = lda[['session', 'mouse_name'] + lda_names].copy()
    lda = lda[lda['session'].isin(set(states_df['eid'].unique()))].reset_index(drop=True)
    return states_df, lda, lda_names


def build_session_covariates(session_df, stim_stats=None):
    """(X, y) for one session, following compare_k2_k3_pilot.build_session_inputs
    and loso_k2_k3_sweep.build_session_covariates: stim = contrastRight -
    contrastLeft, prev_choice and wsls in {-1, +1}, plus a constant bias column.

    stim_stats: (mean, std) to z-score the stimulus with. None z-scores within
    the session, which is what a stand-alone session fit must do; the mouse-level
    fit passes the mouse's pooled stats, as process_bwm_mouse does.

    CAVEAT: this file has no choice column, so choice is reconstructed from
    (which side had contrast, rewarded). On zero-contrast trials reward is
    delivered at random, so the reconstructed choice there is right only about
    half the time (~12% of trials in this file). Every script in this directory
    inherits that same limitation.
    """
    contrast_left = session_df['contrastLeft'].fillna(0).values
    contrast_right = session_df['contrastRight'].fillna(0).values
    stim = contrast_right - contrast_left
    mu, sd = (stim.mean(), stim.std()) if stim_stats is None else stim_stats
    stim = (stim - mu) / sd

    right_correct = session_df['contrastLeft'].isna() & (session_df['rewarded'] == 1)
    right_incorrect = session_df['contrastRight'].isna() & (session_df['rewarded'] == -1)
    choice_right = (right_correct | right_incorrect).astype(int).values

    prev_choice = np.hstack([choice_right[0], choice_right[:-1]])
    prev_choice_bin = 2 * prev_choice - 1

    reward = session_df['rewarded'].values
    prev_reward = np.hstack([reward[0], reward[:-1]])
    wsls = (prev_reward * prev_choice_bin).astype(float)
    wsls[wsls == 0] = -1

    T = len(session_df)
    X = np.column_stack([stim, prev_choice_bin, wsls, np.ones(T)])
    return X, choice_right.astype(float)


def raw_signed_contrast(session_df):
    return (session_df['contrastRight'].fillna(0).values
            - session_df['contrastLeft'].fillna(0).values)


def build_unit_covariates(states_df, eids, pool_zscore):
    """Covariates for one fitting unit (a list of sessions). pool_zscore
    z-scores the stimulus across all of the unit's trials at once."""
    frames = [states_df[states_df['eid'] == e].reset_index(drop=True) for e in eids]
    stats = None
    if pool_zscore and len(frames) > 1:
        pooled = np.concatenate([raw_signed_contrast(f) for f in frames])
        stats = (pooled.mean(), pooled.std())
    built = [build_session_covariates(f, stim_stats=stats) for f in frames]
    return [b[0] for b in built], [b[1] for b in built]


# ------------------------------------------------------- stage 1: fitting ----

def bits_per_trial(ll_model, y):
    """In-sample bits/trial over a constant-P(right) null, as elsewhere here.
    In-sample: a K=2 fit can never do worse than the null, so this measures
    description quality, not generalization."""
    p = np.clip(y.mean(), 1e-9, 1 - 1e-9)
    ll_null = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    return (ll_model - ll_null) / len(y) / np.log(2)


def order_states(W):
    """K=2 state labels are arbitrary per fit, so align them before comparing
    units: 'engaged' is the state with the larger |stimulus weight|, i.e. the
    one whose choices track the stimulus. Returns [engaged_idx, other_idx]."""
    eng = int(np.argmax(np.abs(W[:, 0])))
    return [eng, 1 - eng]


def mean_dwell(map_states, k):
    """Mean run length (trials) of state k, np.nan if never occupied. Runs are
    counted within a sequence so session boundaries never merge two runs."""
    runs, cur = [], 0
    for s in map_states:
        if s == k:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def extract_params(fit, ys):
    """Parameter row for one fitted unit, with states relabeled engaged-first."""
    eng, dis = order_states(fit['W'])
    posts = [p[:, [eng, dis]] for p in fit['posteriors']]
    trans = fit['trans'][np.ix_([eng, dis], [eng, dis])]

    post_all = np.concatenate(posts)
    y_all = np.concatenate(ys)
    n_trans = sum(int(np.sum(np.diff(p.argmax(axis=1)) != 0)) for p in posts)
    runs_eng = [r for p in posts for r in mean_dwell(p.argmax(axis=1), 0)]
    runs_dis = [r for p in posts for r in mean_dwell(p.argmax(axis=1), 1)]

    row = {
        'n_trials': len(y_all), 'n_sessions': len(posts),
        'converged': fit['converged'], 'n_iter': fit['n_iter'],
        'loglik': fit['loglik'], 'bits_per_trial': bits_per_trial(fit['loglik'], y_all),
        'p_stay_engaged': trans[0, 0], 'p_stay_disengaged': trans[1, 1],
        'frac_engaged': float(post_all[:, 0].mean()),
        'n_transitions_per_trial': n_trans / len(y_all),
        'mean_dwell_engaged': float(np.mean(runs_eng)) if runs_eng else np.nan,
        'mean_dwell_disengaged': float(np.mean(runs_dis)) if runs_dis else np.nan,
        'mean_max_posterior': float(post_all.max(axis=1).mean()),
        # A state that is essentially unoccupied has an ill-determined weight
        # vector; flagged rather than dropped at fit time.
        'degenerate': bool(min(post_all[:, 0].mean(), post_all[:, 1].mean()) < 0.02),
    }
    for j, name in enumerate(INPUT_NAMES):
        row[f'w_engaged_{name}'] = fit['W'][eng, j]
        row[f'w_disengaged_{name}'] = fit['W'][dis, j]
    return row, posts


def fit_all(level='session', max_units=None, save_posteriors=True):
    states_df, lda, lda_names = load_data()
    params_path, posteriors_path, _, _ = paths_for(level)

    if level == 'session':
        units = [(row['session'], row['mouse_name'], [row['session']])
                 for _, row in lda.iterrows()]
        pool_zscore = False
        lda_by_unit = lda.set_index('session')[lda_names]
    else:
        grouped = lda.groupby('mouse_name')
        units = [(m, m, g['session'].tolist()) for m, g in grouped]
        pool_zscore = True
        lda_by_unit = grouped[lda_names].mean()   # mouse-mean LDA coordinates
    if max_units is not None:
        units = units[:max_units]

    print(f"Fitting K={NUM_STATES} GLM-HMM at {level} level: {len(units)} units, "
          f"{N_RESTARTS} restart(s) each\n")

    rows, posterior_frames, t0 = [], [], time.time()
    for i, (unit_id, mouse, eids) in enumerate(units):
        inputs, datas = build_unit_covariates(states_df, eids, pool_zscore)
        n_trials = sum(len(y) for y in datas)
        if n_trials < MIN_TRIALS:
            print(f"  skip {unit_id}: only {n_trials} trials", flush=True)
            continue
        fit = fit_glmhmm_multistart(inputs, datas, num_states=NUM_STATES,
                                    n_restarts=N_RESTARTS)
        row, posts = extract_params(fit, datas)
        row.update({'unit_id': unit_id, 'mouse_name': mouse})
        row.update(lda_by_unit.loc[unit_id].to_dict())
        rows.append(row)

        if save_posteriors:
            for eid, p in zip(eids, posts):
                posterior_frames.append(pd.DataFrame({
                    'unit_id': unit_id, 'eid': eid, 'trial_idx': np.arange(len(p)),
                    'p_engaged': p[:, 0]}))

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(units)}] {(time.time() - t0) / 60:.1f} min", flush=True)

    params = pd.DataFrame(rows)
    lead = ['unit_id', 'mouse_name', 'n_sessions', 'n_trials', 'converged', 'degenerate']
    params = params[lead + [c for c in params.columns if c not in lead]]
    params.to_csv(params_path, index=False)
    print(f"\nSaved {len(params)} fits to {params_path}")
    print(f"  non-converged: {(~params['converged']).sum()} | "
          f"degenerate (a state <2% occupied): {params['degenerate'].sum()}")
    print(f"  frac_engaged: median {params['frac_engaged'].median():.3f} "
          f"[{params['frac_engaged'].quantile(.05):.3f}, {params['frac_engaged'].quantile(.95):.3f}]")
    if save_posteriors:
        pd.concat(posterior_frames, ignore_index=True).to_parquet(posteriors_path, index=False)
        print(f"Saved posteriors to {posteriors_path}")
    return params


# ----------------------------------------------------- stage 2: prediction ----

def _cv_r2(X, y, groups, cv):
    """Out-of-fold R2 of a standardized ridge, with alpha chosen inside each
    training fold by RidgeCV's internal LOO, so no leakage."""
    pred = np.empty_like(y)
    # KFold warns if handed groups, GroupKFold requires them.
    splits = cv.split(X, y, groups) if isinstance(cv, GroupKFold) else cv.split(X, y)
    for train, test in splits:
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def _mouse_block_permutation(groups, rng):
    """Permutation that keeps each mouse's sessions together as a block but
    reassigns the blocks, so the within-mouse structure of the target survives
    while its link to the LDA is broken. Blocks of unequal length are laid
    down in shuffled order and re-cut at the original boundaries."""
    blocks = [np.where(groups == g)[0] for g in pd.unique(groups)]
    pool = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
    perm, pos = np.empty(len(groups), dtype=int), 0
    for b in blocks:
        perm[b] = pool[pos:pos + len(b)]
        pos += len(b)
    return perm


def run_prediction(level='session', n_perm=N_PERM, max_components=MAX_COMPONENTS,
                   drop_degenerate=True, seed=0):
    params_path, _, predict_path, univariate_path = paths_for(level)
    params = pd.read_csv(params_path)
    lda_names = sorted([c for c in params.columns if c.startswith('lda_')],
                       key=lambda c: int(c.split('_')[1]))
    max_components = min(max_components, len(lda_names))

    if drop_degenerate:
        n_before = len(params)
        params = params[~params['degenerate'] & params['converged']].reset_index(drop=True)
        print(f"Using {len(params)}/{n_before} units (dropped non-converged / degenerate fits)")

    mice = params['mouse_name'].values
    n_mice = len(pd.unique(mice))
    by_mouse = level == 'session' and n_mice < len(params)
    print(f"{len(params)} units from {n_mice} mice, "
          f"{len(lda_names)} LDA components available")
    print(f"CV: {'GroupKFold, whole mice held out' if by_mouse else 'KFold over independent units'}"
          f" | null: {'mouse-block permutation' if by_mouse else 'plain shuffle'}\n")

    rng = np.random.RandomState(seed)
    main_cv = (GroupKFold(n_splits=min(N_FOLDS, n_mice)) if by_mouse
               else KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed))
    random_cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    targets = [t for t in TARGETS if t in params]
    centered = params.copy()
    if by_mouse:
        cols = lda_names + targets
        centered[cols] = params[cols] - params.groupby('mouse_name')[cols].transform('mean')

    rows = []
    for target in targets:
        y_full = params[target].values.astype(float)
        ok = np.isfinite(y_full)
        y, g = y_full[ok], mice[ok]
        yc = centered[target].values.astype(float)[ok]

        for n_comp in range(1, max_components + 1):
            cols = lda_names[:n_comp]
            X = params.loc[ok, cols].values.astype(float)
            r2_main = _cv_r2(X, y, g, main_cv)
            r2_random = _cv_r2(X, y, g, random_cv) if by_mouse else r2_main
            r2_within = (_cv_r2(centered.loc[ok, cols].values.astype(float), yc, g, main_cv)
                         if by_mouse else np.nan)

            null = np.empty(n_perm)
            for p in range(n_perm):
                perm = (_mouse_block_permutation(g, rng) if by_mouse
                        else rng.permutation(len(y)))
                null[p] = _cv_r2(X, y[perm], g, main_cv)
            pval = (np.sum(null >= r2_main) + 1) / (n_perm + 1)

            rows.append(dict(target=target, n_components=n_comp,
                             r2_heldout=r2_main, r2_random_split=r2_random,
                             r2_within_mouse=r2_within, perm_p=pval,
                             null_mean=null.mean(), null_p95=np.percentile(null, 95),
                             n_units=len(y), n_mice=len(pd.unique(g))))

        best = max([r for r in rows if r['target'] == target], key=lambda r: r['r2_heldout'])
        print(f"{target:26s} best n={best['n_components']:2d}  R2={best['r2_heldout']:+.3f}  "
              f"p={best['perm_p']:.3f}   (random split {best['r2_random_split']:+.3f}, "
              f"within-mouse {best['r2_within_mouse']:+.3f})", flush=True)

    results = pd.DataFrame(rows)
    results['perm_p_fdr'] = _bh_fdr(results['perm_p'].values)
    results.to_csv(predict_path, index=False)

    uni = univariate_correlations(params, lda_names, targets)
    uni.to_csv(univariate_path, index=False)
    print(f"\nSaved {predict_path}\nSaved {univariate_path}")

    sig = results[results['perm_p_fdr'] < 0.05].sort_values('r2_heldout', ascending=False)
    print(f"\n{len(sig)} of {len(results)} (target, n_components) pairs pass FDR<0.05"
          f" on the permutation null:")
    if len(sig):
        print(sig[['target', 'n_components', 'r2_heldout', 'perm_p', 'perm_p_fdr']]
              .head(20).to_string(index=False))
    return results, uni


def univariate_correlations(params, lda_names, targets):
    """Per-component Spearman correlation with each target: over units, and
    (when units are sessions) also over mouse means, which is the conservative
    version since sessions of one mouse are not independent."""
    rows = []
    mouse_means = params.groupby('mouse_name')[lda_names + targets].mean().reset_index()
    for target in targets:
        for comp in lda_names:
            ok = np.isfinite(params[target]) & np.isfinite(params[comp])
            r_u, p_u = spearmanr(params.loc[ok, comp], params.loc[ok, target])
            okm = np.isfinite(mouse_means[target]) & np.isfinite(mouse_means[comp])
            r_m, p_m = spearmanr(mouse_means.loc[okm, comp], mouse_means.loc[okm, target])
            rows.append(dict(target=target, component=comp,
                             rho_units=r_u, p_units=p_u, n_units=int(ok.sum()),
                             rho_mouse_means=r_m, p_mouse_means=p_m, n_mice=int(okm.sum())))
    uni = pd.DataFrame(rows)
    uni['p_mouse_means_fdr'] = _bh_fdr(uni['p_mouse_means'].values)
    return uni


def _bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ------------------------------------------------------------ reliability ----

def reliability(max_units=None):
    """How much of a per-session parameter is signal rather than fit noise.

    Splits each session in half by trial and fits the two halves as two
    independent K=2 GLM-HMMs, then correlates each parameter's first-half
    value against its second-half value across sessions. Both halves come
    from the same session, so a low correlation means the parameter is not
    even reproducible within a session and cannot be predicted by anything.

    Reported next to the ICC (between-mouse share of the variance), which is
    the separate ceiling that applies to a MOUSE-level predictor such as the
    LDA: a target with high reliability but ICC near zero varies session to
    session for real, but not in a way any mouse-level feature can track.
    """
    states_df, lda, _ = load_data()
    sessions = lda['session'].tolist()[:max_units]
    param_cols = [t for t in TARGETS]

    rows_a, rows_b, mice = [], [], []
    t0 = time.time()
    for i, eid in enumerate(sessions):
        sdf = states_df[states_df['eid'] == eid].reset_index(drop=True)
        if len(sdf) < 2 * MIN_TRIALS:
            continue
        half = len(sdf) // 2
        out = []
        for part in (sdf.iloc[:half], sdf.iloc[half:].reset_index(drop=True)):
            X, y = build_session_covariates(part)
            fit = fit_glmhmm_multistart([X], [y], num_states=NUM_STATES, n_restarts=N_RESTARTS)
            row, _ = extract_params(fit, [y])
            out.append(row)
        if any(r['degenerate'] or not r['converged'] for r in out):
            continue
        rows_a.append(out[0])
        rows_b.append(out[1])
        mice.append(lda.loc[lda['session'] == eid, 'mouse_name'].iloc[0])
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(sessions)}] {(time.time() - t0) / 60:.1f} min", flush=True)

    a, b = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    full = pd.read_csv(paths_for('session')[0]) if paths_for('session')[0].exists() else None

    print(f"\nSplit-half reliability over {len(a)} sessions "
          f"(both halves converged and non-degenerate)\n")
    print(f"{'target':26s} {'r(half1,half2)':>15s} {'r_full':>9s} {'rho':>7s} {'ICC(mouse)':>11s}")
    out_rows = []
    for t in param_cols:
        ok = np.isfinite(a[t]) & np.isfinite(b[t])
        r = np.corrcoef(a.loc[ok, t], b.loc[ok, t])[0, 1] if ok.sum() > 2 else np.nan
        rho = spearmanr(a.loc[ok, t], b.loc[ok, t])[0] if ok.sum() > 2 else np.nan
        icc = _icc(full, t) if full is not None else np.nan
        # Spearman-Brown: each half has half the trials, so r understates the
        # reliability of the full-session estimate that stage 1 actually uses.
        r_full = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan
        print(f"{t:26s} {r:>15.3f} {r_full:>9.3f} {rho:>7.3f} {icc:>11.3f}")
        out_rows.append(dict(target=t, r_split_half=r, r_full_spearman_brown=r_full,
                             rho_split_half=rho, icc_between_mouse=icc,
                             n_sessions=int(ok.sum())))
    res = pd.DataFrame(out_rows)
    path = OUT_DIR / 'session_param_reliability.csv'
    res.to_csv(path, index=False)
    print(f"\nSaved {path}")
    return res


def _icc(params, target):
    """Between-mouse share of the variance of `target` across sessions - the
    ceiling on what any predictor constant within a mouse (like the LDA) can
    explain."""
    x = params[[target, 'mouse_name']].dropna()
    if len(x) < 3:
        return np.nan
    g = x.groupby('mouse_name')[target]
    n, m = g.size(), g.mean()
    k = n.mean()
    msb = float((n * (m - x[target].mean()) ** 2).sum() / (len(m) - 1))
    msw = float(((x[target] - x['mouse_name'].map(m)) ** 2).sum() / (len(x) - len(m)))
    return (msb - msw) / (msb + (k - 1) * msw)


# ------------------------------------------------------------- validation ----

def validate(n_mice=5):
    """Refit whole mice the way engaged.py did (sessions pooled, shared
    parameters, stimulus z-scored over the mouse's pooled trials) with this
    module's EM, and compare the posteriors against the p_state1 column the
    ssm fits wrote into merged_behavioral_and_states.pqt.

    Perfect agreement is not expected: engaged.py read choices straight from
    the ONE trials object and masked violation trials, whereas this file has
    no choice column so choices are reconstructed from reward and stimulus
    side (wrong on about half of the ~12% zero-contrast trials, and violation
    trials cannot be identified at all). A high correlation therefore says the
    EM is right; the residual gap is the input reconstruction, and it applies
    equally to compare_k2_k3_pilot.py and loso_k2_k3_sweep.py."""
    states_df, _, _ = load_data()
    counts = states_df.groupby('animal')['eid'].nunique().sort_values(ascending=False)

    print("Validating session_glmhmm_em.py against the stored ssm fits")
    print("(per-mouse pooled refit; correlation of p(state) across all trials)\n")
    for animal in counts.index[:n_mice]:
        adf = states_df[states_df['animal'] == animal]
        eids = list(pd.unique(adf['eid']))
        inputs, datas = build_unit_covariates(states_df, eids, pool_zscore=True)
        stored = np.concatenate([adf[adf['eid'] == e]['p_state1'].values for e in eids])
        fit = fit_glmhmm_multistart(inputs, datas, num_states=2, n_restarts=1)
        ours = np.concatenate([p[:, 0] for p in fit['posteriors']])
        r = max(np.corrcoef(ours, stored)[0, 1], np.corrcoef(1 - ours, stored)[0, 1])
        print(f"  {animal:14s} {len(eids):2d} sessions, {len(ours):5d} trials: r = {r:.4f}"
              f"  (converged={fit['converged']}, iters={fit['n_iter']})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true',
                    help='check the numpy EM against the stored ssm fits, then exit')
    ap.add_argument('--fit', action='store_true', help='stage 1: fit the GLM-HMMs')
    ap.add_argument('--reliability', action='store_true',
                    help='split-half reliability of the per-session parameters, then exit')
    ap.add_argument('--predict', action='store_true', help='stage 2: LDA -> parameters')
    ap.add_argument('--level', choices=('session', 'mouse'), default='session')
    ap.add_argument('--max-units', type=int, default=None, help='stage 1 smoke test')
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--max-components', type=int, default=MAX_COMPONENTS)
    ap.add_argument('--keep-degenerate', action='store_true',
                    help='stage 2: keep non-converged / single-state fits')
    args = ap.parse_args()

    if args.validate:
        validate()
        return
    if args.reliability:
        reliability(max_units=args.max_units)
        return
    if not (args.fit or args.predict):
        ap.error('pass --fit, --predict, --reliability, or --validate')
    if args.fit:
        fit_all(level=args.level, max_units=args.max_units)
    if args.predict:
        if not paths_for(args.level)[0].exists():
            sys.exit(f"{paths_for(args.level)[0]} not found - run --fit first")
        run_prediction(level=args.level, n_perm=args.n_perm,
                       max_components=args.max_components,
                       drop_degenerate=not args.keep_degenerate)


if __name__ == '__main__':
    main()
