"""Population-level encoding: region-average activity pooled across sessions and mice,
with the behavioural LDA axis entering the model as *interactions* on the encoding
kernels.

Difference from `encoding_functions.py`
---------------------------------------
`encoding_functions.fit_session` fits ONE SESSION at a time (targets = neurons, or
neuron-averaged areas) and LDA1 only enters afterwards, as a second-stage regression on
the fitted cvR2 (`lda1_effect`, `lda1_perm_bf`).

Here the model is fitted ONCE across all sessions of a single region:

    target  y_s(t)  = region-mean spike count of session s at bin t,
                      z-scored WITHIN session
    base    B_s(t)  = the usual Musall design (task kernels + motor states),
                      each column z-scored WITHIN session
    LDA     L_s(t)  = z_s * B_s(t)   -- the session's LDA score multiplying the
                      base columns, i.e. LDA1 rescales the encoding kernels

    y ~ B  (base model)        vs        y ~ B + L  (LDA model)

and the quantity of interest is the *cross-validated* gain

    dR2_LDA = cvR2(B + L) - cvR2(B),      folds held out by MOUSE.

Held-out mice (not bins, not sessions) is the only fold structure that makes the
question "does knowing a new animal's LDA position help predict its region activity"
answerable; holding out bins or sessions of a training mouse leaks identity.

Why interactions and not a main effect
--------------------------------------
z_s is one number per session, so a main-effect column is constant within a session.
The target is z-scored within session (mean 0), so a within-session constant predicts
nothing at all -- its least-squares weight is exactly 0. With within-session
normalisation LDA can *only* act by changing kernel gains, which is also the
scientifically interesting claim.

How it stays cheap
------------------
Everything is computed from per-session SUFFICIENT STATISTICS:

    A_s = B_s' B_s      b_s = B_s' y_s      syy_s = y_s' y_s      n_s

Because the LDA block is just the base block scaled by the scalar z_s, the Gram matrix
of the full design for ANY set of sessions and ANY assignment of LDA values is a
weighted sum of those per-session matrices with weights (1, z_s, z_s^2):

    F_s'F_s = [[A_s,        z_s A_s[:,I]     ],
               [z_s A_s[I,:], z_s^2 A_s[I,I] ]]        F_s'y_s = [b_s, z_s b_s[I]]

So the binned data is streamed once (one pass, cached to disk), and every fit, every
fold and every permutation afterwards is small linear algebra on ~p x p matrices. This
is what makes a 2000-permutation null affordable on ~7 million bins.

Null hypothesis
---------------
`perm_null` shuffles the session -> LDA assignment, which by construction leaves
everything *within* a session untouched (design, target, neuron pooling, bin count)
and breaks only the session<->LDA link. Two levels:

  level='session' : shuffle z across sessions.
  level='mouse'   : shuffle mouse-mean z across mice, all sessions of a mouse keeping
                    one value. This is the conservative one -- it matches the
                    held-out-mouse CV, and it cannot be beaten by within-mouse
                    session structure. Report both.
"""
import os
import glob
import pickle

import numpy as np
import pandas as pd

import encoding_functions as ef

# Allen/Beryl labels that are anatomically unusable as a "region": white matter,
# ventricles, and the coarse parents that Beryl assigns to poorly-localised units.
COARSE_LABELS = {
    'root', 'void', 'arb', 'fiber tracts', 'or', 'alv', 'ccb', 'fp', 'scwm', 'cing',
    'em', 'ar', 'bic', 'dhc', 'fx', 'mlf', 'cst', 'py', 'ml', 'sup', 'lfbs', 'opt',
    'st', 'amc', 'ipf', 'tspc', 'tb', 'VS', 'das', 'll', 'isl', 'ec', 'int', 'ccg',
    'cc', 'fa', 'ee', 'sm', 'onl', 'bsc', 'grey', 'CTX', 'HPF', 'MB', 'TH', 'HY',
    'P', 'MY', 'CB', 'STR', 'OLF', 'IB',
}

ALPHAS = np.logspace(-4, 2, 13)      # ridge penalty, in PER-SAMPLE units (see _fit)


# ---------------------------------------------------------------------------
# Anatomy
# ---------------------------------------------------------------------------
def beryl_map(acronyms, br=None):
    """{raw Allen acronym -> Beryl acronym} for the acronyms stored in the neuron-file
    column names. Unmappable labels map to themselves."""
    from iblatlas.atlas import BrainRegions
    br = br or BrainRegions()
    out = {}
    for a in acronyms:
        try:
            ids = br.remap(br.acronym2id(a), source_map='Allen', target_map='Beryl')
            out[a] = br.id2acronym(ids)[0]
        except Exception:
            out[a] = a
    return out


def region_coverage(results_dir='encoding_results', lda=None, min_neurons=(5, 10, 20),
                    drop_coarse=True):
    """How many sessions / mice have how many neurons in each Beryl region.

    Read off the cached per-neuron encoding results (one parquet per pid), which
    already carry area / session / mouse_name -- much cheaper than re-opening every
    binned neuron file. Probes are pooled per session (a session with two probes in
    the region contributes one row with the summed neuron count).

    `lda` : the LDA table; if given, sessions without an LDA score are excluded, so the
    counts are the ones that actually matter for this analysis.

    Also reports `yield_rho`: Spearman correlation between neurons-per-session and
    LDA1. A region where yield tracks LDA1 (e.g. CP, VPM) is a trap -- the number of
    neurons entering the average would itself be a function of the predictor.
    """
    from scipy.stats import spearmanr

    files = sorted(glob.glob(os.path.join(results_dir, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'no cached results in {results_dir}/')
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d['beryl'] = d['area'].astype(str).map(beryl_map(d['area'].astype(str).unique()))

    if lda is not None:
        lda1 = _lda1(lda)
        d = d[d['session'].isin(set(lda1['session']))]
    else:
        lda1 = None

    per = (d.groupby(['beryl', 'session', 'mouse_name']).size()
             .rename('n_neu').reset_index())
    rows = []
    for region, g in per.groupby('beryl'):
        r = dict(region=region, coarse=region in COARSE_LABELS, sessions=len(g),
                 mice=g['mouse_name'].nunique(), median_n=int(g['n_neu'].median()))
        for k in min_neurons:
            ok = g[g['n_neu'] >= k]
            r[f'sess_ge{k}'] = len(ok)
            r[f'mice_ge{k}'] = ok['mouse_name'].nunique()
        if lda1 is not None and len(g) > 4:
            m = g.merge(lda1, on='session')
            rho, p = spearmanr(m['n_neu'], m['lda_1'])
            r['yield_rho'], r['yield_p'] = round(float(rho), 2), round(float(p), 3)
        rows.append(r)
    t = pd.DataFrame(rows)
    if drop_coarse:
        t = t[~t['coarse']].drop(columns='coarse')
    key = f'mice_ge{min_neurons[1]}' if len(min_neurons) > 1 else 'mice'
    return t.sort_values([key, 'sessions'], ascending=False).reset_index(drop=True)


def _lda1(lda, component=0):
    """Session-level LDA score for one component (the raw table names them 0..23)."""
    col = component if component in lda.columns else f'lda_{component + 1}'
    return (lda[['session', col]].rename(columns={col: 'lda_1'})
               .dropna().drop_duplicates('session').reset_index(drop=True))


# ---------------------------------------------------------------------------
# One session -> sufficient statistics
# ---------------------------------------------------------------------------
def _region_spike_cols(df, region, bmap=None):
    cols = [c for c in df.columns if c.endswith('_spike_count')]
    raw = [c.split('_neuron_')[0] for c in cols]
    bmap = bmap or beryl_map(sorted(set(raw)))
    return [c for c, r in zip(cols, raw) if bmap.get(r) == region]


def load_session(pids, neuron_dir, region, bmap=None):
    """Load one session's binned neuron file(s) and pool the region's neurons across
    probes. Returns (df, region_spike_cols).

    The design matrix comes from the first (largest) file; extra probes contribute only
    their region spike columns, inner-joined on `Bin`. If the bin grids do not line up
    the extra probe is dropped (and reported), rather than silently mis-aligned.
    """
    dfs = []
    for pid in pids:
        path = os.path.join(neuron_dir, f'states_neurons_file_{pid}')
        with open(path, 'rb') as f:
            dfs.append((pid, pickle.load(f)))
    dfs.sort(key=lambda t: -len(_region_spike_cols(t[1], region, bmap)))

    base_pid, df = dfs[0]
    df = df.reset_index(drop=True)
    cols = _region_spike_cols(df, region, bmap)
    for pid, other in dfs[1:]:
        ocols = _region_spike_cols(other, region, bmap)
        if not ocols:
            continue
        # neuron ids restart per probe, so the column names collide -- tag them
        ren = {c: f'{c}__{pid[:8]}' for c in ocols}
        add = other[['Bin'] + ocols].drop_duplicates('Bin').rename(columns=ren)
        new = list(ren.values())
        merged = df.merge(add, on='Bin', how='left')
        if len(merged) != len(df) or merged[new].isna().all(axis=None):
            print(f'    ! probe {pid[:8]} bin grid does not match {base_pid[:8]} — dropped')
            continue
        df = merged
        cols += new
    return df, cols


def _zscore(A):
    """Column-wise z-score; constant columns become exactly 0 (they carry no
    information and would otherwise blow up on 0/0)."""
    A = np.asarray(A, dtype=np.float64)
    sd = A.std(axis=0, ddof=1)
    ok = sd > 1e-12
    out = np.zeros_like(A)
    out[:, ok] = (A[:, ok] - A[:, ok].mean(axis=0)) / sd[ok]
    return out


def _rebin_blocks(keep, k):
    """Row groupings for averaging k consecutive bins.

    The peri-trial mask leaves gaps, so blocks are formed only *within* runs of
    contiguous kept bins and any leftover partial block is dropped -- averaging across a
    gap would mix bins that are seconds apart.
    Returns a list of (start, stop) slices into the kept-row array.
    """
    pos = np.flatnonzero(keep)
    if k <= 1:
        return None
    breaks = np.flatnonzero(np.diff(pos) != 1) + 1
    runs = np.split(np.arange(len(pos)), breaks)
    sl = []
    for r in runs:
        m = (len(r) // k) * k
        for j in range(0, m, k):
            sl.append((r[j], r[j] + k))
    return sl


def _apply_rebin(M, sl):
    """Average the rows of M within each block. M may be 1-D or 2-D."""
    if sl is None:
        return M
    out = np.empty((len(sl),) + M.shape[1:], dtype=float)
    for i, (a, b) in enumerate(sl):
        out[i] = M[a:b].mean(axis=0)
    return out


def session_stats(df, region_cols, motor_continuous=False, motor_lags=True,
                  continuous_features='speed', rebin=3):
    """Sufficient statistics for one session.

    y = region-mean spike count over the kept bins, z-scored within session.
    B = the Musall design matrix for this session, each column z-scored within session.

    `rebin` : average this many consecutive 16.7 ms bins before fitting (3 -> 50 ms).
    At 60 Hz the mean spike count of ~20 neurons is dominated by Poisson noise, which
    caps the achievable R2 for every session equally: it does not bias dR2, but it costs
    a lot of power. Averaging within runs of contiguous kept bins cuts that white noise
    by sqrt(rebin) while leaving the kernels (which are all >=0.3 s wide) intact.
    Set rebin=1 for the raw 60 Hz fit.

    Returns dict(cols, col_group, A, b, syy, n) where A = B'B and b = B'y, both in this
    session's own column space (mapped into the union space later by `assemble`).
    """
    X, groups, trial_ids, keep = ef.build_design_matrix(
        df, motor_continuous=motor_continuous, motor_lags=motor_lags,
        continuous_features=continuous_features)
    y = df.loc[keep, region_cols].astype(float).mean(axis=1).values
    if not np.isfinite(y).all():
        raise ValueError('non-finite region average (empty neuron set in some bins)')

    sl = _rebin_blocks(keep, rebin)
    y = _apply_rebin(y, sl)
    B = _zscore(_apply_rebin(X.values.astype(float), sl))
    yz = (y - y.mean()) / y.std(ddof=1)
    col_group = {c: g for g, cs in groups.items() for c in cs}
    return dict(cols=list(X.columns),
                col_group=[col_group.get(c, 'other') for c in X.columns],
                A=B.T @ B, b=B.T @ yz, syy=float(yz @ yz), n=int(len(yz)),
                rebin=rebin, n_neurons=len(region_cols),
                n_trials=int(pd.unique(trial_ids).size))


def sweep_sessions(sessions, neuron_dir, region, cache_dir, motor_continuous=False,
                   motor_lags=True, min_neurons=10, rebin=3, verbose=True):
    """Stream every session once, caching one pickle of sufficient statistics per
    session (resumable: delete the cache dir to recompute).

    `sessions` : DataFrame with columns session, mouse_name, pid (one row per probe).
    Sessions with fewer than `min_neurons` neurons in the region are skipped, because a
    population average over a handful of neurons is mostly single-neuron noise.
    """
    os.makedirs(cache_dir, exist_ok=True)
    bmap = None
    out, skipped = [], []
    grp = sessions.groupby('session')
    for i, (eid, g) in enumerate(grp, 1):
        cache = os.path.join(cache_dir, f'{eid}.pkl')
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                out.append(pickle.load(f))
            continue
        try:
            df, cols = load_session(list(g['pid']), neuron_dir, region, bmap)
            if bmap is None:                       # build the atlas map once, then reuse
                raw = [c.split('_neuron_')[0] for c in df.columns if c.endswith('_spike_count')]
                bmap = beryl_map(sorted(set(raw)))
            if len(cols) < min_neurons:
                skipped.append((eid, len(cols)))
                continue
            st = session_stats(df, cols, motor_continuous=motor_continuous,
                               motor_lags=motor_lags, rebin=rebin)
            st.update(session=eid, mouse_name=g['mouse_name'].iloc[0],
                      pids=list(g['pid']))
            with open(cache, 'wb') as f:
                pickle.dump(st, f)
            out.append(st)
            if verbose:
                print(f'[{i}/{len(grp)}] {eid[:8]}  {st["n_neurons"]:3d} neurons  '
                      f'{st["n"]:7d} bins  {len(st["cols"]):3d} cols')
        except Exception as e:
            print(f'[{i}/{len(grp)}] {eid[:8]} FAILED: {type(e).__name__}: {e}')
    if skipped and verbose:
        print(f'skipped {len(skipped)} sessions with <{min_neurons} neurons in {region}')
    return out


# ---------------------------------------------------------------------------
# Union column space
# ---------------------------------------------------------------------------
def assemble(stats):
    """Map per-session sufficient statistics into a shared column space.

    Sessions can differ in which columns exist (a session with 3 whisker HMM states has
    more one-hot columns than one with 2). Missing columns are simply absent from that
    session's A/b -- i.e. treated as the all-zero column, which is what they are.
    """
    cols, group_of = [], {}
    for st in stats:
        for c, g in zip(st['cols'], st['col_group']):
            if c not in group_of:
                group_of[c] = g
                cols.append(c)
    pos = {c: i for i, c in enumerate(cols)}
    p = len(cols)

    A = np.zeros((len(stats), p, p))
    b = np.zeros((len(stats), p))
    for k, st in enumerate(stats):
        idx = np.array([pos[c] for c in st['cols']])
        A[k][np.ix_(idx, idx)] = st['A']
        b[k][idx] = st['b']
    return dict(cols=cols, groups=np.array([group_of[c] for c in cols]),
                A=A, b=b,
                syy=np.array([st['syy'] for st in stats], float),
                n=np.array([st['n'] for st in stats], int),
                session=np.array([st['session'] for st in stats]),
                mouse=np.array([st['mouse_name'] for st in stats]),
                n_neurons=np.array([st['n_neurons'] for st in stats], int))


def interaction_index(S, which='all'):
    """Column indices of the base block that get multiplied by the LDA score.

    'all'  -> every base column
    'task' -> only the task event kernels
    'motor'-> only the motor-state columns
    """
    if which == 'all':
        return np.arange(len(S['cols']))
    key = {'task': 'task', 'motor': 'motor_states',
           'motor_continuous': 'motor_continuous'}[which]
    return np.flatnonzero(S['groups'] == key)


# ---------------------------------------------------------------------------
# Fitting from sufficient statistics
# ---------------------------------------------------------------------------
def _as_Z(z):
    """Accept a 1-D vector (one component) or an (n_sessions, K) matrix (several)."""
    Z = np.asarray(z, dtype=float)
    return Z[:, None] if Z.ndim == 1 else Z


def _gram(S, idx, z, I):
    """Pooled Gram matrix / cross-product for the sessions in `idx`, for the design
    [B, z_1*B[:,I], ..., z_K*B[:,I]]. Exact -- no data touched, only the cached
    per-session A and b.

    With K LDA components the cross-blocks between interaction sets a and b are
    z_a * z_b * A[I,I], so the whole (p + K*q) Gram is still a weighted sum of the
    per-session matrices. That is what keeps a joint multi-component test as cheap as
    the single-component one.
    """
    Z = _as_Z(z)
    p = S['A'].shape[1]
    q = len(I)
    K = Z.shape[1] if q else 0
    P = p + K * q
    G = np.zeros((P, P))
    g = np.zeros(P)
    for k in idx:
        A, bb = S['A'][k], S['b'][k]
        G[:p, :p] += A
        g[:p] += bb
        if not q:
            continue
        zs = Z[k]
        AI = A[:, I]                       # p x q
        AII = AI[I, :]                     # q x q
        for a in range(K):
            sa = p + a * q
            G[:p, sa:sa + q] += zs[a] * AI
            g[sa:sa + q] += zs[a] * bb[I]
            for b in range(K):
                sb = p + b * q
                G[sa:sa + q, sb:sb + q] += (zs[a] * zs[b]) * AII
    if q:
        G[p:, :p] = G[:p, p:].T
    return G, g, int(S['n'][idx].sum()), float(S['syy'][idx].sum())


def _fit(G, g, n, alpha):
    """Ridge weights. G/n and g/n put `alpha` in per-sample units, so the same alpha
    means the same amount of shrinkage regardless of how many bins a fold happens to
    contain (columns are z-scored, so the normalised Gram has ~unit diagonal)."""
    p = len(g)
    return np.linalg.solve(G / n + alpha * np.eye(p), g / n)


def _sse(G, g, syy, w):
    return float(syy - 2.0 * (w @ g) + w @ G @ w)


def mouse_folds(S, n_splits=5, seed=0):
    """GroupKFold over sessions, grouped by mouse: every session of a mouse is in the
    same fold, so a held-out fold contains only unseen animals."""
    from sklearn.model_selection import GroupKFold
    idx = np.arange(len(S['session']))
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(S['mouse']))))
    return [(idx[tr], idx[te]) for tr, te in gkf.split(idx, groups=S['mouse'])]


def cv_r2(S, z, I, folds, alphas=ALPHAS, inner_splits=3, fixed_alphas=None):
    """Cross-validated R2 of [B, z*B[:,I]] on held-out mice.

    R2 is against the session mean (the target is z-scored within session, so
    predicting 0 = predicting each session's own mean). Pooled over folds as
    1 - sum(SSE)/sum(SST), plus the per-fold values.

    `fixed_alphas`: skip the nested alpha search and reuse these (one per fold) --
    used by the permutation null, where re-selecting alpha on every shuffle would be
    both slow and circular.
    """
    r2_fold, alpha_fold, sse_tot, sst_tot = [], [], 0.0, 0.0
    for f, (tr, te) in enumerate(folds):
        if fixed_alphas is not None:
            alpha = fixed_alphas[f]
        else:
            inner = _inner_folds(S, tr, inner_splits)
            best = (-np.inf, alphas[0])
            for a in alphas:
                sse, sst = 0.0, 0.0
                for itr, ite in inner:
                    w = _fit(*_gram(S, itr, z, I)[:3], a)
                    G, g, _, syy = _gram(S, ite, z, I)
                    sse += _sse(G, g, syy, w)
                    sst += syy
                score = 1 - sse / sst
                if score > best[0]:
                    best = (score, a)
            alpha = best[1]
        w = _fit(*_gram(S, tr, z, I)[:3], alpha)
        G, g, _, syy = _gram(S, te, z, I)
        sse = _sse(G, g, syy, w)
        r2_fold.append(1 - sse / syy)
        alpha_fold.append(alpha)
        sse_tot += sse
        sst_tot += syy
    return dict(cv_r2=1 - sse_tot / sst_tot, r2_folds=np.array(r2_fold),
                alphas=alpha_fold)


def _inner_folds(S, tr, n_splits):
    from sklearn.model_selection import GroupKFold
    mice = S['mouse'][tr]
    k = min(n_splits, len(np.unique(mice)))
    if k < 2:
        return [(tr, tr)]
    gkf = GroupKFold(n_splits=k)
    return [(tr[a], tr[b]) for a, b in gkf.split(tr, groups=mice)]


def delta_r2(S, z, folds, which='all', alphas=ALPHAS, base=None):
    """cvR2 with and without the LDA interaction block, and their difference."""
    base = base or cv_r2(S, z, np.array([], int), folds, alphas)
    I = interaction_index(S, which)
    full = cv_r2(S, z, I, folds, alphas)
    return dict(which=which, n_interactions=len(I),
                cv_r2_base=base['cv_r2'], cv_r2_full=full['cv_r2'],
                dR2=full['cv_r2'] - base['cv_r2'],
                base=base, full=full)


# ---------------------------------------------------------------------------
# Permutation null on the session -> LDA assignment
# ---------------------------------------------------------------------------
def lda_matrix(S, lda, components=(0,), level='session'):
    """(n_sessions, K) matrix of z-scored LDA scores, aligned to S.

    Each component is z-scored separately across sessions. Passing several components
    means they enter the model TOGETHER as one interaction block -- a single joint test
    ("do these K axes, as a set, change how the region encodes task/behaviour"), with no
    multiplicity to correct. That is more powerful than K separate tests when more than
    one component contributes, and less powerful when only one does.

    Each component costs another `len(I)` columns (78 for `which='all'`), so the null
    mean drops further below zero with every one added -- keep K small.
    """
    cols = []
    for k in components:
        cols.append(lda_vector(S, _lda1(lda, k), level=level))
    return np.column_stack(cols)


def lda_vector(S, lda1, level='session'):
    """z-scored LDA score per session, aligned to S. level='mouse' replaces each
    session's score by its mouse's mean, so the predictor carries only between-animal
    information (matches the held-out-mouse CV and the mouse-level null)."""
    m = pd.DataFrame(dict(session=S['session'], mouse=S['mouse'])).merge(
        lda1, on='session', how='left')
    if m['lda_1'].isna().any():
        raise ValueError('some sessions have no LDA score')
    v = m['lda_1'].values.astype(float)
    if level == 'mouse':
        v = m.groupby('mouse')['lda_1'].transform('mean').values.astype(float)
    return (v - v.mean()) / v.std(ddof=1)


def perm_null(S, z, folds, which='all', level='session', n_perm=2000, seed=0,
              observed=None, alphas=ALPHAS):
    """Null distribution of dR2 under shuffling of the session -> LDA assignment.

    level='session' : permute z across sessions.
    level='mouse'   : permute the per-mouse value across mice, keeping all sessions of
                      a mouse together (conservative; use this as the headline test).

    Everything inside a session -- design, target, neuron pooling, bin count -- is
    untouched by construction, so the only thing destroyed is the link being tested.
    The base model does not involve z at all, so its cvR2 is computed once and reused;
    fold alphas are frozen at the observed-data values.
    """
    obs = observed or delta_r2(S, z, folds, which, alphas)
    base_cv = obs['base']['cv_r2']
    fixed = obs['full']['alphas']
    I = interaction_index(S, which)
    rng = np.random.default_rng(seed)

    Z = _as_Z(z)
    if level == 'mouse':
        mice, inv = np.unique(S['mouse'], return_inverse=True)
        per_mouse = np.stack([Z[S['mouse'] == m][0] for m in mice])

    null = np.empty(n_perm)
    for i in range(n_perm):
        # permute whole session (or mouse) ROWS, so the correlation structure between
        # components is preserved and only the session<->LDA link is broken
        if level == 'mouse':
            zp = per_mouse[rng.permutation(len(mice))][inv]
        else:
            zp = Z[rng.permutation(len(Z))]
        full = cv_r2(S, zp, I, folds, fixed_alphas=fixed)
        null[i] = full['cv_r2'] - base_cv
    p = float((1 + np.sum(null >= obs['dR2'])) / (1 + n_perm))
    return dict(which=which, level=level, dR2=obs['dR2'], null=null, p=p,
                null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                z=float((obs['dR2'] - null.mean()) / null.std(ddof=1)),
                cv_r2_base=base_cv, cv_r2_full=obs['cv_r2_full'],
                n_sessions=len(S['session']), n_mice=int(len(np.unique(S['mouse']))),
                observed=obs)


# ---------------------------------------------------------------------------
# Where the gain comes from
# ---------------------------------------------------------------------------
def interaction_weights(S, z, which='all', alpha=None, folds=None):
    """Fit on ALL sessions and return the LDA-interaction weight per base column, with
    its regressor family. Descriptive only -- significance comes from `perm_null`."""
    I = interaction_index(S, which)
    if alpha is None:
        alpha = float(np.median(cv_r2(S, z, I, folds or mouse_folds(S))['alphas']))
    idx = np.arange(len(S['session']))
    w = _fit(*_gram(S, idx, z, I)[:3], alpha)
    p = len(S['cols'])
    q = len(I)
    K = _as_Z(z).shape[1]
    cols = np.array(S['cols'])[I]
    fam = pd.Series(cols).str.replace(r'_b\d+$', '', regex=True) \
                         .str.replace(r'_L\d+$', '', regex=True) \
                         .str.replace(r'_\d+$', '', regex=True)
    out = pd.DataFrame(dict(column=cols, family=fam.values,
                            group=S['groups'][I], w_base=w[:p][I]))
    for a in range(K):
        out[f'w_lda{a + 1}' if K > 1 else 'w_lda'] = w[p + a * q: p + (a + 1) * q]
    return out.assign(alpha=alpha)


def plot_null(res, ax=None):
    """Observed dR2 against its permutation null."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(res['null'], bins=40, color='#b8c6d1', edgecolor='none')
    ax.axvline(res['dR2'], color='#c1442a', lw=2)
    ax.set(xlabel='ΔR²  (LDA interactions)', ylabel='permutations',
           title=f"{res['which']} × LDA1, {res['level']}-level null\n"
                 f"ΔR²={res['dR2']:+.4g}  z={res['z']:+.2f}  p={res['p']:.4f}  "
                 f"({res['n_sessions']} sessions, {res['n_mice']} mice)")
    ax.set_title(ax.get_title(), fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# The 1-degree-of-freedom GAIN test (the non-vacuous version of an "offset")
# ---------------------------------------------------------------------------
# A true LDA main effect is provably worthless here: the offset column is constant
# within a session, and both y and every base column are centred within session, so
#
#     offset . y = sum_s z_s * sum_t y_s(t) = sum_s z_s * 0 = 0
#     offset . B_j = sum_s z_s * sum_t B_j,s(t) = 0
#
# -- the column is EXACTLY orthogonal to the target and to the whole design, so its
# ridge weight is 0 and its dR2 is 0. (Verified numerically: |offset.y| ~ 1e-12.)
# Dropping the within-session centring would turn it into a question about absolute
# firing level, which is confounded by neuron yield / sorting / depth and is already
# answered per-neuron in firing_rate/fr_psth_ldabin.ipynb.
#
# The useful middle ground: let LDA scale the WHOLE fitted response by one number
# instead of reshaping each kernel. Regress on [B, z * (B w0)] where w0 is the base
# model's weights fitted on the TRAINING sessions only. That is 1 column per component
# instead of 78, so the null barely dips below zero -- by far the most powerful test if
# the effect is a uniform gain change rather than a change of kernel shape.
def _gram_gain(S, idx, z, w0):
    """Gram / cross-product for [B, z_1*(B w0), ..., z_K*(B w0)].

    All blocks reduce to the cached per-session A and b:
        c_a . c_b = sum_s z_a,s z_b,s (w0' A_s w0)
        c_a . B   = sum_s z_a,s (w0' A_s)
        c_a . y   = sum_s z_a,s (w0' b_s)
    """
    Z = _as_Z(z)
    p = S['A'].shape[1]
    K = Z.shape[1]
    P = p + K
    G = np.zeros((P, P))
    g = np.zeros(P)
    for k in idx:
        A, bb, zs = S['A'][k], S['b'][k], Z[k]
        G[:p, :p] += A
        g[:p] += bb
        Aw = A @ w0                       # p
        wAw = float(w0 @ Aw)
        wb = float(w0 @ bb)
        for a in range(K):
            G[:p, p + a] += zs[a] * Aw
            g[p + a] += zs[a] * wb
            for b in range(K):
                G[p + a, p + b] += zs[a] * zs[b] * wAw
    G[p:, :p] = G[:p, p:].T
    return G, g, int(S['n'][idx].sum()), float(S['syy'][idx].sum())


def cv_gain(S, z, folds, alphas=ALPHAS, fixed_alphas=None, base_alphas=None):
    """Cross-validated R2 of the 1-df-per-component gain model, held out by mouse.

    w0 is refitted on each fold's TRAINING sessions before the gain column is built,
    so nothing about the held-out mice enters the regressor.
    """
    empty = np.array([], int)
    r2_fold, a_fold, sse_tot, sst_tot = [], [], 0.0, 0.0
    for f, (tr, te) in enumerate(folds):
        a_base = (base_alphas or [None] * len(folds))[f]
        if a_base is None:
            a_base = cv_r2(S, z, empty, [(tr, te)], alphas)['alphas'][0]
        w0 = _fit(*_gram(S, tr, z, empty)[:3], a_base)
        if fixed_alphas is not None:
            alpha = fixed_alphas[f]
        else:
            best = (-np.inf, alphas[0])
            for a in alphas:
                sse, sst = 0.0, 0.0
                for itr, ite in _inner_folds(S, tr, 3):
                    w = _fit(*_gram_gain(S, itr, z, w0)[:3], a)
                    G, g, _, syy = _gram_gain(S, ite, z, w0)
                    sse += _sse(G, g, syy, w)
                    sst += syy
                if 1 - sse / sst > best[0]:
                    best = (1 - sse / sst, a)
            alpha = best[1]
        w = _fit(*_gram_gain(S, tr, z, w0)[:3], alpha)
        G, g, _, syy = _gram_gain(S, te, z, w0)
        sse = _sse(G, g, syy, w)
        r2_fold.append(1 - sse / syy)
        a_fold.append(alpha)
        sse_tot += sse
        sst_tot += syy
    return dict(cv_r2=1 - sse_tot / sst_tot, r2_folds=np.array(r2_fold), alphas=a_fold)


def perm_null_gain(S, z, folds, n_perm=2000, seed=0, alphas=ALPHAS, base=None):
    """Permutation null for the gain test, same session/mouse row-shuffle as
    `perm_null`. Returns both null levels in one pass over the shuffles."""
    empty = np.array([], int)
    base = base or cv_r2(S, z, empty, folds, alphas)
    obs = cv_gain(S, z, folds, alphas, base_alphas=base['alphas'])
    dR2 = obs['cv_r2'] - base['cv_r2']
    Z = _as_Z(z)
    rng = np.random.default_rng(seed)
    mice, inv = np.unique(S['mouse'], return_inverse=True)
    per_mouse = np.stack([Z[S['mouse'] == m][0] for m in mice])
    out = {}
    for level in ('session', 'mouse'):
        null = np.empty(n_perm)
        for i in range(n_perm):
            zp = (per_mouse[rng.permutation(len(mice))][inv] if level == 'mouse'
                  else Z[rng.permutation(len(Z))])
            null[i] = cv_gain(S, zp, folds, fixed_alphas=obs['alphas'],
                              base_alphas=base['alphas'])['cv_r2'] - base['cv_r2']
        out[level] = dict(level=level, dR2=dR2, null=null,
                          p=float((1 + np.sum(null >= dR2)) / (1 + n_perm)),
                          null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                          z=float((dR2 - null.mean()) / null.std(ddof=1)),
                          cv_r2_base=base['cv_r2'], cv_r2_full=obs['cv_r2'],
                          n_components=Z.shape[1])
    return out
