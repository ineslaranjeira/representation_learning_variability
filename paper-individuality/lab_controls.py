"""
LAB CONTROLS FOR EVERY LD1 CORRELATE IN THE PAPER
=================================================
One table: for each correlate of LD1 (and each time-horizon correlation the trait claim rests on), the
correlation as reported, the same correlation with lab removed, and -- where a mouse has several sessions --
the within-mouse version.

WHY THESE THREE COLUMNS
  raw          Spearman across MICE (session values averaged within mouse). The mouse is the unit the
               individuality claim is about; sessions of one mouse are not independent.
  lab          rank partial correlation with lab removed: x and y are ranked across mice, both ranks are
               residualised on lab dummies, and the residuals correlated. p from a permutation that shuffles
               x among mice WITHIN their lab (the null keeps every lab's composition), plus the parametric p
               with the lab dummies' degrees of freedom taken off. This asks: within a lab, do the mice that
               sit higher on LD1 also have more of y?
  within       Spearman of session values minus their mouse's mean (mice with >= 2 sessions). Near zero
               means the correlate is a between-mouse difference, not session-to-session co-fluctuation.
  eta2_lab_y   the share of y's between-mouse variance that sits between labs (how lab-loaded y is).

NEURAL rows are not mouse-level: they rerun the paper's test (neuron or session-region rows, region and
trial-count covariates, LD1 permuted across sessions) and the same test with lab dummies added and LD1
permuted WITHIN lab.

LAB LABELS. The LDA file's `lab` merges hoferlab into mrsicflogel (the SWC mice share the ephys rig).
LAB_SCHEME = 'alyx' splits them back using Alyx's assignment; 'lda' keeps the LDA file's 10 labs.

Sections (run one with `--only <name>`; each writes results/lab_controls/<name>.csv):
  trial     fig_ld1 panel C: accuracy, choice bias, bias shift, RT, elongation, training time
  engage    fig_engagement / fig_engagement_syllables: state switching, persistence, posterior, K = 4,
            fit gain, trial-mode entropy
  syllable  syllable-use summaries (whisking, licking, idle paw) -- needs the syllable file
  neural    fig_neural grid
  horizon   fig_time_horizon: each earlier timepoint vs Proficient, per feature
  ldvar     lab variance per LD (for the figure), not a correlate. Two statistics: eta^2 of lab on the mouse means
            (chance floor ~0.18), and the nested mixed model LD ~ 1 + (1|lab) + (1|lab:mouse) on the session scores,
            lab share = s2_lab / (s2_lab + s2_mouse) (chance 0; lab/variance_partition.py, REML as a cross-check)
`--only all` (default) runs every section whose inputs exist and assembles results/lab_controls_<date>.csv.
"""
import sys
import argparse
import pathlib
import warnings
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata, t as t_dist
from statsmodels.stats.multitest import multipletests

PI = pathlib.Path(__file__).resolve().parent
REPO = PI.parent
for p in [PI, PI / 'neural', PI / 'learning_individuality', PI / 'lab']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---------------------------------------------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------------------------------------------
LDA_FILE = PI / 'clustering' / 'data_files' / 'mouse_LDA_5_bins_raw_shrink0.5_wmouse_360_02-10-2026'
TRIAL_META_FILES = [PI / 'data' / 'session_trial_meta_19-08-2026',      # fig_ld1's file (analysis machine)
                    PI / 'data' / 'session_trial_meta_10-07-2026']      # fallback: 233 of the 260 sessions
TRAINING_FILE = REPO / 'learning_prediction' / 'training_time_05-05-2026'
SYLLABLE_FILES = [PI / 'data' / '8_k_10_bin_syllables_02-10-2026',                  # fig_ld1's path
                  PI / 'clustering' / 'data_files' / '8_k_10_bin_syllables_02-10-2026']
STATES_FILE = PI / 'GLM-HMM' / 'merged_behavioral_and_states.pqt'
PILOT = PI / 'GLM-HMM' / 'k2_k3_pilot'
MODE_ENTROPY_FILE = PI / 'neural' / 'lda1_tables' / 'session_mode_entropy_trial_modes_8_k_10_bin_syllables_02-10-2026_ldasessions.pqt'
HABDIR = REPO / 'habituation'
DMDIR = PI / 'data' / 'design_matrices'
PAW_CACHE = PI / 'figure_generation' / 'cache' / 'cosyne_left_paw_vigor_sessions.csv'
PAW_STATES = 'left_paw_wheel_states_uniform_raw_sessionz_fit-stages'
METHOD = 'pearson'           # 'pearson' (raw values, the default: what the paper reports) or 'spearman' (ranks,
                             # the robustness check), set with --method; the paper's figures
                             # use Pearson (fig_ld1, fig_engagement, fig_neural) and Spearman (fig_engagement_syllables)
OUT_DIR = PI / 'results' / 'lab_controls'
LDVAR_DIR = PI / 'results' / 'lab_controls'      # the lab share per LD does not depend on --method

LAB_SCHEME = 'alyx'          # 'alyx': hoferlab split from mrsicflogel (11 labs); 'lda': the LDA file's 10
ALYX_HOFER = {'SWC_043', 'SWC_060', 'SWC_061'}   # Alyx puts these in hoferlab; the other SWC mice in mrsicflogellab
N_PERM = 10000               # mouse-level within-lab permutations
N_PERM_NEURAL = 2000         # as fig_neural
N_BOOT = 2000                # mouse-level 95% CIs (mice resampled within lab)
N_BOOT_NEURAL = 500          # neural 95% CIs (sessions resampled within lab)
N_TRUNC = 400                # fig_engagement: occupancy / switching over each session's first 400 trials
SEED = 0


# ---------------------------------------------------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------------------------------------------------
def read_lda():
    try:
        return pd.read_pickle(LDA_FILE)
    except ModuleNotFoundError:     # pickled under numpy 2, read under numpy 1
        import numpy.core.numeric, numpy.core.multiarray
        for name, mod in [('numpy._core', np.core), ('numpy._core.numeric', np.core.numeric),
                          ('numpy._core.multiarray', np.core.multiarray)]:
            sys.modules.setdefault(name, mod)
        return pd.read_pickle(LDA_FILE)


LDA = read_lda()
N_AXES = sum(isinstance(c, (int, np.integer)) for c in LDA.columns)
LDA = LDA.rename(columns={0: 'LD1'})


def fix_lab(mouse, lab):
    if LAB_SCHEME == 'alyx' and lab == 'mrsicflogel':
        return 'hoferlab' if mouse in ALYX_HOFER else 'mrsicflogellab'
    return lab


LDA['lab'] = [fix_lab(m, l) for m, l in zip(LDA['mouse_name'], LDA['lab'])]
MOUSE_LAB = LDA.drop_duplicates('mouse_name').set_index('mouse_name')['lab']
SESSION_LD1 = LDA.set_index('session')['LD1']
SESSION_MOUSE = LDA.set_index('session')['mouse_name']


def first_existing(paths):
    return next((p for p in paths if p.exists()), None)


# ---------------------------------------------------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------------------------------------------------
def _residualise(v, D):
    return v - D @ np.linalg.lstsq(D, v, rcond=None)[0]


def _lab_design(labs):
    return np.column_stack([np.ones(len(labs)), pd.get_dummies(pd.Series(labs), drop_first=True).to_numpy(float)])


def eta2(v, labs):
    v = np.asarray(v, float)
    s = pd.Series(v).groupby(np.asarray(labs))
    ss_b = (s.size() * (s.mean() - v.mean()) ** 2).sum()
    return ss_b / ((v - v.mean()) ** 2).sum()


def within_lab_perm(x, labs, n_perm, rng):
    """(n_perm, n) array: x shuffled among the units of each lab."""
    labs = np.asarray(labs)
    out = np.tile(x, (n_perm, 1))
    for l in np.unique(labs):
        idx = np.flatnonzero(labs == l)
        if len(idx) > 1:
            out[:, idx] = rng.permuted(np.tile(x[idx], (n_perm, 1)), axis=1)
    return out


def _tr(v):
    """The values the correlation is computed on: ranks for Spearman, the values for Pearson."""
    return rankdata(v) if METHOD == 'spearman' else np.asarray(v, float)


def _corr(a, b):
    return spearmanr(a, b) if METHOD == 'spearman' else pearsonr(a, b)


def _rho_pair(xv, yv, labs):
    """(Spearman, lab-removed rank partial correlation) for one sample of mice."""
    rx, ry = _tr(xv), _tr(yv)
    D = _lab_design(labs)
    ex, ey = _residualise(rx, D), _residualise(ry, D)
    return np.corrcoef(rx, ry)[0, 1], np.corrcoef(ex, ey)[0, 1]


def boot_ci(xv, yv, labs, rng, n_boot=N_BOOT):
    """95% CIs of both correlations, mice resampled with replacement WITHIN lab (lab sizes kept)."""
    groups = [np.flatnonzero(labs == l) for l in np.unique(labs)]
    out = np.empty((n_boot, 2))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        out[b] = _rho_pair(xv[idx], yv[idx], labs[idx])
    lo, hi = np.nanpercentile(out, [2.5, 97.5], axis=0)
    return dict(rho_lo=lo[0], rho_hi=hi[0], rho_lab_lo=lo[1], rho_lab_hi=hi[1])


def mouse_level(m, x, y, rng):
    """m: one row per mouse with columns x, y, lab."""
    m = m[[x, y, 'lab']].replace([np.inf, -np.inf], np.nan).dropna()
    n, labs = len(m), m['lab'].to_numpy()
    rx, ry = _tr(m[x]), _tr(m[y])
    rho, p = _corr(m[x], m[y])
    D = _lab_design(labs)
    ex, ey = _residualise(rx, D), _residualise(ry, D)
    r_lab = np.corrcoef(ex, ey)[0, 1]
    ci = boot_ci(m[x].to_numpy(float), m[y].to_numpy(float), labs, rng)
    dof = n - D.shape[1] - 1
    p_lab_t = 2 * t_dist.sf(abs(r_lab) * np.sqrt(dof / (1 - r_lab ** 2)), dof)
    # permutation: x's ranks shuffled within lab, every permutation residualised at once (P H, H = D D+ symmetric)
    P = within_lab_perm(rx, labs, N_PERM, rng)
    Pe = P - (P @ D) @ np.linalg.pinv(D)
    r_null = (Pe @ ey) / (np.linalg.norm(Pe, axis=1) * np.linalg.norm(ey))
    p_lab = (1 + np.sum(np.abs(r_null) >= abs(r_lab))) / (1 + N_PERM)
    return dict(n_mice=n, n_labs=len(np.unique(labs)), rho=rho, p=p, rho_lab=r_lab, p_lab=p_lab,
                p_lab_t=p_lab_t, **ci, eta2_lab_y=eta2(m[y], labs), eta2_lab_x=eta2(m[x], labs))


def within_mouse(s, x, y):
    s = s[['mouse_name', x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s.groupby('mouse_name')[x].transform('size') >= 2]
    if s['mouse_name'].nunique() < 5:
        return dict(rho_within=np.nan, p_within=np.nan, n_sessions=np.nan)
    dx = s[x] - s.groupby('mouse_name')[x].transform('mean')
    dy = s[y] - s.groupby('mouse_name')[y].transform('mean')
    r, p = _corr(dx, dy)
    return dict(rho_within=r, p_within=p, n_sessions=len(s))


def correlate(sess, y, label, section, rng, x='LD1', per_session=True, note=''):
    """sess: one row per session (per_session) or per mouse, with mouse_name, x, y."""
    s = sess.copy()
    if 'lab' not in s:
        s['lab'] = s['mouse_name'].map(MOUSE_LAB)
    m = (s.groupby('mouse_name').agg(**{x: (x, 'mean'), y: (y, 'mean'), 'lab': ('lab', 'first')})
         if per_session else s.set_index('mouse_name'))
    row = dict(section=section, correlate=label, x=x, y=y, **mouse_level(m, x, y, rng))
    row.update(within_mouse(s, x, y) if per_session else dict(rho_within=np.nan, p_within=np.nan, n_sessions=np.nan))
    # for reference only: across sessions, as the paper's panels draw it (sessions of one mouse are not independent)
    if per_session:
        ss = s[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        r, p = _corr(ss[x], ss[y])
        se = (np.sqrt((1 + r ** 2 / 2) / (len(ss) - 3)) if METHOD == 'spearman'   # Bonett & Wright (2000)
              else 1 / np.sqrt(len(ss) - 3))                                      # Fisher z
        lo, hi = np.tanh(np.arctanh(r) + np.array([-1.96, 1.96]) * se)
        row.update(rho_session=r, p_session=p, rho_session_lo=lo, rho_session_hi=hi, n_sessions_all=len(ss))
    else:
        row.update(rho_session=np.nan, p_session=np.nan, rho_session_lo=np.nan, rho_session_hi=np.nan,
                   n_sessions_all=np.nan)
    row['note'] = note
    return row


def show(rows, title):
    t = pd.DataFrame(rows)
    cols = ['correlate', 'rho_session', 'p_session', 'n_mice', 'rho', 'p', 'rho_lab', 'p_lab', 'rho_within', 'p_within', 'eta2_lab_y']
    cols = [c for c in cols if c in t]
    print(f'\n=== {title} ===')
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(t[cols].round(3).to_string(index=False))
    return t


# ---------------------------------------------------------------------------------------------------------------
# SECTIONS
# ---------------------------------------------------------------------------------------------------------------
def section_trial(rng):
    """fig_ld1 panel C, same definitions (cell 11)."""
    meta_file = first_existing(TRIAL_META_FILES)
    if meta_file is None:
        print('trial: no trial-meta file -- skipped'); return []
    tf = pd.read_parquet(meta_file)
    tt = pd.read_parquet(TRAINING_FILE)
    tt['log_training'] = np.log(tt['training_days'])
    excluded = tt.loc[tt['log_training'].isin([0, 1]), 'mouse_name'].tolist()
    tt = tt[~tt['mouse_name'].isin(excluded)]
    tf = tf[~tf['mouse_name'].isin(excluded)].copy()
    tf['choice'] = tf['choice'].map({'left': 0, 'right': 1, 'nan': np.nan})
    tf['correct'] = (tf['feedback'] == 'correct').astype(int)
    zc = tf[tf['contrast'] == 0].groupby(['session', 'block'])['choice'].mean().unstack('block')
    bias_df = pd.DataFrame({'session': zc.index, 'bias': (zc[0.8] - zc[0.2]).values})
    hc = tf[tf['contrast'] > 0].copy()
    hc['expected_right'] = (hc['signed_contrast'] < 0).astype(int)
    cb = hc.groupby('session').agg(choice_mean=('choice', 'mean'), expected_right_mean=('expected_right', 'mean')).reset_index()
    cb['choice_bias'] = (cb['choice_mean'] - cb['expected_right_mean']).abs()
    acc = tf[tf['contrast'] > 0.5].groupby('session')['correct'].mean().rename('correct').reset_index()
    ss = tf.groupby(['session', 'mouse_name']).agg(reaction=('reaction', 'median'), elongation=('elongation', 'median')).reset_index()
    with np.errstate(invalid='ignore', divide='ignore'):
        ss['log_reaction'], ss['log_elongation'] = np.log(ss['reaction']), np.log(ss['elongation'])
    beh = (ss.merge(bias_df, on='session').merge(cb[['session', 'choice_bias']], on='session').merge(acc, on='session')
             .merge(LDA[['session', 'mouse_name', 'LD1']], on=['session', 'mouse_name']))
    note = f'{meta_file.name}, {beh["session"].nunique()} sessions'
    rows = [correlate(beh, v, lab, 'trial', rng, note=note) for v, lab in
            [('correct', 'Accuracy (easy trials)'), ('choice_bias', 'Choice bias'), ('bias', 'Block bias shift'),
             ('log_reaction', 'Reaction time (log)'), ('log_elongation', 'Trial elongation (log)')]]
    mice = (LDA.groupby('mouse_name')['LD1'].mean().rename('LD1').reset_index()
               .merge(tt[['mouse_name', 'log_training']], on='mouse_name'))
    rows.append(correlate(mice, 'log_training', 'Training time (log days)', 'trial', rng, per_session=False,
                          note=TRAINING_FILE.name))
    show(rows, f'trial-level correlates ({note})')
    return rows


def run_lengths(x):
    d = np.diff(np.r_[0, x.astype(int), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def section_engage(rng):
    """fig_engagement cell 13's session measures, recomputed (first N_TRUNC trials, engaged = P >= 0.5)."""
    if not STATES_FILE.exists():
        print('engage: no GLM-HMM states file -- skipped'); return []
    tr = pd.read_parquet(STATES_FILE)
    tr = tr[tr['eid'].isin(SESSION_LD1.index)].copy()
    tr['trial_id'] = tr.groupby('eid').cumcount()
    tr['engaged'] = tr['p_state1'] >= 0.5
    tr['correct'] = (tr['rewarded'] == 1).astype(int)
    tr = tr[tr['trial_id'] < N_TRUNC]

    def per_session(g):
        e = g['engaged'].to_numpy()
        p = g['p_state1'].to_numpy().clip(1e-12, 1 - 1e-12)
        a, b = e[:-1], e[1:]
        dis = run_lengths(~e)
        return pd.Series({
            'frac_engaged': e.mean(),
            'log_ratio': np.log(g['p_state1'].mean() / g['p_state2'].mean()),
            'entropy': np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))),
            'p_stay_eng': (a & b).sum() / a.sum() if a.sum() else np.nan,
            'p_stay_dis': (~a & ~b).sum() / (~a).sum() if (~a).sum() else np.nan,
            'switches': (a != b).sum(),
            'short_bouts': (dis <= 2).sum()})
    S = tr.groupby('eid')[['engaged', 'p_state1', 'p_state2']].apply(per_session)
    easy = tr[tr['signed_contrast'].abs() >= 0.25]
    S = S.join(easy.groupby('eid')['correct'].mean().rename('p_easy'))
    loso_f = PILOT / 'loso_k2_k3_k4_k5_wide.csv'
    if loso_f.exists():
        loso = pd.read_csv(loso_f).set_index('held_out_session')
        S = S.join((loso['bpt_K3'] - loso['bpt_K2']).rename('gain_23'), how='left')
        S = S.join((loso['bpt_K4'] - loso['bpt_K2']).rename('gain_24'), how='left')
    post_f = PILOT / 'all_k_posteriors.parquet'
    if post_f.exists():
        post = pd.read_parquet(post_f, filters=[('K', '==', 4)], columns=['eid', 'trial_idx', 'state', 'posterior'])
        dom = (post.sort_values('posterior').drop_duplicates(['eid', 'trial_idx'], keep='last')
                   .sort_values(['eid', 'trial_idx']))
        dom = dom[dom['trial_idx'] < N_TRUNC]
        S = S.join(dom.groupby('eid')['state'].agg(lambda s: int((np.diff(s.to_numpy()) != 0).sum())).rename('switches_k4'))
    if MODE_ENTROPY_FILE.exists():
        S = S.join(pd.read_parquet(MODE_ENTROPY_FILE).set_index('session')['H_modes'], how='outer')
    S = S[S.index.isin(SESSION_LD1.index)]
    S['LD1'] = SESSION_LD1.reindex(S.index)
    S['mouse_name'] = SESSION_MOUSE.reindex(S.index)
    S = S.reset_index().rename(columns={'index': 'session', 'eid': 'session'})
    # check against fig_engagement's printed session-level Pearson (switches: r = -0.3418, n = 244)
    ok = S[['LD1', 'switches']].dropna()
    print(f'engage: {len(ok)} sessions; check switches vs LD1 session-level r = {np.corrcoef(ok["LD1"], ok["switches"])[0, 1]:+.4f} '
          '(fig_engagement: -0.3418, n = 244)')
    # switches with easy-trial accuracy regressed out across sessions, as fig_engagement's W
    w = S[['switches', 'p_easy']].dropna()
    S.loc[w.index, 'switches_resid_acc'] = w['switches'] - np.polyval(np.polyfit(w['p_easy'], w['switches'], 1), w['p_easy'])
    tests = [('switches', 'State switches (K=2)'), ('switches_resid_acc', 'State switches | easy accuracy'),
             ('p_stay_eng', 'P(stay engaged)'), ('p_stay_dis', 'P(stay disengaged)'),
             ('short_bouts', 'Short disengaged bouts (<=2 trials)'), ('entropy', 'Posterior uncertainty'),
             ('log_ratio', 'Engaged log posterior ratio'), ('frac_engaged', 'Fraction engaged (hard)'),
             ('switches_k4', 'State switches (K=4)'), ('gain_23', 'Held-out fit gain K=2->3'),
             ('gain_24', 'Held-out fit gain K=2->4'),
             ('H_modes', 'Trial-mode entropy')]
    rows = [correlate(S, c, lab, 'engage', rng) for c, lab in tests if c in S]
    show(rows, 'engagement correlates')
    return rows


def loading_map(occ, sessions, rng):
    """LD1's syllable loadings (fig_ld1 panel B) at mouse level, with and without lab.

    occ: (sessions, bins, syllables) occupancy, rows in the order of `sessions`. Per cell: Spearman of the
    mouse-mean occupancy with mouse-mean LD1, and the rank partial correlation with lab removed; p for the
    latter from the same within-lab permutation as the table (one shared set of permutations for every cell),
    BH over the cells. Writes results/lab_controls/loading_map.csv (long format, one row per cell)."""
    s = pd.DataFrame({'session': sessions, 'mouse_name': SESSION_MOUSE.reindex(sessions).to_numpy()})
    mice = s['mouse_name'].drop_duplicates().to_numpy()
    M = np.stack([np.nanmean(occ[(s['mouse_name'] == m).to_numpy()], axis=0) for m in mice])   # mice x bins x syl
    nb, ns = M.shape[1:]
    F = M.reshape(len(mice), -1)
    x = LDA.groupby('mouse_name')['LD1'].mean().reindex(mice).to_numpy()
    labs = MOUSE_LAB.reindex(mice).to_numpy()
    D = _lab_design(labs)
    rx = _tr(x)
    RF = np.apply_along_axis(_tr, 0, F)
    flat = RF.std(0) > 0                                              # a cell with no variance across mice
    def corr_cols(a, B):
        a, B = a - a.mean(), B - B.mean(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            return (a @ B) / (np.linalg.norm(a) * np.linalg.norm(B, axis=0))
    r_raw = corr_cols(rx, RF)
    H = D @ np.linalg.pinv(D)
    ex, EF = rx - H @ rx, RF - H @ RF
    r_lab = corr_cols(ex, EF)
    P = within_lab_perm(rx, labs, N_PERM, rng)
    Pe = P - P @ H
    EFn = (EF - EF.mean(0)) / np.where(np.linalg.norm(EF - EF.mean(0), axis=0) > 0, np.linalg.norm(EF - EF.mean(0), axis=0), np.nan)
    null = (Pe / np.linalg.norm(Pe, axis=1, keepdims=True)) @ np.nan_to_num(EFn)
    p_lab = (1 + (np.abs(null) >= np.abs(np.nan_to_num(r_lab))).sum(0)) / (1 + N_PERM)
    p_lab[~flat] = np.nan
    q_lab = np.full_like(p_lab, np.nan)
    q_lab[flat] = multipletests(p_lab[flat], method='fdr_bh')[1]
    # the same test for the map as reported (no lab): x's ranks shuffled across all mice
    RFn = (RF - RF.mean(0)) / np.where(RF.std(0) > 0, np.linalg.norm(RF - RF.mean(0), axis=0), np.nan)
    Pr = np.stack([rng.permutation(rx) for _ in range(N_PERM)])
    Pr = Pr - Pr.mean(1, keepdims=True)
    null_raw = (Pr / np.linalg.norm(Pr, axis=1, keepdims=True)) @ np.nan_to_num(RFn)
    p_raw = (1 + (np.abs(null_raw) >= np.abs(np.nan_to_num(r_raw))).sum(0)) / (1 + N_PERM)
    p_raw[~flat] = np.nan
    q_raw = np.full_like(p_raw, np.nan)
    q_raw[flat] = multipletests(p_raw[flat], method='fdr_bh')[1]
    # qualitative reading, as fig_ld1 panel B: each cell against its own permutation null (2.5-97.5 %), no
    # correction across cells; the q columns are kept for reference only
    raw_lo, raw_hi = np.percentile(null_raw, [2.5, 97.5], axis=0)
    lab_lo, lab_hi = np.percentile(null, [2.5, 97.5], axis=0)
    out = pd.DataFrame({'bin': np.repeat(np.arange(nb), ns), 'syllable': np.tile(np.arange(ns), nb),
                        'r_raw': r_raw, 'null_raw_lo': raw_lo, 'null_raw_hi': raw_hi, 'p_raw': p_raw, 'q_raw': q_raw,
                        'r_lab': r_lab, 'null_lab_lo': lab_lo, 'null_lab_hi': lab_hi, 'p_lab': p_lab, 'q_lab': q_lab})
    out.to_csv(OUT_DIR / 'loading_map.csv', index=False)
    ok = flat & np.isfinite(r_raw) & np.isfinite(r_lab)
    print(f'\n=== LD1 loading map ({len(mice)} mice, {nb} bins x {ns} syllables) ===')
    print(f'pattern similarity, raw vs lab-removed map: r = {np.corrcoef(r_raw[ok], r_lab[ok])[0, 1]:+.3f}')
    print(f'cells with q < 0.05: {(q_raw < 0.05).sum()} as reported, {(q_lab < 0.05).sum()} lab removed (of {ok.sum()})')
    return out


def section_syllable(rng):
    syl_file = first_existing(SYLLABLE_FILES)
    if syl_file is None:
        print('syllable: no syllable file on this machine -- skipped'); return []
    import paper_style as ps
    ps.use_paw_states('uniform')
    syl = pd.read_parquet(syl_file)
    syl['session'] = syl['sample'].str[:36]
    syl = syl[syl['session'].isin(SESSION_LD1.index)]
    trials = (syl.pivot(index=['mouse_name', 'session', 'sample'], columns='broader_label', values='binned_sequence')
              .reset_index().dropna(subset=ps.EPOCH_NAMES))
    codes = ps.paw_codes(np.vstack([np.hstack(r) for r in trials[ps.EPOCH_NAMES].to_numpy()]).astype(float))
    paw, whisk, lick = ps.decode_syllables(codes)
    # fig_ld1 cell 9: per trial, the occupancy of each syllable in each of the 40 bins; per session, its mean
    OCC = np.stack([np.where(np.isnan(paw), np.nan, paw == k) for k in range(ps.N_PAW_STATES)] + [whisk, lick], axis=-1)
    sess_idx = trials['session'].to_numpy()
    order = np.array([x for x in LDA['session'] if x in set(sess_idx)])
    with np.errstate(invalid='ignore'):
        SO = np.stack([np.nanmean(OCC[sess_idx == x], axis=0) for x in order])        # sessions x 40 x 10
    print(f'syllable: {syl_file.relative_to(PI)} | {len(order)} sessions x {SO.shape[1]} bins x {SO.shape[2]} syllables')
    loading_map(SO, order, rng)
    vig_rank = np.argsort(np.array(ps.PAW_VIGOR_ORDER))                                # vigor rank of each paw state
    paw_occ = SO[:, :, :ps.N_PAW_STATES]
    with np.errstate(invalid='ignore'):
        vigor = np.nanmean((paw_occ * vig_rank).sum(-1) / paw_occ.sum(-1), axis=1)
    q = slice(10, 20)                                                                  # the Quiescence epoch's bins
    s = pd.DataFrame({'session': order, 'mouse_name': SESSION_MOUSE.reindex(order).to_numpy(),
                      'LD1': SESSION_LD1.reindex(order).to_numpy(),
                      'whisk': np.nanmean(SO[:, :, 8], 1), 'whisk_q': np.nanmean(SO[:, q, 8], 1),
                      'lick': np.nanmean(SO[:, :, 9], 1), 'paw_idle': np.nanmean(SO[:, :, 0], 1), 'paw_vigor_rank': vigor})
    rows = [correlate(s, c, lab, 'syllable', rng) for c, lab in
            [('whisk', 'Whisking (whole trial)'), ('whisk_q', 'Whisking (quiescence)'), ('lick', 'Licking (whole trial)'),
             ('paw_idle', 'Idle paw state (0)'), ('paw_vigor_rank', 'Mean paw-state vigor rank')]]
    show(rows, 'syllable-use summaries')
    return rows



ENTROPY_CACHE = OUT_DIR / 'entropy_sessions.csv'


def section_entropy(rng):
    """fig_engagement_syllables panel I: per session, syllable entropy H(S) and its split into a between-mode
    part I(S;M) and a within-mode part H(S|M), plus the between share. Computed by running that notebook's own
    cells (up to its panel-I table) and cached, since the decomposition takes a while."""
    if ENTROPY_CACHE.exists():
        dec = pd.read_csv(ENTROPY_CACHE)
    else:
        import json, os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        nb = json.load(open(PI / 'figure_generation' / 'fig_engagement_syllables.ipynb'))
        ns, here, plt_show = {'__name__': 'lab_controls_entropy'}, os.getcwd(), plt.show
        plt.show = lambda *a, **k: plt.close('all')
        os.chdir(PI / 'figure_generation')
        if str(PI / 'figure_generation') not in sys.path:
            sys.path.insert(0, str(PI / 'figure_generation'))
        try:
            for c in nb['cells']:
                src = ''.join(c['source'])
                if c['cell_type'] != 'code':
                    continue
                exec(compile(src, 'fig_engagement_syllables.ipynb', 'exec'), ns)
                if 'dec_res = rho_table(dec' in src:
                    break
        except FileNotFoundError as e:
            print(f'entropy: fig_engagement_syllables needs {e} -- skipped'); return []
        finally:
            os.chdir(here); plt.show = plt_show; plt.close('all')
        dec = ns['dec'][['session', 'H_syl', 'I_between', 'H_within', 'frac_between']]
        dec.to_csv(ENTROPY_CACHE, index=False)
    dec = dec.assign(mouse_name=dec['session'].map(SESSION_MOUSE), LD1=dec['session'].map(SESSION_LD1)).dropna(subset=['LD1'])
    rows = [correlate(dec, c, lab, 'entropy', rng) for c, lab in
            [('H_syl', 'Syllable entropy'), ('I_between', 'Syllable entropy, between modes'),
             ('H_within', 'Syllable entropy, within modes'), ('frac_between', 'Between-mode share of entropy')]]
    show(rows, 'syllable-entropy decomposition')
    return rows


def neural_test(df, y_col, covars, n_perm, seed):
    """fig_neural's test (nm.age_effect_test's arithmetic), with and without lab, LD1 permuted across sessions
    (no lab) or within lab (lab)."""
    d = df.dropna(subset=[y_col, 'lda_1', 'region'] + list(covars)).copy()
    d['lab'] = d['mouse_name'].map(MOUSE_LAB)
    d = d.dropna(subset=['lab']).reset_index(drop=True)
    y = d[y_col].to_numpy(float)
    base = np.column_stack([np.ones(len(d)), pd.get_dummies(d['region'], drop_first=True).to_numpy(float),
                            d[list(covars)].to_numpy(float)])
    labD = pd.get_dummies(d['lab'], drop_first=True).to_numpy(float)
    sess = d.groupby('session').agg(x=('lda_1', 'first'), lab=('lab', 'first'))
    row_sess = d['session'].map({s: i for i, s in enumerate(sess.index)}).to_numpy()
    rng = np.random.default_rng(seed)
    out = dict(n_rows=len(d), n_sessions=len(sess), n_mice=d['mouse_name'].nunique())
    for tag, Z, strat in [('', base, False), ('_lab', np.column_stack([base, labD]), True)]:
        Zp = np.linalg.pinv(Z)
        yr = y - Z @ (Zp @ y)
        x = sess['x'].to_numpy()[row_sess]
        xr = x - Z @ (Zp @ x)
        slope = (xr @ yr) / (xr @ xr)
        P = (within_lab_perm(sess['x'].to_numpy(), sess['lab'].to_numpy(), n_perm, rng) if strat
             else np.stack([rng.permutation(sess['x'].to_numpy()) for _ in range(n_perm)]))
        null = np.empty(n_perm)
        for i0 in range(0, n_perm, 250):
            Xp = P[i0:i0 + 250][:, row_sess].T                    # rows x perms
            Xr = Xp - Z @ (Zp @ Xp)
            null[i0:i0 + 250] = (Xr.T @ yr) / np.einsum('ij,ij->j', Xr, Xr)
        out[f'r_partial{tag}'] = np.corrcoef(xr, yr)[0, 1]
        out[f'p_perm{tag}'] = (1 + np.sum(np.abs(null) >= abs(slope))) / (1 + n_perm)
    # 95% CIs: sessions resampled within lab; a session drawn k times weighs its rows k times. Each resample's
    # regression is solved from per-session sums (Z'Z, Z'[y x], [y x]'[y x]) re-weighted by the draw counts, so
    # it costs a k x k solve instead of a pass over every row. pinv handles a covariate column left empty by a
    # resample (a region whose sessions were all left out).
    x = sess['x'].to_numpy()[row_sess]
    groups = [np.flatnonzero(sess['lab'].to_numpy() == l) for l in np.unique(sess['lab'])]
    Zs = {'': base, '_lab': np.column_stack([base, labD])}
    W = np.column_stack([y, x])
    stats = {}
    for tag, Z in Zs.items():
        k = Z.shape[1]
        A, B, C = np.zeros((len(sess), k, k)), np.zeros((len(sess), k, 2)), np.zeros((len(sess), 2, 2))
        for s in range(len(sess)):
            r = row_sess == s
            A[s], B[s], C[s] = Z[r].T @ Z[r], Z[r].T @ W[r], W[r].T @ W[r]
        stats[tag] = (A, B, C)
    boots = {tag: np.empty(N_BOOT_NEURAL) for tag in Zs}
    for b in range(N_BOOT_NEURAL):
        counts = np.bincount(np.concatenate([rng.choice(g, len(g)) for g in groups]), minlength=len(sess)).astype(float)
        for tag, (A, B, C) in stats.items():
            Ab, Bb, Cb = np.tensordot(counts, A, 1), np.tensordot(counts, B, 1), np.tensordot(counts, C, 1)
            E = Cb - Bb.T @ np.linalg.pinv(Ab) @ Bb          # residual cross-products of y and x
            boots[tag][b] = E[0, 1] / np.sqrt(E[0, 0] * E[1, 1])
    for tag in Zs:
        out[f'r_lo{tag}'], out[f'r_hi{tag}'] = np.nanpercentile(boots[tag], [2.5, 97.5])
    return out


def _neural_tables():
    """fig_neural's tables, cohort and the current LD1 (None if the tables are not on this machine)."""
    import lda1_neural_metrics as nm
    import run_behavior_axes as rba
    cfg = rba.config()
    # the newest build of this QC configuration BY THE DATE IN ITS NAME (load_tables goes by file mtime, which a
    # copy from a backup does not preserve)
    import re
    want = nm.cache_tag(cfg)
    builds = sorted({f.name.split('__')[0] for f in nm.CACHE.glob(f'{want}_*__ff.pqt')
                     if re.fullmatch(r'\d{2}-\d{2}-\d{4}', f.name.split('__')[0][len(want) + 1:])},
                    key=lambda b: pd.to_datetime(b[-10:], dayfirst=True))
    if not builds:
        print(f'neural: no {want}_<date> tables in {nm.CACHE} -- skipped'); return None, nm
    T = nm.load_tables(tag=builds[-1], verbose=False)
    print(f'neural: tables {builds[-1]}')
    # the cached tables may carry an older LD1 (opposite sign): put the current file's LD1 in, by session
    for k in ['fr', 'ff', 'rsc', 'curves', 'sessions']:
        if k in T and 'session' in T[k]:
            T[k]['lda_1'] = T[k]['session'].map(SESSION_LD1)
            T[k]['mouse_name'] = T[k]['session'].map(SESSION_MOUSE)
    return nm.apply_cohort(T, cfg, 'common', verbose=False), nm


def section_neural(rng):
    T, nm = _neural_tables()
    if T is None:
        return []
    rows = []
    for (metric, test), spec in nm.GRID.items():
        if spec is None:
            continue
        tbl, y_col, _, _, _, covars = spec
        r = neural_test(T[tbl], y_col, covars, N_PERM_NEURAL, SEED)
        rows.append(dict(section='neural', correlate=f'{metric}, {test}', x='LD1', y=y_col,
                         n_mice=r['n_mice'], n_sessions=r['n_sessions'],
                         rho=r['r_partial'], p=r['p_perm'], rho_lab=r['r_partial_lab'], p_lab=r['p_perm_lab'],
                         rho_lo=r['r_lo'], rho_hi=r['r_hi'], rho_lab_lo=r['r_lo_lab'], rho_lab_hi=r['r_hi_lab'],
                         note=('not in figure; ' if test == 'Response onset' else '')
                              + f'partial r, {r["n_rows"]} rows, region + {"/".join(covars)} covariates'))
        print(f"  {metric:<18} {test:<19} r = {r['r_partial']:+.3f} (p {r['p_perm']:.4f})   "
              f"+lab: r = {r['r_partial_lab']:+.3f} (p {r['p_perm_lab']:.4f})   {r['n_sessions']} sessions, {r['n_mice']} mice")
    show(rows, 'neural correlates (partial r; p from session permutation, within lab for +lab)')
    return rows


def section_neural_mouse(rng):
    """The fig_neural tests with the MOUSE as the unit. Per test: the metric is adjusted for the same covariates as
    fig_neural (region dummies + trial count [+ neuron count]) by regression on all rows WITHOUT LD1, the residuals are
    averaged within session (every neuron / session-region counts once) and then within mouse, and that is correlated
    with the mouse's LD1 (mean over its recorded sessions) -- the same statistics as the behavioural rows (correlate():
    across sessions, across mice, lab removed, within mouse). Equal weight per mouse, where fig_neural weights rows."""
    T, nm = _neural_tables()
    if T is None:
        return []
    rows = []
    for (metric, test), spec in nm.GRID.items():
        if spec is None or test == 'Response onset':
            continue
        tbl, y_col, _, _, region_col, covars = spec
        d = T[tbl].dropna(subset=[y_col, 'lda_1', region_col] + list(covars)).reset_index(drop=True)
        Z = np.column_stack([np.ones(len(d)), pd.get_dummies(d[region_col], drop_first=True).to_numpy(float),
                             d[list(covars)].to_numpy(float)])
        y = d[y_col].to_numpy(float)
        d['y_adj'] = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        sess = (d.groupby('session').agg(y_adj=('y_adj', 'mean'), LD1=('lda_1', 'first'),
                                         mouse_name=('mouse_name', 'first')).reset_index())
        r = correlate(sess, 'y_adj', f'{metric}, {test}', 'neural_mouse', rng,
                      note=f'{tbl}: covariate-adjusted, session then mouse means; {len(sess)} sessions')
        rows.append(r)
    show(rows, 'neural, mouse as the unit (covariate-adjusted metric, session -> mouse means)')
    return rows


def section_horizon(rng):
    """fig_time_horizon / cosyne panel j: per mouse, each earlier timepoint's mean against the same mouse's
    Proficient mean; Proficient vs itself = odd vs even sessions by date."""
    hab_f = HABDIR / 'hmm_state_measures.csv'
    if not hab_f.exists():
        print(f'horizon: {hab_f} not on this machine -- skipped'); return []
    from session_filters import load_session_table
    sys.path.insert(0, str(PI / '4_mice'))
    from functions import lab_labels
    stage_dirs = {'Early': DMDIR / '1_camera_setup' / 'session_1', 'Late': DMDIR / '1_camera_setup' / 'last_training',
                  'Pre-rec': DMDIR / '1_camera_setup' / 'biased'}
    stage_of = {f.name[14:50]: tp for tp, d in stage_dirs.items() for f in d.glob('design_matrix_*')}
    table = load_session_table()
    if not stage_of:     # design matrices not on this machine: the QC sheet's timepoint for each eid
        stage_of = table.drop_duplicates('eid').set_index('eid')['timepoint'].dropna().to_dict()
        print('horizon: timepoints from the QC sheet (no design matrices here)')
    h = pd.read_csv(hab_f)
    h['timepoint'] = np.where(h['source'] == 'Training', h['eid'].map(stage_of), h['source'])
    h = h.dropna(subset=['timepoint'])
    h['feature'] = h['var'] + '_frac'
    long = [h[['timepoint', 'mouse_name', 'eid', 'feature', 'frac']].rename(columns={'frac': 'value'})]
    if PAW_CACHE.exists():
        long.append(pd.read_csv(PAW_CACHE).assign(feature='paw_vigor').rename(columns={'paw_vigor': 'value'})
                    [['timepoint', 'mouse_name', 'eid', 'feature', 'value']])
    else:
        print(f'horizon: {PAW_CACHE.name} not found -- paw vigor rows skipped (run cosyne_figure once to build it)')
    long = pd.concat(long)
    # lab per mouse: the LDA file where the mouse is in it, otherwise its training-rig name
    tr = table[table['timepoint'].isin(['Early', 'Late'])]
    rig_lab = lab_labels(tr['eid'], tr['mouse_name'], verbose=False).groupby(tr['mouse_name'].to_numpy()).first()
    lab_of = {m: MOUSE_LAB.get(m, fix_lab(m, rig_lab.get(m))) for m in long['mouse_name'].unique()}
    date_of = table.drop_duplicates('eid').set_index('eid')['date']
    m = long.groupby(['feature', 'timepoint', 'mouse_name'])['value'].mean()
    prof = long[long['timepoint'] == 'Proficient'].copy()
    prof['date'] = prof['eid'].map(date_of)
    prof = prof.dropna(subset=['date']).sort_values(['feature', 'mouse_name', 'date'])
    prof['half'] = prof.groupby(['feature', 'mouse_name']).cumcount() % 2
    halves = prof.groupby(['feature', 'mouse_name', 'half'])['value'].mean().unstack('half').dropna()
    rows = []
    names = {'whisk_frac': 'Whisking', 'lick_frac': 'Licking', 'paw_vigor': 'Left-paw vigor'}
    for f in [f for f in names if f in m.index.get_level_values(0)]:
        w = m.loc[f].unstack(0)
        for tp in ['Early', 'Late', 'Pre-rec', 'Proficient']:
            if tp == 'Proficient':
                d = halves.loc[f].set_axis(['earlier', 'proficient'], axis=1).reset_index()
                lab = 'odd vs even Proficient sessions (reliability)'
            elif tp in w:
                d = w[[tp, 'Proficient']].dropna().set_axis(['earlier', 'proficient'], axis=1).reset_index()
                lab = f'{tp} vs Proficient'
            else:
                continue
            d['lab'] = d['mouse_name'].map(lab_of)
            rows.append(correlate(d, 'proficient', f'{names[f]}: {lab}', 'horizon', rng, x='earlier', per_session=False,
                                  note=f'{d["lab"].isna().sum()} mice without a lab dropped' if d['lab'].isna().any() else ''))
    show(rows, 'time horizon (mouse means, each timepoint vs Proficient)')
    return rows


def section_ldvar(rng, n_perm=5000):
    """Lab share of each LD's between-mouse variance (eta^2 of lab on the mouse means), for EVERY discriminant,
    against a null that shuffles lab labels across mice keeping lab sizes (its floor is (labs - 1) / (mice - 1),
    not 0). Also each LD's share of the between-mouse variance, so the lab share can be weighted by how much the axis
    carries. For fig_lab_share, not the correlate table. p one-sided (observed >= null), q = BH over all LDs."""
    cols = [c for c in LDA.columns if isinstance(c, (int, np.integer))]
    mm = LDA.rename(columns={'LD1': 0}).groupby('mouse_name')[[0] + cols].mean()
    X = mm.to_numpy(float)                                                   # mice x LDs
    labs = mm.index.map(MOUSE_LAB).to_numpy()
    def eta2_all(lab_vec):
        G = pd.get_dummies(lab_vec).to_numpy(float)                          # mice x labs
        n = G.sum(0)
        means = (G.T @ X) / n[:, None]
        ss_b = (n[:, None] * (means - X.mean(0)) ** 2).sum(0)
        return ss_b / ((X - X.mean(0)) ** 2).sum(0)
    obs = eta2_all(labs)
    null = np.stack([eta2_all(rng.permutation(labs)) for _ in range(n_perm)])
    between = X.var(0)
    p = (1 + (null >= obs).sum(0)) / (1 + n_perm)
    t = pd.DataFrame({'LD': np.arange(1, X.shape[1] + 1), 'share_between_mouse_var': between / between.sum(),
                      'eta2_lab': obs, 'null_mean': null.mean(0), 'null_95': np.percentile(null, 95, axis=0),
                      'p': p, 'q': multipletests(p, method='fdr_bh')[1]})
    w = t['share_between_mouse_var']
    # the whole embedding: lab share of ALL between-mouse variance (each LD weighted by the variance it carries)
    tot_obs, tot_null = w.to_numpy() @ obs, null @ w.to_numpy()
    pd.DataFrame({'weighted_null': tot_null}).assign(observed=tot_obs).to_csv(LDVAR_DIR / 'ldvar_total_null.csv', index=False)
    print(f'lab share of all between-mouse variance: {tot_obs:.3f}, shuffled labs {tot_null.mean():.3f} '
          f'(95%: {np.percentile(tot_null, 95):.3f}), p = {(1 + (tot_null >= tot_obs).sum()) / (1 + n_perm):.4f}')
    print(f'\n=== lab share of between-mouse variance per LD ({len(mm)} mice, {len(np.unique(labs))} labs, {len(t)} LDs) ===')
    print(t.head(20).round(3).to_string(index=False))
    print(f"weighted by each LD's share of between-mouse variance: lab share {np.sum(w * t['eta2_lab']):.3f} "
          f"(shuffled labs: {np.sum(w * t['null_mean']):.3f}); first 10 LDs: "
          f"{np.sum(w[:10] * t['eta2_lab'][:10]) / w[:10].sum():.3f}")
    return t.merge(ldvar_mixed(rng, n_perm), on='LD')


def ldvar_mixed(rng, n_perm=5000, n_boot=N_BOOT):
    """The same question as a nested linear mixed model on the SESSION scores, LD ~ 1 + (1|lab) + (1|lab:mouse):
    lab share of an LD's inter-individual variance = s2_lab / (s2_lab + s2_mouse). Unlike eta^2 its chance level is 0
    (the components are unbiased, Henderson I -- variance_partition.py), and session noise sits in its own component
    instead of inflating the mouse means. Per LD the raw share can go negative (a lab component near 0); the figure
    clips it, the p and the pooled number use the raw one. Pooled: sum s2_lab over LDs / sum (s2_lab + s2_mouse), the
    trace, so each LD weighs by the variance it carries. Null: lab labels shuffled across mice (lab sizes kept).
    CI: labs resampled with replacement (the unit that limits precision, 11 labs). REML (statsmodels) per LD as a
    cross-check: it is non-negative by construction and weights unbalanced mice differently, so it can differ."""
    import variance_partition as vp
    import statsmodels.formula.api as smf
    cols = ['LD1'] + [c for c in LDA.columns if isinstance(c, (int, np.integer))]
    Y, labs, mice = LDA[cols].to_numpy(float), LDA['lab'].to_numpy(), LDA['mouse_name'].to_numpy()
    def comps(lab_vec, m=mice, Yv=Y):
        vc, _ = vp.nested_variance_components(Yv, lab_vec, m)
        return vc['sigma2_lab'].to_numpy(), vc['sigma2_mouse'].to_numpy(), vc['sigma2_session'].to_numpy()
    def shares(s_lab, s_mouse):
        return s_lab / (s_lab + s_mouse), s_lab.sum() / (s_lab + s_mouse).sum()
    s_lab, s_mouse, s_sess = comps(labs)
    obs, obs_tot = shares(s_lab, s_mouse)

    mouse_lab = MOUSE_LAB.loc[pd.unique(mice)]
    null, null_tot = [], []
    for _ in range(n_perm):
        sh = pd.Series(rng.permutation(mouse_lab.to_numpy()), index=mouse_lab.index)
        a, b = shares(*comps(sh[mice].to_numpy())[:2])
        null.append(a); null_tot.append(b)
    null, null_tot = np.stack(null), np.array(null_tot)

    boot, boot_tot, uniq = [], [], pd.unique(labs)
    for _ in range(n_boot):
        idx, bl, bm = [], [], []
        for rep, l in enumerate(rng.choice(uniq, len(uniq), replace=True)):
            where = np.flatnonzero(labs == l)       # a lab drawn twice becomes two labs with distinct mice
            idx.extend(where); bl.extend([f'{l}#{rep}'] * len(where)); bm.extend(f'{m}#{rep}' for m in mice[where])
        try:
            sl, sm, _ = comps(np.array(bl), np.array(bm), Y[idx])
        except (ValueError, AssertionError):        # fewer than 2 distinct labs drawn
            continue
        boot.append(shares(np.clip(sl, 0, None), sm)[0])       # per LD: as the figure draws it (clipped)
        boot_tot.append(shares(sl, sm)[1])                      # pooled: raw, as observed
    boot, boot_tot = np.stack(boot), np.array(boot_tot)

    reml = []
    for j in range(len(cols)):
        d = pd.DataFrame({'y': Y[:, j], 'lab': labs, 'mouse': mice})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = smf.mixedlm('y ~ 1', d, groups='lab', re_formula='1', vc_formula={'mouse': '0 + C(mouse)'}).fit(reml=True)
        sl, sm = float(f.cov_re.iloc[0, 0]), float(f.vcomp[0])
        reml.append(sl / (sl + sm))

    p = (1 + (null >= obs).sum(0)) / (1 + n_perm)
    # s2_mouse <= 0: the LD carries no inter-individual variance, the share is undefined (not 'all lab')
    undefined = s_mouse <= 0
    obs, p = np.where(undefined, np.nan, obs), np.where(undefined, np.nan, p)
    q = np.full(len(p), np.nan); q[~undefined] = multipletests(p[~undefined], method='fdr_bh')[1]
    t = pd.DataFrame({'LD': np.arange(1, len(cols) + 1), 'sigma2_lab': s_lab, 'sigma2_mouse': s_mouse,
                      'sigma2_session': s_sess, 'mixed_share': obs,
                      'mixed_share_clipped': np.where(undefined, np.nan,
                                                      np.clip(s_lab, 0, None) / (np.clip(s_lab, 0, None) + s_mouse)),
                      'mixed_ci_lo': np.percentile(boot, 2.5, axis=0), 'mixed_ci_hi': np.percentile(boot, 97.5, axis=0),
                      'mixed_null_mean': null.mean(0), 'mixed_null_95': np.percentile(null, 95, axis=0),
                      'mixed_p': p, 'mixed_q': q, 'mixed_share_reml': reml,
                      'mouse_over_session': s_mouse / s_sess})     # < 1: mice differ less than a mouse's sessions do
    p_tot = (1 + (null_tot >= obs_tot).sum()) / (1 + n_perm)
    pd.DataFrame({'null': null_tot}).assign(observed=obs_tot, ci_lo=np.percentile(boot_tot, 2.5),
                                            ci_hi=np.percentile(boot_tot, 97.5), p=p_tot
                                            ).to_csv(LDVAR_DIR / 'ldvar_mixed_total_null.csv', index=False)
    print(f'\n=== mixed model: lab share of each LD\'s inter-individual variance, s2_lab / (s2_lab + s2_mouse) ===')
    print(t.head(20).drop(columns=['sigma2_lab', 'sigma2_mouse', 'sigma2_session']).round(3).to_string(index=False))
    print(f'whole space (trace): lab share {obs_tot:.3f} [{np.percentile(boot_tot, 2.5):.3f}, '
          f'{np.percentile(boot_tot, 97.5):.3f}], shuffled labs {null_tot.mean():.3f} '
          f'(95%: {np.percentile(null_tot, 95):.3f}), p = {p_tot:.4f}; session share of all variance '
          f'{s_sess.sum() / (s_lab.clip(0) + s_mouse + s_sess).sum():.3f}')
    return t


SECTIONS = {'trial': section_trial, 'engage': section_engage, 'entropy': section_entropy, 'syllable': section_syllable,
            'neural': section_neural, 'neural_mouse': section_neural_mouse, 'horizon': section_horizon}


# FDR families. 'paper': every LD1 test the paper's figures show (the rows of fig_lab_controls' compact panels),
# significant or not: everything except the syllable-use summaries (descriptive, LD1 is built from them), the
# neural onset tests (not in fig_neural), two GLM-HMM measures fig_engagement computes but does not draw (brief
# lapses, posterior uncertainty) and the time-horizon rows, which are not LD1 tests (each timepoint against
# Proficient). 'all': literally every row of the table. 'section': within each section, as each figure corrects its
# own tests. The figure fills its dots by the 'paper' family, separately for each column (sessions, mice, lab removed).
# neural_mouse is out too: it re-tests the neural rows with the mouse as the unit (a robustness check, same names), so
# counting it would put every neural test in the family twice.
NOT_PAPER = lambda t: (t['section'].isin(['syllable', 'horizon', 'neural_mouse']) | t['correlate'].str.endswith('Response onset')
                       | t['correlate'].isin(['Short disengaged bouts (<=2 trials)', 'Posterior uncertainty',
                                             # panel I's split of syllable entropy: dropped from the paper (13-10-2026)
                                             'Syllable entropy, between modes', 'Syllable entropy, within modes',
                                             'Between-mode share of entropy']))


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q


def assemble():
    parts = [pd.read_csv(OUT_DIR / f'{s}.csv') for s in SECTIONS if (OUT_DIR / f'{s}.csv').exists()]
    t = pd.concat(parts, ignore_index=True)
    paper = ~NOT_PAPER(t)
    for col in ['p', 'p_lab', 'p_session']:
        if col not in t:
            continue
        q = col.replace('p', 'q', 1)
        t[f'{q}_all'] = bh(t[col])
        t[f'{q}_paper'] = np.where(paper, 0, np.nan)
        t.loc[paper, f'{q}_paper'] = bh(t.loc[paper, col])
        t[f'{q}_section'] = t.groupby('section')[col].transform(lambda x: pd.Series(bh(x), index=x.index))
        t[q] = t[f'{q}_paper']                       # the default: the paper family
    t['in_paper_family'] = paper
    t['survives_lab'] = (t['q_lab'] < 0.05) & (np.sign(t['rho_lab']) == np.sign(t['rho']))
    out = PI / 'results' / f'lab_controls{"" if METHOD == "spearman" else "_" + METHOD}_{date.today().strftime("%d-%m-%Y")}.csv'
    t.to_csv(out, index=False)
    print(f'\nwrote {out.relative_to(PI)} ({len(t)} rows; paper family {paper.sum()} tests)')
    print('lab-removed tests with q < 0.05:  ' + ' | '.join(
        f"{fam}: {(t[f'q_lab_{fam}'] < 0.05).sum()} of {t[f'q_lab_{fam}'].notna().sum()}" for fam in ['section', 'paper', 'all']))
    return t


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='all', choices=['all', 'ldvar', 'assemble'] + list(SECTIONS))
    ap.add_argument('--method', default='pearson', choices=['pearson', 'spearman'])
    a = ap.parse_args()
    METHOD = a.method
    if METHOD != 'spearman':     # its own folder; the neural rows are Pearson partial r either way
        OUT_DIR = PI / 'results' / f'lab_controls_{METHOD}'
        ENTROPY_CACHE = PI / 'results' / 'lab_controls' / 'entropy_sessions.csv'   # data, the same for both
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'LDA: {LDA_FILE.name} | {LDA["session"].nunique()} sessions, {LDA["mouse_name"].nunique()} mice, '
          f'{LDA["lab"].nunique()} labs ({LAB_SCHEME})')
    rng = np.random.default_rng(SEED)
    todo = list(SECTIONS) if a.only == 'all' else [a.only] if a.only in SECTIONS else []
    for s in todo:
        rows = SECTIONS[s](rng)
        if rows:
            pd.DataFrame(rows).to_csv(OUT_DIR / f'{s}.csv', index=False)
    if a.only in ('all', 'ldvar'):
        section_ldvar(rng).to_csv(LDVAR_DIR / 'ldvar.csv', index=False)
    if a.only in ('all', 'assemble'):
        assemble()
