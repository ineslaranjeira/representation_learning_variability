"""
THE LAB EFFECT IN THREE FEATURE SETS, ENCODED THE SAME WAY
===========================================================
Syllables, lightningAction paw states and wheel speed, all as
(sessions x [4 epochs x 10 bins x channels]) matrices built from the SAME trials, the SAME
epoch boundaries and the SAME bins -- then the same nested lab / mouse / session partition
on each. Differences between the three are then differences between the MEASUREMENTS, not
between preprocessing choices.

ENCODING (mirrors functions.binarize, which is what the syllables already go through)

  syllables        7 paw one-hot (one state dropped as reference) + whisk + lick
                   = 9 per timestep  x 40 timesteps = 360
  lightningAction  2 paws x (5 states - 1 reference) = 8 per timestep x 40 = 320
                   `background` is the dropped reference: it means "paw not localised",
                   so it is the non-behavioural level, and it is where the lab artifact sat
                   (wittenlab 2.6% against 0.05-0.2% elsewhere)
  wheel            1 continuous channel per timestep x 40 = 40, z-scored per column
                   (the treatment functions.build_design_matrix gives its 'raw' branch)

A missing bin sets that timestep's whole feature block to NaN, and the per-session average is
a nanmean -- so missing data is skipped rather than counted as zero, exactly as for the
syllables.

WHY WHEEL IS THE DECISIVE ROW: it is the rotary encoder, in physical units, with no camera in
the path. A lab effect present in the video-derived features but absent in the wheel cannot
be something the animals did.
"""
import sys
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / '4_mice'), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import functions
import variance_partition as vp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
N_BINS = 10
LA_STATES = ['background', 'still', 'move', 'wheel_turn', 'groom']


def _pivot(seq, value='binned_sequence'):
    """trial-epoch rows -> (n_trials, 40), keeping only trials complete in all four epochs.
    Same pivot-then-dropna rule build_design_matrix uses, so the trial sets match."""
    w = (seq.pivot(index=['mouse_name', 'session', 'sample', 'trial_type'],
                   columns='broader_label', values=value)
         .reset_index().dropna().sort_values('session'))
    X = np.vstack(w[EPOCHS].apply(lambda r: np.hstack(r), axis=1))
    return w, X


def _per_session(X, sessions, cols=None):
    df = pd.DataFrame(X)
    df['session'] = np.asarray(sessions)
    out = df.groupby('session', sort=False)[list(range(X.shape[1]))].mean()
    if cols is not None:
        out.columns = cols
    return out


def onehot_categorical(seqs, n_states, drop_state=0):
    """(n_trials, T) integer states -> (n_trials, T*(n_states-1)) one-hot, reference dropped.
    NaN in a timestep blanks that timestep's whole block, as functions.binarize does."""
    n, T = seqs.shape
    keep = [s for s in range(n_states) if s != drop_state]
    out = np.zeros((n, T * len(keep)))
    for t in range(T):
        col = seqs[:, t]
        nan = ~np.isfinite(col)
        val = ~nan
        if val.any():
            lab = col[val].astype(int)
            for j, s in enumerate(keep):
                out[np.where(val)[0], t * len(keep) + j] = (lab == s).astype(float)
        if nan.any():
            out[nan, t * len(keep):(t + 1) * len(keep)] = np.nan
    return out


def build_all():
    sets = {}
    syl, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    sets['syllables'] = syl
    mouse = pd.Series(syl.index.map(dd[['mouse_name', 'session']].drop_duplicates()
                                    .set_index('session')['mouse_name']), index=syl.index)

    # ---- wheel: continuous, z-scored per column across sessions
    wh = pd.read_parquet(HERE / 'wheel_10bin_sequences.pqt')
    w, X = _pivot(wh)
    X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    sets['wheel'] = _per_session(X, w['session'])

    # ---- lightningAction: both paws, one-hot with `background` dropped
    la = pd.read_parquet(HERE / 'lightningaction_10bin_sequences.pqt')
    blocks, sess = [], None
    for p in ['paw_r', 'paw_l']:
        wp, Xp = _pivot(la[la['paw'] == p].drop(columns='paw'))
        blocks.append((wp, Xp))
    # keep the trials present for BOTH paws, in the same order
    keys = ['mouse_name', 'session', 'sample', 'trial_type']
    common = blocks[0][0][keys].merge(blocks[1][0][keys], on=keys)
    parts = []
    for (wp, Xp) in blocks:
        idx = wp.reset_index(drop=True).merge(common.assign(_k=1), on=keys, how='inner').index
        wp2 = wp.reset_index(drop=True)
        mask = wp2[keys].apply(tuple, axis=1).isin(set(common[keys].apply(tuple, axis=1)))
        parts.append((wp2[mask], Xp[mask.to_numpy()]))
    sess = parts[0][0]['session']
    Xla = np.hstack([onehot_categorical(P, len(LA_STATES), drop_state=0) for _, P in parts])
    sets['lightningAction'] = _per_session(Xla, sess)
    return sets, mouse


def _embed(M, n_pc=30):
    A = np.asarray(M, float)
    A = np.nan_to_num(A, nan=np.nanmean(A))
    if A.shape[1] <= n_pc:
        return StandardScaler().fit_transform(A)
    return StandardScaler().fit_transform(PCA(min(A.shape)).fit_transform(A)[:, :n_pc])


def id_mouse(norm, y, n_per=3, reps=3, seed=0):
    """Leave-one-session-out mouse identification, balanced training draw -- the same rule
    the identity pipeline uses, so these numbers sit on the same scale as its LOO score."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(reps):
        hits = []
        for t in range(n):
            tr = np.setdiff1d(np.arange(n), t)
            Xt, yt = norm[tr], y[tr]
            idx = []
            for m in np.unique(yt):
                mi = np.where(yt == m)[0]
                idx.extend(rng.choice(mi, min(n_per, len(mi)), replace=False))
            idx = np.sort(np.array(idx))
            k = len(np.unique(yt[idx]))
            if k < 2:
                continue
            clf = LinearDiscriminantAnalysis(priors=np.ones(k) / k, n_components=min(30, k - 1))
            hits.append(clf.fit(Xt[idx], yt[idx]).predict(norm[t:t + 1])[0] == y[t])
        out.append(np.mean(hits))
    return float(np.mean(out))


def id_lab_heldout_mouse(norm, y_lab, groups):
    """Lab decoded from a mouse the classifier has never seen."""
    hits = []
    for g in pd.unique(groups):
        te = groups == g
        tr = ~te
        k = len(np.unique(y_lab[tr]))
        if k < 2:
            continue
        clf = LinearDiscriminantAnalysis(priors=np.ones(k) / k, n_components=min(30, k - 1))
        hits.extend(list(clf.fit(norm[tr], y_lab[tr]).predict(norm[te]) == y_lab[te]))
    return float(np.mean(hits))


def main():
    sets, mouse = build_all()
    common = set.intersection(*[set(v.index) for v in sets.values()])
    order = [s for s in sets['syllables'].index if s in common]
    lab_all = functions.lab_labels(pd.Index(order), mouse_names=mouse[order], verbose=False)
    ms, lb = mouse[order], lab_all
    print(f'COMMON SESSIONS: {len(order)}   mice {ms.nunique()}   labs {lb.nunique()}')
    for k, v in sets.items():
        print(f'  {k:16s} {v.loc[order].shape[1]:4d} features')

    print('\nNESTED PARTITION (share of variance)')
    print(f'  {"feature set":17s} {"lab":>7s} {"mouse":>7s} {"session":>8s}   '
          f'{"lab 95% CI":>16s}   {"eta2 (null)":>16s}   p')
    rows = []
    for k, v in sets.items():
        D = v.loc[order]
        vc, _ = vp.nested_variance_components(D, lb, ms)
        s = vc.clip(lower=0).sum()
        s = s / s.sum()
        b = vp.bootstrap_shares(D, lb, ms, n_boot=200, seed=0)
        ci = b['sigma2_lab'].quantile([.025, .975]).to_numpy()
        o, null, pv = vp.permutation_null_eta2(D, lb, ms, n_perm=2000, seed=0)
        print(f'  {k:17s} {s["sigma2_lab"]:7.3f} {s["sigma2_mouse"]:7.3f} '
              f'{s["sigma2_session"]:8.3f}   [{ci[0]:.3f}, {ci[1]:.3f}]   '
              f'{o:.3f} ({null.mean():.3f})   {pv:.4f}')
        rows.append(dict(feature_set=k, lab=s['sigma2_lab'], mouse=s['sigma2_mouse'],
                         session=s['sigma2_session'], lab_lo=ci[0], lab_hi=ci[1],
                         eta2=o, eta2_null=null.mean(), p=pv, n_features=D.shape[1]))
    pd.DataFrame(rows).to_csv(HERE / 'lab_effect_three_featuresets.csv', index=False)

    print('\nDECODING CONTROL -- how much of each signal is identity, and how much is lab')
    print(f'  {"feature set":17s} {"mouse ID":>9s} {"chance":>8s} {"lab from held-out mouse":>24s} {"chance":>8s}')
    y_mouse = pd.factorize(ms)[0]
    y_lab = pd.factorize(lb)[0]
    for k, v in sets.items():
        nrm = _embed(v.loc[order])
        a_m = id_mouse(nrm, y_mouse)
        a_l = id_lab_heldout_mouse(nrm, y_lab, ms.to_numpy())
        print(f'  {k:17s} {a_m:9.3f} {1 / ms.nunique():8.3f} {a_l:24.3f} {1 / lb.nunique():8.3f}')
        for r in rows:
            if r['feature_set'] == k:
                r['mouse_id_acc'] = a_m
                r['lab_acc_heldout_mouse'] = a_l
    pd.DataFrame(rows).to_csv(HERE / 'lab_effect_three_featuresets.csv', index=False)

    print('\nLAB SHARE PER EPOCH')
    print(f'  {"feature set":17s} ' + ' '.join(f'{e:>16s}' for e in EPOCHS))
    for k, v in sets.items():
        D = v.loc[order]
        per_step = D.shape[1] // (len(EPOCHS) * N_BINS)
        ep = np.repeat(np.arange(len(EPOCHS)), N_BINS * per_step)
        vc, _ = vp.nested_variance_components(D, lb, ms)
        cl = vc.clip(lower=0)
        line = []
        for e in range(len(EPOCHS)):
            m = ep == e
            t = cl[m].sum()
            line.append(f'{t["sigma2_lab"] / t.sum():16.3f}')
        print(f'  {k:17s} ' + ' '.join(line))
    print(f'\nsaved {HERE / "lab_effect_three_featuresets.csv"}')


if __name__ == '__main__':
    main()
