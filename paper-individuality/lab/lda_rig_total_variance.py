"""
HOW MUCH OF THE TOTAL BETWEEN-MOUSE VARIANCE IS RIG GEOMETRY?
=============================================================
The earlier answer covered LD1 only. LD1 is one direction out of 30, so "rig does
not predict LD1" leaves open that rig loads on some other direction the LDA also
uses. This asks the question of the WHOLE space.

WHAT "TOTAL VARIANCE" MEANS HERE -- three nested choices, all reported, because
they answer different questions:

  A. original feature space (360 binarised syllable features, natural scale)
     "how much of the measured behaviour is rig?"  The honest denominator, but
     dominated by whichever features happen to have the largest variance.

  B. the 30 PCs, natural scale
     what survives the dimensionality reduction, still variance-weighted.

  C. the 30 PCs after z-scoring  <- THE ONE THE LDA ACTUALLY SEES
     the pipeline z-scores the PCs, which gives every PC equal weight regardless
     of how much variance it carried. So a rig effect sitting on PC 28 counts as
     much to the LDA as one on PC 1. This is the denominator that matters for
     "does rig contaminate the embedding".

UNIT AND VALIDATION. Mouse level throughout, matching the LD1 analysis: one row
per mouse, so a model cannot pass by memorising a mouse from its other sessions.
Multivariate CV R2 = 1 - sum_j SS_res(j) / sum_j SS_tot(j), summed over dimensions
in their own scale, so dimensions contribute in proportion to the variance they
actually hold. Negative = worse than predicting each dimension's mean.
"""
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GEO = ['nose_x', 'nose_y', 'tube_x', 'tube_y', 'pawNear_x', 'pawNear_y',
       'pawFar_x', 'pawFar_y', 'tube_len_px', 'pupil_diam_px', 'nose_to_tube_px']


def cv_r2_multi(X, Y, reps=40):
    """Multivariate out-of-sample R2, pooled over the columns of Y in their own scale."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    scores = []
    for rep in range(reps):
        pred = np.zeros_like(Y)
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
            D = np.column_stack([np.ones(len(tr)), X[tr]])
            B, *_ = np.linalg.lstsq(D, Y[tr], rcond=None)
            pred[te] = np.column_stack([np.ones(len(te)), X[te]]) @ B
        ss_res = ((Y - pred) ** 2).sum()
        ss_tot = ((Y - Y.mean(0)) ** 2).sum()
        scores.append(1 - ss_res / ss_tot)
    return float(np.mean(scores)), float(np.std(scores))


def in_sample_r2_multi(X, Y):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    D = np.column_stack([np.ones(len(X)), X])
    B, *_ = np.linalg.lstsq(D, Y, rcond=None)
    return 1 - ((Y - D @ B) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def main():
    import lda_allsessions_heldout as L

    # --- rebuild exactly the matrices the notebook's LDA consumes
    parts, names = [], None
    for dtype, fn in L.data_files.items():
        ss, mn = L.build_design_matrix(fn, n_paw_states=L.n_paw_states)
        parts.append(ss.add_prefix(f'{dtype}_'))
        names = mn if names is None else names
    session_syllables = pd.concat(parts, axis=1, join='inner')
    mice = pd.Series(names, index=session_syllables.index)
    counts = mice.value_counts()
    fit_mask = np.isin(mice.values, counts[counts >= L.MIN_SESSIONS].index)

    X_raw = np.asarray(session_syllables)                       # A: 360 features
    pca = PCA(n_components=min(fit_mask.sum(), X_raw.shape[1])).fit(X_raw[fit_mask])
    X_pc = pca.transform(X_raw)[:, :L.MIN_COMPONENTS]           # B: 30 PCs, natural scale
    X_z = StandardScaler().fit(X_pc[fit_mask]).transform(X_pc)  # C: what the LDA sees
    print(f"\n30 PCs retain {pca.explained_variance_ratio_[:L.MIN_COMPONENTS].sum()*100:.1f}% "
          f"of the original feature variance")

    # --- mouse level
    idx = session_syllables.index
    def to_mouse(M):
        return pd.DataFrame(M, index=idx).groupby(mice.values).mean()
    A, B, C = to_mouse(X_raw), to_mouse(X_pc), to_mouse(X_z)

    # --- geometry, rig, lab per mouse
    g = pd.read_csv(HERE / 'lda_rig_geometry_validation.csv', index_col=0)
    z = pd.read_csv(HERE / 'lda_rig_contrast.csv', index_col=0)
    gm = g.groupby(['lab', 'mouse_name']).agg({c: 'median' for c in GEO}).reset_index()
    gm = gm.join(z.groupby('mouse_name')['nose_to_pupil_px'].mean(), on='mouse_name')
    out = []
    for lab, q in gm.groupby('lab'):
        if len(q) < 2:
            out.append(q.assign(rig=1)); continue
        Z = linkage(q[['nose_x', 'nose_y']].to_numpy(), 'complete')
        out.append(q.assign(rig=fcluster(Z, 80.0, criterion='distance')))
    gm = pd.concat(out).dropna(subset=GEO + ['nose_to_pupil_px'])
    gm['labrig'] = gm.lab + '_r' + gm.rig.astype(str)
    gm = gm.set_index('mouse_name')

    keep = [m for m in A.index if m in gm.index]
    A, B, C, gm = A.loc[keep], B.loc[keep], C.loc[keep], gm.loc[keep]
    print(f"{len(keep)} mice with both behaviour and geometry\n")

    preds = {
        'camera aim (nose x,y)':   gm[['nose_x', 'nose_y']].values,
        'aim + tube position':     gm[['nose_x', 'nose_y', 'tube_x', 'tube_y']].values,
        'zoom (nose->pupil)':      gm[['nose_to_pupil_px']].values,
        'ALL measured geometry':   gm[GEO + ['nose_to_pupil_px']].values,
        'rig identity':            pd.get_dummies(gm.labrig, drop_first=True).astype(float).values,
        'lab identity':            pd.get_dummies(gm.lab, drop_first=True).astype(float).values,
    }
    spaces = [('A  360 raw features', A), ('B  30 PCs, natural scale', B),
              ('C  30 PCs z-scored  <- what the LDA sees', C)]
    rows = []
    for sname, Y in spaces:
        print("=" * 76)
        print(f"TOTAL BETWEEN-MOUSE VARIANCE IN:  {sname}")
        print("=" * 76)
        for pname, X in preds.items():
            ins = in_sample_r2_multi(X, Y.values)
            cv, sd = cv_r2_multi(X, Y.values)
            k = np.asarray(X).shape[1]
            print(f"  {pname:24s} k={k:2d}   in-sample={ins:+.3f}   CV R2 = {cv:+.3f} +/- {sd:.3f}")
            rows.append(dict(space=sname, predictor=pname, k=k, in_sample=ins, cv=cv, cv_sd=sd))
        print()
    pd.DataFrame(rows).to_csv(HERE / 'lda_rig_total_variance.csv', index=False)
    print(f"saved {HERE / 'lda_rig_total_variance.csv'}")

    # --- for contrast: the same, per LD, to show where LD1 sits
    print("=" * 76)
    print("FOR CONTRAST -- per discriminant, CV R2 from ALL measured geometry")
    print("=" * 76)
    clustered = L.main()
    lda = clustered.set_index('session')
    ldm = lda.groupby('mouse_name')[[0, 1, 2]].mean().loc[keep]
    Xg = preds['ALL measured geometry']
    for j, nm in enumerate(['LD1', 'LD2', 'LD3']):
        cv, sd = cv_r2_multi(Xg, ldm[[j]].values)
        print(f"  {nm}: CV R2 = {cv:+.3f} +/- {sd:.3f}")


if __name__ == '__main__':
    main()
