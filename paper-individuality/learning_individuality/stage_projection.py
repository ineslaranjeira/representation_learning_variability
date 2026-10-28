"""
The training stages projected into the proficient 1-camera LDA frame.
=====================================================================
The frame is the one of neuromodulators/1_proficient_1cam_ld1 (proficient_1cam.design / embed: syllable_data
.SYLLABLES, shrinkage mouse-weighted LDA, signs aligned to the paper's 2-camera LD1 / LD2), fitted on the
proficient sessions only and never refitted. Each training stage (Early, Late, Pre-rec; QC-sheet exclusions,
missing-data screen, every session kept) is projected into it.

`stage_table` measures, per stage, against the same mice's proficient positions:
  offset            length of the mean displacement (stage - proficient, same mice), all LDs
  offset_LD1-3      share of that displacement (squared) on LD1-3
  LDk shift         mean displacement on LDk, in units of the proficient between-mouse SD
  r LDk             correlation across mice of the stage position with the proficient position
  own mouse         each stage session assigned to the nearest proficient mouse centroid (all mice classes)
"""
import sys
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'neuromodulators'))
import proficient_1cam as pc                    # noqa: E402
import syllable_data as sd                      # noqa: E402
from session_filters import exclusions_by_timepoint   # noqa: E402

STAGES = ['Early', 'Late', 'Pre-rec']


def frame():
    X, mice, _ = pc.design(verbose=False)
    emb, sc = pc.embed(X, mice)
    pc.orient(sc, emb, pc.two_cam_lda().groupby('mouse_name')[['LD1', 'LD2']].mean(), cols=('LD1', 'LD2'))
    return emb, sc


def project_stages(emb, stages=STAGES, qc='filtered_out'):
    """Session scores of every stage in the frame, one frame with a 'stage' column."""
    ex = exclusions_by_timepoint(qc)
    out = []
    for t in stages:
        X, m = pc.lc.proficient_design(sd.syllable_file(t, **sd.SYLLABLES), ex[t], *sd.n_states(**sd.SYLLABLES),
                                       min_sessions_per_mouse=1, verbose=False)
        Z = emb.transform(X.to_numpy())
        out.append(pd.DataFrame(Z, index=X.index, columns=[f'LD{i + 1}' for i in range(Z.shape[1])])
                   .assign(mouse_name=m.to_numpy(), stage=t))
    return pd.concat(out)


def stage_table(sc, P, show=('LD1', 'LD2', 'LD3')):
    lds = [c for c in P if c.startswith('LD')]
    B = sc.groupby('mouse_name')[lds].mean()
    sd_ = B.std()
    rows = []
    for t, Z in P.groupby('stage', sort=False):
        M = Z.groupby('mouse_name')[lds].mean()
        M = M[M.index.isin(B.index)]
        v = (M - B.loc[M.index]).mean()
        D = ((Z[lds].to_numpy()[:, None, :] - B.to_numpy()[None]) ** 2).sum(2)
        rows.append(dict(stage=t, mice=len(M), sessions=len(Z), offset=np.linalg.norm(v),
                         **{'offset_LD1-3': float((v[lds[:3]] ** 2).sum() / (v ** 2).sum())},
                         **{f'{c} shift': v[c] / sd_[c] for c in show},
                         **{f'r {c}': np.corrcoef(M[c], B.loc[M.index, c])[0, 1] for c in show},
                         own_mouse=float(np.mean(B.index[D.argmin(1)] == Z['mouse_name'].to_numpy()))))
    t = pd.DataFrame(rows).set_index('stage')
    t.attrs['typical'] = float(np.linalg.norm(B - B.mean(), axis=1).mean())
    t.attrs['chance'] = 1 / len(B)
    return t


def ld1_retention(sc, P, n_boot=2000, n_perm=10000, seed=0):
    """Per stage: does a mouse keep its LD1 place? Mouse-mean stage LD1 vs the same mouse's proficient LD1.
    Spearman rho (as fig_time_horizon panel e) with BH FDR over the stages, a 95% CI resampling mice, and the
    lab-removed version (ranks residualised on lab dummies, p from permuting within lab), because about half of
    LD1's between-mouse variance is between labs."""
    from scipy.stats import spearmanr, rankdata
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(seed)
    prof = sc.groupby('mouse_name').agg(prof=('LD1', 'mean'), lab=('lab', 'first'))
    rows = []
    for t, Z in P.groupby('stage', sort=False):
        d = Z.groupby('mouse_name')['LD1'].mean().rename('stage').to_frame().join(prof, how='inner')
        x, y, labs = d['prof'].to_numpy(), d['stage'].to_numpy(), d['lab'].to_numpy()
        r = spearmanr(x, y)
        boot = []
        for _ in range(n_boot):
            i = rng.integers(0, len(d), len(d))
            boot.append(spearmanr(x[i], y[i]).statistic)
        D = np.column_stack([np.ones(len(d)), pd.get_dummies(labs, drop_first=True).to_numpy(float)])
        H = D @ np.linalg.pinv(D)
        rx, ry = rankdata(x), rankdata(y)
        ex, ey = rx - H @ rx, ry - H @ ry
        r_lab = np.corrcoef(ex, ey)[0, 1]
        null = np.empty(n_perm)
        for k in range(n_perm):
            xp = rx.copy()
            for l in np.unique(labs):
                idx = np.flatnonzero(labs == l)
                xp[idx] = rng.permutation(xp[idx])
            e = xp - H @ xp
            null[k] = np.corrcoef(e, ey)[0, 1]
        rows.append(dict(stage=t, n_mice=len(d), rho=r.statistic, p=r.pvalue,
                         lo=np.nanpercentile(boot, 2.5), hi=np.nanpercentile(boot, 97.5),
                         rho_lab=r_lab, p_lab=(1 + np.sum(np.abs(null) >= abs(r_lab))) / (1 + n_perm),
                         slope=np.polyfit(x, y, 1)[0]))
    t = pd.DataFrame(rows).set_index('stage')
    t['q'] = multipletests(t['p'], method='fdr_bh')[1]
    t['q_lab'] = multipletests(t['p_lab'], method='fdr_bh')[1]
    return t
