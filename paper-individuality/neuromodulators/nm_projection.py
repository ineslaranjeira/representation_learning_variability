"""
The neuromodulator sessions projected into the BWM proficient 1-camera embedding.
==================================================================================
The frame is 1_proficient_1cam_ld1's (proficient_1cam.design / embed, signs aligned to the 2-camera LD1 /
LD2) and is never refitted. Projected into it, through the same stages paw states and the same pipeline:
  NM         the neuromodulator sessions (nm_lda.design with the stages syllables), ephys and / or biased
             protocol (project(protocol=...)); each session's protocol in the column 'protocol'
  Pre-rec    the BWM mice's own biased sessions before recording: same mice and same recording rig as the
             frame, but the camera at 30 Hz like NM's, 100 ms lick HMM like NM's -- the yardstick for the
             offset that 30 Hz acquisition (+ a few days, biased protocol) produces

Distances are Euclidean in LD coordinates, the LDA's own metric (within-mouse scatter whitened).
"""
import numpy as np
import pandas as pd

import proficient_1cam as pc
import nm_lda as n
import syllable_data as sd
from session_filters import exclusions_by_timepoint

GROUPS = ['BWM proficient', 'Pre-rec (same mice)', 'NM']


def frame():
    """(embedding, session scores with lab) of the BWM proficient 1-camera data, signs as notebook 1."""
    X, mice, _ = pc.design(verbose=False)
    emb, sc = pc.embed(X, mice)
    pc.orient(sc, emb, pc.two_cam_lda().groupby('mouse_name')[['LD1', 'LD2']].mean(), cols=('LD1', 'LD2'))
    return emb, sc


def _scores(emb, X, mice, group):
    Z = emb.transform(X.to_numpy())
    d = pd.DataFrame(Z, index=X.index, columns=[f'LD{i + 1}' for i in range(Z.shape[1])])
    return d.assign(mouse_name=mice.to_numpy(), group=group)


def project(emb, sc, protocol='ephys', min_sessions=3):
    """Session scores of all three groups in the BWM frame (one frame, rows = sessions). protocol: which NM
    sessions ('ephys', 'biased', None = both); their protocol is kept in the column 'protocol'."""
    Xn, mn = n.design(protocol, syllables=sd.SYLLABLES, verbose=False, min_sessions_per_mouse=min_sessions)
    Xp, mp = pc.lc.proficient_design(sd.syllable_file('Pre-rec'), exclusions_by_timepoint('filtered_out')['Pre-rec'],
                                     *sd.n_states(**sd.SYLLABLES), min_sessions_per_mouse=1, verbose=False)
    nm = _scores(emb, Xn, mn, GROUPS[2])
    nm['protocol'] = n.recordings()['protocol'].reindex(nm.index).to_numpy()
    return pd.concat([sc.drop(columns='lab').assign(group=GROUPS[0]), _scores(emb, Xp, mp, GROUPS[1]), nm])


def lds(A):
    return [c for c in A if c.startswith('LD')]


def protocol_consistency(A, show=('LD1', 'LD2', 'LD3', 'LD4', 'LD5')):
    """For the NM mice with sessions of both protocols: do their ephys and biased sessions sit in the same place
    in the BWM frame? Per LD, r of the two protocol means across mice and the mean biased - ephys shift (in BWM
    between-mouse SD units); and cross-protocol identification: each biased session to the nearest of the
    mice's EPHYS centroids, against chance."""
    N = A[A['group'] == GROUPS[2]]
    both = N.groupby('mouse_name')['protocol'].nunique()
    both = both.index[both == 2]
    N = N[N['mouse_name'].isin(both)]
    P = N.groupby(['protocol', 'mouse_name'])[lds(A)].mean()
    E, Bz = P.loc['ephys'], P.loc['biased'].reindex(P.loc['ephys'].index)
    sd_ = A[A['group'] == GROUPS[0]].groupby('mouse_name')[lds(A)].mean().std()
    per_ld = pd.DataFrame({'r(ephys, biased) across mice': [np.corrcoef(E[c], Bz[c])[0, 1] for c in show],
                           'biased - ephys (BWM SD)': [((Bz[c] - E[c]).mean()) / sd_[c] for c in show]}, index=list(show))
    cols = lds(A)
    Zb = N[N['protocol'] == 'biased']
    D = ((Zb[cols].to_numpy()[:, None, :] - E[cols].to_numpy()[None]) ** 2).sum(2)
    ident = pd.Series({'mice with both protocols': len(both), 'biased sessions': len(Zb),
                       'biased session -> own ephys centroid': float(np.mean(E.index[D.argmin(1)] == Zb['mouse_name'].to_numpy())),
                       'chance': 1 / len(both)})
    return per_ld, ident


def mouse_means(A):
    return A.groupby(['group', 'mouse_name'])[lds(A)].mean()


def group_table(A, lab, show=('LD1', 'LD2', 'LD3', 'LD4', 'LD5')):
    """Per group and per BWM lab: mean and SD of the mouse means on each LD in units of the BWM between-mouse
    SD, and the distance of the group centroid from the BWM centre (all LDs). Expected distance of a random
    group of n BWM mice: about (mean BWM mouse distance) / sqrt(n)."""
    MM = mouse_means(A)
    B = MM.loc[GROUPS[0]]
    c0, sd_ = B.mean(), B.std()
    typical = np.linalg.norm(B - c0, axis=1).mean()
    rows = []
    groups = [(f'lab: {l}', g) for l, g in B.groupby(lab.reindex(B.index))] + \
             [(g, MM.loc[g]) for g in GROUPS[1:]]
    for name, g in groups:
        rows.append(dict(group=name, n_mice=len(g), centroid_dist=np.linalg.norm(g.mean() - c0),
                         chance_dist=typical / np.sqrt(len(g)) * np.sqrt(max(len(B) - len(g), 0) / (len(B) - 1)),
                         **{f'{c} mean': (g[c].mean() - c0[c]) / sd_[c] for c in show},
                         **{f'{c} sd': g[c].std() / sd_[c] for c in show}))
    return pd.DataFrame(rows).set_index('group'), typical


def offset_decomposition(A, lab):
    """The NM group offset (from the BWM centre) against two candidate causes: the Pre-rec offset (Pre-rec minus
    proficient, same mice and rig: 30 Hz acquisition + a few days) and mainenlab's own offset among the BWM mice."""
    MM = mouse_means(A)
    B = MM.loc[GROUPS[0]]
    c0 = B.mean()
    v_nm = MM.loc[GROUPS[2]].mean() - c0
    P = MM.loc[GROUPS[1]]
    v_rig = (P - B.loc[P.index]).mean()
    v_lab = B[lab.reindex(B.index) == 'mainenlab'].mean() - c0
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))   # noqa: E731
    coef = np.linalg.lstsq(np.c_[v_rig, v_lab], v_nm, rcond=None)[0]
    fit = np.c_[v_rig, v_lab] @ coef
    return pd.Series({'|NM offset|': np.linalg.norm(v_nm), '|Pre-rec offset|': np.linalg.norm(v_rig),
                      '|mainenlab offset|': np.linalg.norm(v_lab), 'cos(NM, Pre-rec)': cos(v_nm, v_rig),
                      'cos(NM, mainenlab)': cos(v_nm, v_lab), 'cos chance SD': 1 / np.sqrt(len(v_nm)),
                      'NM ~ a Pre-rec: a': coef[0], 'NM ~ b mainenlab: b': coef[1],
                      'R2 of NM offset': 1 - ((v_nm - fit) ** 2).sum() / (v_nm ** 2).sum()})


def loso_nearest_centroid(Z, y):
    """Leave-one-session-out: each session to the nearest mouse centroid of the other sessions."""
    hit = []
    for i in range(len(Z)):
        m = np.ones(len(Z), bool)
        m[i] = False
        cls = np.unique(y[m])
        C = np.stack([Z[m][y[m] == c].mean(0) for c in cls])
        hit.append(cls[((C - Z[i]) ** 2).sum(1).argmin()] == y[i])
    return float(np.mean(hit))


def nm_identification(A, dims=(3, 10, 20, None), n_shuffle=20, seed=0):
    """NM mouse identity in the BWM frame, first d LDs (None = all)."""
    Z = A[A['group'] == GROUPS[2]]
    y = Z['mouse_name'].to_numpy()
    cols = lds(A)
    rng = np.random.default_rng(seed)
    out = {f'LD1-{d or len(cols)}': loso_nearest_centroid(Z[cols[:d]].to_numpy(), y) for d in dims}
    out['shuffled (all LDs)'] = np.mean([loso_nearest_centroid(Z[cols].to_numpy(), rng.permutation(y))
                                         for _ in range(n_shuffle)])
    out['chance'] = 1 / len(np.unique(y))
    return pd.Series(out)


def prerec_identification(A, dims=(3, 10, None)):
    """Each Pre-rec session assigned to the nearest BWM proficient mouse centroid (all BWM mice are classes)."""
    MM = mouse_means(A)
    B = MM.loc[GROUPS[0]]
    Zp = A[A['group'] == GROUPS[1]]
    cols = lds(A)
    out = {}
    for d in dims:
        C = B[cols[:d]]
        D = ((Zp[cols[:d]].to_numpy()[:, None, :] - C.to_numpy()[None]) ** 2).sum(2)
        out[f'LD1-{d or len(cols)}'] = float(np.mean(C.index[D.argmin(1)] == Zp['mouse_name'].to_numpy()))
    out['chance'] = 1 / len(B)
    out['sessions'] = len(Zp)
    return pd.Series(out)
