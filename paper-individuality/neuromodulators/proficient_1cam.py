"""
The proficient BWM mouse embedding on 1-CAMERA syllables, and the tools to read its LD1.
=======================================================================================
The syllables are the ones learning_individuality uses (syllable_data.SYLLABLES: left paw + wheel
states fitted on the four stages, k = 8, whisk, lick, 10 bins per trial phase), so this is the frame
every 1-camera dataset (training stages, and later the neuromodulator mice) can be projected into.
The embedding is lda_shrinkage_core's: proficient_design() (QC sheet, missing-data screen, >= 3
sessions per mouse) -> MouseEmbedding (mouse-weighted shrinkage LDA, one-SE shrinkage).

`as_lab_controls_lda` writes the result in the layout of clustering/data_files/mouse_LDA_*, so the
paper's LD1 correlate battery (lab_controls.py) runs on it unchanged: `use_in_lab_controls` swaps it
in for the 2-camera file.
"""
import os
import sys
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent
for p in [PI, PI / 'learning_individuality', PI / '4_mice']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import syllable_data as sd                      # noqa: E402
import lda_shrinkage_core as lc                 # noqa: E402
from session_filters import exclusions_by_timepoint   # noqa: E402

OUT = PI / 'data' / 'neuromodulators' / 'bwm_1cam_embedding'
TWO_CAM_LDA = PI / 'clustering' / 'data_files' / 'mouse_LDA_5_bins_raw_shrink0.5_wmouse_360_02-10-2026'
ALYX_HOFER = {'SWC_043', 'SWC_060', 'SWC_061'}   # as lab_controls.py: Alyx puts these in hoferlab


def cached(name, make):
    """Load OUT/<name>.pkl, or make() it and save it there."""
    import pickle
    f = OUT / f'{name}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    OUT.mkdir(parents=True, exist_ok=True)
    obj = make()
    pickle.dump(obj, open(f, 'wb'))
    return obj


def vigor_tables():
    """The proficient per-session vigor and epoch-timing tables of vigor/lda (read, not recomputed), and its
    mouse-level per-epoch speeds plus the pre-quiescence / quiescence gate (log ratio)."""
    V = PI / 'vigor' / 'lda'
    ses = pd.read_csv(V / 'vigor_sessions_states.csv').set_index('session')
    tim = pd.read_csv(V / 'epoch_timing.csv').set_index('session')
    mm = pd.read_csv(V / 'vigor_mouse_level.csv').set_index('mouse_name')
    return ses, tim, epoch_speeds(mm)


def epoch_speeds(t):
    """Per-epoch wheel / paw speed columns of a vigor table, plus each signal's gate."""
    E = t[[c for c in t if c.startswith(('wheel_speed_', 'paw_speed_'))]].copy()
    for s in ['wheel', 'paw']:
        E[f'{s}_gate'] = np.log(E[f'{s}_speed_Pre-quiescence'] / E[f'{s}_speed_Quiescence'])
    return E


def design(syllables=None, qc='filtered_out', **kw):
    """(session x feature frame, mouse per session, syllable file) of the proficient 1-camera data."""
    s = dict(sd.SYLLABLES, **(syllables or {}))
    f = sd.syllable_file('Proficient', **s)
    k, kw_ = sd.n_states(**s)
    X, mice = lc.proficient_design(f, exclusions_by_timepoint(qc)['Proficient'], k, kw_, **kw)
    return X, mice, f


def labs(sessions, mice):
    """Session -> lab from the QC sheet's rig name, hoferlab split from mrsicflogel as lab_controls does."""
    from functions import lab_labels
    lab = lab_labels(sessions, mouse_names=mice, verbose=False)
    lab = pd.Series(np.asarray(lab), index=list(sessions))
    return pd.Series([('hoferlab' if m in ALYX_HOFER else 'mrsicflogellab') if l == 'mrsicflogel' else l
                      for m, l in zip(mice, lab)], index=lab.index)


def embed(X, mice, seed=0):
    """MouseEmbedding on the design; returns (embedding, session-score frame LD1.., mouse_name, lab)."""
    emb = lc.MouseEmbedding(X.to_numpy(), mice.to_numpy(), rng=np.random.default_rng(seed))
    Z = emb.transform(X.to_numpy())
    sc = pd.DataFrame(Z, index=X.index, columns=[f'LD{i + 1}' for i in range(Z.shape[1])])
    sc['mouse_name'] = mice.to_numpy()
    sc['lab'] = labs(X.index, mice.to_numpy()).to_numpy()
    return emb, sc


def orient(sc, emb, ref, cols=('LD1', 'LD2', 'LD3')):
    """Flip the sign of each LD so it correlates positively (mouse means) with ref[col] -- an
    eigenvector's sign is arbitrary. ref: per-mouse frame. Returns the signs applied."""
    signs = {}
    mm = sc.groupby('mouse_name')[list(cols)].mean().join(ref, rsuffix='_ref', how='inner')
    for i, c in enumerate(cols):
        r = ref.columns[i] if c not in ref else c
        s = np.sign(np.corrcoef(mm[c], mm[r if r in mm else c + '_ref'])[0, 1]) or 1.0
        sc[c] *= s
        emb.W[:, int(c[2:]) - 1] *= s
        signs[c] = s
    return signs


def as_lab_controls_lda(sc):
    """Session scores in the layout lab_controls.read_lda() expects: 'LD1', then int columns 1.. for
    LD2.., mouse_name, lab, session."""
    lds = [c for c in sc if c.startswith('LD')]
    out = sc[lds].copy()
    out.columns = ['LD1'] + list(range(1, len(lds)))
    out['mouse_name'] = sc['mouse_name'].to_numpy()
    out['lab'] = sc['lab'].to_numpy()
    out['session'] = sc.index.to_numpy()
    return out.reset_index(drop=True)


def paper_lab_controls():
    """PI/lab_controls.py (4_mice/ has its own lab_controls.py, which a plain import would find first)."""
    import importlib.util
    if 'paper_lab_controls' not in sys.modules:
        spec = importlib.util.spec_from_file_location('paper_lab_controls', PI / 'lab_controls.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules['paper_lab_controls'] = mod
        spec.loader.exec_module(mod)
    return sys.modules['paper_lab_controls']


def two_cam_lda():
    """The paper's 2-camera embedding (lab_controls.LDA_FILE), session scores LD1..LD3."""
    t = paper_lab_controls().LDA.rename(columns={1: 'LD2', 2: 'LD3'})
    return t.set_index('session')[['LD1', 'LD2', 'LD3', 'mouse_name']]


def use_in_lab_controls(lda_table, out_dir):
    """Point lab_controls' module globals at another embedding (its functions read them at call time)."""
    lab = paper_lab_controls()
    lab.LDA = lda_table
    lab.MOUSE_LAB = lda_table.drop_duplicates('mouse_name').set_index('mouse_name')['lab']
    lab.SESSION_LD1 = lda_table.set_index('session')['LD1']
    lab.SESSION_MOUSE = lda_table.set_index('session')['mouse_name']
    lab.OUT_DIR = pathlib.Path(out_dir)
    lab.LDVAR_DIR = pathlib.Path(out_dir)
    lab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    return lab


def loading_profile(sc, X, ld='LD1', n_paw_states=8, n_wheel_states=0):
    """Structure coefficients at mouse level: Pearson r of each (bin, feature) mouse-mean occupancy
    with the mouse-mean LD. Reference paw state restored (sd.reconstruct). -> (bins x features)."""
    R = sd.reconstruct(X.to_numpy(), n_paw_states, n_wheel_states)         # sessions x bins x feats
    mice = sc['mouse_name'].to_numpy()
    um = pd.unique(mice)
    M = np.stack([R[mice == m].mean(0) for m in um])
    x = sc.groupby('mouse_name')[ld].mean().reindex(um).to_numpy()
    F = M.reshape(len(um), -1)
    Fc, xc = F - F.mean(0), x - x.mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (xc @ Fc) / (np.linalg.norm(xc) * np.linalg.norm(Fc, axis=0))
    return pd.DataFrame(r.reshape(M.shape[1:]), columns=sd.feature_names(n_paw_states, n_wheel_states))


def mouse_corr(sc, metrics, cols=('LD1', 'LD2', 'LD3'), lab_col='lab', n_perm=5000, seed=0):
    """Per metric: Pearson r across mouse means with each LD, raw and lab-centred (both variables
    centred within lab), p for the latter from a within-lab permutation of the LD. `metrics` is
    indexed by session, or by mouse_name (already one row per mouse)."""
    rng = np.random.default_rng(seed)
    if metrics.index.name == 'mouse_name':
        mm = sc.groupby('mouse_name').agg({**{c: 'mean' for c in cols}, lab_col: 'first'}).join(metrics, how='inner')
    else:
        d = sc.join(metrics, how='inner')
        mm = d.groupby('mouse_name').agg({**{c: 'mean' for c in list(cols) + list(metrics.columns)},
                                          lab_col: 'first'})
    rows = []
    for y in metrics.columns:
        for c in cols:
            m = mm[[c, y, lab_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(m) < 8:
                continue
            g = m.groupby(lab_col)
            xc, yc = (m[c] - g[c].transform('mean')).to_numpy(), (m[y] - g[y].transform('mean')).to_numpy()
            r_lab = np.corrcoef(xc, yc)[0, 1]
            labs_ = m[lab_col].to_numpy()
            null = np.empty(n_perm)
            for i in range(n_perm):
                xp = xc.copy()
                for l in np.unique(labs_):
                    idx = np.flatnonzero(labs_ == l)
                    xp[idx] = rng.permutation(xp[idx])
                null[i] = np.corrcoef(xp, yc)[0, 1]
            r = np.corrcoef(m[c], m[y])[0, 1]
            n = len(m)
            from scipy.stats import t as tdist
            p = 2 * tdist.sf(abs(r) * np.sqrt((n - 2) / (1 - r ** 2)), n - 2)
            rows.append(dict(metric=y, ld=c, n_mice=n, r=r, p=p, r_lab=r_lab,
                             p_lab=(1 + np.sum(np.abs(null) >= abs(r_lab))) / (1 + n_perm)))
    return pd.DataFrame(rows)
