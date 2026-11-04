"""
Is a mouse's day-to-day consistency related to its other traits?

Consistency = each mouse's own session-to-session variance of the whole syllable profile, sampling noise removed
(figures/mouse_consistency.csv, written by fig_session_partition.py). Low = consistent mouse.

Against, one value per mouse:
    LD1                 the mouse's mean LD1 (saved embedding)
    learning speed      log training days (learning_prediction/training_time_05-05-2026, as lab_controls.py:
                        mice with log days of exactly 0 or 1 dropped)
    state switches      mean GLM-HMM engagement-state switches per session (lab_controls' session values; the
                        one association that was between-mouse only in within_between_models.py)

Pearson and Spearman across mice; p from 10,000 permutations of consistency across mice. Also the partial
correlation controlling the number of sessions (a mouse with 3 sessions has a noisier consistency estimate) and,
as a sensitivity, with lab removed as well.
Writes figures/consistency_correlates.{csv,png,pdf}.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent
REPO = PI.parent
sys.path.insert(0, str(PI))
try:
    import numpy._core.numeric  # noqa: F401
except ModuleNotFoundError:     # the LDA file was pickled under numpy 2
    import numpy.core.numeric, numpy.core.multiarray
    sys.modules['numpy._core.numeric'], sys.modules['numpy._core.multiarray'] = np.core.numeric, np.core.multiarray
import paper_style as ps

N_PERM, SEED = 10000, 0
FIGDIR = HERE / 'figures'
LDA_FILE = PI / 'clustering' / 'data_files' / 'mouse_LDA_5_bins_raw_shrink0.5_wmouse_360_02-10-2026'
TRAINING_FILE = REPO / 'learning_prediction' / 'training_time_05-05-2026'
CORRELATES = HERE / 'cache' / 'correlate_sessions.pqt'          # within_between_models.py's capture


def load():
    M = pd.read_csv(FIGDIR / 'mouse_consistency.csv')
    lda = pd.read_pickle(LDA_FILE)
    M['LD1'] = M['mouse_name'].map(lda.groupby('mouse_name')[0].mean())
    tt = pd.read_parquet(TRAINING_FILE)
    tt['log_training'] = np.log(tt['training_days'])
    tt = tt[~tt['log_training'].isin([0, 1])]
    M['learning (log training days)'] = M['mouse_name'].map(tt.groupby('mouse_name')['log_training'].first())
    cs = pd.read_parquet(CORRELATES)
    sw = cs[cs['correlate'] == 'State switches (K=2)'].groupby('mouse_name')['y'].mean()
    M['state switches'] = M['mouse_name'].map(sw)
    return M


def residualise(v, Z):
    Z = np.column_stack([np.ones(len(v)), Z])
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


def test(M, col, rng):
    d = M[['day_to_day_variance', col, 'n_sessions', 'lab']].dropna()
    x, y = d['day_to_day_variance'].to_numpy(float), d[col].to_numpy(float)
    r, rho = np.corrcoef(x, y)[0, 1], stats.spearmanr(x, y)[0]
    null = np.array([np.corrcoef(rng.permutation(x), y)[0, 1] for _ in range(N_PERM)])
    p = (1 + (np.abs(null) >= abs(r)).sum()) / (1 + N_PERM)
    Zn = d[['n_sessions']].to_numpy(float)
    r_n, p_n = stats.pearsonr(residualise(x, Zn), residualise(y, Zn))
    Zl = np.column_stack([Zn, pd.get_dummies(d['lab'], drop_first=True).to_numpy(float)])
    r_l, p_l = stats.pearsonr(residualise(x, Zl), residualise(y, Zl))
    return dict(correlate=col, n_mice=len(d), pearson_r=r, p_perm=p, spearman_rho=rho,
                r_given_n_sessions=r_n, p_given_n_sessions=p_n, r_given_n_sessions_and_lab=r_l,
                p_given_n_sessions_and_lab=p_l)


def main():
    rng = np.random.default_rng(SEED)
    M = load()
    r_ns = stats.spearmanr(M['day_to_day_variance'], M['n_sessions'])
    print(f"{len(M)} mice; consistency vs number of sessions: rho = {r_ns[0]:+.2f} (p = {r_ns[1]:.3f})")
    cols = ['LD1', 'learning (log training days)', 'state switches']
    RES = pd.DataFrame([test(M, c, rng) for c in cols])
    with pd.option_context('display.width', 220):
        print(RES.round(3).to_string(index=False))
    RES.to_csv(FIGDIR / 'consistency_correlates.csv', index=False)

    ps.use('paper')
    fs = plt.rcParams['font.size']
    fig, axs = plt.subplots(1, 3, figsize=(7.09, 2.2), gridspec_kw={'wspace': 0.45}, sharey=True)
    for ax, col, (_, r) in zip(axs, cols, RES.iterrows()):
        d = M[['day_to_day_variance', col]].dropna()
        ax.scatter(d[col], d['day_to_day_variance'], s=10, color='#8FCB8B', edgecolor='0.3', lw=0.3)
        b = np.polyfit(d[col], d['day_to_day_variance'], 1)
        xx = np.linspace(d[col].min(), d[col].max(), 2)
        ax.plot(xx, np.polyval(b, xx), color='0.3', lw=0.8, ls='-' if r['p_perm'] < 0.05 else '--')
        ax.set_xlabel(col.replace('learning (log training days)', 'Training time (log days)')
                      .replace('state switches', 'State switches / session').replace('LD1', 'LD1 (mouse mean)'))
        ax.set_title(f"r = {r['pearson_r']:+.2f}, p = {r['p_perm']:.3f}\n"
                     f"| n sessions: {r['r_given_n_sessions']:+.2f}  | + lab: {r['r_given_n_sessions_and_lab']:+.2f}",
                     fontsize=fs * 0.7)
    axs[0].set_ylabel("A mouse's day-to-day\nvariance")
    fig.text(0.5, -0.06, f'One point per mouse ({len(M)}). Higher = less consistent. p: consistency permuted across mice.',
             ha='center', fontsize=fs * 0.65, color='0.35')
    for path in (FIGDIR / 'consistency_correlates.png', FIGDIR / 'consistency_correlates.pdf'):
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print('wrote', path.relative_to(PI))


if __name__ == '__main__':
    main()
