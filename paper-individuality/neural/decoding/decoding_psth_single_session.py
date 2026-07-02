"""Time-resolved decodability PSTH for a single example session.
Decodes stimulus side, block, choice and feedback from the population at each time bin
(aligned to stimulus onset), with a matched control line overlaid:
  - block  -> IBL pseudosession null (generative, biased-only)
  - stimulus / choice / feedback -> label-shuffle null
Options: cap trial count (N_TRIALS) and balance classes (BALANCE_CLASSES).
"""
import numpy as np, pandas as pd, pickle, os, warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings('ignore')

prefix = '/home/ines/repositories/representation_learning_variability/paper-individuality/'
firing_rates_dir = prefix + 'data/firing_rates/'
trials_path = prefix + '4_mice/all_trials_04-05-2026'
DROP = ['root', 'void']

N_COMP = 20; CV_FOLDS = 5; N_CONTROL = 20; SEED = 0
SMOOTH_BINS = 3            # trailing (causal) window ~50 ms for the decode feature
N_TRIALS = None            # cap #trials per variable (None = use all available)
BALANCE_CLASSES = True     # downsample to equal class sizes (stim/choice/feedback)
MIN_NEURONS = 40           # for auto-picking a good example session
SESSION_SKIP = 1           # skip this many qualifying sessions (0 = first, 1 = second, ...)
BLK_MEAN = 60; BLK_MIN = 20; BLK_MAX = 100
rng = np.random.default_rng(SEED)

# ---------------- helpers ----------------
def gen_pseudo_biased(n, rng):
    out = []; side = float(rng.choice([0.2, 0.8]))
    while len(out) < n:
        x = rng.exponential(BLK_MEAN)
        while x < BLK_MIN or x > BLK_MAX: x = rng.exponential(BLK_MEAN)
        out += [side] * int(x); side = 0.8 if side == 0.2 else 0.2
    return np.array(out[:n])

def make_folds(X, y):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED); folds = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        nc = min(N_COMP, Xtr.shape[1], Xtr.shape[0] - 1)
        pca = PCA(n_components=nc, random_state=SEED).fit(Xtr)
        folds.append((tr, te, pca.transform(Xtr), pca.transform(Xte)))
    return folds

def acc_labels(folds, y):
    a = []
    for tr, te, Ztr, Zte in folds:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2: a.append(np.nan); continue
        a.append(balanced_accuracy_score(y[te], LDA().fit(Ztr, y[tr]).predict(Zte)))
    return np.nanmean(a)

def balance(idx, y, rng):
    classes, counts = np.unique(y, return_counts=True); m = counts.min()
    keep = np.concatenate([rng.choice(idx[y == c], m, replace=False) for c in classes])
    return np.sort(keep)

# ---------------- pick + load an example session ----------------
trials_df = pd.read_parquet(trials_path)
pkl_files = sorted([f for f in os.listdir(firing_rates_dir) if f.startswith('firing_rate_')])
chosen = None; seen = 0
for fn in pkl_files:
    with open(os.path.join(firing_rates_dir, fn), 'rb') as f: d = pickle.load(f)
    d = d[~d['area'].isin(DROP)]
    if d['neuron_id'].nunique() < MIN_NEURONS: continue
    sess = d['session'].iloc[0]
    tl = trials_df[trials_df.session == sess]
    if (tl.block.isin([0.2, 0.8]).sum() >= 150) and (tl.contrast > 0).sum() >= 150:
        if seen == SESSION_SKIP:
            chosen = (fn, d, sess); break
        seen += 1
fn, d, session = chosen
tcols = sorted([c for c in d.columns if c.startswith('t_')], key=lambda x: float(x.split('_')[1]))
tsec = np.array([float(c.split('_')[1]) for c in tcols])
d = d.copy(); d['nuid'] = d['pid'].astype(str) + '__' + d['neuron_id'].astype(str)
piv = d.pivot_table(index='trial_id', columns='nuid', values=tcols)      # trials x (tcol, nuid)
arr = np.stack([piv[tc].values for tc in tcols], axis=-1)                # [trial, neuron, bin]
trial_ids = piv.index.values.astype(int)
# trailing-window smoothing along time
cs = np.cumsum(arr, axis=2); sm = arr.copy()
sm[:, :, SMOOTH_BINS - 1:] = (cs[:, :, SMOOTH_BINS - 1:] -
    np.concatenate([np.zeros_like(cs[:, :, :1]), cs[:, :, :-SMOOTH_BINS]], axis=2)) / SMOOTH_BINS
for i in range(SMOOTH_BINS - 1): sm[:, :, i] = cs[:, :, i] / (i + 1)
arr = sm
ok = ~np.isnan(arr).any(axis=(1, 2))                                    # trials with complete data
arr, trial_ids = arr[ok], trial_ids[ok]
print(f"session {session[:8]} | file {fn[-10:]} | {arr.shape[1]} neurons | {arr.shape[0]} complete trials")

# ---------------- trial labels ----------------
lab = trials_df[trials_df.session == session].set_index('trial_id')
lab = lab.reindex(trial_ids)
choice_code = (lab['choice'].values == 'right').astype(float)           # left=0 right=1
correct = lab['correct'].values
stim_side = np.where(correct == 1, choice_code, 1 - choice_code)        # stimulus on chosen side if correct
contrast = lab['contrast'].values
block = lab['block'].values

VARS = {
    'stimulus (L/R)': dict(mask=(contrast > 0), y=stim_side, ctrl='shuffle'),
    'block (0.2/0.8)': dict(mask=np.isin(block, [0.2, 0.8]), y=(block == 0.8).astype(float), ctrl='pseudo'),
    'choice (L/R)':    dict(mask=~np.isnan(choice_code), y=choice_code, ctrl='shuffle'),
    'feedback (corr)': dict(mask=~np.isnan(correct), y=correct, ctrl='shuffle'),
}

# ---------------- decode across time ----------------
results = {}
for name, cfg in VARS.items():
    m = cfg['mask'] & ~np.isnan(cfg['y']); idx = np.where(m)[0]; y_all = cfg['y'][idx].astype(int)
    order = np.arange(len(idx))                                         # chronological position (block pseudo)
    if BALANCE_CLASSES:
        kb = balance(np.arange(len(idx)), y_all, rng); idx, y_all, order = idx[kb], y_all[kb], order[kb]
    if N_TRIALS and len(idx) > N_TRIALS:
        s = np.sort(rng.choice(len(idx), N_TRIALS, replace=False)); idx, y_all, order = idx[s], y_all[s], order[s]
    span = int(cfg['mask'].sum())                                       # full biased span (for pseudo)
    # controls (fixed across bins; X unchanged so only labels vary)
    if cfg['ctrl'] == 'pseudo':
        ctrls = []
        for _ in range(N_CONTROL):
            pf = gen_pseudo_biased(span, rng); ctrls.append((pf[order] == 0.8).astype(int))
    else:
        ctrls = [rng.permutation(y_all) for _ in range(N_CONTROL)]
    real_t, ctrl_t = [], []
    for b in range(arr.shape[2]):
        X = arr[idx, :, b]
        if len(np.unique(y_all)) < 2 or min(np.bincount(y_all)) < CV_FOLDS:
            real_t.append(np.nan); ctrl_t.append([np.nan] * N_CONTROL); continue
        folds = make_folds(X, y_all)
        real_t.append(acc_labels(folds, y_all))
        ctrl_t.append([acc_labels(folds, c) for c in ctrls])
    results[name] = dict(real=np.array(real_t), ctrl=np.array(ctrl_t), n=len(idx))
    print(f"  {name:18s}: n={len(idx):3d} trials, peak real acc={np.nanmax(real_t):.3f}")

# ---------------- plot ----------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, (name, r) in zip(axes.ravel(), results.items()):
    cm = np.nanmean(r['ctrl'], axis=1); clo = np.nanpercentile(r['ctrl'], 5, axis=1); chi = np.nanpercentile(r['ctrl'], 95, axis=1)
    ax.fill_between(tsec, clo, chi, color='gray', alpha=0.25)
    ax.plot(tsec, cm, color='gray', lw=1.5, ls='--', label='control')
    ax.plot(tsec, r['real'], color='crimson', lw=2, label='real')
    ax.axvline(0, color='k', ls=':', lw=1); ax.axhline(0.5, color='k', ls='-', lw=0.6, alpha=0.5)
    ax.set_title(f"{name}  (n={r['n']} trials)", fontsize=11); ax.set_ylabel('balanced accuracy')
    ax.legend(fontsize=9, loc='upper left'); ax.set_ylim(0.4, 1.0)
for ax in axes[1]: ax.set_xlabel('time from stim onset (s)')
fig.suptitle(f"Decodability along the trial — session {session[:8]} ({arr.shape[1]} neurons, all regions pooled)\n"
             f"smooth={SMOOTH_BINS} bins, k={N_COMP} PCs, balance={BALANCE_CLASSES}", fontsize=12)
plt.tight_layout()
out_png = prefix + f'neural/decoding/decoding_psth_single_session_{session}.png'
plt.savefig(out_png, dpi=110, bbox_inches='tight'); print(f"saved {out_png}")
