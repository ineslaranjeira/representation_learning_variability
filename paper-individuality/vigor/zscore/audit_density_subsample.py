"""
AUDIT OF THE ORIGINAL 3.2 -> 3.3 PIPELINE (density-weighted subsample, K = 10)
===========================================================================
Rebuilds the original 3.3 fit (PCA 95%, KMeans(10, seed 2024)) from the supersession file
and, on 25 random sessions, compares:
  * the density-weighted 2,000-frame training subsample vs the full session (mean shift,
    SD ratio, and cluster occupancy: total variation vs a uniform random 2,000);
  * the labels the same frames get when z-scored by the subsample's own stats (training)
    vs by full-session stats (what labelling uses);
  * whether the pooled z-score in 3.3 does anything; skew of the raw amplitudes.
Reads data/paw_subsampled_wavelets18ago (the original subsamples, renamed 2026-09-30).
Run from paper-individuality/:  python vigor/zscore/audit_density_subsample.py
"""
import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pathlib
B=str(pathlib.Path(__file__).resolve().parents[2]/'data')+'/'
ALL=['l_paw_x','l_paw_y','r_paw_x','r_paw_y']+[f'{p}_{a}{f}' for p in ['l_paw','r_paw'] for a in 'xy' for f in ['0.5','1.0','2.0','4.0','8.0','16.0','32.0']]
USE=[f'{p}_{a}{f}' for p in ['l_paw','r_paw'] for a in 'xy' for f in ['0.5','1.0','2.0','4.0','8.0']]
ui=[ALL.index(c) for c in USE]
sup=np.load(B+'session_zscored_supersession_wavelets_paw08-19-2026')
print('supersession', sup.shape)
X=stats.zscore(sup[:,ui],axis=0)
print('pooled z-score is no-op? max|mean|, std range:', np.abs(sup[:,ui].mean(0)).max().round(4), sup[:,ui].std(0).min().round(3), sup[:,ui].std(0).max().round(3))
pca=PCA(20).fit(X); n=np.where(np.cumsum(pca.explained_variance_ratio_)>=.95)[0][0]+1
P=pca.transform(X)[:,:n]
km=KMeans(10,random_state=2024).fit(P); C=km.cluster_centers_
gm,gs=X.mean(0),X.std(0)
def lab(Z): return np.argmin(cdist(pca.transform((Z-gm)/gs)[:,:n],C),1)
sub=sorted(os.listdir(B+'paw_subsampled_wavelets18ago'))
rng=np.random.default_rng(0); pick=rng.choice(len(sub),25,replace=False)
rows=[]; skew=[]; OB=[]; OS=[]; OU=[]
for k in pick:
    f=sub[k]; s,m=f[:36],f[37:-4]
    fn=B+f'paw_wavelets/paw_vel_wavelets_{s}_{m}'
    if not os.path.exists(fn): continue
    raw=np.load(B+'paw_subsampled_wavelets18ago/'+f)[:,ui]
    full=pd.read_parquet(fn)[USE].dropna().to_numpy()
    mu,sd=full.mean(0),full.std(0)
    unif=full[rng.choice(len(full),2000,replace=False)]
    zb=stats.zscore(raw,0)                        # what training used
    zf=(raw-mu)/sd                                # same frames, full-session stats (what propagation uses)
    Lb,Lf=lab(zb),lab(zf)
    Lsess=lab((full-mu)/sd)                        # propagated labels, whole session
    Lu=lab((unif-mu)/sd)
    occ=lambda L: np.bincount(L,minlength=10)/len(L)
    rows.append(dict(mean_shift_sd=np.median((raw.mean(0)-mu)/sd), std_ratio=np.median(raw.std(0)/sd),
        relabel=np.mean(Lb!=Lf), TV_biased_vs_session=.5*np.abs(occ(Lf)-occ(Lsess)).sum(),
        TV_uniform_vs_session=.5*np.abs(occ(Lu)-occ(Lsess)).sum(), n_frames=len(full)))
    skew.append(stats.skew(full,0))
    OB.append(occ(Lb)); OS.append(occ(Lsess)); OU.append(occ(Lu))
df=pd.DataFrame(rows); print(df.describe().round(3).T[['mean','50%','min','max']])
print('median skew per feature', np.round(np.median(skew,0),1))
print('min value of raw features (positive amplitudes?)', full.min(0).round(3)[:5])
# rare vs common: occupancy in density-biased training sample vs whole session, pooled

vig=np.array([P[km.labels_==c].shape[0] for c in range(10)])
power=np.array([X[km.labels_==c].mean() for c in range(10)])
t=pd.DataFrame(dict(mean_z_power=power, session_occ=np.mean(OS,0), trainingset_occ=np.mean(OB,0), uniform_occ=np.mean(OU,0))).sort_values('mean_z_power')
t['train/session']=t.trainingset_occ/t.session_occ
print(t.round(3))
