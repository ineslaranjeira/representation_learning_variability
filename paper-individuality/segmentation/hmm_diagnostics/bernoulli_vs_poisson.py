"""Bernoulli vs Poisson emissions for the binary lick signal (training sessions, 30 Hz).
Same CV setup, same kappa, same folds. Compares held-out bits and state agreement.
Env: OUTCSV."""
import sys, os, re, pickle, csv, time
SEG=os.path.dirname(os.path.abspath(__file__))+'/..'
sys.path.insert(0,SEG); os.chdir(SEG)
os.environ.setdefault("JAX_PLATFORM_NAME","cpu")
import numpy as np, pandas as pd
import jax.numpy as jnp, jax.random as jr
from dynamax.hidden_markov_model import PoissonHMM, BernoulliHMM
from segmentation_functions import cross_validate_poismodel
NB, KAPPA = 5, 3.0
OUT=os.environ['OUTCSV']
q=pd.read_csv('hmm_diagnostics/lick_training_quality.csv')
q=q[(q.med_bout_licks>1)&(q.n_seg>5)].nlargest(10,'n_licks')   # healthy sessions
dmi={}
for p in ['../data/design_matrices/1_camera_setup/session_1','../data/design_matrices/1_camera_setup/extra_bwm']:
    for f in os.listdir(p):
        if f.startswith('design_matrix'): dmi[f.split('_')[2]]=os.path.join(p,f)
def dwell(s):
    ch=np.where(np.diff(s)!=0)[0]; return np.diff(np.concatenate(([0],ch+1,[len(s)])))
fh=open(OUT,'w',newline=''); w=csv.DictWriter(fh,fieldnames=['mouse','eid','model','bits','raw_ll_per_frame','med_dwell','n_seg','occ','agreement','secs'])
w.writeheader()
for _,r in q.iterrows():
    eid=[k for k in dmi if k.startswith(r.eid)][0]
    x=pd.read_parquet(dmi[eid])[['Lick count']].dropna().values
    nt,ed=x.shape; short=np.array(x[:(nt//NB)*NB])
    tr=jnp.stack(jnp.split(short,NB)); fold_len=len(short)/NB
    seqs={}
    for name in ('poisson','bernoulli'):
        t0=time.time()
        try:
            m=(PoissonHMM(2,ed,transition_matrix_stickiness=KAPPA) if name=='poisson'
               else BernoulliHMM(2,ed,transition_matrix_stickiness=KAPPA))
            em = tr if name=='poisson' else tr.astype(bool)
            vll,fitp,_,bll=cross_validate_poismodel(m,jr.PRNGKey(0),em,NB,'em')
            bits=(np.asarray(vll)-np.asarray(bll))/fold_len*np.log(2)
            raw=np.asarray(vll)/fold_len
            fold=int(np.nanargmax(bits))
            mdl=(PoissonHMM(2,ed,transition_matrix_stickiness=KAPPA) if name=='poisson'
                 else BernoulliHMM(2,ed,transition_matrix_stickiness=KAPPA))
            kw=dict(initial_probs=fitp[0].probs[fold],
                    transition_matrix=np.asarray(fitp[1].transition_matrix)[fold])
            kw['emission_rates' if name=='poisson' else 'emission_probs']=getattr(fitp[2],'rates' if name=='poisson' else 'probs')[fold]
            p_,_=mdl.initialize(key=jr.PRNGKey(0),method='prior',**kw)
            s=np.asarray(mdl.most_likely_states(p_,short if name=='poisson' else short.astype(bool)))
            par=np.asarray(getattr(fitp[2],'rates' if name=='poisson' else 'probs'))[fold].ravel()
            if par[1]<par[0]: s=1-s
            seqs[name]=s; d=dwell(s)
            w.writerow(dict(mouse=r.mouse,eid=r.eid,model=name,bits=round(float(np.nanmean(bits)),5),raw_ll_per_frame=round(float(np.nanmean(raw)),6),
                med_dwell=float(np.median(d)),n_seg=len(d),occ=round(float(s.mean()),4),
                agreement='',secs=round(time.time()-t0,1)))
        except Exception as e:
            w.writerow(dict(mouse=r.mouse,eid=r.eid,model=name,bits=f'ERR {type(e).__name__}: {e}'[:90],secs=round(time.time()-t0,1)))
        fh.flush()
    if len(seqs)==2:
        a=(seqs['poisson']==seqs['bernoulli']).mean()
        w.writerow(dict(mouse=r.mouse,eid=r.eid,model='AGREEMENT',agreement=round(float(max(a,1-a)),4)))
        fh.flush()
    print(f'{r.mouse} {r.eid} done',flush=True)
fh.close(); print('DONE')
