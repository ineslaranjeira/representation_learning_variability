"""Kappa scan for the 60 Hz two-camera lick fits -- the cohort that also has whisker fits.

Kappa grid is scaled to the number of state exits in THIS cohort so the scan spans the
same "nothing -> roughly double the dwell" range as the whisker and 30 Hz lick scans.
Env: OUTCSV."""
import sys, os, re, pickle, csv, time
SEG=os.path.dirname(os.path.abspath(__file__))+'/..'
sys.path.insert(0,SEG); os.chdir(SEG)
os.environ.setdefault("JAX_PLATFORM_NAME","cpu")
import numpy as np, pandas as pd
import jax.numpy as jnp, jax.random as jr
from dynamax.hidden_markov_model import PoissonHMM
from segmentation_functions import cross_validate_poismodel
NB=5; OUT=os.environ['OUTCSV']
F='../data/hmm/grid_search/5_prior_em_zsc_False'; DM='../data/design_matrices/'
KAPPAS=[0.0,1e3,1e4,5e4,2e5,5e5]
props=pd.read_csv('/tmp/claude-1000/-home-ines-repositories-representation-learning-variability/6ebdc898-39ec-4101-b74f-546a4d1990f4/scratchpad/lick60_props.csv')
# spread across the lick-rate range so the scan is not all high-lick sessions
props=props.sort_values('lick_hz')
pick=set(props.iloc[[int(x) for x in np.linspace(0,len(props)-1,8)]].eid)
files={}
for f in os.listdir(F):
    if not f.startswith('best_results_Lick count_'): continue
    m=re.search(r'([0-9a-f-]{36})$',f)
    if m and m.group(1)[:8] in pick: files[m.group(1)]=f[len('best_results_Lick count_'):-36]
def dwell(s):
    ch=np.where(np.diff(s)!=0)[0]; return np.diff(np.concatenate(([0],ch+1,[len(s)])))
fh=open(OUT,'w',newline='')
w=csv.DictWriter(fh,fieldnames=['modality','mouse','eid','lick_hz','kappa','bits','raw_ll',
                                'med_dwell_f','med_dwell_ms','n_seg','occ','agree_vs_k0','secs'])
w.writeheader(); t00=time.time()
for eid,mouse in sorted(files.items(),key=lambda kv:kv[1]):
    x=pd.read_parquet(DM+f'design_matrix_{eid}_{mouse}')[['Lick count']].dropna().values
    nt,ed=x.shape; short=np.array(x[:(nt//NB)*NB])
    tr=jnp.stack(jnp.split(short,NB)); fl=len(short)/NB
    lhz=float(props[props.eid==eid[:8]].lick_hz.iloc[0])
    s0=None
    for kap in KAPPAS:
        t0=time.time()
        try:
            m=PoissonHMM(2,ed,transition_matrix_stickiness=kap)
            vll,fp,_,bll=cross_validate_poismodel(m,jr.PRNGKey(0),tr,NB,'em')
            bits=(np.asarray(vll)-np.asarray(bll))/fl*np.log(2); raw=np.asarray(vll)/fl
            fold=int(np.nanargmax(bits))
            md=PoissonHMM(2,ed,transition_matrix_stickiness=kap)
            p,_=md.initialize(key=jr.PRNGKey(0),method='prior',
                initial_probs=fp[0].probs[fold],
                transition_matrix=np.asarray(fp[1].transition_matrix)[fold],
                emission_rates=fp[2].rates[fold])
            s=np.asarray(md.most_likely_states(p,short))
            r=np.asarray(fp[2].rates)[fold].ravel()
            if r[1]<r[0]: s=1-s
            d=dwell(s); ag=''
            if s0 is not None:
                a=(s==s0).mean(); ag=round(float(max(a,1-a)),4)
            w.writerow(dict(modality='lick60',mouse=mouse,eid=eid[:8],lick_hz=lhz,kappa=kap,
                bits=round(float(np.nanmean(bits)),5),raw_ll=round(float(np.nanmean(raw)),6),
                med_dwell_f=float(np.median(d)),med_dwell_ms=round(float(np.median(d))/60*1000,1),
                n_seg=len(d),occ=round(float(s.mean()),4),agree_vs_k0=ag,
                secs=round(time.time()-t0,1)))
            if kap==0.0: s0=s
        except Exception as e:
            print('ERR',eid[:8],kap,type(e).__name__,e,flush=True)
        fh.flush()
    print(f'{mouse} {eid[:8]} done  ({time.time()-t00:.0f}s elapsed)',flush=True)
fh.close(); print('DONE')
