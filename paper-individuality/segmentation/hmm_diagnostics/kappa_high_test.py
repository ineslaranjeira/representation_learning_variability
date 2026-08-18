"""Is kappa >= 1e4 significantly WORSE on raw held-out LL? Paired across folds.
Saves per-fold raw LL so the test is rigorous rather than based on fold means.
Env: OUTCSV."""
import sys,os,re,pickle,csv,time
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
props=pd.read_csv('/tmp/claude-1000/-home-ines-repositories-representation-learning-variability/6ebdc898-39ec-4101-b74f-546a4d1990f4/scratchpad/lick60_props.csv').sort_values('lick_hz')
pick=set(props.iloc[[int(x) for x in np.linspace(0,len(props)-1,8)]].eid)
files={}
for f in os.listdir(F):
    if not f.startswith('best_results_Lick count_'): continue
    m=re.search(r'([0-9a-f-]{36})$',f)
    if m and m.group(1)[:8] in pick: files[m.group(1)]=f[len('best_results_Lick count_'):-36]
fh=open(OUT,'w',newline='')
w=csv.DictWriter(fh,fieldnames=['eid','mouse','kappa']+[f'fold{i}' for i in range(NB)])
w.writeheader()
for eid,mouse in sorted(files.items(),key=lambda kv:kv[1]):
    x=pd.read_parquet(DM+f'design_matrix_{eid}_{mouse}')[['Lick count']].dropna().values
    nt,ed=x.shape; short=np.array(x[:(nt//NB)*NB])
    tr=jnp.stack(jnp.split(short,NB)); fl=len(short)/NB
    for kap in KAPPAS:
        try:
            m_=PoissonHMM(2,ed,transition_matrix_stickiness=kap)
            vll,_,_,_=cross_validate_poismodel(m_,jr.PRNGKey(0),tr,NB,'em')
            raw=np.asarray(vll)/fl
            w.writerow(dict(eid=eid[:8],mouse=mouse,kappa=kap,
                            **{f'fold{i}':round(float(raw[i]),8) for i in range(NB)}))
        except Exception as e:
            print('ERR',eid[:8],kap,e,flush=True)
        fh.flush()
    print(f'{mouse} {eid[:8]} done',flush=True)
fh.close(); print('DONE')
