"""Fit (no-lag, signed-velocity paws, per neuron) every available session that
contains at least one of the 5 most common regions. Per-session parquet cache."""
import os, pickle, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import encoding_functions as ef

NFDIR = "/home/ines/repositories/representation_learning_variability/paper-individuality/data/neuron_files/"
OUTDIR = "encoding_results_nolag_vel"
os.makedirs(OUTDIR, exist_ok=True)

target = pd.read_csv('_nolag_target_pids.csv')['pid'].tolist()
print(f"target sessions: {len(target)}", flush=True)

for k, pid in enumerate(target):
    cache = os.path.join(OUTDIR, f'neuron_{pid}.parquet')
    if os.path.exists(cache):
        print(f'[{k+1}/{len(target)}] {pid[:8]} cached', flush=True)
        continue
    try:
        with open(os.path.join(NFDIR, f'states_neurons_file_{pid}'), 'rb') as f:
            df = pickle.load(f)
        r = ef.fit_session(df, motor_continuous=True, continuous_features='velocity',
                           motor_lags=False, unit='neuron', n_shuffles=0)
        r['pid'] = pid
        r['session'] = df['session'].iloc[0]
        r['mouse_name'] = df['mouse_name'].iloc[0]
        r.to_parquet(cache)
        print(f'[{k+1}/{len(target)}] {pid[:8]} {len(r)} neurons cvR2={r["cv_r2"].mean():.3f}', flush=True)
    except Exception as e:
        print(f'[{k+1}/{len(target)}] {pid[:8]} FAILED {type(e).__name__}: {e}', flush=True)

print('DONE', flush=True)
