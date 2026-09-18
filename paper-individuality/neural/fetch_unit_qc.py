"""
PER-NEURON SPIKE-SORTING QC, fetched once and cached
====================================================
The firing-rate pkl files carry no unit quality metrics -- they hold binned peri-stimulus
rates and nothing else -- so a presence ratio cannot be recomputed from them: the presence
ratio is defined over the WHOLE recording (fraction of 10 s bins containing at least one
spike from that unit, `presence_window` = 10 in ibllib's METRICS_PARAMS), and the pkl files
only ever saw -0.5..1.0 s around each stimulus.

So it is fetched from the source the spike files were built from. `neural_spike_files_v2`
loaded each probe with

    ssl.load_spike_sorting(good_units=True, revision='2025-05-26')

and named each neuron `<acronym>_neuron_<cluster_id>_spike_count`, where <cluster_id> is the
row position in that revision's FULL cluster table (good_units=True filters the spikes, not
the cluster table). This script re-opens the same table at the same revision and pulls the
metrics column for those rows. Two things are asserted rather than assumed, because a
silent mis-join here would attach one neuron's quality to another's activity:

    * every neuron we hold must come back with label == 1  (good_units=True guarantees it)
    * the acronym in the neuron's own name must equal the cluster table's acronym

Output: one row per neuron, keyed by `nuid` = '<pid>__<neuron_id>', which is exactly the key
lda1_neural_metrics builds. Written to lda1_tables/unit_qc_<date>.pqt.

    python fetch_unit_qc.py
"""
import gc
import os
import pickle
import sys
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(ROOT)
FR_DIR = os.path.join(PAPER, 'data', 'firing_rates')
CACHE = os.path.join(ROOT, 'lda1_tables')
REVISION = '2025-05-26'          # the revision neural_spike_files_v2 built the pkl files from
SORTER = 'iblsorter'

CHECKPOINT_EVERY = 20
PARTIAL = os.path.join(CACHE, 'unit_qc_partial.pqt')   # resume file, deleted on success

KEEP_COLS = ['presence_ratio', 'presence_ratio_std', 'firing_rate', 'label',
             'amp_median', 'noise_cutoff', 'slidingRP_viol', 'spike_count', 'drift']


def parse_cluster_id(neuron_id):
    """'MRN_neuron_42_spike_count' -> 42. The int is the row position in the full cluster
    table, which is what spikes['clusters'] holds."""
    return int(neuron_id.split('_neuron_')[1].split('_spike_count')[0])


def parse_acronym(neuron_id):
    return neuron_id.split('_neuron_')[0]


def main(max_files=None):
    from one.api import ONE
    from brainbox.io.one import SpikeSortingLoader
    one = ONE()

    files = sorted(f for f in os.listdir(FR_DIR) if f.startswith('firing_rate_'))
    if max_files:
        files = files[:max_files]

    # RESUME. Each probe costs a download, so ~400 of them is long enough that the run will
    # occasionally be interrupted -- and an interrupted run that kept everything in memory
    # loses the lot. Progress is checkpointed every CHECKPOINT_EVERY probes to a partial
    # file, and a fresh start picks up whatever is already in it.
    rows, failed, done = [], [], set()
    if os.path.exists(PARTIAL):
        prev = pd.read_parquet(PARTIAL)
        rows.append(prev)
        done = set(prev['pid'].unique())
        print(f'resuming: {len(prev)} neurons over {len(done)} probes already fetched',
              flush=True)

    def checkpoint():
        pd.concat(rows, ignore_index=True).to_parquet(PARTIAL, index=False)

    for i, fn in enumerate(files):
        try:
            with open(os.path.join(FR_DIR, fn), 'rb') as f:
                d = pickle.load(f)
            pid = d['pid'].iloc[0]
            if pid in done:
                continue
            nu = d.drop_duplicates('neuron_id')[['neuron_id']].copy()
            nu['cluster_id'] = nu['neuron_id'].map(parse_cluster_id)
            nu['name_acronym'] = nu['neuron_id'].map(parse_acronym)

            # CLUSTERS ONLY -- NEVER THE SPIKES TABLE. The obvious call here is
            # `ssl.load_spike_sorting(good_units=True, ...)`, and it is what this script
            # used at first: it loads the probe's passingSpikes table, tens of millions of
            # spike times, to compute nothing we need. Held across ~400 probes it grew to
            # 15 GB of resident memory and the kernel OOM-killed the run at probe 160,
            # twice, with no Python traceback -- a SIGKILL bypasses try/except entirely.
            # Every metric we want is already in clusters.metrics on disk; the acronym for
            # the id guard comes from the channels object. Peak memory this way is flat at
            # ~0.9 GB no matter how many probes are processed.
            ssl = SpikeSortingLoader(one=one, pid=pid)
            ssl.download_spike_sorting(spike_sorter=SORTER, revision=REVISION,
                                       objects=['clusters', 'channels'])
            cl = ssl._load_object(ssl.files['clusters'], wildcards=one.wildcards)
            met = cl['metrics']
            if nu['cluster_id'].max() >= len(met):
                raise IndexError(f'cluster_id {nu.cluster_id.max()} past table of {len(met)}')
            sub = met.loc[nu['cluster_id'].values]

            ch = ssl.load_channels(spike_sorter=SORTER, revision=REVISION)
            acronym = np.asarray(ch['acronym'])[
                np.asarray(cl['channels'])[nu['cluster_id'].values]]

            # the two guards -- a mis-join would be silent otherwise
            assert (sub['label'].values == 1).all(), 'a neuron came back not-good'
            assert (acronym == nu['name_acronym'].values).all(), 'acronym mismatch'

            out = pd.DataFrame({'pid': pid, 'neuron_id': nu['neuron_id'].values,
                                'nuid': pid + '__' + nu['neuron_id'].values,
                                'cluster_id': nu['cluster_id'].values,
                                'acronym': acronym})
            for c in KEEP_COLS:
                out[c] = sub[c].values if c in sub.columns else np.nan
            rows.append(out)
            del ssl, cl, met, sub, ch, d, nu
            gc.collect()
        except Exception as e:
            failed.append((fn, f'{type(e).__name__}: {e}'))
            print(f'  FAIL {fn}: {type(e).__name__}: {e}', flush=True)
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(files):
            if rows:
                checkpoint()
            print(f'{i+1}/{len(files)} probes, {sum(len(r) for r in rows)} neurons, '
                  f'{len(failed)} failed (checkpointed)', flush=True)

    qc = pd.concat(rows, ignore_index=True)
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"unit_qc_{date.today().strftime('%d-%m-%Y')}.pqt")
    qc.to_parquet(p, index=False)
    if os.path.exists(PARTIAL):
        os.remove(PARTIAL)
    print(f'\nwrote {p}: {len(qc)} neurons over {qc.pid.nunique()} probes')
    print(f"presence_ratio: median {qc.presence_ratio.median():.3f}, "
          f"<=0.95 in {(qc.presence_ratio <= 0.95).mean():.1%} of neurons, "
          f"NaN in {qc.presence_ratio.isna().mean():.1%}")
    if failed:
        print(f'{len(failed)} probes failed:')
        for fn, e in failed[:20]:
            print('  ', fn, e)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
