"""
BUILD session_ordinals_task.parquet -- the session axis of dataset_timepoints_overview.ipynb
=============================================================================================
For every mouse with an Early session in the QC sheet (session_filters.load_session_table), the full list of its
sessions from Alyx (ONE, remote), keeping TASK sessions only -- training, biased and ephys choiceWorld -- and
dropping habituation, so the first training session is ordinal 1 (as that notebook's header describes). Columns:
eid, mouse_name, start_time, task_protocol, kind ('training' | 'biased' | 'ephys'), train_ord (1-based, by start
time). ~1 min for 58 mice. Run from learning_individuality/:  python build_session_ordinals.py
"""
import sys
import pathlib
import pandas as pd
from one.api import ONE

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from session_filters import load_session_table   # noqa: E402

OUT = HERE / 'session_ordinals_task.parquet'


def kind_of(protocol):
    p = (protocol or '').lower()
    if 'habituation' in p or 'choiceworld' not in p:
        return None
    if 'ephys' in p:
        return 'ephys'
    if 'biased' in p:
        return 'biased'
    if 'training' in p:
        return 'training'
    return None


def main():
    tab = load_session_table().drop_duplicates('eid').dropna(subset=['mouse_name', 'timepoint'])
    mice = sorted(tab.loc[tab['timepoint'] == 'Early', 'mouse_name'].unique())
    one = ONE(mode='remote')
    rows = []
    for k, m in enumerate(mice):
        for s in one.alyx.rest('sessions', 'list', subject=m):
            kind = kind_of(s.get('task_protocol'))
            if kind:
                rows.append(dict(eid=s['id'], mouse_name=m, start_time=s['start_time'],
                                 task_protocol=s['task_protocol'], kind=kind))
        print(f'{k + 1}/{len(mice)} {m}', flush=True)
    df = pd.DataFrame(rows).sort_values(['mouse_name', 'start_time'])
    df['train_ord'] = df.groupby('mouse_name').cumcount() + 1
    df.to_parquet(OUT, index=False)
    print(f'wrote {OUT.name}: {len(df)} task sessions, {df["mouse_name"].nunique()} mice')


if __name__ == '__main__':
    main()
