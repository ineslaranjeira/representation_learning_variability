"""
Which dopamine mice have photometry during TRAINING, and from which session on?
==============================================================================
Queries Alyx (mainenlab, DAT-Cre lines) for every session of every DA mouse, and per session
whether it has photometry datasets (processed `photometry.signal.pqt`, raw
`_neurophotometrics_fpData.raw.pqt`) and the wheel (`_ibl_wheel.position.npy`). The fibre target
is read from `photometryROI.locations.pqt` of one session per mouse (2 KB download).

Outputs (next to this file):
    da_inventory_sessions.csv   one row per session
    da_inventory_mice.csv       one row per mouse: training sessions, how many have photometry
                                (and wheel), the ordinal of the first training session with
                                photometry, biased/ephys sessions with photometry, fibre target

Session classes come from the task protocol: habituation / training (trainingChoiceWorld,
incl. the _NPH_ variants and trainingPhaseChoiceWorld) / biased / ephys. A mouse with biased
sessions reached the training criterion; one without either did not, or is still training.

What this does NOT check: signal quality (the pose-QC table in neuromodulators/ covers biased and
ephys sessions only), and whether the photometry signal covers the whole session.

Usage (iblenv, needs Alyx access):  python da_inventory.py
"""
import re
from pathlib import Path

import pandas as pd
from one.api import ONE

HERE = Path(__file__).resolve().parent
BASE_URL = 'https://alyx.internationalbrainlab.org'
CLASSES = ['habituation', 'training', 'biased', 'ephys', 'passive']


def session_class(protocol):
    p = str(protocol).lower()
    return next((k for k in CLASSES if k in p), 'other')


def session_table(one):
    subjects = [s for s in one.alyx.rest('subjects', 'list', lab='mainenlab')
                if s.get('line') and re.search(r'dat', s['line'], re.I)]
    rows = []
    for s in subjects:
        nick = s['nickname']
        sessions = one.alyx.rest('sessions', 'list', subject=nick)
        if not sessions:
            continue
        dsets = []
        for pattern in ('photometry', 'fpData'):
            dsets += one.alyx.rest('datasets', 'list', subject=nick, django=f'name__icontains,{pattern}') or []
        wheel = one.alyx.rest('datasets', 'list', subject=nick, name='_ibl_wheel.position.npy') or []
        eid = lambda d: d['session'].split('/')[-1] if d.get('session') else None
        names = {}
        for d in dsets:
            names.setdefault(eid(d), set()).add(d['name'])
        has_wheel = {eid(d) for d in wheel}
        for x in sessions:
            n = names.get(x['id'], set())
            rows.append(dict(subject=nick, line=s['line'], eid=x['id'], date=x['start_time'][:10],
                             number=x['number'], protocol=x['task_protocol'],
                             session_class=session_class(x['task_protocol']),
                             signal='photometry.signal.pqt' in n,
                             raw='_neurophotometrics_fpData.raw.pqt' in n,
                             wheel=x['id'] in has_wheel))
    return pd.DataFrame(rows).sort_values(['subject', 'date', 'number']).reset_index(drop=True)


def fibre_target(one, eid):
    try:
        loc = one.load_dataset(eid, 'photometryROI.locations.pqt', collection='alf/photometry')
        return ';'.join(loc['brain_region'].astype(str))
    except Exception:
        return None


def mouse_table(sessions, one=None):
    rows = []
    for subject, g in sessions.groupby('subject'):
        tr = g[g['session_class'] == 'training'].reset_index(drop=True)
        with_ph = tr.index[tr['signal']]
        last_ph = g[g['signal']]
        rows.append(dict(
            subject=subject, line=g['line'].iloc[0],
            target=fibre_target(one, last_ph['eid'].iloc[-1]) if one is not None and len(last_ph) else None,
            first_training=tr['date'].min(), last_training=tr['date'].max(),
            n_training=len(tr), n_training_signal=int(tr['signal'].sum()),
            n_training_signal_wheel=int((tr['signal'] & tr['wheel']).sum()),
            first_signal_training_session=int(with_ph[0]) + 1 if len(with_ph) else None,
            n_biased=int((g['session_class'] == 'biased').sum()),
            n_biased_signal=int(((g['session_class'] == 'biased') & g['signal']).sum()),
            n_ephys_signal=int(((g['session_class'] == 'ephys') & g['signal']).sum()),
            reached_biased=bool((g['session_class'] == 'biased').any())))
    return pd.DataFrame(rows)


def main():
    one = ONE(base_url=BASE_URL, silent=True)
    sessions = session_table(one)
    sessions.to_csv(HERE / 'da_inventory_sessions.csv', index=False)
    mice = mouse_table(sessions, one)
    mice.to_csv(HERE / 'da_inventory_mice.csv', index=False)
    print(mice.to_string(index=False))


if __name__ == '__main__':
    main()
