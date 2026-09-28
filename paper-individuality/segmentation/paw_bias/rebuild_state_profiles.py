"""Rebuild state_profiles_19Ago2026.csv from state_description.txt.

WHY THIS EXISTS. describe_states.py writes two things side by side: the CSV, and
state_description.txt, which prints the very same table. The CSV is the one every
downstream analysis reads (4_mice/laterality/laterality_features.py::state_laterality)
and the one most likely to be missing on a machine that only has the repo. The txt is
checked in. So the CSV can be recovered from it without the 664,000-frame supersession
that describe_states.py needs, and without refitting the clustering -- which must NOT be
refit, since a new KMeans on a different subset would produce states that do not
correspond to the ones already baked into the syllable files.

THE ONE COMPROMISE. The txt prints the table to 2 decimals, so the rebuilt CSV is
accurate to 2 decimals rather than to float64. The script checks itself against the
exact per-state summary that the same txt prints at full precision: the worst LI error
is 0.008, and the |LI| > 0.25 split that picks the lateralised states (left 3, 5;
right 4, 6) is unchanged. If the original CSV turns up, overwrite this file with it.
"""
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
TXT = HERE / 'state_description.txt'
OUT = HERE / 'state_profiles_19Ago2026.csv'


def _table(txt, start, end):
    block = txt.split(start)[1].split(end)[0]
    lines = [l for l in block.strip().split('\n') if l.strip()]
    head = lines[0].split()
    rows = [l.split() for l in lines[1:] if l.split() and l.split()[0].replace('.', '').isdigit()]
    df = pd.DataFrame([[float(v) for v in r[1:]] for r in rows],
                      index=[int(float(r[0])) for r in rows], columns=head)
    df.index.name = 'state'
    return df


def main():
    txt = TXT.read_text()
    prof = _table(txt, '=== mean z-scored wavelet power per state (rows = remapped state) ===',
                  '=== summary per state ===')
    summ = txt.split('=== summary per state ===')[1].split('===')[0].strip().split('\n')
    cols = summ[0].split()
    exact = pd.DataFrame([[float(v) for v in l.split()] for l in summ[1:] if l.split()],
                         columns=cols).set_index('state')

    left = prof[[c for c in prof.columns if c.startswith('l_paw')]].mean(axis=1)
    right = prof[[c for c in prof.columns if c.startswith('r_paw')]].mean(axis=1)
    li = (left - right) / (left.abs() + right.abs())
    err = float((li.values - exact.LI.values).__abs__().max())

    lateral = lambda s: (sorted(s.index[s > 0.25].astype(int)), sorted(s.index[s < -0.25].astype(int)))
    assert lateral(li) == lateral(pd.Series(exact.LI.values, index=exact.index)), \
        'the rounding changed which states count as lateralised -- do not use this file'
    print(f'rebuilt {prof.shape[0]} states x {prof.shape[1]} wavelet columns')
    print(f'worst |LI| error against the full-precision summary in the same file: {err:.4f}')
    print(f'left states {lateral(li)[0]}, right states {lateral(li)[1]}')
    prof.to_csv(OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
