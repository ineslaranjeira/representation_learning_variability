"""
Run segmentation/3.3_wavelet_clusters_uniform.ipynb once per (LOG, NORM).

The notebook has LOG and NORM as knobs; this writes a temporary copy per combination next
to it (so its relative imports and paths hold), executes it headless, and deletes the copy.
Each run writes data/paw_states_uniform_{raw|log}_{session|global}z/. The executed copies
go to ./executed/ so the plots of every variant can be looked at.

Needs 3.2_wavelet_subsample_uniform.ipynb to have been run first.
"""
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
SEG = HERE.parents[1] / 'segmentation'
NB = SEG / '3.3_wavelet_clusters_uniform.ipynb'
OUT = HERE / 'executed'
KNOBS = {'LOG': "LOG    = True ", 'NORM': "NORM   = 'session'  "}


def run(log, norm):
    src = NB.read_text()
    assert all(k in src for k in KNOBS.values()), 'knob lines changed -- update KNOBS'
    src = (src.replace(KNOBS['LOG'], f'LOG    = {log} ')
              .replace(KNOBS['NORM'], f"NORM   = '{norm}'".ljust(len(KNOBS['NORM']))))
    tmp = SEG / f'_tmp_3.3_{log}_{norm}.ipynb'
    tmp.write_text(src)
    try:
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                        '--ExecutePreprocessor.timeout=6000', '--output-dir', str(OUT),
                        '--output', f'3.3_uniform_{"log" if log else "raw"}_{norm}z', str(tmp)],
                       check=True, capture_output=True)
    finally:
        tmp.unlink()
    return log, norm


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        for log, norm in ex.map(lambda a: run(*a), [(l, n) for l in (True, False)
                                                    for n in ('session', 'global')]):
            print('done', log, norm)
