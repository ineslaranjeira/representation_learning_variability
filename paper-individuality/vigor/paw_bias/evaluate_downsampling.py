"""Score the three candidate fixes against the five criteria fixed in test_downsampling.py."""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, '.')
from analyse_paw_bias import icc_and_perm, mouse_level_corr

VARIANTS = [('raw', 'RAW (current pipeline)'),
            ('quant', 'QUANT (snap to right-cam grid)'),
            ('match', 'MATCH (noise floors equalised)'),
            ('whiten', 'WHITEN (divide by own 32 Hz)')]
FRQ = ['0.5', '1.0', '2.0', '4.0', '8.0']


def main(csv):
    df = pd.read_csv(csv)
    if 'err' in df:
        print(f'{df.err.notna().sum()} failures'); df = df[df.err.isna()]
    print(f'== {len(df)} sessions, {df.mouse.nunique()} mice '
          f'({(df.mouse.value_counts() >= 3).sum()} with >=3) ==\n')

    print('estimated positional noise sd (right-cam px): '
          f'left {df.sd_L.median():.3f}, right {df.sd_R.median():.3f}; '
          f'noise added to left paw by MATCH: {df.sigma_added.median():.3f}\n')

    # ---- C1: are the noise floors actually matched? ----
    print('C1  LI[32 Hz power] -> 0   (0 = noise floors equalised)')
    for v, lab in VARIANTS:
        c = f'{v}_li_noisepow'
        if c not in df:
            continue
        s = df[c]
        print(f'    {lab:34s} mean={s.mean():+.4f}  median={s.median():+.4f}  '
              f'|mean| {"PASS" if abs(s.mean()) < 0.05 else "fail"}')

    # ---- C2: does the amplitude index still track the noise index? ----
    print('\nC2  r(LI[band power], LI[32 Hz of the RAW data]) -> 0')
    ref = df.raw_li_noisepow
    for v, lab in VARIANTS:
        c = f'{v}_li_bandpow'
        if c not in df:
            continue
        r, p = stats.pearsonr(df[c], ref)
        print(f'    {lab:34s} r={r:+.3f}  p={p:.1e}')

    # ---- the decisive question: does the fix move the metric at all? ----
    print('\n**  how much does each fix actually MOVE the metric it is meant to fix? **')
    print(f'    {"metric":16s} {"RAW mean":>10s} {"QUANT":>10s} {"MATCH":>10s} '
          f'{"r(raw,quant)":>13s} {"r(raw,match)":>13s}')
    for met in ['li_bandpow', 'li_spmed']:
        row = [df[f'raw_{met}'].mean(), df[f'quant_{met}'].mean(), df[f'match_{met}'].mean()]
        rq = stats.pearsonr(df[f'raw_{met}'], df[f'quant_{met}'])[0]
        rm = stats.pearsonr(df[f'raw_{met}'], df[f'match_{met}'])[0]
        print(f'    {met:16s} {row[0]:+10.4f} {row[1]:+10.4f} {row[2]:+10.4f} '
              f'{rq:13.4f} {rm:13.4f}')
    print('    (r ~ 1.000 means the fix left the metric essentially untouched)')

    # ---- C3: per-frequency trend ----
    print('\nC3  per-frequency LI (mean over sessions) -- a monotonic slide with frequency is the '
          'noise-floor signature')
    print(f'    {"variant":34s} ' + ' '.join(f'{f+"Hz":>9s}' for f in FRQ) + '    slope')
    for v, lab in VARIANTS[:3]:
        vals = [df[f'{v}_li_pow_{float(f)}'].mean() for f in FRQ]
        sl = np.polyfit(np.log2([float(f) for f in FRQ]), vals, 1)[0]
        print(f'    {lab:34s} ' + ' '.join(f'{x:+9.4f}' for x in vals) + f'  {sl:+.4f}/oct')

    # ---- C4: does the individuality survive? ----
    print('\nC4  between-mouse ICC(1) survives the fix')
    for v, lab in VARIANTS:
        c = f'{v}_li_bandpow'
        if c not in df:
            continue
        r = icc_and_perm(df, c, n_perm=2000)
        if r:
            print(f'    {lab:34s} ICC={r["icc"]:.3f}  p_perm={r["p_perm"]:.4f}  '
                  f'({r["n_sess"]} sess, {r["n_mice"]} mice)')
    for v, lab in [('raw', 'RAW  li_spmed'), ('match', 'MATCH li_spmed')]:
        r = icc_and_perm(df, f'{v}_li_spmed', n_perm=2000)
        if r:
            print(f'    {lab:34s} ICC={r["icc"]:.3f}  p_perm={r["p_perm"]:.4f}')

    # ---- C5: does the task link survive? ----
    print('\nC5  mouse-level correlation with the task survives the fix')
    for tgt in ['wheel_bias', 'frac_right_choice']:
        if tgt not in df:
            continue
        for v, lab in VARIANTS:
            c = f'{v}_li_bandpow'
            if c not in df:
                continue
            r = mouse_level_corr(df, c, tgt, n_perm=3000)
            if r:
                print(f'    {tgt:18s} {lab:34s} r={r["r"]:+.3f} rho={r["rho"]:+.3f} '
                      f'p_perm={r["p_perm"]:.4f} (n={r["n_mice"]})')
        print()


if __name__ == '__main__':
    main(sys.argv[1])
