"""
BUILD wheel/vigor_sessions.parquet ON THIS MACHINE -- session-level wheel vigor, no warping
===========================================================================================
whisk_wheel_engagement.ipynb reads wheel/vigor_sessions.parquet (built on the Mac by vigor/wheel/wheel_vigor.py
from data/training/training_design_matrices/, which is not on the linux machine). This rebuilds it with the SAME
metric function (wheel_vigor.session_vigor: 30 Hz common grid, winsorised speeds, |v| > 0.1 = moving) from the
local raw design matrices:
  source 'training'       first training session  (data/design_matrices/1_camera_setup/session_1)
  source 'last_training'  last training session   (data/design_matrices/1_camera_setup/last_training)
The sessions are the ones in the legacy syllable files the notebook uses (whisk_wheel_data.SYL_FILES).
Run from learning_individuality/:  python build_local_vigor.py
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'vigor', 'wheel'))
import whisk_wheel_data as wd           # noqa: E402
import wheel_preprocessing as wp        # noqa: E402
import wheel_vigor as wv                # noqa: E402

OUT = os.path.join(HERE, 'wheel', 'vigor_sessions.parquet')
parts = []
for tp, source in [('Early', 'training'), ('Late', 'last_training')]:
    s = pd.read_parquet(wd.SYL_FILES[tp], columns=['sample', 'mouse_name'])
    table = (s.assign(session=s['sample'].str[:36])[['mouse_name', 'session']].drop_duplicates()
              .assign(source=source))
    wp.SOURCE_DIR[source] = wd.DM_DIRS[tp] + '/'        # session_vigor reads the design matrix from here
    v = wv.build_vigor(table)
    print(f'{tp}: {len(v)} of {len(table)} sessions from {wd.DM_DIRS[tp]}')
    parts.append(v)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.concat(parts, ignore_index=True).to_parquet(OUT, index=False)
print('wrote', OUT)
