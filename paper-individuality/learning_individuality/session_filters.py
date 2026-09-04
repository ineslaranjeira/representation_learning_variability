"""
Session exclusions, driven by the curated QC sheet.
=========================================================================
Single source of truth for "which sessions do we drop", replacing the hardcoded
`prob_sessions` / `session_1` / `last_training` lists that used to be copied between
scripts.

The sheet is `individuality-paper_data_3sep26.csv`. Two things about it matter:
  * it has a TWO-ROW header (merged spreadsheet cells), so it must be read with
    header=1 or every column comes back as 'Unnamed: N';
  * `source_file` identifies which dataset a session belongs to, and
    `task_protocol` corroborates it.

`Used in paper` takes exactly three values -- 'yes', 'filtered out',
'need to revise' -- which gives the two strictness levels below.
"""
from pathlib import Path
import pandas as pd

CSV_NAME = 'individuality-paper_data_4Sep26.csv'

# source_file -> timepoint label used by the analysis scripts.
# Pre-rec spans three files because those sessions are the 1st/2nd/3rd biased
# session before recording (up to 3 per mouse).
SOURCE_TO_TIMEPOINT = {
    'first_training_eids.csv':        'Early',
    'last_training_eids.csv':         'Late',
    'biased_before_ephys_1_eids.csv': 'Pre-rec',
    'biased_before_ephys_2_eids.csv': 'Pre-rec',
    'biased_before_ephys_3_eids.csv': 'Pre-rec',
    'bwm_qc_new_08-03-2026':          'Proficient',
}

# The filter parameter. 'filtered_out' takes `Used in paper` literally;
# 'filtered_out_and_revise' additionally drops the sessions still under review.
EXCLUDE_LEVELS = {
    'filtered_out':            ('filtered out',),
    'filtered_out_and_revise': ('filtered out', 'need to revise'),
}


def load_session_table(csv_path=None):
    """The QC sheet as a tidy frame with an added `timepoint` column."""
    path = Path(csv_path) if csv_path else Path(__file__).with_name(CSV_NAME)
    df = pd.read_csv(path, header=1, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in ('eid', 'source_file', 'Used in paper'):
        if col not in df.columns:
            raise ValueError(f"{path.name} has no {col!r} column -- did the header "
                             f"layout change? (it is read with header=1)")
    df = df.dropna(subset=['eid'])
    df['timepoint'] = df['source_file'].map(SOURCE_TO_TIMEPOINT)
    unknown = sorted(df.loc[df['timepoint'].isna(), 'source_file'].dropna().unique())
    if unknown:
        print(f"!! source_file values not mapped to a timepoint: {unknown}")
    return df


def sessions_to_exclude(timepoint=None, strictness='filtered_out', csv_path=None,
                        table=None):
    """
    eids to drop. `timepoint` None means every timepoint pooled.

    strictness : 'filtered_out'            -> Used in paper == 'filtered out'
                 'filtered_out_and_revise' -> that PLUS 'need to revise'
    """
    if strictness not in EXCLUDE_LEVELS:
        raise ValueError(f"strictness must be one of {sorted(EXCLUDE_LEVELS)}")
    df = table if table is not None else load_session_table(csv_path)
    if timepoint is not None:
        df = df[df['timepoint'] == timepoint]
    return set(df.loc[df['Used in paper'].isin(EXCLUDE_LEVELS[strictness]), 'eid'])


def exclusions_by_timepoint(strictness='filtered_out', csv_path=None):
    """{timepoint: set(eids to drop)} -- ready to hand to a loader."""
    table = load_session_table(csv_path)
    labels = [l for l in dict.fromkeys(SOURCE_TO_TIMEPOINT.values())]
    return {lab: sessions_to_exclude(lab, strictness, table=table) for lab in labels}


def report_exclusions(sessions_by_timepoint, strictness='filtered_out', csv_path=None):
    """
    Print what the filter does to each dataset, given {timepoint: set(eids in file)}.

    Sessions present in a file but ABSENT from the sheet are KEPT, and counted in
    the `unlisted` column -- dropping data merely because it is not listed would be
    the wrong default, but it should be visible.
    """
    table = load_session_table(csv_path)
    print(f"session exclusions  (strictness = {strictness!r})")
    print(f"  {'timepoint':11s} {'in file':>8s} {'listed':>7s} {'unlisted':>9s} "
          f"{'excluded':>9s} {'kept':>6s}")
    out = {}
    for lab, in_file in sessions_by_timepoint.items():
        in_file = set(in_file)
        listed = set(table.loc[table['timepoint'] == lab, 'eid'])
        drop = sessions_to_exclude(lab, strictness, table=table) & in_file
        out[lab] = drop
        print(f"  {lab:11s} {len(in_file):8d} {len(in_file & listed):7d} "
              f"{len(in_file - listed):9d} {len(drop):9d} {len(in_file - drop):6d}")
    return out


if __name__ == '__main__':
    for lvl in EXCLUDE_LEVELS:
        ex = exclusions_by_timepoint(lvl)
        print(f"{lvl:26s} " + "  ".join(f"{k}:{len(v)}" for k, v in ex.items()))
