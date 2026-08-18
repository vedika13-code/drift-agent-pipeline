"""
Injects controlled, KNOWN drift events into the clean synthetic taxi data.

Each event has a day it starts on, a type, and a `ground_truth_fix` describing
the correct repair. This ground truth is used ONLY for evaluation (not visible
to the detector/agent), so we can score detection recall/precision and repair
correctness objectively.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class DriftEvent:
    name: str
    start_day: int
    kind: str  # "schema_rename" | "unit_shift" | "missing_field" | "distribution_shift"
    column: str
    apply_fn: Callable[[pd.DataFrame], pd.DataFrame]
    ground_truth_fix: Dict[str, Any] = field(default_factory=dict)


def _rename_column(df, start_day, old, new):
    df = df.copy()
    mask = df["day"] >= start_day
    # once renamed downstream, the column literally doesn't exist under old name
    renamed_part = df.loc[mask].rename(columns={old: new})
    before_part = df.loc[~mask]
    # combine: after start_day the frame has `new`, before it has `old`.
    # To keep a single frame walkable window-by-window we instead just rename
    # the physical column for all rows >= start_day and leave a NaN-old column.
    df.loc[mask, new] = df.loc[mask, old]
    df.loc[mask, old] = np.nan
    return df


def _unit_shift(df, start_day, column, factor):
    df = df.copy()
    mask = df["day"] >= start_day
    df.loc[mask, column] = df.loc[mask, column] * factor
    return df


def _missing_field(df, start_day, column, frac_missing=1.0):
    df = df.copy()
    mask = df["day"] >= start_day
    if frac_missing >= 0.999:
        df.loc[mask, column] = np.nan
    else:
        idx = df.loc[mask].sample(frac=frac_missing, random_state=1).index
        df.loc[idx, column] = np.nan
    return df


def _distribution_shift(df, start_day, column, shift_mult):
    df = df.copy()
    mask = df["day"] >= start_day
    df.loc[mask, column] = df.loc[mask, column] * shift_mult
    return df


def default_drift_scenarios():
    """Four canonical drift events at staggered days, matching the proposal:
    schema drift, unit shift, missing fields, statistical distribution shift."""
    events = [
        DriftEvent(
            name="rename_trip_distance",
            start_day=15,
            kind="schema_rename",
            column="trip_distance",
            apply_fn=lambda df: _rename_column(df, 15, "trip_distance", "trip_dist_miles"),
            ground_truth_fix={"op": "rename_column", "from": "trip_dist_miles", "to": "trip_distance"},
        ),
        DriftEvent(
            name="tip_amount_unit_shift_cents",
            start_day=25,
            kind="unit_shift",
            column="tip_amount",
            apply_fn=lambda df: _unit_shift(df, 25, "tip_amount", 100.0),
            ground_truth_fix={"op": "unit_convert", "column": "tip_amount", "factor": 1 / 100.0},
        ),
        DriftEvent(
            name="passenger_count_missing",
            start_day=38,
            kind="missing_field",
            column="passenger_count",
            apply_fn=lambda df: _missing_field(df, 38, "passenger_count", frac_missing=0.9),
            ground_truth_fix={"op": "impute_missing", "column": "passenger_count", "strategy": "median"},
        ),
        DriftEvent(
            name="fare_surge_distribution_shift",
            start_day=48,
            kind="distribution_shift",
            column="fare_amount",
            apply_fn=lambda df: _distribution_shift(df, 48, "fare_amount", 1.8),
            ground_truth_fix={"op": "rescale_column", "column": "fare_amount", "factor": 1 / 1.8},
        ),
    ]
    return events


def apply_all_drift(df: pd.DataFrame, events=None):
    events = events or default_drift_scenarios()
    out = df.copy()
    for ev in sorted(events, key=lambda e: e.start_day):
        out = ev.apply_fn(out)
    return out, events


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.generate_data import generate_clean_taxi_data

    df = generate_clean_taxi_data()
    dirty, events = apply_all_drift(df)
    print(dirty.columns.tolist())
    print(dirty[dirty["day"] == 20][["trip_distance", "trip_dist_miles"]].head())
    print(dirty[dirty["day"] == 40][["trip_distance"]].describe())
