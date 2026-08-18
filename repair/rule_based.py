"""
Rule-based baseline repair layer. Hard-coded heuristics with static thresholds
-- exactly the kind of thing the proposal argues doesn't generalize. It knows
about a fixed set of "known" renames/unit factors and otherwise gives up.
This intentionally-narrow baseline is what the LLM agent is compared against.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

KNOWN_RENAMES = {"trip_dist_miles": "trip_distance"}   # must be hard-coded in advance
KNOWN_UNIT_COLUMNS = {"trip_distance": {"expected_mean_range": (0.5, 8.0), "km_to_mi": 1 / 1.60934}}


def propose_rule_based_patch(day: int, finding_kind: str, column: str,
                              window_df: pd.DataFrame, detail: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Returns a structured patch op, or None if the baseline has no rule for it."""

    if finding_kind == "schema" and detail.get("issue") == "unexpected_new_column":
        if column in KNOWN_RENAMES:
            return {"op": "rename_column", "from": column, "to": KNOWN_RENAMES[column],
                     "source": "rule_based"}
        return None  # unknown new column -> baseline cannot generalize, gives up

    if finding_kind == "missing":
        # baseline's only trick: fill missing with column median if it's numeric
        if pd.api.types.is_numeric_dtype(window_df[column]) or column in ("passenger_count",):
            return {"op": "impute_missing", "column": column, "strategy": "median",
                     "source": "rule_based"}
        return None

    if finding_kind == "statistical":
        if column in KNOWN_UNIT_COLUMNS:
            lo, hi = KNOWN_UNIT_COLUMNS[column]["expected_mean_range"]
            cur_mean = detail.get("cur_mean")
            if cur_mean is not None and cur_mean > hi:
                # naive static-threshold guess: assume km->mi conversion needed
                return {"op": "unit_convert", "column": column,
                         "factor": KNOWN_UNIT_COLUMNS[column]["km_to_mi"], "source": "rule_based"}
        return None  # no static rule matches (e.g. surge-pricing shift) -> baseline misses it

    return None


def apply_patch(df: pd.DataFrame, patch: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    op = patch["op"]
    if op == "rename_column":
        if patch["from"] in df.columns:
            df[patch["to"]] = df[patch["from"]]
            df = df.drop(columns=[patch["from"]])
    elif op == "unit_convert":
        col = patch["column"]
        if col in df.columns:
            df[col] = df[col] * patch["factor"]
    elif op == "impute_missing":
        col = patch["column"]
        if col in df.columns:
            strategy = patch.get("strategy", "median")
            if strategy == "median":
                fill_val = df[col].median()
            elif strategy == "mean":
                fill_val = df[col].mean()
            else:
                fill_val = 0
            df[col] = df[col].fillna(fill_val)
    elif op == "rescale_column":
        col = patch["column"]
        if col in df.columns:
            df[col] = df[col] * patch["factor"]
    else:
        raise ValueError(f"Unknown patch op: {op}")
    return df
