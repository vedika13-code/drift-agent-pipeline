"""
Baseline drift detector: rolling-window KS-test + PSI for numeric columns,
plus a schema diff against a registered reference schema. This is the
"rule-based / static-threshold" baseline the proposal calls out as not
generalizing -- we deliberately keep it simple and threshold-driven.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

KS_ALPHA = 0.01          # p-value threshold for KS test
PSI_THRESHOLD = 0.2       # standard PSI "significant shift" threshold
NULL_RATE_THRESHOLD = 0.15  # fraction of nulls that trips a "missing field" flag


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = reference[~pd.isna(reference)]
    current = current[~pd.isna(current)]
    if len(reference) < 5 or len(current) < 5:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(reference, quantiles))
    if len(cut_points) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=cut_points)
    cur_counts, _ = np.histogram(current, bins=cut_points)
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


@dataclass
class DriftFinding:
    day: int
    kind: str            # "schema" | "statistical" | "missing"
    column: str
    detail: Dict[str, Any]
    severity: str          # "info" | "warn" | "critical"


class DriftDetector:
    # columns that are structural/partitioning keys, not measured pipeline values --
    # comparing their distribution across windows is meaningless (e.g. "day" is by
    # construction a single constant value within any single-day window).
    NON_STATISTICAL_COLUMNS = {"day"}

    def __init__(self, reference_df: pd.DataFrame, reference_schema: Dict[str, str],
                 numeric_columns: Optional[List[str]] = None):
        self.reference_df = reference_df
        self.reference_schema = reference_schema
        self.numeric_columns = numeric_columns or [
            c for c, dt in reference_schema.items()
            if dt in ("float64", "int64") and c not in self.NON_STATISTICAL_COLUMNS
        ]

    def check_schema(self, day: int, window_df: pd.DataFrame) -> List[DriftFinding]:
        findings = []
        current_cols = set(window_df.columns)
        expected_cols = set(self.reference_schema.keys())

        for missing_col in (expected_cols - current_cols):
            findings.append(DriftFinding(day, "schema", missing_col,
                                          {"issue": "column_missing_from_frame"}, "critical"))
        for new_col in (current_cols - expected_cols):
            # only flag if it actually carries data in this window
            if window_df[new_col].notna().any():
                findings.append(DriftFinding(day, "schema", new_col,
                                              {"issue": "unexpected_new_column"}, "warn"))
        for col in (expected_cols & current_cols):
            if col not in window_df.columns:
                continue
            null_rate = window_df[col].isna().mean()
            if null_rate >= NULL_RATE_THRESHOLD:
                findings.append(DriftFinding(day, "missing", col,
                                              {"null_rate": round(float(null_rate), 3)},
                                              "critical" if null_rate > 0.5 else "warn"))
        return findings

    def check_statistics(self, day: int, window_df: pd.DataFrame) -> List[DriftFinding]:
        findings = []
        for col in self.numeric_columns:
            if col not in window_df.columns:
                continue
            ref_vals = pd.to_numeric(self.reference_df[col], errors="coerce").dropna().to_numpy()
            cur_vals = pd.to_numeric(window_df[col], errors="coerce").dropna().to_numpy()
            if len(cur_vals) < 5 or len(ref_vals) < 5:
                continue
            stat, p_value = ks_2samp(ref_vals, cur_vals)
            psi = population_stability_index(ref_vals, cur_vals)
            drifted = (p_value < KS_ALPHA) or (psi >= PSI_THRESHOLD)
            if drifted:
                severity = "critical" if psi >= 0.5 or p_value < 1e-6 else "warn"
                findings.append(DriftFinding(
                    day, "statistical", col,
                    {"ks_stat": round(float(stat), 4), "p_value": float(p_value), "psi": round(psi, 4),
                     "ref_mean": round(float(np.mean(ref_vals)), 3),
                     "cur_mean": round(float(np.mean(cur_vals)), 3)},
                    severity))
        return findings

    def run_window(self, day: int, window_df: pd.DataFrame) -> List[DriftFinding]:
        return self.check_schema(day, window_df) + self.check_statistics(day, window_df)
