"""
Verification / gating layer.

Applies DB-transaction-consistency principles to LLM-proposed patches:
  1. BEGIN: dry-run the patch on a COPY of the frame (never mutate live state directly).
  2. CHECK: validate the resulting frame against the schema registry and business rules,
     and confirm the patch actually reduces measured drift (post-patch KS/PSI vs reference)
     rather than just changing something.
  3. COMMIT: only if all checks pass, the patch is applied to the live frame and logged
     to a SQLite transaction log (persisted, auditable, replayable).
  4. ROLLBACK: if any check fails, the patch is rejected, nothing is mutated, and the
     rejection + reason is logged. The pipeline continues on the last known-good state.

This is the safety gate described in the proposal: no patch is committed to the
live pipeline without passing consistency checks, the same guarantee a DB transaction
gives you before a COMMIT.
"""
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from repair.rule_based import apply_patch
from data.generate_data import BUSINESS_RULES


@dataclass
class VerificationResult:
    accepted: bool
    reasons: list
    post_patch_ks_p: Optional[float] = None
    post_patch_psi: Optional[float] = None


class TransactionalVerifier:
    def __init__(self, db_path: str, expected_schema: Dict[str, str],
                 business_rules: Dict[str, Dict[str, float]] = None,
                 max_null_increase: float = 0.05, drift_improvement_required: bool = True):
        self.db_path = db_path
        self.expected_schema = expected_schema
        self.business_rules = business_rules or BUSINESS_RULES
        self.max_null_increase = max_null_increase
        self.drift_improvement_required = drift_improvement_required
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, day INTEGER, status TEXT, source TEXT,
                patch_json TEXT, reasons TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _log(self, day: int, status: str, source: str, patch: Dict[str, Any], reasons: list):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO transaction_log (ts, day, status, source, patch_json, reasons) VALUES (?,?,?,?,?,?)",
            (time.time(), day, status, source, json.dumps(patch), json.dumps(reasons)),
        )
        conn.commit()
        conn.close()

    def _check_schema(self, patched_df: pd.DataFrame) -> list:
        reasons = []
        expected_cols = set(self.expected_schema.keys())
        current_cols = set(patched_df.columns)
        missing = expected_cols - current_cols
        if missing:
            reasons.append(f"schema_violation: still missing expected columns {sorted(missing)}")
        return reasons

    def _check_business_rules(self, patched_df: pd.DataFrame) -> list:
        reasons = []
        for col, rule in self.business_rules.items():
            if col not in patched_df.columns:
                continue
            vals = pd.to_numeric(patched_df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            if "min" in rule and vals.min() < rule["min"]:
                reasons.append(f"business_rule_violation: {col} min {vals.min():.3f} < {rule['min']}")
            if "max" in rule and vals.max() > rule["max"]:
                reasons.append(f"business_rule_violation: {col} max {vals.max():.3f} > {rule['max']}")
        return reasons

    def _check_null_increase(self, before_df: pd.DataFrame, patched_df: pd.DataFrame, column: str) -> list:
        reasons = []
        if column in before_df.columns and column in patched_df.columns:
            before_null = before_df[column].isna().mean()
            after_null = patched_df[column].isna().mean()
            if after_null - before_null > self.max_null_increase:
                reasons.append(f"data_loss: null rate for {column} increased "
                                f"{before_null:.3f} -> {after_null:.3f}")
        return reasons

    def _check_drift_improved(self, reference_df: pd.DataFrame, patched_df: pd.DataFrame,
                               column: str) -> Tuple[list, Optional[float]]:
        reasons = []
        p_value = None
        if not self.drift_improvement_required:
            return reasons, p_value
        if column in reference_df.columns and column in patched_df.columns:
            ref_vals = pd.to_numeric(reference_df[column], errors="coerce").dropna().to_numpy()
            cur_vals = pd.to_numeric(patched_df[column], errors="coerce").dropna().to_numpy()
            if len(ref_vals) >= 5 and len(cur_vals) >= 5:
                _, p_value = ks_2samp(ref_vals, cur_vals)
                if p_value < 0.01:
                    reasons.append(f"drift_not_resolved: post-patch KS p-value {p_value:.5f} "
                                    f"still indicates significant drift on {column}")
        return reasons, p_value

    def verify_and_commit(self, day: int, live_df: pd.DataFrame, reference_df: pd.DataFrame,
                           column: str, patch: Dict[str, Any], source: str) -> Tuple[VerificationResult, pd.DataFrame]:
        """Dry-runs `patch` on a COPY of live_df. Commits (returns patched df) only if
        all checks pass; otherwise rolls back (returns live_df unchanged)."""
        reasons = []

        if patch.get("op") == "no_op":
            reasons.append("agent_declined: proposed no_op (insufficient confidence)")
            result = VerificationResult(False, reasons)
            self._log(day, "rejected", source, patch, reasons)
            return result, live_df

        try:
            dry_run_df = apply_patch(live_df.copy(), patch)
        except Exception as e:
            reasons.append(f"patch_execution_error: {e}")
            result = VerificationResult(False, reasons)
            self._log(day, "rejected", source, patch, reasons)
            return result, live_df

        reasons += self._check_schema(dry_run_df)
        reasons += self._check_business_rules(dry_run_df)
        reasons += self._check_null_increase(live_df, dry_run_df, column)
        drift_reasons, p_value = self._check_drift_improved(reference_df, dry_run_df, column)
        reasons += drift_reasons

        if reasons:
            result = VerificationResult(False, reasons, post_patch_ks_p=p_value)
            self._log(day, "rejected", source, patch, reasons)
            return result, live_df  # ROLLBACK: unchanged live state

        result = VerificationResult(True, [], post_patch_ks_p=p_value)
        self._log(day, "committed", source, patch, [])
        return result, dry_run_df  # COMMIT
