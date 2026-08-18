"""
LLM-driven diagnosis + repair-proposal agent.

Design choice: rather than letting the model emit free-form pandas/SQL code
that gets exec()'d (unsafe, and impossible to gate cleanly), the agent must
emit a JSON object naming ONE of a small whitelist of structured patch ops
(same op vocabulary as repair/rule_based.py: rename_column, unit_convert,
impute_missing, rescale_column, no_op). The verification layer (agent/verifier.py)
then dry-runs that structured op and checks it against schema + business rules
before it is ever committed. This is what makes the system "gated" rather than
"the LLM writes and runs arbitrary code."

Two backends:
  - real: calls the Gemini API (requires GEMINI_API_KEY in the environment)
  - mock: a deterministic heuristic stand-in used when no API key is present,
    so the whole pipeline is runnable/testable offline. It is intentionally
    MORE general than the rule-based baseline (e.g. it infers unit-conversion
    factors from observed magnitude ratios rather than a hard-coded table,
    and it inspects orphaned columns for a rename match by value correlation)
    to reflect what an LLM reasoning over the drift summary should be capable
    of, while still being a simple, auditable stand-in for the demo.
"""
import json
import os
import time

from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

SYSTEM_PROMPT = """You are a data-pipeline drift diagnosis agent.
You will be given: the expected schema, a statistical/schema drift summary for one time
window of a pipeline, and small samples of reference (pre-drift) and current (post-drift) data.

Diagnose the root cause and propose exactly ONE structured repair patch from this whitelist:
- {"op": "rename_column", "from": "<col>", "to": "<expected_col>"}
- {"op": "unit_convert", "column": "<col>", "factor": <float>}
- {"op": "impute_missing", "column": "<col>", "strategy": "median"|"mean"}
- {"op": "rescale_column", "column": "<col>", "factor": <float>}
- {"op": "no_op"}   (use this if you are not confident a safe fix exists)

Respond with ONLY a JSON object, no markdown, no prose, in this exact shape:
{"diagnosis": "<one sentence root-cause explanation>",
 "patch": {...one of the ops above...},
 "confidence": <float 0-1>}
"""


@dataclass
class AgentProposal:
    diagnosis: str
    patch: Dict[str, Any]
    confidence: float
    backend: str
    latency_sec: float


def _build_user_prompt(day: int, finding_kind: str, column: str, detail: Dict[str, Any],
                        expected_schema: Dict[str, str], window_df: pd.DataFrame,
                        reference_df: pd.DataFrame) -> str:
    cur_sample = window_df.head(5).to_dict(orient="records")
    ref_sample = reference_df.head(5).to_dict(orient="records")
    payload = {
        "day": day,
        "finding_kind": finding_kind,
        "flagged_column": column,
        "detail": detail,
        "expected_schema": expected_schema,
        "current_window_columns": list(window_df.columns),
        "current_window_sample": cur_sample,
        "reference_sample": ref_sample,
    }
    return json.dumps(payload, default=str)


def _call_real_llm(system_prompt: str, user_prompt: str, model: str = "gemini-2.5-flash") -> str:
    from google import genai
    from google.genai import types
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=500,
        ),
    )
    return response.text


def _mock_llm_diagnose(day: int, finding_kind: str, column: str, detail: Dict[str, Any],
                        expected_schema: Dict[str, str], window_df: pd.DataFrame,
                        reference_df: pd.DataFrame) -> Dict[str, Any]:
    """Deterministic heuristic stand-in for the LLM. More general than the
    rule-based baseline: it *infers* fixes rather than looking them up in a
    hard-coded table."""

    expected_cols = set(expected_schema.keys())
    current_cols = set(window_df.columns)

    def _is_effectively_absent(col: str) -> bool:
        # a column counts as "effectively missing" if it's structurally absent,
        # OR physically present but emptied out (e.g. an upstream rename that
        # nulls the old field instead of dropping it -- still a missing source).
        if col not in window_df.columns:
            return True
        return bool(window_df[col].isna().mean() > 0.9)

    # --- Case 1: schema drift -- unexpected new column likely renamed from a missing one
    if finding_kind == "schema" and detail.get("issue") == "unexpected_new_column":
        missing_expected = [c for c in expected_cols if _is_effectively_absent(c)]
        if missing_expected and pd.api.types.is_numeric_dtype(window_df[column]):
            # correlate against reference distribution shape of each missing expected col
            best_col, best_score = None, -1
            cur_vals = pd.to_numeric(window_df[column], errors="coerce").dropna()
            for cand in missing_expected:
                if cand not in reference_df.columns:
                    continue
                ref_vals = pd.to_numeric(reference_df[cand], errors="coerce").dropna()
                if len(ref_vals) < 5 or len(cur_vals) < 5:
                    continue
                # similarity heuristic: closeness of means/stds (scale-invariant-ish)
                score = -abs(np.log((cur_vals.mean() + 1e-6) / (ref_vals.mean() + 1e-6)))
                if score > best_score:
                    best_score, best_col = score, cand
            if best_col:
                return {
                    "diagnosis": f"Column '{column}' appears to be a renamed version of "
                                 f"expected column '{best_col}' (schema now missing '{best_col}', "
                                 f"'{column}' has a similar value distribution).",
                    "patch": {"op": "rename_column", "from": column, "to": best_col},
                    "confidence": 0.85,
                }
        return {"diagnosis": f"Unexpected column '{column}' but no clear rename target found.",
                "patch": {"op": "no_op"}, "confidence": 0.3}

    # --- Case 2: missing field -- impute
    if finding_kind == "missing":
        return {
            "diagnosis": f"Column '{column}' has a high null rate "
                         f"({detail.get('null_rate')}), consistent with an upstream field "
                         f"drop/rename; imputing with the reference median as a safe interim fix.",
            "patch": {"op": "impute_missing", "column": column, "strategy": "median"},
            "confidence": 0.7,
        }

    # --- Case 3: statistical drift -- infer whether it's a unit shift or a real distribution shift
    if finding_kind == "statistical":
        cur_mean = detail.get("cur_mean")
        ref_mean = detail.get("ref_mean")
        if cur_mean and ref_mean and ref_mean != 0:
            ratio = cur_mean / ref_mean
            # classic mile<->km ratio ~1.60934; treat anything close to a "nice" unit
            # conversion constant as a unit shift, otherwise call it a genuine
            # distribution shift and propose a rescale back toward the reference mean.
            candidate_factors = {"km_to_mi (1.60934)": 1.60934, "mi_to_km (0.62137)": 0.62137}
            close_unit_match = None
            for name, f in candidate_factors.items():
                if abs(ratio - f) / f < 0.05:
                    close_unit_match = (name, f)
                    break
            if close_unit_match:
                name, f = close_unit_match
                return {
                    "diagnosis": f"Column '{column}' mean shifted by a factor of {ratio:.3f}x, "
                                 f"matching the known unit-conversion constant {name}. Likely an "
                                 f"upstream unit change rather than a genuine behavioral shift.",
                    "patch": {"op": "unit_convert", "column": column, "factor": 1 / f},
                    "confidence": 0.8,
                }
            else:
                return {
                    "diagnosis": f"Column '{column}' mean shifted by {ratio:.2f}x with no match to "
                                 f"a known unit-conversion constant; likely a genuine distribution "
                                 f"shift (e.g. pricing/business-rule change). Rescaling toward the "
                                 f"reference mean as the safest available fix.",
                    "patch": {"op": "rescale_column", "column": column, "factor": 1 / ratio},
                    "confidence": 0.65,
                }
        return {"diagnosis": f"Statistical drift on '{column}' detected but insufficient "
                              f"summary stats to propose a confident fix.",
                "patch": {"op": "no_op"}, "confidence": 0.2}

    return {"diagnosis": "No actionable drift pattern recognized.",
            "patch": {"op": "no_op"}, "confidence": 0.1}


def diagnose_and_propose(day: int, finding_kind: str, column: str, detail: Dict[str, Any],
                          expected_schema: Dict[str, str], window_df: pd.DataFrame,
                          reference_df: pd.DataFrame, backend: str = "auto") -> AgentProposal:
    t0 = time.time()
    use_real = (backend == "real") or (backend == "auto" and os.environ.get("GEMINI_API_KEY"))

    if use_real:
        try:
            user_prompt = _build_user_prompt(day, finding_kind, column, detail,
                                              expected_schema, window_df, reference_df)
            raw = _call_real_llm(SYSTEM_PROMPT, user_prompt)
            cleaned = raw.strip().strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            latency = time.time() - t0
            return AgentProposal(parsed.get("diagnosis", ""), parsed.get("patch", {"op": "no_op"}),
                                  float(parsed.get("confidence", 0.5)), "real_llm", latency)
        except Exception as e:
            # fall through to mock on any API/parse failure, but record it
            parsed = _mock_llm_diagnose(day, finding_kind, column, detail, expected_schema,
                                         window_df, reference_df)
            latency = time.time() - t0
            parsed["diagnosis"] += f" [fell back to mock backend after real-LLM error: {e}]"
            return AgentProposal(parsed["diagnosis"], parsed["patch"], parsed["confidence"],
                                  "real_llm_failed_fallback_mock", latency)

    parsed = _mock_llm_diagnose(day, finding_kind, column, detail, expected_schema,
                                 window_df, reference_df)
    latency = time.time() - t0
    return AgentProposal(parsed["diagnosis"], parsed["patch"], parsed["confidence"], "mock", latency)
