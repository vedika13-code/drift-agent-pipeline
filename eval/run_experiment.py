"""
End-to-end experiment: generates synthetic taxi data, injects known drift events,
then runs TWO independent pipelines day-by-day over it:
  1. baseline: rolling KS/PSI detector + hard-coded rule-based repair
  2. agentic:  same detector + LLM-agent diagnosis/repair, gated by the
               transactional verifier before any commit

Both are logged to their own SQLite transaction log and a shared metrics log,
then compared on: detection precision/recall, repair correctness,
time-to-repair, % unsafe patches caught (adversarial test), false-positive rate.

Run with:  python -m eval.run_experiment
Optional:  GEMINI_API_KEY=AIza... python -m eval.run_experiment   (uses real LLM)
"""
import os
import sys
import json
import time

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.generate_data import generate_clean_taxi_data, EXPECTED_SCHEMA, BUSINESS_RULES
from drift.injector import apply_all_drift, default_drift_scenarios
from drift.detector import DriftDetector
from repair.rule_based import propose_rule_based_patch, apply_patch
from agent.llm_agent import diagnose_and_propose
from agent.verifier import TransactionalVerifier
from eval.metrics import summarize_system_log

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
WARMUP_DAYS = 10  # days 0-9 form the reference window; processing starts day 10


def event_detect_columns(ev):
    if ev.kind == "schema_rename":
        return [ev.ground_truth_fix["from"], ev.ground_truth_fix["to"]]
    return [ev.column]


def is_repair_correct(patch, ev):
    if ev is None or patch is None:
        return False
    gt = ev.ground_truth_fix
    # unit_convert and rescale_column are functionally identical (both do
    # df[col] *= factor); treat them as the same op family for scoring since
    # what matters is whether the resulting VALUE is correct, not the label
    # the agent chose to describe it with.
    scale_ops = {"unit_convert", "rescale_column"}
    same_op_family = (patch.get("op") == gt.get("op")) or (
        patch.get("op") in scale_ops and gt.get("op") in scale_ops
    )
    if not same_op_family:
        return False
    if gt["op"] == "rename_column":
        return patch.get("to") == gt.get("to")
    if gt["op"] in scale_ops:
        f, gtf = patch.get("factor"), gt.get("factor")
        if f is None or gtf is None:
            return False
        return abs(f - gtf) / abs(gtf) < 0.15
    if gt["op"] == "impute_missing":
        return patch.get("column") == gt.get("column")
    return False


def match_event(finding_column, active_events):
    for ev in active_events:
        if finding_column in event_detect_columns(ev):
            return ev
    return None


def baseline_propose(day, finding, window, reference_df, expected_schema):
    t0 = time.time()
    patch = propose_rule_based_patch(day, finding.kind, finding.column, window, finding.detail)
    latency = time.time() - t0
    return patch, "rule_based", 1.0 if patch else 0.0, latency


def agent_propose(day, finding, window, reference_df, expected_schema):
    proposal = diagnose_and_propose(day, finding.kind, finding.column, finding.detail,
                                     expected_schema, window, reference_df, backend="auto")
    patch = proposal.patch
    if patch.get("op") == "no_op":
        patch = None
    return patch, proposal.backend, proposal.confidence, proposal.latency_sec


def run_system(system_name, propose_fn, dirty_df, reference_df, events, expected_schema, db_path):
    verifier = TransactionalVerifier(db_path, expected_schema, business_rules=BUSINESS_RULES)
    detector = DriftDetector(reference_df, expected_schema)
    learned_fixes = []
    log_rows = []
    event_detected_day, event_fixed_day = {}, {}
    days = sorted(dirty_df["day"].unique())

    for day in days:
        if day < WARMUP_DAYS:
            continue
        window = dirty_df[dirty_df["day"] == day].copy()
        for p in learned_fixes:
            try:
                window = apply_patch(window, p)
            except Exception:
                pass

        findings = detector.run_window(day, window)
        active_events = [ev for ev in events if day >= ev.start_day]

        for finding in findings:
            matched_event = match_event(finding.column, active_events)
            is_tp = matched_event is not None
            log_rows.append({"system": system_name, "record_type": "detection", "day": day,
                              "column": finding.column, "kind": finding.kind, "is_true_positive": is_tp})
            if matched_event and matched_event.name not in event_detected_day:
                event_detected_day[matched_event.name] = day

            patch, backend, confidence, latency = propose_fn(day, finding, window, reference_df, expected_schema)
            if patch is None:
                log_rows.append({"system": system_name, "record_type": "repair_attempt", "day": day,
                                  "column": finding.column, "accepted": False,
                                  "reasons": ["no_patch_proposed"], "repair_correct": False,
                                  "latency_sec": latency, "confidence": confidence, "patch": None})
                continue

            v_result, patched_window = verifier.verify_and_commit(
                day, window, reference_df, finding.column, patch, system_name)
            correct = is_repair_correct(patch, matched_event) if v_result.accepted else False
            log_rows.append({"system": system_name, "record_type": "repair_attempt", "day": day,
                              "column": finding.column, "accepted": v_result.accepted,
                              "reasons": v_result.reasons, "repair_correct": correct,
                              "latency_sec": latency, "confidence": confidence, "patch": patch})
            if v_result.accepted:
                window = patched_window
                learned_fixes.append(patch)
                if matched_event and matched_event.name not in event_fixed_day:
                    event_fixed_day[matched_event.name] = day

    for ev in events:
        if ev.name in event_fixed_day:
            log_rows.append({"system": system_name, "record_type": "time_to_repair",
                              "event": ev.name, "ttr_days": event_fixed_day[ev.name] - ev.start_day})
        else:
            log_rows.append({"system": system_name, "record_type": "missed_event", "event": ev.name})

    return log_rows, event_detected_day, event_fixed_day


def adversarial_unsafe_patch_test(reference_df, expected_schema, db_path, n=20):
    """Feeds the verifier deliberately unsafe/malformed patches (simulating a
    hallucinating or buggy agent) and measures how many it correctly rejects.
    This is the '% unsafe patches caught' metric."""
    import numpy as np
    rng = np.random.default_rng(3)
    verifier = TransactionalVerifier(db_path, expected_schema, business_rules=BUSINESS_RULES)
    window = reference_df.copy()

    unsafe_patches = []
    for _ in range(n):
        kind = rng.choice(["bad_rename", "extreme_unit", "negative_scale", "wrong_column", "huge_factor"])
        if kind == "bad_rename":
            unsafe_patches.append(("trip_distance", {"op": "rename_column", "from": "trip_distance",
                                                       "to": "totally_made_up_column"}))
        elif kind == "extreme_unit":
            unsafe_patches.append(("fare_amount", {"op": "unit_convert", "column": "fare_amount",
                                                     "factor": 500.0}))  # would blow past business max
        elif kind == "negative_scale":
            unsafe_patches.append(("trip_distance", {"op": "rescale_column", "column": "trip_distance",
                                                       "factor": -3.0}))  # produces negative distances
        elif kind == "wrong_column":
            unsafe_patches.append(("passenger_count", {"op": "impute_missing", "column": "nonexistent_col",
                                                          "strategy": "median"}))
        else:
            unsafe_patches.append(("fare_amount", {"op": "rescale_column", "column": "fare_amount",
                                                     "factor": 1000.0}))

    caught = 0
    results = []
    for i, (col, patch) in enumerate(unsafe_patches):
        v_result, _ = verifier.verify_and_commit(9000 + i, window, reference_df, col, patch, "adversarial_test")
        was_caught = not v_result.accepted
        caught += int(was_caught)
        results.append({"patch": patch, "caught": was_caught, "reasons": v_result.reasons})

    return {"n_unsafe_patches": n, "n_caught": caught,
            "pct_unsafe_caught": round(100 * caught / n, 1), "details": results}


def main():
    print("=" * 70)
    print("Generating synthetic taxi data + injecting controlled drift...")
    clean_df = generate_clean_taxi_data()
    dirty_df, events = apply_all_drift(clean_df)
    reference_df = dirty_df[dirty_df["day"] < WARMUP_DAYS].copy()

    print(f"Injected {len(events)} drift events:")
    for ev in events:
        print(f"  - day {ev.start_day:>2}: {ev.name} ({ev.kind}) on '{ev.column}'")

    baseline_db = os.path.join(LOG_DIR, "baseline_transactions.db")
    agent_db = os.path.join(LOG_DIR, "agent_transactions.db")
    for p in (baseline_db, agent_db):
        if os.path.exists(p):
            os.remove(p)

    print("\nRunning BASELINE (rolling-stats + rule-based repair)...")
    baseline_log, baseline_detect_day, baseline_fix_day = run_system(
        "baseline", baseline_propose, dirty_df, reference_df, events, EXPECTED_SCHEMA, baseline_db)

    backend_note = "REAL Gemini API" if os.environ.get("GEMINI_API_KEY") else "mock heuristic backend (no GEMINI_API_KEY set)"
    print(f"\nRunning AGENTIC system ({backend_note})...")
    agent_log, agent_detect_day, agent_fix_day = run_system(
        "agent", agent_propose, dirty_df, reference_df, events, EXPECTED_SCHEMA, agent_db)

    print("\nRunning adversarial unsafe-patch test against the verifier...")
    adversarial_results = adversarial_unsafe_patch_test(reference_df, EXPECTED_SCHEMA, agent_db, n=20)

    all_rows = baseline_log + agent_log
    with open(os.path.join(LOG_DIR, "run_log.json"), "w") as f:
        json.dump(all_rows, f, indent=2, default=str)

    baseline_summary = summarize_system_log(all_rows, "baseline")
    agent_summary = summarize_system_log(all_rows, "agent")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for label, summary in [("BASELINE (rule-based)", baseline_summary), ("AGENTIC (LLM + verifier)", agent_summary)]:
        print(f"\n--- {label} ---")
        for k, v in summary.items():
            print(f"  {k:32s}: {v}")

    print(f"\n--- Verifier safety gate (adversarial unsafe-patch test) ---")
    print(f"  unsafe patches submitted        : {adversarial_results['n_unsafe_patches']}")
    print(f"  unsafe patches caught/rejected  : {adversarial_results['n_caught']}")
    print(f"  % unsafe patches caught         : {adversarial_results['pct_unsafe_caught']}%")

    print("\n--- Per-event outcome (start day -> detected day -> fixed day) ---")
    for ev in events:
        b_det = baseline_detect_day.get(ev.name, "never")
        b_fix = baseline_fix_day.get(ev.name, "never")
        a_det = agent_detect_day.get(ev.name, "never")
        a_fix = agent_fix_day.get(ev.name, "never")
        print(f"  {ev.name:32s} start={ev.start_day:>2} | baseline: det={b_det} fix={b_fix} | "
              f"agent: det={a_det} fix={a_fix}")

    results_table = pd.DataFrame([baseline_summary, agent_summary])
    results_table.to_csv(os.path.join(LOG_DIR, "comparison_summary.csv"), index=False)
    with open(os.path.join(LOG_DIR, "adversarial_results.json"), "w") as f:
        json.dump(adversarial_results, f, indent=2, default=str)

    print(f"\nFull run log:        {os.path.join(LOG_DIR, 'run_log.json')}")
    print(f"Comparison summary:  {os.path.join(LOG_DIR, 'comparison_summary.csv')}")
    print(f"Adversarial results: {os.path.join(LOG_DIR, 'adversarial_results.json')}")
    print(f"Transaction logs:    {baseline_db}, {agent_db}")


if __name__ == "__main__":
    main()
