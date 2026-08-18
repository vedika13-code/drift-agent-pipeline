"""Metric computations for the baseline vs. agentic comparison."""
from typing import List, Dict, Any
import pandas as pd


def precision_recall(true_positive: int, false_positive: int, false_negative: int):
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else float("nan")
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0 else float("nan"))
    return precision, recall, f1


def summarize_system_log(rows: List[Dict[str, Any]], system_name: str) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    df = df[df["system"] == system_name]
    if df.empty:
        return {"system": system_name, "note": "no rows"}

    detections = df[df["record_type"] == "detection"]
    tp = int((detections["is_true_positive"] == True).sum())
    fp = int((detections["is_true_positive"] == False).sum())
    fn_rows = df[df["record_type"] == "missed_event"]
    fn = len(fn_rows)
    precision, recall, f1 = precision_recall(tp, fp, fn)

    repairs = df[df["record_type"] == "repair_attempt"]
    n_repair_attempts = len(repairs)
    n_committed = int((repairs["accepted"] == True).sum())
    n_rejected = int((repairs["accepted"] == False).sum())
    n_correct = int((repairs["repair_correct"] == True).sum())
    repair_correctness = n_correct / n_committed if n_committed else float("nan")

    ttr = df[(df["record_type"] == "time_to_repair")]
    avg_ttr_days = ttr["ttr_days"].mean() if not ttr.empty else float("nan")

    avg_latency = repairs["latency_sec"].mean() if not repairs.empty else float("nan")

    return {
        "system": system_name,
        "detection_precision": round(precision, 3) if precision == precision else None,
        "detection_recall": round(recall, 3) if recall == recall else None,
        "detection_f1": round(f1, 3) if f1 == f1 else None,
        "false_positive_windows": fp,
        "true_positive_windows": tp,
        "missed_events": fn,
        "repair_attempts": n_repair_attempts,
        "repairs_committed": n_committed,
        "repairs_rejected_by_verifier": n_rejected,
        "repair_correctness_rate": round(repair_correctness, 3) if repair_correctness == repair_correctness else None,
        "avg_time_to_repair_days": round(avg_ttr_days, 2) if avg_ttr_days == avg_ttr_days else None,
        "avg_proposal_latency_sec": round(avg_latency, 5) if avg_latency == avg_latency else None,
    }
