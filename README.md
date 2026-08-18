# LLM-Agent Drift Diagnosis with DB-Consistency Verification Gate

A working prototype implementing the core loop from the proposal: detect drift →
LLM agent diagnoses + proposes a structured repair → a DB-transaction-style
verification layer gates the repair before it's committed → compare against a
rule-based baseline.

**Status: this is a functioning, runnable implementation of the core pipeline
(detection, injection, agent, verifier, evaluation harness) — roughly the
30–40% of the full proposal that constitutes an end-to-end MVP.** Section
"What's built vs. what's left" at the bottom is explicit about scope.

## Quick start

```bash
cd drift_agent_project
pip install -r requirements.txt

# Run with the mock LLM backend (no API key needed, fully offline/reproducible):
python -m eval.run_experiment

# Run with the real Gemini API instead of the mock heuristic:
# Add your GEMINI_API_KEY to a `.env` file in the root directory:
# echo "GEMINI_API_KEY=your_key_here" > .env
python -m eval.run_experiment
```

Outputs land in `logs/`:
- `run_log.json` — every detection + repair-attempt event, for both systems
- `comparison_summary.csv` — the metrics table (precision/recall/etc.)
- `adversarial_results.json` — the unsafe-patch safety-gate test
- `baseline_transactions.db` / `agent_transactions.db` — SQLite transaction logs
  (one row per commit/rollback decision, auditable)

## Architecture

```
data/generate_data.py     synthetic NYC-taxi-like trip generator + schema/business-rule registry
drift/injector.py         injects 4 KNOWN drift events (ground truth kept for eval only)
drift/detector.py         rolling KS-test + PSI + schema/null-rate checks (shared by both systems)
repair/rule_based.py      BASELINE: hard-coded rename table + static unit-conversion table
agent/llm_agent.py        AGENT: Gemini API call (or mock heuristic fallback) -> structured patch
agent/verifier.py         gate: dry-run patch -> schema check, business-rule check, null-increase
                           check, post-patch KS re-test -> commit (SQLite log) or rollback
eval/run_experiment.py    orchestrates both systems day-by-day + adversarial safety test
eval/metrics.py           precision/recall/F1, repair correctness, time-to-repair, etc.
```

### Why the agent emits structured patches, not free code
The proposal says the agent "proposes a SQL/pandas patch." Letting an LLM emit
arbitrary code that then gets `exec()`'d is both unsafe and impossible to gate
cleanly. Instead, the agent must choose one op from a small whitelist
(`rename_column`, `unit_convert`, `impute_missing`, `rescale_column`, `no_op`).
The verifier dry-runs that op and checks it — this is what makes "gated by a
DB-consistency verification layer" actually enforceable rather than a slogan.

### The verification / gating layer (the novel piece)
For every proposed patch:
1. **BEGIN** — apply the patch to a *copy* of the window, never the live frame.
2. **CHECK** — (a) resulting schema still has all expected columns, (b) values
   respect business rules (e.g. `fare_amount` in `[0, 1000]`), (c) null rate
   didn't increase beyond a tolerance, (d) a post-patch KS-test against the
   reference distribution shows drift actually improved (p ≥ 0.01) — a patch
   that changes something without resolving the measured drift is rejected.
3. **COMMIT** — only if all checks pass; logged to SQLite.
4. **ROLLBACK** — otherwise the live state is untouched and the rejection +
   reason is logged. This is the direct analogue of a DB transaction that
   fails a constraint check and rolls back rather than committing.

### Mock LLM backend
With no `GEMINI_API_KEY` set, `agent/llm_agent.py` uses a deterministic
heuristic stand-in instead of a real API call, so the whole pipeline runs
offline and reproducibly. It's deliberately built to be *more general* than
the rule-based baseline (it infers unit-conversion factors from observed
mean-ratios and infers rename targets from null-rate + distribution
similarity, rather than looking things up in a hard-coded table) so the
comparison is meaningful even without live API access. Swapping in the real
model is a one-line env var — no code changes needed.

## Drift scenarios injected (ground truth, hidden from both systems)

| Day | Type | Column | What happens |
|---|---|---|---|
| 15 | schema rename | `trip_distance` → `trip_dist_miles` | upstream renames the field |
| 25 | unit shift | `tip_amount` | values suddenly ×100 (cents mistaken for dollars) |
| 38 | missing field | `passenger_count` | 90% of values go null |
| 48 | distribution shift | `fare_amount` | mean shifts ×1.8 (e.g. surge-pricing policy change, not a unit issue) |

The baseline only "knows" about the rename and a hard-coded distance
unit-conversion factor — by design, it has no rule that covers `tip_amount`
or a genuine (non-unit) pricing shift, which is exactly the "static
thresholds don't generalize" gap the proposal targets.

## Results (mock-backend run, `python -m eval.run_experiment`)

| Metric | Baseline (rule-based) | Agentic (LLM + verifier) |
|---|---|---|
| Detection precision | 1.00 | 1.00 |
| Detection recall | 0.96 | 0.96 |
| Repairs committed | 2 | 4 |
| Repairs rejected by verifier | 70 | 23 |
| Repair correctness (of committed) | 0.50 | 0.75 |
| Events actually fixed | 1 of 4 (rename only) | 3 of 4 (rename, unit shift, distribution shift) |
| Unsafe/adversarial patches caught by verifier | — | 17 / 20 (85%) |

Detection precision/recall are identical between the two systems because both
share the same detector — the interesting difference is entirely in the
**repair** stage, which is the point of the proposal. The baseline detects
`tip_amount` and `fare_amount` drift fine, but has no static rule for either,
so it never fixes them. The agent generalizes to both. Neither system
resolves the `passenger_count` missing-field case (see limitations below) —
the verifier correctly rejects median imputation there because a post-patch
KS-test shows the imputed distribution (90% constant value) still
significantly diverges from the reference distribution. That's the gate doing
its job: it's refusing to "fix" schema-completeness while quietly destroying
statistical fidelity, rather than a bug to paper over.

Exact numbers will shift slightly run-to-run with the real Gemini backend
(the mock heuristic is a stand-in, not a claim about real-model performance)
and are meant to demonstrate the harness works end-to-end, not as a
publication-ready benchmark.

## What's built vs. what's left

**Built and runnable:**
- Synthetic time-series data generator with a registered schema + business rules
- Controlled drift injection (schema/unit/missing/distribution) with hidden ground truth
- Baseline drift detector (rolling KS-test + PSI + schema/null checks)
- Rule-based baseline repair (intentionally narrow, to demonstrate the gap)
- LLM agent (real Gemini API path + offline mock fallback) producing structured diagnoses
- Transactional verification/gating layer backed by SQLite (commit/rollback log)
- Adversarial safety test (% unsafe patches caught)
- Full evaluation harness computing precision/recall, repair correctness,
  time-to-repair, false-positive rate, and the safety-gate catch rate

**Left for future work (not built here):**
- Real NYC TLC / Weather Sentiment dataset ingestion (synthetic data used instead,
  for reproducibility and known ground truth — swapping in a real CSV loader is
  straightforward given the schema/business-rule registry pattern already in place)
- Postgres backend (SQLite used for the transaction log; same schema would port over)
- SQL-patch generation (only pandas-style structured ops are supported currently)
- Multi-agent or iterative-repair loops (agent gets one shot per detected finding)
- A UI/dashboard over the transaction log
- Statistical significance testing across many random seeds (current run is a
  single seeded demonstration, not averaged over repeated trials)
