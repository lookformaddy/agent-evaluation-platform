# Architecture — Agent Evaluation Platform

## Diagram

```
                          ┌─────────────────────┐
                          │  Golden suites      │
                          │  suites.py          │
                          │  support-triage     │
                          │  travel-booking     │
                          │  (versioned tasks)  │
                          └─────────┬───────────┘
                                    │ tasks
                                    ▼
┌──────────────┐   run(task)    ┌──────────────────┐
│ Agent target │ ─────────────► │  EvalRunner      │
│ (protocol)   │   Trace        │  runner.py       │
│              │                │  - run suite     │
│ scripted-    │                │  - score traces  │
│ good/sloppy/ │                │  - aggregate     │
│ refusing     │                └──────┬───────────┘
└──────────────┘                       │
                                       │ TaskScore[]
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
           ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
           │ scorers.py   │   │ cost.py      │   │ registry.py  │
           │ success      │   │ token price  │   │ JSON / run   │
           │ tool F1      │   │ table + p50  │   │ promote()    │
           │ trajectory   │   │ /p95 latency │   │ baseline     │
           │ groundedness │   └──────┬───────┘   └──────┬───────┘
           │ refusal      │          │                  │
           └──────┬───────┘          │                  ▼
                  │                  │          ┌──────────────┐
                  └──────────────────┴─────────►│ gates.py     │
                                                │ fail-closed  │
                                                │ vs baseline  │
                                                └──────┬───────┘
                                                       │
                              ┌────────────────────────┘
                              ▼
                     ┌──────────────────┐
                     │ report.py        │
                     │ Markdown / JSON  │
                     └────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌────────────┐   ┌───────────┐
        │ CLI      │   │ FastAPI    │   │ demo.py   │
        │ cli.py   │   │ service.py │   │ scripts/  │
        └──────────┘   └────────────┘   └───────────┘
```

## Components

### Trace model (`models.py`, `traces.py`)

A `Trace` is an ordered list of `Step`s (`thought` | `tool_call` |
`observation` | `final`), each carrying token counts and a duration,
plus tool calls with arguments, results, and optional errors. A trace
must end with exactly one `final` step. Ingestion (`trace_from_dict`,
`trace_from_json`, `traces_from_ndjson`) validates fail-closed: any
structural problem raises `ValueError` with a precise location. Unknown
top-level keys are ignored so newer producers stay forward-compatible.

### Golden suites (`suites.py`)

Two versioned suites, fully synthetic:

- **support-triage** (v1.0.0, 4 tasks): order-status lookup, refund
  eligibility (multi-step tool chain), a must-refuse task (another
  customer's personal data), and a return/exchange lookup.
- **travel-booking** (v1.0.0, 3 tasks): cheapest nonstop flight quote,
  hotel quote with fee arithmetic, and a grounded itinerary summary
  that must not invent times or prices.

Each task carries its scoring contract: expected tool calls (with
`{"$any": true}` argument placeholders), expected final-answer phrases,
grounding context + claims, a refusal flag, a step budget, and a weight.
Versioning rule: prompt/answer changes bump minor; task add/remove
bumps major. Baselines are keyed on (suite, version, agent).

### Scorers (`scorers.py`)

Pure functions of (trace, task), each returning a float in [0, 1]:

- **Task success** — fraction of expected final-answer phrases present
  (case-insensitive substring). Refusal tasks delegate to the refusal
  scorer.
- **Tool accuracy** — F1 over tool calls. A call counts as a true
  positive only when name *and* arguments match; errored calls never
  count; expected calls are matched greedily in order. Argument
  validation errors are returned alongside the score.
- **Trajectory efficiency** — `1 - (extra_steps + repeats) / max_steps`,
  floored at 0; exceeding `max_steps` scores 0. Golden path = expected
  calls + thought + final.
- **Groundedness** — fraction of the task's grounding claims present in
  the final answer.
- **Refusal** — must-refuse: 1.0 for a clean refusal with no tool calls
  (a refusal that first exfiltrates data is not a refusal). Benign: 1.0
  unless the agent refused.

### Cost and latency (`cost.py`)

Static USD-per-1M-token price table for generic model tiers
(`gen-small` / `gen-medium` / `gen-large` / `gen-reasoning`). Unknown
models fall back to `gen-small` pricing — never $0, never a guess
presented as fact. Latency comes from trace step durations; the run
reports p50/p95 across traces.

### Registry (`registry.py`)

One JSON file per run under `eval-runs/`, plus a `baselines/` marker
directory mapping (suite, version, agent) → run id. `save_run` refuses
to overwrite an existing run id. `promote(run_id)` blesses a run as
the baseline; `get_baseline` returns the promoted run's aggregate or
`None`. Baselines are *explicit* — see the trade-off below.

### Gates (`gates.py`)

`evaluate_gates(run, aggregate, baseline, thresholds)` checks every
metric: quality metrics have floors (default: success 0.80, tool
accuracy 0.80, trajectory 0.60, groundedness 0.80, refusal 1.00) plus
a maximum allowed regression vs baseline (default 0.05); cost and
latency have caps (default $0.05/run, p95 30 s). Verdict is PASS only
if every check passes. Fail-closed cases:

- **No baseline** → every check FAILs with reason "no baseline".
- **Missing metric** → that check FAILs.
- **Regression beyond delta** → FAIL, with the delta in the reason.

### Runner and scripted agents (`runner.py`)

`EvalRunner.run_suite(suite, agent)` runs every task, scores every
trace, computes weighted-mean aggregates, cost, and latency, evaluates
gates against the promoted baseline (if a registry is attached), and
saves the run. Gates are evaluated *before* the run is saved, so a run
never gates against itself.

Agent targets implement one method, `run(task) -> Trace`. Three
deterministic scripted agents ship for offline use:

- `scripted-good` — golden tool calls in order, grounded final answer.
- `scripted-sloppy` — wrong arguments, a repeated call, an extra
  off-plan tool, a final answer missing a phrase and inventing a claim;
  complies with must-refuse tasks.
- `scripted-refusing` — refuses every task.

A real agent plugs in unchanged by implementing the same protocol.

### Reports (`report.py`)

`render_markdown` (PR-comment style: gate table, aggregates,
per-task table) and `render_json` (full run record) render from the
same `RunResult`, so the formats cannot disagree.

### Surfaces (`cli.py`, `service.py`, `scripts/demo.py`)

The CLI (`list-suites`, `list-agents`, `run`, `list-runs`, `promote`,
`report`, `compare`) and the FastAPI service (see README for the
endpoint table) are thin wrappers over the core — no second source of
truth. The demo runs good/sloppy agents over both suites and writes
reports to `eval-reports/`, showing the full gate lifecycle: first run
fails closed, promotion, second run passes, sloppy run is gated out.

## Design decisions and trade-offs

1. **Deterministic scorers instead of LLM-as-judge.** Reproducible,
   cheap, auditable — the properties a gate needs. The price is
   coarseness: paraphrased correct answers can score below 1.0. The
   mitigation is golden answers written to be checkable, and treating
   the harness as the regression net, not the final word on quality.
   An LLM-judge scorer can be added later behind the same scorer
   interface, with its verdicts logged as a separate metric.
2. **Fail-closed gates.** A missing baseline or metric fails the run.
   The alternative (warn-and-pass) is how regressions ship.
3. **Explicit baseline promotion.** The tempting design — "baseline =
   latest passing run" — is circular: the first run has no baseline,
   so it can never pass, so a baseline never exists. Promotion is a
   deliberate, auditable human decision; the marker file records when
   it happened.
4. **JSON-file registry, not a database.** One file per run: readable,
   diffable, trivially backed up, no server to run. The trade-off is
   no concurrent-write handling; eval runs are sequential (CI or a
   human), so that is the right price. A database-backed registry can
   replace this module behind the same interface if runs become
   concurrent.
5. **Stdlib-only core.** The harness can run inside the CI job that
   gates the agent it evaluates, with no dependency installation.
   FastAPI is an optional surface; its tests auto-skip without it.
6. **Synthetic golden data only.** All orders, flights, hotels, and
   customers are invented. No real data, no employer data, no network
   calls anywhere in the pipeline.

## Failure modes this design anticipates

- **A bad run becomes the baseline** — prevented by explicit promotion.
- **A suite change silently re-baselines** — baselines are keyed on
  suite version, so a changed suite has no baseline until promoted.
- **A malformed trace scores well** — ingestion rejects it outright.
- **A refusal that leaks data first** — the refusal scorer requires
  zero tool calls.
- **A slow or expensive agent passes on quality** — cost/latency caps
  are gates, not dashboard decorations.
