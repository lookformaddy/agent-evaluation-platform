# Agent Evaluation Platform

Evaluation harness for agentic systems: versioned golden task suites,
deterministic scorers, per-run cost/latency accounting, fail-closed
regression gates against a promoted baseline, and a versioned eval-run
registry — with a CLI, a FastAPI service, and a demo that runs end to
end with no network and no API keys.

## The enterprise problem

Shipping an agent without an eval harness is shipping blind. A prompt
tweak, a tool change, or a model swap can silently regress task success,
start calling the wrong tools, hallucinate prices, or comply with a
request it should refuse — and the first person to notice is a customer.
This platform answers the question every team needs before promoting an
agent change: **"Is this version at least as good as the last one we
shipped, on the tasks we care about, at a cost we can afford?"**

It is built for the loop that actually runs in production: golden tasks
checked into version control, scored deterministically, gated against a
blessed baseline, and reported in a form a reviewer can read in a PR.

## What it evaluates

| Scorer | What it measures |
|---|---|
| **Task success** | Fraction of expected final-answer phrases present |
| **Tool accuracy** | F1 over tool calls, with argument validation (`{"$any": true}` placeholders allowed) |
| **Trajectory efficiency** | Penalty for steps beyond the golden path and repeated identical calls |
| **Groundedness** | Fraction of grounding claims supported by the final answer |
| **Refusal** | Clean refusal of must-refuse tasks; no refusal of benign ones |

Plus per-run **cost** (USD, from a static per-model token price table)
and **latency** (p50/p95 across traces), and **regression gates** that
compare every aggregate against the promoted baseline.

## Architecture overview

```
golden suites (suites.py) ──► EvalRunner ──► scorers (pure functions)
        ▲                          │                 │
        │                          ▼                 ▼
  versioned tasks            TaskScore[]        cost / latency
        │                          │                 │
        └──────────► registry (JSON files) ◄────────┘
                          │  promote() = bless a baseline
                          ▼
                    gates (fail-closed) ──► Markdown / JSON report
                          │
              CLI (ageval) · FastAPI service · demo.py
```

- **Core is pure stdlib.** Scorers, cost accounting, gates, and the
  registry have no dependencies, so the harness can run anywhere —
  including inside the CI job that gates the agent it evaluates.
- **Traces are validated fail-closed** (`traces.py`): a malformed trace
  (unknown step kind, no final step, negative tokens) raises instead of
  scoring well by accident.
- **Agents are one-method targets** (`run(task) -> Trace`). Three
  deterministic scripted agents ship with the repo so the whole
  pipeline runs offline: `scripted-good` (follows the golden plan),
  `scripted-sloppy` (wrong args, repeats, extra tool, invented claim),
  `scripted-refusing` (refuses everything).
- **Baselines are promoted explicitly** (`registry.promote`). There is
  no "latest passing run" auto-baseline: the first run for a suite has
  no baseline, so its gates fail closed, and it can only anchor future
  gates once a human has reviewed and promoted it.
- **Suites are versioned.** Changing a task's prompt/expected answer
  bumps the minor version; adding/removing a task bumps the major
  version. Baselines are keyed on (suite, version, agent), so a suite
  change never silently compares against a stale baseline.

See [docs/architecture.md](docs/architecture.md) for the full design,
including the diagram and the trade-offs.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# List the golden suites and scripted agents
PYTHONPATH=src python3 -m ageval.cli list-suites
PYTHONPATH=src python3 -m ageval.cli list-agents

# First run: gates FAIL (no baseline yet — fail-closed by design)
PYTHONPATH=src python3 -m ageval.cli run --suite support-triage --agent scripted-good

# Review the report, then promote the run as the baseline
PYTHONPATH=src python3 -m ageval.cli list-runs
PYTHONPATH=src python3 -m ageval.cli promote --run-id <run-id>

# Second run passes; the sloppy agent is gated out
PYTHONPATH=src python3 -m ageval.cli run --suite support-triage --agent scripted-good
PYTHONPATH=src python3 -m ageval.cli run --suite support-triage --agent scripted-sloppy

# Full end-to-end demo (both suites, reports in ./eval-reports/)
make demo

# Tests (plain pytest passes from the project root, with or without FastAPI)
make test
```

## API service

```bash
uvicorn ageval.service:app --host 0.0.0.0 --port 8000
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + suite list |
| GET | `/suites` | Golden suites and their tasks |
| POST | `/runs` | Run a suite against an agent (201) |
| GET | `/runs` | List runs in the registry |
| GET | `/runs/{id}` | One run record |
| GET | `/runs/{id}/report?format=markdown\|json` | Run report |
| GET | `/runs/{id}/gates` | Gate verdict for a run |

## Key design decisions and trade-offs

1. **Deterministic scorers, no LLM-as-judge.** Every scorer is a pure
   function of (trace, task). LLM judges are powerful but
   non-deterministic, expensive, and hard to audit; substring/F1-based
   scorers are explainable and reproducible, which is what a gate needs.
   The trade-off is coarseness — a paraphrased correct answer can score
   below 1.0 — so golden answers are written to be checkable, and the
   harness is the regression net, not the final word on quality.
2. **Fail-closed gates.** A missing baseline, a missing metric, or a
   regression beyond the allowed delta all fail the run. The alternative
   (warn and pass) is how regressions ship.
3. **Explicit baseline promotion.** Auto-baselines from "the latest
   passing run" are circular — the first run can never pass, so no
   baseline ever exists. Promotion is a deliberate, auditable act.
4. **JSON-file registry, not a database.** One file per run, human
   readable, diffable, and trivially backed up. The trade-off is no
   concurrent-write handling; for a harness whose runs are sequential
   (CI or a human), that is the right price.
5. **Synthetic golden data only.** All orders, flights, hotels, and
   customers are invented. No real data, no employer data, no network.
6. **FastAPI is optional.** The core runs on stdlib alone; the service
   and its tests auto-skip when FastAPI is absent, so `make test` works
   on a bare Python install and the full suite runs in CI.

## What this demonstrates

Golden-task design and suite versioning · deterministic, auditable
scoring (success, tool-use F1, trajectory, groundedness, refusal) ·
fail-closed regression gating · explicit baseline promotion · per-run
cost/latency accounting · versioned eval-run registry · trace
validation · CLI + FastAPI service · CI-ready pytest suite.

## Project layout

```
src/ageval/
  models.py      dataclasses: traces, tasks, scores, runs, gates
  traces.py      fail-closed trace ingestion (JSON / NDJSON)
  suites.py      versioned golden suites (support-triage, travel-booking)
  scorers.py     deterministic scorers
  cost.py        token pricing + latency percentiles
  gates.py       fail-closed regression gates vs baseline
  registry.py    JSON-file run registry + baseline promotion
  runner.py      EvalRunner + three scripted agent targets
  report.py      Markdown / JSON reports
  cli.py         `ageval` CLI
  service.py     FastAPI service
tests/           pytest suite (API tests auto-skip without FastAPI)
scripts/demo.py  end-to-end demo over both suites
docs/            architecture + CI definition (see below)
```

## CI

`docs/github-actions-ci.yml` is the CI definition. It ships under
`docs/` (not `.github/workflows/`) because the GitHub App used to push
this repo lacks the Workflows permission; activate it by moving it to
`.github/workflows/ci.yml`.
