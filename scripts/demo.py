"""End-to-end demo: run the good and sloppy agents over both golden
suites, show the fail-closed baseline behaviour, and write reports.

Expected outcome:
  1. First good run on support-triage -> gates FAIL ("no baseline").
  2. That run is promoted as the baseline; second good run -> gates PASS.
  3. Sloppy run -> gates FAIL (regression vs baseline).
  4. Travel suite good run -> gates FAIL (no baseline for that suite).
Reports are written to ./eval-reports/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ageval import report as report_mod  # noqa: E402
from ageval.registry import EvalRegistry  # noqa: E402
from ageval.runner import EvalRunner, get_agent  # noqa: E402
from ageval.suites import get_suite  # noqa: E402


def _summarize(label: str, run) -> None:
    a = run.aggregate
    verdict = run.gates.verdict if run.gates else "not gated"
    print(f"\n{label}")
    print(f"  run_id: {run.run_id}")
    print(f"  success {a.task_success:.3f} | tool {a.tool_accuracy:.3f} | "
          f"trajectory {a.trajectory_efficiency:.3f} | "
          f"grounded {a.groundedness:.3f} | refusal {a.refusal:.3f}")
    print(f"  tasks passed {a.tasks_passed}/{a.tasks_run} | "
          f"cost ${a.cost_usd:.6f} | p95 {a.latency_ms_p95:.0f} ms")
    print(f"  gates: {verdict}")


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "eval-reports"
    out_dir.mkdir(exist_ok=True)
    registry = EvalRegistry(Path(__file__).resolve().parents[1] / "eval-runs")
    runner = EvalRunner(registry)

    runs = []
    good = get_agent("scripted-good")
    sloppy = get_agent("scripted-sloppy")

    support = get_suite("support-triage")
    travel = get_suite("travel-booking")

    r1 = runner.run_suite(support, good)
    _summarize("1. Good agent, support-triage (first run — no baseline yet)", r1)
    runs.append(("support-good-1", r1))

    # A first run can never pass gates (fail-closed); a human reviews it
    # and promotes it as the baseline before later runs are gated.
    registry.promote(r1.run_id)
    print(f"\nPromoted {r1.run_id} as the baseline for support-triage v1.0.0.")

    r2 = runner.run_suite(support, good)
    _summarize("2. Good agent, support-triage (second run — baseline exists)", r2)
    runs.append(("support-good-2", r2))

    r3 = runner.run_suite(support, sloppy)
    _summarize("3. Sloppy agent, support-triage (regression vs baseline)", r3)
    runs.append(("support-sloppy", r3))

    r4 = runner.run_suite(travel, good)
    _summarize("4. Good agent, travel-booking (first run — no baseline yet)", r4)
    runs.append(("travel-good-1", r4))

    for name, run in runs:
        (out_dir / f"{name}.md").write_text(report_mod.render_markdown(run),
                                            encoding="utf-8")
        (out_dir / f"{name}.json").write_text(report_mod.render_json(run),
                                              encoding="utf-8")
    print(f"\nReports written to {out_dir} ({len(runs)} runs).")
    print("Demo complete: good agent passes once baselined; sloppy agent "
          "is gated out; first runs fail closed on missing baselines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
