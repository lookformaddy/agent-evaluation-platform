"""Markdown and JSON reports for eval runs.

Reports are the human-facing surface of the harness: a Markdown summary
suitable for a PR comment or a run review, and a JSON dump of the full
run record for downstream tooling. Both render from the same
:class:`RunResult`, so the two formats can never disagree.
"""
from __future__ import annotations

import json

from ageval.models import RunResult


def render_markdown(run: RunResult) -> str:
    lines = [
        f"# Eval Run Report — `{run.run_id}`",
        "",
        f"- **Suite:** {run.suite_id} (v{run.suite_version})",
        f"- **Agent:** {run.agent_id} (v{run.agent_version})",
        f"- **Created:** {run.created_at}",
        "",
    ]
    if run.gates is not None:
        lines.append(f"## Gate Verdict: {run.gates.verdict}")
        lines.append("")
        lines.append("| Metric | Threshold | Actual | Baseline | Result |")
        lines.append("|---|---|---|---|---|")
        for r in run.gates.results:
            actual = f"{r.actual:.4f}" if r.actual is not None else "—"
            base = f"{r.baseline:.4f}" if r.baseline is not None else "—"
            mark = "PASS" if r.passed else "FAIL"
            lines.append(
                f"| {r.metric} | {r.threshold} | {actual} | {base} | "
                f"{mark} — {r.reason} |"
            )
        lines.append("")
    if run.aggregate is not None:
        a = run.aggregate
        lines.append("## Aggregate Scores")
        lines.append("")
        lines.append(f"- Task success: **{a.task_success:.3f}**")
        lines.append(f"- Tool accuracy: **{a.tool_accuracy:.3f}**")
        lines.append(f"- Trajectory efficiency: **{a.trajectory_efficiency:.3f}**")
        lines.append(f"- Groundedness: **{a.groundedness:.3f}**")
        lines.append(f"- Refusal: **{a.refusal:.3f}**")
        lines.append(f"- Tasks passed: {a.tasks_passed}/{a.tasks_run}")
        lines.append(f"- Cost: ${a.cost_usd:.6f} · p50 latency {a.latency_ms_p50:.0f} ms · "
                     f"p95 latency {a.latency_ms_p95:.0f} ms")
        lines.append("")
    if run.scores:
        lines.append("## Per-Task Scores")
        lines.append("")
        lines.append("| Task | Success | Tool | Trajectory | Grounded | Refusal | Steps |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in run.scores:
            lines.append(
                f"| {s.task_id} | {s.task_success:.2f} | {s.tool_accuracy:.2f} | "
                f"{s.trajectory_efficiency:.2f} | {s.groundedness:.2f} | "
                f"{s.refusal:.2f} | {s.steps_used} |"
            )
            for err in s.tool_call_errors:
                lines.append(f"| | | | | | | ⚠ {err} |")
        lines.append("")
    return "\n".join(lines)


def render_json(run: RunResult) -> str:
    return json.dumps(run.to_dict(), indent=2, sort_keys=True)
