"""CLI for the agent evaluation platform.

Commands:
  list-suites                 list the golden suites and their tasks
  list-agents                 list the deterministic scripted agents
  run --suite S --agent A     run a suite, score it, gate it vs baseline
  list-runs                   list runs in the registry
  promote --run-id R          mark a run as the baseline for its suite+agent
  report --run-id R           print the Markdown report for a run
  compare --a R1 --b R2       side-by-side aggregate comparison
"""
from __future__ import annotations

import argparse
import sys

from ageval import report as report_mod
from ageval.registry import EvalRegistry
from ageval.runner import AGENTS, EvalRunner, get_agent
from ageval.suites import SUITES, get_suite

DEFAULT_REGISTRY = "./eval-runs"


def _registry(path: str) -> EvalRegistry:
    return EvalRegistry(path)


def cmd_list_suites(_args: argparse.Namespace) -> int:
    for suite in SUITES.values():
        print(f"{suite.suite_id} (v{suite.version}) — {suite.description}")
        for task in suite.tasks:
            print(f"  {task.task_id}: {task.prompt[:80]}")
    return 0


def cmd_list_agents(_args: argparse.Namespace) -> int:
    for agent_id, agent in AGENTS.items():
        print(f"{agent_id} (v{agent.agent_version})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    suite = get_suite(args.suite)
    agent = get_agent(args.agent)
    registry = _registry(args.registry)
    runner = EvalRunner(registry)
    run = runner.run_suite(suite, agent)
    print(report_mod.render_markdown(run))
    print(f"Run saved: {run.run_id}")
    return 0 if (run.gates is None or run.gates.verdict == "PASS") else 1


def cmd_list_runs(args: argparse.Namespace) -> int:
    for run in _registry(args.registry).list_runs():
        verdict = run.gates.verdict if run.gates else "not gated"
        agg = run.aggregate
        success = f"{agg.task_success:.3f}" if agg else "—"
        print(f"{run.run_id} | {run.suite_id} v{run.suite_version} | "
              f"{run.agent_id} | success {success} | gates {verdict}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    run = _registry(args.registry).promote(args.run_id)
    print(f"Promoted {run.run_id} as baseline for "
          f"{run.suite_id} v{run.suite_version} / {run.agent_id}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run = _registry(args.registry).get_run(args.run_id)
    print(report_mod.render_markdown(run))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    registry = _registry(args.registry)
    a = registry.get_run(args.a)
    b = registry.get_run(args.b)
    if a.aggregate is None or b.aggregate is None:
        print("Both runs need aggregates to compare.", file=sys.stderr)
        return 2
    print(f"{'metric':<24}{a.run_id[:28]:<30}{b.run_id[:28]}")
    for metric in ("task_success", "tool_accuracy", "trajectory_efficiency",
                   "groundedness", "refusal", "cost_usd", "latency_ms_p95"):
        va = getattr(a.aggregate, metric)
        vb = getattr(b.aggregate, metric)
        print(f"{metric:<24}{va:<30.4f}{vb:.4f}  (Δ {vb - va:+.4f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ageval", description="Evaluation harness for agentic systems.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-suites").set_defaults(func=cmd_list_suites)
    sub.add_parser("list-agents").set_defaults(func=cmd_list_agents)

    p_run = sub.add_parser("run")
    p_run.add_argument("--suite", default="support-triage")
    p_run.add_argument("--agent", default="scripted-good")
    p_run.add_argument("--registry", default=DEFAULT_REGISTRY)
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list-runs")
    p_list.add_argument("--registry", default=DEFAULT_REGISTRY)
    p_list.set_defaults(func=cmd_list_runs)

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("--run-id", required=True)
    p_promote.add_argument("--registry", default=DEFAULT_REGISTRY)
    p_promote.set_defaults(func=cmd_promote)

    p_rep = sub.add_parser("report")
    p_rep.add_argument("--run-id", required=True)
    p_rep.add_argument("--registry", default=DEFAULT_REGISTRY)
    p_rep.set_defaults(func=cmd_report)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--a", required=True)
    p_cmp.add_argument("--b", required=True)
    p_cmp.add_argument("--registry", default=DEFAULT_REGISTRY)
    p_cmp.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
