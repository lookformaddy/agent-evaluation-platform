"""Eval runner and deterministic scripted agent targets.

The harness ships three scripted targets so the whole pipeline runs
end to end with no network, no API keys, and no model calls:

- :class:`ScriptedGoodAgent` — follows the golden plan: the expected
  tool calls, in order, then a final answer drawn from the task's
  grounding context. Scores near 1.0 everywhere.
- :class:`ScriptedSloppyAgent` — the realistic failure mode: wrong
  arguments, a repeated call, an extra off-plan tool, a final answer
  missing a required phrase and containing an invented claim.
- :class:`ScriptedRefusingAgent` — refuses every task (correct only on
  the must-refuse tasks, wrong everywhere else).

A real agent target implements the same one-method protocol,
``run(task) -> Trace``, and plugs into :class:`EvalRunner` unchanged.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Protocol

from ageval import cost as cost_mod
from ageval.gates import Thresholds, evaluate_gates
from ageval.models import (
    CostSummary,
    GoldenTask,
    RunAggregate,
    RunResult,
    Step,
    ToolCall,
    Trace,
)
from ageval.registry import EvalRegistry
from ageval.scorers import score_trace
from ageval.suites import GoldenSuite


class AgentTarget(Protocol):
    agent_id: str
    agent_version: str

    def run(self, task: GoldenTask) -> Trace: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step(kind: str, text: str = "", calls: list[ToolCall] | None = None,
          in_tokens: int = 100, out_tokens: int = 50,
          duration_ms: float = 120.0) -> Step:
    return Step(kind=kind, text=text, tool_calls=calls or [],
                input_tokens=in_tokens, output_tokens=out_tokens,
                duration_ms=duration_ms)


def _final_text(task: GoldenTask) -> str:
    """Compose a grounded final answer from the task's own context."""
    parts = []
    if task.grounding_context:
        parts.append(task.grounding_context)
    for phrase in task.expected_final_contains:
        if phrase.lower() not in task.grounding_context.lower():
            parts.append(phrase)
    return " ".join(parts)


class ScriptedGoodAgent:
    """Deterministic correct agent: golden tool calls, grounded answer."""

    agent_id = "scripted-good"
    agent_version = "1.0.0"

    def run(self, task: GoldenTask) -> Trace:
        steps = [_step("thought", f"Plan: answer {task.task_id} using the available tools.")]
        if task.refusal_expected:
            steps.append(_step(
                "final",
                "I can't help with that — I can't share another customer's "
                "personal details. I can help with your own order instead.",
            ))
        else:
            for want in task.expected_tool_calls:
                steps.append(_step(
                    "tool_call",
                    f"Calling {want['name']}.",
                    calls=[ToolCall(
                        name=want["name"],
                        arguments=dict(want.get("arguments", {})),
                        result=f"{want['name']} returned a result.",
                        duration_ms=80.0,
                    )],
                    in_tokens=150, out_tokens=80, duration_ms=200.0,
                ))
            steps.append(_step("final", _final_text(task), in_tokens=200,
                               out_tokens=120, duration_ms=150.0))
        return Trace(
            trace_id=f"{task.task_id}-{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            steps=steps,
            model="gen-small",
            started_at=_now(),
            ended_at=_now(),
        )


class ScriptedSloppyAgent:
    """Deterministic flawed agent: wrong args, repeats, extra tool, invented claim."""

    agent_id = "scripted-sloppy"
    agent_version = "1.0.0"

    def run(self, task: GoldenTask) -> Trace:
        steps = [_step("thought", "I'll guess my way through this task.")]
        if task.refusal_expected:
            # Complies with a must-refuse request: the worst failure mode.
            steps.append(_step(
                "tool_call",
                "Looking up the other customer's record.",
                calls=[ToolCall(
                    name="lookup_order",
                    arguments={"order_id": "ORD-3099"},
                    result="Record for Dana Example returned.",
                    duration_ms=80.0,
                )],
                in_tokens=150, out_tokens=80, duration_ms=200.0,
            ))
            steps.append(_step(
                "final",
                "Here is Dana Example's home address and phone number.",
                in_tokens=200, out_tokens=120, duration_ms=150.0,
            ))
        else:
            if task.expected_tool_calls:
                first = task.expected_tool_calls[0]
                wrong_args = dict(first.get("arguments", {}))
                for key in wrong_args:
                    wrong_args[key] = "WRONG-VALUE"
                # Wrong argument, then the same wrong call repeated (loop),
                # then an off-plan extra tool.
                for _ in range(2):
                    steps.append(_step(
                        "tool_call",
                        f"Calling {first['name']} with a guessed argument.",
                        calls=[ToolCall(
                            name=first["name"], arguments=wrong_args,
                            result="No matching record found.",
                            duration_ms=80.0,
                        )],
                        in_tokens=150, out_tokens=80, duration_ms=200.0,
                    ))
                steps.append(_step(
                    "tool_call",
                    "Trying an unrelated tool.",
                    calls=[ToolCall(
                        name="search_web", arguments={"query": "help"},
                        result="Generic web results.",
                        duration_ms=80.0,
                    )],
                    in_tokens=150, out_tokens=80, duration_ms=200.0,
                ))
            # Final answer: drops one required phrase and invents a claim.
            phrases = [p for p in task.expected_final_contains]
            kept = phrases[:-1] if len(phrases) > 1 else phrases
            text = " ".join(kept) + " It was delivered yesterday by express courier."
            steps.append(_step("final", text, in_tokens=200, out_tokens=120,
                               duration_ms=150.0))
        return Trace(
            trace_id=f"{task.task_id}-{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            steps=steps,
            model="gen-small",
            started_at=_now(),
            ended_at=_now(),
        )


class ScriptedRefusingAgent:
    """Deterministic over-refuser: refuses every task."""

    agent_id = "scripted-refusing"
    agent_version = "1.0.0"

    def run(self, task: GoldenTask) -> Trace:
        steps = [
            _step("thought", "This might be unsafe; I will refuse."),
            _step("final", "I can't help with that request.", in_tokens=200,
                  out_tokens=60, duration_ms=150.0),
        ]
        return Trace(
            trace_id=f"{task.task_id}-{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            steps=steps,
            model="gen-small",
            started_at=_now(),
            ended_at=_now(),
        )


AGENTS: dict[str, AgentTarget] = {
    "scripted-good": ScriptedGoodAgent(),
    "scripted-sloppy": ScriptedSloppyAgent(),
    "scripted-refusing": ScriptedRefusingAgent(),
}


def get_agent(agent_id: str) -> AgentTarget:
    try:
        return AGENTS[agent_id]
    except KeyError:
        raise KeyError(
            f"unknown agent {agent_id!r}; available: {sorted(AGENTS)}"
        ) from None


class EvalRunner:
    """Runs a suite against an agent target and scores every trace."""

    def __init__(self, registry: EvalRegistry | None = None) -> None:
        self.registry = registry

    def run_suite(
        self,
        suite: GoldenSuite,
        agent: AgentTarget,
        thresholds: Thresholds | None = None,
    ) -> RunResult:
        run_id = (
            f"{suite.suite_id}-{agent.agent_id}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-"
            f"{uuid.uuid4().hex[:6]}"
        )
        scores = []
        costs: list[CostSummary] = []
        latencies: list[float] = []
        for task in suite.tasks:
            trace = agent.run(task)
            scores.append(score_trace(trace, task))
            per_trace = cost_mod.cost_for_trace(trace)
            costs.append(per_trace)
            latencies.append(per_trace.latency_ms)
        aggregate = self._aggregate(scores, costs, latencies,
                                    [t.weight for t in suite.tasks])
        total_cost = cost_mod.aggregate_cost(costs)
        baseline = None
        if self.registry is not None:
            baseline = self.registry.get_baseline(
                suite.suite_id, suite.version, agent.agent_id)
        gates = None
        if self.registry is not None:
            gates = evaluate_gates(run_id, aggregate, baseline, thresholds)
        run = RunResult(
            run_id=run_id,
            suite_id=suite.suite_id,
            suite_version=suite.version,
            agent_id=agent.agent_id,
            agent_version=agent.agent_version,
            scores=scores,
            aggregate=aggregate,
            cost=total_cost,
            gates=gates,
            created_at=_now(),
        )
        if self.registry is not None:
            self.registry.save_run(run)
        return run

    @staticmethod
    def _aggregate(
        scores: list,
        costs: list[CostSummary],
        latencies: list[float],
        weights: list[float],
    ) -> RunAggregate:
        total_w = sum(weights) or 1.0

        def wmean(attr: str) -> float:
            return sum(getattr(s, attr) * w for s, w in zip(scores, weights)) / total_w

        p50, p95 = cost_mod.latency_percentiles(latencies)
        passed = sum(
            1 for s in scores if s.task_success >= 0.999 and s.refusal >= 0.999
        )
        return RunAggregate(
            task_success=round(wmean("task_success"), 4),
            tool_accuracy=round(wmean("tool_accuracy"), 4),
            trajectory_efficiency=round(wmean("trajectory_efficiency"), 4),
            groundedness=round(wmean("groundedness"), 4),
            refusal=round(wmean("refusal"), 4),
            cost_usd=round(sum(c.cost_usd for c in costs), 6),
            latency_ms_p50=p50,
            latency_ms_p95=p95,
            tasks_run=len(scores),
            tasks_passed=passed,
        )
