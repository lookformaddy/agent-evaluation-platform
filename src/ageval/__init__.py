"""Agent Evaluation Platform (ageval).

Evaluation harness for agentic systems: golden task suites, deterministic
scorers (task success, tool-call accuracy, trajectory efficiency,
groundedness, refusal appropriateness), per-run cost/latency accounting,
baseline comparison with fail-closed regression gates, and a versioned
eval-run registry. Pure stdlib core; FastAPI is an optional thin wrapper.
"""
from __future__ import annotations

from ageval.models import (
    CostSummary,
    GateReport,
    GateResult,
    GoldenTask,
    RunAggregate,
    RunResult,
    Step,
    TaskScore,
    ToolCall,
    Trace,
)

__all__ = [
    "CostSummary",
    "GateReport",
    "GateResult",
    "GoldenTask",
    "RunAggregate",
    "RunResult",
    "Step",
    "TaskScore",
    "ToolCall",
    "Trace",
]

__version__ = "1.0.0"
