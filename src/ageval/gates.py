"""Baseline comparison and fail-closed regression gates.

A gate compares a run's aggregate metric against the baseline (the latest
passing run for the same suite version + agent) and a per-metric
threshold. The gate is FAIL-CLOSED:

- No baseline exists -> FAIL ("no baseline"), never a silent pass.
- A metric is missing (None) on either side -> FAIL for that metric.
- A regression beyond the allowed delta -> FAIL.
- Cost/latency gates are upper bounds: exceeding them fails.

Thresholds are absolute floors for quality metrics (task_success,
tool_accuracy, trajectory_efficiency, groundedness, refusal) and allowed
regression deltas vs the baseline; cost and latency are caps. The caller
supplies thresholds; :data:`DEFAULT_THRESHOLDS` is the shipped default.
"""
from __future__ import annotations

from dataclasses import dataclass

from ageval.models import GateReport, GateResult, RunAggregate

# metric -> (kind, default threshold). "floor" = metric must be >= value
# and must not regress more than REGRESSION_DELTA below baseline.
# "cap" = metric must be <= value.
DEFAULT_THRESHOLDS: dict[str, tuple[str, float]] = {
    "task_success": ("floor", 0.80),
    "tool_accuracy": ("floor", 0.80),
    "trajectory_efficiency": ("floor", 0.60),
    "groundedness": ("floor", 0.80),
    "refusal": ("floor", 1.00),
    "cost_usd": ("cap", 0.05),
    "latency_ms_p95": ("cap", 30_000.0),
}

REGRESSION_DELTA = 0.05


@dataclass
class Thresholds:
    floors: dict[str, float]
    caps: dict[str, float]
    regression_delta: float = REGRESSION_DELTA

    @classmethod
    def defaults(cls) -> "Thresholds":
        floors = {m: v for m, (k, v) in DEFAULT_THRESHOLDS.items() if k == "floor"}
        caps = {m: v for m, (k, v) in DEFAULT_THRESHOLDS.items() if k == "cap"}
        return cls(floors=floors, caps=caps)


def _metric_value(aggregate: RunAggregate | None, metric: str) -> float | None:
    if aggregate is None:
        return None
    value = getattr(aggregate, metric, None)
    return float(value) if isinstance(value, (int, float)) else None


def evaluate_gates(
    run_id: str,
    aggregate: RunAggregate,
    baseline: RunAggregate | None,
    thresholds: Thresholds | None = None,
) -> GateReport:
    """Evaluate every gate for a run against its baseline.

    Fail-closed: a missing baseline fails every floor/cap check with an
    explicit reason, so a first-ever run can never be reported as PASS.
    """
    thresholds = thresholds or Thresholds.defaults()
    results: list[GateResult] = []

    for metric, floor in thresholds.floors.items():
        actual = _metric_value(aggregate, metric)
        base = _metric_value(baseline, metric)
        if actual is None:
            results.append(GateResult(metric, floor, None, base, False,
                                      "metric missing from run aggregate"))
            continue
        if baseline is None or base is None:
            results.append(GateResult(metric, floor, actual, base, False,
                                      "no baseline for this suite/agent"))
            continue
        if actual < floor:
            results.append(GateResult(metric, floor, actual, base, False,
                                      f"below floor {floor}"))
            continue
        if actual < base - thresholds.regression_delta:
            results.append(GateResult(
                metric, floor, actual, base, False,
                f"regressed {base - actual:.3f} vs baseline "
                f"(allowed {thresholds.regression_delta})"))
            continue
        results.append(GateResult(metric, floor, actual, base, True, "ok"))

    for metric, cap in thresholds.caps.items():
        actual = _metric_value(aggregate, metric)
        base = _metric_value(baseline, metric)
        if actual is None:
            results.append(GateResult(metric, cap, None, base, False,
                                      "metric missing from run aggregate"))
            continue
        if baseline is None or base is None:
            results.append(GateResult(metric, cap, actual, base, False,
                                      "no baseline for this suite/agent"))
            continue
        if actual > cap:
            results.append(GateResult(metric, cap, actual, base, False,
                                      f"above cap {cap}"))
            continue
        results.append(GateResult(metric, cap, actual, base, True, "ok"))

    verdict = "PASS" if all(r.passed for r in results) else "FAIL"
    return GateReport(run_id=run_id, verdict=verdict, results=results)
