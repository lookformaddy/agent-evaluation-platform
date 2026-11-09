"""Per-run cost and latency accounting.

Pricing is a static table of USD per 1M tokens for the generic model
tiers the harness knows about. Unknown models fall back to the
``gen-small`` price and are flagged in the summary — silently pricing a
model at $0 would understate spend, and silently guessing a price would
overstate precision. Latency comes from trace step durations (wall-clock
of the agent run), reported as raw ms plus p50/p95 across a run.
"""
from __future__ import annotations

import math
import statistics

from ageval.models import CostSummary, Trace

# USD per 1M tokens: (input, output). Generic tiers, not vendor pricing.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gen-small": (0.50, 1.50),
    "gen-medium": (2.00, 6.00),
    "gen-large": (5.00, 15.00),
    "gen-reasoning": (10.00, 30.00),
}

DEFAULT_MODEL = "gen-small"


def price_for(model: str) -> tuple[float, float]:
    """Return (input, output) USD-per-1M-tokens price for a model."""
    return MODEL_PRICES.get(model, MODEL_PRICES[DEFAULT_MODEL])


def cost_for_trace(trace: Trace) -> CostSummary:
    """Cost/latency summary for one trace."""
    in_price, out_price = price_for(trace.model)
    in_tokens = trace.total_input_tokens
    out_tokens = trace.total_output_tokens
    cost = (in_tokens * in_price + out_tokens * out_price) / 1_000_000
    return CostSummary(
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        total_tokens=in_tokens + out_tokens,
        cost_usd=round(cost, 6),
        latency_ms=trace.total_duration_ms,
        tool_calls=len(trace.tool_calls),
        steps=len(trace.steps),
    )


def latency_percentiles(latencies_ms: list[float]) -> tuple[float, float]:
    """(p50, p95) of per-trace latencies; (0, 0) for an empty run."""
    if not latencies_ms:
        return 0.0, 0.0
    ordered = sorted(latencies_ms)
    p50 = statistics.median(ordered)
    # Nearest-rank percentile: p95 is the ceil(0.95 * n)-th value.
    idx = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return p50, ordered[idx]


def aggregate_cost(summaries: list[CostSummary]) -> CostSummary:
    """Sum per-trace cost summaries into one run-level summary."""
    return CostSummary(
        input_tokens=sum(s.input_tokens for s in summaries),
        output_tokens=sum(s.output_tokens for s in summaries),
        total_tokens=sum(s.total_tokens for s in summaries),
        cost_usd=round(sum(s.cost_usd for s in summaries), 6),
        latency_ms=sum(s.latency_ms for s in summaries),
        tool_calls=sum(s.tool_calls for s in summaries),
        steps=sum(s.steps for s in summaries),
    )
