"""Trace ingestion and validation.

Traces arrive as JSON (single objects or NDJSON streams) from any agent
implementation. Ingestion is fail-closed: a malformed trace — missing
fields, unknown step kinds, a trace that does not end in a final step,
negative token counts — raises ``ValueError`` with a precise message
instead of producing a score that looks fine. Validated traces are the
only input the scorers ever see.
"""
from __future__ import annotations

import json
from typing import Any

from ageval.models import Step, ToolCall, Trace

VALID_STEP_KINDS = {"thought", "tool_call", "observation", "final"}


def _tool_call_from_dict(raw: Any, where: str) -> ToolCall:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: tool call must be a JSON object")
    for key in ("name", "arguments"):
        if key not in raw:
            raise ValueError(f"{where}: tool call missing required field {key!r}")
    try:
        return ToolCall(
            name=raw["name"],
            arguments=raw["arguments"],
            result=raw.get("result", ""),
            duration_ms=float(raw.get("duration_ms", 0.0)),
            error=raw.get("error"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: invalid tool call: {exc}") from exc


def _step_from_dict(raw: Any, where: str) -> Step:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: step must be a JSON object")
    if "kind" not in raw:
        raise ValueError(f"{where}: step missing required field 'kind'")
    if raw["kind"] not in VALID_STEP_KINDS:
        raise ValueError(f"{where}: unknown step kind {raw['kind']!r}")
    try:
        return Step(
            kind=raw["kind"],
            text=raw.get("text", ""),
            tool_calls=[
                _tool_call_from_dict(tc, f"{where}.tool_calls[{i}]")
                for i, tc in enumerate(raw.get("tool_calls", []))
            ],
            input_tokens=int(raw.get("input_tokens", 0)),
            output_tokens=int(raw.get("output_tokens", 0)),
            duration_ms=float(raw.get("duration_ms", 0.0)),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(where):
            raise
        raise ValueError(f"{where}: invalid step: {exc}") from exc


def trace_from_dict(raw: Any) -> Trace:
    """Validate one raw trace dict and return a :class:`Trace`.

    Raises ``ValueError`` on any structural problem. Unknown top-level
    keys are ignored so newer producers stay forward-compatible.
    """
    if not isinstance(raw, dict):
        raise ValueError("trace must be a JSON object")
    for key in ("trace_id", "task_id", "agent_id", "agent_version", "steps"):
        if key not in raw:
            raise ValueError(f"trace missing required field {key!r}")
    if not isinstance(raw["steps"], list) or not raw["steps"]:
        raise ValueError("trace 'steps' must be a non-empty list")
    steps = [
        _step_from_dict(s, f"steps[{i}]") for i, s in enumerate(raw["steps"])
    ]
    try:
        return Trace(
            trace_id=raw["trace_id"],
            task_id=raw["task_id"],
            agent_id=raw["agent_id"],
            agent_version=raw["agent_version"],
            steps=steps,
            model=raw.get("model", "gen-small"),
            started_at=raw.get("started_at", ""),
            ended_at=raw.get("ended_at", ""),
        )
    except ValueError as exc:
        raise ValueError(f"invalid trace: {exc}") from exc


def trace_from_json(text: str) -> Trace:
    """Parse a single JSON trace document."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"trace is not valid JSON: {exc}") from exc
    return trace_from_dict(raw)


def traces_from_ndjson(text: str) -> list[Trace]:
    """Parse an NDJSON stream of traces; blank lines are skipped.

    Fails on the first malformed line (with its line number) — a partial
    import would silently skew a suite's aggregates.
    """
    traces: list[Trace] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            traces.append(trace_from_json(line))
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
    return traces


def trace_to_json(trace: Trace) -> str:
    return json.dumps(trace.to_dict(), indent=2, sort_keys=True)
