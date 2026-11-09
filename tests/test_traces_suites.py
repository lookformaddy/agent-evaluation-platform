"""Tests for trace ingestion/validation and the golden suites."""
import json

import pytest

from ageval.models import Trace
from ageval.suites import SUITES, get_suite, list_suites
from ageval.traces import trace_from_dict, trace_from_json, traces_from_ndjson

VALID_TRACE = {
    "trace_id": "tr-1",
    "task_id": "support-001",
    "agent_id": "scripted-good",
    "agent_version": "1.0.0",
    "model": "gen-small",
    "steps": [
        {"kind": "thought", "text": "Plan.", "input_tokens": 100,
         "output_tokens": 50, "duration_ms": 120},
        {"kind": "tool_call", "text": "Calling lookup_order.",
         "tool_calls": [{"name": "lookup_order",
                         "arguments": {"order_id": "ORD-1042"},
                         "result": "delivered", "duration_ms": 80}],
         "input_tokens": 150, "output_tokens": 80, "duration_ms": 200},
        {"kind": "final", "text": "Order ORD-1042 was delivered.",
         "input_tokens": 200, "output_tokens": 120, "duration_ms": 150},
    ],
}


def test_valid_trace_ingests():
    trace = trace_from_dict(VALID_TRACE)
    assert isinstance(trace, Trace)
    assert trace.final_text == "Order ORD-1042 was delivered."
    assert len(trace.tool_calls) == 1
    assert trace.total_input_tokens == 450
    assert trace.total_output_tokens == 250
    assert trace.total_duration_ms == 470


def test_missing_field_rejected():
    raw = dict(VALID_TRACE)
    del raw["task_id"]
    with pytest.raises(ValueError, match="task_id"):
        trace_from_dict(raw)


def test_unknown_step_kind_rejected():
    raw = json.loads(json.dumps(VALID_TRACE))
    raw["steps"][0]["kind"] = "teleport"
    with pytest.raises(ValueError, match="unknown step kind"):
        trace_from_dict(raw)


def test_trace_not_ending_in_final_rejected():
    raw = json.loads(json.dumps(VALID_TRACE))
    raw["steps"] = raw["steps"][:2]
    with pytest.raises(ValueError, match="must end with a final step"):
        trace_from_dict(raw)


def test_two_final_steps_rejected():
    raw = json.loads(json.dumps(VALID_TRACE))
    raw["steps"].append({"kind": "final", "text": "Again."})
    with pytest.raises(ValueError, match="exactly one final step"):
        trace_from_dict(raw)


def test_negative_tokens_rejected():
    raw = json.loads(json.dumps(VALID_TRACE))
    raw["steps"][0]["input_tokens"] = -5
    with pytest.raises(ValueError):
        trace_from_dict(raw)


def test_tool_call_step_without_calls_rejected():
    raw = json.loads(json.dumps(VALID_TRACE))
    raw["steps"][1]["tool_calls"] = []
    with pytest.raises(ValueError, match="at least one tool call"):
        trace_from_dict(raw)


def test_invalid_json_rejected():
    with pytest.raises(ValueError, match="not valid JSON"):
        trace_from_json("{not json")


def test_ndjson_round_trip_and_bad_line():
    text = json.dumps(VALID_TRACE) + "\n" + json.dumps(VALID_TRACE) + "\n"
    traces = traces_from_ndjson(text)
    assert len(traces) == 2
    with pytest.raises(ValueError, match="line 2"):
        traces_from_ndjson(json.dumps(VALID_TRACE) + "\n{bad}\n")


def test_suites_are_versioned_and_nonempty():
    suites = list_suites()
    assert {s.suite_id for s in suites} == {"support-triage", "travel-booking"}
    for suite in suites:
        assert suite.version
        assert len(suite.tasks) >= 3
        for task in suite.tasks:
            assert task.suite_id == suite.suite_id
            assert task.max_steps >= 1


def test_support_suite_has_refusal_task():
    suite = get_suite("support-triage")
    refusal_tasks = [t for t in suite.tasks if t.refusal_expected]
    assert len(refusal_tasks) == 1
    assert refusal_tasks[0].expected_tool_calls == []


def test_unknown_suite_raises():
    with pytest.raises(KeyError):
        get_suite("no-such-suite")


def test_suite_get_task():
    suite = get_suite("travel-booking")
    task = suite.get("travel-001")
    assert "SFO" in task.prompt
    with pytest.raises(KeyError):
        suite.get("travel-999")
