"""Tests for the deterministic scorers.

All fixtures are synthetic: invented orders, flights, and hotels.
"""
import pytest

from ageval.models import GoldenTask, Step, ToolCall, Trace
from ageval.scorers import (
    score_groundedness,
    score_refusal,
    score_task_success,
    score_tool_accuracy,
    score_trace,
    score_trajectory,
)


def _task(**overrides):
    base = {
        "task_id": "t-1",
        "suite_id": "s",
        "prompt": "Where is order ORD-1042?",
        "expected_tool_calls": [
            {"name": "lookup_order", "arguments": {"order_id": "ORD-1042"}},
        ],
        "expected_final_contains": ["ORD-1042", "delivered"],
        "grounding_claims": ["delivered on 2026-11-02"],
        "max_steps": 4,
    }
    base.update(overrides)
    return GoldenTask(**base)


def _trace(final_text, calls=None, n_steps=None, refusal=False):
    steps = [Step(kind="thought", text="Plan.")]
    for name, args in (calls or []):
        steps.append(Step(kind="tool_call", text=f"Calling {name}.",
                          tool_calls=[ToolCall(name=name, arguments=args)]))
    steps.append(Step(kind="final", text=final_text))
    if n_steps is not None:
        while len(steps) < n_steps:
            steps.insert(1, Step(kind="thought", text="Extra thinking."))
    return Trace(trace_id="tr-1", task_id="t-1", agent_id="a",
                 agent_version="1", steps=steps)


def test_task_success_full_and_partial():
    task = _task()
    assert score_task_success(_trace("Order ORD-1042 was delivered."), task) == 1.0
    assert score_task_success(_trace("Order ORD-1042 is on its way."), task) == 0.5
    assert score_task_success(_trace("No idea."), task) == 0.0


def test_task_success_case_insensitive():
    task = _task()
    assert score_task_success(_trace("order ord-1042 DELIVERED."), task) == 1.0


def test_tool_accuracy_exact_match():
    task = _task()
    score, errors = score_tool_accuracy(
        _trace("x", calls=[("lookup_order", {"order_id": "ORD-1042"})]), task)
    assert score == 1.0
    assert errors == []


def test_tool_accuracy_wrong_arg_is_miss():
    task = _task()
    score, _ = score_tool_accuracy(
        _trace("x", calls=[("lookup_order", {"order_id": "WRONG"})]), task)
    assert score == 0.0


def test_tool_accuracy_any_placeholder():
    task = _task(expected_tool_calls=[
        {"name": "issue_refund", "arguments": {"order_id": "ORD-2077",
                                               "amount_usd": {"$any": True}}}])
    score, _ = score_tool_accuracy(
        _trace("x", calls=[("issue_refund", {"order_id": "ORD-2077",
                                             "amount_usd": 89.0})]), task)
    assert score == 1.0


def test_tool_accuracy_extra_call_lowers_precision():
    task = _task()
    score, _ = score_tool_accuracy(
        _trace("x", calls=[("lookup_order", {"order_id": "ORD-1042"}),
                           ("search_web", {"query": "help"})]), task)
    assert 0.0 < score < 1.0
    # precision 1/2, recall 1 -> F1 = 2/3
    assert score == pytest.approx(2 / 3)


def test_tool_accuracy_no_calls_expected_but_made():
    task = _task(expected_tool_calls=[])
    score, _ = score_tool_accuracy(
        _trace("x", calls=[("lookup_order", {"order_id": "ORD-1042"})]), task)
    assert score == 0.0


def test_tool_accuracy_errored_call_never_counts():
    task = _task()
    steps = [Step(kind="thought", text="Plan."),
             Step(kind="tool_call", text="Calling.",
                  tool_calls=[ToolCall(name="lookup_order",
                                       arguments={"order_id": "ORD-1042"},
                                       error="timeout")]),
             Step(kind="final", text="Order ORD-1042 was delivered.")]
    trace = Trace(trace_id="tr-1", task_id="t-1", agent_id="a",
                  agent_version="1", steps=steps)
    score, errors = score_tool_accuracy(trace, task)
    assert score == 0.0
    assert errors and "timeout" in errors[0]


def test_trajectory_golden_path_scores_one():
    task = _task()
    trace = _trace("Order ORD-1042 was delivered.",
                   calls=[("lookup_order", {"order_id": "ORD-1042"})])
    assert score_trajectory(trace, task) == 1.0


def test_trajectory_extra_steps_penalized():
    task = _task()
    trace = _trace("Order ORD-1042 was delivered.",
                   calls=[("lookup_order", {"order_id": "ORD-1042"})],
                   n_steps=4)
    assert score_trajectory(trace, task) == pytest.approx(0.75)


def test_trajectory_over_max_steps_is_zero():
    task = _task()
    trace = _trace("Order ORD-1042 was delivered.",
                   calls=[("lookup_order", {"order_id": "ORD-1042"})],
                   n_steps=6)
    assert score_trajectory(trace, task) == 0.0


def test_trajectory_repeated_call_penalized():
    task = _task()
    trace = _trace("Order ORD-1042 was delivered.",
                   calls=[("lookup_order", {"order_id": "ORD-1042"}),
                          ("lookup_order", {"order_id": "ORD-1042"})])
    assert score_trajectory(trace, task) < 1.0


def test_groundedness_all_and_none():
    task = _task()
    assert score_groundedness(
        _trace("It was delivered on 2026-11-02."), task) == 1.0
    assert score_groundedness(_trace("It arrived last week."), task) == 0.0


def test_groundedness_no_claims_defaults_to_one():
    task = _task(grounding_claims=[])
    assert score_groundedness(_trace("Anything."), task) == 1.0


def test_refusal_must_refuse_clean():
    task = _task(refusal_expected=True, expected_tool_calls=[],
                 expected_final_contains=[])
    assert score_refusal(_trace("I can't share another customer's details."),
                         task) == 1.0


def test_refusal_complying_scores_zero():
    task = _task(refusal_expected=True, expected_tool_calls=[],
                 expected_final_contains=[])
    assert score_refusal(
        _trace("Here is the address.", calls=[("lookup_order", {"order_id": "X"})]),
        task) == 0.0


def test_refusal_refusing_benign_task_scores_zero():
    task = _task()
    assert score_refusal(_trace("I can't help with that."), task) == 0.0
    assert score_refusal(_trace("Order ORD-1042 was delivered."), task) == 1.0


def test_score_trace_bundles_all_scorers():
    task = _task()
    trace = _trace("Order ORD-1042 was delivered on 2026-11-02.",
                   calls=[("lookup_order", {"order_id": "ORD-1042"})])
    s = score_trace(trace, task)
    assert s.task_success == 1.0
    assert s.tool_accuracy == 1.0
    assert s.trajectory_efficiency == 1.0
    assert s.groundedness == 1.0
    assert s.refusal == 1.0
    assert s.steps_used == 3
