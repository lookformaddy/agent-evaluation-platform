"""Core data models for the agent evaluation platform.

All models are plain dataclasses with ``to_dict``/``from_dict`` round-trips
so traces, suites, and run results can move between JSON files, the CLI,
and the HTTP service without a second schema. Validation is fail-closed:
constructors reject impossible values (negative durations, empty ids)
rather than letting a bad record silently score well.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _require_non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass
class ToolCall:
    """One tool invocation inside a trace step."""

    name: str
    arguments: dict
    result: str = ""
    duration_ms: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "tool name")
        if not isinstance(self.arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        if self.duration_ms < 0:
            raise ValueError("tool duration_ms must be >= 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Step:
    """One step of an agent trace.

    ``kind`` is one of: ``thought`` | ``tool_call`` | ``observation`` |
    ``final``. A trace must end with exactly one ``final`` step; the
    runner and the ingestion validator both enforce this.
    """

    kind: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"thought", "tool_call", "observation", "final"}:
            raise ValueError(f"unknown step kind: {self.kind!r}")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be >= 0")
        if self.duration_ms < 0:
            raise ValueError("step duration_ms must be >= 0")
        if self.kind == "tool_call" and not self.tool_calls:
            raise ValueError("a tool_call step must carry at least one tool call")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trace:
    """A complete agent run for one task: ordered steps plus metadata."""

    trace_id: str
    task_id: str
    agent_id: str
    agent_version: str
    steps: list[Step] = field(default_factory=list)
    model: str = "gen-small"
    started_at: str = ""
    ended_at: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.trace_id, "trace_id")
        _require_non_empty(self.task_id, "task_id")
        _require_non_empty(self.agent_id, "agent_id")
        if not self.steps:
            raise ValueError("a trace must contain at least one step")
        if self.steps[-1].kind != "final":
            raise ValueError("a trace must end with a final step")
        if sum(1 for s in self.steps if s.kind == "final") != 1:
            raise ValueError("a trace must contain exactly one final step")

    @property
    def final_text(self) -> str:
        return self.steps[-1].text

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [tc for s in self.steps for tc in s.tool_calls]

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GoldenTask:
    """One golden task: prompt, ground truth, and scoring contract.

    ``expected_tool_calls`` lists the tool calls a correct agent makes, in
    order, as ``{"name": ..., "arguments": {...}}`` — argument values may be
    exact, or a ``{"$any": true}`` placeholder when only presence matters.
    ``expected_final_contains`` lists phrases the final answer must contain
    (case-insensitive substring match). ``grounding_context`` is the text the
    final answer's factual claims must be supported by. ``refusal_expected``
    marks tasks the agent must refuse (e.g. a request to exfiltrate data);
    for those tasks the success scorer is replaced by the refusal scorer.
    """

    task_id: str
    suite_id: str
    prompt: str
    expected_tool_calls: list[dict] = field(default_factory=list)
    expected_final_contains: list[str] = field(default_factory=list)
    grounding_context: str = ""
    grounding_claims: list[str] = field(default_factory=list)
    refusal_expected: bool = False
    max_steps: int = 8
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require_non_empty(self.task_id, "task_id")
        _require_non_empty(self.suite_id, "suite_id")
        _require_non_empty(self.prompt, "prompt")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.weight <= 0:
            raise ValueError("weight must be > 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskScore:
    """Per-task scores from every scorer (each in [0, 1])."""

    task_id: str
    task_success: float
    tool_accuracy: float
    trajectory_efficiency: float
    groundedness: float
    refusal: float
    steps_used: int
    tool_call_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CostSummary:
    """Per-run cost/latency accounting."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    tool_calls: int
    steps: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunAggregate:
    """Suite-level aggregates (weighted means over task scores)."""

    task_success: float
    tool_accuracy: float
    trajectory_efficiency: float
    groundedness: float
    refusal: float
    cost_usd: float
    latency_ms_p50: float
    latency_ms_p95: float
    tasks_run: int
    tasks_passed: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateResult:
    """One regression-gate check against the baseline."""

    metric: str
    threshold: float
    actual: float | None
    baseline: float | None
    passed: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateReport:
    """Fail-closed gate verdict: any failed check blocks the run."""

    run_id: str
    verdict: str  # PASS | FAIL
    results: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    """One full eval run: suite, agent, scores, cost, gates, registry id."""

    run_id: str
    suite_id: str
    suite_version: str
    agent_id: str
    agent_version: str
    scores: list[TaskScore] = field(default_factory=list)
    aggregate: RunAggregate | None = None
    cost: CostSummary | None = None
    gates: GateReport | None = None
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
