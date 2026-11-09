"""Versioned eval-run registry.

Runs are stored as one JSON document per run under a directory (the
default is ``./eval-runs/``), named ``<run_id>.json``. Each record carries
the suite id + version, agent id + version, per-task scores, aggregates,
cost, and the gate verdict — everything needed to reproduce a comparison
later. The registry is append-only in spirit: ``save_run`` refuses to
overwrite an existing run id.

Baselines are promoted explicitly, never auto-derived: a run can only
become a baseline when an operator (or a CI step) calls
:meth:`EvalRegistry.promote` on it. This avoids the circularity of
"the baseline is the latest passing run" — the first run for a suite
has no baseline, so its gates fail closed, and it can only anchor
future gates once a human has reviewed it and promoted it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ageval.models import RunAggregate, RunResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvalRegistry:
    """JSON-file registry of eval runs."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._baselines_dir = self.directory / "baselines"
        self._baselines_dir.mkdir(exist_ok=True)

    def _path(self, run_id: str) -> Path:
        if "/" in run_id or "\\" in run_id or run_id in {"", ".", ".."}:
            raise ValueError(f"invalid run_id: {run_id!r}")
        return self.directory / f"{run_id}.json"

    @staticmethod
    def _baseline_key(suite_id: str, suite_version: str, agent_id: str) -> str:
        return f"{suite_id}__{suite_version}__{agent_id}"

    def save_run(self, run: RunResult) -> Path:
        """Persist a run; raises if the run id already exists."""
        path = self._path(run.run_id)
        if path.exists():
            raise ValueError(f"run {run.run_id!r} already exists in the registry")
        if not run.created_at:
            run.created_at = _now()
        path.write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True),
                        encoding="utf-8")
        return path

    def get_run(self, run_id: str) -> RunResult:
        path = self._path(run_id)
        if not path.exists():
            raise KeyError(f"run {run_id!r} not found in the registry")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _run_from_dict(raw)

    def list_runs(self) -> list[RunResult]:
        runs = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                runs.append(_run_from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, KeyError):
                continue  # skip corrupt records rather than failing the listing
        return sorted(runs, key=lambda r: r.created_at)

    def promote(self, run_id: str) -> RunResult:
        """Mark a run as the baseline for its suite version + agent.

        Overwrites any previous baseline for the same key — promotion is
        an explicit, auditable decision, so the latest promotion wins.
        """
        run = self.get_run(run_id)
        key = self._baseline_key(run.suite_id, run.suite_version, run.agent_id)
        marker = self._baselines_dir / f"{key}.json"
        marker.write_text(json.dumps(
            {"run_id": run.run_id, "promoted_at": _now()},
            indent=2, sort_keys=True), encoding="utf-8")
        return run

    def get_baseline(
        self, suite_id: str, suite_version: str, agent_id: str
    ) -> RunAggregate | None:
        """The promoted baseline's aggregate for this suite version + agent."""
        key = self._baseline_key(suite_id, suite_version, agent_id)
        marker = self._baselines_dir / f"{key}.json"
        if not marker.exists():
            return None
        try:
            run_id = json.loads(marker.read_text(encoding="utf-8"))["run_id"]
            run = self.get_run(run_id)
        except (ValueError, KeyError):
            return None
        return run.aggregate


def _run_from_dict(raw: dict) -> RunResult:
    from ageval.models import (
        CostSummary,
        GateReport,
        GateResult,
        RunAggregate,
        TaskScore,
    )

    agg = raw.get("aggregate")
    cost = raw.get("cost")
    gates = raw.get("gates")
    return RunResult(
        run_id=raw["run_id"],
        suite_id=raw["suite_id"],
        suite_version=raw["suite_version"],
        agent_id=raw["agent_id"],
        agent_version=raw["agent_version"],
        scores=[TaskScore(**s) for s in raw.get("scores", [])],
        aggregate=RunAggregate(**agg) if agg else None,
        cost=CostSummary(**cost) if cost else None,
        gates=GateReport(
            run_id=gates["run_id"],
            verdict=gates["verdict"],
            results=[GateResult(**r) for r in gates.get("results", [])],
        ) if gates else None,
        created_at=raw.get("created_at", ""),
    )
