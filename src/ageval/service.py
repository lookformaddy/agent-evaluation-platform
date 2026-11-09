"""FastAPI surface for the agent evaluation platform.

Thin HTTP wrapper over the pure-stdlib core: it owns the registry and
the runner, and translates domain errors into HTTP responses
(ValueError/KeyError -> 400/404). All scoring, gating, and cost logic
lives in the core, so the API adds no second source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ageval import report as report_mod
from ageval.registry import EvalRegistry
from ageval.runner import EvalRunner, get_agent
from ageval.suites import SUITES, get_suite

REGISTRY_DIR = os.environ.get("AGEVAL_REGISTRY", "./eval-runs")
ACTOR = os.environ.get("AGEVAL_ACTOR", "api")

app = FastAPI(title="Agent Evaluation Platform", version="1.0.0")

registry = EvalRegistry(REGISTRY_DIR)
runner = EvalRunner(registry)


class RunRequest(BaseModel):
    suite_id: str = "support-triage"
    agent_id: str = "scripted-good"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "suites": sorted(SUITES)}


@app.get("/suites")
def list_suites() -> dict:
    return {
        suite.suite_id: {
            "version": suite.version,
            "description": suite.description,
            "tasks": [t.task_id for t in suite.tasks],
        }
        for suite in SUITES.values()
    }


@app.post("/runs", status_code=201)
def run_eval(req: RunRequest) -> dict:
    try:
        suite = get_suite(req.suite_id)
        agent = get_agent(req.agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    run = runner.run_suite(suite, agent)
    return run.to_dict()


@app.get("/runs")
def list_runs() -> dict:
    runs = registry.list_runs()
    return {"runs": [r.to_dict() for r in runs]}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return registry.get_run(run_id).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}/report", response_model=None)
def get_report(run_id: str, format: str = "markdown"):  # noqa: A002
    try:
        run = registry.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if format == "json":
        return run.to_dict()
    if format == "markdown":
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(report_mod.render_markdown(run))
    raise HTTPException(status_code=400, detail="format must be markdown or json")


@app.get("/runs/{run_id}/gates")
def get_gates(run_id: str) -> dict:
    try:
        run = registry.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if run.gates is None:
        raise HTTPException(status_code=404, detail="run was not gated")
    return run.gates.to_dict()
