"""
FastAPI Application for ITR Fraud Detection Environment.
Serves the environment as an HTTP API following OpenEnv spec.
"""

import sys
import os
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ITRAction, ITRObservation, ITRState, ITRStepResult
from server.itr_environment import ITREnvironment

# ── App Setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ITR Fraud Detection Environment",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to detect "
        "fraud in Indian Income Tax Returns."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (per container)
env = ITREnvironment()


# ── Request/Response Models ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    action: ITRAction


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    episode_id: str
    step_count: int
    task_id: str
    is_done: bool
    current_score: float


# ── OpenEnv Endpoints ─────────────────────────────────────────────────────

@app.post("/reset", response_model=Dict[str, Any])
async def reset(request: ResetRequest):
    """Reset the environment for a new episode."""
    try:
        observation = env.reset(task_id=request.task_id)
        return {
            "observation": observation.model_dump(),
            "info": {
                "task_id": request.task_id,
                "message": f"Environment reset for task: {request.task_id}",
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """Execute an action in the environment."""
    try:
        result = env.step(request.action)
        return StepResponse(
            observation=result.observation.model_dump(),
            reward=result.reward,
            done=result.done,
            info=result.info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=StateResponse)
async def state():
    """Get current episode state."""
    s = env.state()
    return StateResponse(
        episode_id=s.episode_id,
        step_count=s.step_count,
        task_id=s.task_id,
        is_done=s.is_done,
        current_score=s.current_score,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "itr_fraud_detection"}


@app.get("/info")
async def info():
    """Environment information."""
    return {
        "name": "ITR Fraud Detection Environment",
        "version": "1.0.0",
        "description": "AI agents detect fraud in Indian Income Tax Returns",
        "tasks": ["easy", "medium", "hard"],
        "action_types": [
            "investigate_field",
            "cross_reference",
            "request_document",
            "flag_anomaly",
            "render_verdict",
        ],
        "document_types": [
            "form_16",
            "bank_statement",
            "rent_receipts",
            "investment_proofs",
            "capital_gains_statement",
            "business_books",
            "related_party_records",
        ],
    }


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
