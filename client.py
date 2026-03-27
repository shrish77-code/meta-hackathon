"""
ITR Fraud Detection Environment Client.
Provides a typed Python client for interacting with the environment server.
"""

from __future__ import annotations

import httpx
from typing import Any, Dict, Optional

from models import ITRAction, ITRObservation, ITRState, ITRStepResult


class ITRFraudEnv:
    """
    Client for the ITR Fraud Detection OpenEnv environment.

    Supports both sync/async usage:
        # Sync usage
        with ITRFraudEnv(base_url="http://localhost:8000") as client:
            obs = client.reset(task_id="easy")
            result = client.step(ITRAction(...))

        # Direct usage (local, no server)
        from server.itr_environment import ITREnvironment
        env = ITREnvironment()
        obs = env.reset(task_id="easy")
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.Client] = None

    def __enter__(self):
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)
        return self

    def __exit__(self, *args):
        if self._client:
            self._client.close()

    def _ensure_client(self):
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        self._ensure_client()
        response = self._client.post("/reset", json={"task_id": task_id})
        response.raise_for_status()
        return response.json()

    def step(self, action: ITRAction) -> Dict[str, Any]:
        """Execute an action in the environment."""
        self._ensure_client()
        response = self._client.post(
            "/step",
            json={"action": action.model_dump()},
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> Dict[str, Any]:
        """Get current episode state."""
        self._ensure_client()
        response = self._client.get("/state")
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        self._ensure_client()
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def info(self) -> Dict[str, Any]:
        """Get environment info."""
        self._ensure_client()
        response = self._client.get("/info")
        response.raise_for_status()
        return response.json()

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
