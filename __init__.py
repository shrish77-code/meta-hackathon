"""
ITR Fraud Detection Environment for OpenEnv
An environment where AI agents learn to detect fraud in Income Tax Returns.
"""

from .models import ITRAction, ITRObservation, ITRState, ITRStepResult
from .client import ITRFraudEnv

__all__ = ["ITRAction", "ITRObservation", "ITRState", "ITRStepResult", "ITRFraudEnv"]
