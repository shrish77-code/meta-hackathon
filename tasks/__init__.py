"""Task definitions for ITR Fraud Detection environment."""

from .task_easy import EasyTask
from .task_medium import MediumTask
from .task_hard import HardTask

TASKS = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}

__all__ = ["TASKS", "EasyTask", "MediumTask", "HardTask"]
