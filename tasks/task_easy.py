"""
Task 1 — Easy: "Obvious Red Flags"
Single ITR with 1-2 blatant anomalies (income mismatch, over-limit deductions).
Max steps: 10
"""

from typing import Any, Dict, List, Set


class EasyTask:
    """Easy difficulty task: detect obvious fraud patterns."""

    TASK_ID = "easy"
    TASK_NAME = "Obvious Red Flags"
    TASK_DESCRIPTION = (
        "You are reviewing an Income Tax Return (ITR) for potential fraud. "
        "This filing contains 1-2 OBVIOUS irregularities that a basic audit would catch. "
        "Investigate the return, flag any anomalies you find, and render your final verdict. "
        "Look for: income mismatches with TDS records, deductions exceeding statutory limits."
    )
    MAX_STEPS = 10
    SEED = 101

    # Expected fraud patterns for this task
    EXPECTED_PATTERNS = {"income_mismatch", "80c_over_limit"}

    @staticmethod
    def grade(
        flagged_anomalies: List[Dict[str, str]],
        verdict: str | None,
        verdict_confidence: float | None,
        actual_patterns: Set[str],
        actual_is_fraudulent: bool,
        steps_taken: int,
        max_steps: int,
    ) -> float:
        """
        Grade agent performance on the easy task.
        Returns score 0.0 - 1.0
        """
        score = 0.0

        # ── Anomaly Detection (50%) ───────────────────────────────────────
        if actual_patterns:
            flagged_fields = {a.get("field", "").lower() for a in flagged_anomalies}
            flagged_reasons = " ".join(a.get("reason", "").lower() for a in flagged_anomalies)

            # Check if key anomaly areas were flagged
            detected = 0
            for pattern in actual_patterns:
                if pattern == "income_mismatch":
                    if any("salary" in f or "income" in f for f in flagged_fields) or "salary" in flagged_reasons or "mismatch" in flagged_reasons:
                        detected += 1
                elif pattern == "80c_over_limit":
                    if any("80c" in f or "deduction" in f for f in flagged_fields) or "80c" in flagged_reasons or "limit" in flagged_reasons:
                        detected += 1

            anomaly_score = detected / len(actual_patterns) if actual_patterns else 0
            score += 0.5 * anomaly_score

        # ── Verdict Accuracy (40%) ────────────────────────────────────────
        if verdict is not None:
            verdict_lower = verdict.lower()
            if actual_is_fraudulent and verdict_lower in ("fraudulent", "suspicious"):
                score += 0.4
            elif not actual_is_fraudulent and verdict_lower == "legitimate":
                score += 0.4
            elif actual_is_fraudulent and verdict_lower == "legitimate":
                score += 0.0  # Missed fraud entirely
            elif not actual_is_fraudulent and verdict_lower in ("fraudulent", "suspicious"):
                score += 0.1  # False positive, partial credit

        # ── Efficiency (10%) ──────────────────────────────────────────────
        if steps_taken > 0:
            efficiency = max(0, 1.0 - (steps_taken / max_steps))
            score += 0.1 * efficiency

        return round(min(1.0, max(0.0, score)), 4)
