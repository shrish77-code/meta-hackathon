"""
Task 2 — Medium: "The Subtle Cheat"
ITR with 2-3 subtle inconsistencies requiring cross-referencing.
Max steps: 15
"""

from typing import Any, Dict, List, Set


class MediumTask:
    """Medium difficulty task: detect subtle fraud patterns."""

    TASK_ID = "medium"
    TASK_NAME = "The Subtle Cheat"
    TASK_DESCRIPTION = (
        "You are auditing an ITR that appears mostly compliant on the surface. "
        "However, there are 2-3 SUBTLE inconsistencies hidden in this return. "
        "You will need to cross-reference fields, request supporting documents, "
        "and look for patterns like phantom HRA claims, suspicious capital gains "
        "through penny stocks, and computation errors. Be thorough but efficient."
    )
    MAX_STEPS = 15
    SEED = 202

    EXPECTED_PATTERNS = {"phantom_hra", "penny_stock_ltcg", "tax_computation_error"}

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
        Grade agent performance on the medium task.
        Weighted score considering anomaly severity and investigation depth.
        Returns score 0.0 - 1.0
        """
        score = 0.0

        # ── Anomaly Detection (40%) ───────────────────────────────────────
        pattern_weights = {
            "phantom_hra": 0.3,
            "penny_stock_ltcg": 0.4,
            "tax_computation_error": 0.3,
        }

        if actual_patterns:
            flagged_fields = {a.get("field", "").lower() for a in flagged_anomalies}
            flagged_reasons = " ".join(a.get("reason", "").lower() for a in flagged_anomalies)

            pattern_score = 0.0
            for pattern in actual_patterns:
                weight = pattern_weights.get(pattern, 0.33)
                if pattern == "phantom_hra":
                    if any("hra" in f for f in flagged_fields) or "hra" in flagged_reasons or "rent" in flagged_reasons:
                        pattern_score += weight
                elif pattern == "penny_stock_ltcg":
                    if any("capital" in f or "ltcg" in f or "stock" in f for f in flagged_fields) or \
                       any(k in flagged_reasons for k in ["penny", "ltcg", "capital gain", "stock"]):
                        pattern_score += weight
                elif pattern == "tax_computation_error":
                    if any("tax" in f or "liability" in f for f in flagged_fields) or \
                       any(k in flagged_reasons for k in ["computation", "tax", "liability", "mismatch"]):
                        pattern_score += weight

            score += 0.4 * min(1.0, pattern_score)

        # ── Investigation Depth (20%) ─────────────────────────────────────
        # Did the agent actually investigate (cross-reference, request docs)?
        investigation_actions = len(flagged_anomalies)
        depth_score = min(1.0, investigation_actions / 2.0)  # At least 2 flags expected
        score += 0.2 * depth_score

        # ── Verdict Accuracy (30%) ────────────────────────────────────────
        if verdict is not None:
            verdict_lower = verdict.lower()
            if actual_is_fraudulent and verdict_lower == "fraudulent":
                score += 0.3
            elif actual_is_fraudulent and verdict_lower == "suspicious":
                score += 0.2
            elif not actual_is_fraudulent and verdict_lower == "legitimate":
                score += 0.3
            elif actual_is_fraudulent and verdict_lower == "legitimate":
                score += 0.0

        # ── Efficiency (10%) ──────────────────────────────────────────────
        if steps_taken > 0:
            efficiency = max(0, 1.0 - (steps_taken / max_steps))
            score += 0.1 * efficiency

        return round(min(1.0, max(0.0, score)), 4)
