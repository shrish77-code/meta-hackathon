"""
Task 3 — Hard: "The Elaborate Scheme"
Multi-year ITR with complex fraud involving shell companies, penny stocks,
income splitting, and high-value cash transactions.
Max steps: 25
"""

from typing import Any, Dict, List, Set


class HardTask:
    """Hard difficulty task: detect complex multi-layered fraud."""

    TASK_ID = "hard"
    TASK_NAME = "The Elaborate Scheme"
    TASK_DESCRIPTION = (
        "You are investigating a complex ITR filing suspected of elaborate fraud. "
        "This case involves MULTIPLE layers of deception: shell company invoicing, "
        "multi-year income shifting patterns, suspicious high-value cash transactions, "
        "excessive refund claims, and potentially fraudulent donation receipts. "
        "You must request and cross-reference multiple documents, analyze historical "
        "patterns across assessment years, and build a comprehensive case. "
        "Your verdict must be well-supported with evidence."
    )
    MAX_STEPS = 25
    SEED = 303

    EXPECTED_PATTERNS = {
        "shell_company_invoicing",
        "multi_year_income_shifting",
        "high_value_cash_transactions",
        "excessive_refund",
        "suspicious_80g_donations",
    }

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
        Grade agent performance on the hard task.
        Multi-dimensional: anomaly detection + pattern recognition + verdict quality.
        Returns score 0.0 - 1.0
        """
        score = 0.0

        # ── Anomaly Detection & Pattern Recognition (45%) ─────────────────
        pattern_weights = {
            "shell_company_invoicing": 0.25,
            "multi_year_income_shifting": 0.25,
            "high_value_cash_transactions": 0.20,
            "excessive_refund": 0.15,
            "suspicious_80g_donations": 0.15,
        }

        keyword_map = {
            "shell_company_invoicing": ["shell", "business", "related party", "invoic", "expense"],
            "multi_year_income_shifting": ["income shift", "year", "variation", "historical", "fluctuat", "previous"],
            "high_value_cash_transactions": ["cash", "high value", "transaction", "deposit", "property"],
            "excessive_refund": ["refund", "excessive", "disproportionate", "claim"],
            "suspicious_80g_donations": ["donation", "80g", "related", "suspicious"],
        }

        if actual_patterns:
            flagged_fields = {a.get("field", "").lower() for a in flagged_anomalies}
            flagged_reasons = " ".join(a.get("reason", "").lower() for a in flagged_anomalies)
            all_flagged_text = " ".join(flagged_fields) + " " + flagged_reasons

            pattern_score = 0.0
            patterns_found = 0
            for pattern in actual_patterns:
                weight = pattern_weights.get(pattern, 0.2)
                keywords = keyword_map.get(pattern, [])
                if any(kw in all_flagged_text for kw in keywords):
                    pattern_score += weight
                    patterns_found += 1

            score += 0.45 * min(1.0, pattern_score)

        # ── Verdict Quality (30%) ─────────────────────────────────────────
        if verdict is not None:
            verdict_lower = verdict.lower()
            if actual_is_fraudulent and verdict_lower == "fraudulent":
                verdict_score = 1.0
                # Bonus for high confidence on correct verdict
                if verdict_confidence and verdict_confidence > 0.8:
                    verdict_score = 1.0
                score += 0.3 * verdict_score
            elif actual_is_fraudulent and verdict_lower == "suspicious":
                score += 0.2  # Partial credit
            elif not actual_is_fraudulent and verdict_lower == "legitimate":
                score += 0.3
            elif actual_is_fraudulent and verdict_lower == "legitimate":
                score += 0.0  # Critical miss

        # ── Comprehensiveness (15%) ───────────────────────────────────────
        # How many distinct anomaly types were flagged?
        unique_fields = {a.get("field", "") for a in flagged_anomalies}
        comprehensiveness = min(1.0, len(unique_fields) / 3.0)
        score += 0.15 * comprehensiveness

        # ── Efficiency (10%) ──────────────────────────────────────────────
        if steps_taken > 0:
            efficiency = max(0, 1.0 - (steps_taken / max_steps))
            score += 0.1 * efficiency

        return round(min(1.0, max(0.0, score)), 4)
