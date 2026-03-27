"""
ITR Fraud Detection Environment — Core Environment Logic
Implements the OpenEnv Environment base class with step(), reset(), state().
"""

from __future__ import annotations

import json
import sys
import uuid
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, ".")

from models import (
    ActionType,
    DocumentResult,
    DocumentType,
    ITRAction,
    ITRFiling,
    ITRObservation,
    ITRReward,
    ITRState,
    ITRStepResult,
    InvestigationResult,
    Severity,
    VerdictType,
)
from data.itr_generator import ITRGenerator
from tasks import TASKS


class ITREnvironment:
    """
    OpenEnv-compliant environment for ITR fraud detection.

    The agent acts as a tax auditor reviewing Income Tax Returns.
    They investigate fields, cross-reference data, request documents,
    flag anomalies, and render a final verdict.
    """

    def __init__(self):
        self._state: Optional[ITRState] = None
        self._filing: Optional[ITRFiling] = None
        self._task = None
        self._investigation_results: List[InvestigationResult] = []
        self._document_results: List[DocumentResult] = []
        self._flagged_anomalies: List[Dict[str, str]] = []
        self._investigated_fields: Set[str] = set()
        self._cross_referenced: Set[str] = set()
        self._requested_docs: Set[str] = set()
        self._verdict_rendered: bool = False
        self._cumulative_reward: float = 0.0
        self._last_action_result: Optional[str] = None
        self._final_verdict: Optional[str] = None
        self._final_confidence: Optional[float] = None

    # ── OpenEnv API ───────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> ITRObservation:
        """Reset the environment for a new episode."""
        # Load task config
        task_class = TASKS.get(task_id)
        if not task_class:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")

        self._task = task_class
        gen = ITRGenerator(seed=task_class.SEED)

        # Generate filing based on difficulty
        if task_id == "easy":
            self._filing = gen.generate_easy_fraud()
        elif task_id == "medium":
            self._filing = gen.generate_medium_fraud()
        elif task_id == "hard":
            self._filing = gen.generate_hard_fraud()

        # Reset state
        self._state = ITRState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            is_done=False,
            current_score=0.0,
        )

        # Reset tracking
        self._investigation_results = []
        self._document_results = []
        self._flagged_anomalies = []
        self._investigated_fields = set()
        self._cross_referenced = set()
        self._requested_docs = set()
        self._verdict_rendered = False
        self._cumulative_reward = 0.0
        self._last_action_result = None
        self._final_verdict = None
        self._final_confidence = None

        return self._build_observation()

    def step(self, action: ITRAction) -> ITRStepResult:
        """Execute an action and return the result."""
        if self._state is None or self._filing is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state.is_done:
            return ITRStepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"message": "Episode already done. Call reset() to start a new one."},
            )

        self._state.step_count += 1
        reward = ITRReward(total=0.0)

        # Process action
        if action.action_type == ActionType.INVESTIGATE_FIELD:
            reward = self._handle_investigate(action, reward)
        elif action.action_type == ActionType.CROSS_REFERENCE:
            reward = self._handle_cross_reference(action, reward)
        elif action.action_type == ActionType.REQUEST_DOCUMENT:
            reward = self._handle_request_document(action, reward)
        elif action.action_type == ActionType.FLAG_ANOMALY:
            reward = self._handle_flag_anomaly(action, reward)
        elif action.action_type == ActionType.RENDER_VERDICT:
            reward = self._handle_render_verdict(action, reward)

        # Check termination
        done = False
        if self._verdict_rendered:
            done = True
        elif self._state.step_count >= self._task.MAX_STEPS:
            done = True
            self._last_action_result = "⚠️ Maximum steps reached. Episode terminated without verdict."
            reward.penalty -= 0.1
            reward.total -= 0.1

        self._state.is_done = done
        self._cumulative_reward += reward.total

        # Compute final score if done
        info: Dict[str, Any] = {"reward_breakdown": reward.model_dump()}
        if done:
            final_score = self._task.grade(
                flagged_anomalies=self._flagged_anomalies,
                verdict=self._final_verdict,
                verdict_confidence=self._final_confidence,
                actual_patterns=set(self._filing.fraud_patterns),
                actual_is_fraudulent=self._filing.is_fraudulent,
                steps_taken=self._state.step_count,
                max_steps=self._task.MAX_STEPS,
            )
            self._state.current_score = final_score
            info["final_score"] = final_score
            info["actual_is_fraudulent"] = self._filing.is_fraudulent
            info["actual_patterns"] = self._filing.fraud_patterns
            info["anomaly_details"] = self._filing.anomaly_details

        return ITRStepResult(
            observation=self._build_observation(),
            reward=reward.total,
            done=done,
            info=info,
            reward_breakdown=reward,
        )

    def state(self) -> ITRState:
        """Return current episode state."""
        if self._state is None:
            return ITRState()
        return self._state

    # ── Action Handlers ───────────────────────────────────────────────────

    def _handle_investigate(self, action: ITRAction, reward: ITRReward) -> ITRReward:
        """Handle investigate_field action."""
        field = action.field_name or ""

        if field in self._investigated_fields:
            self._last_action_result = f"⚠️ Already investigated '{field}'. No new information."
            reward.penalty -= 0.02
            reward.total -= 0.02
            return reward

        self._investigated_fields.add(field)
        finding = self._get_field_investigation(field)
        self._investigation_results.append(finding)
        self._last_action_result = f"🔍 Investigation of '{field}': {finding.finding}"

        if finding.suspicious:
            reward.investigation_reward += 0.05
            reward.total += 0.05
        else:
            reward.investigation_reward += 0.01
            reward.total += 0.01

        return reward

    def _handle_cross_reference(self, action: ITRAction, reward: ITRReward) -> ITRReward:
        """Handle cross_reference action."""
        field_a = action.field_a or ""
        field_b = action.field_b or ""
        cross_key = f"{field_a}|{field_b}"

        if cross_key in self._cross_referenced:
            self._last_action_result = f"⚠️ Already cross-referenced '{field_a}' with '{field_b}'."
            reward.penalty -= 0.02
            reward.total -= 0.02
            return reward

        self._cross_referenced.add(cross_key)
        finding = self._get_cross_reference(field_a, field_b)
        self._investigation_results.append(finding)
        self._last_action_result = f"🔗 Cross-reference '{field_a}' ↔ '{field_b}': {finding.finding}"

        if finding.suspicious:
            reward.investigation_reward += 0.08
            reward.total += 0.08
        else:
            reward.investigation_reward += 0.02
            reward.total += 0.02

        return reward

    def _handle_request_document(self, action: ITRAction, reward: ITRReward) -> ITRReward:
        """Handle request_document action."""
        doc_type = action.document_type
        if doc_type is None:
            self._last_action_result = "❌ Must specify document_type."
            reward.penalty -= 0.02
            reward.total -= 0.02
            return reward

        doc_key = doc_type.value
        if doc_key in self._requested_docs:
            self._last_action_result = f"⚠️ Already requested '{doc_key}'."
            reward.penalty -= 0.02
            reward.total -= 0.02
            return reward

        self._requested_docs.add(doc_key)
        result = self._get_document(doc_type)
        self._document_results.append(result)
        self._last_action_result = f"📄 Document '{doc_key}': {result.content_summary}"

        if result.discrepancies:
            reward.investigation_reward += 0.06
            reward.total += 0.06
        else:
            reward.investigation_reward += 0.02
            reward.total += 0.02

        return reward

    def _handle_flag_anomaly(self, action: ITRAction, reward: ITRReward) -> ITRReward:
        """Handle flag_anomaly action."""
        anomaly = {
            "field": action.anomaly_field or "",
            "reason": action.anomaly_reason or "",
            "severity": (action.anomaly_severity or Severity.MEDIUM).value,
        }

        # Check if this is a real anomaly
        field = anomaly["field"].lower()
        reason = anomaly["reason"].lower()
        is_correct = False

        for pattern_field, detail in self._filing.anomaly_details.items():
            if (pattern_field.lower() in field or field in pattern_field.lower() or
                any(keyword in reason for keyword in detail.lower().split()[:5])):
                is_correct = True
                break

        self._flagged_anomalies.append(anomaly)

        if is_correct:
            self._last_action_result = f"✅ Anomaly flagged in '{anomaly['field']}': noted for review."
            reward.anomaly_reward += 0.10
            reward.total += 0.10
        else:
            self._last_action_result = f"⚠️ Anomaly flagged in '{anomaly['field']}': noted, but may not be significant."
            reward.penalty -= 0.05
            reward.total -= 0.05

        return reward

    def _handle_render_verdict(self, action: ITRAction, reward: ITRReward) -> ITRReward:
        """Handle render_verdict action."""
        self._verdict_rendered = True
        verdict = action.verdict
        confidence = action.confidence or 0.5
        explanation = action.explanation or ""

        self._final_verdict = verdict.value if verdict else None
        self._final_confidence = confidence

        is_correct = (
            (self._filing.is_fraudulent and verdict in (VerdictType.FRAUDULENT, VerdictType.SUSPICIOUS))
            or (not self._filing.is_fraudulent and verdict == VerdictType.LEGITIMATE)
        )

        if is_correct:
            reward.verdict_reward += 0.3
            reward.total += 0.3
            self._last_action_result = f"⚖️ Verdict rendered: {verdict.value} (confidence: {confidence:.0%})"
        else:
            reward.verdict_reward -= 0.2
            reward.total -= 0.2
            self._last_action_result = f"⚖️ Verdict rendered: {verdict.value} (confidence: {confidence:.0%})"

        # Efficiency bonus
        steps_used = self._state.step_count
        max_steps = self._task.MAX_STEPS
        efficiency = max(0, 1.0 - (steps_used / max_steps))
        reward.efficiency_bonus = round(0.05 * efficiency, 4)
        reward.total += reward.efficiency_bonus

        return reward

    # ── Investigation Logic ───────────────────────────────────────────────

    def _get_field_investigation(self, field: str) -> InvestigationResult:
        """Get investigation details for a specific field."""
        field_lower = field.lower()

        # Check if field matches any known anomaly
        for anomaly_field, detail in self._filing.anomaly_details.items():
            if anomaly_field.lower() in field_lower or field_lower in anomaly_field.lower():
                return InvestigationResult(
                    finding=f"Potential irregularity detected in {field}.",
                    suspicious=True,
                    detail=detail,
                )

        # Return normal findings for non-anomalous fields
        return self._get_normal_field_data(field)

    def _get_normal_field_data(self, field: str) -> InvestigationResult:
        """Return normal investigation data for a field."""
        field_lower = field.lower()
        f = self._filing
        field_data = {
            "income.salary": InvestigationResult(
                finding=f"Salary income: ₹{f.income.salary:,.0f}",
                suspicious=False,
                detail=f"Employer: {f.tds_entries[0].source if f.tds_entries else 'N/A'}. Salary appears consistent with profile.",
            ),
            "income.business_profession": InvestigationResult(
                finding=f"Business/Profession income: ₹{f.income.business_profession:,.0f}",
                suspicious=f.income.business_profession > 2000000,
                detail=f"Employment type: {f.taxpayer.employer_type}.",
            ),
            "income.capital_gains_long": InvestigationResult(
                finding=f"Long-term capital gains: ₹{f.income.capital_gains_long:,.0f}",
                suspicious=f.income.capital_gains_long > 500000,
                detail=f"Check if gains are from listed equity (exempt up to ₹1.25L) or other assets.",
            ),
            "income.capital_gains_short": InvestigationResult(
                finding=f"Short-term capital gains: ₹{f.income.capital_gains_short:,.0f}",
                suspicious=False,
                detail="Taxed at 20% (listed) or slab rate (unlisted).",
            ),
            "deductions.section_80c": InvestigationResult(
                finding=f"Section 80C deduction: ₹{f.deductions.section_80c:,.0f} (limit: ₹1,50,000)",
                suspicious=f.deductions.section_80c > 150000,
                detail=f"Covers PPF, ELSS, LIC, NSC, etc. {'⚠️ EXCEEDS LIMIT' if f.deductions.section_80c > 150000 else 'Within limit.'}",
            ),
            "deductions.section_80d": InvestigationResult(
                finding=f"Section 80D deduction: ₹{f.deductions.section_80d:,.0f}",
                suspicious=False,
                detail="Medical insurance premium.",
            ),
            "deductions.hra_exemption": InvestigationResult(
                finding=f"HRA exemption claimed: ₹{f.deductions.hra_exemption:,.0f}",
                suspicious=f.deductions.hra_exemption > f.income.salary * 0.5,
                detail=f"City: {f.taxpayer.city_tier}. HRA as % of salary: {f.deductions.hra_exemption / max(1, f.income.salary) * 100:.0f}%",
            ),
            "deductions.section_80g": InvestigationResult(
                finding=f"Section 80G donations: ₹{f.deductions.section_80g:,.0f}",
                suspicious=f.deductions.section_80g > 100000,
                detail=f"{'Large donation amount — verify recipient legitimacy.' if f.deductions.section_80g > 100000 else 'Within normal range.'}",
            ),
            "refund_claimed": InvestigationResult(
                finding=f"Refund claimed: ₹{f.refund_claimed:,.0f}",
                suspicious=f.refund_claimed > f.total_tax_liability * 0.5,
                detail=f"Tax liability: ₹{f.total_tax_liability:,.0f}. TDS total: ₹{sum(t.tds_deducted for t in f.tds_entries):,.0f}.",
            ),
            "high_value_transactions": InvestigationResult(
                finding=f"High-value transactions: {len(f.high_value_transactions)} found, total ₹{sum(t.amount for t in f.high_value_transactions):,.0f}",
                suspicious=len(f.high_value_transactions) > 0 and sum(t.amount for t in f.high_value_transactions) > f.income.total,
                detail="; ".join(f"{t.transaction_type}: ₹{t.amount:,.0f} ({t.description})" for t in f.high_value_transactions) if f.high_value_transactions else "No high-value transactions reported.",
            ),
            "previous_years": InvestigationResult(
                finding=f"Previous years: {len(f.previous_years)} years of data available.",
                suspicious=any(abs(py.total_income - f.income.total) > f.income.total * 0.5 for py in f.previous_years),
                detail="; ".join(f"{py.assessment_year}: ₹{py.total_income:,.0f} income, ₹{py.tax_paid:,.0f} tax" for py in f.previous_years),
            ),
            "total_tax_liability": InvestigationResult(
                finding=f"Declared tax liability: ₹{f.total_tax_liability:,.0f}",
                suspicious=False,
                detail=f"Filing status: {f.taxpayer.filing_status}. Assessment year: {f.taxpayer.assessment_year}.",
            ),
            "bank_accounts_count": InvestigationResult(
                finding=f"Bank accounts: {f.bank_accounts_count}",
                suspicious=f.bank_accounts_count > 5,
                detail=f"{'Unusually high number of bank accounts.' if f.bank_accounts_count > 5 else 'Normal.'}",
            ),
        }

        # Fuzzy match
        for key, result in field_data.items():
            if key in field_lower or field_lower in key:
                return result

        return InvestigationResult(
            finding=f"No specific data available for '{field}'.",
            suspicious=False,
            detail="Try investigating specific fields like 'income.salary', 'deductions.section_80c', 'high_value_transactions', etc.",
        )

    def _get_cross_reference(self, field_a: str, field_b: str) -> InvestigationResult:
        """Cross-reference two fields for discrepancies."""
        f = self._filing
        a, b = field_a.lower(), field_b.lower()
        combined = f"{a}|{b}"

        # Salary vs TDS check
        if ("salary" in a and "tds" in b) or ("tds" in a and "salary" in b):
            if f.tds_entries:
                tds_salary = f.tds_entries[0].amount_credited
                declared = f.income.salary
                if abs(tds_salary - declared) > 50000:
                    return InvestigationResult(
                        finding=f"❌ MISMATCH: Declared salary ₹{declared:,.0f} vs TDS records showing ₹{tds_salary:,.0f}",
                        suspicious=True,
                        detail=f"Discrepancy of ₹{abs(tds_salary - declared):,.0f}. Employer TDS shows different amount credited.",
                    )
                return InvestigationResult(
                    finding=f"✓ Salary (₹{declared:,.0f}) matches TDS records (₹{tds_salary:,.0f})",
                    suspicious=False,
                    detail="No discrepancy found.",
                )

        # HRA vs rent/city
        if "hra" in combined and ("rent" in combined or "city" in combined or "salary" in combined):
            hra = f.deductions.hra_exemption
            salary = f.income.salary
            ratio = hra / max(1, salary)
            if ratio > 0.5:
                return InvestigationResult(
                    finding=f"❌ HRA (₹{hra:,.0f}) is {ratio:.0%} of salary — unusually high",
                    suspicious=True,
                    detail=f"For {f.taxpayer.city_tier} location. HRA exemption exceeds 50% of basic salary.",
                )
            return InvestigationResult(
                finding=f"✓ HRA (₹{hra:,.0f}) is {ratio:.0%} of salary — within normal range",
                suspicious=False,
                detail=f"City tier: {f.taxpayer.city_tier}",
            )

        # Income vs high-value transactions
        if "income" in combined and ("transaction" in combined or "high_value" in combined):
            total_txn = sum(t.amount for t in f.high_value_transactions)
            income = f.income.total
            if total_txn > income * 1.5:
                return InvestigationResult(
                    finding=f"❌ High-value transactions (₹{total_txn:,.0f}) exceed 150% of income (₹{income:,.0f})",
                    suspicious=True,
                    detail="Transactions significantly exceed declared income — source of funds questionable.",
                )
            return InvestigationResult(
                finding=f"✓ Transactions (₹{total_txn:,.0f}) proportionate to income (₹{income:,.0f})",
                suspicious=False,
                detail="No major discrepancy.",
            )

        # Tax liability vs income
        if "tax" in combined and "income" in combined:
            expected_tax = self._compute_expected_tax()
            declared_tax = f.total_tax_liability
            if abs(expected_tax - declared_tax) > expected_tax * 0.2:
                return InvestigationResult(
                    finding=f"❌ Declared tax (₹{declared_tax:,.0f}) deviates from expected (₹{expected_tax:,.0f})",
                    suspicious=True,
                    detail=f"Difference: ₹{abs(expected_tax - declared_tax):,.0f}. Tax computation may be incorrect.",
                )
            return InvestigationResult(
                finding=f"✓ Tax liability (₹{declared_tax:,.0f}) is close to expected (₹{expected_tax:,.0f})",
                suspicious=False,
                detail="Tax computation appears reasonable.",
            )

        # Previous years comparison
        if "previous" in combined or "year" in combined:
            if f.previous_years:
                incomes = [py.total_income for py in f.previous_years]
                max_variation = max(incomes) / max(1, min(incomes))
                if max_variation > 2.0:
                    return InvestigationResult(
                        finding=f"❌ Income varies {max_variation:.1f}x across years — suspicious pattern",
                        suspicious=True,
                        detail=f"Incomes: {', '.join(f'₹{i:,.0f}' for i in incomes)}. Possible income shifting.",
                    )
            return InvestigationResult(
                finding="✓ Year-over-year income variation within normal range",
                suspicious=False,
                detail="No unusual patterns detected.",
            )

        return InvestigationResult(
            finding=f"Cross-reference of '{field_a}' and '{field_b}': no specific comparison rule.",
            suspicious=False,
            detail="Try cross-referencing: salary↔tds, hra↔salary, income↔transactions, tax↔income.",
        )

    def _get_document(self, doc_type: DocumentType) -> DocumentResult:
        """Simulate document request."""
        f = self._filing
        discrepancies = []

        if doc_type == DocumentType.FORM_16:
            if f.tds_entries:
                tds = f.tds_entries[0]
                if abs(tds.amount_credited - f.income.salary) > 50000:
                    discrepancies.append(
                        f"Form 16 shows ₹{tds.amount_credited:,.0f} but declared salary is ₹{f.income.salary:,.0f}"
                    )
            return DocumentResult(
                document_type="Form 16",
                available=True,
                content_summary=f"Form 16 from {f.tds_entries[0].source if f.tds_entries else 'N/A'}. "
                               f"Gross salary: ₹{f.tds_entries[0].amount_credited if f.tds_entries else 0:,.0f}. "
                               f"TDS deducted: ₹{f.tds_entries[0].tds_deducted if f.tds_entries else 0:,.0f}.",
                discrepancies=discrepancies,
            )

        elif doc_type == DocumentType.RENT_RECEIPTS:
            has_hra = f.deductions.hra_exemption > 0
            if "phantom_hra" in f.fraud_patterns:
                return DocumentResult(
                    document_type="Rent Receipts",
                    available=False,
                    content_summary="⚠️ Rent receipts NOT available. Taxpayer claims HRA but cannot produce receipts.",
                    discrepancies=["HRA claimed without supporting rent receipts"],
                )
            return DocumentResult(
                document_type="Rent Receipts",
                available=has_hra,
                content_summary=f"Rent receipts {'available' if has_hra else 'not applicable (no HRA claimed)'}.",
                discrepancies=discrepancies,
            )

        elif doc_type == DocumentType.CAPITAL_GAINS_STATEMENT:
            if f.income.capital_gains_long > 500000:
                penny_stocks = [t for t in f.high_value_transactions if "penny" in t.description.lower() or "shares" in t.description.lower()]
                if penny_stocks:
                    discrepancies.append(f"LTCG from suspected penny stocks: {', '.join(t.description for t in penny_stocks)}")
            return DocumentResult(
                document_type="Capital Gains Statement",
                available=True,
                content_summary=f"STCG: ₹{f.income.capital_gains_short:,.0f}, LTCG: ₹{f.income.capital_gains_long:,.0f}.",
                discrepancies=discrepancies,
            )

        elif doc_type == DocumentType.BANK_STATEMENT:
            cash_deposits = [t for t in f.high_value_transactions if t.transaction_type == "cash_deposit"]
            if cash_deposits:
                discrepancies.append(f"Large cash deposits: {', '.join(f'₹{t.amount:,.0f} on {t.date}' for t in cash_deposits)}")
            return DocumentResult(
                document_type="Bank Statement",
                available=True,
                content_summary=f"Accounts: {f.bank_accounts_count}. "
                               f"Total credits consistent with declared income of ₹{f.income.total:,.0f}.",
                discrepancies=discrepancies,
            )

        elif doc_type == DocumentType.INVESTMENT_PROOFS:
            if f.deductions.section_80c > 150000:
                discrepancies.append(f"80C claims total ₹{f.deductions.section_80c:,.0f} — exceeds ₹1.5L limit")
            return DocumentResult(
                document_type="Investment Proofs",
                available=True,
                content_summary=f"80C investments: ₹{f.deductions.section_80c:,.0f}. "
                               f"80D premium: ₹{f.deductions.section_80d:,.0f}.",
                discrepancies=discrepancies,
            )

        elif doc_type == DocumentType.BUSINESS_BOOKS:
            if "shell_company_invoicing" in f.fraud_patterns:
                discrepancies.append("Multiple invoices from entities with matching addresses and directors")
                discrepancies.append("No evidence of actual goods/services delivered for high-value invoices")
            return DocumentResult(
                document_type="Business Books",
                available=f.taxpayer.employer_type == "self_employed",
                content_summary=f"Business income: ₹{f.income.business_profession:,.0f}.",
                discrepancies=discrepancies,
            )

        elif doc_type == DocumentType.RELATED_PARTY_RECORDS:
            if "shell_company_invoicing" in f.fraud_patterns or "suspicious_80g_donations" in f.fraud_patterns:
                discrepancies.append("Related party transactions detected with circular fund flows")
            return DocumentResult(
                document_type="Related Party Records",
                available=True,
                content_summary="Records of transactions with related parties and associated entities.",
                discrepancies=discrepancies,
            )

        return DocumentResult(
            document_type=doc_type.value,
            available=False,
            content_summary="Document type not recognized.",
            discrepancies=[],
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_observation(self) -> ITRObservation:
        """Build the current observation for the agent."""
        f = self._filing

        # Create ITR summary (what the agent sees — EXCLUDING hidden ground truth)
        itr_summary = {
            "filing_id": f.filing_id,
            "taxpayer": {
                "pan_category": f.taxpayer.pan_category,
                "filing_status": f.taxpayer.filing_status,
                "assessment_year": f.taxpayer.assessment_year,
                "age_bracket": f.taxpayer.age_bracket,
                "city_tier": f.taxpayer.city_tier,
                "employer_type": f.taxpayer.employer_type,
            },
            "income": {
                "salary": f.income.salary,
                "business_profession": f.income.business_profession,
                "house_property": f.income.house_property,
                "capital_gains_short": f.income.capital_gains_short,
                "capital_gains_long": f.income.capital_gains_long,
                "other_sources": f.income.other_sources,
                "total": f.income.total,
            },
            "deductions": {
                "section_80c": f.deductions.section_80c,
                "section_80d": f.deductions.section_80d,
                "section_80e": f.deductions.section_80e,
                "section_80g": f.deductions.section_80g,
                "hra_exemption": f.deductions.hra_exemption,
                "standard_deduction": f.deductions.standard_deduction,
                "other_deductions": f.deductions.other_deductions,
                "total": f.deductions.total,
            },
            "tds_entries": [
                {
                    "source": t.source,
                    "amount_credited": t.amount_credited,
                    "tds_deducted": t.tds_deducted,
                }
                for t in f.tds_entries
            ],
            "advance_tax_paid": f.advance_tax_paid,
            "self_assessment_tax": f.self_assessment_tax,
            "total_tax_liability": f.total_tax_liability,
            "refund_claimed": f.refund_claimed,
            "high_value_transactions": [
                {
                    "type": t.transaction_type,
                    "amount": t.amount,
                    "date": t.date,
                    "description": t.description,
                }
                for t in f.high_value_transactions
            ],
            "previous_years": [
                {
                    "assessment_year": py.assessment_year,
                    "total_income": py.total_income,
                    "tax_paid": py.tax_paid,
                    "refund_claimed": py.refund_claimed,
                }
                for py in f.previous_years
            ],
            "bank_accounts_count": f.bank_accounts_count,
        }

        return ITRObservation(
            itr_summary=itr_summary,
            last_action_result=self._last_action_result,
            investigation_results=self._investigation_results.copy(),
            document_results=self._document_results.copy(),
            flagged_anomalies=self._flagged_anomalies.copy(),
            step_number=self._state.step_count if self._state else 0,
            max_steps=self._task.MAX_STEPS if self._task else 10,
            task_description=self._task.TASK_DESCRIPTION if self._task else "",
        )

    def _compute_expected_tax(self) -> float:
        """Compute expected tax for cross-reference."""
        f = self._filing
        taxable = f.income.total - f.deductions.total
        # Simplified new regime
        if taxable <= 300000:
            return 0
        slabs = [
            (300000, 0), (700000, 0.05), (1000000, 0.10),
            (1200000, 0.15), (1500000, 0.20), (float("inf"), 0.30),
        ]
        tax = 0.0
        prev = 0
        for limit, rate in slabs:
            if taxable <= prev:
                break
            taxable_in_slab = min(taxable, limit) - prev
            tax += max(0, taxable_in_slab) * rate
            prev = limit
        return round(tax * 1.04, 2)
