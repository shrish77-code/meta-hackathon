"""
ITR Fraud Detection Environment - Pydantic Models
Defines typed Action, Observation, and State models for OpenEnv compliance.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    INVESTIGATE_FIELD = "investigate_field"
    CROSS_REFERENCE = "cross_reference"
    REQUEST_DOCUMENT = "request_document"
    FLAG_ANOMALY = "flag_anomaly"
    RENDER_VERDICT = "render_verdict"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VerdictType(str, Enum):
    LEGITIMATE = "legitimate"
    SUSPICIOUS = "suspicious"
    FRAUDULENT = "fraudulent"


class DocumentType(str, Enum):
    FORM_16 = "form_16"
    BANK_STATEMENT = "bank_statement"
    RENT_RECEIPTS = "rent_receipts"
    INVESTMENT_PROOFS = "investment_proofs"
    CAPITAL_GAINS_STATEMENT = "capital_gains_statement"
    BUSINESS_BOOKS = "business_books"
    RELATED_PARTY_RECORDS = "related_party_records"


# ─── ITR Data Models ─────────────────────────────────────────────────────────

class IncomeBreakdown(BaseModel):
    salary: float = Field(description="Income from salary")
    business_profession: float = Field(description="Income from business/profession")
    house_property: float = Field(description="Income from house property")
    capital_gains_short: float = Field(description="Short-term capital gains")
    capital_gains_long: float = Field(description="Long-term capital gains")
    other_sources: float = Field(description="Income from other sources (interest, dividends, etc.)")

    @property
    def total(self) -> float:
        return (
            self.salary + self.business_profession + self.house_property
            + self.capital_gains_short + self.capital_gains_long + self.other_sources
        )


class Deductions(BaseModel):
    section_80c: float = Field(description="80C deductions (PPF, ELSS, LIC, etc.) - limit ₹1.5L")
    section_80d: float = Field(description="80D medical insurance - limit ₹25K/₹50K")
    section_80e: float = Field(description="80E education loan interest")
    section_80g: float = Field(description="80G donations")
    hra_exemption: float = Field(description="HRA exemption claimed")
    standard_deduction: float = Field(default=50000.0, description="Standard deduction ₹50K")
    other_deductions: float = Field(description="Other deductions")

    @property
    def total(self) -> float:
        return (
            self.section_80c + self.section_80d + self.section_80e
            + self.section_80g + self.hra_exemption + self.standard_deduction
            + self.other_deductions
        )


class TDSEntry(BaseModel):
    source: str = Field(description="TDS source (employer, bank, tenant, etc.)")
    amount_credited: float = Field(description="Amount on which TDS was deducted")
    tds_deducted: float = Field(description="TDS amount deducted")
    tds_deposited: float = Field(description="TDS deposited with govt")


class HighValueTransaction(BaseModel):
    transaction_type: str = Field(description="Type: cash_deposit, property, shares, etc.")
    amount: float
    date: str
    description: str


class PreviousYearComparison(BaseModel):
    assessment_year: str
    total_income: float
    tax_paid: float
    refund_claimed: float
    major_deductions: float


class TaxpayerProfile(BaseModel):
    pan_category: str = Field(description="P=Individual, C=Company, F=Firm, etc.")
    filing_status: str = Field(description="Original/Revised/Belated")
    assessment_year: str
    age_bracket: str = Field(description="below_60, 60_to_80, above_80")
    city_tier: str = Field(description="metro, tier_1, tier_2, rural")
    employer_type: str = Field(description="govt, psu, private, self_employed")


class ITRFiling(BaseModel):
    """Complete ITR filing data that the agent will review."""
    filing_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taxpayer: TaxpayerProfile
    income: IncomeBreakdown
    deductions: Deductions
    tds_entries: List[TDSEntry]
    advance_tax_paid: float
    self_assessment_tax: float
    total_tax_liability: float
    refund_claimed: float
    high_value_transactions: List[HighValueTransaction]
    previous_years: List[PreviousYearComparison]
    bank_accounts_count: int
    is_fraudulent: bool = Field(exclude=True, description="Ground truth - hidden from agent")
    fraud_patterns: List[str] = Field(exclude=True, description="List of fraud indicators - hidden")
    anomaly_details: Dict[str, str] = Field(exclude=True, description="Field-level anomaly explanations")


# ─── OpenEnv Models ──────────────────────────────────────────────────────────

class ITRAction(BaseModel):
    """Action the agent can take in the ITR fraud detection environment."""
    action_type: ActionType = Field(description="Type of action to perform")

    # For investigate_field
    field_name: Optional[str] = Field(default=None, description="Field to investigate (e.g., 'income.salary', 'deductions.section_80c')")

    # For cross_reference
    field_a: Optional[str] = Field(default=None, description="First field to cross-reference")
    field_b: Optional[str] = Field(default=None, description="Second field to cross-reference")

    # For request_document
    document_type: Optional[DocumentType] = Field(default=None, description="Type of document to request")

    # For flag_anomaly
    anomaly_field: Optional[str] = Field(default=None, description="Field where anomaly was detected")
    anomaly_reason: Optional[str] = Field(default=None, description="Explanation of why this is anomalous")
    anomaly_severity: Optional[Severity] = Field(default=None, description="Severity of the anomaly")

    # For render_verdict
    verdict: Optional[VerdictType] = Field(default=None, description="Final verdict")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in verdict (0-1)")
    explanation: Optional[str] = Field(default=None, description="Explanation for the verdict")


class InvestigationResult(BaseModel):
    """Result of an investigation or cross-reference action."""
    finding: str = Field(description="What was found")
    suspicious: bool = Field(description="Whether this finding is suspicious")
    detail: str = Field(description="Detailed explanation")


class DocumentResult(BaseModel):
    """Result of a document request."""
    document_type: str
    available: bool
    content_summary: str
    discrepancies: List[str] = Field(default_factory=list)


class ITRObservation(BaseModel):
    """What the agent observes after each action."""
    itr_summary: Dict[str, Any] = Field(description="Current ITR data visible to the agent")
    last_action_result: Optional[str] = Field(default=None, description="Result of the last action taken")
    investigation_results: List[InvestigationResult] = Field(default_factory=list)
    document_results: List[DocumentResult] = Field(default_factory=list)
    flagged_anomalies: List[Dict[str, str]] = Field(default_factory=list)
    step_number: int = Field(default=0)
    max_steps: int = Field(default=10)
    available_actions: List[str] = Field(
        default_factory=lambda: [a.value for a in ActionType]
    )
    task_description: str = Field(default="", description="Description of the current task")


class ITRState(BaseModel):
    """Episode state tracking."""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = Field(default=0)
    task_id: str = Field(default="easy")
    is_done: bool = Field(default=False)
    current_score: float = Field(default=0.0)


class ITRReward(BaseModel):
    """Reward breakdown for transparency."""
    total: float = Field(description="Total reward for this step")
    anomaly_reward: float = Field(default=0.0, description="Reward from correct anomaly flags")
    investigation_reward: float = Field(default=0.0, description="Reward from useful investigations")
    penalty: float = Field(default=0.0, description="Penalties for false flags, redundant actions")
    verdict_reward: float = Field(default=0.0, description="Reward for correct final verdict")
    efficiency_bonus: float = Field(default=0.0, description="Bonus for efficient investigation")


class ITRStepResult(BaseModel):
    """Result of a step in the environment."""
    observation: ITRObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    reward_breakdown: Optional[ITRReward] = None
