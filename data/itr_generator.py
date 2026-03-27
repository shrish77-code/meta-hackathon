"""
Synthetic ITR Data Generator
Generates realistic Indian Income Tax Return filings with configurable fraud patterns.
Uses seeded randomness for reproducibility.
"""

import random
from typing import Dict, List, Optional, Tuple

from models import (
    Deductions,
    HighValueTransaction,
    ITRFiling,
    IncomeBreakdown,
    PreviousYearComparison,
    TDSEntry,
    TaxpayerProfile,
)


class ITRGenerator:
    """Generates synthetic ITR filings with optional fraud patterns."""

    EMPLOYER_NAMES = [
        "Tata Consultancy Services", "Infosys Ltd", "Wipro Technologies",
        "Reliance Industries", "HDFC Bank", "State Bank of India",
        "Bharti Airtel", "HCL Technologies", "ICICI Bank", "Bajaj Finance",
    ]

    PENNY_STOCK_NAMES = [
        "XYZ Micro Finance Ltd", "ABC Agro Industries",
        "PQR Trading Co", "MNO Infra Developers",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _rand(self, low: float, high: float) -> float:
        return round(self.rng.uniform(low, high), 2)

    def _rand_int(self, low: int, high: int) -> int:
        return self.rng.randint(low, high)

    # ── Profile Generation ────────────────────────────────────────────────

    def _generate_profile(self) -> TaxpayerProfile:
        return TaxpayerProfile(
            pan_category=self.rng.choice(["P", "P", "P", "P", "C", "F"]),
            filing_status=self.rng.choice(["Original", "Original", "Revised", "Belated"]),
            assessment_year=self.rng.choice(["AY 2025-26", "AY 2024-25"]),
            age_bracket=self.rng.choice(["below_60", "below_60", "below_60", "60_to_80", "above_80"]),
            city_tier=self.rng.choice(["metro", "metro", "tier_1", "tier_2", "rural"]),
            employer_type=self.rng.choice(["private", "private", "govt", "psu", "self_employed"]),
        )

    # ── Legitimate Filing ─────────────────────────────────────────────────

    def generate_legitimate(self) -> ITRFiling:
        """Generate a clean, legitimate ITR filing."""
        profile = self._generate_profile()
        salary = self._rand(400000, 2500000)

        income = IncomeBreakdown(
            salary=salary,
            business_profession=0.0 if profile.employer_type != "self_employed" else self._rand(100000, 800000),
            house_property=self._rand(-200000, 200000) if self.rng.random() > 0.6 else 0.0,
            capital_gains_short=self._rand(0, 50000) if self.rng.random() > 0.7 else 0.0,
            capital_gains_long=self._rand(0, 100000) if self.rng.random() > 0.8 else 0.0,
            other_sources=self._rand(5000, 80000),
        )

        # Legitimate deductions within limits
        deductions = Deductions(
            section_80c=min(self._rand(50000, 150000), 150000),
            section_80d=self._rand(5000, 25000),
            section_80e=self._rand(0, 50000) if self.rng.random() > 0.8 else 0.0,
            section_80g=self._rand(0, 20000) if self.rng.random() > 0.7 else 0.0,
            hra_exemption=self._rand(60000, min(salary * 0.4, 300000)) if profile.city_tier == "metro" else 0.0,
            other_deductions=self._rand(0, 20000) if self.rng.random() > 0.8 else 0.0,
        )

        # TDS consistent with salary
        tds_entries = [
            TDSEntry(
                source=self.rng.choice(self.EMPLOYER_NAMES),
                amount_credited=salary,
                tds_deducted=round(salary * self._rand(0.08, 0.20), 2),
                tds_deposited=round(salary * self._rand(0.08, 0.20), 2),
            )
        ]
        if income.other_sources > 10000:
            tds_entries.append(TDSEntry(
                source="Bank Interest TDS",
                amount_credited=income.other_sources,
                tds_deducted=round(income.other_sources * 0.10, 2),
                tds_deposited=round(income.other_sources * 0.10, 2),
            ))

        taxable = income.total - deductions.total
        tax_liability = self._compute_tax(taxable, profile.age_bracket)
        total_tds = sum(t.tds_deducted for t in tds_entries)

        return ITRFiling(
            taxpayer=profile,
            income=income,
            deductions=deductions,
            tds_entries=tds_entries,
            advance_tax_paid=max(0, tax_liability - total_tds - self._rand(0, 10000)),
            self_assessment_tax=self._rand(0, 5000),
            total_tax_liability=tax_liability,
            refund_claimed=max(0, total_tds - tax_liability + self._rand(0, 5000)),
            high_value_transactions=[],
            previous_years=self._generate_prev_years(income.total),
            bank_accounts_count=self._rand_int(1, 3),
            is_fraudulent=False,
            fraud_patterns=[],
            anomaly_details={},
        )

    # ── Fraudulent Filings ────────────────────────────────────────────────

    def generate_easy_fraud(self) -> ITRFiling:
        """Generate ITR with obvious fraud patterns."""
        filing = self.generate_legitimate()
        filing.is_fraudulent = True
        patterns = []
        anomalies: Dict[str, str] = {}

        # Pattern 1: Income mismatch — declared salary doesn't match TDS
        real_salary = filing.income.salary
        declared_salary = round(real_salary * self._rand(0.4, 0.65), 2)
        filing.income.salary = declared_salary
        patterns.append("income_mismatch")
        anomalies["income.salary"] = (
            f"Declared salary ₹{declared_salary:,.0f} but TDS records show "
            f"₹{real_salary:,.0f} credited by employer. "
            f"Discrepancy of ₹{real_salary - declared_salary:,.0f}."
        )

        # Pattern 2: 80C deduction exceeds ₹1.5L limit
        inflated_80c = self._rand(180000, 320000)
        filing.deductions.section_80c = inflated_80c
        patterns.append("80c_over_limit")
        anomalies["deductions.section_80c"] = (
            f"Section 80C deduction of ₹{inflated_80c:,.0f} exceeds "
            f"the statutory limit of ₹1,50,000."
        )

        filing.fraud_patterns = patterns
        filing.anomaly_details = anomalies
        return filing

    def generate_medium_fraud(self) -> ITRFiling:
        """Generate ITR with subtle fraud patterns requiring cross-referencing."""
        filing = self.generate_legitimate()
        filing.is_fraudulent = True
        patterns = []
        anomalies: Dict[str, str] = {}

        # Pattern 1: Phantom HRA — claims HRA but no rent receipts exist
        if filing.taxpayer.city_tier == "metro":
            hra = self._rand(250000, 500000)
        else:
            hra = self._rand(150000, 300000)
            filing.taxpayer.city_tier = "metro"  # Claims metro for higher HRA
        filing.deductions.hra_exemption = hra
        patterns.append("phantom_hra")
        anomalies["deductions.hra_exemption"] = (
            f"HRA exemption of ₹{hra:,.0f} claimed, but no rent receipts "
            f"are available. Rent paid exceeds 50% of salary which is unusual."
        )

        # Pattern 2: Suspicious LTCG through penny stocks
        penny_gain = self._rand(800000, 2500000)
        filing.income.capital_gains_long = penny_gain
        filing.high_value_transactions.append(HighValueTransaction(
            transaction_type="shares",
            amount=penny_gain,
            date="2024-12-15",
            description=f"Sale of {self.rng.choice(self.PENNY_STOCK_NAMES)} shares",
        ))
        patterns.append("penny_stock_ltcg")
        anomalies["income.capital_gains_long"] = (
            f"LTCG of ₹{penny_gain:,.0f} from penny stock transactions. "
            f"Classic pattern of round-tripping black money through "
            f"artificially inflated share prices."
        )

        # Pattern 3: Inconsistent tax computation
        filing.total_tax_liability = round(filing.total_tax_liability * 0.6, 2)
        patterns.append("tax_computation_error")
        anomalies["total_tax_liability"] = (
            f"Declared tax liability appears significantly lower than "
            f"what the income and deductions would compute to."
        )

        filing.fraud_patterns = patterns
        filing.anomaly_details = anomalies
        return filing

    def generate_hard_fraud(self) -> ITRFiling:
        """Generate ITR with complex, multi-layered fraud."""
        filing = self.generate_legitimate()
        filing.is_fraudulent = True
        patterns = []
        anomalies: Dict[str, str] = {}

        # Pattern 1: Shell company invoicing — business expenses to related parties
        filing.income.business_profession = self._rand(2000000, 5000000)
        filing.taxpayer.employer_type = "self_employed"
        ghost_expenses = self._rand(1500000, 3000000)
        # Hidden: business income is inflated with fake expenses to shell companies
        patterns.append("shell_company_invoicing")
        anomalies["income.business_profession"] = (
            f"Business income of ₹{filing.income.business_profession:,.0f} with "
            f"extremely high expenses (₹{ghost_expenses:,.0f}) routed to "
            f"entities with suspicious patterns. Cross-reference business "
            f"books to find related-party transactions."
        )

        # Pattern 2: Multi-year income shifting
        prev_years = []
        base_income = filing.income.total
        for i, ay in enumerate(["AY 2023-24", "AY 2022-23", "AY 2021-22"]):
            # Artificially varying income to dodge slab thresholds
            shifted = base_income * self._rand(0.3, 0.7) if i < 2 else base_income * 1.8
            prev_years.append(PreviousYearComparison(
                assessment_year=ay,
                total_income=round(shifted, 2),
                tax_paid=round(shifted * 0.1, 2),
                refund_claimed=self._rand(50000, 200000),
                major_deductions=self._rand(100000, 400000),
            ))
        filing.previous_years = prev_years
        patterns.append("multi_year_income_shifting")
        anomalies["previous_years"] = (
            f"Income shows abnormal year-to-year variation: fluctuating between "
            f"₹{prev_years[0].total_income:,.0f} and ₹{prev_years[2].total_income:,.0f}. "
            f"Pattern consistent with income shifting to stay below higher tax slabs."
        )

        # Pattern 3: Suspicious high-value transactions
        filing.high_value_transactions = [
            HighValueTransaction(
                transaction_type="cash_deposit",
                amount=self._rand(1000000, 3000000),
                date="2024-11-10",
                description="Large cash deposit in savings account",
            ),
            HighValueTransaction(
                transaction_type="property",
                amount=self._rand(5000000, 15000000),
                date="2024-08-20",
                description="Property purchase — value seems under-reported vs circle rate",
            ),
        ]
        patterns.append("high_value_cash_transactions")
        anomalies["high_value_transactions"] = (
            f"Multiple high-value transactions totaling "
            f"₹{sum(t.amount for t in filing.high_value_transactions):,.0f} "
            f"inconsistent with declared income of ₹{filing.income.total:,.0f}."
        )

        # Pattern 4: Excessive refund claim
        filing.refund_claimed = self._rand(300000, 800000)
        patterns.append("excessive_refund")
        anomalies["refund_claimed"] = (
            f"Refund claimed of ₹{filing.refund_claimed:,.0f} is disproportionate "
            f"to total tax liability and TDS deducted."
        )

        # Pattern 5: Related party donations for 80G
        filing.deductions.section_80g = self._rand(200000, 500000)
        patterns.append("suspicious_80g_donations")
        anomalies["deductions.section_80g"] = (
            f"Donations of ₹{filing.deductions.section_80g:,.0f} claimed under 80G "
            f"to entities that appear to be related parties."
        )

        filing.fraud_patterns = patterns
        filing.anomaly_details = anomalies
        filing.bank_accounts_count = self._rand_int(5, 12)
        return filing

    # ── Helpers ────────────────────────────────────────────────────────────

    def _compute_tax(self, taxable_income: float, age_bracket: str) -> float:
        """Simplified Indian income tax computation (new regime)."""
        if taxable_income <= 300000:
            return 0
        slabs = [
            (300000, 0),
            (700000, 0.05),
            (1000000, 0.10),
            (1200000, 0.15),
            (1500000, 0.20),
            (float("inf"), 0.30),
        ]
        tax = 0.0
        prev = 0
        for limit, rate in slabs:
            if taxable_income <= prev:
                break
            taxable_in_slab = min(taxable_income, limit) - prev
            tax += max(0, taxable_in_slab) * rate
            prev = limit
        # Cess
        tax *= 1.04
        return round(tax, 2)

    def _generate_prev_years(self, current_income: float) -> List[PreviousYearComparison]:
        """Generate plausible previous year data."""
        prev = []
        for ay in ["AY 2024-25", "AY 2023-24"]:
            variation = self._rand(0.85, 1.15)
            income = round(current_income * variation, 2)
            prev.append(PreviousYearComparison(
                assessment_year=ay,
                total_income=income,
                tax_paid=round(income * self._rand(0.08, 0.18), 2),
                refund_claimed=self._rand(0, 30000),
                major_deductions=self._rand(50000, 200000),
            ))
        return prev


# ── Factory functions for tasks ───────────────────────────────────────────────

def generate_task_scenario(
    difficulty: str, seed: int = 42
) -> Tuple[ITRFiling, ITRFiling]:
    """
    Generate a pair of (fraudulent, legitimate) ITR filings for a task.
    Returns: (target_filing, reference_filing)
    """
    gen = ITRGenerator(seed=seed)

    if difficulty == "easy":
        fraud = gen.generate_easy_fraud()
        legit = gen.generate_legitimate()
    elif difficulty == "medium":
        fraud = gen.generate_medium_fraud()
        legit = gen.generate_legitimate()
    elif difficulty == "hard":
        fraud = gen.generate_hard_fraud()
        legit = gen.generate_legitimate()
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    return fraud, legit
