---
title: ITR Fraud Detection
emoji: 🏦
colorFrom: red
colorTo: black
sdk: docker
pinned: false
---

# ITR Fraud Detection Environment 🏦🔍

An **OpenEnv-compliant** environment where AI agents learn to detect fraud in **Indian Income Tax Returns (ITR)**. Built for the Meta PyTorch × Hugging Face OpenEnv Hackathon.

## 🎯 What It Does

This environment simulates the work of a tax auditor at the Indian Income Tax Department. An AI agent:

1. **Reviews** an ITR filing (income, deductions, TDS, high-value transactions)
2. **Investigates** specific fields for irregularities
3. **Cross-references** data points to find discrepancies
4. **Requests** supporting documents (Form 16, rent receipts, bank statements)
5. **Flags** anomalies with reasoning and severity
6. **Renders** a final verdict: legitimate, suspicious, or fraudulent

This is a **real-world task** — not a game. Tax auditors perform these exact steps daily to detect billions in tax fraud.

---

## 🏛️ Project Relevance & The AI Auditor

### Why This Idea Matters
India's tax collection system is transitioning to a "Faceless Assessment" model. Manually auditing millions of ITR filings is physically impossible and prone to human error or bias. Our environment provides a training ground for **AI Auditors** that can:
- **Scalability**: Process thousands of returns per minute.
- **Consistency**: Apply the same audit logic to every citizen fairly.
- **Complexity**: Identify multi-layered "layering" techniques used in money laundering and tax evasion.

### Role of Large Language Models (LLMs)
Modern LLMs (like GPT-4o, Gemini 1.5 Pro) are uniquely suited for this task because:
- **Tax Law Training**: These models have read thousands of pages of tax codes, case law, and IT Department circulars (Section 80C, 80D, HRA rules, etc.).
- **Pattern Recognition**: They are excellent at spotting inconsistencies between text-heavy descriptions (standard business books) and numerical data.
- **Reasoning**: Unlike traditional static rule-engines, an AI Agent can *decide* which document to ask for next based on previous findings, mimicking the curiosity of a human auditor.
- **Explainability**: They don't just flag fraud; they explain *why* it's fraud in natural language, making them perfect assistants for human review.

---

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

### Run the Baseline Agent (No API Key Needed)

```bash
python inference.py --local
```

### Run Inference Script Required by Hackathon

```bash
export HF_TOKEN="your-huggingface-or-openai-key-here"
export API_BASE_URL="https://router.huggingface.co/v1" # (Optional: defaults to OpenAI)
export MODEL_NAME="gpt-4o-mini" # (Optional: defaults to gpt-4o-mini)
python inference.py
```

### Start the Server

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Use the Client

```python
from client import ITRFraudEnv
from models import ITRAction, ActionType, VerdictType

with ITRFraudEnv(base_url="http://localhost:8000") as client:
    # Reset for easy task
    obs = client.reset(task_id="easy")
    
    # Investigate salary
    result = client.step(ITRAction(
        action_type=ActionType.INVESTIGATE_FIELD,
        field_name="income.salary"
    ))
    
    # Render verdict
    result = client.step(ITRAction(
        action_type=ActionType.RENDER_VERDICT,
        verdict=VerdictType.FRAUDULENT,
        confidence=0.9,
        explanation="Income mismatch with TDS records"
    ))
```

---

## 📋 Tasks

| Task | Difficulty | Fraud Patterns | Max Steps |
|------|-----------|----------------|-----------|
| **Obvious Red Flags** | Easy | Income mismatch, 80C over-limit | 10 |
| **The Subtle Cheat** | Medium | Phantom HRA, penny stock LTCG, tax computation error | 15 |
| **The Elaborate Scheme** | Hard | Shell companies, income shifting, cash deposits, fake donations | 25 |

Each task has a programmatic grader that scores performance from **0.0 to 1.0** based on:
- Anomaly detection accuracy
- Verdict correctness
- Investigation completeness
- Efficiency (fewer steps = better)

---

## 🎮 Action Space

| Action | Description | Key Parameters |
|--------|-------------|----------------|
| `investigate_field` | Drill into a specific field | `field_name` |
| `cross_reference` | Compare two data points | `field_a`, `field_b` |
| `request_document` | Request supporting docs | `document_type` |
| `flag_anomaly` | Flag suspicious finding | `anomaly_field`, `anomaly_reason`, `anomaly_severity` |
| `render_verdict` | Final decision | `verdict`, `confidence`, `explanation` |

### Available Document Types
- `form_16` — Employer TDS certificate
- `bank_statement` — Bank account records
- `rent_receipts` — For HRA verification
- `investment_proofs` — 80C/80D proofs
- `capital_gains_statement` — Stock trading records
- `business_books` — Business accounting
- `related_party_records` — Related entity transactions

---

## 📊 Observation Space

Each observation contains:
- **ITR Summary**: Full tax return data (income, deductions, TDS, transactions, previous years)
- **Last Action Result**: What happened from the last action
- **Investigation Results**: Findings from investigations
- **Document Results**: Requested documents and any discrepancies
- **Flagged Anomalies**: List of anomalies the agent has flagged
- **Step Info**: Current step number, max steps, task description

---

## 🏆 Reward Function

| Component | Value | Description |
|-----------|-------|-------------|
| Correct anomaly flag | +0.10 | Flagging a real irregularity |
| Useful investigation | +0.05 | Investigating a relevant field |
| Useful cross-reference | +0.08 | Finding a real discrepancy |
| Useful document | +0.06 | Document reveals discrepancies |
| Correct verdict | +0.30 | Right final call |
| Efficiency bonus | up to +0.05 | Fewer steps = more reward |
| False flag | -0.05 | Flagging a non-issue |
| Redundant action | -0.02 | Repeating an investigation |
| Timeout | -0.10 | Running out of steps |

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t itr-fraud-env -f server/Dockerfile .

# Run
docker run -p 8000:8000 itr-fraud-env

# Test
curl http://localhost:8000/health
```

---

## 📁 Project Structure

```
├── models.py              # Pydantic Action/Observation/State models
├── client.py              # HTTP client (ITRFraudEnv)
├── inference.py           # Baseline inference script
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Package configuration
├── tasks/
│   ├── task_easy.py       # Easy: Obvious red flags
│   ├── task_medium.py     # Medium: Subtle inconsistencies
│   └── task_hard.py       # Hard: Multi-layered fraud
├── data/
│   └── itr_generator.py   # Synthetic ITR data generator
└── server/
    ├── itr_environment.py # Core environment logic
    ├── app.py             # FastAPI server
    ├── requirements.txt   # Server dependencies
    └── Dockerfile         # Container image
```

---

## 🔧 OpenEnv API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode (`{"task_id": "easy"}`) |
| `/step` | POST | Execute action |
| `/state` | GET | Get episode state |
| `/health` | GET | Health check |
| `/info` | GET | Environment metadata |

---

## 📜 License

MIT
