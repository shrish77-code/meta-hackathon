"""
Baseline Inference Script for ITR Fraud Detection Environment.

Uses the Google Gemini API to run a model against all 3 tasks.
Produces reproducible baseline scores.

Usage:
    export GEMINI_API_KEY="your-key"
    python baseline.py

Or run locally without API:
    python baseline.py --local
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# Attempt to load from .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))

from models import ActionType, ITRAction, DocumentType, Severity, VerdictType
from server.itr_environment import ITREnvironment


def run_gemini_agent(env: ITREnvironment, task_id: str) -> Dict[str, Any]:
    """Run Google Gemini-powered agent on a task."""
    try:
        from google import genai
    except ImportError:
        print("⚠️  google-genai package not installed. Run: pip install google-genai")
        return run_heuristic_agent(env, task_id)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set. Falling back to heuristic agent.")
        return run_heuristic_agent(env, task_id)

    client = genai.Client(api_key=api_key)
    obs = env.reset(task_id=task_id)

    system_prompt = """You are an expert income tax auditor for the Indian Income Tax Department.
You are reviewing an ITR (Income Tax Return) filing for potential fraud.

You MUST respond with a valid JSON action. Available action types:
1. investigate_field - Investigate a specific field. Params: field_name (e.g., "income.salary", "deductions.section_80c")
2. cross_reference - Cross-check two fields. Params: field_a, field_b (e.g., "salary", "tds")
3. request_document - Request a document. Params: document_type (form_16, bank_statement, rent_receipts, investment_proofs, capital_gains_statement, business_books, related_party_records)
4. flag_anomaly - Flag a suspicious field. Params: anomaly_field, anomaly_reason, anomaly_severity (low/medium/high/critical)
5. render_verdict - Final decision. Params: verdict (legitimate/suspicious/fraudulent), confidence (0-1), explanation

Respond ONLY with a single JSON object, no extra text. Examples:
{"action_type": "investigate_field", "field_name": "income.salary"}
{"action_type": "flag_anomaly", "anomaly_field": "deductions.section_80c", "anomaly_reason": "Exceeds 1.5L limit", "anomaly_severity": "high"}
{"action_type": "render_verdict", "verdict": "fraudulent", "confidence": 0.9, "explanation": "Multiple anomalies found"}
"""

    steps = []
    done = False
    total_reward = 0.0

    while not done:
        user_msg = json.dumps({
            "step": obs.step_number,
            "max_steps": obs.max_steps,
            "task": obs.task_description,
            "itr_data": obs.itr_summary,
            "last_result": obs.last_action_result,
            "flagged_so_far": obs.flagged_anomalies,
            "investigations": [r.model_dump() for r in obs.investigation_results[-3:]],
        }, indent=2, default=str)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                {"role": "user", "parts": [{"text": system_prompt + "\n\nCurrent ITR data:\n" + user_msg}]},
            ],
            config={
                "temperature": 0.1,
                "max_output_tokens": 1000,
                "response_mime_type": "application/json",
            },
        )

        action_text = response.text.strip()
        # Parse JSON from response — try multiple strategies
        action_data = None
        # Strategy 1: Direct parse (works when response_mime_type is set)
        try:
            action_data = json.loads(action_text)
        except (json.JSONDecodeError, TypeError):
            pass
        # Strategy 2: Extract from markdown code block
        if action_data is None and "```" in action_text:
            try:
                block = action_text.split("```")[1]
                if block.startswith("json"):
                    block = block[4:]
                action_data = json.loads(block.strip())
            except (json.JSONDecodeError, TypeError, IndexError):
                pass
        # Strategy 3: Find first { ... } using simple scan
        if action_data is None:
            import re
            match = re.search(r'\{[^{}]*\}', action_text, re.DOTALL)
            if match:
                try:
                    action_data = json.loads(match.group())
                except (json.JSONDecodeError, TypeError):
                    pass
        # Final fallback
        if action_data is None:
            print(f"\n[DEBUG] Could not parse JSON! Raw model output:\n{action_text}\n")
            action_data = {
                "action_type": "render_verdict",
                "verdict": "suspicious",
                "confidence": 0.5,
                "explanation": "Unable to determine",
            }

        # Build action
        action = ITRAction(**action_data)
        result = env.step(action)
        obs = result.observation
        done = result.done
        total_reward += result.reward

        steps.append({
            "step": obs.step_number,
            "action": action_data,
            "reward": result.reward,
            "done": done,
        })

    return {
        "task_id": task_id,
        "steps": steps,
        "total_reward": total_reward,
        "final_score": result.info.get("final_score", 0.0),
        "num_steps": len(steps),
    }


def run_heuristic_agent(env: ITREnvironment, task_id: str) -> Dict[str, Any]:
    """
    Rule-based baseline agent that follows a systematic audit procedure.
    This provides reproducible scores without requiring an API key.
    """
    obs = env.reset(task_id=task_id)
    done = False
    total_reward = 0.0
    steps = []

    # Step 1: Investigate salary
    action = ITRAction(
        action_type=ActionType.INVESTIGATE_FIELD,
        field_name="income.salary",
    )
    result = env.step(action)
    obs = result.observation
    done = result.done
    total_reward += result.reward
    steps.append({"action": "investigate income.salary", "reward": result.reward})

    if done:
        return _build_result(task_id, steps, total_reward, result)

    # Step 2: Cross-reference salary with TDS
    action = ITRAction(
        action_type=ActionType.CROSS_REFERENCE,
        field_a="salary",
        field_b="tds",
    )
    result = env.step(action)
    obs = result.observation
    done = result.done
    total_reward += result.reward
    steps.append({"action": "cross_reference salary↔tds", "reward": result.reward})

    if done:
        return _build_result(task_id, steps, total_reward, result)

    # Step 3: Request Form 16
    action = ITRAction(
        action_type=ActionType.REQUEST_DOCUMENT,
        document_type=DocumentType.FORM_16,
    )
    result = env.step(action)
    obs = result.observation
    done = result.done
    total_reward += result.reward
    steps.append({"action": "request Form 16", "reward": result.reward})

    if done:
        return _build_result(task_id, steps, total_reward, result)

    # Step 4: Investigate deductions
    action = ITRAction(
        action_type=ActionType.INVESTIGATE_FIELD,
        field_name="deductions.section_80c",
    )
    result = env.step(action)
    obs = result.observation
    done = result.done
    total_reward += result.reward
    steps.append({"action": "investigate deductions.section_80c", "reward": result.reward})

    if done:
        return _build_result(task_id, steps, total_reward, result)

    # Step 5: Check for suspicious findings — flag anomalies based on observations
    for inv in obs.investigation_results:
        if inv.suspicious and not done:
            action = ITRAction(
                action_type=ActionType.FLAG_ANOMALY,
                anomaly_field=inv.finding.split(":")[0] if ":" in inv.finding else "detected_field",
                anomaly_reason=inv.detail[:200],
                anomaly_severity=Severity.HIGH,
            )
            result = env.step(action)
            obs = result.observation
            done = result.done
            total_reward += result.reward
            steps.append({"action": f"flag_anomaly: {inv.finding[:50]}", "reward": result.reward})
            if done:
                return _build_result(task_id, steps, total_reward, result)

    # Check document discrepancies
    for doc in obs.document_results:
        for disc in doc.discrepancies:
            if not done:
                action = ITRAction(
                    action_type=ActionType.FLAG_ANOMALY,
                    anomaly_field=doc.document_type,
                    anomaly_reason=disc[:200],
                    anomaly_severity=Severity.HIGH,
                )
                result = env.step(action)
                obs = result.observation
                done = result.done
                total_reward += result.reward
                steps.append({"action": f"flag_anomaly from doc: {disc[:40]}", "reward": result.reward})
                if done:
                    return _build_result(task_id, steps, total_reward, result)

    # Additional investigations for medium/hard tasks
    if task_id in ("medium", "hard") and not done:
        for field in ["deductions.hra_exemption", "income.capital_gains_long", "high_value_transactions", "previous_years"]:
            if not done:
                action = ITRAction(
                    action_type=ActionType.INVESTIGATE_FIELD,
                    field_name=field,
                )
                result = env.step(action)
                obs = result.observation
                done = result.done
                total_reward += result.reward
                steps.append({"action": f"investigate {field}", "reward": result.reward})

                if result.observation.investigation_results and result.observation.investigation_results[-1].suspicious and not done:
                    inv = result.observation.investigation_results[-1]
                    action = ITRAction(
                        action_type=ActionType.FLAG_ANOMALY,
                        anomaly_field=field,
                        anomaly_reason=inv.detail[:200],
                        anomaly_severity=Severity.HIGH,
                    )
                    result = env.step(action)
                    obs = result.observation
                    done = result.done
                    total_reward += result.reward
                    steps.append({"action": f"flag {field}", "reward": result.reward})
                    if done:
                        return _build_result(task_id, steps, total_reward, result)

    # Hard task: additional document requests
    if task_id == "hard" and not done:
        for doc_type in [DocumentType.BANK_STATEMENT, DocumentType.BUSINESS_BOOKS, DocumentType.RELATED_PARTY_RECORDS]:
            if not done:
                action = ITRAction(
                    action_type=ActionType.REQUEST_DOCUMENT,
                    document_type=doc_type,
                )
                result = env.step(action)
                obs = result.observation
                done = result.done
                total_reward += result.reward
                steps.append({"action": f"request {doc_type.value}", "reward": result.reward})

                if result.observation.document_results and result.observation.document_results[-1].discrepancies and not done:
                    doc = result.observation.document_results[-1]
                    for disc in doc.discrepancies:
                        if not done:
                            action = ITRAction(
                                action_type=ActionType.FLAG_ANOMALY,
                                anomaly_field=doc.document_type,
                                anomaly_reason=disc[:200],
                                anomaly_severity=Severity.CRITICAL,
                            )
                            result = env.step(action)
                            obs = result.observation
                            done = result.done
                            total_reward += result.reward
                            steps.append({"action": f"flag from {doc.document_type}", "reward": result.reward})
                            if done:
                                return _build_result(task_id, steps, total_reward, result)

    # Final verdict
    if not done:
        # Decide verdict based on anomalies found
        has_anomalies = len(obs.flagged_anomalies) > 0
        has_suspicious = any(
            inv.suspicious for inv in obs.investigation_results
        )

        if has_anomalies or has_suspicious:
            verdict = VerdictType.FRAUDULENT
            confidence = min(0.95, 0.5 + len(obs.flagged_anomalies) * 0.15)
        else:
            verdict = VerdictType.LEGITIMATE
            confidence = 0.7

        action = ITRAction(
            action_type=ActionType.RENDER_VERDICT,
            verdict=verdict,
            confidence=confidence,
            explanation=f"Based on {len(obs.flagged_anomalies)} anomalies detected and {len(obs.investigation_results)} investigations.",
        )
        result = env.step(action)
        obs = result.observation
        done = result.done
        total_reward += result.reward
        steps.append({"action": f"verdict: {verdict.value}", "reward": result.reward})

    return _build_result(task_id, steps, total_reward, result)


def _build_result(task_id, steps, total_reward, result):
    return {
        "task_id": task_id,
        "steps": steps,
        "total_reward": total_reward,
        "final_score": result.info.get("final_score", 0.0),
        "num_steps": len(steps),
    }


def main():
    parser = argparse.ArgumentParser(description="ITR Fraud Detection Baseline")
    parser.add_argument("--local", action="store_true", help="Use local heuristic agent (no API key needed)")
    parser.add_argument("--task", type=str, default=None, help="Run specific task (easy/medium/hard)")
    args = parser.parse_args()

    # Fix Windows encoding
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    env = ITREnvironment()
    tasks = [args.task] if args.task else ["easy", "medium", "hard"]

    print("=" * 70)
    print("  ITR FRAUD DETECTION - BASELINE INFERENCE")
    print("=" * 70)

    all_scores = {}

    for task_id in tasks:
        print(f"\n{'-' * 50}")
        print(f"  Task: {task_id.upper()}")
        print(f"{'-' * 50}")

        if args.local:
            result = run_heuristic_agent(env, task_id)
        else:
            result = run_gemini_agent(env, task_id)

        score = result["final_score"]
        all_scores[task_id] = score

        print(f"\n  Steps taken: {result['num_steps']}")
        print(f"  Total reward: {result['total_reward']:.4f}")
        print(f"  Final score:  {score:.4f}")

        # Print step-by-step
        print(f"\n  Step-by-step:")
        for i, step in enumerate(result["steps"]):
            action_str = step.get("action", str(step.get("action", "")))
            print(f"    {i+1}. {action_str} (reward: {step['reward']:+.4f})")

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for task_id, score in all_scores.items():
        bar = "#" * int(score * 30) + "." * (30 - int(score * 30))
        print(f"  {task_id:8s} [{bar}] {score:.4f}")
    avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0
    print(f"\n  Average Score: {avg:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
