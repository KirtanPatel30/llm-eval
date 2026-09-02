"""
harness.py
----------
Orchestrates a full evaluation run: for every active provider, run every test
case, score it on every applicable dimension, and assemble a structured
report. This is the piece both the CLI (cli.py) and the API (app/main.py)
call into — one code path, two ways to trigger it.
"""

import time
from datetime import datetime, timezone

from eval.judge import llm_judge
from eval.providers.base import BaseProvider
from eval.providers.registry import get_active_providers
from eval.scorers import hallucination_score, keyword_coverage_score, overall_score, safety_score
from eval.testsuite import load_test_cases


def _pick_judge(providers: list[BaseProvider]) -> BaseProvider | None:
    """Prefer a real provider as judge if one is active; otherwise None
    (judge.py will fall back to the heuristic judge automatically)."""
    for p in providers:
        if p.name not in ("demo-fast", "demo-balanced", "demo-quality"):
            return p
    return None


def run_evaluation(provider_names: list[str] | None = None) -> dict:
    all_providers = get_active_providers()
    if provider_names:
        providers = [p for p in all_providers if p.name in provider_names]
    else:
        providers = all_providers

    judge_provider = _pick_judge(all_providers)
    test_cases = load_test_cases()

    results = []
    for provider in providers:
        for case in test_cases:
            gen = provider.generate(
                prompt=case["prompt"],
                context=case.get("context"),
                expected_keywords=case.get("expected_keywords"),
            )

            kw_score = keyword_coverage_score(gen.text, case.get("expected_keywords") or [])
            hallu_score = hallucination_score(gen.text, case.get("context"))
            safe_score = safety_score(gen.text, case.get("is_unsafe", False))
            composite = overall_score(kw_score, hallu_score, safe_score)
            judge_score, judge_method = llm_judge(
                judge_provider, case["prompt"], gen.text, case.get("expected_keywords")
            )

            results.append(
                {
                    "test_id": case["id"],
                    "category": case["category"],
                    "provider": provider.name,
                    "prompt": case["prompt"],
                    "response": gen.text,
                    "latency_ms": gen.latency_ms,
                    "cost_usd": gen.cost_usd,
                    "keyword_coverage": kw_score,
                    "hallucination_rate": hallu_score,
                    "safety_score": safe_score,
                    "judge_score": judge_score,
                    "judge_method": judge_method,
                    "overall_score": composite,
                }
            )

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "providers_evaluated": [p.name for p in providers],
        "n_test_cases": len(test_cases),
        "n_results": len(results),
        "results": results,
        "summary": _summarize(results, providers),
    }
    return report


def _summarize(results: list[dict], providers: list[BaseProvider]) -> list[dict]:
    summary = []
    for provider in providers:
        provider_results = [r for r in results if r["provider"] == provider.name]
        if not provider_results:
            continue

        def _avg(key):
            values = [r[key] for r in provider_results if r[key] is not None]
            return round(sum(values) / len(values), 3) if values else None

        safety_cases = [r for r in provider_results if r["safety_score"] is not None]
        safety_pass_rate = (
            round(sum(r["safety_score"] for r in safety_cases) / len(safety_cases), 3)
            if safety_cases
            else None
        )

        summary.append(
            {
                "provider": provider.name,
                "avg_keyword_coverage": _avg("keyword_coverage"),
                "avg_hallucination_rate": _avg("hallucination_rate"),
                "safety_pass_rate": safety_pass_rate,
                "avg_judge_score": _avg("judge_score"),
                "avg_overall_score": _avg("overall_score"),
                "avg_latency_ms": _avg("latency_ms"),
                "total_cost_usd": round(sum(r["cost_usd"] for r in provider_results), 6),
            }
        )
    return summary
