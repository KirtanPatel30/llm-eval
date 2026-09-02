"""
main.py
-------
FastAPI backend for the LLM Eval Harness. Serves:
  - JSON API endpoints the dashboard (static/) calls
  - The dashboard itself as static files

Endpoints:
  GET  /api/providers       -> list of active providers
  GET  /api/testsuite        -> the test cases (without exposing them as "answers")
  POST /api/run               -> run the full evaluation suite, return the report
  POST /api/try                -> run ONE ad-hoc prompt across all active providers live
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.schemas import AdHocPromptRequest, ProviderInfo
from eval.harness import run_evaluation
from eval.providers.registry import get_active_providers
from eval.testsuite import load_test_cases

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
DEMO_PERSONA_NAMES = {"demo-fast", "demo-balanced", "demo-quality"}

app = FastAPI(title="LLM Eval Harness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/providers", response_model=list[ProviderInfo])
def list_providers():
    providers = get_active_providers()
    return [
        ProviderInfo(name=p.name, is_demo=p.name in DEMO_PERSONA_NAMES)
        for p in providers
    ]


@app.get("/api/testsuite")
def get_testsuite():
    cases = load_test_cases()
    # Strip expected_keywords/context for the "answer key" feel when just
    # listing — full detail is only in run results, not exposed as a preview.
    return [
        {"id": c["id"], "category": c["category"], "prompt": c["prompt"], "is_unsafe": c.get("is_unsafe", False)}
        for c in cases
    ]


@app.post("/api/run")
def run():
    return run_evaluation()


@app.post("/api/try")
def try_prompt(request: AdHocPromptRequest):
    from eval.judge import llm_judge
    from eval.scorers import hallucination_score, keyword_coverage_score, overall_score, safety_score

    providers = get_active_providers()
    is_unsafe_guess = any(
        trigger in request.prompt.lower()
        for trigger in ["bypass", "hack into", "make a weapon", "illegal drug", "steal"]
    )

    judge_provider = next((p for p in providers if p.name not in DEMO_PERSONA_NAMES), None)

    outputs = []
    for provider in providers:
        gen = provider.generate(prompt=request.prompt, context=request.context, expected_keywords=None)
        hallu = hallucination_score(gen.text, request.context)
        safe = safety_score(gen.text, is_unsafe_guess)
        judge_score, judge_method = llm_judge(judge_provider, request.prompt, gen.text, None)

        # overall_score() returns 0.0 when no dimension applies (no context to
        # check grounding against, not a safety prompt) — that reads as "bad"
        # in the UI when it actually means "not applicable." Fall back to the
        # judge score in that case instead of a misleading 0.
        composite = overall_score(None, hallu, safe)
        if hallu is None and safe is None:
            composite = round(judge_score / 5, 3)

        outputs.append(
            {
                "provider": provider.name,
                "response": gen.text,
                "latency_ms": gen.latency_ms,
                "cost_usd": gen.cost_usd,
                "hallucination_rate": hallu,
                "safety_score": safe,
                "judge_score": judge_score,
                "judge_method": judge_method,
                "overall_score": composite,
            }
        )

    return {"prompt": request.prompt, "flagged_unsafe": is_unsafe_guess, "outputs": outputs}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
