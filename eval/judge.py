"""
judge.py
--------
"LLM-as-judge" quality scoring: ask a model to rate another model's response
on a 1-5 helpfulness rubric.

If a real provider (OpenAI/Anthropic) is available, this uses it as the judge
— a real judge call. If not, it degrades automatically to a heuristic judge
based on response length, structure, and keyword density, so the pipeline
never breaks and the demo works with zero API keys.

This graceful-degradation pattern (try the real thing, fall back to a
reasonable heuristic) is itself worth pointing to in an interview — it's how
production systems stay usable when a dependency isn't configured, rather
than hard-failing.
"""

from eval.providers.base import BaseProvider, GenerationResult


def heuristic_judge(response_text: str, expected_keywords: list[str] | None) -> int:
    """Rough 1-5 quality proxy when no real judge model is configured.
    Rewards reasonable length and keyword relevance; penalizes very short or
    very rambling answers."""
    word_count = len(response_text.split())
    keyword_hits = 0
    if expected_keywords:
        text_lower = response_text.lower()
        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)

    score = 2  # baseline
    if 8 <= word_count <= 80:
        score += 1
    if keyword_hits > 0:
        score += min(2, keyword_hits)
    if word_count > 150:
        score -= 1  # rambling penalty

    return max(1, min(5, score))


def llm_judge(
    judge_provider: BaseProvider | None,
    prompt: str,
    response_text: str,
    expected_keywords: list[str] | None,
) -> tuple[int, str]:
    """Returns (score_1_to_5, judge_method). Tries a real judge provider first,
    falls back to the heuristic. `judge_method` tells the report which path
    was actually used, for transparency."""
    if judge_provider is not None and judge_provider.is_available():
        rubric_prompt = (
            f"Rate the following response to the prompt on a scale of 1-5 for "
            f"helpfulness. Respond with ONLY the number.\n\n"
            f"Prompt: {prompt}\n\nResponse: {response_text}"
        )
        try:
            result: GenerationResult = judge_provider.generate(rubric_prompt)
            digits = "".join(c for c in result.text if c.isdigit())
            if digits:
                score = max(1, min(5, int(digits[0])))
                return score, f"llm-judge:{judge_provider.name}"
        except Exception:
            pass  # fall through to heuristic on any API failure

    return heuristic_judge(response_text, expected_keywords), "heuristic-fallback"
