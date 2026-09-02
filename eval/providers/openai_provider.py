"""
openai_provider.py
-------------------
Real OpenAI adapter. Only activates if OPENAI_API_KEY is set in the
environment — otherwise is_available() returns False and the registry skips
it in favor of the demo personas. Uses raw HTTP via httpx rather than the
official SDK to keep the dependency footprint small.
"""

import os
import time

import httpx

from eval.providers.base import BaseProvider, GenerationResult

# Approximate public pricing as of early 2026, per 1K tokens (input+output blended).
# Update if OpenAI changes pricing — this is only used for the cost column in reports.
_PRICE_PER_1K = 0.002


class OpenAIProvider(BaseProvider):
    name = "openai-gpt4o-mini"
    model_name = "gpt-4o-mini"

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        context: str | None = None,
        expected_keywords: list[str] | None = None,
    ) -> GenerationResult:
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}" if context else prompt

        start = time.perf_counter()
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 300,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.perf_counter() - start) * 1000

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", len(full_prompt.split()))
        output_tokens = usage.get("completion_tokens", len(text.split()))
        cost_usd = round((input_tokens + output_tokens) / 1000 * _PRICE_PER_1K, 6)

        return GenerationResult(
            text=text,
            latency_ms=round(latency_ms, 1),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            provider_name=self.name,
            model_name=self.model_name,
        )
