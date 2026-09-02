"""
anthropic_provider.py
----------------------
Real Anthropic adapter. Only activates if ANTHROPIC_API_KEY is set — mirrors
openai_provider.py's structure exactly, which is the point: adding a new
provider to this harness means writing one small file that matches this
shape, nothing else in the codebase changes.
"""

import os
import time

import httpx

from eval.providers.base import BaseProvider, GenerationResult

_PRICE_PER_1K = 0.003  # approximate blended input+output price, update as needed


class AnthropicProvider(BaseProvider):
    name = "anthropic-claude-haiku"
    model_name = "claude-haiku-4-5-20251001"

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

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
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model_name,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": full_prompt}],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.perf_counter() - start) * 1000

        text = "".join(block.get("text", "") for block in data.get("content", []))
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", len(full_prompt.split()))
        output_tokens = usage.get("output_tokens", len(text.split()))
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
