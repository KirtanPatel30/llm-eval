"""
base.py
-------
Every model provider (real API or demo persona) implements this same interface.
This is what makes the harness provider-agnostic: swap OpenAI for Anthropic for
a local model, and nothing else in the codebase has to change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    provider_name: str
    model_name: str


class BaseProvider(ABC):
    """Interface every provider (real or mock) must implement."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: str | None = None,
        expected_keywords: list[str] | None = None,
    ) -> GenerationResult:
        """Given a prompt (and optional grounding context), return a GenerationResult.

        `expected_keywords` is only ever used by the demo/mock providers to
        generate plausible-looking varied responses without a real model call.
        Real providers (OpenAI/Anthropic) ignore it entirely — they never see
        the answer key.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether this provider is usable right now (e.g. API key present)."""
        return True
