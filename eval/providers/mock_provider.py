"""
mock_provider.py
-----------------
Three deterministic "demo personas" that let the entire harness run and be
demoed with zero API keys. Each persona has different quality/speed/cost
characteristics so the comparison charts show meaningful, realistic-looking
differences — this is what a recruiter sees out of the box.

Determinism: responses are generated from a hash of the prompt, so the same
prompt always produces the same "model behavior" for a given persona. This
makes runs reproducible, which matters for an eval tool — you want to trust
that a score difference reflects a real change, not run-to-run randomness.

IMPORTANT: unlike real providers, personas are given `expected_keywords` as a
convenience so they can fabricate plausible varied answers. Real providers
(openai_provider.py, anthropic_provider.py) never receive the answer key —
this parameter is a demo-only convenience, not part of how a real eval works.
"""

import hashlib
import random
import time

from eval.providers.base import BaseProvider, GenerationResult

FABRICATED_FACTS = [
    "approximately 42% higher than industry average",
    "first documented in 1987",
    "used by over 3 million organizations worldwide",
    "roughly 12.4 kilometers in total length",
    "developed originally for aerospace applications",
]


class MockPersona(BaseProvider):
    def __init__(
        self,
        name: str,
        base_latency_ms: float,
        latency_jitter_ms: float,
        cost_per_1k_tokens: float,
        keyword_coverage_rate: float,
        hallucination_rate: float,
        refusal_reliability: float,
        verbosity: int,
    ):
        self.name = name
        self.model_name = name
        self.base_latency_ms = base_latency_ms
        self.latency_jitter_ms = latency_jitter_ms
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.keyword_coverage_rate = keyword_coverage_rate
        self.hallucination_rate = hallucination_rate
        self.refusal_reliability = refusal_reliability
        self.verbosity = verbosity

    def _rng_for(self, prompt: str) -> random.Random:
        seed = int(hashlib.sha256(f"{self.name}:{prompt}".encode()).hexdigest(), 16) % (2**32)
        return random.Random(seed)

    def generate(
        self,
        prompt: str,
        context: str | None = None,
        expected_keywords: list[str] | None = None,
    ) -> GenerationResult:
        rng = self._rng_for(prompt)
        start = time.perf_counter()

        is_unsafe_prompt = any(
            trigger in prompt.lower()
            for trigger in ["bypass", "hack into", "make a weapon", "illegal drug", "steal"]
        )

        sentences = []

        if is_unsafe_prompt:
            if rng.random() < self.refusal_reliability:
                sentences.append(
                    "I can't help with that request as it could facilitate harm."
                )
            else:
                # Weaker personas occasionally fail to refuse — this is the
                # kind of gap a real safety eval is specifically built to catch.
                sentences.append(
                    "Here's a general overview of how that process might work."
                )
        else:
            keywords = expected_keywords or []
            n_to_include = round(len(keywords) * self.keyword_coverage_rate)
            included = rng.sample(keywords, min(n_to_include, len(keywords))) if keywords else []

            if included:
                sentences.append(
                    "Based on the available information, key points include "
                    + ", ".join(included) + "."
                )
            else:
                sentences.append("Here is a general response addressing the prompt.")

            if context and rng.random() < self.hallucination_rate:
                sentences.append(
                    "Notably, this is " + rng.choice(FABRICATED_FACTS) + "."
                )

            for _ in range(max(0, self.verbosity - 1)):
                sentences.append(
                    rng.choice(
                        [
                            "This is supported by several independent observations.",
                            "Further context can refine this answer.",
                            "There are some edge cases worth considering separately.",
                        ]
                    )
                )

        text = " ".join(sentences)

        # Simulated latency (does not actually sleep — keeps the demo fast)
        latency_ms = max(20.0, rng.gauss(self.base_latency_ms, self.latency_jitter_ms))

        input_tokens = max(1, len(prompt.split()) + len((context or "").split()))
        output_tokens = max(1, len(text.split()))
        cost_usd = round(
            (input_tokens + output_tokens) / 1000 * self.cost_per_1k_tokens, 6
        )

        return GenerationResult(
            text=text,
            latency_ms=round(latency_ms, 1),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            provider_name=self.name,
            model_name=self.model_name,
        )


def build_demo_personas() -> list[MockPersona]:
    return [
        MockPersona(
            name="demo-fast",
            base_latency_ms=180,
            latency_jitter_ms=30,
            cost_per_1k_tokens=0.0002,
            keyword_coverage_rate=0.45,
            hallucination_rate=0.35,
            refusal_reliability=0.6,
            verbosity=1,
        ),
        MockPersona(
            name="demo-balanced",
            base_latency_ms=650,
            latency_jitter_ms=90,
            cost_per_1k_tokens=0.0015,
            keyword_coverage_rate=0.75,
            hallucination_rate=0.15,
            refusal_reliability=0.9,
            verbosity=2,
        ),
        MockPersona(
            name="demo-quality",
            base_latency_ms=1400,
            latency_jitter_ms=200,
            cost_per_1k_tokens=0.006,
            keyword_coverage_rate=0.95,
            hallucination_rate=0.03,
            refusal_reliability=0.99,
            verbosity=3,
        ),
    ]
