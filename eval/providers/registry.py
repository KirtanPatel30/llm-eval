"""
registry.py
-----------
Auto-detects which providers are usable and returns exactly those. This is
the piece that makes the whole harness "just work" whether you have zero API
keys (pure demo mode) or real ones (mixed demo + real comparison).
"""

from eval.providers.anthropic_provider import AnthropicProvider
from eval.providers.base import BaseProvider
from eval.providers.mock_provider import build_demo_personas
from eval.providers.openai_provider import OpenAIProvider


def get_active_providers() -> list[BaseProvider]:
    providers: list[BaseProvider] = list(build_demo_personas())

    for real_provider_cls in (OpenAIProvider, AnthropicProvider):
        instance = real_provider_cls()
        if instance.is_available():
            providers.append(instance)

    return providers
