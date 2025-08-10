"""
VLM provider implementations.
"""
from .base import VLMProvider
from .ollama import OllamaProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .google import GoogleProvider
from .cv_provider import CVProvider

__all__ = [
    'VLMProvider',
    'OllamaProvider', 
    'AnthropicProvider',
    'OpenAIProvider',
    'GoogleProvider',
    'CVProvider'
]


def get_provider(provider_name: str, **kwargs) -> VLMProvider:
    """Factory function to get a VLM provider by name."""
    providers = {
        'ollama': OllamaProvider,
        'anthropic': AnthropicProvider,
        'openai': OpenAIProvider,
        'google': GoogleProvider,
        'opencv': CVProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    
    return providers[provider_name](**kwargs)