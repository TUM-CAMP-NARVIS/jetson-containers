import os

from .base import VLMAdapter


def get_adapter() -> VLMAdapter:
    """Return the adapter for the VLM_MODEL env var."""
    model = os.environ.get("VLM_MODEL", "gemma")

    if model == "vila":
        from .vila_adapter import VilaAdapter
        return VilaAdapter()
    elif model == "gemma":
        from .gemma_adapter import GemmaAdapter
        return GemmaAdapter()
    elif model == "llava":
        from .llava_adapter import LlavaAdapter
        return LlavaAdapter()
    else:
        raise ValueError(f"Unknown VLM_MODEL: {model}. Use vila, gemma, or llava.")
