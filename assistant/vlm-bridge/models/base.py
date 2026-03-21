from abc import ABC, abstractmethod
from typing import AsyncIterator
from PIL import Image


class VLMAdapter(ABC):
    """Abstract base for VLM model adapters."""

    @abstractmethod
    def load(self):
        """Load model into GPU memory. Called once at startup."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        images: list[Image.Image],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> AsyncIterator[str] | str:
        """Generate a response. If stream=True, yield chunks."""
        ...
