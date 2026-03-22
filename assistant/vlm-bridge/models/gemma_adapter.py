import os
import torch
from typing import AsyncIterator
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from .base import VLMAdapter


class GemmaAdapter(VLMAdapter):
    def __init__(self):
        self.model_id = os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
        self.model = None
        self.processor = None

    def load(self):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="cuda",
        ).eval()

    def model_name(self) -> str:
        return self.model_id

    async def generate(
        self,
        messages: list[dict],
        images: list[Image.Image],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> AsyncIterator[str] | str:
        gemma_messages = []
        image_idx = 0
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                gemma_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        if image_idx < len(images):
                            parts.append({"type": "image", "image": images[image_idx]})
                            image_idx += 1
                if not any(p.get("type") == "text" and p.get("text") for p in parts):
                    parts.append({"type": "text", "text": ""})
                gemma_messages.append({"role": role, "content": parts})

        inputs = self.processor.apply_chat_template(
            gemma_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )

        generated = output[0][input_len:]
        text = self.processor.decode(generated, skip_special_tokens=True)

        if stream:
            async def _stream():
                yield text
            return _stream()
        return text
