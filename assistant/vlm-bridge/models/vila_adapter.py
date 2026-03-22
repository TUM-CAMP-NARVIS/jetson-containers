import os
import torch
from typing import AsyncIterator
from PIL import Image

from .base import VLMAdapter


class VilaAdapter(VLMAdapter):
    def __init__(self):
        self.model_path = os.environ.get(
            "VILA_MODEL_PATH", "Efficient-Large-Model/VILA1.5-3b"
        )
        self.conv_mode = os.environ.get("VILA_CONV_MODE", "vicuna_v1")
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None

    def load(self):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        import transformers

        # Patch resize_token_embeddings to avoid MAGMA requirement on Jetson
        _orig_resize = transformers.PreTrainedModel.resize_token_embeddings

        def _safe_resize(self_model, new_num_tokens=None, **kwargs):
            kwargs["mean_resizing"] = False
            return _orig_resize(self_model, new_num_tokens, **kwargs)

        transformers.PreTrainedModel.resize_token_embeddings = _safe_resize

        model_name = get_model_name_from_path(self.model_path)
        if model_name is None:
            model_name = self.model_path.split("/")[-1]
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(self.model_path, model_name)
        )

        # Restore original
        transformers.PreTrainedModel.resize_token_embeddings = _orig_resize

    def model_name(self) -> str:
        return self.model_path

    async def generate(
        self,
        messages: list[dict],
        images: list[Image.Image],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> AsyncIterator[str] | str:
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        # Extract last user text prompt
        prompt_text = ""
        for msg in reversed(messages):
            content = msg.get("content", "")
            if msg.get("role") == "user":
                if isinstance(content, str):
                    prompt_text = content
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text" and part.get("text"):
                            prompt_text = part["text"]
                break

        # Build conversation with image token
        conv = conv_templates[self.conv_mode].copy()
        if images:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        else:
            inp = prompt_text

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(full_prompt, self.tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        # Process image
        image_tensor = None
        if images:
            image_tensor = (
                self.image_processor.preprocess(images[0], return_tensors="pt")[
                    "pixel_values"
                ]
                .half()
                .cuda()
            )

        # VILA uses media={"image": [tensors]} instead of images=
        media = {}
        if image_tensor is not None:
            media = {"image": [image_tensor]}

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                media=media,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_tokens,
                use_cache=True,
            )

        text = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        if stream:
            async def _stream():
                yield text
            return _stream()
        return text
