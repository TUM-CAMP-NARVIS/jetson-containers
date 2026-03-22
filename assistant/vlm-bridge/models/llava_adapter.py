import os
import torch
from typing import AsyncIterator
from PIL import Image

from .base import VLMAdapter


class LlavaAdapter(VLMAdapter):
    def __init__(self):
        self.model_path = os.environ.get(
            "LLAVA_MODEL_PATH", "liuhaotian/llava-v1.5-7b"
        )
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None

    def load(self):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_name = get_model_name_from_path(self.model_path)
        if model_name is None:
            model_name = self.model_path.split("/")[-1]
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(self.model_path, model_name)
        )

    def model_name(self) -> str:
        return self.model_path

    def _get_conv_mode(self) -> str:
        name = self.model_path.lower()
        if "llama-2" in name:
            return "llava_llama_2"
        elif "v1" in name:
            return "llava_v1"
        elif "mpt" in name:
            return "mpt"
        return "llava_v0"

    async def generate(
        self,
        messages: list[dict],
        images: list[Image.Image],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> AsyncIterator[str] | str:
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

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

        conv = conv_templates[self._get_conv_mode()].copy()
        if self.model.config.mm_use_im_start_end:
            inp = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + prompt_text
            )
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        image_tensor = None
        if images:
            image_tensor = (
                self.image_processor.preprocess(images[0], return_tensors="pt")[
                    "pixel_values"
                ]
                .half()
                .cuda()
            )

        stop_str = (
            conv.sep
            if conv.sep_style != SeparatorStyle.TWO
            else conv.sep2
        )
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        text = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        if stream:
            async def _stream():
                yield text
            return _stream()
        return text
