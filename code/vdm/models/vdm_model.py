"""Minimal Visual Jev choice scorer backed by Qwen3-VL."""
from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from .prompts import option_token_strings, prefix_text, suffix_text

MAX_CHOICES = 16


class VisualJev:
    """Score 2–16 choices for one question about one image."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
        max_pixels: int = 32768,
        adapter_path: str | None = None,
    ) -> None:
        self.device = device
        self.compute_dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.image_processor.max_pixels = max_pixels
        self.tokenizer = self.processor.tokenizer
        self.config = AutoConfig.from_pretrained(model_path)

        model_kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                    llm_int8_skip_modules=["model.visual", "lm_head"],
                ),
                device_map={"": device},
            )

        backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, **model_kwargs
        )
        if not load_in_4bit:
            backbone = backbone.to(device)
        if adapter_path:
            from peft import PeftModel

            self.adapter_model = PeftModel.from_pretrained(backbone, adapter_path)
            self.adapter_model.eval()
            # PEFT wraps Qwen; the scoring path needs the underlying Qwen modules.
            self.backbone = self.adapter_model.get_base_model()
        else:
            self.backbone = backbone
        self.backbone.eval()
        self.option_ids = self._option_ids()

    def _option_ids(self) -> torch.Tensor:
        ids = []
        for option in option_token_strings(MAX_CHOICES):
            tokens = self.tokenizer(option, add_special_tokens=False)["input_ids"]
            if len(tokens) != 1:
                raise ValueError(f"option marker {option!r} is not one token: {tokens}")
            ids.append(tokens[0])
        return torch.tensor(ids, device=self.device)

    @torch.inference_mode()
    def score_choices(self, image, question: str, choices: Sequence[str]) -> torch.Tensor:
        if not question.strip():
            raise ValueError("question must not be empty")
        if not 2 <= len(choices) <= MAX_CHOICES:
            raise ValueError(f"choices must contain 2 to {MAX_CHOICES} items")

        prefix = prefix_text()
        suffix = suffix_text(question, choices)
        encoded = self.processor(text=[prefix], images=[image], return_tensors="pt")
        prefix_ids = encoded["input_ids"][0]
        suffix_ids = torch.tensor(
            self.tokenizer(suffix, add_special_tokens=False)["input_ids"],
            dtype=torch.long,
        )
        input_ids = torch.cat([prefix_ids, suffix_ids])[None].to(self.device)
        image_grid = encoded["image_grid_thw"].to(self.device)
        token_types = (input_ids == self.config.image_token_id).to(torch.int32)
        position_ids, _ = self.backbone.model.get_rope_index(
            input_ids,
            mm_token_type_ids=token_types,
            image_grid_thw=image_grid,
        )

        output = self.backbone.model(
            input_ids=input_ids,
            pixel_values=encoded["pixel_values"].to(self.device, self.compute_dtype),
            image_grid_thw=image_grid,
            mm_token_type_ids=token_types,
            position_ids=position_ids,
            use_cache=False,
        )
        final_hidden = output.last_hidden_state[0, -1]
        vocabulary_logits = self.backbone.lm_head(final_hidden.to(self.backbone.dtype))
        return vocabulary_logits.index_select(0, self.option_ids[:len(choices)]).float()
