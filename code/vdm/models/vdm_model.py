"""Minimal Visual Jev choice scorer backed by Qwen3-VL."""
from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import DynamicCache

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
    def score_choices(
        self, image, question: str, choices: Sequence[str], shared_context: str = ""
    ) -> torch.Tensor:
        if not question.strip():
            raise ValueError("question must not be empty")
        if not 2 <= len(choices) <= MAX_CHOICES:
            raise ValueError(f"choices must contain 2 to {MAX_CHOICES} items")

        prefix = prefix_text(shared_context)
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

    @torch.inference_mode()
    def score_questions(
        self,
        image,
        questions: Sequence[tuple[str, Sequence[str]]],
        batch_size: int = 2,
        shared_context: str = "",
    ) -> list[torch.Tensor]:
        """Share image encoding and prefix KV across questions about one image."""
        if not questions:
            raise ValueError("questions must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for question, choices in questions:
            if not question.strip() or not 2 <= len(choices) <= MAX_CHOICES:
                raise ValueError(f"each question needs text and 2 to {MAX_CHOICES} choices")
        if len(questions) == 1:
            question, choices = questions[0]
            return [self.score_choices(image, question, choices, shared_context)]

        encoded = self.processor(
            text=[prefix_text(shared_context)], images=[image], return_tensors="pt"
        )
        prefix_ids = encoded["input_ids"][0]
        prefix_len = len(prefix_ids)
        image_grid = encoded["image_grid_thw"].to(self.device)
        suffixes = []
        positions = []
        prefix_positions = None
        for question, choices in questions:
            suffix_ids = torch.tensor(
                self.tokenizer(suffix_text(question, choices), add_special_tokens=False)["input_ids"],
                dtype=torch.long,
            )
            suffixes.append(suffix_ids)
            full_ids = torch.cat([prefix_ids, suffix_ids])[None].to(self.device)
            token_types = (full_ids == self.config.image_token_id).to(torch.int32)
            position_ids, _ = self.backbone.model.get_rope_index(
                full_ids, mm_token_type_ids=token_types, image_grid_thw=image_grid
            )
            if prefix_positions is None:
                prefix_positions = position_ids[:, :, :prefix_len]
            positions.append(position_ids[:, 0, prefix_len:])

        vl_model = self.backbone.model
        visual = vl_model.get_image_features(
            encoded["pixel_values"].to(self.device, self.compute_dtype),
            image_grid,
            return_dict=True,
        )
        image_embeds = torch.cat(visual.pooler_output, dim=0)
        prefix_ids = prefix_ids.to(self.device)
        visual_mask = prefix_ids == self.config.image_token_id
        prefix_embeds = vl_model.get_input_embeddings()(prefix_ids).clone()
        prefix_embeds[visual_mask] = image_embeds.to(prefix_embeds.dtype)
        prefix_cache = DynamicCache()
        vl_model.language_model(
            inputs_embeds=prefix_embeds[None],
            position_ids=prefix_positions,
            past_key_values=prefix_cache,
            visual_pos_masks=visual_mask[None],
            deepstack_visual_embeds=visual.deepstack_features,
            use_cache=True,
        )

        results = []
        for start in range(0, len(questions), batch_size):
            chunk_suffixes = suffixes[start:start + batch_size]
            chunk_positions = positions[start:start + batch_size]
            max_suffix_len = max(len(ids) for ids in chunk_suffixes)
            count = len(chunk_suffixes)
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            ids = torch.full((count, max_suffix_len), pad_id, dtype=torch.long)
            pos = torch.zeros((3, count, max_suffix_len), dtype=torch.long)
            attention = torch.zeros((count, prefix_len + max_suffix_len), dtype=torch.long)
            attention[:, :prefix_len] = 1
            for row, (suffix_ids, suffix_pos) in enumerate(zip(chunk_suffixes, chunk_positions)):
                size = len(suffix_ids)
                ids[row, -size:] = suffix_ids
                pos[:, row, -size:] = suffix_pos
                attention[row, -size:] = 1

            cache = self._fork_cache(prefix_cache, count)
            suffix_embeds = vl_model.get_input_embeddings()(ids.to(self.device))
            output = vl_model.language_model(
                inputs_embeds=suffix_embeds,
                position_ids=pos.to(self.device),
                attention_mask=attention.to(self.device),
                past_key_values=cache,
                use_cache=True,
            )
            hidden = output.last_hidden_state[:, -1]
            vocabulary_logits = self.backbone.lm_head(hidden.to(self.backbone.dtype))
            option_logits = vocabulary_logits.index_select(-1, self.option_ids).float()
            for row, (_, choices) in enumerate(questions[start:start + batch_size]):
                results.append(option_logits[row, :len(choices)])
        return results

    @staticmethod
    def _fork_cache(cache: DynamicCache, batch_size: int) -> DynamicCache:
        forked = DynamicCache()
        for layer_index, layer in enumerate(cache.layers):
            keys = layer.keys.expand(batch_size, -1, -1, -1).contiguous().clone()
            values = layer.values.expand(batch_size, -1, -1, -1).contiguous().clone()
            forked.update(keys, values, layer_index)
        return forked
