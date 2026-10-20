"""Prompt text for Visual Jev multiple-choice inference."""
from __future__ import annotations

from typing import Sequence

LETTERS = [chr(ord("A") + index) for index in range(16)]
SYSTEM = (
    "You are a visual decision model. You inspect the image and answer each "
    "question by choosing exactly one of the given options. You never explain."
)


def prefix_text(shared_context: str = "") -> str:
    context = f"\n{shared_context.strip()}" if shared_context.strip() else ""
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|><|image_pad|><|vision_end|>{context}"
    )


def suffix_text(question: str, choices: Sequence[str]) -> str:
    lines = [f"\n\nQuestion: {question.strip()}", "Options:"]
    lines.extend(f"{LETTERS[index]}. {choice}" for index, choice in enumerate(choices))
    lines.append("Reply with the single letter of the correct option.")
    return "\n".join(lines) + "<|im_end|>\n<|im_start|>assistant\nAnswer:"


def option_token_strings(count: int) -> list[str]:
    return [f" {LETTERS[index]}" for index in range(count)]
