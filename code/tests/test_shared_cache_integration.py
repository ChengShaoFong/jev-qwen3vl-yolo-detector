"""Optional GPU parity check: set VISUAL_JEV_RUN_GPU_TESTS=1 to run it."""
from __future__ import annotations

import os
import sys
import unittest

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.quickstart import resolve_model_path
from vdm.models.vdm_model import VisualJev


@unittest.skipUnless(os.environ.get("VISUAL_JEV_RUN_GPU_TESTS") == "1", "requires local CUDA model")
class SharedCacheParityTests(unittest.TestCase):
    def test_shared_batch_agrees_with_independent_questions(self):
        model = VisualJev(
            resolve_model_path("Qwen/Qwen3-VL-2B-Instruct"),
            load_in_4bit=True,
            max_pixels=32768,
        )
        image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "figures", "demo_bottle.jpg"
        ))
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        questions = [
            ("Is there a bottle in the image?", ["yes", "no"]),
            ("What color is the bottle?", ["green", "blue", "yellow", "black"]),
        ]
        independent = [model.score_choices(image, question, choices)
                       for question, choices in questions]
        shared = model.score_questions(image, questions, batch_size=2)
        for baseline, cached in zip(independent, shared):
            self.assertEqual(baseline.argmax().item(), cached.argmax().item())
            self.assertTrue(torch.allclose(
                torch.softmax(baseline, -1), torch.softmax(cached, -1), atol=0.02
            ))


if __name__ == "__main__":
    unittest.main()
