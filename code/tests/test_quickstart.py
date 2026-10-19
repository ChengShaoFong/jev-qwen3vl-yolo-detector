from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
import sys
import unittest

from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.quickstart import classify_image, resolve_model_path


class _NoPersonDetector:
    def detect(self, image):
        return []


class _RecordingModel:
    def __init__(self):
        self.received_size = None

    def score_choices(self, image, question, choices):
        self.received_size = image.size
        return torch.tensor([2.0, 1.0])


class QuickstartTests(unittest.TestCase):
    def test_hugging_face_name_prefers_local_model(self):
        resolved = resolve_model_path("Qwen/Qwen3-VL-2B-Instruct")
        self.assertTrue(resolved.endswith(os.path.join("models", "Qwen3-VL-2B-Instruct")))
        self.assertTrue(os.path.isdir(resolved))

    def test_no_person_falls_back_to_full_image(self):
        model = _RecordingModel()
        image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "figures", "demo_bottle.jpg"
        ))
        with Image.open(image_path) as source:
            expected_size = source.size
        output = io.StringIO()
        with redirect_stdout(output):
            classify_image(
                model,
                _NoPersonDetector(),
                torch,
                Image,
                image_path,
                "Is the image bright?",
                ["yes", "no"],
            )

        self.assertEqual(model.received_size, expected_size)
        self.assertIn("running Jev on the full image", output.getvalue())
        self.assertIn("prediction: yes", output.getvalue())


if __name__ == "__main__":
    unittest.main()
