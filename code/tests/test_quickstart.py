from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
import sys
import unittest
from unittest.mock import patch

from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.quickstart import classify_image, parse_args, resolve_model_path


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
    def test_default_cli_uses_4b_with_adapter(self):
        image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "figures", "demo_bottle.jpg"
        ))
        with patch.object(sys, "argv", ["quickstart.py", "--image", image_path,
                                        "--interactive", "--use-yolo"]):
            _, args = parse_args()
        self.assertEqual(args.model, "Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(args.adapter, "guanxuyu/visual-jev-4b-answer-sft")
        self.assertFalse(args.base_only)

    def test_2b_requires_base_only_with_default_adapter(self):
        image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "figures", "demo_bottle.jpg"
        ))
        argv = ["quickstart.py", "--image", image_path, "--interactive", "--use-yolo",
                "--model", "Qwen/Qwen3-VL-2B-Instruct"]
        with patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()
        with patch.object(sys, "argv", [*argv, "--base-only"]):
            _, args = parse_args()
        self.assertTrue(args.base_only)

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
