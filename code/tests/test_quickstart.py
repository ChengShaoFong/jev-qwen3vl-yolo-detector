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

from examples.quickstart import (
    classify_image, classify_questions, interactive, load_request_file, parse_args,
    resolve_model_path,
)


class _NoPersonDetector:
    def detect(self, image):
        return []


class _RecordingModel:
    def __init__(self):
        self.received_size = None

    def score_choices(self, image, question, choices):
        self.received_size = image.size
        return torch.tensor([2.0, 1.0])


class _BatchModel:
    def __init__(self):
        self.calls = []

    def score_questions(self, image, questions, **kwargs):
        self.calls.append((image.size, questions, kwargs))
        return [torch.tensor([2.0, 1.0]) for _ in questions]


class QuickstartTests(unittest.TestCase):
    def test_interactive_batches_questions_before_inference(self):
        model = _BatchModel()
        image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "figures", "demo_bottle.jpg"
        ))
        answers = ["", "Is there a bottle?", "yes, no",
                   "What color is it?", "green, blue", "", "", ""]
        output = io.StringIO()
        with patch("builtins.input", side_effect=answers) as input_mock, \
                redirect_stdout(output):
            interactive(model, None, torch, Image, image_path, batch_size=2)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][1], [
            ("Is there a bottle?", ["yes", "no"]),
            ("What color is it?", ["green", "blue"]),
        ])
        self.assertEqual(model.calls[0][2]["batch_size"], 2)
        self.assertIn("Next question [Enter to run]: ", [
            call.args[0] for call in input_mock.call_args_list
        ])

    def test_interactive_single_question_uses_original_path(self):
        model = _RecordingModel()
        image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "figures", "demo_bottle.jpg"
        ))
        answers = ["", "Is there a bottle?", "yes, no", "", "", ""]
        with patch("builtins.input", side_effect=answers), redirect_stdout(io.StringIO()):
            interactive(model, None, torch, Image, image_path)
        self.assertIsNotNone(model.received_size)

    def test_request_file_uses_named_questions(self):
        path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "examples", "bottle_questions.json"
        ))
        questions, state = load_request_file(path)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0][1], ["yes", "no"])
        self.assertTrue(state)

    def test_ppe_request_file_has_five_questions_and_uncertainty_option(self):
        path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "examples", "ppe_questions.json"
        ))
        questions, state = load_request_file(path)
        self.assertEqual(len(questions), 5)
        self.assertIn("visible", state)
        for question, choices in questions:
            self.assertTrue(question.startswith("Is any visible person missing"))
            self.assertEqual(choices, ["yes", "no", "unclear", "no person visible"])

    def test_multiple_questions_share_one_model_call_per_image(self):
        model = _BatchModel()
        image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "figures", "demo_bottle.jpg"
        ))
        questions = [("Bottle present?", ["yes", "no"]),
                     ("Bottle green?", ["yes", "no"])]
        output = io.StringIO()
        with redirect_stdout(output):
            classify_questions(
                model, _NoPersonDetector(), torch, Image, image_path, questions,
                batch_size=2, shared_context="Inventory image",
            )
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][2], {
            "batch_size": 2, "shared_context": "Inventory image",
        })
        self.assertEqual(output.getvalue().count("prediction: yes"), 2)

    def test_default_cli_uses_4b_with_adapter(self):
        image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "figures", "demo_bottle.jpg"
        ))
        with patch.object(sys, "argv", ["quickstart.py", "--image", image_path]):
            _, args = parse_args()
        self.assertEqual(args.model, "Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(args.adapter, "guanxuyu/visual-jev-4b-answer-sft")
        self.assertFalse(args.base_only)
        self.assertFalse(args.use_yolo)
        self.assertTrue(args.interactive)

    def test_yolo_can_be_enabled_explicitly(self):
        image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "figures", "demo_bottle.jpg"
        ))
        with patch.object(sys, "argv", ["quickstart.py", "--image", image_path,
                                        "--interactive", "--use-yolo"]):
            _, args = parse_args()
        self.assertTrue(args.use_yolo)

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

    def test_yolo_disabled_uses_full_image(self):
        model = _RecordingModel()
        image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "figures", "demo_bottle.jpg"
        ))
        with Image.open(image_path) as source:
            expected_size = source.size
        output = io.StringIO()
        with redirect_stdout(output):
            classify_image(
                model, None, torch, Image, image_path,
                "Is there a bottle?", ["yes", "no"],
            )
        self.assertEqual(model.received_size, expected_size)
        self.assertIn("YOLO disabled; running Jev on the full image", output.getvalue())


if __name__ == "__main__":
    unittest.main()
