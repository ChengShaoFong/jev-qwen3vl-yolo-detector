from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vdm.inference.person_crop import YoloPersonDetector, padded_box


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensor([[50.2, 10.1, 90.8, 80.9], [5, 5, 20, 20]])
        self.conf = _FakeTensor([0.9, 0.8])
        self.cls = _FakeTensor([0, 2])

    def __len__(self):
        return 2


class _FakeYolo:
    names = {0: "person", 2: "car"}

    def __init__(self, model_path):
        self.model_path = model_path
        self.predict_args = None

    def predict(self, **kwargs):
        self.predict_args = kwargs
        return [types.SimpleNamespace(boxes=_FakeBoxes())]


class PersonCropTests(unittest.TestCase):
    def test_padded_box_is_expanded_and_clamped(self):
        self.assertEqual(padded_box((2, 4, 12, 24), (20, 30), 0.5), (0, 0, 17, 30))

    def test_detector_returns_only_person_crops(self):
        ultralytics = types.SimpleNamespace(YOLO=_FakeYolo)
        with patch.dict(sys.modules, {"ultralytics": ultralytics}):
            detector = YoloPersonDetector("fake.pt", confidence=0.4, padding=0.0)

        image = Image.new("RGB", (100, 100), "white")
        crops = detector.detect(image)

        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].box, (50, 10, 91, 81))
        self.assertEqual(crops[0].image.size, (41, 71))
        self.assertEqual(detector.model.predict_args["classes"], [0])
        self.assertEqual(detector.model.predict_args["device"], "cpu")

    def test_rejects_model_without_person_class(self):
        class NoPersonYolo(_FakeYolo):
            names = {2: "car"}

        ultralytics = types.SimpleNamespace(YOLO=NoPersonYolo)
        with patch.dict(sys.modules, {"ultralytics": ultralytics}):
            with self.assertRaisesRegex(ValueError, "no 'person' class"):
                YoloPersonDetector("fake.pt")


if __name__ == "__main__":
    unittest.main()
