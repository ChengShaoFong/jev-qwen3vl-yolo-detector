"""Optional YOLO person detection for Visual Jev inference."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class PersonCrop:
    """One detected person and the PIL image cropped around that person."""

    image: Any
    box: tuple[int, int, int, int]
    confidence: float


class YoloPersonDetector:
    """Lazily isolate the Ultralytics-specific person detection behavior."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.25,
        padding: float = 0.05,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("YOLO confidence must be between 0 and 1")
        if not 0.0 <= padding <= 1.0:
            raise ValueError("YOLO padding must be between 0 and 1")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO mode requires ultralytics; install code/requirements.txt first"
            ) from exc

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.padding = padding
        self.device = device
        names = self.model.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.person_class_ids = {
            int(class_id)
            for class_id, name in names.items()
            if str(name).strip().casefold() == "person"
        }
        if not self.person_class_ids:
            raise ValueError(f"YOLO model {model_path!r} has no 'person' class")

    def detect(self, image: Any) -> list[PersonCrop]:
        """Return left-to-right person crops from a PIL image."""
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            classes=sorted(self.person_class_ids),
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        crops = []
        for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids):
            if int(class_id) not in self.person_class_ids:
                continue
            box = padded_box(coordinates, image.size, self.padding)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crops.append(PersonCrop(image=image.crop(box), box=box, confidence=float(confidence)))

        return sorted(crops, key=lambda crop: (crop.box[0], crop.box[1]))


def padded_box(
    coordinates: list[float] | tuple[float, ...],
    image_size: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    """Expand an xyxy box by a fraction of its size and clamp it to the image."""
    x1, y1, x2, y2 = coordinates
    width, height = image_size
    pad_x = max(0.0, x2 - x1) * padding
    pad_y = max(0.0, y2 - y1) * padding
    return (
        max(0, math.floor(x1 - pad_x)),
        max(0, math.floor(y1 - pad_y)),
        min(width, math.ceil(x2 + pad_x)),
        min(height, math.ceil(y2 + pad_y)),
    )
