"""Interactive YOLO person cropping followed by Visual Jev classification."""
from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter


def resolve_model_path(requested_path):
    """Prefer models/<model-name> so the fixed command works offline."""
    if os.path.isdir(requested_path):
        return requested_path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_path = os.path.join(project_root, "models", requested_path.rsplit("/", 1)[-1])
    return local_path if os.path.isdir(local_path) else requested_path


def classify_image(model, detector, torch, Image, image_path, question, choices):
    """Classify every detected person, or the full image when none is found."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    detections = detector.detect(image)
    if detections:
        print(f"YOLO found {len(detections)} person(s).", flush=True)
        inputs = [(detection, detection.image) for detection in detections]
    else:
        print("YOLO found no people; running Jev on the full image.", flush=True)
        inputs = [(None, image)]

    started = perf_counter()
    for index, (detection, input_image) in enumerate(inputs, start=1):
        if detection is not None:
            print(
                f"\n=== Person {index}: confidence={detection.confidence:.3f}, "
                f"box={detection.box} ==="
            )

        logits = model.score_choices(input_image, question, choices)
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        for choice, probability in zip(choices, probabilities):
            print(f"  {choice}: {probability:.3f}")
        prediction = choices[max(range(len(probabilities)), key=probabilities.__getitem__)]
        print(f"  prediction: {prediction}")

    print(f"Inference: {perf_counter() - started:.1f}s", flush=True)


def interactive(model, detector, torch, Image, initial_image):
    """Keep both models loaded while accepting image/question requests."""
    image_path = initial_image
    print(
        "Interactive mode: press Enter to reuse the image; "
        "leave the question blank to quit.",
        flush=True,
    )
    while True:
        try:
            new_path = input(f"Image [{image_path}]: ").strip().strip('"')
            if new_path:
                image_path = new_path
            question = input("Question: ").strip()
            if not question:
                return
            choices = [item.strip() for item in input(
                "Choices (comma-separated): "
            ).split(",")]
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not os.path.isfile(image_path):
            print(f"Image not found: {image_path}", flush=True)
            continue
        if not 2 <= len(choices) <= 16 or any(not choice for choice in choices):
            print("Enter 2 to 16 non-empty choices separated by commas.", flush=True)
            continue

        try:
            classify_image(model, detector, torch, Image, image_path, question, choices)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Could not process request: {exc}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="initial local image path")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default="guanxuyu/visual-jev-4b-answer-sft",
                        help="Visual Jev LoRA adapter; ignored with --base-only")
    parser.add_argument("--base-only", action="store_true",
                        help="skip the adapter and use only the Qwen base model")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="load language weights in 4-bit NF4")
    parser.add_argument("--max-pixels", type=int, default=32768)
    parser.add_argument("--device", choices=["auto", "cuda", "mps"], default="auto")
    parser.add_argument("--interactive", action="store_true", required=True)
    parser.add_argument("--use-yolo", action="store_true", required=True)
    parser.add_argument("--yolo-model", default="yolo11n.pt")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-padding", type=float, default=0.05)
    parser.add_argument("--yolo-device", default="cpu")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        parser.error(f"image not found: {args.image}")
    if not args.base_only and args.adapter == "guanxuyu/visual-jev-4b-answer-sft" \
            and os.path.basename(args.model.rstrip("/\\")) != "Qwen3-VL-4B-Instruct":
        parser.error("the default Visual Jev adapter requires Qwen3-VL-4B-Instruct; "
                     "use --base-only or select a compatible --adapter")
    if not 0.0 <= args.yolo_confidence <= 1.0:
        parser.error("--yolo-confidence must be between 0 and 1")
    if not 0.0 <= args.yolo_padding <= 1.0:
        parser.error("--yolo-padding must be between 0 and 1")
    return parser, args


def main():
    parser, args = parse_args()
    from tqdm.auto import tqdm

    progress = tqdm(total=3, desc="Importing libraries", ascii=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} stages",
                    dynamic_ncols=True)
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    from PIL import Image

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vdm.inference.person_crop import YoloPersonDetector
    from vdm.models.vdm_model import VisualJev
    progress.update(1)

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            parser.error("no supported accelerator found; CUDA or Apple MPS is required")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but it is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested, but it is not available")
    if args.load_in_4bit and device != "cuda":
        parser.error("--load-in-4bit requires CUDA")

    progress.set_description("Loading YOLO")
    detector = YoloPersonDetector(
        model_path=args.yolo_model,
        confidence=args.yolo_confidence,
        padding=args.yolo_padding,
        device=args.yolo_device,
    )
    progress.update(1)

    progress.set_description("Loading Visual Jev")
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    model_path = resolve_model_path(args.model)
    adapter_path = None if args.base_only else resolve_model_path(args.adapter)
    if model_path != args.model:
        print(f"\nUsing local model: {model_path}", flush=True)
    if adapter_path and adapter_path != args.adapter:
        print(f"Using local adapter: {adapter_path}", flush=True)
    model = VisualJev(
        model_path,
        device=device,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        max_pixels=args.max_pixels,
        adapter_path=adapter_path,
    )
    progress.update(1)
    progress.set_description("Ready")
    progress.close()

    precision = "4-bit NF4" if args.load_in_4bit else str(dtype)
    print(f"Using {device} with {precision}", flush=True)
    print(f"Visual Jev adapter: {adapter_path or 'disabled (base-only)'}", flush=True)
    interactive(model, detector, torch, Image, args.image)


if __name__ == "__main__":
    main()
