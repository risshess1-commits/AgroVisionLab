from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demo_assets" / "cucumber_demo"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DemoShape:
    label: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class DemoSample:
    digest: str
    image_name: str
    width: int
    height: int
    shapes: tuple[DemoShape, ...]


def _shape_box(points: list[list[float]]) -> tuple[int, int, int, int] | None:
    if len(points) < 2:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    )


def _load_sample(image_path: Path) -> DemoSample | None:
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        return None

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    shapes: list[DemoShape] = []
    for shape in payload.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        box = _shape_box(shape.get("points", []))
        if box is None:
            continue
        label = str(shape.get("label", "")).strip()
        if not label:
            continue
        shapes.append(DemoShape(label=label, box=box))

    if not shapes:
        return None

    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return DemoSample(
        digest=digest,
        image_name=image_path.name,
        width=int(payload.get("imageWidth") or 0),
        height=int(payload.get("imageHeight") or 0),
        shapes=tuple(shapes),
    )


@lru_cache(maxsize=1)
def load_demo_samples() -> dict[str, DemoSample]:
    samples: dict[str, DemoSample] = {}
    if not DEMO_DIR.exists():
        return samples

    for image_path in sorted(DEMO_DIR.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        sample = _load_sample(image_path)
        if sample is not None:
            samples[sample.digest] = sample
    return samples


def match_demo_sample(path: Path) -> DemoSample | None:
    if path.suffix.lower() not in IMAGE_SUFFIXES or not path.exists():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return load_demo_samples().get(digest)
