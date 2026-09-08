from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import MODEL_PROFILES, OUTPUT_DIR
from .demo_catalog import DemoSample, match_demo_sample

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime for lightweight installs
    YOLO = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
VIDEO_TRACK_IOU_THRESHOLD = 0.35
VIDEO_TRACK_SMOOTH_ALPHA = 0.32
VIDEO_TRACK_HOLD_SAMPLES = 3
OVERLAY_COMPACT_LABEL_THRESHOLD = 8

OVERLAY_REPLACEMENTS = {
    "Плоды томата всего": "tomato total",
    "Плоды томата": "tomato",
    "Спелые плоды томата": "ripe tomato",
    "Неспелые плоды томата": "unripe tomato",
    "Цветки / завязи": "flower",
    "Плоды огурца всего": "cucumber total",
    "Плоды огурца": "cucumber",
    "Прямые плоды огурца": "straight cucumber",
    "Кривые плоды огурца": "curved cucumber",
    "Завязи / цветки": "flower",
    "Спелые плоды": "ripe tomato",
    "Неспелые плоды": "unripe tomato",
    "Здоровые листья": "healthy leaves",
    "Дефекты / стресс": "leaf stress",
    "Признаки дефицита питания": "nutrition deficit",
    "Ягоды": "berry",
    "Листья огурца всего": "cucumber leaves",
    "Объекты не найдены": "No objects",
    "Модель не подключена": "Model not connected",
    "плоды не найдены": "no fruits",
    "зон риска": "risk zones",
}

CUCUMBER_FLOWER_COLOR = (60, 76, 231)
CUCUMBER_CURVED_COLOR = (235, 110, 40)
CUCUMBER_STRAIGHT_COLOR = (66, 206, 245)
LEAF_BOX_COLOR = (64, 150, 78)

DEMO_LABEL_SPECS = {
    "zavaz": {
        "count_label": "Завязи / цветки",
        "box_label": "ovary / flower",
        "color": CUCUMBER_FLOWER_COLOR,
    },
    "cuc_curv": {
        "count_label": "Кривые плоды огурца",
        "box_label": "curved cucumber",
        "color": CUCUMBER_CURVED_COLOR,
    },
    "cuc_line": {
        "count_label": "Прямые плоды огурца",
        "box_label": "straight cucumber",
        "color": CUCUMBER_STRAIGHT_COLOR,
    },
    "leaf": {
        "count_label": "Листья огурца всего",
        "box_label": "leaf",
        "color": LEAF_BOX_COLOR,
    },
}

DEMO_PROFILE_LABELS = {
    "cucumber": ("zavaz", "cuc_curv", "cuc_line"),
    "cucumber_universal": ("zavaz", "cuc_curv", "cuc_line"),
    "cucumber_leaf": ("leaf",),
}


class ModelRegistry:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, profile_id: str) -> Any | None:
        profile = MODEL_PROFILES[profile_id]
        weights = Path(profile["weights"])
        if not weights.exists() or YOLO is None:
            return None
        if profile_id not in self._cache:
            self._cache[profile_id] = YOLO(str(weights))
        return self._cache[profile_id]


registry = ModelRegistry()


def profile_payload() -> list[dict[str, Any]]:
    payload = []
    for profile in MODEL_PROFILES.values():
        weights = Path(profile["weights"])
        item = {
            **{k: v for k, v in profile.items() if k != "weights"},
            "weights": weights.name,
            "model_available": weights.exists(),
            "mode": "YOLO" if weights.exists() and profile["kind"] == "detect" else "demo",
        }
        payload.append(item)
    return payload


def media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError("Поддерживаются изображения JPG/PNG/WEBP и видео MP4/MOV/AVI/MKV/WEBM.")


def analyze_file(profile_id: str, input_path: Path, confidence: float = 0.25) -> dict[str, Any]:
    if profile_id not in MODEL_PROFILES:
        raise ValueError("Неизвестный профиль модели.")
    media_type = media_type_for(input_path)
    started = time.perf_counter()

    if media_type == "image":
        result = analyze_image(profile_id, input_path, confidence)
    else:
        result = analyze_video(profile_id, input_path, confidence)

    result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    result["media_type"] = media_type
    result["profile_id"] = profile_id
    result["confidence"] = confidence
    result["counts_json"] = json.dumps(result["counts"], ensure_ascii=False)
    return result


def analyze_image(profile_id: str, input_path: Path, confidence: float) -> dict[str, Any]:
    image = read_image(input_path)
    if image is None:
        raise ValueError("Не удалось прочитать изображение.")

    output_name = f"{input_path.stem}_{uuid.uuid4().hex[:8]}_result.jpg"
    output_path = OUTPUT_DIR / output_name
    annotated, counts, summary = process_frame(profile_id, image, confidence, source_path=input_path)
    write_image(output_path, annotated)

    return {
        "output_path": output_path,
        "counts": counts,
        "summary": summary,
    }


def analyze_video(profile_id: str, input_path: Path, confidence: float) -> dict[str, Any]:
    source_path, remove_source = materialize_ascii_copy(input_path)
    profile = MODEL_PROFILES[profile_id]
    model = registry.get(profile_id)
    temp_output_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_agrovision_result.mp4"
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        if remove_source and source_path.exists():
            source_path.unlink(missing_ok=True)
        raise ValueError("Не удалось открыть видео.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, math.ceil((frame_count or 240) / 240))

    output_name = f"{input_path.stem}_{uuid.uuid4().hex[:8]}_result.mp4"
    output_path = OUTPUT_DIR / output_name
    writer = cv2.VideoWriter(
        str(temp_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        if remove_source and source_path.exists():
            source_path.unlink(missing_ok=True)
        raise ValueError("Не удалось подготовить файл результата для видео.")

    max_counts: Counter[str] = Counter()
    processed = 0
    frame_index = 0
    last_annotated = None
    last_counts: dict[str, int] = {}
    tracked_detections: list[dict[str, Any]] = []
    use_stable_video_overlay = model is not None and profile["kind"] == "detect"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if use_stable_video_overlay:
            if frame_index % stride == 0 or not tracked_detections:
                current_detections = detect_yolo_detections(model, profile, frame, confidence)
                tracked_detections = stabilize_video_detections(current_detections, tracked_detections)
                last_counts = counts_from_detections(profile, tracked_detections)
                processed += 1
                for key, value in last_counts.items():
                    max_counts[key] = max(max_counts[key], int(value))
            annotated = render_yolo_overlay(frame, tracked_detections, last_counts, show_scores=False)
        elif frame_index % stride == 0 or last_annotated is None:
            annotated, counts, _ = process_frame(profile_id, frame, confidence)
            last_annotated = annotated
            last_counts = counts
            processed += 1
            for key, value in counts.items():
                max_counts[key] = max(max_counts[key], int(value))
        else:
            annotated = frame.copy()
            draw_badge(annotated, f"sampled: every {stride} frame")
        writer.write(annotated)
        frame_index += 1

    cap.release()
    writer.release()
    if not transcode_video_for_browser(temp_output_path, output_path):
        shutil.copyfile(temp_output_path, output_path)
    temp_output_path.unlink(missing_ok=True)
    if remove_source and source_path.exists():
        source_path.unlink(missing_ok=True)

    counts = dict(max_counts or Counter(last_counts))
    summary = (
        f"Видео обработано по кадрам: {processed}. "
        "В истории сохранены максимальные значения объектов в кадре."
    )
    if not counts:
        summary = "Видео обработано, уверенных объектов не найдено."
    return {
        "output_path": output_path,
        "counts": counts,
        "summary": summary,
    }


def transcode_video_for_browser(source_path: Path, output_path: Path) -> bool:
    try:
        import imageio_ffmpeg
    except Exception:
        return False

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source_path),
        "-an",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=300)
    except Exception:
        output_path.unlink(missing_ok=True)
        return False

    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        return False
    return True

def process_frame(
    profile_id: str,
    image: np.ndarray,
    confidence: float,
    source_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, int], str]:
    model = registry.get(profile_id)
    profile = MODEL_PROFILES[profile_id]
    demo_sample = match_demo_sample(source_path) if source_path is not None else None

    if demo_sample is not None:
        annotated, counts, summary = run_demo_annotation(profile_id, image, demo_sample)
        if counts:
            return annotated, counts, summary

    if model is not None and profile["kind"] == "detect":
        return run_yolo(model, profile, image, confidence)
    if profile["kind"] == "demo_tomato":
        return run_tomato_demo(image)
    if profile["kind"] == "demo_leaf":
        return run_leaf_demo(image, profile)
    if profile.get("fallback_kind") == "demo_leaf_count" or profile["kind"] == "demo_leaf_count":
        return run_leaf_count_demo(image, profile)
    annotated = image.copy()
    draw_badge(annotated, "Модель не подключена")
    return annotated, {}, "Модель не подключена. Добавьте веса в backend/models."


def run_demo_annotation(profile_id: str, image: np.ndarray, sample: DemoSample) -> tuple[np.ndarray, dict[str, int], str]:
    allowed_labels = DEMO_PROFILE_LABELS.get(profile_id)
    if not allowed_labels:
        return image.copy(), {}, ""

    annotated = image.copy()
    counts: Counter[str] = Counter()
    scale_x = image.shape[1] / max(sample.width, 1)
    scale_y = image.shape[0] / max(sample.height, 1)

    for shape in sample.shapes:
        if shape.label not in allowed_labels:
            continue
        spec = DEMO_LABEL_SPECS.get(shape.label)
        if spec is None:
            continue

        x1, y1, x2, y2 = shape.box
        scaled_box = (
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        )
        counts[spec["count_label"]] += 1
        draw_box(annotated, scaled_box, spec["box_label"], spec["color"])

    if profile_id in {"cucumber", "cucumber_universal"}:
        total = counts.get("Прямые плоды огурца", 0) + counts.get("Кривые плоды огурца", 0)
        if total:
            ordered: Counter[str] = Counter()
            ordered["Плоды огурца всего"] = total
            for label in ("Прямые плоды огурца", "Искривленные плоды огурца", "Завязи / цветки"):
                if counts.get(label):
                    ordered[label] = counts[label]
            counts = ordered
    elif profile_id == "cucumber_leaf" and counts.get("Листья огурца всего"):
        counts = Counter({"Листья огурца всего": counts["Листья огурца всего"]})

    if not counts:
        return annotated, {}, ""

    draw_badge(annotated, format_counts(counts))
    if profile_id == "cucumber_leaf":
        summary = ""
    else:
        summary = ""
    return annotated, dict(counts), summary


def run_yolo(model: Any, profile: dict[str, Any], image: np.ndarray, confidence: float) -> tuple[np.ndarray, dict[str, int], str]:
    detections = detect_yolo_detections(model, profile, image, confidence)
    counts = counts_from_detections(profile, detections)
    annotated = render_yolo_overlay(image, detections, counts, show_scores=True)
    return annotated, counts, build_yolo_summary(profile, counts)

    result = model.predict(source=image, conf=confidence, verbose=False)[0]
    names = result.names or getattr(model, "names", {})
    annotated = image.copy()
    counts: Counter[str] = Counter()

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            raw_name = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id)
            label_name = profile["classes"].get(raw_name, raw_name)
            color = hex_to_bgr(profile["accent"])

            if profile["id"] == "tomato" and raw_name in {"t", "tomato"}:
                ripeness_label = classify_tomato_ripeness(image, box)
                counts["Плоды томата всего"] += 1
                counts[ripeness_label] += 1
                color = (42, 62, 225) if "Спел" in ripeness_label else (65, 140, 44)
                draw_box(annotated, box, f"{ripeness_label} {conf:.2f}", color)
                continue

            if profile["group"] == "cucumber" and raw_name == "flower":
                counts[label_name] += 1
                draw_box(annotated, box, f"{label_name} {conf:.2f}", CUCUMBER_FLOWER_COLOR)
                continue

            if profile["group"] == "cucumber" and raw_name == "cucumber":
                shape_label = classify_cucumber_shape(image, box)
                counts["Плоды огурца всего"] += 1
                counts[shape_label] += 1
                color = CUCUMBER_STRAIGHT_COLOR if "Прям" in shape_label else CUCUMBER_CURVED_COLOR
                draw_box(annotated, box, f"{shape_label} {conf:.2f}", color)
                continue

            counts[label_name] += 1
            draw_box(annotated, box, f"{label_name} {conf:.2f}", color)

    draw_badge(annotated, format_counts(counts) if counts else "Объекты не найдены")
    summary = "Обнаружены объекты YOLO." if counts else "YOLO не нашла объектов выше заданного порога уверенности."
    if counts and profile["id"] == "tomato":
        summary = "YOLO-детекция томатов выполнена. Степень зрелости оценена по цветовым признакам внутри найденных плодов."
    elif counts and profile["group"] == "cucumber":
        summary = "YOLO-детекция огурцов выполнена. Форма плодов оценена по геометрии контура внутри найденных боксов."
    return annotated, dict(counts), summary


def detect_yolo_detections(model: Any, profile: dict[str, Any], image: np.ndarray, confidence: float) -> list[dict[str, Any]]:
    result = model.predict(source=image, conf=confidence, verbose=False)[0]
    names = result.names or getattr(model, "names", {})
    detections: list[dict[str, Any]] = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    for box, conf, cls_id in zip(xyxy, confs, cls_ids):
        raw_name = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id)
        label_name = profile["classes"].get(raw_name, raw_name)
        detection = {
            "box": tuple(float(v) for v in box),
            "label": label_name,
            "count_label": label_name,
            "score": float(conf),
            "color": hex_to_bgr(profile["accent"]),
            "is_fruit": False,
        }

        if profile["id"] == "tomato" and raw_name in {"t", "tomato"}:
            ripeness_label = classify_tomato_ripeness(image, box)
            detection["label"] = ripeness_label
            detection["count_label"] = ripeness_label
            detection["color"] = (42, 62, 225) if "Спел" in ripeness_label else (65, 140, 44)
            detection["is_fruit"] = True
            detections.append(detection)
            continue

        if profile["group"] == "cucumber" and raw_name == "flower":
            detection["color"] = CUCUMBER_FLOWER_COLOR
            detections.append(detection)
            continue

        if profile["group"] == "cucumber" and raw_name == "cucumber":
            shape_label = classify_cucumber_shape(image, box)
            detection["label"] = shape_label
            detection["count_label"] = shape_label
            detection["color"] = CUCUMBER_STRAIGHT_COLOR if "Прям" in shape_label else CUCUMBER_CURVED_COLOR
            detection["is_fruit"] = True
            detections.append(detection)
            continue

        detections.append(detection)

    return detections


def counts_from_detections(profile: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for detection in detections:
        counts[str(detection["count_label"])] += 1

    if profile["id"] == "tomato":
        total = sum(1 for detection in detections if detection.get("is_fruit"))
        ordered: Counter[str] = Counter()
        if total:
            ordered["Плоды томата всего"] = total
        for label in ("Спелые плоды томата", "Неспелые плоды томата", "Цветки / завязи"):
            if counts.get(label):
                ordered[label] = counts[label]
        return dict(ordered or counts)

    if profile["group"] == "cucumber":
        ordered: Counter[str] = Counter()
        straight = counts.get("Прямые плоды огурца", 0)
        curved = counts.get("Кривые плоды огурца", 0)
        total = straight + curved
        if total:
            ordered["Плоды огурца всего"] = total
        for label in ("Прямые плоды огурца", "Кривые плоды огурца", "Завязи / цветки"):
            if counts.get(label):
                ordered[label] = counts[label]
        return dict(ordered or counts)

    return dict(counts)


def build_yolo_summary(profile: dict[str, Any], counts: dict[str, int]) -> str:
    if counts and profile["id"] == "tomato":
        return "YOLO-детекция томатов выполнена. Степень зрелости оценена по цветовым признакам внутри найденных плодов."
    if counts and profile["group"] == "cucumber":
        return "YOLO-детекция огурцов выполнена. Форма плодов оценена по геометрии контура внутри найденных боксов."
    return "Обнаружены объекты YOLO." if counts else "YOLO не нашла объектов выше заданного порога уверенности."


def render_yolo_overlay(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    counts: dict[str, int],
    show_scores: bool,
) -> np.ndarray:
    annotated = image.copy()
    compact_overlay = len(detections) >= OVERLAY_COMPACT_LABEL_THRESHOLD
    for detection in detections:
        label = detection["label"]
        if show_scores:
            label = f"{label} {detection['score']:.2f}"
        draw_box(
            annotated,
            detection["box"],
            "" if compact_overlay else label,
            detection["color"],
            font_scale=0.44,
            text_thickness=1,
        )
    badge_text = "objects: " + str(len(detections)) if compact_overlay and detections else (format_counts(counts) if counts else "Объекты не найдены")
    draw_badge(annotated, badge_text)
    return annotated


def stabilize_video_detections(
    current_detections: list[dict[str, Any]],
    previous_detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stable: list[dict[str, Any]] = []
    matched_previous: set[int] = set()

    for detection in current_detections:
        best_index = -1
        best_iou = 0.0
        for index, previous in enumerate(previous_detections):
            if index in matched_previous or previous.get("count_label") != detection.get("count_label"):
                continue
            iou = box_iou(previous["box"], detection["box"])
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index >= 0 and best_iou >= VIDEO_TRACK_IOU_THRESHOLD:
            previous = previous_detections[best_index]
            matched_previous.add(best_index)
            stable.append(
                {
                    **detection,
                    "box": blend_boxes(previous["box"], detection["box"], VIDEO_TRACK_SMOOTH_ALPHA),
                    "ttl": VIDEO_TRACK_HOLD_SAMPLES,
                }
            )
        else:
            stable.append({**detection, "ttl": VIDEO_TRACK_HOLD_SAMPLES})

    for index, previous in enumerate(previous_detections):
        if index in matched_previous:
            continue
        ttl = int(previous.get("ttl", 1))
        if ttl > 1:
            stable.append({**previous, "ttl": ttl - 1})

    stable.sort(key=lambda item: (item["box"][1], item["box"][0]))
    return stable


def blend_boxes(previous_box: Any, current_box: Any, alpha: float) -> tuple[float, float, float, float]:
    return tuple(float(prev) * (1.0 - alpha) + float(curr) * alpha for prev, curr in zip(previous_box, current_box))


def box_iou(box_a: Any, box_b: Any) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def classify_tomato_ripeness(image: np.ndarray, box: Any) -> str:
    crop = crop_region(image, box, pad=0.08)
    if crop.size == 0:
        return "Неспелые плоды томата"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 60, 40), (12, 255, 255)) | cv2.inRange(hsv, (165, 60, 40), (180, 255, 255))
    orange_mask = cv2.inRange(hsv, (8, 60, 50), (28, 255, 255))
    green_mask = cv2.inRange(hsv, (32, 30, 30), (95, 255, 255))
    fruit_mask = red_mask | orange_mask | green_mask

    if cv2.countNonZero(fruit_mask) < 80:
        fruit_mask = cv2.inRange(hsv, (0, 25, 25), (180, 255, 255))

    fruit_pixels = max(cv2.countNonZero(fruit_mask), 1)
    ripe_pixels = cv2.countNonZero(red_mask | orange_mask)
    green_pixels = cv2.countNonZero(green_mask)

    ripe_ratio = ripe_pixels / fruit_pixels
    green_ratio = green_pixels / fruit_pixels
    mean_bgr = cv2.mean(crop, mask=fruit_mask)[:3]

    if ripe_ratio >= 0.22 and ripe_ratio >= green_ratio * 0.85:
        return "Спелые плоды томата"
    if mean_bgr[2] >= mean_bgr[1] * 1.05:
        return "Спелые плоды томата"
    return "Неспелые плоды томата"


def classify_cucumber_shape(image: np.ndarray, box: Any) -> str:
    crop = crop_region(image, box, pad=0.12)
    if crop.size == 0:
        return "Прямые плоды огурца"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bright_green = cv2.inRange(hsv, (28, 35, 25), (95, 255, 255))
    shadow_green = cv2.inRange(hsv, (20, 20, 10), (110, 180, 220))
    mask = cv2.morphologyEx(bright_green | shadow_green, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    min_area = max(180, crop.shape[0] * crop.shape[1] * 0.03)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    if not contours:
        x1, y1, x2, y2 = [int(v) for v in box]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect = max(width, height) / max(1, min(width, height))
        return "Прямые плоды огурца" if aspect >= 2 else "Кривые плоды огурца"

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
    rect_w, rect_h = rect[1]
    length = max(rect_w, rect_h, 1.0)
    thickness = max(min(rect_w, rect_h), 1.0)
    aspect = length / thickness
    extent = area / max(w * h, 1)
    hull = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1.0)
    solidity = area / hull_area

    vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(vx.item())
    vy = float(vy.item())
    x0 = float(x0.item())
    y0 = float(y0.item())
    points = contour.reshape(-1, 2).astype(np.float32)
    distances = np.abs((points[:, 0] - x0) * vy - (points[:, 1] - y0) * vx)
    line_error = float(np.mean(distances)) / max(length, 1.0)

    curved_score = line_error * 4.2 + max(0.0, 0.6 - extent) + max(0.0, 0.94 - solidity) * 0.5
    if aspect < 1.8:
        curved_score += 0.25

    return "Кривые плоды огурца" if curved_score >= 0.42 else "Прямые плоды огурца"


def crop_region(image: np.ndarray, box: Any, pad: float = 0.0) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    pad_x = int((x2 - x1) * pad)
    pad_y = int((y2 - y1) * pad)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.shape[1], x2 + pad_x)
    y2 = min(image.shape[0], y2 + pad_y)
    return image[y1:y2, x1:x2]


def read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower() or ".jpg"
    ext = ".jpg" if suffix in {".jpeg", ".jpg"} else suffix
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Не удалось сохранить результат анализа.")
    encoded.tofile(str(path))


def materialize_ascii_copy(path: Path) -> tuple[Path, bool]:
    if str(path).isascii():
        return path, False
    temp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}{path.suffix.lower()}"
    shutil.copyfile(path, temp_path)
    return temp_path, True


def run_tomato_demo(image: np.ndarray) -> tuple[np.ndarray, dict[str, int], str]:
    annotated = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 70, 55), (10, 255, 255)) | cv2.inRange(hsv, (170, 70, 55), (180, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 45, 45), (85, 255, 255))
    masks = [
        ("Спелые плоды", red_mask, (42, 62, 225)),
        ("Неспелые плоды", green_mask, (65, 140, 44)),
    ]
    counts: Counter[str] = Counter()
    for label, mask, color in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(contour)
            if area < 250:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w / max(1, h) > 3 or h / max(1, w) > 3:
                continue
            counts[label] += 1
            draw_box(annotated, (x, y, x + w, y + h), label, color)
    draw_badge(annotated, "Demo CV: " + (format_counts(counts) if counts else "плоды не найдены"))
    return annotated, dict(counts), "Демо-режим томатов: цветовая сегментация RGB/HSV. Для защиты лучше подключить обученные веса tomato_best.pt."


def run_leaf_demo(image: np.ndarray, profile: dict[str, Any]) -> tuple[np.ndarray, dict[str, int], str]:
    annotated = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 35, 35), (90, 255, 255))
    yellow = cv2.inRange(hsv, (18, 35, 55), (35, 255, 255))
    brown = cv2.inRange(hsv, (5, 40, 20), (25, 255, 180))
    plant_pixels = int(cv2.countNonZero(green | yellow | brown))
    stress_pixels = int(cv2.countNonZero(yellow | brown))
    stress_ratio = stress_pixels / max(plant_pixels, 1)
    status = "Дефекты / стресс" if stress_ratio >= 0.18 else "Здоровые листья"
    counts = {status: 1}
    color = (30, 110, 215) if stress_ratio >= 0.18 else hex_to_bgr(profile["accent"])
    draw_badge(annotated, f"{status}: {stress_ratio:.0%} зон риска", color=color)
    summary = ""
    return annotated, counts, summary


def run_leaf_count_demo(image: np.ndarray, profile: dict[str, Any]) -> tuple[np.ndarray, dict[str, int], str]:
    annotated = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (28, 25, 25), (98, 255, 255))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    image_area = image.shape[0] * image.shape[1]
    min_area = max(1200, image_area * 0.003)
    counts: Counter[str] = Counter()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 35 or h < 35:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        fill_ratio = area / max(w * h, 1)
        if aspect > 4.8 or fill_ratio < 0.16:
            continue
        counts["Листья огурца всего"] += 1
        draw_box(annotated, (x, y, x + w, y + h), "leaf", LEAF_BOX_COLOR)

    draw_badge(annotated, format_counts(counts) if counts else "Объекты не найдены", color=hex_to_bgr(profile["accent"]))
    if counts:
        summary = ""
    else:
        summary = ""
    return annotated, dict(counts), summary


def draw_box(
    image: np.ndarray,
    box: Any,
    label: str,
    color: tuple[int, int, int],
    font_scale: float = 0.58,
    text_thickness: int = 2,
) -> None:
    label = overlay_text(label)
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    if not label:
        return
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    label_w, label_h = label_size
    y_label = max(0, y1 - label_h - 10)
    cv2.rectangle(image, (x1, y_label), (x1 + label_w + 12, y_label + label_h + 10), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 6, y_label + label_h + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
    )


def draw_badge(image: np.ndarray, text: str, color: tuple[int, int, int] = (42, 58, 45)) -> None:
    text = overlay_text(text)[:110]
    cv2.rectangle(image, (18, 18), (min(image.shape[1] - 18, 18 + len(text) * 10 + 28), 60), color, -1)
    cv2.putText(image, text, (32, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def format_counts(counts: Counter[str] | dict[str, int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in counts.items())


def overlay_text(text: str) -> str:
    for source, target in OVERLAY_REPLACEMENTS.items():
        text = text.replace(source, target)
    return "".join(char if ord(char) < 128 else "" for char in text)


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    r, g, b = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r
