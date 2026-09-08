from __future__ import annotations

from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND_DIR / "models"
STORAGE_DIR = BACKEND_DIR / "storage"
DB_PATH = BACKEND_DIR / "smart_farm.sqlite3"

UPLOAD_DIR = STORAGE_DIR / "uploads"
OUTPUT_DIR = STORAGE_DIR / "outputs"

FREE_UPLOAD_LIMIT = 3

MODEL_PROFILES = {
    "cucumber": {
        "id": "cucumber",
        "group": "cucumber",
        "title": "Огурцы",
        "short_title": "Огурцы custom",
        "task": "Детекция плодов огурца, подсчет завязей и разделение на прямые и кривые плоды",
        "kind": "detect",
        "weights": MODELS_DIR / "customized_teplitza_cucumber.pt",
        "classes": {
            "flower": "Завязи / цветки",
            "cucumber": "Плоды огурца",
        },
        "accent": "#1f9d63",
        "mode_note": "Веса YOLO подключены",
        "confidence_default": 0.10,
        "confidence_min": 0.05,
        "confidence_max": 0.95,
        "confidence_step": 0.05,
        "confidence_note": "Порог уверенности отсеивает слабые детекции: высокий порог делает результат чище, но повышает риск пропусков.",
        "confidence_range_note": "",
    },
    "cucumber_universal": {
        "id": "cucumber_universal",
        "group": "cucumber",
        "title": "Огурцы, универсальная модель",
        "short_title": "Огурцы universal",
        "task": "Сравнительная детекция плодов, оценка формы и подсчет завязей",
        "kind": "detect",
        "weights": MODELS_DIR / "universal_cucumber.pt",
        "classes": {
            "flower": "Завязи / цветки",
            "cucumber": "Плоды огурца",
        },
        "accent": "#3b82f6",
        "mode_note": "Веса YOLO подключены",
        "confidence_default": 0.15,
        "confidence_min": 0.05,
        "confidence_max": 0.95,
        "confidence_step": 0.05,
        "confidence_note": "Порог уверенности нужен для балансировки между полнотой и чистотой детекции, особенно на кадрах с перекрытием листьев и мелкими завязями.",
        "confidence_range_note": "",
    },
    "tomato": {
        "id": "tomato",
        "group": "tomato",
        "title": "Томаты",
        "short_title": "Томаты",
        "task": "Детекция и подсчет плодов, оценка степени зрелости",
        "kind": "detect",
        "weights": MODELS_DIR / "tomato_best.pt",
        "classes": {
            "t": "Плоды томата",
            "f": "Цветки / завязи",
            "tomato": "Плоды томата",
            "ripe_tomato": "Спелые плоды томата",
            "unripe_tomato": "Неспелые плоды томата",
        },
        "accent": "#e14b37",
        "mode_note": "Веса YOLO подключены",
        "confidence_default": 0.25,
        "confidence_min": 0.05,
        "confidence_max": 0.95,
        "confidence_step": 0.05,
        "confidence_note": "Порог уверенности позволяет управлять компромиссом между пропусками и ложными срабатываниями при подсчете плодов.",
        "confidence_range_note": "",
    },
    "lettuce": {
        "id": "lettuce",
        "group": "lettuce",
        "title": "Листовой салат",
        "short_title": "Салат",
        "task": "Классификация здоровья листьев",
        "kind": "demo_leaf",
        "weights": MODELS_DIR / "lettuce_health.pt",
        "classes": {
            "healthy": "Здоровые листья",
            "deficit": "Признаки дефицита питания",
        },
        "accent": "#6a9f2b",
        "mode_note": "Базовый CV-режим",
        "confidence_default": 0.25,
        "confidence_min": 0.05,
        "confidence_max": 0.95,
        "confidence_step": 0.05,
        "confidence_note": "",
        "confidence_range_note": "",
    },
    "cucumber_leaf": {
        "id": "cucumber_leaf",
        "group": "cucumber_leaf",
        "title": "Листья огурца",
        "short_title": "Листья огурца",
        "task": "Детекция и подсчет листьев огурца",
        "kind": "detect",
        "fallback_kind": "demo_leaf_count",
        "weights": MODELS_DIR / "cucumber_leaf.pt",
        "classes": {
            "leaf": "Листья огурца всего",
        },
        "accent": "#0f766e",
        "mode_note": "Веса YOLO подключены",
        "confidence_default": 0.25,
        "confidence_min": 0.05,
        "confidence_max": 0.95,
        "confidence_step": 0.05,
        "confidence_note": "Порог уверенности помогает управлять компромиссом между полнотой подсчета листьев и количеством ложных боксов на фоне перекрытия и бликов.",
        "confidence_range_note": "",
    },
    "strawberry": {
        "id": "strawberry",
        "group": "strawberry",
        "title": "Клубника",
        "short_title": "Клубника",
        "task": "Детекция и подсчет ягод",
        "kind": "detect",
        "weights": MODELS_DIR / "strawberry_best2.pt",
        "classes": {
            "berry": "Ягоды",
        },
        "accent": "#c02652",
        "mode_note": "Веса YOLO подключены",
        "confidence_default": 0.25,
        "confidence_min": 0.05,
        "confidence_max": 0.95,
        "confidence_step": 0.05,
        "confidence_note": "Порог уверенности помогает отсеивать слабые срабатывания на фоне листвы и бликов.",
        "confidence_range_note": "",
    },
}


def ensure_runtime_dirs() -> None:
    for path in (STORAGE_DIR, UPLOAD_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)
