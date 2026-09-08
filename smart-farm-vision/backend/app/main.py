from __future__ import annotations

import json
import shutil
import uuid
from io import StringIO
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import MODEL_PROFILES, STORAGE_DIR, UPLOAD_DIR, ensure_runtime_dirs
from .db import (
    activate_payment,
    can_analyze,
    get_analysis,
    get_stats,
    get_user_by_token,
    init_db,
    insert_analysis,
    list_history,
    login_or_create,
    remaining_uploads,
)
from .inference import analyze_file, profile_payload


ensure_runtime_dirs()
init_db()

app = FastAPI(title="Smart Farm Vision API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(STORAGE_DIR)), name="media")
FRONTEND_DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class LoginPayload(BaseModel):
    company: str
    password: str


def parse_counts(raw: str | dict | None) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def recommendation_for(profile_id: str, counts: dict) -> str:
    if not counts:
        return "Объекты не обнаружены. Рекомендуется повторить съемку при равномерном освещении и проверить порог уверенности модели."

    normalized = {str(key).lower(): int(value) for key, value in counts.items()}

    if profile_id == "lettuce":
        stress = sum(value for key, value in normalized.items() if "дефицит" in key or "стресс" in key or "дефект" in key)
        if stress:
            return "Зафиксированы признаки стресса листьев. Рекомендуется визуальный осмотр культуры и проверка режима питания и освещения. Урожай с розеток рекомендуется снимать, товарный вид утрачивается, а горечь в листьях нарастает."
        return "Критичных признаков стресса не выявлено. Определена здоровая листовая масса. Рекомендуется продолжить качественное питание и влажность почвы 75–80%, планировать сбор урожая через несколько дней."

    if profile_id == "cucumber_leaf":
        leaves = sum(value for key, value in normalized.items() if "лист" in key)
        if leaves >= 25:
            return "Определена листовая масса в большом количестве. Продолжайте поддерживать сбалансированный показатель концентрации растворённых минеральных солей, необходимых для полноценного роста и развития растений, влажность 75–80% для профилактики паутинного клеща. Сбор урожая рекомендуется проводить ежедневно, чтобы не перегружать побеги и стимулировать генерацию новых завязей."
        if leaves > 0:
            return "Листья огурца обнаружены и подсчитаны. Продолжайте поддерживать сбалансированный показатель концентрации растворённых минеральных солей, необходимых для полноценного роста и развития растений, влажность 75–80% для профилактики паутинного клеща. Сбор урожая рекомендуется проводить ежедневно, чтобы не перегружать побеги и стимулировать генерацию новых завязей."
        return "Листья не были выделены достаточно уверенно. Рекомендована коррекция питания и подкормки растения."

    if profile_id == "tomato":
        ripe = sum(value for key, value in normalized.items() if "спел" in key and "неспел" not in key)
        unripe = sum(value for key, value in normalized.items() if "неспел" in key)
        flowers = sum(value for key, value in normalized.items() if "цвет" in key or "завяз" in key)
        fruits = sum(value for key, value in normalized.items() if "томат" in key and "всего" in key)
        if not fruits:
            fruits = ripe + unripe

        if fruits and ripe and unripe:
            if ripe > unripe:
                return "Преобладают спелые плоды томата, рекомендуется собирать урожай, чтобы освободить место для налива остальных кистей. Снимайте красные плоды каждые 2–3 дня и немного сократите полив, чтобы избежать водянистого вкуса и растрескивания. "
            return "Обнаружены неспелые плоды томата. Подкормите кусты калием и магнием. При хорошем освещении и ночной температуре около 16–18°C массовое покраснение начнётся уже через полторы-две недели."
        if fruits and flowers:
            return "Куст находится в активной стадии. Чтобы растение справилось с такой нагрузкой, обеспечьте равномерный полив без перепадов влажности. Аккуратно удаляйте листья, затеняющие зреющие кисти, но не более 2–3 листов за один подход."
        if fruits:
            return "Плоды томата обнаружены уверенно. Чтобы растение справилось с такой нагрузкой, обеспечьте равномерный полив без перепадов влажности. Аккуратно удаляйте листья, затеняющие зреющие кисти, но не более 2–3 листов за один подход."
        return "Для томатов стоит повторить съемку на более контрастном кадре, чтобы улучшить разбор зрелости и количества плодов."

    if profile_id.startswith("cucumber"):
        fruits = sum(value for key, value in normalized.items() if "огур" in key and "всего" in key)
        if not fruits:
            fruits = sum(value for key, value in normalized.items() if "огур" in key and "прям" not in key and "крив" not in key)
        flowers = sum(value for key, value in normalized.items() if "завяз" in key or "цвет" in key)
        straight = sum(value for key, value in normalized.items() if "прям" in key)
        curved = sum(value for key, value in normalized.items() if "крив" in key)

        if curved and curved >= max(1, straight):
            return "Зафиксирована заметная доля искривленных плодов. Крючковатые и грушевидные зеленцы часто сигналят о несбалансированном питании или нерегулярном поливе. Рекомендуется применить калийную подкормку, временно убрав избыток азота. "
        if flowers > fruits and fruits:
            return "Завязей больше, чем сформированных плодов. Рекомендуется тщательно следить за режимом полива и усилить фосфорно-калийное питание. Полезно повторить мониторинг через 1-2 дня и сравнить динамику роста."
        if fruits:
            return "Плоды огурца обнаружены уверенно. Рекомендуется тщательно следить за водным режимом и усилить фосфорно-калийное питание."
        return "Для огурцов полезно повторить съемку крупнее или под другим углом, чтобы форма плодов читалась стабильнее."

    if profile_id == "strawberry":
        return "Куст в стадии активного плодоношения с высоким процентом окрашенных плодов, убедитесь в отсутствии серой гнили под прилистниками. Немедленно соберите спелые ягоды, чтобы избежать перезревания."

    return "Результат сохранен. Используйте историю обработок для сравнения динамики по датам."


def attach_analysis_extras(row: dict) -> dict:
    counts = parse_counts(row.get("counts_json"))
    row["counts"] = counts
    row["output_url"] = public_url(row["output_path"])
    row["report_url"] = f"/api/analysis/{row['id']}/report"
    row["recommendation"] = recommendation_for(row["profile_id"], counts)
    return row


def public_url(path: str | Path) -> str:
    path = Path(path)
    try:
        rel = path.resolve().relative_to(STORAGE_DIR.resolve())
    except ValueError:
        rel = path.name
    return "/media/" + str(rel).replace("\\", "/")


def current_user(authorization: Annotated[str | None, Header()] = None) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Требуется вход.")
    token = authorization.split(" ", 1)[1].strip()
    user = get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Сессия не найдена.")
    user["token"] = token
    return user


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/auth/login")
def login(payload: LoginPayload) -> dict:
    try:
        return login_or_create(payload.company, payload.password)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/me")
def me(user: dict = Depends(current_user)) -> dict:
    return user


@app.get("/api/profiles")
def profiles() -> list[dict]:
    return profile_payload()


@app.post("/api/payment/activate")
def payment(user: dict = Depends(current_user)) -> dict:
    return activate_payment(int(user["id"]))


@app.get("/api/history")
def history(user: dict = Depends(current_user)) -> list[dict]:
    rows = list_history(int(user["id"]))
    for row in rows:
        attach_analysis_extras(row)
    return rows


@app.get("/api/analysis/{analysis_id}/report")
def analysis_report(analysis_id: int, user: dict = Depends(current_user)) -> StreamingResponse:
    row = get_analysis(int(user["id"]), analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Анализ не найден.")

    row = attach_analysis_extras(row)
    profile = MODEL_PROFILES.get(row["profile_id"], {})
    counts_text = "\n".join(f"- {label}: {value}" for label, value in row["counts"].items()) or "- объекты не обнаружены"
    text = f"""AgroVision Lab. Отчет по анализу #{row['id']}

Компания: {user['company']}
Дата обработки: {row['created_at']}
Профиль модели: {profile.get('title', row['profile_id'])}
Задача: {profile.get('task', 'не указана')}
Тип материала: {row['media_type']}
Файл: {row['original_name']}
Порог уверенности: {row['confidence']}
Время обработки: {row['elapsed_ms']} мс

Найденные объекты:
{counts_text}

Краткое резюме:
{row['summary']}

Рекомендация:
{row['recommendation']}
"""
    return StreamingResponse(
        iter(["\ufeff" + text]),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="agrovision_report_{analysis_id}.txt"'},
    )


@app.get("/api/history/export")
def export_history(user: dict = Depends(current_user)) -> StreamingResponse:
    rows = list_history(int(user["id"]), limit=1000)
    buffer = StringIO()
    buffer.write("\ufeff")
    writer = __import__("csv").DictWriter(
        buffer,
        fieldnames=["id", "created_at", "profile_id", "media_type", "original_name", "counts_json", "elapsed_ms", "summary"],
        extrasaction="ignore",
        delimiter=";",
    )
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="agrovision_history.csv"'},
    )


@app.get("/api/stats")
def stats(user: dict = Depends(current_user)) -> dict:
    return get_stats(int(user["id"]))


@app.post("/api/analyze")
def analyze(
    profile_id: Annotated[str, Form()],
    confidence: Annotated[float, Form()] = 0.25,
    file: UploadFile = File(...),
    user: dict = Depends(current_user),
) -> dict:
    if profile_id not in MODEL_PROFILES:
        raise HTTPException(status_code=400, detail="Неизвестный профиль модели.")
    if not can_analyze(int(user["id"])):
        raise HTTPException(status_code=402, detail="Бесплатный лимит 3 загрузок исчерпан.")

    suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
    upload_name = f"{uuid.uuid4().hex}{suffix}"
    upload_path = UPLOAD_DIR / upload_name
    with upload_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    try:
        result = analyze_file(profile_id, upload_path, confidence)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = {
        **result,
        "original_name": file.filename or upload_name,
        "upload_path": str(upload_path),
        "output_path": str(result["output_path"]),
    }
    analysis_id = insert_analysis(int(user["id"]), record)
    fresh_user = get_user_by_token((user.get("token") or "")) if user.get("token") else None
    return {
        "id": analysis_id,
        "profile_id": profile_id,
        "media_type": result["media_type"],
        "counts": result["counts"],
        "summary": result["summary"],
        "recommendation": recommendation_for(profile_id, result["counts"]),
        "elapsed_ms": result["elapsed_ms"],
        "output_url": public_url(result["output_path"]),
        "report_url": f"/api/analysis/{analysis_id}/report",
        "original_name": file.filename or upload_name,
        "model_available": Path(MODEL_PROFILES[profile_id]["weights"]).exists(),
        "me": fresh_user,
    }


@app.post("/api/analyze-batch")
def analyze_batch(
    profile_id: Annotated[str, Form()],
    confidence: Annotated[float, Form()] = 0.25,
    files: list[UploadFile] = File(...),
    user: dict = Depends(current_user),
) -> dict:
    if profile_id not in MODEL_PROFILES:
        raise HTTPException(status_code=400, detail="Неизвестный профиль модели.")
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не переданы.")
    remaining = remaining_uploads(int(user["id"]))
    if remaining is not None and len(files) > remaining:
        raise HTTPException(status_code=402, detail=f"Доступно загрузок: {remaining}. Для пакетной обработки оформите подписку.")

    items = []
    for file in files:
        suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
        upload_name = f"{uuid.uuid4().hex}{suffix}"
        upload_path = UPLOAD_DIR / upload_name
        with upload_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)

        try:
            result = analyze_file(profile_id, upload_path, confidence)
        except Exception as exc:
            items.append({"original_name": file.filename or upload_name, "error": str(exc)})
            continue

        record = {
            **result,
            "original_name": file.filename or upload_name,
            "upload_path": str(upload_path),
            "output_path": str(result["output_path"]),
        }
        analysis_id = insert_analysis(int(user["id"]), record)
        items.append(
            {
                "id": analysis_id,
                "profile_id": profile_id,
                "media_type": result["media_type"],
                "original_name": file.filename or upload_name,
                "counts": result["counts"],
                "summary": result["summary"],
                "recommendation": recommendation_for(profile_id, result["counts"]),
                "elapsed_ms": result["elapsed_ms"],
                "output_url": public_url(result["output_path"]),
                "report_url": f"/api/analysis/{analysis_id}/report",
                "model_available": Path(MODEL_PROFILES[profile_id]["weights"]).exists(),
            }
        )

    fresh_user = get_user_by_token((user.get("token") or "")) if user.get("token") else None
    return {"items": items, "me": fresh_user}


if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")
