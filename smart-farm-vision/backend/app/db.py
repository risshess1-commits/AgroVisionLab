from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DB_PATH, FREE_UPLOAD_LIMIT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_paid INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL,
                media_type TEXT NOT NULL,
                original_name TEXT NOT NULL,
                upload_path TEXT NOT NULL,
                output_path TEXT NOT NULL,
                counts_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                confidence REAL NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return digest, salt


def login_or_create(company: str, password: str) -> dict[str, Any]:
    company = company.strip()
    if not company:
        raise ValueError("Название компании не может быть пустым.")
    if len(password) < 3:
        raise ValueError("Пароль должен содержать не менее 3 символов.")

    with connect() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE lower(company) = lower(?)",
            (company,),
        ).fetchone()
        if user is None:
            digest, salt = hash_password(password)
            cur = conn.execute(
                """
                INSERT INTO users(company, password_hash, salt, is_paid, created_at)
                VALUES (?, ?, ?, 0, ?)
                """,
                (company, digest, salt, utc_now()),
            )
            user_id = cur.lastrowid
        else:
            expected, _ = hash_password(password, user["salt"])
            if not secrets.compare_digest(expected, user["password_hash"]):
                raise PermissionError("Неверный пароль.")
            user_id = int(user["id"])

        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO sessions(token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, utc_now()),
        )
        return get_user_payload(conn, user_id, token)


def get_user_payload(conn: sqlite3.Connection, user_id: int, token: str | None = None) -> dict[str, Any]:
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    used = conn.execute("SELECT COUNT(*) AS c FROM analyses WHERE user_id = ?", (user_id,)).fetchone()["c"]
    payload = {
        "id": user["id"],
        "company": user["company"],
        "is_paid": bool(user["is_paid"]),
        "used_uploads": int(used),
        "free_upload_limit": FREE_UPLOAD_LIMIT,
        "remaining_uploads": max(0, FREE_UPLOAD_LIMIT - int(used)) if not user["is_paid"] else None,
    }
    if token:
        payload["token"] = token
    return payload


def get_user_by_token(token: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT users.id
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()
        if row is None:
            return None
        return get_user_payload(conn, int(row["id"]))


def activate_payment(user_id: int) -> dict[str, Any]:
    with connect() as conn:
        conn.execute("UPDATE users SET is_paid = 1 WHERE id = ?", (user_id,))
        return get_user_payload(conn, user_id)


def can_analyze(user_id: int) -> bool:
    with connect() as conn:
        user = conn.execute("SELECT is_paid FROM users WHERE id = ?", (user_id,)).fetchone()
        if user and user["is_paid"]:
            return True
        used = conn.execute("SELECT COUNT(*) AS c FROM analyses WHERE user_id = ?", (user_id,)).fetchone()["c"]
        return int(used) < FREE_UPLOAD_LIMIT


def remaining_uploads(user_id: int) -> int | None:
    with connect() as conn:
        user = conn.execute("SELECT is_paid FROM users WHERE id = ?", (user_id,)).fetchone()
        if user and user["is_paid"]:
            return None
        used = conn.execute("SELECT COUNT(*) AS c FROM analyses WHERE user_id = ?", (user_id,)).fetchone()["c"]
        return max(0, FREE_UPLOAD_LIMIT - int(used))


def insert_analysis(user_id: int, record: dict[str, Any]) -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO analyses(
                user_id, profile_id, media_type, original_name, upload_path,
                output_path, counts_json, summary, confidence, elapsed_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                record["profile_id"],
                record["media_type"],
                record["original_name"],
                record["upload_path"],
                record["output_path"],
                record["counts_json"],
                record["summary"],
                float(record["confidence"]),
                int(record["elapsed_ms"]),
                utc_now(),
            ),
        )
        return int(cur.lastrowid)


def list_history(user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, profile_id, media_type, original_name, output_path, counts_json,
                   summary, confidence, elapsed_ms, created_at
            FROM analyses
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def get_analysis(user_id: int, analysis_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, profile_id, media_type, original_name, output_path, counts_json,
                   summary, confidence, elapsed_ms, created_at
            FROM analyses
            WHERE user_id = ? AND id = ?
            """,
            (user_id, analysis_id),
        ).fetchone()
        return dict(row) if row else None


def get_stats(user_id: int) -> dict[str, Any]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT profile_id, media_type, counts_json, elapsed_ms, created_at
            FROM analyses
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,),
        ).fetchall()

    by_profile: dict[str, int] = {}
    by_media: dict[str, int] = {}
    objects: dict[str, int] = {}
    total_elapsed = 0

    for row in rows:
        by_profile[row["profile_id"]] = by_profile.get(row["profile_id"], 0) + 1
        by_media[row["media_type"]] = by_media.get(row["media_type"], 0) + 1
        total_elapsed += int(row["elapsed_ms"])
        try:
            counts = json.loads(row["counts_json"])
        except json.JSONDecodeError:
            counts = {}
        for label, value in counts.items():
            objects[label] = objects.get(label, 0) + int(value)

    total = len(rows)
    return {
        "total_analyses": total,
        "avg_elapsed_ms": round(total_elapsed / total) if total else 0,
        "by_profile": by_profile,
        "by_media": by_media,
        "objects": objects,
    }
