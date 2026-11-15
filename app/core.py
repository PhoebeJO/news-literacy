"""共用：使用者、文章向量快取、卡片格式。"""
import hashlib
import secrets
from datetime import timedelta

import numpy as np

from . import db
from .config import EVENT_WINDOW_HOURS
from .embed import from_blob

_corpus = None


def invalidate() -> None:
    global _corpus
    _corpus = None


def corpus() -> dict:
    """有向量的文章一次載入記憶體：rows（依 id）、M（矩陣）、idx（id → 列號）。"""
    global _corpus
    if _corpus is None:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT a.*, s.name AS source, s.country FROM articles a "
                "JOIN sources s USING(source_id) WHERE embedding IS NOT NULL "
                "ORDER BY published_at DESC").fetchall()
        _corpus = {
            "rows": {r["article_id"]: r for r in rows},
            "ids": [r["article_id"] for r in rows],
            "idx": {r["article_id"]: i for i, r in enumerate(rows)},
            "M": (np.vstack([from_blob(r["embedding"]) for r in rows])
                  if rows else np.zeros((0, 1), dtype=np.float32)),
        }
    return _corpus


def recent_ids(C: dict, hours: int = EVENT_WINDOW_HOURS) -> list[int]:
    cutoff = db.iso(db.now() - timedelta(hours=hours))
    return [i for i in C["ids"] if C["rows"][i]["published_at"] >= cutoff]


def get_user(conn, uid: str):
    """uid 是聊天介面給的識別碼（正式版是 LINE userId）；只存雜湊。"""
    h = hashlib.sha256(uid.encode()).hexdigest()
    row = conn.execute("SELECT * FROM users WHERE line_uid_hash=?", (h,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO users (line_uid_hash, click_token, created_at) VALUES (?,?,?)",
            (h, secrets.token_urlsafe(12), db.iso(db.now())))
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE line_uid_hash=?", (h,)).fetchone()
    return row


def card(row, token: str, reason: str = "", other_view: bool = False, score=None) -> dict:
    return {
        "article_id": row["article_id"],
        "title": row["title"],
        "summary": row["summary"][:120],
        "source": row["source"],
        "country": row["country"],
        "topic": row["topic_label"] or "",
        "time": db.tw_str(row["published_at"]),
        "url": f"/r/{row['article_id']}?t={token}",   # 先經過點擊追蹤再轉址
        "reason": reason,
        "other_view": other_view,
        "score": None if score is None else round(float(score), 2),
    }


def source_read_counts(conn, user_id: int, days: int = 7) -> dict:
    cutoff = db.iso(db.now() - timedelta(days=days))
    rows = conn.execute(
        "SELECT s.name, COUNT(DISTINCT i.article_id) n FROM interactions i "
        "JOIN articles a USING(article_id) JOIN sources s USING(source_id) "
        "WHERE i.user_id=? AND i.action IN ('click','like') AND i.at>=? GROUP BY s.name",
        (user_id, cutoff)).fetchall()
    return {r["name"]: r["n"] for r in rows}
