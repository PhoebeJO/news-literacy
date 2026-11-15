"""SQLite 資料層。表格對應單元 6 報告三-4 的 ERD（正式版換成 Supabase/pgvector）。"""
import sqlite3
from datetime import datetime, timezone, timedelta

from .config import DB_PATH, DATA_DIR, SOURCES

TW = timezone(timedelta(hours=8))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY,
    line_uid_hash TEXT UNIQUE NOT NULL,   -- 雜湊，不存原始 ID
    click_token   TEXT UNIQUE NOT NULL,   -- 點擊追蹤用的隨機代號
    chosen_topics TEXT DEFAULT '',        -- 冷啟動選的領域，逗號分隔
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    source_id INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    country   TEXT NOT NULL,
    ownership TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id     INTEGER PRIMARY KEY,
    label        TEXT NOT NULL,
    first_seen   TEXT NOT NULL,
    source_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS articles (
    article_id   INTEGER PRIMARY KEY,
    source_id    INTEGER NOT NULL REFERENCES sources(source_id),
    event_id     INTEGER,
    title        TEXT NOT NULL,
    url          TEXT UNIQUE NOT NULL,
    summary      TEXT NOT NULL,           -- 只存 RSS 摘要，不存全文
    topic_label  TEXT,
    topic_by     TEXT,                    -- feed = 來自 RSS 分類；knn = 向量最近質心
    embedding    BLOB,
    published_at TEXT NOT NULL,
    fetched_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS interactions (
    id         INTEGER PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(user_id),
    article_id INTEGER NOT NULL REFERENCES articles(article_id),
    action     TEXT NOT NULL,             -- click / like / dislike / other_view
    simulated  INTEGER DEFAULT 0,         -- 1 = 「示範紀錄」產生的假資料
    at         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendations (
    id            INTEGER PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(user_id),
    article_id    INTEGER NOT NULL REFERENCES articles(article_id),
    reason        TEXT NOT NULL,
    is_other_view INTEGER NOT NULL,
    sent_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_checks (
    id               INTEGER PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(user_id),
    claim            TEXT NOT NULL,
    cofacts_category TEXT,
    match_count      INTEGER NOT NULL,
    at               TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS query_log (
    id      INTEGER PRIMARY KEY,
    user_id INTEGER,
    intent  TEXT NOT NULL,
    by      TEXT NOT NULL,                -- rule / llm
    at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_usage (
    id         INTEGER PRIMARY KEY,
    purpose    TEXT NOT NULL,
    model      TEXT NOT NULL,
    tokens_in  INTEGER,
    tokens_out INTEGER,
    at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_pub ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_inter_user ON interactions(user_id, at);
"""


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse(s: str) -> datetime:
    return datetime.fromisoformat(s)


def tw_str(s: str) -> str:
    """UTC ISO 字串 → 台灣時間 'MM/DD HH:MM'。"""
    return parse(s).astimezone(TW).strftime("%m/%d %H:%M")


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        for s in SOURCES:
            conn.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?)",
                (s["id"], s["name"], s["country"], s["ownership"]),
            )
