"""離線批次（正式版由 GitHub Actions 每 3 小時執行）：抓新聞 → 向量 → 領域分類 → 事件分群。

單獨執行：python -m app.pipeline
"""
import html
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import httpx
import numpy as np

from . import db
from .config import (SOURCES, TOPICS, EVENT_WINDOW_HOURS, EVENT_SIM_THRESHOLD,
                     FRESH_DAYS)
from .embed import Embedder, article_text, to_blob, from_blob, to_trad, unit

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36"}


# ---------- 1. 抓取 ----------

def _clean(s: str) -> str:
    s = re.sub(r"(?s)^\s*<!\[CDATA\[(.*?)\]\]>\s*$", r"\1", s or "")
    s = html.unescape(re.sub(r"(?s)<[^>]+>", " ", html.unescape(s)))
    return re.sub(r"\s+", " ", s).strip()


def _parse_time(s: str) -> datetime | None:
    s = (s or "").strip()
    try:
        dt = parsedate_to_datetime(s)          # RSS: RFC 822
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))   # Atom / RDF
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_feed(xml: str) -> list[dict]:
    """同時支援 RSS <item>、Atom <entry>（公視）、RDF（DW）。"""
    out = []
    for b in re.findall(r"(?s)<(?:item|entry)\b.*?</(?:item|entry)>", xml):
        t = re.search(r"(?s)<title[^>]*>(.*?)</title>", b)
        l = (re.search(r"(?s)<link[^>]*>\s*(?:<!\[CDATA\[)?\s*(https?://[^<\]\s]+)", b)
             or re.search(r'<link[^>]*href="(https?://[^"]+)"', b))
        d = re.search(r"(?s)<(?:pubDate|updated|dc:date|published)>(.*?)</", b)
        s = re.search(r"(?s)<(?:description|summary)[^>]*>(.*?)</(?:description|summary)>", b)
        when = _parse_time(d.group(1)) if d else None
        if not (t and l and when):
            continue
        url = html.unescape(l.group(1)).split("?at_medium")[0]
        out.append({
            "title": _clean(t.group(1))[:200],
            "url": url,
            "summary": (_clean(s.group(1)) if s else "")[:300],
            "published_at": when,
        })
    return out


def fetch_all() -> dict:
    stats = {}
    fetched_at = db.iso(db.now())
    with httpx.Client(headers=UA, follow_redirects=True, timeout=20) as client, db.connect() as conn:
        for src in SOURCES:
            new = 0
            for url, topic in src["feeds"]:
                try:
                    r = client.get(url)
                    r.raise_for_status()
                    items = parse_feed(r.content.decode("utf-8", errors="replace"))
                except httpx.HTTPError as e:
                    stats.setdefault("errors", []).append(f"{src['name']} {url}: {e}")
                    continue
                for it in items:
                    if src.get("simplified"):   # DW 是簡體，轉成繁體再存（其他家不動，避免改到原標題用字）
                        it["title"], it["summary"] = to_trad(it["title"]), to_trad(it["summary"])
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO articles "
                        "(source_id,title,url,summary,topic_label,topic_by,published_at,fetched_at) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (src["id"], it["title"], it["url"], it["summary"], topic,
                         "feed" if topic else None, db.iso(it["published_at"]), fetched_at),
                    )
                    new += cur.rowcount
            stats[src["name"]] = new
    return stats


# ---------- 2. 向量 + 領域分類 ----------

def embed_all() -> int:
    cutoff = db.iso(db.now() - timedelta(days=FRESH_DAYS))
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT article_id,title,summary,topic_label,topic_by FROM articles "
            "WHERE published_at>=?", (cutoff,)).fetchall()
        if not rows:
            return 0
        texts = [article_text(r["title"], r["summary"]) for r in rows]
        emb = Embedder()
        emb.fit(texts)
        vecs = np.vstack([emb.embed(t) for t in texts])

        # 領域分類：RSS 有分類的文章算出各領域質心，沒分類的指派給最近的質心
        cents, names = [], []
        for topic in TOPICS:
            idx = [i for i, r in enumerate(rows) if r["topic_by"] == "feed" and r["topic_label"] == topic]
            if idx:
                cents.append(unit(vecs[idx].mean(axis=0)))
                names.append(topic)
        cents = np.vstack(cents) if cents else None

        for i, r in enumerate(rows):
            topic, by = r["topic_label"], r["topic_by"]
            if by != "feed" and cents is not None:
                topic, by = names[int(np.argmax(cents @ vecs[i]))], "knn"
            conn.execute(
                "UPDATE articles SET embedding=?, topic_label=?, topic_by=? WHERE article_id=?",
                (to_blob(vecs[i]), topic, by, r["article_id"]))
        conn.execute("UPDATE articles SET embedding=NULL WHERE published_at<?", (cutoff,))
    return len(rows)


# ---------- 3. 事件分群 ----------

def cluster_events() -> dict:
    """72 小時內的報導依時間排序，逐篇指派給最相似的事件（與質心相似度 ≥ 門檻），否則開新事件。"""
    cutoff = db.iso(db.now() - timedelta(hours=EVENT_WINDOW_HOURS))
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT article_id,source_id,title,published_at,embedding FROM articles "
            "WHERE published_at>=? AND embedding IS NOT NULL ORDER BY published_at", (cutoff,)
        ).fetchall()
        clusters: list[dict] = []
        sums = np.zeros((0, 0), dtype=np.float32)
        for r in rows:
            v = from_blob(r["embedding"])
            best, best_sim = -1, 0.0
            if clusters:
                cents = sums / np.linalg.norm(sums, axis=1, keepdims=True)
                sims = cents @ v
                best = int(np.argmax(sims))
                best_sim = float(sims[best])
            if best >= 0 and best_sim >= EVENT_SIM_THRESHOLD:
                clusters[best]["members"].append(r)
                sums[best] += v
            else:
                clusters.append({"members": [r]})
                sums = v[None, :].copy() if not len(sums) else np.vstack([sums, v])

        conn.execute("DELETE FROM events")
        conn.execute("UPDATE articles SET event_id=NULL")
        multi = 0
        for i, c in enumerate(clusters, start=1):
            m = c["members"]
            n_src = len({x["source_id"] for x in m})
            multi += n_src >= 2
            conn.execute("INSERT INTO events VALUES (?,?,?,?)",
                         (i, m[0]["title"], m[0]["published_at"], n_src))
            conn.executemany("UPDATE articles SET event_id=? WHERE article_id=?",
                             [(i, x["article_id"]) for x in m])
    return {"articles": len(rows), "events": len(clusters), "multi_source_events": multi}


def run() -> dict:
    db.init()
    result = {"fetched_new": fetch_all()}
    result["embedded"] = embed_all()
    result["clustering"] = cluster_events()
    return result


if __name__ == "__main__":
    import json
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run(), ensure_ascii=False, indent=2))
