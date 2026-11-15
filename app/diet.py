"""E. 新聞飲食報告：近 7 天讀了哪些領域／來源、外媒占比、來源多樣性（正規化 Shannon entropy）。"""
import math
from collections import Counter
from datetime import timedelta

from . import db
from .config import SOURCES
from .core import corpus, card, source_read_counts
from .recommend import other_views


def shannon_diversity(counts: list[int], n_categories: int) -> float:
    """H = −Σ p_i·log(p_i) / log(N)。0 = 全讀同一家；1 = 每家讀得一樣多。"""
    total = sum(counts)
    if total == 0 or n_categories < 2:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts if c)
    return h / math.log(n_categories)


def report(conn, user) -> dict:
    uid = user["user_id"]
    cutoff = db.iso(db.now() - timedelta(days=7))
    rows = conn.execute(
        "SELECT DISTINCT a.article_id, a.topic_label, a.event_id, s.name, s.country "
        "FROM interactions i JOIN articles a USING(article_id) JOIN sources s USING(source_id) "
        "WHERE i.user_id=? AND i.action IN ('click','like') AND i.at>=?", (uid, cutoff)).fetchall()
    simulated = conn.execute(
        "SELECT COUNT(*) FROM interactions WHERE user_id=? AND simulated=1", (uid,)).fetchone()[0]
    total = len(rows)
    if total == 0:
        return {"total": 0}

    by_source = Counter(r["name"] for r in rows)
    by_topic = Counter(r["topic_label"] or "未分類" for r in rows)
    foreign = sum(1 for r in rows if r["country"] != "台灣")
    n = len(SOURCES)
    h = shannon_diversity(list(by_source.values()), n)
    top_source, top_n = by_source.most_common(1)[0]

    # 建議：使用者讀過的事件中，找一家他較少讀的媒體的版本
    C = corpus()
    read_counts = source_read_counts(conn, uid)
    read_ids = {r["article_id"] for r in rows}
    suggestion = None
    for r in rows:
        alts = other_views(C, r["article_id"], read_ids, read_counts)
        if alts:
            a = C["rows"][r["article_id"]]
            suggestion = card(alts[0], user["click_token"],
                              f"你讀過{a['source']}對這件事的報導，{alts[0]['source']}也有報導", True)
            break

    return {
        "total": total,
        "sources": [{"name": s["name"], "count": by_source.get(s["name"], 0),
                     "pct": round(100 * by_source.get(s["name"], 0) / total)} for s in SOURCES],
        "topics": [{"name": t, "count": c, "pct": round(100 * c / total)}
                   for t, c in by_topic.most_common()],
        "foreign_pct": round(100 * foreign / total),
        "diversity": round(h, 2),
        "n_sources": n,
        "top_source": top_source,
        "top_source_pct": round(100 * top_n / total),
        "unread_sources": [s["name"] for s in SOURCES if not by_source.get(s["name"])],
        "suggestion": suggestion,
        "simulated": simulated,
    }
