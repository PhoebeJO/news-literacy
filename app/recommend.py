"""D. 素養版推薦：興趣向量 → 餘弦相似度 → MMR 多樣性重排 → 固定混入「另一觀點」→ 每則附理由。"""
from collections import Counter
from datetime import timedelta

import numpy as np

from . import db
from .config import (ACTION_WEIGHT, HALF_LIFE_DAYS, MMR_LAMBDA, SAME_SOURCE_PENALTY,
                     REC_K)
from .core import corpus, recent_ids, card, source_read_counts
from .embed import unit


def interest_profile(conn, user_id: int, C: dict):
    """① 興趣建模：點過／按讚的新聞向量加權平均，越近期權重越高（半衰期 7 天），👎 是負權重。"""
    rows = conn.execute(
        "SELECT article_id, action, at FROM interactions WHERE user_id=? ORDER BY at DESC",
        (user_id,)).fetchall()
    vec = np.zeros(C["M"].shape[1], dtype=np.float32)
    reads, seen = [], set()
    now = db.now()
    for r in rows:
        seen.add(r["article_id"])
        i = C["idx"].get(r["article_id"])
        if i is None:
            continue
        age_days = (now - db.parse(r["at"])).total_seconds() / 86400
        decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
        vec += ACTION_WEIGHT.get(r["action"], 0.0) * decay * C["M"][i]
        if r["action"] in ("click", "like") and r["article_id"] not in reads:
            reads.append(r["article_id"])
    return unit(vec), reads, seen


def mmr(C: dict, cand: list[int], rel: np.ndarray, k: int) -> list[int]:
    """③ MMR：score = λ·相關度 − (1−λ)·與已選新聞的最大重複度（同一家媒體加罰、同一事件視為重複）。"""
    picked: list[int] = []
    pool = list(range(len(cand)))
    while pool and len(picked) < k:
        best, best_score = None, -1e9
        for j in pool:
            red = 0.0
            for p in picked:
                a, b = C["rows"][cand[j]], C["rows"][cand[p]]
                sim = float(C["M"][C["idx"][cand[j]]] @ C["M"][C["idx"][cand[p]]])
                if a["source_id"] == b["source_id"]:
                    sim += SAME_SOURCE_PENALTY
                if a["event_id"] is not None and a["event_id"] == b["event_id"]:
                    sim = 1.0       # 同一事件視為完全重複：同事件的其他報導留給「另一觀點」那一格
                red = max(red, sim)
            score = MMR_LAMBDA * float(rel[j]) - (1 - MMR_LAMBDA) * red
            if score > best_score:
                best, best_score = j, score
        picked.append(best)
        pool.remove(best)
    return [cand[j] for j in picked]


def other_views(C: dict, article_id: int, exclude: set, read_counts: dict) -> list:
    """④ 同一事件、不同媒體的報導。排序：不同國家優先 → 使用者較少讀的媒體 → 較新。"""
    a = C["rows"].get(article_id)
    if a is None or a["event_id"] is None:
        return []
    alts = [r for r in C["rows"].values()
            if r["event_id"] == a["event_id"] and r["source_id"] != a["source_id"]
            and r["article_id"] not in exclude]
    alts.sort(key=lambda r: r["published_at"], reverse=True)
    alts.sort(key=lambda r: (r["country"] == a["country"], read_counts.get(r["source"], 0)))
    return alts


def recommend(conn, user) -> dict:
    C = corpus()
    trace = []
    token, uid = user["click_token"], user["user_id"]
    profile, reads, seen = interest_profile(conn, uid, C)
    read_counts = source_read_counts(conn, uid)
    recently_sent = {r[0] for r in conn.execute(
        "SELECT article_id FROM recommendations WHERE user_id=? AND sent_at>=?",
        (uid, db.iso(db.now() - timedelta(hours=24))))}

    pool = recent_ids(C)
    cand = [i for i in pool if i not in seen and i not in recently_sent]
    if len(cand) < REC_K + 1:
        cand = [i for i in pool if i not in seen] or pool
    if not cand:
        return {"cards": [], "trace": ["資料庫沒有新聞，請先按「重新抓取新聞」"], "mode": "empty"}

    chosen = [t for t in (user["chosen_topics"] or "").split(",") if t]
    cards, picks = [], []

    if reads or float(np.linalg.norm(profile)) > 0:
        mode = "profile"
        trace.append(f"① 興趣向量：{len(reads)} 則讀過的新聞（時間衰減半衰期 {HALF_LIFE_DAYS:g} 天）")
    elif chosen:
        mode = "cold_topics"
        vecs = [C["M"][C["idx"][i]] for i in C["ids"] if C["rows"][i]["topic_label"] in chosen]
        profile = unit(np.mean(vecs, axis=0)) if vecs else profile
        trace.append(f"⑥ 冷啟動：用你選的領域 {'、'.join(chosen)} 的質心當興趣向量")
    else:
        mode = "cold_events"
        trace.append("⑥ 冷啟動：還沒有閱讀紀錄也沒選領域 → 推多家媒體都有報導的事件")

    if mode == "cold_events":
        with_n = sorted(cand, key=lambda i: C["rows"][i]["published_at"], reverse=True)
        ev_sizes = dict(conn.execute("SELECT event_id, source_count FROM events").fetchall())
        with_n.sort(key=lambda i: ev_sizes.get(C["rows"][i]["event_id"], 1), reverse=True)
        used_events, used_sources = set(), Counter()
        for i in with_n:
            r = C["rows"][i]
            if r["event_id"] in used_events or used_sources[r["source"]] >= 2:
                continue
            used_events.add(r["event_id"])
            used_sources[r["source"]] += 1
            n = ev_sizes.get(r["event_id"], 1)
            reason = f"{n} 家媒體都在報導的事件" if n >= 2 else "最新報導"
            picks.append(i)
            cards.append(card(r, token, reason))
            if len(picks) == REC_K:
                break
    else:
        rel = C["M"][[C["idx"][i] for i in cand]] @ profile
        top = np.argsort(-rel)[:40]                       # ② 相關度前 40 名進入重排
        short = [cand[j] for j in top]
        picks = mmr(C, short, rel[top], REC_K)
        trace.append(f"② 餘弦相似度：{len(cand)} 則候選取前 {len(short)} 名")
        trace.append(f"③ MMR 重排（λ={MMR_LAMBDA}）：選出 {len(picks)} 則，"
                     f"來自 {len({C['rows'][i]['source'] for i in picks})} 家媒體")
        topic_n = Counter(C["rows"][i]["topic_label"] for i in reads if i in C["rows"])
        score_of = {cand[j]: rel[j] for j in top}
        for i in picks:
            r = C["rows"][i]
            if mode == "cold_topics":
                reason = (f"你選了「{r['topic_label']}」領域" if r["topic_label"] in chosen
                          else "和你選的領域內容相近")
            elif topic_n.get(r["topic_label"]):
                reason = f"你最近讀了 {topic_n[r['topic_label']]} 則「{r['topic_label']}」新聞"
            else:
                near = max((x for x in reads if x in C["idx"]),
                           key=lambda x: float(C["M"][C["idx"][x]] @ C["M"][C["idx"][i]]),
                           default=None)
                reason = (f"和你讀過的〈{C['rows'][near]['title'][:14]}…〉內容相近"
                          if near else "和你的興趣向量相近")
            cards.append(card(r, token, reason, score=score_of[i]))

    # ④ 固定混入一則「另一觀點」
    exclude = set(picks) | seen
    ov = None
    for n, anchor in enumerate(picks, start=1):
        alts = other_views(C, anchor, exclude, read_counts)
        if alts:
            a = C["rows"][anchor]
            ov = (alts[0], f"另一個觀點：第 {n} 則是{a['source']}的報導，"
                           f"這是{alts[0]['source']}（{alts[0]['country']}）對同一事件的報導")
            break
    if ov is None:
        for anchor in reads[:10]:
            alts = other_views(C, anchor, exclude, read_counts)
            if alts:
                a = C["rows"][anchor]
                ov = (alts[0], f"另一個觀點：你讀過{a['source']}的〈{a['title'][:12]}…〉，"
                               f"這是{alts[0]['source']}（{alts[0]['country']}）對同一事件的報導")
                break
    if ov is None:
        events = conn.execute(
            "SELECT event_id, source_count FROM events WHERE source_count>=2 "
            "ORDER BY source_count DESC").fetchall()
        for e in events:
            members = [r for r in C["rows"].values()
                       if r["event_id"] == e["event_id"] and r["article_id"] not in exclude]
            if members:
                members.sort(key=lambda r: read_counts.get(r["source"], 0))
                m = members[0]
                ov = (m, f"另一個觀點：{e['source_count']} 家媒體都在報導的事件，"
                         f"這是你較少讀的{m['source']}（{m['country']}）的版本")
                break
    if ov:
        cards.append(card(ov[0], token, ov[1], other_view=True))
        trace.append(f"④ 另一觀點：{ov[0]['source']}（事件 #{ov[0]['event_id']}）")
    else:
        trace.append("④ 另一觀點：目前沒有多家媒體共同報導的事件可用")

    sent = db.iso(db.now())
    conn.executemany(
        "INSERT INTO recommendations (user_id, article_id, reason, is_other_view, sent_at) "
        "VALUES (?,?,?,?,?)",
        [(uid, c["article_id"], c["reason"], int(c["other_view"]), sent) for c in cards])
    conn.commit()
    trace.append("⑤ 每則附推薦理由，寫入 recommendations 表")
    return {"cards": cards, "trace": trace, "mode": mode}
