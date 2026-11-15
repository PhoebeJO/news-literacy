"""線上對話流程（單元 6 Chatflow 的程式版）：
前端 Guardrail → 意圖識別 → 參數提取 → 各分支 → 回覆，並留下每一步的 trace。"""
import json
import re
from datetime import timedelta

from . import db, llm, knowledge, factcheck, compare, diet
from .config import TOPICS
from .core import corpus, recent_ids, card
from .recommend import recommend, other_views
from .core import source_read_counts

INJECTION = re.compile(r"忽略.{0,8}(指令|提示|規則)|ignore .{0,20}instruction|system prompt|"
                       r"你的\s*(prompt|提示詞|系統提示)|\bDAN\b|扮演.{0,6}沒有限制", re.I)
STANCE = re.compile(r"(哪|那)(一)?(家|個媒體|間).{0,8}(可信|中立|公正|偏頗|客觀|最好|能信)|"
                    r"哪個(政黨|黨|候選人).{0,6}(對|好|可信)|該(投|支持)(誰|哪)")
INTENTS = {
    "A": "議題對照", "B": "訊息查證", "C": "名詞問題", "D": "推薦", "E": "我的報告",
    "TOPICS": "選領域", "SEED": "示範紀錄", "DELETE": "刪除紀錄", "OTHER": "其他",
}
MENU = ["推薦新聞", "我的新聞報告", "選領域", "什麼是同溫層？", "載入示範紀錄"]
HELP = ("我是新聞素養助手（Demo），可以：\n"
        "・推薦新聞：依你的興趣推薦，最後一則固定是另一個觀點\n"
        "・我的新聞報告：看你這週的新聞飲食與來源多樣性\n"
        "・輸入議題（例：「微軟當機的新聞」）：多家媒體並排對照\n"
        "・貼上轉傳訊息：到 Cofacts 查有沒有查核紀錄\n"
        "・問名詞（例：「什麼是框架效應」）\n"
        "・刪除我的紀錄：清除你的閱讀與查詢紀錄\n"
        "我不會判斷哪家媒體中立或偏頗，只呈現差異與來源。")


def classify_rule(text: str) -> str:
    t = text.strip()
    if "刪除" in t and "紀錄" in t:
        return "DELETE"
    if "示範紀錄" in t or "示範資料" in t:
        return "SEED"
    if t.startswith("領域") or "選領域" in t:
        return "TOPICS"
    if re.search(r"報告|飲食|多樣性", t):
        return "E"
    if re.search(r"推薦|推播|推新聞|今日新聞|今天有什麼|有什麼新聞", t):
        return "D"
    if re.search(r"查證|真的嗎|是真的|真的假的|謠言|闢謠|https?://", t) or len(t) >= 40:
        return "B"
    if re.search(r"什麼是|是什麼|什麼意思|解釋|定義|為什麼|怎麼(分辨|判斷|辨別)|如何(分辨|判斷|辨別)", t) or knowledge.has_term(t):
        return "C"
    if re.search(r"新聞|報導|怎麼報|對照|比較|各家", t):
        return "A"
    if re.search(r"^(你好|哈囉|嗨|hi|hello|你是誰|你能做什麼|help|說明|功能)", t, re.I):
        return "OTHER"
    return "A" if 2 <= len(t) <= 12 else "OTHER"


def classify(text: str, trace: list) -> tuple[str, str]:
    rule = classify_rule(text)
    if llm.available() and rule in ("A", "B", "C", "OTHER"):
        try:
            out = llm.generate("intent", (
                "判斷使用者訊息的意圖，只輸出 JSON：{\"intent\": \"A|B|C|OTHER\"}。\n"
                "A=想看某個議題的新聞或多家對照；B=貼了一則訊息想查證真假；"
                "C=問媒體識讀名詞；OTHER=其他。\n<message>" + text[:300] + "</message>"), json_mode=True)
            intent = json.loads(out).get("intent")
            if intent in ("A", "B", "C", "OTHER"):
                return intent, "llm"
        except Exception as e:
            trace.append(f"意圖識別 LLM 失敗，改用規則：{e}")
    return rule, "rule"


def extract_topic(text: str) -> str:
    t = re.sub(r"幫我(看|查|找)(一下)?|我想(看|知道)|想知道|請問|有關|關於|最近|這幾天|今天|"
               r"各家|媒體|怎麼報導?|怎麼說|的?新聞|的?報導|對照|比較|有哪些|有什麼|一下|"
               r"[?？!！。，、：:\s]", " ", text)
    return re.sub(r"的$", "", " ".join(t.split()))


def seed_demo(conn, user) -> str:
    """模擬一位「只讀同一家、同一領域」的讀者，方便展示推薦與飲食報告。資料標記 simulated=1。"""
    C = corpus()
    pool = [C["rows"][i] for i in recent_ids(C)]
    if len(pool) < 12:
        return "新聞還不夠，請先按「重新抓取新聞」。"
    from collections import Counter
    src, _ = Counter(r["source"] for r in pool).most_common(1)[0]
    mine = [r for r in pool if r["source"] == src]
    topic, _ = Counter(r["topic_label"] for r in mine).most_common(1)[0]
    multi = {r[0] for r in conn.execute("SELECT event_id FROM events WHERE source_count>=2")}
    main = sorted((r for r in mine if r["topic_label"] == topic),
                  key=lambda r: r["event_id"] not in multi)[:8]
    others = [r for r in pool if r["source"] != src and r["country"] == "台灣"][:2]
    picked = main + others
    now = db.now()
    conn.execute("DELETE FROM interactions WHERE user_id=? AND simulated=1", (user["user_id"],))
    for n, r in enumerate(picked):
        at = now - timedelta(hours=6 + n * 11)      # 分散在過去約 5 天
        conn.execute(
            "INSERT INTO interactions (user_id, article_id, action, simulated, at) VALUES (?,?,?,1,?)",
            (user["user_id"], r["article_id"], "click", db.iso(at)))
    conn.commit()
    return (f"已載入示範紀錄（模擬資料）：過去 5 天點了 {len(picked)} 則新聞，"
            f"其中 {len(main)} 則是{src}的「{topic}」新聞。\n"
            "現在試試「推薦新聞」和「我的新聞報告」。")


def handle(uid_user, text: str) -> dict:
    """回傳 {intent, intent_by, trace, messages, quick}。"""
    text = (text or "").strip()[:1000]
    trace, msgs, quick = [], [], MENU
    with db.connect() as conn:
        user = uid_user(conn)

        # --- 前端 Guardrail（規則版；正式版再加一層 LLM）---
        if INJECTION.search(text):
            trace.append("前端 Guardrail：疑似提示注入 → 固定回覆")
            intent, by = "OTHER", "guardrail"
            msgs.append({"type": "text", "text": "這個要求我沒辦法照做。我是新聞素養助手，可以幫你對照新聞、查證訊息。"})
        elif STANCE.search(text):
            trace.append("前端 Guardrail：要求替媒體或政黨表態 → 婉拒，改提供對照")
            intent, by = "OTHER", "guardrail"
            msgs.append({"type": "text", "text": "我不會判斷哪家媒體或哪個政黨比較可信。"
                         "我能做的是把不同媒體對同一件事的報導並排給你看，由你自己比較。"
                         "輸入一個議題試試，例如「微軟當機的新聞」。"})
        else:
            trace.append("前端 Guardrail：通過")
            intent, by = classify(text, trace)
            trace.append(f"意圖識別（{llm.label() if by == 'llm' else '規則'}）：{intent} {INTENTS[intent]}")

            if intent == "D":
                rec = recommend(conn, user)
                trace += rec["trace"]
                if rec["cards"]:
                    lead = {"profile": "依你的閱讀紀錄推薦，最後一則是另一個觀點：",
                            "cold_topics": "依你選的領域推薦，最後一則是另一個觀點：",
                            "cold_events": "你還沒有閱讀紀錄，先推多家媒體都在報導的事件。"
                                           "點連結、按 👍👎 之後，推薦會跟著調整："}[rec["mode"]]
                    msgs += [{"type": "text", "text": lead}, {"type": "cards", "cards": rec["cards"]}]
                else:
                    msgs.append({"type": "text", "text": "目前沒有可以推薦的新聞，請先按右上角「重新抓取新聞」。"})

            elif intent == "E":
                rep = diet.report(conn, user)
                trace.append("SQL 統計近 7 天 interactions → 來源分佈 → 正規化 Shannon entropy")
                if rep["total"] == 0:
                    msgs.append({"type": "text", "text": "你這週還沒有閱讀紀錄。先點幾則推薦新聞的連結，"
                                 "或按「載入示範紀錄」看看報告長什麼樣子。"})
                else:
                    msgs.append({"type": "report", "report": rep})

            elif intent == "A":
                topic = extract_topic(text)
                if len(topic) < 2:
                    msgs.append({"type": "text", "text": "想看哪個議題？例如「微軟當機的新聞」。"})
                else:
                    res = compare.find(conn, user, topic)
                    trace += res.pop("trace")
                    if res["found"] == 0:
                        msgs.append({"type": "text", "text": f"近 7 天的資料庫裡找不到和「{topic}」有關的報導。"
                                     "Demo 只收 6 家媒體 RSS 上的近期新聞，換個關鍵字試試。"})
                    else:
                        msgs.append({"type": "compare", "data": res})

            elif intent == "B":
                res = factcheck.check(conn, user, text)
                trace += res.pop("trace")
                msgs.append({"type": "factcheck", "data": res})

            elif intent == "C":
                hit = knowledge.search(text)
                if hit:
                    trace.append(f"知識庫檢索：命中〈{hit['name']}〉（{hit['by']}）")
                    msgs.append({"type": "kb", "name": hit["name"], "text": hit["text"]})
                else:
                    trace.append("知識庫檢索：沒有夠相近的條目 → 不編造")
                    names = "、".join(e["name"] for e in knowledge.entries())
                    msgs.append({"type": "text", "text": f"識讀指南裡沒有這個名詞。目前收錄：{names}。"})

            elif intent == "TOPICS":
                picked = [t for t in TOPICS if t in text] if text.startswith("領域") else None
                if picked:
                    conn.execute("UPDATE users SET chosen_topics=? WHERE user_id=?",
                                 (",".join(picked), user["user_id"]))
                    conn.commit()
                    trace.append(f"寫入 users.chosen_topics = {picked}")
                    msgs.append({"type": "text", "text": f"已記下你關注的領域：{'、'.join(picked)}。按「推薦新聞」試試。"})
                else:
                    chosen = [t for t in (user["chosen_topics"] or "").split(",") if t]
                    msgs.append({"type": "topics", "topics": TOPICS, "chosen": chosen})

            elif intent == "SEED":
                msgs.append({"type": "text", "text": seed_demo(conn, user)})

            elif intent == "DELETE":
                for table in ("interactions", "recommendations", "fact_checks", "query_log"):
                    conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user["user_id"],))
                conn.execute("UPDATE users SET chosen_topics='' WHERE user_id=?", (user["user_id"],))
                conn.commit()
                trace.append("刪除該使用者的 interactions / recommendations / fact_checks / query_log")
                msgs.append({"type": "text", "text": "已刪除你的閱讀、推薦、查證紀錄與領域設定。"})

            else:
                msgs.append({"type": "text", "text": HELP})

        if intent != "DELETE":
            conn.execute("INSERT INTO query_log (user_id, intent, by, at) VALUES (?,?,?,?)",
                         (user["user_id"], intent, by, db.iso(db.now())))
            conn.commit()
    return {"intent": intent, "intent_name": INTENTS[intent], "intent_by": by,
            "trace": trace, "messages": msgs, "quick": quick}


def postback(uid_user, article_id: int, action: str) -> dict:
    """卡片上的 👍 / 👎 /「換個觀點」按鈕（正式版是 LINE postback）。"""
    with db.connect() as conn:
        user = uid_user(conn)
        C = corpus()
        if article_id not in C["rows"] or action not in ("like", "dislike", "other_view"):
            return {"messages": [{"type": "text", "text": "找不到這則新聞。"}]}
        conn.execute("INSERT INTO interactions (user_id, article_id, action, at) VALUES (?,?,?,?)",
                     (user["user_id"], article_id, action, db.iso(db.now())))
        conn.commit()
        if action != "other_view":
            word = "之後會多推類似的內容" if action == "like" else "之後會少推類似的內容"
            return {"messages": [{"type": "text", "text": f"收到，{word}。"}],
                    "trace": [f"寫入 interactions（{action}）→ 下次推薦的興趣向量會更新"]}
        alts = other_views(C, article_id, {article_id}, source_read_counts(conn, user["user_id"]))
        a = C["rows"][article_id]
        if not alts:
            return {"messages": [{"type": "text", "text":
                    f"〈{a['title'][:18]}…〉目前資料庫裡只有{a['source']}一家報導，找不到其他媒體的版本。"}],
                    "trace": ["事件分群：這則所屬事件只有一家媒體"]}
        cards = [card(r, user["click_token"], f"{r['source']}（{r['country']}）對同一事件的報導", True)
                 for r in alts[:3]]
        return {"messages": [{"type": "text", "text": f"同一事件的其他媒體報導（你剛看的是{a['source']}）："},
                             {"type": "cards", "cards": cards}],
                "trace": [f"事件 #{a['event_id']}：找到 {len(alts)} 則其他媒體的報導"]}
