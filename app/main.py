"""FastAPI 入口。啟動：python -m uvicorn app.main:app --port 8765

/            聊天介面（模擬 LINE 對話）
/api/chat    對話（正式版是 LINE Webhook）
/api/postback 卡片按鈕
/r/{id}      點擊追蹤：記錄後 302 轉址到原文
/api/stats   儀表板數據（正式版是 Streamlit）
/api/refresh 手動執行離線批次（正式版是 GitHub Actions cron）
"""
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from . import db, pipeline, router, llm
from .config import ROOT, SOURCES
from .core import get_user, invalidate

_refresh_lock = threading.Lock()
_state = {"refreshing": False, "last": None, "at": 0.0}
REFRESH_EVERY = 3 * 3600     # 自動更新間隔（秒）
REFRESH_COOLDOWN = 600       # 手動更新的最短間隔：公開上線後避免被連續觸發去打媒體網站


def refresh() -> None:
    if not _refresh_lock.acquire(blocking=False):
        return
    _state["refreshing"] = True
    try:
        _state["last"] = pipeline.run()
        invalidate()
    except Exception as e:      # 抓取失敗不讓伺服器掛掉，狀態顯示在儀表板
        _state["last"] = {"error": str(e)}
    finally:
        _state["refreshing"] = False
        _state["at"] = time.time()
        _refresh_lock.release()


def _auto_refresh() -> None:
    """伺服器醒著的期間每 3 小時抓一次（正式版由 GitHub Actions cron 負責）。"""
    while True:
        time.sleep(REFRESH_EVERY)
        refresh()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    with db.connect() as conn:
        latest = conn.execute("SELECT MAX(fetched_at) FROM articles").fetchone()[0]
    stale = latest is None or (db.now() - db.parse(latest)).total_seconds() > REFRESH_EVERY
    if stale:                   # 資料庫是空的或資料太舊：啟動時先抓一次
        threading.Thread(target=refresh, daemon=True).start()
    threading.Thread(target=_auto_refresh, daemon=True).start()
    yield


app = FastAPI(title="新聞素養助手 Demo", lifespan=lifespan)


class ChatIn(BaseModel):
    uid: str = Field(min_length=8, max_length=80)
    text: str = Field(max_length=2000)


class PostbackIn(BaseModel):
    uid: str = Field(min_length=8, max_length=80)
    article_id: int
    action: str


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/chat")
def chat(body: ChatIn):
    if _state["refreshing"]:
        with db.connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL").fetchone()[0] == 0:
                return {"intent": "OTHER", "intent_name": "其他", "intent_by": "system", "trace": [],
                        "messages": [{"type": "text", "text": "正在抓取新聞並建立向量，約 20 秒後再試一次。"}],
                        "quick": router.MENU}
    return router.handle(lambda conn: get_user(conn, body.uid), body.text)


@app.post("/api/postback")
def postback(body: PostbackIn):
    return router.postback(lambda conn: get_user(conn, body.uid), body.article_id, body.action)


@app.get("/r/{article_id}")
def track(article_id: int, t: str = ""):
    """點擊追蹤：LINE 訊息裡的連結先到這裡，記一筆 click 再轉址到原文。"""
    with db.connect() as conn:
        art = conn.execute("SELECT url FROM articles WHERE article_id=?", (article_id,)).fetchone()
        if art is None:
            raise HTTPException(404, "article not found")
        user = conn.execute("SELECT user_id FROM users WHERE click_token=?", (t,)).fetchone()
        if user:
            conn.execute("INSERT INTO interactions (user_id, article_id, action, at) VALUES (?,?,?,?)",
                         (user["user_id"], article_id, "click", db.iso(db.now())))
    return RedirectResponse(art["url"], status_code=302)


@app.post("/api/refresh")
def api_refresh():
    if _state["refreshing"]:
        return {"status": "already_running"}
    if time.time() - _state["at"] < REFRESH_COOLDOWN:
        return {"status": "too_soon"}
    threading.Thread(target=refresh, daemon=True).start()
    return {"status": "started"}


@app.get("/api/stats")
def stats():
    with db.connect() as conn:
        def q(sql, *a):
            return conn.execute(sql, a).fetchall()

        def one(sql, *a):
            return conn.execute(sql, a).fetchone()[0]

        per_source = {r["name"]: r["n"] for r in q(
            "SELECT s.name, COUNT(a.article_id) n FROM sources s LEFT JOIN articles a USING(source_id) "
            "GROUP BY s.source_id")}
        events = [dict(r) for r in q(
            "SELECT e.event_id, e.label, e.source_count, "
            "(SELECT COUNT(*) FROM articles a WHERE a.event_id=e.event_id) n_articles, "
            "(SELECT GROUP_CONCAT(DISTINCT s.name) FROM articles a JOIN sources s USING(source_id) "
            " WHERE a.event_id=e.event_id) sources "
            "FROM events e WHERE e.source_count>=2 ORDER BY e.source_count DESC, n_articles DESC LIMIT 12")]
        # 推薦成效：推出去之後，同一位使用者有沒有點那一則（不含示範用的模擬資料）
        rec = q("SELECT r.is_other_view, COUNT(*) sent, SUM(EXISTS("
                "SELECT 1 FROM interactions i WHERE i.user_id=r.user_id AND i.article_id=r.article_id "
                "AND i.action='click' AND i.simulated=0 AND i.at>=r.sent_at)) clicked "
                "FROM recommendations r GROUP BY r.is_other_view")
        rec = {("other_view" if r["is_other_view"] else "interest"):
               {"sent": r["sent"], "clicked": r["clicked"] or 0} for r in rec}
        likes = one("SELECT COUNT(*) FROM interactions WHERE action='like'")
        dislikes = one("SELECT COUNT(*) FROM interactions WHERE action='dislike'")
        return {
            "refreshing": _state["refreshing"],
            "last_refresh": _state["last"],
            "llm_enabled": llm.available(), "llm_label": llm.label(),
            "articles_total": one("SELECT COUNT(*) FROM articles"),
            "articles_per_source": [{"name": s["name"], "country": s["country"],
                                     "ownership": s["ownership"], "count": per_source.get(s["name"], 0)}
                                    for s in SOURCES],
            "latest_fetch": (lambda v: db.tw_str(v) if v else None)(one("SELECT MAX(fetched_at) FROM articles")),
            "events_total": one("SELECT COUNT(*) FROM events"),
            "events_multi": one("SELECT COUNT(*) FROM events WHERE source_count>=2"),
            "top_events": events,
            "users": one("SELECT COUNT(*) FROM users"),
            "intents": [dict(r) for r in q("SELECT intent, COUNT(*) n FROM query_log GROUP BY intent ORDER BY n DESC")],
            "recommendations": rec,
            "clicks_real": one("SELECT COUNT(*) FROM interactions WHERE action='click' AND simulated=0"),
            "clicks_simulated": one("SELECT COUNT(*) FROM interactions WHERE simulated=1"),
            "likes": likes, "dislikes": dislikes,
            "fact_checks": [dict(r) for r in q(
                "SELECT cofacts_category category, COUNT(*) n FROM fact_checks GROUP BY 1 ORDER BY n DESC")],
            "llm_usage": [dict(r) for r in q(
                "SELECT purpose, COUNT(*) calls, SUM(tokens_in) tokens_in, SUM(tokens_out) tokens_out "
                "FROM llm_usage GROUP BY purpose")],
        }
