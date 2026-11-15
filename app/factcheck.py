"""B. 訊息查證：抽出關鍵句 → Cofacts GraphQL（POST）→ 結果寫入 fact_checks。不自行判斷真假。"""
import re
from collections import Counter

import httpx

from . import db

COFACTS = "https://api.cofacts.tw/graphql"
QUERY = """query($t:String){ListArticles(filter:{moreLikeThis:{like:$t,minimumShouldMatch:"50%"}},
orderBy:[{_score:DESC}],first:3){totalCount edges{node{id text replyCount
articleReplies(status:NORMAL){reply{type text}}}}}}"""
TYPE_ZH = {"RUMOR": "含有不實訊息", "NOT_RUMOR": "含有正確訊息",
           "OPINIONATED": "含有個人意見", "NOT_ARTICLE": "不在查證範圍"}
CHANNELS = [
    {"name": "台灣事實查核中心", "url": "https://tfc-taiwan.org.tw/"},
    {"name": "MyGoPen", "url": "https://www.mygopen.com/"},
    {"name": "Cofacts 真的假的", "url": "https://cofacts.tw/"},
]


def extract_claim(text: str) -> str:
    """參數提取（規則版）：去掉網址、表情、口頭語，取最長的一句並截到 60 字。
    單元 5 實測：整段長文丟去查常常 0 筆，所以要縮成一句關鍵說法。"""
    t = re.sub(r"https?://\S+", " ", text)
    t = re.sub(r"(幫我)?查證(一下)?[:：]?|這是真的(嗎|假的)?[?？]?|是真的嗎[?？]?|真的假的[?？]?|請問", " ", t)
    t = re.sub(r"[^\w一-鿿，。、！？!?,.%％ ]", " ", t)
    parts = [p.strip() for p in re.split(r"[。！？!?\n]", t) if p.strip()]
    return (max(parts, key=len) if parts else t.strip())[:60]


def check(conn, user, text: str) -> dict:
    claim = extract_claim(text)
    trace = [f"參數提取（規則）：claim =「{claim}」"]
    if len(claim) < 4:
        return {"claim": claim, "error": "請貼上想查證的那句話（至少幾個字）", "trace": trace}
    try:
        r = httpx.post(COFACTS, json={"query": QUERY, "variables": {"t": claim}}, timeout=20)
        r.raise_for_status()
        data = r.json()["data"]["ListArticles"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
        trace.append(f"Cofacts 查詢失敗：{e}")
        return {"claim": claim, "error": "目前連不上 Cofacts，請直接到下列查證管道查詢",
                "channels": CHANNELS, "trace": trace}

    matches, top_category = [], "NO_MATCH"
    for n, edge in enumerate(data["edges"]):
        node = edge["node"]
        types = Counter(ar["reply"]["type"] for ar in node["articleReplies"])
        first = next((ar["reply"]["text"] for ar in node["articleReplies"]), "")
        if n == 0:
            top_category = types.most_common(1)[0][0] if types else "NO_REPLY"
        matches.append({
            "text": re.sub(r"\s+", " ", node["text"])[:110],
            "url": f"https://cofacts.tw/article/{node['id']}",
            "labels": [f"{TYPE_ZH.get(t, t)} ×{c}" for t, c in types.most_common()],
            "reply": re.sub(r"\s+", " ", first)[:140],
        })
    trace.append(f"Cofacts POST：相似訊息 {data['totalCount']} 筆，取前 {len(matches)} 筆")
    conn.execute(
        "INSERT INTO fact_checks (user_id, claim, cofacts_category, match_count, at) VALUES (?,?,?,?,?)",
        (user["user_id"], claim, top_category, data["totalCount"], db.iso(db.now())))
    conn.commit()
    trace.append(f"寫入 fact_checks（分類 {top_category}）")
    return {"claim": claim, "total": data["totalCount"], "matches": matches,
            "channels": CHANNELS, "trace": trace}
