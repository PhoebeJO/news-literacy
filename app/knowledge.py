"""C. 名詞／識讀問題：檢索媒體識讀指南（一個名詞一塊），回答時附引用。"""
import re

import numpy as np

from .config import ROOT
from .embed import Embedder

GUIDE = ROOT / "app" / "media_literacy.md"
_entries = None


def entries() -> list[dict]:
    global _entries
    if _entries is None:
        _entries = []
        for block in re.split(r"(?m)^## ", GUIDE.read_text(encoding="utf-8"))[1:]:
            head, _, body = block.partition("\n")
            names = [n.strip() for n in head.split("／")]
            _entries.append({"name": names[0], "aliases": names, "text": body.strip()})
    return _entries


def search(query: str) -> dict | None:
    q = query.lower()
    for e in entries():                       # 先比對名詞本身
        if any(a.lower() in q for a in e["aliases"]):
            return {**e, "by": "名詞比對"}
    emb = Embedder()                          # 再用向量相似度
    qv = emb.embed(query)
    sims = [float(qv @ emb.embed(" ".join(e["aliases"]) + " " + e["text"])) for e in entries()]
    best = int(np.argmax(sims))
    if sims[best] >= 0.08:
        return {**entries()[best], "by": f"向量檢索（相似度 {sims[best]:.2f}）"}
    return None


def has_term(query: str) -> bool:
    q = query.lower()
    return any(a.lower() in q for e in entries() for a in e["aliases"])
