"""把文字轉成向量。

Demo 版不需要 API Key：中文字元 bigram + TF-IDF，雜湊到固定維度後做 L2 正規化。
正式版把 Embedder 換成 Gemini Embedding（語意向量）即可，其餘推薦／分群程式不用改，
因為它們只依賴「向量 + 餘弦相似度」。
"""
import math
import re
import zlib
from collections import Counter

import numpy as np
from opencc import OpenCC

from .config import EMBED_DIM, IDF_PATH

_cc = OpenCC("s2t")
_CJK = re.compile(r"[一-鿿]+")
_LATIN = re.compile(r"[a-z0-9]{2,}")


def to_trad(text: str) -> str:
    """簡體 → 繁體（DW 中文是簡體，不轉的話和台灣媒體比不起來）。"""
    return _cc.convert(text or "")


def tokens(text: str) -> list[str]:
    text = to_trad(text).lower()
    out: list[str] = []
    for run in _CJK.findall(text):
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    out.extend(_LATIN.findall(text))
    return out


def _slot(tok: str) -> int:
    return zlib.crc32(tok.encode("utf-8")) % EMBED_DIM


class Embedder:
    def __init__(self) -> None:
        self.idf = np.ones(EMBED_DIM, dtype=np.float32)
        if IDF_PATH.exists():
            self.idf = np.load(IDF_PATH)

    def fit(self, texts: list[str]) -> None:
        """用整個語料算 IDF：越多文章都有的字詞，權重越低。"""
        df = np.zeros(EMBED_DIM, dtype=np.float32)
        for t in texts:
            for s in {_slot(tok) for tok in tokens(t)}:
                df[s] += 1
        n = max(len(texts), 1)
        self.idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
        np.save(IDF_PATH, self.idf)

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(EMBED_DIM, dtype=np.float32)
        for tok, c in Counter(tokens(text)).items():
            v[_slot(tok)] += 1 + math.log(c)
        v *= self.idf
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v


def article_text(title: str, summary: str) -> str:
    # 標題重複一次 = 標題權重加倍
    return f"{title} {title} {summary}"


def to_blob(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v
