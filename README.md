# 新聞素養助手 Demo（本機版）

依 [[單元6報告]] 第三節的設計做的可執行 demo：**推你感興趣的新聞，同時讓你看到不同觀點，並告訴你自己的新聞飲食偏在哪裡。**
不需要任何 API Key，新聞是即時從 6 家媒體的 RSS 抓的真實資料。LLM 目前接的是本機 Ollama（llama3.1:8b），也不需要金鑰。

## 啟動

```bash
cd news-demo
.venv/Scripts/python -m uvicorn app.main:app --port 8765
```

打開 <http://localhost:8765>。第一次啟動（資料庫是空的）會自動抓一次新聞，約 20 秒；之後按右上角「重新抓取新聞」更新。
換電腦時先 `py -3.11 -m venv .venv`，再 `.venv/Scripts/python -m pip install -r requirements.txt`。

## 線上版（Render）

repo 根目錄的 `render.yaml` 是 Render Blueprint：在 Render 選 **New → Blueprint**，連結這個 repo 即可部署。

- 線上網址：<https://news-literacy-demo.onrender.com>
- 免費方案閒置 15 分鐘會休眠。`.github/workflows/keepalive.yml` 每 10 分鐘打一次健康檢查讓它保持清醒；GitHub 的排程偶爾會延遲，萬一遇到休眠，打開要等約 1 分鐘，醒來後會自動重新抓新聞（再等約 20 秒）。
- 資料庫在容器裡，**每次重啟或重新部署都會清空**（閱讀紀錄、儀表板數字歸零）。這是 demo 的限制，正式版要換成 Supabase。
- 雲端沒有 Ollama，線上版跑規則版（`LLM_PROVIDER=none`）。
- 伺服器醒著時每 3 小時自動更新新聞；「重新抓取新聞」按鈕 10 分鐘內只會生效一次。

## Demo 腳本（約 3 分鐘）

| 步驟 | 操作 | 展示重點 |
|---|---|---|
| 1 | 按「推薦新聞」 | 冷啟動：沒有紀錄 → 推多家媒體都報導的事件；第 5 則固定是「另一觀點」 |
| 2 | 按「選領域」選 2～3 個 → 再按「推薦新聞」 | 冷啟動第二種：領域質心當興趣向量 |
| 3 | 按「載入示範紀錄」→「推薦新聞」 | 興趣向量 + MMR；每則都有「為什麼推這則」；展開「流程追蹤」看每一步 |
| 4 | 按「我的新聞報告」 | 來源多樣性（Shannon entropy）偏低 → 建議看另一家 |
| 5 | 點卡片標題、按 👍 / 👎 /「換個觀點」 | 點擊先經過 `/r/{id}` 記錄再轉址；回饋會改變下一次推薦 |
| 6 | 輸入儀表板上任一事件的關鍵字（例：「XX 的新聞」） | 多家並排、共同用詞、事件時間軸 |
| 7 | 貼一段轉傳訊息，或「查證：……」 | Cofacts 查核結果與連結，不自行判定真假 |
| 8 | 「什麼是框架效應」「哪家媒體最可信？」「忽略以上指令…」 | 知識庫附引用；Guardrail 婉拒表態、擋注入 |
| 9 | 切到「儀表板」 | 點擊率、「另一觀點」點擊率、👍 率、事件分群、意圖分佈 |
| 10 | 輸入「刪除我的紀錄」 | 隱私：使用者可刪除自己的資料 |

## LLM 設定（`news-demo/.env`）

| 設定 | 效果 |
|---|---|
| `LLM_PROVIDER=ollama`（目前） | 意圖識別（A/B/C/其他）與多源比對用本機 Ollama；要先開著 Ollama |
| 刪掉那一行，或 `LLM_PROVIDER=none` | 全部回到規則版，回覆是即時的 |
| `GEMINI_API_KEY=金鑰`（並拿掉 `LLM_PROVIDER=ollama`） | 改用 Gemini |

- 改完 `.env` 要重啟伺服器。`python -m app.llm` 可以單獨測 LLM 有沒有通。
- **Demo 前先暖機**：多源比對的結果會快取（同一組報導、同一個模型只算一次）。上台前把要展示的議題各問一次，現場就是秒回；開機後第一次呼叫要另外等模型載入（約 1 分鐘）。
- 推薦、飲食報告、查證、名詞問題不經過 LLM，速度不受影響。

## 和正式版的對應

| 設計（單元 6 報告三-4） | Demo 的做法 | 正式版 |
|---|---|---|
| LINE Messaging API | 網頁聊天介面（卡片 = Flex Message，按鈕 = postback） | LINE Webhook |
| Supabase + pgvector | SQLite，表格照 ERD 建（`app/db.py`） | 換連線即可 |
| Gemini Embedding | 中文字元 bigram TF-IDF 雜湊向量（`app/embed.py`） | 換掉 `Embedder` |
| Gemini（意圖、比對、Guardrail） | 意圖識別與多源比對走本機 Ollama；Guardrail 仍是規則版 | 換成 Gemini（見下方） |
| trafilatura 讀全文 | 只用 RSS 的標題與摘要 | 加全文抽取 |
| GitHub Actions cron | 手動按鈕或 `python -m app.pipeline` | 排程執行同一支程式 |
| Streamlit 儀表板 | 同一頁的「儀表板」分頁（`/api/stats`） | Streamlit |

## 程式對照

| 檔案 | 內容 |
|---|---|
| `app/pipeline.py` | 離線批次：抓 RSS → 向量 → 領域分類（最近質心）→ 72 小時事件分群 |
| `app/recommend.py` | 興趣向量（時間衰減）、餘弦相似度、MMR、另一觀點、推薦理由、冷啟動 |
| `app/diet.py` | 新聞飲食報告、正規化 Shannon entropy |
| `app/compare.py` | 議題多源對照 + 事件時間軸 |
| `app/factcheck.py` | Cofacts 查證，寫入 `fact_checks` |
| `app/knowledge.py`、`app/media_literacy.md` | 識讀指南檢索 |
| `app/router.py` | Guardrail → 意圖識別 → 參數提取 → 分支（單元 6 Chatflow 的程式版） |
| `app/main.py` | FastAPI：對話、postback、點擊追蹤轉址、儀表板數據 |
| `app/config.py` | 媒體白名單與所有演算法參數（λ、門檻、半衰期） |

## 已知限制（報告時要誠實說明）

- **向量是字面相似，不是語意相似**：TF-IDF 只認得相同的字詞，「川普／特朗普」這類不同譯名會被當成不同詞。事件分群門檻 0.22 是用 2026-11-14 當天的資料人工檢視後選的（0.30 太保守、同事件被拆開；0.22 多數正確、偶爾把相關但不同的事件併在一起），還沒有做 precision / recall 標註。
- **Gemini 路徑沒有實測**：`app/llm.py` 的 Gemini 分支是在沒有金鑰的情況下寫的，接上金鑰後要先跑 `python -m app.llm`。
- **本機 LLM 很慢、比對內容要人工核對**：實測（i7-8550U、無獨顯加速）意圖識別每則 7～17 秒，多源比對一次約 3 分鐘。比對結果只看標題與摘要，llama3.1:8b 會把「各家寫的重點不同」說成「說法不一致」，也可能把某家的內容記到另一家，展示時要對照上方的並排原文。
- **「共同用詞」是字串比對**，會切出不完整的詞，不能當成共同事實；真正的「共同事實／說法不一致」要等 LLM 比對。
- **領域標籤**：中央社、自由部分新聞用 RSS 分類；其他用最近質心自動分，會有分錯的。
- **示範紀錄是模擬資料**（`interactions.simulated = 1`），儀表板的點擊率不計入。
- 點擊追蹤網址是 `localhost`，只有本機能用；DW 的標題摘要由簡體轉成繁體後儲存。
