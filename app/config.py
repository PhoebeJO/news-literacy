"""設定：媒體白名單、領域、演算法參數。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "news.db"
IDF_PATH = DATA_DIR / "idf.npy"

# 媒體白名單：ownership 只寫可查證的事實，不寫立場標籤
# feeds: (RSS 網址, 領域標籤或 None)；None 代表之後用向量最近質心分類
SOURCES = [
    {
        "id": 1, "name": "中央社", "country": "台灣",
        "ownership": "國家通訊社（財團法人中央通訊社）",
        "feeds": [
            ("https://feeds.feedburner.com/rsscna/politics", "政治"),
            ("https://feeds.feedburner.com/rsscna/intworld", "國際"),
            ("https://feeds.feedburner.com/rsscna/mainland", "兩岸"),
            ("https://feeds.feedburner.com/rsscna/finance", "財經"),
            ("https://feeds.feedburner.com/rsscna/technology", "科技"),
            ("https://feeds.feedburner.com/rsscna/lifehealth", "生活"),
            ("https://feeds.feedburner.com/rsscna/social", "社會"),
            ("https://feeds.feedburner.com/rsscna/sport", "運動"),
        ],
    },
    {
        "id": 2, "name": "公視新聞網", "country": "台灣",
        "ownership": "公共媒體（財團法人公共電視文化事業基金會）",
        "feeds": [("https://news.pts.org.tw/xml/newsfeed.xml", None)],
    },
    {
        "id": 3, "name": "自由時報", "country": "台灣",
        "ownership": "民營報紙（自由時報企業股份有限公司）",
        "feeds": [
            ("https://news.ltn.com.tw/rss/politics.xml", "政治"),
            ("https://news.ltn.com.tw/rss/world.xml", "國際"),
            ("https://news.ltn.com.tw/rss/business.xml", "財經"),
            ("https://news.ltn.com.tw/rss/all.xml", None),
        ],
    },
    {
        "id": 4, "name": "BBC 中文", "country": "英國",
        "ownership": "公共廣播（英國廣播公司 BBC）",
        "feeds": [("https://feeds.bbci.co.uk/zhongwen/trad/rss.xml", None)],
    },
    {
        "id": 5, "name": "德國之聲 DW", "country": "德國", "simplified": True,
        "ownership": "公共國際廣播（德國聯邦政府預算資助）",
        "feeds": [("https://rss.dw.com/rdf/rss-chi-all", None)],
    },
    {
        "id": 6, "name": "法廣 RFI", "country": "法國",
        "ownership": "公共國際廣播（France Médias Monde）",
        "feeds": [("https://www.rfi.fr/tw/rss", None)],
    },
]

TOPICS = ["政治", "國際", "兩岸", "財經", "科技", "生活", "社會", "運動"]

# 演算法參數
EMBED_DIM = 8192          # 雜湊後的向量維度
EVENT_WINDOW_HOURS = 72   # 事件分群的時間窗
EVENT_SIM_THRESHOLD = 0.22  # 與事件質心的餘弦相似度門檻（見 README 的調校紀錄）
FRESH_DAYS = 14           # 只保留近 14 天新聞的向量
HALF_LIFE_DAYS = 7.0      # 興趣向量的時間衰減半衰期
MMR_LAMBDA = 0.7          # MMR：相關度 vs 多樣性
SAME_SOURCE_PENALTY = 0.3  # MMR 中同一家媒體的額外重複度
REC_K = 4                 # 興趣推薦則數（第 5 則固定是「另一觀點」）
ACTION_WEIGHT = {"click": 1.0, "like": 2.0, "dislike": -1.0, "other_view": 0.0}
READ_ACTIONS = ("click", "like")   # 算進「讀過」的行為
MIN_SOURCES_TO_COMPARE = 3
