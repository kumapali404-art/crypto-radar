"""engine.py - konfigurasi, pengumpul data (berita, pasar, makro) dan mesin sinyal.
Semua angka dihitung di sini (kode), bukan oleh LLM."""
import calendar, csv, hashlib, html, io, logging, os, re, sqlite3, time, json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cached_property
from urllib.parse import quote

import feedparser, numpy as np, pandas as pd, requests, yfinance as yf

log = logging.getLogger("macro")


def env(k, d=None):
    v = os.getenv(k)
    return d if v is None or v.strip() == "" else v.strip()


# ============ KONFIGURASI ============
TZ = env("TIMEZONE", "Asia/Jakarta")
FRED_KEY = env("FRED_API_KEY")
FRESH_H = int(env("LOOKBACK_HOURS", "6"))      # berita yang dianalisis
MAX_ITEMS = int(env("MAX_ITEMS", "18"))
MIN_SCORE = int(env("MIN_SCORE", "3"))
DB = env("DB_PATH", "seen.db")
UA = {"User-Agent": "Mozilla/5.0 (macro-wire-bot)"}


def _watch():
    """WATCH_LEVELS="US 30Y:5.5:above,USD/JPY:160:above" """
    out = {}
    for part in env("WATCH_LEVELS", "US 30Y:5.5:above").split(","):
        try:
            name, lvl, d = part.rsplit(":", 2)
            out[name.strip()] = (float(lvl), d.strip())
        except ValueError:
            pass
    return out


WATCH = _watch()


def gn(q):
    return f"https://news.google.com/rss/search?q={quote(q + ' when:1d')}&hl=en-US&gl=US&ceid=US:en"


FEEDS = [
    ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml", 1),
    ("BLS", "https://www.bls.gov/feed/bls_latest.rss", 1),
    ("ECB", "https://www.ecb.europa.eu/rss/press.html", 1),
    ("Bank of Japan", "https://www.boj.or.jp/en/rss/whatsnew.xml", 1),
    ("CNBC Economy", "https://www.cnbc.com/id/20910258/device/rss/rss.html", 2),
    ("CNBC Finance", "https://www.cnbc.com/id/10000664/device/rss/rss.html", 2),
    ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/", 2),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex", 2),
    ("FXStreet", "https://www.fxstreet.com/rss/news", 2),
    ("Nikkei Asia", "https://asia.nikkei.com/rss/feed/nar", 2),
    ("Japan Times", "https://www.japantimes.co.jp/feed/", 2),
] + [("GNews", gn(q), 2) for q in [
    "Treasury yields 30-year", "Treasury auction demand", "Federal Reserve rate Powell",
    "Fed officials speech rates outlook", "Bank of Japan rate Ueda", "Japan inflation CPI",
    "JGB yield record", "Japan government bond buyers demand", "US dollar index DXY",
    "yen USD/JPY intervention", "US CPI inflation PCE", "US jobs report payrolls",
    "term premium deficit", "why dollar strengthening", "bond selloff yields surge",
    "oil prices inflation expectations"]]

GOOD = ("reuters", "bloomberg", "wsj", "wall street journal", "financial times", "ft.com",
        "cnbc", "marketwatch", "barron", "nikkei", "japan times", "kyodo", "nhk", "jiji",
        "investing.com", "fxstreet", "yahoo", "the economist", "associated press", "apnews",
        "ap news", "axios", "politico", "fortune", "business insider", "conference board",
        "globe and mail", "morningstar", "dailyfx", "trading economics", "tradingeconomics",
        "guardian", "bbc", "cnn", "nytimes", "new york times", "washington post", "forbes",
        "federalreserve", "treasury.gov", "bls.gov", "boj.or.jp", "mof.go.jp", "ecb.europa")
BAD_TITLE = re.compile(
    r"\?\s*$|should you|stocks? to (buy|watch)|price (prediction|forecast)|nibble|"
    r"blood in the streets|top \d+|podcast|\bvideo\b|opinion:|sponsored|horoscope", re.I)

TICKERS = [
    ("US 5Y", "^FVX", "yield"), ("US 10Y", "^TNX", "yield"), ("US 30Y", "^TYX", "yield"),
    ("DXY", "DX-Y.NYB", "px"), ("USD/JPY", "JPY=X", "px"), ("EUR/USD", "EURUSD=X", "px"),
    ("Nikkei 225", "^N225", "px"), ("S&P 500", "^GSPC", "px"), ("VIX", "^VIX", "px"),
    ("Gold", "GC=F", "px"), ("WTI", "CL=F", "px"), ("BTC", "BTC-USD", "px"),
]
FRED_DAILY = {"DGS1MO": "1M", "DGS3MO": "3M", "DGS6MO": "6M", "DGS1": "1Y", "DGS2": "2Y",
              "DGS3": "3Y", "DGS5": "5Y", "DGS7": "7Y", "DGS10": "10Y", "DGS20": "20Y",
              "DGS30": "30Y"}
FRED_EXTRA = ["DFII10", "T10YIE", "T5YIFR", "THREEFYTP10", "DFF", "BAMLH0A0HYM2"]
FRED_MACRO = [  # (label, series, units)
    ("US CPI YoY", "CPIAUCSL", "pc1"), ("US Core CPI YoY", "CPILFESL", "pc1"),
    ("US Core PCE YoY", "PCEPILFE", "pc1"), ("US Unemployment", "UNRATE", "lin"),
    ("US Payrolls (chg, ribu)", "PAYEMS", "chg"), ("JP CPI YoY", "JPNCPIALLMINMEI", "pc1"),
    ("JP Policy Rate", "IRSTCB01JPM156N", "lin"),
]
KEYWORDS = [
    (r"\b(fed|fomc|powell|federal reserve)\b", 3), (r"bank of japan|\bboj\b|ueda|\bjgb", 3),
    (r"treasury yield|10-year|30-year|\byields?\b|term premium", 3),
    (r"\bcpi\b|inflation|\bpce\b|\bppi\b", 3),
    (r"rate (cut|hike|decision)|interest rate|policy rate", 3),
    (r"payrolls|jobs report|unemployment|jobless", 2),
    (r"\bdollar\b|\bdxy\b|\byen\b|usd/jpy|intervention", 2),
    (r"\bgdp\b|recession|tariff|deficit|debt|auction|quantitative", 2),
    (r"\becb\b|lagarde|\bpboc\b|bank of england|\bboe\b", 1), (r"\boil\b|opec|\bgold\b", 1),
]
TOPICS = [
    ("CENTRAL BANK", r"\bfed\b|fomc|powell|\bboj\b|bank of japan|ueda|\becb\b|rate (cut|hike)|interest rate"),
    ("INFLATION", r"cpi|inflation|\bpce\b|\bppi\b|payroll|jobs|unemployment|\bgdp\b"),
    ("YIELDS", r"yield|treasury|jgb|auction|deficit|debt|term premium"),
    ("FX", r"dollar|dxy|\byen\b|usd/jpy|intervention|euro"),
]
THEMES = {
    "term premium / pasokan obligasi": r"term premium|deficit|debt|auction|issuance|supply",
    "intervensi yen": r"intervention|stealth|yen.*(slide|slump|weak)",
    "kenaikan suku bunga BoJ": r"boj.*(hike|raise)|ueda|japan.*rate hike",
    "JGB / yield Jepang": r"\bjgb|japan.*yield|japanese government bond",
    "inflasi": r"inflation|\bcpi\b|\bpce\b|\bppi\b",
    "jalur suku bunga Fed": r"rate (cut|hike)|fomc|powell|fed officials",
    "minyak / energi": r"\boil\b|crude|opec|brent|energy prices",
    "tarif / perdagangan": r"tariff|trade war|trade deal",
    "pasar tenaga kerja": r"payroll|jobs report|unemployment|jobless",
}


# ============ BERITA ============
@dataclass
class Item:
    id: str
    title: str
    source: str
    tier: int
    link: str
    pub: datetime
    summary: str
    score: int = 0
    body: str = ""
    web: bool = False      # hasil pencarian web (verifikasi lebih rendah)


def clean(t):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t or ""))).strip()


def is_english(t):
    letters = [c for c in t if c.isalpha()]
    return not letters or sum(c.isascii() for c in letters) / len(letters) > 0.9


def good_source(name):
    n = (name or "").lower()
    return any(g in n for g in GOOD)


def fetch_feed(feed):
    name, url, tier = feed
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        entries = feedparser.parse(r.content).entries
    except Exception as e:
        log.warning("feed gagal [%s]: %s", name, e)
        return []
    out = []
    for e in entries:
        t = e.get("published_parsed") or e.get("updated_parsed")
        title, link = clean(e.get("title", "")), e.get("link", "")
        if not (t and title and link):
            continue
        src = name
        if name == "GNews" and " - " in title:
            title, src = title.rsplit(" - ", 1)
        summ = clean(e.get("summary", ""))[:400]
        if summ.lower().startswith(title.lower()[:40]):
            summ = ""
        uid = hashlib.sha1(re.sub(r"\W+", "", title.lower()).encode()).hexdigest()[:16]
        out.append(Item(uid, title, src, tier, link,
                        datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc), summ))
    return out


def toks(t):
    return {w for w in re.findall(r"[a-z0-9\.%]+", t.lower()) if len(w) > 2}


def dedupe(items):
    kept = []
    for it in items:
        t = toks(it.title)
        if not any(len(t & toks(k.title)) / max(1, len(t | toks(k.title))) > 0.6 for k in kept):
            kept.append(it)
    return kept


def collect():
    """Berita relevan 24 jam terakhir dari sumber kredibel (juga dipakai deteksi momentum tema)."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    with ThreadPoolExecutor(8) as ex:
        items = [i for b in ex.map(fetch_feed, FEEDS) for i in b if i.pub >= cutoff
                 and (i.tier == 1 or good_source(i.source)) and is_english(i.title)
                 and not BAD_TITLE.search(i.title)]
    for i in items:
        blob = f"{i.title} {i.summary}".lower()
        s = sum(w for p, w in KEYWORDS if re.search(p, blob))
        i.score = s + (3 if i.tier == 1 and s else 0)
    items = [i for i in items if i.score >= MIN_SCORE]
    items.sort(key=lambda x: (-x.score, x.tier, -x.pub.timestamp()))
    items = dedupe(items)
    items.sort(key=lambda x: (-x.score, -x.pub.timestamp()))
    log.info("berita relevan 24j: %d", len(items))
    return items


def extract_text(raw):
    try:
        import trafilatura
        t = trafilatura.extract(raw, include_comments=False, include_tables=False)
        if t:
            return t
    except Exception:
        pass
    paras = re.findall(r"<p[^>]*>(.*?)</p>", raw, flags=re.S | re.I)
    return " ".join(p for p in (clean(x) for x in paras) if len(p) > 60)


def enrich(items, limit=12):
    """Ambil isi artikel (bila terbuka) agar analisis tidak hanya berdasar headline."""
    targets = [i for i in items if "news.google.com" not in i.link][:limit]

    def go(i):
        try:
            r = requests.get(i.link, headers=UA, timeout=8)
            if r.ok and "html" in r.headers.get("content-type", ""):
                i.body = extract_text(r.text)[:1800]
        except Exception:
            pass
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(go, targets))
    log.info("isi artikel terambil: %d/%d", sum(1 for i in targets if i.body), len(targets))


# ============ DB ============
def db():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, ts REAL)")
    c.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
    return c


def unseen(items):
    with db() as c:
        seen = {r[0] for r in c.execute("SELECT id FROM seen")}
    return [i for i in items if i.id not in seen]


def mark_seen(items):
    now = time.time()
    with db() as c:
        c.executemany("INSERT OR IGNORE INTO seen VALUES (?,?)", [(i.id, now) for i in items])
        c.execute("DELETE FROM seen WHERE ts < ?", (now - 7 * 86400,))


def kv_get(k, default=None):
    with db() as c:
        r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else default


def kv_set(k, v):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v)))


# ============ PASAR ============
@dataclass
class Quote:
    label: str
    kind: str            # "yield" (perubahan dalam bp) | "px" (perubahan dalam %)
    s: pd.Series         # harga/yield harian, index tz-naive

    @property
    def last(self):
        return float(self.s.iloc[-1])

    @property
    def prev(self):
        return float(self.s.iloc[-2])

    @property
    def unit(self):
        return "bp" if self.kind == "yield" else "%"

    @property
    def asof(self):
        return self.s.index[-1]

    def fmt(self):
        return f"{self.last:,.3f}" if self.kind == "yield" else f"{self.last:,.2f}"

    def chg(self, n=1):
        if len(self.s) <= n:
            return None
        a, b = float(self.s.iloc[-1]), float(self.s.iloc[-1 - n])
        return (a - b) * 100 if self.kind == "yield" else (a / b - 1) * 100

    def changes(self):
        return self.s.diff() * 100 if self.kind == "yield" else self.s.pct_change() * 100

    @cached_property
    def note(self):
        s = self.s
        if len(s) < 60:
            return ""
        last, hist = s.iloc[-1], s.iloc[:-1]
        yrs = max(1, round((s.index[-1] - s.index[0]).days / 365))
        recent = hist.iloc[-20:]
        for word, hit, older in (("highest", last > recent.max(), hist[hist >= last]),
                                 ("lowest", last < recent.min(), hist[hist <= last])):
            if hit:
                return f"{word} in {yrs}+ yrs" if older.empty else f"{word} since {older.index[-1]:%b %Y}"
        return ""

    @property
    def flag(self):
        w = WATCH.get(self.label)
        if w and w[1] == "above" and self.last >= w[0]:
            return f"≥{w[0]:g}"
        if w and w[1] == "below" and self.last <= w[0]:
            return f"≤{w[0]:g}"
        return ""


@dataclass
class Market:
    quotes: dict = field(default_factory=dict)
    fred: dict = field(default_factory=dict)       # series_id -> pd.Series
    macro: dict = field(default_factory=dict)      # label -> (value, prev, date)
    auctions: list = field(default_factory=list)

    def q(self, label):
        return self.quotes.get(label)

    def f(self, sid):
        s = self.fred.get(sid)
        return s if s is not None and len(s) else None


def one_quote(t):
    label, tk, kind = t
    for attempt in range(2):
        try:
            h = yf.Ticker(tk).history(period="10y", interval="1d")["Close"].dropna()
            if len(h) < 25:
                return None
            h.index = h.index.tz_localize(None).normalize()
            h = h[~h.index.duplicated(keep="last")]
            return Quote(label, kind, h.astype(float))
        except Exception as e:
            log.warning("quote gagal [%s] (%d): %s", tk, attempt + 1, e)
            time.sleep(1)
    return None


def load_jgb():
    """Yield JGB harian dari Kementerian Keuangan Jepang (gratis)."""
    base = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"
    for name in ("jgbcm_all.csv", "jgbcm.csv"):
        try:
            r = requests.get(base + name, headers=UA, timeout=30)
            r.raise_for_status()
            rows = list(csv.reader(io.StringIO(r.content.decode("utf-8", "ignore"))))
            hi = next(i for i, row in enumerate(rows) if row and row[0].strip().lower() == "date")
            head = [h.strip() for h in rows[hi]]
            idx, recs = [], []
            for row in rows[hi + 1:]:
                try:
                    y, m, d = (int(x) for x in row[0].strip().replace("-", "/").split("/"))
                    ts = pd.Timestamp(y, m, d)
                except (ValueError, IndexError):
                    continue
                vals = {}
                for c, v in zip(head[1:], row[1:]):
                    try:
                        vals[c] = float(v)
                    except ValueError:
                        pass
                idx.append(ts)
                recs.append(vals)
            df = pd.DataFrame(recs, index=idx).sort_index()
            if len(df) >= 3:
                out = {}
                for col in ("2Y", "5Y", "10Y", "20Y", "30Y", "40Y"):
                    if col in df:
                        s = df[col].dropna().iloc[-2600:]
                        if len(s) >= 3:
                            out[f"JGB {col}"] = Quote(f"JGB {col}", "yield", s)
                if out:
                    return out
        except Exception as e:
            log.warning("JGB gagal (%s): %s", name, e)
    return {}


def fred_series(sid, n=300, units="lin"):
    for attempt in range(2):
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=20,
                             params=dict(series_id=sid, api_key=FRED_KEY, file_type="json",
                                         sort_order="desc", limit=n, units=units))
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            obs = {pd.Timestamp(o["date"]): float(o["value"])
                   for o in r.json()["observations"] if o["value"] not in (".", "")}
            return pd.Series(obs, dtype=float).sort_index()
        except Exception as e:
            log.warning("FRED gagal [%s]: %s", sid, e)
    return pd.Series(dtype=float)


def load_fred(mk):
    if not FRED_KEY:
        return
    ids = list(FRED_DAILY) + FRED_EXTRA
    with ThreadPoolExecutor(6) as ex:
        for sid, s in zip(ids, ex.map(fred_series, ids)):
            if len(s):
                mk.fred[sid] = s
        macro = list(ex.map(lambda t: fred_series(t[1], 4, t[2]), FRED_MACRO))
    for (label, _, _), s in zip(FRED_MACRO, macro):
        if len(s):
            mk.macro[label] = (float(s.iloc[-1]), float(s.iloc[-2]) if len(s) > 1 else None,
                               s.index[-1].strftime("%Y-%m"))


def load_auctions():
    """Hasil lelang Treasury 10Y/30Y dari TreasuryDirect (gratis)."""
    out = []
    end = datetime.now()
    start = end - timedelta(days=420)
    for typ, pat, name in (("Bond", r"^(30|29)-Year", "30Y"), ("Note", r"^(10|9)-Year", "10Y")):
        try:
            r = requests.get("https://www.treasurydirect.gov/TA_WS/securities/search", headers=UA,
                             timeout=25, params=dict(format="json", type=typ,
                                                     startDate=start.strftime("%Y-%m-%d"),
                                                     endDate=end.strftime("%Y-%m-%d")))
            r.raise_for_status()
            rows = []
            for x in r.json():
                if not re.match(pat, str(x.get("securityTerm", ""))):
                    continue
                try:
                    rows.append(dict(date=pd.Timestamp(str(x["auctionDate"])[:10]),
                                     hy=float(x["highYield"]), btc=float(x["bidToCoverRatio"])))
                except (KeyError, ValueError, TypeError):
                    continue
            rows.sort(key=lambda z: z["date"])
            if rows:
                out.append(dict(name=name, rows=rows[-8:]))
        except Exception as e:
            log.warning("lelang %s gagal: %s", name, e)
    return out


def load_market():
    mk = Market()
    with ThreadPoolExecutor(6) as ex:
        for q in ex.map(one_quote, TICKERS):
            if q:
                mk.quotes[q.label] = q
    mk.quotes.update(load_jgb())
    load_fred(mk)
    mk.auctions = load_auctions()
    log.info("pasar: %d kuotasi, %d seri FRED, %d lelang", len(mk.quotes), len(mk.fred), len(mk.auctions))
    return mk


def us_jp_gap(mk):
    us, jp, fx = mk.q("US 10Y"), mk.q("JGB 10Y"), mk.q("USD/JPY")
    if not (us and jp and fx):
        return None
    df = pd.concat([us.s, jp.s, fx.s], axis=1, keys=["us", "jp", "fx"]).dropna()
    if len(df) < 5:
        return None
    df["gap"] = df["us"] - df["jp"]
    return df


# ============ MESIN SINYAL ============
@dataclass
class Signal:
    kind: str
    key: str            # stabil antar-run, dipakai membedakan sinyal baru vs berlanjut
    text: str
    w: float            # bobot kepentingan
    id: str = ""
    new: bool = True


def sig_since(mk, last):
    if not last or not last.get("vals"):
        return []
    parts = []
    for lab in ("US 10Y", "US 30Y", "DXY", "USD/JPY", "Gold", "WTI", "S&P 500"):
        q, p = mk.q(lab), last["vals"].get(lab)
        if not q or not p:
            continue
        ch, th, u = ((q.last - p) * 100, 2, "bp") if q.kind == "yield" else ((q.last / p - 1) * 100, 0.25, "%")
        if abs(ch) >= th:
            parts.append(f"{lab} {ch:+.1f}{u}")
    if not parts:
        return []
    mins = (time.time() - last["ts"]) / 60
    return [Signal("since", "since", f"Sejak laporan {mins:.0f} menit lalu: " + ", ".join(parts), 9)]


def sig_levels(mk):
    out = []
    for q in mk.quotes.values():
        if q.flag:
            out.append(Signal("level", f"flag:{q.label}", f"{q.label} {q.fmt()}: level pantau {q.flag} tersentuh", 7))
        if q.note:
            out.append(Signal("level", f"ext:{q.label}:{q.note}", f"{q.label} {q.fmt()}: {q.note}", 6.5))
    return out


def sig_moves(mk):
    out = []
    for q in mk.quotes.values():
        r = q.changes().dropna()
        if len(r) < 70:
            continue
        sd = r.iloc[-61:-1].std()
        if sd and abs(r.iloc[-1] / sd) >= 2:
            z = r.iloc[-1] / sd
            out.append(Signal("move", f"z:{q.label}:{'+' if z > 0 else '-'}",
                              f"{q.label} {r.iloc[-1]:+.1f}{q.unit} d/d = {z:+.1f}σ dibanding 60 sesi terakhir",
                              6 + min(abs(z), 4)))
    return out


def sig_streak(mk):
    out = []
    for q in mk.quotes.values():
        if q.label in ("VIX", "BTC") or len(q.s) < 12:
            continue
        d = q.s.diff().dropna()
        sign, n = np.sign(d.iloc[-1]), 0
        for v in d.iloc[::-1]:
            if sign != 0 and np.sign(v) == sign:
                n += 1
            else:
                break
        if n >= 4:
            out.append(Signal("streak", f"streak:{q.label}:{int(sign)}",
                              f"{q.label} {'naik' if sign > 0 else 'turun'} {n} sesi beruntun ({q.chg(n):+.1f}{q.unit} kumulatif)", 3))
    return out


def sig_divergence(mk):
    out = []
    y, dx, jp, sp, au = mk.q("US 10Y"), mk.q("DXY"), mk.q("USD/JPY"), mk.q("S&P 500"), mk.q("Gold")
    if not y:
        return out
    cy = y.chg(5)
    if cy is None:
        return out

    def g(q):
        return q.chg(5) if q else None
    cd, cj, cs, ca = g(dx), g(jp), g(sp), g(au)
    if cy >= 8 and cd is not None and cd <= -0.3:
        out.append(Signal("div", "div:y-dxy", f"Divergensi 5 sesi: US 10Y {cy:+.0f}bp tetapi DXY {cd:+.1f}%", 8))
    if cy >= 8 and cj is not None and cj <= -0.5:
        out.append(Signal("div", "div:y-jpy", f"Divergensi 5 sesi: US 10Y {cy:+.0f}bp tetapi USD/JPY {cj:+.1f}%", 8))
    if cy >= 8 and cs is not None and cs >= 1:
        out.append(Signal("div", "div:y-spx", f"5 sesi: S&P 500 {cs:+.1f}% bersamaan dengan US 10Y {cy:+.0f}bp", 6))
    if cy <= -8 and cd is not None and cd >= 0.5:
        out.append(Signal("div", "div:ydn-dxy", f"Divergensi 5 sesi: US 10Y {cy:+.0f}bp tetapi DXY {cd:+.1f}%", 8))
    real = mk.f("DFII10")
    if real is not None and len(real) > 6 and ca is not None:
        cr = (real.iloc[-1] - real.iloc[-6]) * 100
        if (ca >= 1.5 and cr >= 8) or (ca <= -1.5 and cr <= -8):
            out.append(Signal("div", "div:gold-real", f"Emas {ca:+.1f}% dan real yield 10Y {cr:+.0f}bp searah dalam 5 sesi (biasanya berlawanan)", 8))
    return out


def sig_corr(mk):
    out = []
    for a, b in (("S&P 500", "US 10Y"), ("DXY", "US 10Y"), ("USD/JPY", "US 10Y"),
                 ("Gold", "DXY"), ("BTC", "S&P 500")):
        qa, qb = mk.q(a), mk.q(b)
        if not qa or not qb:
            continue
        df = pd.concat([qa.changes(), qb.changes()], axis=1, keys=["a", "b"]).dropna()
        if len(df) < 130:
            continue
        c20, c120 = df.iloc[-20:].corr().iloc[0, 1], df.iloc[-120:].corr().iloc[0, 1]
        flip = np.sign(c20) != np.sign(c120) and abs(c20) >= 0.3
        if flip or abs(c20 - c120) >= 0.5:
            out.append(Signal("corr", f"corr:{a}:{b}:{int(flip)}",
                              f"Korelasi {a} vs {b}: {c20:+.2f} (20 sesi) vs {c120:+.2f} (120 sesi), "
                              f"{'berbalik arah' if flip else 'bergeser tajam'}", 7))
    return out


def sig_usjp(mk):
    df = us_jp_gap(mk)
    if df is None:
        return []
    gap = df["gap"]
    t = f"Selisih US 10Y–JGB 10Y {gap.iloc[-1]:.2f}pp"
    if len(gap) > 1:
        t += f" ({(gap.iloc[-1] - gap.iloc[-2]) * 100:+.0f}bp d/d)"
    if len(df) > 22:
        t += f", {(gap.iloc[-1] - gap.iloc[-22]) * 100:+.0f}bp 1M; USD/JPY {(df['fx'].iloc[-1] / df['fx'].iloc[-22] - 1) * 100:+.1f}% 1M"
    if len(df) > 80:
        c = pd.concat([gap.diff(), df["fx"].pct_change()], axis=1).dropna().iloc[-60:].corr().iloc[0, 1]
        t += f"; korelasi 60 sesi {c:+.2f}"
    return [Signal("usjp", "usjp", t, 6)]


def sig_decomp(mk):
    out = []
    nom, real, be = mk.f("DGS10"), mk.f("DFII10"), mk.f("T10YIE")
    if nom is not None and real is not None and be is not None:
        df = pd.concat({"nom": nom, "real": real, "be": be}, axis=1).dropna()
        for n in (5, 20):
            if len(df) > n:
                ch = (df.iloc[-1] - df.iloc[-1 - n]) * 100
                if abs(ch["nom"]) >= 8:
                    out.append(Signal("decomp", f"dec:{n}",
                                      f"US 10Y {ch['nom']:+.0f}bp dalam {n} hari kerja (s.d. {df.index[-1]:%d %b}): "
                                      f"real yield {ch['real']:+.0f}bp, breakeven inflasi {ch['be']:+.0f}bp", 8))
    tp = mk.f("THREEFYTP10")
    if tp is not None and len(tp) > 21:
        out.append(Signal("decomp", "tp", f"Term premium 10Y (Kim-Wright) {tp.iloc[-1]:.2f}%, "
                          f"{(tp.iloc[-1] - tp.iloc[-21]) * 100:+.0f}bp dalam 20 hari kerja (data {tp.index[-1]:%d %b})", 6))
    f5 = mk.f("T5YIFR")
    if f5 is not None and len(f5) > 21 and abs(f5.iloc[-1] - f5.iloc[-21]) * 100 >= 8:
        out.append(Signal("decomp", "5y5y", f"Ekspektasi inflasi 5y5y {f5.iloc[-1]:.2f}%, "
                          f"{(f5.iloc[-1] - f5.iloc[-21]) * 100:+.0f}bp dalam 20 hari kerja", 6))
    return out


def _classify(ds, dl):
    """Gerakan kurva: bull/bear steepening/flattening dari perubahan sisi pendek & panjang."""
    return ("bear" if ds + dl > 0 else "bull") + " " + ("steepening" if dl > ds else "flattening")


def sig_curve(mk):
    out = []
    s2, s10, s3m, s5, s30 = (mk.f(x) for x in ("DGS2", "DGS10", "DGS3MO", "DGS5", "DGS30"))
    if s2 is not None and s10 is not None:
        df = pd.concat([s2, s10], axis=1, keys=["a", "b"]).dropna()
        if len(df) > 22:
            sp = (df["b"] - df["a"]) * 100
            d2, d10 = df["a"].iloc[-1] - df["a"].iloc[-22], df["b"].iloc[-1] - df["b"].iloc[-22]
            if abs(d10) * 100 >= 8 or abs(sp.iloc[-1] - sp.iloc[-22]) >= 8:
                out.append(Signal("curve", "curve:2s10s",
                                  f"Kurva 1 bulan: {_classify(d2, d10)}; 2Y {d2 * 100:+.0f}bp, 10Y {d10 * 100:+.0f}bp; "
                                  f"2s10s {sp.iloc[-1]:+.0f}bp (s.d. {df.index[-1]:%d %b})", 7))
    if s3m is not None and s10 is not None:
        df = pd.concat([s3m, s10], axis=1, keys=["a", "b"]).dropna()
        if len(df) and df["b"].iloc[-1] < df["a"].iloc[-1]:
            out.append(Signal("curve", "curve:3m10y", f"Kurva 3M–10Y terbalik (inverted): {(df['b'].iloc[-1] - df['a'].iloc[-1]) * 100:+.0f}bp", 6))
    q10, q30 = mk.q("US 10Y"), mk.q("US 30Y")
    if q10 and q30 and len(q10.s) > 22:
        df = pd.concat([q10.s, q30.s], axis=1, keys=["a", "b"]).dropna()
        sp = (df["b"] - df["a"]) * 100
        if len(sp) > 30:
            dm = sp.iloc[-1] - sp.iloc[-22]
            if abs(dm) >= 5:
                pct = (sp < sp.iloc[-1]).mean() * 100
                out.append(Signal("curve", "curve:10s30s",
                                  f"Spread 10s30s {sp.iloc[-1]:.0f}bp, {dm:+.0f}bp dalam 1 bulan "
                                  f"({_classify(df['a'].iloc[-1] - df['a'].iloc[-22], df['b'].iloc[-1] - df['b'].iloc[-22])}); "
                                  f"persentil {pct:.0f} dari 10 thn", 6))
    return out


def sig_policy(mk):
    out = []
    s2, ff = mk.f("DGS2"), mk.f("DFF")
    if s2 is not None and ff is not None:
        gap = (s2.iloc[-1] - ff.iloc[-1]) * 100
        if abs(gap) >= 15:
            out.append(Signal("policy", "pol2y", f"US 2Y {s2.iloc[-1]:.2f}% vs Fed Funds efektif {ff.iloc[-1]:.2f}%: selisih {gap:+.0f}bp "
                              f"(konsisten dengan ekspektasi {'pelonggaran' if gap < 0 else 'pengetatan'} ke depan)", 6))
    m = mk.macro
    if ff is not None and "US Core CPI YoY" in m:
        v, _, d = m["US Core CPI YoY"]
        out.append(Signal("policy", "realff", f"Fed Funds {ff.iloc[-1]:.2f}% vs Core CPI YoY {v:.2f}% ({d}): suku bunga riil kebijakan {ff.iloc[-1] - v:+.2f}pp", 4))
    if "JP Policy Rate" in m and "JP CPI YoY" in m:
        r, _, d1 = m["JP Policy Rate"]
        c, _, d2 = m["JP CPI YoY"]
        out.append(Signal("policy", "realboj", f"Suku bunga BoJ {r:.2f}% ({d1}) vs CPI Jepang YoY {c:.2f}% ({d2}): suku bunga riil {r - c:+.2f}pp", 5))
    return out


def sig_credit(mk):
    out = []
    hy = mk.f("BAMLH0A0HYM2")
    if hy is None or len(hy) < 30:
        return out
    lvl, d1m = hy.iloc[-1], (hy.iloc[-1] - hy.iloc[-22]) * 100
    pct = (hy < lvl).mean() * 100
    if abs(d1m) >= 20 or pct >= 90 or pct <= 10:
        out.append(Signal("credit", "hy", f"Spread kredit HY (OAS) {lvl * 100:.0f}bp, {d1m:+.0f}bp 1M, persentil {pct:.0f} dari {len(hy)} observasi", 6))
    sp = mk.q("S&P 500")
    if sp and sp.chg(21) is not None and sp.chg(21) >= 2 and d1m >= 20:
        out.append(Signal("credit", "hy-spx", f"S&P 500 {sp.chg(21):+.1f}% 1M sementara spread HY melebar {d1m:+.0f}bp (ekuitas dan kredit tidak sejalan)", 8))
    return out


def sig_auction(mk):
    out = []
    for a in mk.auctions:
        rows = a["rows"]
        last = rows[-1]
        age = (datetime.now() - last["date"].to_pydatetime()).days
        if age > 14 or len(rows) < 3:
            continue
        prior = rows[:-1][-6:]
        avg = sum(r["btc"] for r in prior) / len(prior)
        dhy = (last["hy"] - rows[-2]["hy"]) * 100
        out.append(Signal("auction", f"auc:{a['name']}:{last['date']:%Y%m%d}",
                          f"Lelang Treasury {a['name']} ({last['date']:%d %b}): high yield {last['hy']:.3f}% ({dhy:+.1f}bp vs lelang sebelumnya), "
                          f"bid-to-cover {last['btc']:.2f} vs rata-rata {avg:.2f}", 8 if age <= 3 else 5))
    return out


def sig_jgb(mk):
    j10, j30 = mk.q("JGB 10Y"), mk.q("JGB 30Y")
    if not (j10 and j30):
        return []
    df = pd.concat([j10.s, j30.s], axis=1, keys=["a", "b"]).dropna()
    if len(df) < 23:
        return []
    sp = (df["b"] - df["a"]) * 100
    dm = sp.iloc[-1] - sp.iloc[-22]
    if abs(dm) >= 5:
        return [Signal("jgb", "jgbcurve", f"Spread JGB 10s30s {sp.iloc[-1]:.0f}bp, {dm:+.0f}bp dalam 1 bulan", 5)]
    return []


def sig_commodity(mk):
    out = []
    wti, be = mk.q("WTI"), mk.f("T10YIE")
    if wti and be is not None and len(be) > 22 and wti.chg(21) is not None:
        cb = (be.iloc[-1] - be.iloc[-22]) * 100
        if abs(wti.chg(21)) >= 8 and abs(cb) >= 8 and np.sign(wti.chg(21)) == np.sign(cb):
            out.append(Signal("commodity", "wti-be", f"WTI {wti.chg(21):+.1f}% dan breakeven 10Y {cb:+.0f}bp bergerak searah dalam 1 bulan", 6))
    vix = mk.q("VIX")
    if vix and len(vix.s) > 250:
        pct = (vix.s.iloc[-250:] < vix.last).mean() * 100
        if pct >= 90 or pct <= 10:
            out.append(Signal("commodity", "vix", f"VIX {vix.fmt()} di persentil {pct:.0f} dalam 1 tahun", 5))
    return out


def sig_themes(news):
    now = datetime.now(timezone.utc)
    out = []
    for name, pat in THEMES.items():
        c24 = c6 = 0
        for i in news:
            if re.search(pat, f"{i.title} {i.summary}".lower()):
                c24 += 1
                c6 += now - i.pub <= timedelta(hours=6)
        base = (c24 - c6) / 3
        if c6 >= 3 and c6 >= 1.5 * max(base, 0.5):
            out.append(Signal("theme", f"theme:{name}", f"Momentum narasi '{name}': {c6} berita dalam 6 jam terakhir vs rata-rata {base:.1f} per 6 jam sebelumnya", 5))
    return out


def build_signals(mk, news, last):
    fns = (lambda: sig_since(mk, last), lambda: sig_levels(mk), lambda: sig_moves(mk),
           lambda: sig_divergence(mk), lambda: sig_corr(mk), lambda: sig_usjp(mk),
           lambda: sig_decomp(mk), lambda: sig_curve(mk), lambda: sig_policy(mk),
           lambda: sig_credit(mk), lambda: sig_auction(mk), lambda: sig_jgb(mk),
           lambda: sig_commodity(mk), lambda: sig_themes(news), lambda: sig_streak(mk))
    out = []
    for fn in fns:
        try:
            out += fn()
        except Exception:
            log.exception("sinyal gagal")
    out.sort(key=lambda s: -s.w)
    for n, s in enumerate(out, 1):
        s.id = f"S{n}"
    return out
