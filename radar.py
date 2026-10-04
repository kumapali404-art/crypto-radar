"""Crypto Radar v3: OKX screener (metode Xander: Daily bias -> H4 struktur -> M15 fib 0.5/0.618 + POI) -> Telegram."""
import os, re, csv, json, math, time, threading
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests, feedparser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= KONFIGURASI (boleh diedit) =================
INTERVAL_MIN = int(os.getenv("INTERVAL_MIN", "30"))   # 30 atau 60: satu laporan per slot waktu
FORCE = os.getenv("FORCE", "ya") == "ya"              # "ya" = abaikan penjaga slot (untuk tes manual)
NARRATIVE_EVERY_H = 2                                  # laporan narasi/pantau dikirim tiap N jam (walau tanpa setup)
TOP_N = int(os.getenv("TOP_N", "20"))                  # koin besar yang dianalisis dalam (urut OI)
MIN_OI_USD = 20_000_000
NARR_MIN_OI = 3_000_000                                # batas OI untuk pemindai narasi (termasuk small cap)
MAX_SPREAD_BPS = 6.0
WATCHLIST = ["PUMP", "VVV", "UNI", "AAVE"]             # koin pilihanmu: selalu dianalisis jika ada di OKX
MIN_GRADE = os.getenv("MIN_GRADE", "A")
MAX_ALERTS = 3
COOLDOWN_H = 4
FEE = 0.0005
FILL_H, EVAL_H = 4, 12                                 # entry limit berlaku 4 jam; hasil dinilai setelah 12 jam
NEWS_BLACKOUT_UTC = []                                 # mis. ["2026-10-09 12:30"] (UTC), +-45 menit
BLACKOUT_MIN = 45
OCT10_TS = 1760054400000
MODELS = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash-latest"]
NARASI = {
    "DeFi/DEX": ["UNI", "AAVE", "LDO", "CRV", "MKR", "SNX", "COMP", "PENDLE", "ENA", "JUP", "DYDX", "GMX", "CAKE", "SUSHI"],
    "AI": ["TAO", "FET", "RENDER", "VVV", "VIRTUAL", "AI16Z", "ARKM", "WLD", "GRT", "AIXBT", "KAITO"],
    "Meme/Launchpad": ["DOGE", "SHIB", "PEPE", "WIF", "BONK", "PUMP", "FARTCOIN", "TRUMP", "POPCAT", "PNUT", "FLOKI", "BOME", "BRETT"],
    "Layer-1/2": ["SOL", "AVAX", "SUI", "APT", "SEI", "TON", "NEAR", "ADA", "DOT", "ATOM", "ARB", "OP", "STRK", "POL", "MNT", "TIA", "INJ"],
    "Perp DEX": ["HYPE", "DYDX", "GMX", "ASTER"],
    "RWA/Oracle": ["ONDO", "LINK", "PYTH", "API3"],
    "Privasi": ["XMR", "ZEC", "DASH"],
}
LLAMA_ALIAS = {"UNI": "uniswap", "AAVE": "aave", "PUMP": "pump", "HYPE": "hyperliquid", "LDO": "lido", "MKR": "sky", "CRV": "curve", "JUP": "jupiter", "ENA": "ethena", "PENDLE": "pendle"}
SIGNALS_CSV, SLOT_FILE = "data/signals.csv", "data/last_slot.txt"
# ===============================================================
BASE = "https://www.okx.com/api/v5/"
HDR = {"User-Agent": "Mozilla/5.0 (compatible; CryptoRadar/3.0)"}
BARS = ["5m", "15m", "30m", "1H", "4H", "1Dutc"]
GR = {"A+": 4, "A": 3, "B+": 2, "B": 1}
LOG = []
def log(m): print(m, flush=True); LOG.append(m)

# ---------------- akses data OKX ----------------
_lock = threading.Lock(); _next = {}
def throttle(cat, gap):
    with _lock:
        now = time.time(); t = max(now, _next.get(cat, 0)); _next[cat] = t + gap
    if t > now: time.sleep(t - now)

def okx(path, gap=0.15, **p):
    cat = "rubik" if "rubik" in path else "x"
    for _ in range(3):
        try:
            throttle(cat, max(gap, 0.45) if cat == "rubik" else gap)
            j = requests.get(BASE + path, params=p, headers=HDR, timeout=20).json()
            if j.get("code") == "0": return j.get("data", [])
            if j.get("code") in ("50011", "50061"): time.sleep(1.5); continue
            log(f"OKX {path} {p.get('instId') or p.get('ccy') or ''} -> code={j.get('code')} {str(j.get('msg'))[:60]}")
            return []
        except Exception:
            time.sleep(.5)
    log(f"OKX {path} GAGAL"); return []

def candles(inst, bar, limit=300):
    rows = okx("market/candles", instId=inst, bar=bar, limit=limit)
    out = [[int(r[0])] + [float(x) for x in r[1:6]] + [int(r[8]) if len(r) > 8 else 1] for r in reversed(rows)]
    return np.array(out, float) if out else np.zeros((0, 7))   # ts,o,h,l,c,vol,confirm

def cl(a): return a[a[:, 6] == 1] if len(a) else a

def fetch_coin(inst):
    sym = inst.split("-")[0]; d = {"inst": inst, "sym": sym, "c": {b: candles(inst, b) for b in BARS}}
    rb = okx("rubik/stat/contracts/open-interest-volume", ccy=sym, period="5m")
    d["oi5"] = np.array([[int(r[0]), float(r[1])] for r in reversed(rb)]) if rb else np.zeros((0, 2))
    fr = okx("public/funding-rate", instId=inst); d["fund"] = float(fr[0]["fundingRate"]) if fr else None
    d["fh"] = [float(x["fundingRate"]) for x in okx("public/funding-rate-history", instId=inst, limit=60)]
    lg = sh = 0.0; now = time.time() * 1000
    for blk in okx("public/liquidation-orders", instType="SWAP", state="filled", uly=f"{sym}-USDT"):
        for x in blk.get("details", []):
            if now - float(x["ts"]) <= 3600_000:
                if x.get("posSide") == "long": lg += float(x["sz"])
                elif x.get("posSide") == "short": sh += float(x["sz"])
    d["liq"] = (lg, sh)
    bk = okx("market/books", instId=inst, sz=5); d["spread"] = None
    if bk and bk[0].get("asks") and bk[0].get("bids"):
        a0, b0 = float(bk[0]["asks"][0][0]), float(bk[0]["bids"][0][0]); d["spread"] = (a0 - b0) / ((a0 + b0) / 2) * 1e4
    oc = okx("market/history-candles", instId=inst, bar="1Dutc", after=str(OCT10_TS + 86400000), limit=2)
    d["oct10"] = next((float(r[3]) for r in oc if int(r[0]) == OCT10_TS), None)
    return d

# ---------------- indikator ----------------
def ema(x, n):
    k = 2 / (n + 1); o = np.empty_like(x); o[0] = x[0]
    for i in range(1, len(x)): o[i] = x[i] * k + o[i - 1] * (1 - k)
    return o

def rsi(c, n=8):
    o = np.full(len(c), np.nan)
    if len(c) < n + 2: return o
    d = np.diff(c); up = np.where(d > 0, d, 0.); dn = np.where(d < 0, -d, 0.)
    au, ad = up[:n].mean(), dn[:n].mean(); o[n] = 100 if ad == 0 else 100 - 100 / (1 + au / ad)
    for i in range(n, len(d)):
        au = (au * (n - 1) + up[i]) / n; ad = (ad * (n - 1) + dn[i]) / n
        o[i + 1] = 100 if ad == 0 else 100 - 100 / (1 + au / ad)
    return o

def sma(x, m):
    o = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - m + 1):i + 1]
        if len(w) == m and not np.isnan(w).any(): o[i] = w.mean()
    return o

def stoch(h, l, c, k=5, d=3, s=3):
    raw = np.full(len(c), np.nan)
    for i in range(k - 1, len(c)):
        hh, ll = h[i - k + 1:i + 1].max(), l[i - k + 1:i + 1].min(); raw[i] = 50 if hh == ll else 100 * (c[i] - ll) / (hh - ll)
    K = sma(raw, s); return K, sma(K, d)

def atr(h, l, c, n=14):
    o = np.full(len(c), np.nan)
    if len(c) < n + 2: return o
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1]))); a = tr[:n].mean(); o[n] = a
    for i in range(n, len(tr)): a = (a * (n - 1) + tr[i]) / n; o[i + 1] = a
    return o

def last(x):
    x = x[~np.isnan(x)]; return float(x[-1]) if len(x) else np.nan

def swings(h, l, n=2):
    hi, lo = [], []
    for i in range(n, len(h) - n):
        if h[i] == h[i - n:i + n + 1].max(): hi.append((i, h[i]))
        if l[i] == l[i - n:i + n + 1].min(): lo.append((i, l[i]))
    return hi, lo

def ema_state(a):
    """EMA21 & EMA50: 1 = bullish, -1 = bearish, 0 = sideways."""
    c = a[:, 4]
    if len(c) < 60: return 0
    e21, e50 = ema(c, 21), ema(c, 50); at = last(atr(a[:, 2], a[:, 3], c)); gap = abs(e21[-1] - e50[-1])
    if c[-1] > e21[-1] > e50[-1] and e21[-1] > e21[-6] and gap > .25 * at: return 1
    if c[-1] < e21[-1] < e50[-1] and e21[-1] < e21[-6] and gap > .25 * at: return -1
    return 0

def structure(a):
    hi, lo = swings(a[:, 2], a[:, 3], 2)
    if len(hi) < 2 or len(lo) < 2: return 0
    if hi[-1][1] > hi[-2][1] and lo[-1][1] > lo[-2][1]: return 1
    if hi[-1][1] < hi[-2][1] and lo[-1][1] < lo[-2][1]: return -1
    return 0

def wick(a):
    o, h, l, c = a[-1, 1], a[-1, 2], a[-1, 3], a[-1, 4]; rg = h - l
    if rg <= 0: return 0
    if (min(o, c) - l) / rg >= .5 and c >= l + .6 * rg: return 1
    if (h - max(o, c)) / rg >= .5 and c <= h - .6 * rg: return -1
    return 0

def fvg(a, look=60):
    h, l = a[:, 2], a[:, 3]; n = len(h); best = None
    for i in range(max(2, n - look), n):
        if l[i] > h[i - 2] and all(l[j] >= l[i] for j in range(i + 1, n)): best = ("bull", h[i - 2], l[i])
        if h[i] < l[i - 2] and all(h[j] <= h[i] for j in range(i + 1, n)): best = ("bear", h[i], l[i - 2])
    return best

def touches(a, level, at):
    h, l = a[-120:, 2], a[-120:, 3]; cnt = 0; prev = False
    for x, y in zip(h, l):
        t = abs(x - level) <= .25 * at or abs(y - level) <= .25 * at
        if t and not prev: cnt += 1
        prev = t
    return cnt

# ---------------- metode Xander: zona entry M15 ----------------
def xander(a15, a1, d):
    """Impuls M15 searah bias -> fib 0.5-0.618 -> POI (order block, liquidity, S/R 1H, FVG) -> zona valid."""
    h, l, c, o = a15[:, 2], a15[:, 3], a15[:, 4], a15[:, 1]; n = len(c); at = last(atr(h, l, c))
    if n < 80 or np.isnan(at): return None
    hi, lo = swings(h, l, 3); hi = [(i, x) for i, x in hi if i >= n - 120]; lo = [(i, x) for i, x in lo if i >= n - 120]
    if not hi or not lo: return None
    if d > 0:
        ih, H = max(hi, key=lambda t: t[1]); cand = [(i, x) for i, x in lo if i < ih]
        if not cand: return None
        iL, L = min(cand, key=lambda t: t[1]); i0, i1 = iL, ih
        if c[-1] >= H or c[-1] <= L: return None
        z_hi, z_lo = H - .5 * (H - L), H - .618 * (H - L)
    else:
        il, L = min(lo, key=lambda t: t[1]); cand = [(i, x) for i, x in hi if i < il]
        if not cand: return None
        iH, H = max(cand, key=lambda t: t[1]); i0, i1 = iH, il
        if c[-1] <= L or c[-1] >= H: return None
        z_lo, z_hi = L + .5 * (H - L), L + .618 * (H - L)
    R = H - L
    if R < 2.5 * at: return None
    poi = []
    for i in range(max(i0, 1), i1):                       # order block di dalam impuls
        if d > 0 and c[i] < o[i] and c[i + 1] > o[i + 1] and c[i + 1] > h[i] and (c[i + 1] - o[i + 1]) >= .8 * at:
            hit = l[i] <= z_hi + .1 * at and h[i] >= z_lo - .1 * at
        elif d < 0 and c[i] > o[i] and c[i + 1] < o[i + 1] and c[i + 1] < l[i] and (o[i + 1] - c[i + 1]) >= .8 * at:
            hit = l[i] <= z_hi + .1 * at and h[i] >= z_lo - .1 * at
        else: hit = False
        if hit: poi.append("order block"); break
    hi2, lo2 = swings(h, l, 2)                           # liquidity: swing lama (sebelum puncak impuls) di dalam zona
    if any(z_lo - .15 * at <= x <= z_hi + .15 * at for i, x in (lo2 if d > 0 else hi2) if i < i1): poi.append("liquidity")
    if a1 is not None and len(a1) > 20:                  # support/resistance 1H yang terbentuk sebelum puncak impuls
        hh, ll = swings(a1[:, 2], a1[:, 3], 2); t_top = a15[i1, 0]
        if any(z_lo - .15 * at <= x <= z_hi + .15 * at for i, x in hh + ll if a1[i, 0] < t_top): poi.append("S/R 1H")
    g = fvg(a15)
    if g and g[2] >= z_lo - .1 * at and g[1] <= z_hi + .1 * at: poi.append("FVG")
    e21 = float(ema(c, 21)[-1]); px = c[-1]
    near = (z_lo - .25 * at <= e21 <= z_hi + .25 * at) or abs(px - e21) <= .5 * at
    band_lo, band_hi = z_lo - .2 * at, z_hi + .2 * at
    if band_lo <= px <= band_hi: state = "DI ZONA"
    elif d > 0 and px > band_hi: state = "MENDEKAT" if px - band_hi <= at else "MENUNGGU"
    elif d < 0 and px < band_lo: state = "MENDEKAT" if band_lo - px <= at else "MENUNGGU"
    else: return None                                    # zona sudah tertembus
    touched = any(l[-k] <= z_hi + .1 * at and h[-k] >= z_lo - .1 * at for k in range(1, 5))
    conf = bool(touched and (wick(a15) == d or (d > 0 and c[-1] > o[-1] and c[-1] > h[-2]) or (d < 0 and c[-1] < o[-1] and c[-1] < l[-2])))
    mid = (z_lo + z_hi) / 2; stop = z_lo - .6 * at if d > 0 else z_hi + .6 * at
    t1 = H if d > 0 else L; t2 = H + .272 * R if d > 0 else L - .272 * R
    rr = abs(t1 - mid) / max(abs(mid - stop), 1e-12)
    dist = ((px - z_hi) / px * 100) if d > 0 else ((z_lo - px) / px * 100)
    return {"zlo": z_lo, "zhi": z_hi, "H": H, "L": L, "poi": poi, "state": state, "conf": conf, "e21": e21, "near_e21": bool(near),
            "entry": mid, "stop": stop, "t1": t1, "t2": t2, "rr": rr, "dist": dist, "at15": at, "touch": touches(a15, mid, at)}

# ---------------- statistik ----------------
def zscore(x, v):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 20 or x.std() == 0: return 0.0
    return float((v - x.mean()) / x.std())

def perf(ret, ann=365):
    r = np.asarray(ret, float); r = r[~np.isnan(r)]
    if len(r) < 30: return {}
    mu, sd = r.mean(), r.std(ddof=1); dn = r[r < 0]; ds = math.sqrt((dn ** 2).mean()) if len(dn) else 0
    if sd < 1e-5: return {}
    eq = np.cumprod(1 + r)
    return {"sharpe": mu / sd * math.sqrt(ann) if sd else np.nan, "sortino": mu / ds * math.sqrt(ann) if ds else np.nan,
            "vol": sd * math.sqrt(ann), "maxdd": float((eq / np.maximum.accumulate(eq) - 1).min())}

def beta_corr(r, rb):
    n = min(len(r), len(rb))
    if n < 30: return np.nan, np.nan
    r, rb = r[-n:], rb[-n:]; v = np.var(rb, ddof=1)
    return (np.cov(r, rb)[0, 1] / v if v > 0 else np.nan), np.corrcoef(r, rb)[0, 1]

def tstat(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0

def summarize(ev, name):
    if len(ev) < 30: return f"{name}: baru {len(ev)} kejadian, terlalu sedikit untuk disimpulkan."
    net = np.array([e[1] for e in sorted(ev)]); k = sum(e[2] for e in ev); t = tstat(net)
    v = "belum terbukti berguna" if abs(t) < 2 else "terbukti menguntungkan" if t > 0 else "cenderung rugi"
    return f"{name}: dicoba {len(ev)}x, benar {100*k/len(ev):.0f}%, setelah biaya rata-rata {100*net.mean():+.3f}% per trade -> {v}."

def pooled_h4(all4h):
    ev = []
    for a in all4h:
        o, c = a[:, 1], a[:, 4]; col = np.sign(c - o)
        for i in range(2, len(a) - 1):
            if col[i] != 0 and col[i] == col[i - 1] == col[i - 2]:
                g = col[i] * (c[i + 1] / o[i + 1] - 1); ev.append((a[i, 0], g - 2 * FEE, g > 0))
    return summarize(ev, "3 candle H4 searah, lalu lanjut searah")

def pooled_oi(items):
    ev = []
    for c5, oi in items:
        m = {int(r[0]): r[4] for r in c5}; ts = [int(t) for t in oi[:, 0] if int(t) in m]
        if len(ts) < 60: continue
        om = {int(r[0]): r[1] for r in oi}; px = np.array([m[t] for t in ts]); ov = np.array([om[t] for t in ts])
        ch = ov[3:] / ov[:-3] - 1; z = [zscore(ch, v) for v in ch]
        for i in range(3, len(ts) - 6):
            if z[i - 3] > 1.5 and px[i] > px[i - 3]:
                g = px[i + 6] / px[i] - 1; ev.append((ts[i], g - 2 * FEE, g > 0))
    return summarize(ev, "OI melonjak + harga naik, lalu 30 menit berikutnya naik")

# ---------------- analisis per koin ----------------
def analyze(d, tk, oiusd, btc):
    C = {b: cl(d["c"][b]) for b in BARS}
    if any(len(C[b]) < 60 for b in ["1Dutc", "4H", "1H", "15m", "5m"]): return None
    px = float(tk["last"]); chg24 = px / float(tk["open24h"]) - 1
    a1d, a4, a1, a15, a5 = C["1Dutc"], C["4H"], C["1H"], C["15m"], C["5m"]
    dr = ema_state(a1d)
    r = {"sym": d["sym"], "inst": d["inst"], "px": px, "chg24": chg24, "oiusd": oiusd, "dir": dr,
         "s_d": dr, "s_4h": ema_state(a4), "s_1h": ema_state(a1), "str4h": structure(a4)}
    r["atr_pct"] = last(atr(a1[:, 2], a1[:, 3], a1[:, 4])) / px * 100
    rs = rsi(a15[:, 4], 8); K, D = stoch(a15[:, 2], a15[:, 3], a15[:, 4]); r["rsi"], r["K"], r["D"] = last(rs), last(K), last(D)
    p15 = a5[-1, 4] / a5[-4, 4] - 1; r["p15"] = p15
    oi = d["oi5"]; r["oi15"], r["oiz"] = 0.0, 0.0
    if len(oi) >= 30:
        x = oi[3:, 1] / oi[:-3, 1] - 1; r["oi15"] = float(x[-1]); r["oiz"] = zscore(x[:-1], x[-1])
    r["quad"] = ("long baru masuk" if r["oi15"] > 0 else "short covering") if p15 > 0 else ("short baru masuk" if r["oi15"] > 0 else "long menyerah")
    v = a15[:, 5]; r["volz"] = zscore(v[-97:-1], v[-1]) if len(v) > 40 else 0.0
    r["fund"] = d["fund"]; r["fz"] = zscore(d["fh"], d["fund"]) if d["fund"] is not None and len(d["fh"]) >= 20 else 0.0
    r["liq"] = d["liq"]; r["spread"] = d["spread"]
    dc = a1d[:, 4]; ret = dc[1:] / dc[:-1] - 1; r["perf"] = perf(ret[-90:], 365)
    isb = btc is not None and d["sym"] != "BTC"
    b, rho = beta_corr(ret, btc["ret"]) if isb else (1.0, 1.0)
    r["beta"], r["rho"] = b, rho; r["resid24"] = chg24 - (b if not np.isnan(b) else 0) * btc["chg24"] if isb else chg24
    r["oct10"] = d["oct10"]; r["oct10_dist"] = (px / d["oct10"] - 1) * 100 if d["oct10"] else None
    r["zone"] = xander(a15, a1, dr) if dr else None
    return r

def score(r, now):
    dr, z = r["dir"], r["zone"]; sc = {}
    sc["bias+struktur"] = 10 + (15 if r["str4h"] == dr else 0) + (5 if r["s_4h"] == dr else 0)
    sc["zona"] = min(3, len(z["poi"])) * 5 + (5 if z["near_e21"] else 0)
    sc["posisi di zona"] = 10 if z["state"] == "DI ZONA" else 5
    sc["konfirmasi"] = 10 if z["conf"] else 0
    newpos = r["p15"] * dr > 0 and r["oi15"] > 0
    lg, sh = r["liq"]; liqb = ((sh - lg) / (lg + sh)) * dr if lg + sh > 0 else 0
    sc["posisi pasar"] = (6 * min(1, max(0, r["oiz"]) / 2) if newpos else 0) + 2 * (1 - min(3, max(0, r["fz"] * dr)) / 3) + 2 * max(0, liqb)
    sc["volume"] = 5 * min(1, max(0, r["volz"]) / 2)
    sp = r["spread"]; sc["likuiditas"] = (3 if sp is not None and sp <= MAX_SPREAD_BPS else 0) + (2 if r["oiusd"] >= 100e6 else 1)
    sc["RR"] = 10 if z["rr"] >= 2 else 6 if z["rr"] >= 1.5 else 3 if z["rr"] >= 1 else 0
    tot = sum(sc.values()); g = "A+" if tot >= 80 else "A" if tot >= 70 else "B+" if tot >= 60 else "B"; why = []
    if not z["conf"]:
        why.append("menunggu konfirmasi")
        if GR[g] > 2: g = "B+"
    if r["str4h"] != dr:
        why.append("struktur H4 belum searah")
        if GR[g] > 1: g = "B"
    if sp is None or sp > MAX_SPREAD_BPS:
        why.append("spread lebar/tidak ada data")
        if GR[g] > 2: g = "B+"
    if g == "A+" and z["rr"] < 2: g = "A"
    if r["oiz"] > 3 and r["p15"] * dr > 0: why.append("OI melonjak: rawan kejar harga")
    for t in NEWS_BLACKOUT_UTC:
        try:
            tt = datetime.strptime(t, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            if abs((now - tt).total_seconds()) <= BLACKOUT_MIN * 60: g = "B"; why.append("jendela berita")
        except Exception: pass
    return tot, g, sc, why

# ---------------- papan skor (simulasi entry limit di zona) ----------------
COLS = ["ts", "sym", "dir", "grade", "score", "state", "zlo", "zhi", "entry", "stop", "t1", "rr", "oiz", "touch", "status", "R"]
def read_signals():
    if not os.path.exists(SIGNALS_CSV): return []
    with open(SIGNALS_CSV) as f: return list(csv.DictReader(f))

def write_signals(rows):
    os.makedirs("data", exist_ok=True)
    with open(SIGNALS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); [w.writerow({c: r.get(c, "") for c in COLS}) for r in rows]

def candles5(inst, t0, t1):
    out = []; after = t1 + 300000
    for _ in range(6):
        rows = okx("market/history-candles", instId=inst, bar="5m", after=str(after), limit=100)
        if not rows: break
        out += [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in rows]
        oldest = int(rows[-1][0])
        if oldest <= t0: break
        after = oldest
    return sorted([x for x in out if t0 <= x[0] <= t1])

def simulate(s):
    t0 = int(s["ts"]); d = int(s["dir"]); entry, stop, t1 = float(s["entry"]), float(s["stop"]), float(s["t1"])
    cs = candles5(f"{s['sym']}-USDT-SWAP", t0, t0 + EVAL_H * 3600_000)
    if len(cs) < 10: return None
    risk = abs(entry - stop); cost = 2 * FEE * entry / risk; filled = False
    for ts, o, h, l, c in cs:
        if not filled:
            if ts > t0 + FILL_H * 3600_000: break
            if (d > 0 and h >= t1) or (d < 0 and l <= t1): return "NOFILL", 0.0       # target tercapai sebelum entry
            if (d > 0 and l <= entry) or (d < 0 and h >= entry): filled = True
            else: continue
        if (