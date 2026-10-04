"""bot.py - analis LLM, laporan Telegram, dan alur kerja.
  python bot.py --once            satu siklus (mode otomatis: :00 = FULL brief, :30 = FLASH)
  python bot.py --once --full     paksa FULL brief
  python bot.py --dry-run         cetak ke terminal + simpan preview_N.png, tidak kirim
  python bot.py                   loop tiap INTERVAL_MIN menit"""
import argparse, hashlib, html, json, logging, re, sys, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

import charts
import engine as eng
from engine import Item, env

log = logging.getLogger("macro")
esc = html.escape

TG_TOKEN, TG_CHAT = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
GEMINI_KEY = env("GEMINI_API_KEY")
GEMINI_MODEL = env("GEMINI_MODEL", "gemini-2.5-flash")
THINK = int(env("GEMINI_THINKING", "2048"))
RESEARCH = env("RESEARCH", "true").lower() == "true"       # pencarian web via Gemini (gratis)
SCENARIOS = env("INCLUDE_AI_VIEW", "true").lower() == "true"
CHARTS = env("CHARTS", "hourly").lower()                   # hourly = hanya FULL | always = juga FLASH | off
FLASH_QUIET = env("FLASH_QUIET", "true").lower() == "true"  # :30 hanya kirim jika ada yang baru
INTERVAL_MIN = int(env("INTERVAL_MIN", "30"))
LLM_ERR = ""

# ============ PROMPT ============
SYSTEM = """Kamu analis makro senior di meja berita obligasi dan FX (gaya Bloomberg). Pembaca: trader
profesional yang membuat keputusan sendiri. Tugasmu menyajikan FAKTA yang terverifikasi, lalu analisis
berlapis yang jujur soal ketidakpastian. Kamu TIDAK memberi rekomendasi beli/jual/ukuran posisi.

SUMBER: hanya blok input: SINYAL {S#} (dihitung kode, otoritatif), LEVEL PASAR, DATA MAKRO RESMI,
BERITA [n]. Pengetahuanmu sendiri hanya untuk memahami istilah, BUKAN untuk angka, tanggal, nama, atau
penyebab. Hierarki kebenaran: SINYAL & data resmi > berita tier 1 (resmi) > media tier 2 > item WEB
(hasil pencarian, perlakukan sebagai laporan sekunder). Jika sumber berbeda, tampilkan keduanya.

TIGA LAPIS (selalu berlabel jelas):
- FAKTA: ada di input. Berita diberi [n]; sinyal diberi {S#}.
- INFERENSI: kesimpulanmu dari fakta. Sebut dasarnya, pakai "konsisten dengan"/"kemungkinan".
- HIPOTESIS: belum terbukti; hanya di scenarios.

GAYA WIRE: kalimat pertama berisi angka dan perubahan ("Yield Treasury 30 tahun naik 2.7bp ke 5.63%,
tertinggi sejak Jul 2007"). Aktif, pendek, tanpa adjektiva emosional (anjlok/meroket/panik/mengejutkan).
Angka persis dengan satuan dan periodenya. Penyebab hanya sebagai fakta bila sumber menyatakannya
("Menurut Reuters, ..."); selain itu tulis sebagai inferensi.

CARI YANG TIDAK KENTARA: (a) data bergerak tapi berita diam, atau sebaliknya; (b) hubungan antar-aset
yang berubah; (c) dekomposisi penyebab (real yield vs breakeven vs term premium); (d) efek lanjutan dan
siapa yang terdampak; (e) apa yang "dihargai" pasar menurut kurva; (f) kontradiksi narasi vs data.
Tiap pola WAJIB punya penjelasan alternatif. Bukti kurang = daftar kosong. Satu pola kuat lebih baik
daripada empat lemah.

LARANGAN: mengarang angka/tanggal/kutipan; menyebut peristiwa yang tidak ada di input; prediksi
bertanggal; probabilitas angka; menyalin headline sensasional; rekomendasi transaksi.
Gabungkan berita duplikat. Buang berita yang tidak relevan dengan makro AS/Jepang/global.
Bahasa Indonesia; istilah pasar tetap Inggris.

Keluaran JSON valid saja:
{"headline":"1 kalimat inti situasi (faktual, ada angka)",
 "drivers":["FAKTA: ... [n]/{S#}" atau "INFERENSI: ... dasar: ..."  (2-4 butir penggerak utama)],
 "since_last":"perubahan vs LAPORAN SEBELUMNYA (1-2 kalimat) atau kosong",
 "stories":[{"tag":"RATES|CENTRAL BANK|INFLATION|DATA|FX|FISCAL|OIL|OTHER","headline":"maks 90 karakter",
             "lead":"maks 2 kalimat, angka dulu [n]","details":["maks 3 fakta [n]"],"src":[n]}],
 "patterns":[{"pattern":"pola tersembunyi (INFERENSI)","evidence":"bukti: angka {S#} / [n]",
              "alt":"penjelasan alternatif","falsifier":"apa yang akan membantah","confidence":"rendah|sedang|tinggi"}],
 "contradictions":["narasi vs data tidak sejalan, dengan angka"],
 "scenarios":[{"name":"...","if":"pemicu","then":"implikasi lintas aset (hipotesis)","signpost":"penanda konfirmasi/bantahan"}],
 "watch":["event/data yang disebut di input"],
 "data_gaps":["hal penting yang TIDAK bisa disimpulkan dari data ini"],
 "key_questions":["2-4 pertanyaan penilaian untuk pembaca"]}

CONTOH GAYA (bukan data):
 headline: "Yield Treasury 30 tahun naik ke 5.63%, tertinggi sejak Jul 2007"
 pattern: "Kenaikan yield didorong real yield, bukan inflasi" | evidence: "{S2}: real yield +21bp, breakeven +5bp (5 hari)"
          | alt: "Pasokan obligasi (lelang) bisa menaikkan term premium tanpa mengubah real yield terukur" | falsifier: "breakeven naik > real yield dalam 5 hari ke depan" """

FULL_RULES = "Batas: stories maks 5, patterns maks 4, contradictions maks 3, scenarios maks 3 (tanpa probabilitas angka), watch maks 5, data_gaps maks 3."
FLASH_RULES = ("MODE FLASH (update singkat): fokus HANYA pada hal BARU. stories maks 3, patterns maks 2, "
               "contradictions maks 1, scenarios kosong, watch maks 3, data_gaps kosong, key_questions maks 2.")


def gemini(body, models):
    """Panggil Gemini dengan rantai fallback model. Mengembalikan JSON respons atau None."""
    global LLM_ERR
    for model in dict.fromkeys(models):
        b = json.loads(json.dumps(body))
        if "2.5" not in model:
            b.get("generationConfig", {}).pop("thinkingConfig", None)
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              headers={"x-goog-api-key": GEMINI_KEY}, json=b, timeout=150)
            if r.status_code != 200:
                LLM_ERR = f"{model} HTTP {r.status_code}"
                log.error("Gemini %s: %s", LLM_ERR, r.text[:300])
                continue
            return r.json()
        except Exception as e:
            LLM_ERR = f"{model}: {type(e).__name__}"
            log.error("Gemini gagal (%s): %s", model, e)
    return None


def parts_text(resp):
    parts = resp["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


# ============ RISET WEB (Gemini + Google Search, gratis) ============
def research(sigs, now):
    """Cari laporan kredibel yang menjelaskan sinyal; tiap klaim dikembalikan sebagai Item bersumber."""
    if not (GEMINI_KEY and RESEARCH and sigs):
        return []
    top = "\n".join(f"- {s.text}" for s in sigs[:6])
    prompt = (f"Sekarang {now:%Y-%m-%d %H:%M} ({eng.TZ}). Cari laporan TERBARU (12 jam terakhir) dari sumber kredibel "
              "(Reuters, Bloomberg, WSJ, FT, CNBC, Nikkei, The Fed, BoJ, Kemenkeu Jepang, US Treasury) yang "
              f"menjelaskan atau terkait pergerakan pasar obligasi/FX/inflasi AS dan Jepang berikut:\n{top}\n\n"
              "Tulis 6-10 poin fakta ringkas. Tiap poin: apa yang dilaporkan, angka persis, siapa sumbernya. "
              "Jangan berspekulasi atau menambah pendapatmu. Jika tidak ada laporan kredibel, tulis 'tidak ditemukan'.")
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 3000,
                                 "thinkingConfig": {"thinkingBudget": 512}}}
    resp = gemini(body, ["gemini-2.5-flash", "gemini-2.0-flash"])
    if not resp:
        return []
    try:
        gm = resp["candidates"][0].get("groundingMetadata", {})
        chunks = [c.get("web", {}) for c in gm.get("groundingChunks", [])]
        items, seen = [], set()
        for sp in gm.get("groundingSupports", []):
            seg = (sp.get("segment", {}).get("text") or "").strip()
            idx = sp.get("groundingChunkIndices", [])
            if len(seg) < 40 or not idx or idx[0] >= len(chunks):
                continue
            w = chunks[idx[0]]
            name, uri = w.get("title", ""), w.get("uri", "")
            if not uri or not eng.good_source(name):
                continue
            uid = hashlib.sha1(seg.lower().encode()).hexdigest()[:16]
            if uid in seen:
                continue
            seen.add(uid)
            items.append(Item(uid, seg[:110], name, 2, uri, datetime.now(timezone.utc), seg, score=6, web=True))
        log.info("riset web: %d klaim bersumber", len(items))
        return items[:8]
    except Exception:
        log.exception("riset web gagal parse")
        return []


# ============ ANALISIS ============
def build_user(items, mk, sigs, last, flash):
    def news_line(n, i):
        body = (i.body or i.summary)[:900]
        tag = "WEB" if i.web else f"tier {i.tier}"
        return f"[{n}] ({tag} | {i.source} | {i.pub:%Y-%m-%d %H:%MZ}) {i.title}" + (f"\n    {body}" if body and body != i.title else "")
    news = "\n".join(news_line(n, i) for n, i in enumerate(items, 1)) or "(tidak ada berita baru)"
    sig_txt = "\n".join(f"{{{s.id}}} {'[BARU] ' if s.new else ''}{s.text}" for s in sigs[:25]) or "(tidak ada)"
    lv = []
    for q in mk.quotes.values():
        c1 = q.chg(1)
        c21 = q.chg(21)
        lv.append(f"{q.label}: {q.fmt()} (1D {c1:+.1f}{q.unit}" + (f", 1M {c21:+.1f}{q.unit}" if c21 is not None else "")
                  + f", data {q.asof:%d %b})" + (f" [{q.note}]" if q.note else ""))
    macro = "\n".join(f"{k}: {v[0]:.2f} (periode {v[2]})" for k, v in mk.macro.items()) or "(tidak tersedia)"
    prev = ""
    if last and last.get("headline"):
        prev = f"LAPORAN SEBELUMNYA:\nHeadline: {last['headline']}\nDipantau: {'; '.join(last.get('watch', []))}\n\n"
    return (f"WAKTU: {datetime.now(ZoneInfo(eng.TZ)):%A %d %b %Y %H:%M} {eng.TZ}\n\n"
            f"SINYAL (dihitung kode, otoritatif; [BARU] = baru muncul):\n{sig_txt}\n\n"
            f"LEVEL PASAR:\n" + "\n".join(lv) + f"\n\nDATA MAKRO RESMI:\n{macro}\n\n{prev}"
            f"BERITA:\n{news}\n\n" + (FLASH_RULES if flash else FULL_RULES)
            + ("\nKosongkan scenarios." if (flash or not SCENARIOS) else "") + "\nHasilkan JSON sesuai skema.")


def sanitize(a):
    """Paksa tipe data agar keluaran LLM yang melenceng tidak merusak laporan."""
    def S(x):
        return x if isinstance(x, str) else ("" if x is None else str(x))

    def L(x):
        return x if isinstance(x, list) else ([] if x in (None, "") else [x])

    def D(x):
        return [d for d in L(x) if isinstance(d, dict)]
    src = lambda s: [int(n) for n in L(s.get("src")) if str(n).isdigit()]
    return dict(
        headline=S(a.get("headline")), since_last=S(a.get("since_last")),
        drivers=[S(x) for x in L(a.get("drivers"))][:4],
        stories=[dict(tag=S(s.get("tag")), headline=S(s.get("headline")), lead=S(s.get("lead")),
                      details=[S(x) for x in L(s.get("details"))][:3], src=src(s)) for s in D(a.get("stories"))][:5],
        patterns=[{k: S(p.get(k)) for k in ("pattern", "evidence", "alt", "falsifier", "confidence")} for p in D(a.get("patterns"))][:4],
        contradictions=[S(x) for x in L(a.get("contradictions"))][:3],
        scenarios=[{k: S(s.get(k)) for k in ("name", "if", "then", "signpost")} for s in D(a.get("scenarios"))][:3],
        watch=[S(x) for x in L(a.get("watch"))][:5], data_gaps=[S(x) for x in L(a.get("data_gaps"))][:3],
        key_questions=[S(x) for x in L(a.get("key_questions"))][:4])


def analyze(items, mk, sigs, last, flash):
    global LLM_ERR
    if not GEMINI_KEY:
        LLM_ERR = "GEMINI_API_KEY belum diisi"
        return None, ""
    user = build_user(items, mk, sigs, last, flash)
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json",
                                 "maxOutputTokens": 9000, "thinkingConfig": {"thinkingBudget": THINK}}}
    for model in dict.fromkeys([GEMINI_MODEL, "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]):
        resp = gemini(body, [model])
        if not resp:
            continue
        try:
            txt = parts_text(resp)
            return sanitize(json.loads(txt[txt.index("{"): txt.rindex("}") + 1])), user
        except Exception as e:
            LLM_ERR = f"{model}: JSON tidak valid"
            log.error("parse JSON gagal (%s): %s", model, e)
    return None, user


def rule_based(items):
    stories = []
    for n, i in enumerate(items[:6], 1):
        tag = next((t for t, p in eng.TOPICS if re.search(p, i.title.lower())), "OTHER")
        stories.append(dict(tag=tag, headline=i.title, lead=(i.body or i.summary)[:260], details=[], src=[n]))
    return sanitize(dict(stories=stories))


# ---------- audit angka: tandai angka yang tidak ada di data input ----------
def numbers(text):
    out = set()
    for m in re.finditer(r"\d[\d,]*\.?\d*", text):
        t = m.group().rstrip(".").replace(",", "")
        try:
            out.add((round(abs(float(t)), 3), "." in t))
        except ValueError:
            pass
    return out


def iter_strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, list):
        for x in o:
            yield from iter_strings(x)
    elif isinstance(o, dict):
        for v in o.values():
            yield from iter_strings(v)


def audit(ana, corpus):
    have = {v for v, _ in numbers(corpus)}
    have |= {round(v, 1) for v in have} | {round(v, 2) for v in have} | {float(round(v)) for v in have}
    bad = set()
    for t in iter_strings(ana):
        t = re.sub(r"\[[\d,\s]+\]|\{S\d+\}|\(S\d+\)", "", t)
        for v, dec in numbers(t):
            if v in have or (not dec and (v <= 31 or 1990 <= v <= 2100)):
                continue
            bad.add(v)
    return sorted(bad)[:8]


def renumber(ana, items):
    """Tampilkan hanya sumber yang dikutip, dengan nomor berurutan."""
    used = set()
    pat = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
    for t in iter_strings({k: v for k, v in ana.items()}):
        for g in pat.findall(t):
            used.update(int(x) for x in re.split(r"\s*,\s*", g))
    for s in ana.get("stories", []):
        used.update(s.get("src", []))
    order = sorted(n for n in used if 1 <= n <= len(items))
    m = {old: i for i, old in enumerate(order, 1)}

    def sub(mo):
        nums = [m[int(x)] for x in re.split(r"\s*,\s*", mo.group(1)) if int(x) in m]
        return "[" + ",".join(map(str, nums)) + "]" if nums else ""

    def fix(o, key=None):
        if isinstance(o, str):
            return pat.sub(sub, o)
        if isinstance(o, list):
            return [m[n] for n in o if n in m] if key == "src" else [fix(x) for x in o]
        if isinstance(o, dict):
            return {k: fix(v, k) for k, v in o.items()}
        return o
    return fix(ana), [items[n - 1] for n in order]


# ============ RENDER (Telegram HTML) ============
def cites(text, items):
    def sub(m):
        out = [f'<a href="{esc(items[int(n) - 1].link)}">{n}</a>'
               for n in re.split(r"\s*,\s*", m.group(1)) if n.isdigit() and 1 <= int(n) <= len(items)]
        return "[" + ",".join(out) + "]" if out else ""
    t = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", sub, esc(str(text)))
    return re.sub(r"\{(S\d+)\}", r"(\1)", t)


def story_html(s, items):
    src = [n for n in s.get("src", []) if 1 <= n <= len(items)]
    names = ", ".join(dict.fromkeys(items[n - 1].source for n in src))
    when = max((items[n - 1].pub for n in src), default=None)
    meta = " · ".join(x for x in (s.get("tag", ""), names, f"{when:%H:%MZ}" if when else "") if x)
    out = [f"<b>{esc(s.get('headline', ''))}</b>", f"<i>{esc(meta)}</i>"]
    if s.get("lead"):
        out.append(f"(MACRO WIRE) -- {cites(s['lead'], items)}")
    out += [f"• {cites(b, items)}" for b in s.get("details", [])]
    return "\n".join(out)


def build_text(mk, sigs, ana, items, now, flash, bad_nums, header=True):
    a, P = ana or {}, []
    tag = "⚡ FLASH" if flash else "🟧 MACRO WIRE"
    if header:
        us = mk.q("US 10Y") if mk else None
        asof = f" · data UST per {us.asof:%d %b}" if us else ""
        P.append(f"<b>{tag}</b> · {now:%d %b %Y %H:%M} {now.tzname()}{esc(asof)}\n━━━━━━━━━━━━━━━━━━")
    if a.get("headline"):
        t = f"<b>BIG PICTURE</b>\n{cites(a['headline'], items)}"
        if a.get("drivers"):
            t += "\n" + "\n".join(f"• {cites(d, items)}" for d in a["drivers"])
        P.append(t)
    if a.get("since_last"):
        P.append(f"<b>SEJAK LAPORAN TERAKHIR</b>\n{cites(a['since_last'], items)}")
    if a.get("stories"):
        P.append("<b>TOP STORIES</b>")
        P += [story_html(s, items) for s in a["stories"]]
    shown = [s for s in sigs if s.new][:10] if flash else sigs[:12]
    if shown:
        lines = "\n".join(f"{s.id} · {'🆕 ' if (s.new and not flash) else ''}{esc(s.text)}" for s in shown)
        P.append(f"<b>SIGNALS</b> <i>(dihitung dari data, bukan opini)</i>\n<blockquote expandable>{lines}</blockquote>")
    if a.get("patterns"):
        rows = []
        for n, p in enumerate(a["patterns"], 1):
            rows.append(f"<b>{n}. {cites(p['pattern'], items)}</b>\nBukti: {cites(p['evidence'], items)}\n"
                        f"Alternatif: {cites(p['alt'], items)}\n"
                        + (f"Dibantah jika: {cites(p['falsifier'], items)}\n" if p.get("falsifier") else "")
                        + f"<i>Keyakinan: {esc(p.get('confidence') or '-')}</i>")
        P.append("<b>POLA &amp; HAL YANG TIDAK TERLIHAT</b> <i>(inferensi, bukan fakta)</i>\n\n" + "\n\n".join(rows))
    if a.get("contradictions"):
        P.append("<b>KONTRADIKSI NARASI VS DATA</b>\n" + "\n".join(f"⚠ {cites(c, items)}" for c in a["contradictions"]))
    if a.get("scenarios") and SCENARIOS and not flash:
        rows = [f"<b>{esc(s['name'])}</b>\nJika: {cites(s['if'], items)}\nMaka: {cites(s['then'], items)}\n"
                f"Penanda: {cites(s['signpost'], items)}" for s in a["scenarios"]]
        P.append("<b>SKENARIO</b> <i>(hipotesis, bukan prediksi; keputusan di kamu)</i>\n\n" + "\n\n".join(rows))
    if a.get("watch"):
        P.append("<b>PANTAU</b>\n" + "\n".join("• " + cites(w, items) for w in a["watch"]))
    if a.get("data_gaps"):
        P.append("<b>YANG TIDAK BISA DISIMPULKAN</b>\n" + "\n".join("• " + cites(g, items) for g in a["data_gaps"]))
    if a.get("key_questions"):
        P.append("<b>UNTUK PENILAIANMU</b>\n" + "\n".join(f"? {cites(q, items)}" for q in a["key_questions"]))
    if not a and not shown:
        P.append("Tidak ada berita baru maupun sinyal data yang signifikan sejak update terakhir.")
    if items:
        src = "\n".join(f'{n}. <a href="{esc(i.link)}">{esc(i.source)}</a> · {i.pub:%H:%MZ}'
                        f'{" ★" if i.tier == 1 else ""}{" · web" if i.web else ""}' for n, i in enumerate(items, 1))
        P.append(f"<b>SOURCES</b> <i>(★ resmi)</i>\n<blockquote expandable>{src}</blockquote>")
    foot = "Analisis otomatis dari headline/RSS, data pasar, dan pencarian web; verifikasi ke sumber. Bukan saran investasi."
    if bad_nums:
        foot += " ⚠ Angka di luar data input (periksa manual): " + ", ".join(f"{v:g}" for v in bad_nums) + "."
    if LLM_ERR and (a.get("stories") or sigs):
        foot += f" Mode tanpa AI: {LLM_ERR}."
    P.append(f"<i>{esc(foot)}</i>")
    return split("\n\n".join(P))


def split(text, limit=3800):
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) + 2 > limit:
            out.append(cur)
            cur = ""
        cur += ("\n\n" if cur else "") + para
    return out + ([cur] if cur else [])


# ============ TELEGRAM ============
def tg(method, **kw):
    r = None
    for _ in range(3):
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/{method}", timeout=90, **kw)
        if r.status_code == 429:
            try:
                time.sleep(r.json().get("parameters", {}).get("retry_after", 3) + 1)
            except Exception:
                time.sleep(4)
            continue
        break
    return r


def send_text(msgs):
    for m in msgs:
        r = tg("sendMessage", json=dict(chat_id=TG_CHAT, text=m, parse_mode="HTML", disable_web_page_preview=True))
        if r.status_code == 400 and "parse" in r.text.lower():       # fallback: teks polos
            plain = html.unescape(re.sub(r"<[^>]+>", "", m))[:4000]
            r = tg("sendMessage", json=dict(chat_id=TG_CHAT, text=plain, disable_web_page_preview=True))
        if not r.ok:
            log.error("telegram gagal: %s %s", r.status_code, r.text[:200])
            return False
        time.sleep(0.4)
    return True


def send_album(imgs, caption):
    try:
        if len(imgs) == 1:
            r = tg("sendPhoto", data=dict(chat_id=TG_CHAT, caption=caption, parse_mode="HTML"),
                   files={"photo": ("c.png", imgs[0], "image/png")})
        else:
            media, files = [], {}
            for i, b in enumerate(imgs[:10]):
                m = {"type": "photo", "media": f"attach://p{i}"}
                if i == 0:
                    m.update(caption=caption, parse_mode="HTML")
                media.append(m)
                files[f"p{i}"] = (f"p{i}.png", b, "image/png")
            r = tg("sendMediaGroup", data=dict(chat_id=TG_CHAT, media=json.dumps(media)), files=files)
        if not r.ok:
            log.error("album gagal: %s %s", r.status_code, r.text[:200])
        return r.ok
    except Exception:
        log.exception("album gagal")
        return False


# ============ ALUR KERJA ============
def mark_new_signals(sigs):
    """Sinyal 'baru' = kunci belum terlihat 6 jam terakhir. Status disimpan antar-run."""
    state, now = eng.kv_get("sigstate", {}), time.time()
    for s in sigs:
        s.new = (s.kind == "since") or (now - state.get(s.key, 0) > 6 * 3600)
    return {**{k: v for k, v in state.items() if now - v < 48 * 3600}, **{s.key: now for s in sigs}}


def run_once(dry=False, force=None):
    global LLM_ERR
    LLM_ERR = ""
    now = datetime.now(ZoneInfo(eng.TZ))
    flash = (force == "flash") or (force is None and not dry and now.minute >= 30)

    mk = eng.load_market()
    if not mk.quotes:
        log.error("data pasar kosong, siklus dilewati")
        return
    all_news = eng.collect()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=eng.FRESH_H)
    fresh = eng.unseen([i for i in all_news if i.pub >= cutoff])
    items = fresh[:eng.MAX_ITEMS]
    last = eng.kv_get("last_report")
    sigs = eng.build_signals(mk, all_news, last)
    new_state = mark_new_signals(sigs)
    new_sigs = [s for s in sigs if s.new and s.kind != "since"]

    idle = not items and not new_sigs
    if idle and not dry and ((flash and FLASH_QUIET) or now.weekday() >= 5):
        log.info("tidak ada yang baru (flash/akhir pekan), tidak kirim")
        return

    eng.enrich(items)
    strong = any(s.w >= 7 for s in new_sigs)
    web = research(sigs if not flash else new_sigs, now) if (not flash or strong or items) else []
    pool = items + web
    ana, user = analyze(pool, mk, sigs if not flash else (new_sigs or sigs[:3]), last, flash) if (pool or sigs) else (None, "")
    ana = ana or (rule_based(pool) if pool else {})
    bad = audit(ana, user) if user else []
    ana, used = renumber(ana, pool)

    want = CHARTS != "off" and (not flash or CHARTS == "always")
    imgs = charts.make_all(mk) if want else []
    cap = f"<b>{'⚡ FLASH' if flash else '🟧 MACRO WIRE'}</b> · {now:%d %b %Y %H:%M} {now.tzname()}"

    if dry:
        for i, b in enumerate(imgs, 1):
            open(f"preview_{i}.png", "wb").write(b)
        print("\n\n----- BREAK -----\n\n".join(build_text(mk, sigs, ana, used, now, flash, bad, not imgs)))
        return
    album_ok = send_album(imgs, cap) if imgs else False
    if send_text(build_text(mk, sigs, ana, used, now, flash, bad, not album_ok)):
        eng.mark_seen(fresh)
        eng.kv_set("sigstate", new_state)
        eng.kv_set("last_report", dict(ts=time.time(), headline=re.sub(r"\{S\d+\}|\[[\d,\s]+\]", "", ana.get("headline", "")).strip(),
                                       watch=[re.sub(r"\{S\d+\}|\[[\d,\s]+\]", "", w).strip() for w in ana.get("watch", [])],
                                       vals={q.label: q.last for q in mk.quotes.values()}))
        log.info("terkirim [%s]: %d berita, %d web, %d sinyal, %d gambar", "flash" if flash else "full",
                 len(items), len(web), len(sigs), len(imgs))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--flash", action="store_true")
    a = ap.parse_args()
    force = "full" if a.full else "flash" if a.flash else None
    if not a.dry_run and not (TG_TOKEN and TG_CHAT):
        sys.exit("Isi TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID dulu")
    if a.once or a.dry_run:
        return run_once(a.dry_run, force)
    while True:
        try:
            run_once(False, force)
        except Exception:
            log.exception("siklus gagal")
        time.sleep(INTERVAL_MIN * 60 - time.time() % (INTERVAL_MIN * 60))


if __name__ == "__main__":
    main()
