"""charts.py - grafik gaya terminal (latar hitam, aksen oranye)."""
import io, logging, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import engine as eng

plt.rcParams.update({"font.family": "DejaVu Sans Mono", "axes.unicode_minus": False})
log = logging.getLogger("macro")
BG, FG, OR, GR, GRID, BL, GN, RD = "#000000", "#F2F2F2", "#FF9F1C", "#8A8A8A", "#262626", "#4DA3FF", "#3DDC84", "#FF5C5C"


def frame(fig, title, sub, source, H):
    lines = textwrap.wrap(title, 52)[:2]
    y = 0.975
    fig.patch.set_facecolor(BG)
    fig.text(0.03, y, "\n".join(lines), color=FG, fontsize=12, fontweight="bold", va="top", linespacing=1.3)
    y -= 0.27 / H * len(lines) + 0.012
    fig.text(0.03, y, sub, color=OR, fontsize=8.5, va="top")
    fig.text(0.03, 0.015, f"Source: {source}", color=GR, fontsize=7, va="bottom")
    fig.text(0.97, 0.015, "MACRO WIRE", color=OR, fontsize=8, fontweight="bold", ha="right", va="bottom")
    return y - 0.4 / H


def style(ax):
    ax.set_facecolor(BG)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GR)
    ax.tick_params(colors=GR, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=BG)
    plt.close(fig)
    return buf.getvalue()


def endlab(ax, x, y, text, col):
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", color=col, fontsize=8,
                va="center", annotation_clip=False)


def months(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


# ---------- 1. Kurva imbal hasil ----------
def chart_curve(mk):
    ref = None
    for sid in eng.FRED_DAILY:
        s = mk.f(sid)
        if s is not None:
            ref = s.index[-1] if ref is None else max(ref, s.index[-1])
    if ref is None:
        return None

    def curve(days):
        out = []
        for sid in eng.FRED_DAILY:
            s = mk.f(sid)
            sub = s[s.index <= ref - pd.Timedelta(days=days)] if s is not None else []
            out.append(float(sub.iloc[-1]) if len(sub) else np.nan)
        return np.array(out)

    now, w1, m1, y1 = curve(0), curve(7), curve(30), curve(365)
    labs = list(eng.FRED_DAILY.values())
    pos = {l: i for i, l in enumerate(labs)}
    sp210 = (now[pos["10Y"]] - now[pos["2Y"]]) * 100
    sp1030 = (now[pos["30Y"]] - now[pos["10Y"]]) * 100
    H = 4.7
    fig, ax = plt.subplots(figsize=(7.2, H))
    top = frame(fig, f"US Treasury Curve: 2s10s {sp210:+.0f}bp, 10s30s {sp1030:+.0f}bp",
                f"Par yields, %, as of {ref:%d %b %Y}", "FRED (US Treasury)", H)
    fig.subplots_adjust(left=0.08, right=0.95, top=top, bottom=0.14)
    style(ax)
    x = np.arange(len(labs))
    for arr, lab, col, lw, ls in ((y1, "1Y ago", GR, 1.2, "--"), (m1, "1M ago", BL, 1.4, "-"),
                                  (w1, "1W ago", FG, 1.2, "-"), (now, "Now", OR, 2.4, "-")):
        ok = ~np.isnan(arr)
        ax.plot(x[ok], arr[ok], color=col, lw=lw, ls=ls, label=lab, marker="o" if lab == "Now" else None, ms=3.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.legend(frameon=False, labelcolor=FG, fontsize=8, loc="best")
    return png(fig)


# ---------- 2. Dekomposisi yield 10Y ----------
def chart_decomp(mk):
    nom, real, be = mk.f("DGS10"), mk.f("DFII10"), mk.f("T10YIE")
    if nom is None or real is None or be is None:
        return None
    df = pd.concat({"nom": nom, "real": real, "be": be}, axis=1).dropna().iloc[-90:]
    if len(df) < 20:
        return None
    last = df.iloc[-1]
    tp = mk.f("THREEFYTP10")
    H = 5.4 if tp is not None else 4.6
    fig = plt.figure(figsize=(7.2, H))
    top = frame(fig, f"US 10Y at {last['nom']:.2f}% = Real {last['real']:.2f}% + Breakeven {last['be']:.2f}%",
                f"Daily, %, through {df.index[-1]:%d %b %Y}", "FRED", H)
    if tp is not None:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.1], hspace=0.12, left=0.08, right=0.84, top=top, bottom=0.1)
        ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    else:
        fig.subplots_adjust(left=0.08, right=0.84, top=top, bottom=0.12)
        ax, ax2 = fig.add_subplot(111), None
    style(ax)
    ax.stackplot(df.index, df["real"].clip(lower=0), df["be"], colors=[BL, GR], alpha=0.45)
    ax.plot(df.index, df["nom"], color=OR, lw=2)
    ax.set_ylim(0, float(df["nom"].max()) * 1.12)
    endlab(ax, df.index[-1], last["nom"], f"Nominal {last['nom']:.2f}", OR)
    endlab(ax, df.index[-1], last["real"] / 2, f"Real {last['real']:.2f}", BL)
    endlab(ax, df.index[-1], last["real"] + last["be"] / 2, f"BE {last['be']:.2f}", GR)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    if ax2 is not None:
        style(ax2)
        t = tp[tp.index >= df.index[0]]
        ax2.plot(t.index, t.values, color=GN, lw=1.6)
        endlab(ax2, t.index[-1], t.iloc[-1], f"TP {t.iloc[-1]:.2f}", GN)
        ax2.text(0.01, 0.9, "Term premium 10Y (Kim-Wright)", transform=ax2.transAxes, color=GN, fontsize=7.5, va="top")
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(labelbottom=False)
    return png(fig)


# ---------- 3. USD/JPY vs selisih yield AS-Jepang ----------
def chart_usjp(mk):
    df = eng.us_jp_gap(mk)
    if df is None or len(df) < 40:
        return None
    df = df.iloc[-250:]
    fx, gap = df["fx"], df["gap"]
    H = 4.7
    fig, ax = plt.subplots(figsize=(7.2, H))
    top = frame(fig, f"USD/JPY at {fx.iloc[-1]:.2f}; US–Japan 10Y gap at {gap.iloc[-1]:.2f}pp",
                "Daily; USD/JPY (left, orange) vs US 10Y minus JGB 10Y, pp (right, blue)", "Yahoo Finance, MOF Japan", H)
    fig.subplots_adjust(left=0.1, right=0.86, top=top, bottom=0.12)
    style(ax)
    ax.plot(fx.index, fx.values, color=OR, lw=1.8)
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    for sp in ("top", "left", "bottom"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(GR)
    ax2.tick_params(colors=BL, labelsize=8)
    ax2.plot(gap.index, gap.values, color=BL, lw=1.6)
    months(ax)
    return png(fig)


# ---------- 4. Yield Treasury ----------
def chart_yields(mk):
    q30 = mk.q("US 30Y")
    if not q30:
        return None
    title = (f"US 30-Year Yield at {q30.last:.2f}%, {q30.note}" if q30.note
             else f"US 30-Year Yield at {q30.last:.2f}% ({q30.chg():+.1f}bp on the day)")
    H = 4.7
    fig, ax = plt.subplots(figsize=(7.2, H))
    top = frame(fig, title, "Treasury yields, %, daily close, last 6 months", "Yahoo Finance", H)
    fig.subplots_adjust(left=0.08, right=0.84, top=top, bottom=0.12)
    style(ax)
    lo, hi = 99, 0
    for label, col in (("US 30Y", OR), ("US 10Y", BL), ("US 5Y", GR)):
        q = mk.q(label)
        if not q:
            continue
        s = q.s.iloc[-130:]
        ax.plot(s.index, s.values, color=col, lw=1.7)
        endlab(ax, s.index[-1], s.iloc[-1], f"{label} {s.iloc[-1]:.2f}", col)
        lo, hi = min(lo, s.min()), max(hi, s.max())
    w = eng.WATCH.get("US 30Y")
    if w and lo * 0.97 <= w[0] <= hi * 1.03:
        ax.axhline(w[0], color=GR, ls="--", lw=0.9)
        ax.text(ax.get_xlim()[0], w[0], f" watch {w[0]:g}%", color=GR, fontsize=7, va="bottom")
    months(ax)
    return png(fig)


def make_all(mk):
    out = []
    for fn in (chart_curve, chart_decomp, chart_usjp, chart_yields):
        try:
            b = fn(mk)
            if b:
                out.append(b)
        except Exception:
            log.exception("chart gagal: %s", fn.__name__)
    return out
