"""그림 s2n-recheck.png: 11-11 재측정(1번 깃 ξ0.3; S7 → 걸이 다시 → S8) 대 기존 무리 — 모드별 주파수 차, 평균 FRF,
타격별 시계열(43번). 입력은 s2n_recheck.py 출력(첫 인자 = 그 출력 폴더)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ident-s3"))
import groove_lib as gl  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out_s2n")
J = json.load(open(os.path.join(D, "s2n_recheck.json")))
Z = np.load(os.path.join(D, "s2n_frf.npz"))
TODAY, REF = J["today"], J["ref"]
OTHER_TODAY = [g for g in J["groups"] if g.startswith("11-11") and g != TODAY]
MODES = ["3164", "3250", "3459", "3589", "3748", "3804", "4225", "4967", "5745", "7462", "8321"]
KCOL = {"disk": "C0", "panel": "C1", "mixed": "C3", "hub": "C3", "back": "C2", "none": "0.45"}
NHIT = {g: J["res"][g]["qc"]["used"] for g in J["groups"]}
LAB = {"10-30 S3–5": "10-30 S3-5 (after re-hang, gloves)", "10-30 S2": "10-30 S2 (before re-hang)",
       "10-28 S1": "10-28 S1 (first hanging of specimen 2)", "11-11 S7": "11-11 S7 (16:55, 9 hits)",
       "11-11 S8": "11-11 S8 (17:12, hanger re-hung, 19 hits)"}
sec = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + float(s[6:])

fig = plt.figure(figsize=(12, 10.4))
ax = fig.add_axes([0.065, 0.695, 0.925, 0.265])
bx = fig.add_axes([0.065, 0.385, 0.58, 0.22])
fx = fig.add_axes([0.715, 0.385, 0.275, 0.22])
cxs = [fig.add_axes([0.065 + 0.315 * i, 0.06, 0.28, 0.215]) for i in range(3)]

# (a) 모드별 차: 오늘 − 기준, 그리고 다른 무리 − 기준
x = np.arange(len(MODES))
ax.axhspan(-0.05, 0.05, color="0.9", lw=0, zorder=0)
T1 = {g: J["res"][g]["qc"]["T1"] for g in J["res"]}
T0 = {g: J["res"][g]["qc"]["T0"] for g in J["res"]}
dT1, dT0 = T1[TODAY] - T1[REF], T0[TODAY] - T0[REF]
ax.axhspan(gl.DF_T * dT1, gl.DF_T * dT0, color="C3", alpha=0.18, lw=0, zorder=0)
ax.axhline(0, color="k", lw=0.6)
for g, mk, dx in (("10-30 S2", "s", -0.3), ("10-28 S1", "^", 0.3)):
    ys = [J["prev"][n][g]["d_pct"] for n in MODES]
    es = [J["prev"][n][g]["se_pct"] for n in MODES]
    ax.errorbar(x + dx, ys, es, fmt=mk, ms=5, mfc="none", color="0.55", capsize=2, lw=0.7, label=LAB[g] + " - S3-5")
for g in OTHER_TODAY:
    ys = [J["prev"][n][g]["d_pct"] for n in MODES]
    es = [J["prev"][n][g]["se_pct"] for n in MODES]
    ax.errorbar(x + 0.12, ys, es, fmt="D", ms=4.5, color="0.2", capsize=2, lw=0.8, label=LAB[g] + " - S3-5")
for i, n in enumerate(MODES):
    c = J["comp"][n][REF]
    ax.errorbar(i - 0.08, c["d_pct"], c["se_pct"], fmt="o", ms=7, color=KCOL[gl.MODE[n].kind], capsize=2.5, lw=1.1)
ax.plot([], [], "o", color="k", ms=7, label=LAB[TODAY] + " - S3-5 (colour = mode kind)")
ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n{gl.MODE[n].fem or '?'}\n{gl.MODE[n].kind}" for n in MODES], fontsize=7.5)
for t, n in zip(ax.get_xticklabels(), MODES):
    t.set_color(KCOL[gl.MODE[n].kind])
ax.set_ylabel("frequency change vs 10-30 S3-5 [%]\n(20 N equiv., pooled fit, ±1 s.e.)", fontsize=8.5)
ax.set_ylim(-0.62, 0.62)
ax.set_xlim(-0.6, len(MODES) - 0.4)
ax.text(len(MODES) - 0.45, -0.46, "grey band: ±0.05 % re-hang floor (doc 30)", fontsize=7.5, ha="right", va="bottom", color="0.35")
ax.text(len(MODES) - 0.45, -0.49, f"red band: expected from temperature (ai1 {dT1:+.1f} K, ai0 {dT0:+.1f} K) = "
        f"{gl.DF_T * dT1:+.2f}…{gl.DF_T * dT0:+.2f} %", fontsize=7.5, ha="right", va="top", color="C3")
ax.legend(fontsize=7.3, loc="upper left", ncol=2, frameon=False)
ax.set_title("(a) vane 1 xi0.3, back plate: each group vs 10-30 S3-5. h2 pair 2134/2142 not resolved at xi0.3 today "
             "(S7 0/9, S8 7/19 hits)", fontsize=9, loc="left")

# (b) 평균 FRF
fr = Z["fr"]
groups = list(Z["groups"])
sty = {TODAY: dict(color="k", lw=0.9), "10-30 S3–5": dict(color="C0", lw=0.8, alpha=0.85),
       "10-28 S1": dict(color="C2", lw=0.6, alpha=0.7)}
for g in OTHER_TODAY:
    sty[g] = dict(color="0.55", lw=0.7, ls="--")
for i, g in enumerate(groups):
    if g in sty:
        bx.plot(fr / 1e3, 20 * np.log10(np.abs(Z[f"H{i}"]) + 1e-12), label=g, **sty[g])
bx.set_xlim(1.9, 9.0)
bx.set_ylim(-45, 25)
bx.set_xlabel("frequency [kHz]")
bx.set_ylabel("|H1| [dB re 1 Pa/N]")
fr_ = J["frac"]
bx.legend(fontsize=7.3, loc="lower left", ncol=4, frameon=False)
fs7 = f", vs S7 {fr_[OTHER_TODAY[0]]:.2f}" if OTHER_TODAY else ""
bx.set_title(f"(b) mean FRF. Shape FRAC 1.9-9 kHz: {TODAY[-2:]} vs S3-5 {fr_[REF]:.2f}{fs7}; "
             f"earlier hangings vs S3-5 {fr_[REF + '~10-30 S2']:.2f}-{fr_[REF + '~10-28 S1']:.2f}", fontsize=8.5, loc="left")

# (f) 오늘 타격별 시계열(기준 대비 %)
t00 = min(sec(J["res"][g]["hits"][0]["ts"]) for g in [TODAY] + OTHER_TODAY)
for n, col in (("3164", "C1"), ("3748", "C0"), ("8321", "C5"), ("3804", "C3"), ("4225", "C4")):
    f_ref = J["res"][REF]["modes"][n]["pool"]["f"]
    sl = gl.MODE[n].slope
    for g in OTHER_TODAY + [TODAY]:
        t, y = [], []
        for h in J["res"][g]["hits"]:
            v = h["f"].get(n)
            if v and h["N"] >= 5:
                t.append((sec(h["ts"]) - t00) / 60)
                y.append(100 * (v[0] * (1 - sl * np.log2(h["N"] / 20)) / f_ref - 1))
        fx.plot(t, y, ".", color=col, ms=4, label=n if g == TODAY else None)
fx.axhline(0, color="k", lw=0.6)
fx.axhspan(-0.05, 0.05, color="0.9", lw=0, zorder=0)
fx.set_ylim(-0.42, 0.12)
fx.set_xlabel("time since 16:55 [min]")
fx.set_ylabel("per-hit f(20 N) vs S3-5 [%]", fontsize=8)
fx.legend(fontsize=7, ncol=3, loc="lower center", frameon=False, handletextpad=0.1)
if OTHER_TODAY:
    fx.text(0.5, 0.075, "S7", ha="center", fontsize=8)
    fx.text(17.5, 0.075, "S8 (re-hung)", ha="center", fontsize=8)
fx.set_title("(f) today's hits (S7, then S8)", fontsize=8.5, loc="left")

# (c)–(e) 확대
zooms = [((2.10, 2.17), ["2134", "2142"], "(c) h2 pair band"), ((3.13, 3.28), ["3164", "3250"], "(d) 3164 panel / 3250 mixed (hub 0.15)"),
         ((3.72, 3.83), ["3748", "3804"], "(e) 3748 disk / 3804 hub")]
for cx, ((lo, hi), names, title) in zip(cxs, zooms):
    sel = (fr / 1e3 >= lo) & (fr / 1e3 <= hi)
    for i, g in enumerate(groups):
        if g in sty:
            cx.plot(fr[sel] / 1e3, 20 * np.log10(np.abs(Z[f"H{i}"][sel]) + 1e-12), **sty[g])
    for n in names:
        for g, ls in ((TODAY, "-"), (REF, "--")):
            M = J["res"][g]["modes"][n]
            f = M.get("pool", {}).get("f", np.nan)
            if np.isfinite(f):
                cx.axvline(f / 1e3, color=sty[g]["color"], ls=ls, lw=0.6, alpha=0.8)
    cx.set_xlim(lo, hi)
    cx.set_xlabel("frequency [kHz]")
    cx.set_title(title, fontsize=8.5, loc="left")
cxs[0].set_ylabel("|H1| [dB re 1 Pa/N]")
cxs[0].text(0.02, 0.04, f"{TODAY[-2:]}: no pooled fit\n(dashed = S3-5 poles)", transform=cxs[0].transAxes, fontsize=7)
cxs[1].text(0.02, 0.97, f"vertical lines: pooled-fit poles\n(solid {TODAY[-2:]}, dashed S3-5)", transform=cxs[1].transAxes,
            fontsize=7, va="top")
fig.savefig(os.path.join(HERE, "..", "s2n-recheck.png"), dpi=130)
print("saved")
