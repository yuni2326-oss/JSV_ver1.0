"""그림 t-series.png(44번): 11-11 한 점 테이프 시험(T-a·T-b·T-c·T0, 1번 깃 ξ0.6) 요약.
(a) h2 짝 두 극점의 주파수 — 측정(묶음 적합) 대 FEM 예측(기준 × (1 + 예측))
(b) 패널 모드 변화 — T-b·T-c 측정 대 FEM
(c) T0(테이프를 다 뗀 뒤) − 기준: 돌아오지 않은 몫
입력: groove-test/out/<상태>.json(groove_state), fe-nominal/s3_pred_poles.csv."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "groove-test", "out")
PRED = pd.read_csv(os.path.join(HERE, "..", "fe-nominal", "s3_pred_poles.csv"))
PRED = PRED.assign(mode=PRED["mode"].astype(str))
BASE = "A-1111-x06"
STATES = [(BASE, "baseline\n(S2 17:24)"), ("T-a", "T-a 30 mg\nvane 2 (S3)"), ("T-b", "T-b 30 mg\nvane 3 -14° (S4)"),
          ("T-c", "T-c 10 mg\nbay 3-4 (S6)"), ("T0", "T0 tape off\n(S7 18:19)")]
J = {s: json.load(open(os.path.join(OUT, f"{s}.json"))) for s, _ in STATES}


def comb(s, n):
    c = J[s]["modes"][n]["comb"]
    return c["f"], c["se"]


def pred(s, n):
    r = PRED[(PRED.state == s) & (PRED["mode"] == n)]
    return float(r.pred_pct.iloc[0]) if len(r) else np.nan


fig = plt.figure(figsize=(13, 8.6))
ax = fig.add_axes([0.08, 0.55, 0.53, 0.39])
bx = fig.add_axes([0.68, 0.55, 0.31, 0.39])
cx = fig.add_axes([0.06, 0.08, 0.93, 0.33])

# (a) h2 짝
x = np.arange(len(STATES))
for n, col in (("2134", "C0"), ("2142", "C3")):
    f0 = comb(BASE, n)[0]
    ys, es = zip(*[comb(s, n) for s, _ in STATES])
    ax.errorbar(x - 0.06 + (0.12 if n == "2142" else 0), ys, es, fmt="o", color=col, ms=7, capsize=3, label=f"measured {n} component")
    pp = [f0 * (1 + pred(s, n) / 100) if s.startswith("T-") else np.nan for s, _ in STATES]
    ax.plot(x + 0.2 + (0.06 if n == "2142" else 0), pp, "_", color=col, ms=18, mew=2.5, label=f"FEM prediction {n}")
for i, (s, _) in enumerate(STATES):
    a, b = comb(s, "2134")[0], comb(s, "2142")[0]
    ax.text(i, 2126.6, f"split\n{b - a:.1f} Hz", ha="center", fontsize=7.5, color="0.3")
ax.axhspan(2137.0, 2141.5, xmin=0.62, xmax=0.78, color="0.92", zorder=0)
ax.text(3, 2141.9, "pair merged:\ncomponents unreliable", ha="center", fontsize=7, color="0.35")
ax.set_xticks(x)
ax.set_xticklabels([l for _, l in STATES], fontsize=8)
ax.set_ylabel("h2 pair pole frequency [Hz]\n(20 N equiv., pooled fit ±1 s.e.)", fontsize=8.5)
ax.set_ylim(2125.5, 2146.0)
ax.legend(fontsize=7.3, ncol=2, loc="upper left", frameon=False)
ax.set_title("(a) h2 pair: T-a moves only the upper component, T-b only the lower one (as FEM)", fontsize=9, loc="left")

# (b) 패널 모드: T-b·T-c 측정 대 FEM
modes = ["3164", "3639", "8321", "3748"]
w = 0.2
for k, (s, col) in enumerate((("T-b", "C2"), ("T-c", "C1"))):
    f0 = [comb(BASE, n)[0] for n in modes]
    m = [100 * (comb(s, n)[0] / comb(BASE, n)[0] - 1) for n in modes]
    e = [100 * np.hypot(comb(s, n)[1], comb(BASE, n)[1]) / comb(BASE, n)[0] for n in modes]
    p = [pred(s, n) for n in modes]
    xx = np.arange(len(modes)) + (k - 0.5) * 2 * w
    bx.bar(xx - w / 2, p, w, color="none", edgecolor=col, hatch="//", lw=1, label=f"{s} FEM")
    bx.errorbar(xx + w / 2, m, e, fmt="s", color=col, ms=5, capsize=2, label=f"{s} measured")
bx.axhline(0, color="k", lw=0.6)
bx.set_xticks(np.arange(len(modes)))
bx.set_xticklabels(["3164\npanel", "3639\npanel pair*", "8321\npanel", "3748\ndisk"], fontsize=8)
bx.set_ylabel("change vs baseline [%]", fontsize=8.5)
bx.legend(fontsize=7, loc="lower left", frameon=False)
bx.set_title("(b) panel modes: FEM vs measured", fontsize=9, loc="left")
bx.text(0.98, 0.03, "* 3639 FEM = pair mean;\n  one component ≈ 2×", transform=bx.transAxes, ha="right", fontsize=6.8, color="0.3")

# (c) T0 − 기준
names = ["2134", "2142", "3164", "3459", "3589", "3639", "3748", "3804", "4225", "4967", "5745", "8321"]
kind = {"2134": "C3", "2142": "C3", "3748": "C3", "3164": "C0", "3459": "C0", "3639": "C0", "8321": "C0",
        "3589": "0.5", "3804": "C4", "4225": "C4", "4967": "C2", "5745": "C2"}
for i, n in enumerate(names):
    f0, s0 = comb(BASE, n)
    f1, s1 = comb("T0", n)
    d = 100 * (f1 / f0 - 1)
    e = 100 * np.hypot(s0, s1) / f0
    cx.errorbar(i, d, e, fmt="o", color=kind[n], ms=6, capsize=2)
cx.axhspan(-0.06, 0.06, color="0.92", zorder=0)
cx.axhline(0, color="k", lw=0.6)
cx.set_xticks(np.arange(len(names)))
cx.set_xticklabels(names, fontsize=8)
cx.set_ylim(-0.45, 0.2)
cx.set_ylabel("T0 − baseline [%]", fontsize=8.5)
t1 = np.mean([h["T1"] for h in J["T0"]["hits"]]) - np.mean([h["T1"] for h in J[BASE]["hits"]])
t0_ = np.mean([h["T0"] for h in J["T0"]["hits"]]) - np.mean([h["T0"] for h in J[BASE]["hits"]])
cx.set_title(f"(c) T0 (all tape removed, 55 min after baseline) − baseline: most modes stay 0.05-0.10 % lower\n"
             f"grey = ±0.06 % same-hanging threshold; temperature ai0 {t0_:+.2f} K, ai1 {t1:+.2f} K (cooling would raise f); "
             "red disk, blue panel, violet hub, green back, grey unidentified", fontsize=8.3, loc="left")
fig.savefig(os.path.join(HERE, "..", "groove-test", "t-series.png"), dpi=130)
print("saved")
