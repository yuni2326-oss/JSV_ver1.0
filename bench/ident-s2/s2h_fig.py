"""그림: 표준 두께 FE(앞판 0.40, 뒤판 0.60 mm) 대 2번 시편 측정 극점 — (a) 측정/FE − 1, (b) FE 구역 운동에너지 몫."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
J = json.load(open(OUT + "s2h_nominal.json"))
fe = {e["label"]: e for e in J["fe"]}
# (측정 극점, FE 이름, 확정 여부) — 확정 = 28번 반지름 비로 짝지은 것, 후보 = 표준 두께 FE에서 새로 본 것
PAIRS = [(2133.80, "h2:m2n0", 1), (2142.28, "h2:m2n0", 1), (3163.65, "h3:m3n0", 1), (3249.89, "h0:m0n1", 1), (3458.89, "h1:m1n1", 1),
         (3638.99, "h2:m4n0", 1), (3747.87, "h3:m3n1", 1), (3804.07, "h0:m6n1", 1), (4225.49, "h1:m1n1#2", 1), (4966.99, "h2:m4n0#2", 1),
         (5617.22, "h0:m6n0", 1), (5744.69, "h1:m5n0", 1), (6438.86, "h2:m2n1", 1), (6558.41, "h2:m2n1", 0), (6817.09, "h1:m7n0", 0),
         (7093.13, "h3:m3n2", 0), (7574.17, "h2:m8n0", 0), (8321.24, "h3:m3n1#2", 1)]
x = np.arange(len(PAIRS))
col = {0: "C0", 1: "C1", 2: "C2", 3: "C3"}
fig, (ax, bx) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
ax.axhspan(-2.5, 2.5, color="0.93", zorder=0)
for i, (f, lab, ok) in enumerate(PAIRS):
    e = fe[lab]
    ax.plot(i, 100 * (f / e["f"] - 1), "x", color="0.55", ms=5)
    ax.plot(i, 100 * (f / e["fm"] - 1), "o", ms=7, mfc=col[e["h"]] if ok else "white", mec=col[e["h"]], mew=1.5)
for h in range(4):
    ax.plot([], [], "o", color=col[h], label=f"h{h}")
ax.plot([], [], "o", mfc="white", mec="k", label="new candidate")
ax.plot([], [], "x", color="0.55", label=f"FE without missing mass ({J['mass']:.2f} g)")
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("measured / FE - 1  [%]")
ax.set_ylim(-9, 7)
ax.legend(fontsize=7, ncol=3, loc="upper right")
ax.set_title(f"(a) specimen 2 vs FE with sheet gauges front 0.40 / back 0.60 mm, missing {41.3186 - J['mass']:.2f} g spread like the FE mass; grey band = +/-2.5 %",
             fontsize=8.5, loc="left")
bot = np.zeros(len(PAIRS))
for j, (k, c) in enumerate([("front plate", "C0"), ("back plate", "C3"), ("hub", "C4"), ("vanes", "C1"), ("inlet collar", "C2")]):
    v = np.array([fe[lab]["T"][j] for _, lab, _ in PAIRS])
    bx.bar(x, v, bottom=bot, color=c, width=0.6, label=k)
    bot += v
bx.set_ylabel("FE kinetic share")
bx.set_xticks(x)
bx.set_xticklabels([f"{f:.0f}\n{lab}" for f, lab, _ in PAIRS], fontsize=6.3, rotation=50, ha="right")
bx.legend(fontsize=7, ncol=5, loc="upper center")
bx.set_ylim(0, 1.28)
bx.set_title("(b) FE kinetic-energy share by part", fontsize=8.5, loc="left")
fig.tight_layout()
fig.savefig(OUT + "s2h_nominal.png", dpi=170)
print("saved")
