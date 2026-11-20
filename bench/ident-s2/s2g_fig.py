"""그림: (a) 캘리퍼 수준 치수 오차 하나가 모드별 주파수를 얼마나 움직이나, (b) 치수 가정별 측정/FE − 1 (빠진 질량은 FE 분포대로)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
J = json.load(open(OUT + "s2g_fit.json"))
info = J[-1]
case = {c["case"]: np.array(c["err"]) for c in J[:-1]}
chk = pd.read_csv(OUT + "s2g_check.csv", index_col=0)
names = info["names"]
S = np.array(info["S_pct_per_mm"]) * np.array(info["step_mm"])
T = np.array(info["T"])
order = np.argsort(T[:, 1])                      # 뒤판 몫 순서
x = np.arange(len(names))
fig, (ax, bx) = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
lab = {"tf": "front plate +0.02 mm", "tb": "back plate +0.02 mm", "tv": "vane +0.1 mm", "ch": "plate gap +0.1 mm", "colt": "inlet-collar wall +0.05 mm"}
keys = ["tf", "tb", "tv", "ch", "colt"]
w = 0.16
for i, k in enumerate(keys):
    j = info["params"].index(k)
    ax.bar(x + (i - 2) * w, S[order, j], width=w, label=lab[k], color=["C0", "C3", "C1", "C4", "C2"][i])
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("frequency change  [%]")
ax.set_title("(a) one caliper-size dimension error, FE (specimen-2 dims, 32x16)", fontsize=9, loc="left")
ax.legend(fontsize=6.8, ncol=3, loc="upper left")
ax.set_ylim(-0.8, 5.6)
bx.axhspan(-1, 1, color="0.92", zorder=0)
pts = [(case["옛 FE 그대로"], "s", "none", "0.45", "specimen-1 dims (front 0.375, back 0.59, vane 1.2), FE mass 36.78 g"),
       (case["옛 FE + 41.32 g 분포대로"], "s", "0.45", "0.45", "same + missing 4.54 g spread like the FE mass (41.32 g)"),
       (case["2번 시편 치수 + 빠진 질량 FE 분포대로"], "o", "C3", "C3", "specimen-2 dims (front 0.40, back 0.58, vane 1.0) + missing mass"),
       (chk["c_nominal_unif"].to_numpy(), "D", "C2", "C2", "sheet gauges front 0.40, back 0.60, vane 1.0 + missing mass (FE run)"),
       (chk["c_unif_fit"].to_numpy(), "^", "C0", "C0", "best fit inside caliper error: front 0.406, back 0.609 + missing mass (FE run)")]
for e, m, fc, ec, l in pts:
    bx.plot(x, e[order], m, mfc=fc, mec=ec, color=ec, ms=6, label=l, ls="none")
bx.axhline(0, color="k", lw=0.6)
bx.set_ylabel("measured / FE - 1  [%]")
bx.set_ylim(-8, 9.5)
bx.set_title("(b) specimen 2 (41.32 g), 10-30 poles; grey band = +/-1 %", fontsize=9, loc="left")
bx.legend(fontsize=6.6, loc="upper left")
fm = np.array(info["fmeas"])
bx.set_xticks(x)
bx.set_xticklabels([f"{names[i].replace('h3:m3n0+h0:m0n1', 'm3n0+m0n1')}\n{fm[i]:.0f} Hz\nback {T[i, 1]:.2f}" for i in order], fontsize=6.6, rotation=0)
fig.tight_layout()
fig.savefig(OUT + "s2g_dims.png", dpi=170)
print("saved")
