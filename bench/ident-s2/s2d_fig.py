"""그림: 다시 매달기 전후 모드 주파수 변화(1번 깃 ξ0.3, 20 N 환산 타격별 중앙값) + 타격별 시계열 + h2 대역 FRF."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
S = json.load(open(OUT + "s2d_rehang_summary.json"))
rows = json.load(open(OUT + "s2d_rehang.json"))
H2 = json.load(open(OUT + "s2d_h2.json"))
ft = pd.read_csv(OUT + "s2c_mass_fe_fractions.csv").set_index("label")
SL = {"2134": -0.00059, "2142": -0.00059, "3459": -0.00032, "3639": -0.00051, "3804": -0.00036, "4225": -0.00050, "7462": -0.00071, "8321": -0.00046}
FE = {"3164": "h3:m3n0", "3250": "h0:m0n1", "3459": "h1:m1n1", "3589": None, "3639": "h2:m4n0", "3748": "h3:m3n1", "3804": "h0:m6n1",
      "4225": "h1:m1n1#2", "4967": "h2:m4n1", "6439": "h2:m2n1", "7462": None, "8321": "h3:m3n1#2"}
REF = "10-30 16:21 전"; POST = ["10-30 19:07 직후", "10-30 19:17 +10분", "10-30 19:28 +21분"]
LAB = {"10-28 17:08": "10-28 17:08", REF: "10-30 16:21 before", POST[0]: "19:07 (0 min)", POST[1]: "19:17 (+10 min)", POST[2]: "19:28 (+21 min)"}
fig = plt.figure(figsize=(12, 8.2))
ax = fig.add_axes([0.06, 0.6, 0.92, 0.34]); bx = fig.add_axes([0.06, 0.07, 0.6, 0.38]); cx = fig.add_axes([0.73, 0.07, 0.25, 0.38])
modes = [m for m in FE if m in S]
x = np.arange(len(modes) + 1)
# h2 짝: 전 = 같은 매달기 ξ0.6 묶음 적합, 후 = S3·S4 묶음 적합(S5는 분해 안 됨)
b0 = H2["10-30 15:55 전 ξ0.6|2"]
for j, (g, mk) in enumerate(zip(POST, ["o", "s", "^"])):
    xs, ys, es = [], [], []
    key = f"{g}|2"
    if g != POST[2]:
        q = H2[key]
        for c in (0, 1):
            xs.append(-0.15 + 0.3 * c + (j - 1) * 0.08); ys.append(100 * (q["f"][c] / b0["f"][c] - 1)); es.append(100 * np.hypot(q["sd"][c], b0["sd"][c]) / b0["f"][c])
    for i, m in enumerate(modes):
        if g in S[m]:
            xs.append(i + 1 + (j - 1) * 0.18); ys.append(S[m][g]["d_pct"]); es.append(S[m][g]["se_pct"])
    ax.errorbar(xs, ys, es, fmt=mk, ms=5, mfc="none", capsize=2, lw=0.8, label=LAB[g])
ax.axhline(0, color="k", lw=0.6)
ax.axhspan(-0.02 - 0.005, -0.02 + 0.005, color="C3", alpha=0.15, lw=0)
ax.text(len(modes) + 0.3, -0.035, "ai1 +0.9 K -> -0.02 %", color="C3", fontsize=6.5, va="top", ha="right")
ax.set_xticks(x)
tl = ["h2 pair 2134/2142\nh2:m2n0\n[0.01]"]
for m in modes:
    lab = FE[m]
    th = (ft.loc[lab, "T_hub"] + ft.loc[lab, "T_collar"]) if lab else np.nan
    tl.append(f"{m}\n{lab or '?'}\n" + ("" if np.isnan(th) else f"[{th:.2f}]"))
ax.set_xticklabels(tl, fontsize=7)
ax.set_ylabel("after re-hang vs before [%]")
ax.set_ylim(-0.32, 0.15)
ax.legend(fontsize=7, loc="lower left", ncol=3)
ax.set_title("(a) vane 1 xi0.3, 20 N medians per group; h2 pair = pooled fits vs xi0.6 of the same hanging before; [ ] = FE kinetic share hub+collar", fontsize=8.5, loc="left")
# (b) 타격별 시계열
c20 = lambda r, n: r["f"][n][0] * (1 - SL.get(n, 0.0) * np.log2(r["N"] / 20))
grp = ["10-28 17:08", REF] + POST
for m, col in zip(["4225", "3804", "3459", "3164", "8321"], ["C3", "C2", "C0", "C1", "C4"]):
    ref = S[m][REF]["med"]
    for gi, g in enumerate(grp):
        R = [r for r in rows if r["g"] == g and r["f"].get(m)]
        if not R:
            continue
        v = np.array([100 * (c20(r, m) / ref - 1) for r in R])
        xx = gi + np.linspace(-0.3, 0.3, len(v)) + {"4225": -0.04, "3804": -0.02, "3459": 0, "3164": 0.02, "8321": 0.04}[m]
        bx.plot(xx, v, "o", color=col, ms=3, alpha=0.8, label=m if gi == 0 else None)
        bx.plot([gi - 0.33, gi + 0.33], [S[m][g]["d_pct"]] * 2, color=col, lw=1.5)
bx.axhline(0, color="k", lw=0.6)
bx.set_xticks(range(len(grp))); bx.set_xticklabels([LAB[g] for g in grp], fontsize=7.5)
bx.axvline(1.5, color="0.5", ls="--", lw=0.8); bx.text(1.53, 0.2, "re-hung (gloves)", fontsize=7, color="0.4")
bx.set_ylim(-0.4, 0.3); bx.set_ylabel("per-hit f (20 N) vs 16:21 median [%]")
bx.legend(fontsize=7, ncol=5, loc="lower right", title="mode [Hz]", title_fontsize=7)
bx.set_title("(b) per-hit values (lines = group medians); 10-28 group was the first after a specimen change", fontsize=8.5, loc="left")
# (c) h2 대역 FRF
z = np.load(OUT + "s2d_frf.npz", allow_pickle=False); fr = z["fr"]; names = list(z["names"])
for i, n in enumerate(names):
    s = (fr > 2110) & (fr < 2165)
    cx.plot(fr[s], 20 * np.log10(np.abs(z[f"H{i}"][s])), lw=1, label=LAB.get(n, n))
cx.set_xlabel("Hz"); cx.set_ylabel("|H1| [dB re 1 Pa/N]"); cx.legend(fontsize=6.3, loc="lower center")
cx.set_title("(c) h2 band at vane 1 xi0.3", fontsize=8.5, loc="left")
fig.savefig(OUT + "s2d_rehang.png", dpi=160)
print("saved")
