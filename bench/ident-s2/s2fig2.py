"""그림: 2번 시편 깃 1–6 ξ0.3 — (a) 깃 합 FRF와 깨끗한 극점(원주 차수 색), (b) 측정 대 FE(같은 h 후보), (c) 깃 순서(시간)에 따른 20 N 주파수."""
import sys, os, json, csv
sys.path.insert(0, sys.argv[1])
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from w22joint import vane_avg

OUT = sys.argv[2]
Z = np.load(os.path.join(OUT, "s2", "s2_spec.npz")); fr = Z["fr"]; mz = (~Z["late"]) & (Z["N"] >= 5)
H = vane_avg(Z["F"][mz].astype(complex), Z["X"][mz].astype(complex), Z["vane"][mz], Z["N"][mz], -0.001, fr)
P = sum(np.abs(H[v]) ** 2 for v in range(1, 7))
rows = json.load(open(os.path.join(OUT, "s2", "s2fe.json")))
col = {0: "C0", 1: "C1", 2: "C2", 3: "C3"}
fig = plt.figure(figsize=(13, 9))
ax1 = fig.add_subplot(2, 1, 1); ax2 = fig.add_subplot(2, 2, 3); ax3 = fig.add_subplot(2, 2, 4)
s = (fr >= 1900) & (fr <= 10400)
ax1.semilogy(fr[s], P[s] / P[s].max(), "k", lw=0.7)
for r in rows:
    ls = "-" if r["clean"] else ":"
    ax1.axvline(r["f"], color=col[r["h"]], lw=0.9, ls=ls, alpha=0.8)
    ax1.text(r["f"], 2.0, f"{r['f']:.0f} h{r['h']}", rotation=90, fontsize=6.5, va="bottom", color=col[r["h"]])
for h in range(4):
    ax1.plot([], [], color=col[h], label=f"h{h}")
ax1.plot([], [], color="gray", ls=":", label="not clean (MPC < 0.75, h share < 0.6 or boot < 0.9)")
ax1.set_xlim(1900, 10400); ax1.set_ylim(1e-5, 30); ax1.set_xlabel("frequency [Hz]"); ax1.set_ylabel("Σ|H1|² over vanes (norm.)")
ax1.legend(fontsize=7, ncol=5, loc="lower right"); ax1.set_title("(a) specimen 2, vanes 1–6 ξ0.3, microphone a few mm above the back plate (clean hits, 20 N)", fontsize=9)
# (b) FE 대비
for r in rows:
    if not r["fe"]:
        continue
    lab, f_fe = r["fe"][0]
    d = 100 * (r["f"] / f_fe - 1)
    ax2.plot(f_fe, d, "o" if r["clean"] else "x", color=col[r["h"]], ms=6)
    ax2.text(f_fe, d + 0.3, lab, fontsize=6, ha="center", color=col[r["h"]])
for f0, f1, lab in ((2259.3, 2041.0, "S1 2041"), (2259.3, 2062.4, "S1 2062")):
    ax2.plot(f0, 100 * (f1 / f0 - 1), "s", mfc="none", color="C2", ms=6)
ax2.text(2300, -9.6, "specimen 1 (dented)\nh2 pair", fontsize=6.5, color="C2")
ax2.axhline(0, color="k", lw=0.5); ax2.set_xlabel("FE frequency (ff32x16h6w62c40) [Hz]"); ax2.set_ylabel("measured / FE − 1 [%]")
ax2.set_ylim(-11, 5.5); ax2.set_title("(b) nearest FE mode of the same h within 8 % (candidate only; three h1 poles share h1:m1n1); o clean, x not clean", fontsize=8.5)
# (c) 깃 순서별 20 N 주파수(시리즈 중앙 대비)
T = json.load(open(os.path.join(OUT, "s2", "s2track.json")))
SL = {"2132": -0.00059, "3458": -0.00032, "3638": -0.00051, "3803": -0.00036, "4221": -0.00050, "7453": -0.00071, "8315": -0.00046}
for k, (name, sl) in enumerate(SL.items()):
    vals = {v: [r["f"][name][0] * (1 - sl * np.log2(r["N"] / 20)) for r in T if r["v"] == v and r["f"][name]] for v in range(1, 7)}
    md = np.median(np.concatenate([vals[v] for v in vals if vals[v]]))
    xs = [v + (k - 3) * 0.06 for v in range(1, 7) if len(vals[v]) >= 2]
    ys = [100 * (np.median(vals[v]) / md - 1) for v in range(1, 7) if len(vals[v]) >= 2]
    ax3.plot(xs, ys, "o-", ms=3, lw=0.8, label=name + " Hz")
ax3.axhline(0, color="k", lw=0.5); ax3.set_xticks(range(1, 7)); ax3.set_xticklabels(["v1\n17:08", "v2\n17:11", "v3\n17:14", "v4\n17:17", "v5\n17:22", "v6\n17:38"], fontsize=7)
ax3.set_ylabel("20 N vane median / series median − 1 [%]"); ax3.legend(fontsize=6.5, ncol=2)
ax3.set_title("(c) vane order = time order: +0.05–0.15 % rise from v1 to v6 (within-vane slope ≈ 0)", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "s2", "s2_specimen2.png"), dpi=120)
print("saved")
