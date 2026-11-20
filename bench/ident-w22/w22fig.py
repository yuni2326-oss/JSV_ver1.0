"""impact_testw22 그림: (a) 깃 6곳 합 FRF와 원주 차수, (b) 극점별 깃 패턴과 cos h(θ−θ0), (c) 타격별 주파수 시간 추이."""
import sys, os, json
sys.path.insert(0, sys.argv[1])
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from w22joint import vane_avg

OUT = sys.argv[2]; png = sys.argv[3]
z = np.load(os.path.join(OUT, "w22_spec.npz"), allow_pickle=False)
fr = z["fr"]; F = z["F"].astype(complex); X = z["X"].astype(complex); V = z["vane"]
H = vane_avg(F, X, V)
P = json.load(open(os.path.join(OUT, "w22c_slope0.0.json")))
hits = json.load(open(os.path.join(OUT, "w22d_hits.json")))
HC = {0: "C2", 1: "C0", 2: "C3", 3: "C4"}
CLEAN = [2041.0, 2062.4, 3061.6, 3489.3, 3630.6, 3778.0, 4273.8, 6866.1, 7291.6, 7823.6]
fig = plt.figure(figsize=(14, 14))
gs = GridSpec(4, 5, figure=fig, height_ratios=[1.15, 1, 1, 1.1], hspace=0.75, wspace=0.3)
ax = fig.add_subplot(gs[0, :])
for v in range(1, 7):
    ax.plot(fr / 1e3, 20 * np.log10(np.abs(H[v])), lw=0.6, label=f"vane {v}")
ax.set_xlim(1.9, 8.0); ax.set_ylim(-75, 5)
ymax = 2
for p in P:
    if p["mpc"] < 0.75 or max(p["P"]) < 0.6 or p["rel"] < -12:
        continue
    c = HC[p["h"]]
    ax.axvline(p["f"] / 1e3, color=c, lw=1.0, alpha=0.6, ls="--")
    ax.text(p["f"] / 1e3, ymax, f"h{p['h']}", color=c, fontsize=8, ha="center", va="bottom", rotation=0)
ax.set_xlabel("frequency [kHz]"); ax.set_ylabel("|H1| [dB re 1 Pa/N]")
leg = ax.legend(ncol=6, fontsize=8, loc="lower left")
for l_ in leg.get_lines():
    l_.set_linewidth(2.0)
ax.set_title("(a) vane-average FRF, mic fixed, hammer on vanes 1-6 (dashed = poles with a clean circumferential order, MPC >= 0.75)", fontsize=10, loc="left")
th = np.linspace(0, 360, 361); thv = 60.0 * np.arange(6)
for i, f0 in enumerate(CLEAN):
    p = min(P, key=lambda q: abs(q["f"] - f0))
    a = np.array(p["a"]); h = p["h"]
    axp = fig.add_subplot(gs[1 + i // 5, i % 5])
    A = (a * np.exp(-1j * h * np.radians(thv))).sum()
    if h == 0:
        cur = np.full_like(th, a.mean())
    elif h == 3:
        cur = (a * np.cos(np.radians(3 * thv))).sum() / 6 * np.cos(np.radians(3 * th))
    else:
        C = abs(A) / 3; t0 = -np.degrees(np.angle(A)) / h
        cur = C * np.cos(np.radians(h * (th - t0)))
    axp.plot(th, cur, color=HC[h], lw=1.2)
    axp.plot(thv, a, "ko", ms=4)
    for k in range(6):
        axp.text(thv[k], a[k] + (0.12 if a[k] >= 0 else -0.22), str(k + 1), fontsize=7, ha="center")
    axp.axhline(0, color="0.6", lw=0.5)
    axp.set_xticks([0, 120, 240, 360]); axp.set_ylim(-1.35, 1.35); axp.set_xlim(-15, 375)
    axp.set_title(f"{p['f']:.1f} Hz: h{h} ({max(p['P']):.2f}), MPC {p['mpc']:.2f}", fontsize=8.5, color=HC[h])
    if i % 5 == 0:
        axp.set_ylabel("residue (signed)")
    if i // 5 == 1:
        axp.set_xlabel("vane angle [deg]")
bb = fig.axes[1].get_position()
fig.text(bb.x0, bb.y1 + 0.032, "(b) residue of each pole at vanes 1-6 (dots, numbered) and fitted cos h(theta - theta0) of the dominant order (energy share in brackets)", fontsize=10)
ax = fig.add_subplot(gs[3, :])
sl = {"2041 h2": -0.0009, "2062 h2": -0.0009, "4274 h2": -0.0007, "5113": -0.0004, "7292 h3": -0.0004}
sec = lambda s: (int(s[:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8]) - 20 * 3600) / 60
mk = {"2041 h2": "o", "2062 h2": "s", "4274 h2": "^", "5113": "v", "7292 h3": "D"}
for n, s_ in sl.items():
    pts = [(sec(r["ts"]), r["f"][n] * (1 - s_ * np.log2(r["N"] / 20)), r["v"]) for r in hits if r["f"][n] and not (n.startswith("20") and r["v"] == 2)]
    if n == "4274 h2":
        pts = [q for q in pts if q[2] not in (1, 4)]
    t = np.array([q[0] for q in pts]); f = np.array([q[1] for q in pts])
    ref = np.median([q[1] for q in pts if q[2] in (3, 4)] or f)
    ax.plot(t, (f / ref - 1) * 100, mk[n], ms=4, ls="none", label=n + " Hz", alpha=0.85)
for v, (a_, b_) in {1: (2.0, 7.5), 2: (8.0, 10.9), 3: (11.0, 14.2), 4: (14.4, 17.3), 5: (17.6, 21.9), 6: (22.3, 29.5)}.items():
    ax.axvspan(a_, b_, color=f"C{v-1}", alpha=0.07); ax.text((a_ + b_) / 2, 0.33, f"vane {v}", ha="center", fontsize=8)
ax.set_ylim(-0.4, 0.4); ax.set_xlim(1.5, 30)
ax.set_xlabel("time after 20:00 [min]"); ax.set_ylabel("frequency at 20 N,\nrel. to vanes 3-4 [%]")
ax.legend(fontsize=8, ncol=5, loc="lower left")
ax.grid(alpha=0.3)
ax.set_title("(c) per-hit pole frequency in time order (vane 2 omitted for the 2 kHz pair, vanes 1 and 4 for 4274 Hz: near a node)", fontsize=10, loc="left")
fig.savefig(png, dpi=120, bbox_inches="tight")
print("saved", png)
