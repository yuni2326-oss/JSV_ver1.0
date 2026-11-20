"""그림: 깃 높이 테이퍼(바깥 3.00 → 안쪽 3.19 mm) — (a) 간격 분포, (b) E2-I 자유-자유 FEM 주파수 변화(질량 그대로),
(c) 측정/FEM − 1(41.32 g 맞춤), (d) 설치 상태 조립체 커널(낮은 세 계열), (e) 레일 지지 m0n0 커널."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/taper/"
G3 = "/home/super/project/docs/_generated_g3/data/paper3/"
FIG = "/home/super/project-paper3/docs/paper3-jsv/revision-r1/E2I-pilot-20261022/s2m-taper.png"
C = json.load(open(T + "s2m_compare.json"))
V = C["variants"]
VAR = [("taper319_32x16", "taper, front plate conical", "C3", "o"), ("taper319b_32x16", "taper, back plate conical", "C2", "s"),
       ("nom_ch319_32x16", "uniform 3.19 mm", "0.45", "^")]
VAR = [v for v in VAR if v[0] in V]
a, b = 14.89, 36.465

fig = plt.figure(figsize=(11.5, 9.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.15, 1.05], hspace=0.55, wspace=0.22)
ax = fig.add_subplot(gs[0, 0])
r = np.linspace(a, b, 200)
ax.plot(r, np.full_like(r, 3.00), color="C0", lw=2, label="G3 manuscript: uniform 3.00 mm")
ax.plot(r, 3.00 + 0.19 * (b - r) / (b - a), color="C3", lw=2, label="taper 3.19 -> 3.00 mm (linear)")
ax.plot(r, np.full_like(r, 3.19), color="0.45", lw=1.2, ls="--", label="uniform 3.19 mm")
ax.plot([b - 0.3], [3.00], "kv", ms=7)
ax.annotate("specimen 2, outlet: total 4.00 mm\n(inside jaws 2.95-3.00 mm)", (b - 0.3, 3.0), (27.0, 3.12), fontsize=7.5,
            arrowprops=dict(arrowstyle="-", lw=0.6))
ax.annotate("specimen 2, deep: total 4.19 mm", (16.5, 3.19), (17.5, 3.215), fontsize=7.5)
ax.set_xlim(a - 0.5, b + 0.5)
ax.set_ylim(2.96, 3.25)
ax.set_xlabel("radius r [mm]  (a = 14.89 inlet, b = 36.465 outlet)")
ax.set_ylabel("plate gap = vane height [mm]")
ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.06))
ax.set_title("(a) gap distribution (plates 0.40 / 0.60 mm kept)", fontsize=9, loc="left")

bx = fig.add_subplot(gs[0, 1])
labs = [l for l in V[VAR[0][0]]["modes"] if V[VAR[0][0]]["modes"][l][0] < 9000]
x = np.arange(len(labs))
w = 0.8 / len(VAR)
for k, (tag, nm, col, mk) in enumerate(VAR):
    d = [V[tag]["modes"][l][2] for l in labs]
    bx.bar(x - 0.4 + w * (k + 0.5), d, width=w, color=col, label=nm)
bx.axhline(0, color="k", lw=0.6)
bx.set_xticks(x)
bx.set_xticklabels(labs, rotation=70, fontsize=6.5)
bx.set_ylabel("FEM f change vs 3.00 [%]")
bx.legend(fontsize=7)
bx.set_title("(b) free-free FEM, mass as modelled (+0.15 / +0.31 g)", fontsize=9, loc="left")

cx = fig.add_subplot(gs[1, :])
P = V[VAR[0][0]]["pairs"]
xx = np.arange(len(P))
cx.axhspan(-2.5, 2.5, color="0.93", zorder=0)
cx.plot(xx, [p["e300"] for p in P], "o", ms=7, mfc="C0", mec="C0", label="uniform 3.00 (G3)")
for tag, nm, col, mk in VAR:
    cx.plot(xx, [p["evar"] for p in V[tag]["pairs"]], mk, ms=6, mfc="white", mec=col, mew=1.5, label=nm)
cx.axhline(0, color="k", lw=0.6)
cx.set_xticks(xx)
cx.set_xticklabels([f"{p['f']:.0f}\n{p['label']}" for p in P], fontsize=6.3)
cx.set_ylabel("measured / FEM - 1 [%]")
st = "; ".join(f"{nm}: rms {V[tag]['s11'][tag]['rms']:.2f} %" for tag, nm, _, _ in VAR)
cx.set_title(f"(c) specimen 2 poles vs FEM (missing mass spread like the FEM mass, 41.32 g); grey band = +/-2.5 %\n"
             f"    11 pairs: uniform 3.00 rms {V[VAR[0][0]]['s11']['3.00']['rms']:.2f} %; {st}", fontsize=8.3, loc="left")
cx.legend(fontsize=7, ncol=4, loc="lower right")


def kern(path, lab):
    k = pd.read_csv(path)
    s = k[k.label == lab].sort_values("xi")
    return s.xi.to_numpy(), s.k.to_numpy()


dx = fig.add_subplot(gs[2, 0])
if os.path.exists(T + "kernels_k6_taper319_24x12.csv"):
    for lab, col in [("h1:m1n0", "C1"), ("h0:m0n0", "C0"), ("h2:m2n0", "C2")]:
        xa, ya = kern(G3 + "rev1_n6_kernels_ff24x12h6w62c50k92.csv", lab)
        xb, yb = kern(T + "kernels_k6_taper319_24x12.csv", lab)
        dx.plot(xa, ya, "-", color=col, lw=1.6, label=f"{lab} uniform 3.00")
        dx.plot(xb, yb, "--", color=col, lw=1.6, label=f"{lab} taper")
    dx.axhline(0, color="k", lw=0.6)
    dx.legend(fontsize=6.5, ncol=2, loc="upper left")
dx.set_xlabel("damage centre xi")
dx.set_ylabel("kernel k = eta / S")
dx.set_title("(d) installed state (hub clamped), 24x12: lowest three families", fontsize=9, loc="left")

ex = fig.add_subplot(gs[2, 1])
if os.path.exists(T + "kernels_k1_taper319_24x12.csv"):
    for lab, col in [("h0:m0n0", "C0"), ("h1:m1n0", "C1"), ("h2:m2n0", "C2")]:
        xa, ya = kern(G3 + "rev1_n1_kernels_fit24x12v1z2c2w62.csv", lab)
        xb, yb = kern(T + "kernels_k1_taper319_24x12.csv", lab)
        ex.plot(xa, ya, "-", color=col, lw=1.6, label=f"{lab} uniform 3.00")
        ex.plot(xb, yb, "--", color=col, lw=1.6, label=f"{lab} taper")
    ex.axhline(0, color="k", lw=0.6)
    ex.legend(fontsize=6.5, ncol=1, loc="lower left")
ex.set_xlabel("damage centre xi")
ex.set_title("(e) rail support (both plates clamped at r = a), 24x12", fontsize=9, loc="left")
fig.savefig(FIG, dpi=150, bbox_inches="tight")
print(FIG)
