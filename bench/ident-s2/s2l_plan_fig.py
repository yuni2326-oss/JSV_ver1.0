"""다음 실험(E2-I 2단계) 그림: (a) 앞판 홈 위치(흡입구 쪽에서 본 개략도), (b) 뒤판 타격점, (c) 홈 깊이별 예측 주파수 변화(G3 FEM 32×16)."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

sys.path.insert(0, "/home/super/project-paper3")
from impeller_fingerprint.rev1 import n1_mesh as nm

D = "/home/super/project/docs/_generated_g3/data/paper3/"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
a, b = 14.89, 36.465
ext = b - a
fs = nm.FitSpec()
r_c = lambda xi: a + xi * ext
W = 0.2 * ext


def rings(ax, title, grooves, hits, mirror=False):
    ax.add_patch(Circle((0, 0), b, fc="0.93", ec="k", lw=1.0))
    ax.add_patch(Circle((0, 0), a, fc="white", ec="k", lw=1.0))
    ax.add_patch(Circle((0, 0), a + 0.52, fc="none", ec="0.4", lw=0.8, ls="--"))
    for xi, col in grooves:
        ro, ri = min(r_c(xi) + W / 2, b), r_c(xi) - W / 2
        ax.add_patch(Wedge((0, 0), ro, 0, 360, width=ro - ri, fc=col, alpha=0.55, ec=col))
    rr_m = np.linspace(fs.r_vane_in, fs.b, 60)
    cam = np.degrees(fs.camber(rr_m))
    sgn = -1.0 if mirror else 1.0
    for k in range(6):
        th = np.radians(-90 + 60 * k + sgn * (cam - cam[0]))
        ax.plot(rr_m * 1e3 * np.cos(th), rr_m * 1e3 * np.sin(th), color="0.35", lw=1.0)
    for xi, col, mk in hits:
        r = r_c(xi)
        for k in range(6):
            th = np.radians(-90 + 60 * k + sgn * (np.interp(r * 1e-3, rr_m, cam) - cam[0]))
            ax.plot(r * np.cos(th), r * np.sin(th), mk, color=col, ms=6, mec="k", mew=0.5)
    ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9, loc="left")


fig = plt.figure(figsize=(11.5, 6.6))
ax1 = fig.add_axes([0.01, 0.22, 0.31, 0.70])
ax2 = fig.add_axes([0.34, 0.22, 0.31, 0.70])
rings(ax1, "(a) front plate (inlet side): two ring grooves", [(0.9, "C3"), (0.3, "C0")], [], mirror=True)
for xi, col, txt in [(0.9, "C3", "outer groove  r = 32.2–36.5 mm\n(outermost 4.3 mm band, xi 0.8–1.0)"),
                     (0.3, "C0", "inner groove  r = 19.2–23.5 mm\n(xi 0.2–0.4, centre 21.4 mm)")]:
    ang = -60 if xi > 0.5 else -120
    rr_ = min(r_c(xi), b - 1.0) if xi > 0.5 else r_c(xi)
    ax1.annotate(txt, xy=(rr_ * np.cos(np.radians(ang)), rr_ * np.sin(np.radians(ang))),
                 xytext=(0.52 if xi > 0.5 else 0.0, -0.14), textcoords="axes fraction", fontsize=7.2, color=col,
                 arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
ax1.text(0, 0, "inlet\ncollar", ha="center", va="center", fontsize=7)
rings(ax2, "(b) back plate: hammer points on the vane lines", [],
      [(0.3, "C0", "o"), (0.6, "C1", "o"), (0.85, "C2", "s")])
ax2.text(0, -45, "o xi 0.3 (21.4 mm)   o xi 0.6 (27.8 mm)   ■ xi 0.85 (33.2 mm, new)", ha="center", fontsize=7.2)
for xi, col in [(0.3, "C0"), (0.6, "C1"), (0.85, "C2")]:
    ax2.plot([], [], "o", color=col, label=f"xi {xi}")
ax2.text(0, 0, "bore", ha="center", va="center", fontsize=7)

ax3 = fig.add_axes([0.71, 0.47, 0.27, 0.44])
g = pd.read_csv(D + "rev1_n6_grooves_ff32x16h6w62c50.csv")
labs = [("h2:m2n0", "2134/2142 Hz  h2:m2n0 (disk)", "C2", "-"), ("h3:m3n1", "3748 Hz  h3:m3n1 (disk)", "C3", "-"),
        ("h1:m1n1", "3459 Hz  h1:m1n1 (panel)", "C1", "--"), ("h2:m2n1", "6439 Hz  h2:m2n1 (panel)", "C4", "--")]
for xi, ls_, mk in [(0.9, "-", "o"), (0.3, ":", "s")]:
    for lab, name, col, _ in labs:
        s = g[np.isclose(g.xi, xi) & (g.label == lab)].sort_values("depth_mm")
        y = np.where(s.df_pct_res.notna(), s.df_pct_res, s.df_pct_lin)
        ax3.plot(s.depth_mm, y, ls_, marker=mk, color=col, ms=4, label=name if xi == 0.9 else None)
ax3.axhspan(-0.15, 0.15, color="0.9", zorder=0)
ax3.axhline(0, color="k", lw=0.6)
ax3.set_xlabel("groove depth [mm]  (front plate 0.40 mm)")
ax3.set_ylabel("frequency change [%]")
ax3.set_title("(c) predicted shift, G3 FEM free-free\nsolid = outer groove, dotted = inner groove", fontsize=8.5, loc="left")
ax3.set_ylim(-5, 0.6)
ax3.legend(fontsize=6.5, loc="lower left")
ax4 = fig.add_axes([0.71, 0.08, 0.27, 0.25])
for lab, name, col, _ in labs[:2]:
    s = g[np.isclose(g.xi, 0.9) & (g.label == lab)].sort_values("depth_mm")
    y = np.where(s.df_pct_res.notna(), s.df_pct_res, s.df_pct_lin)
    ax4.plot(s.depth_mm, y, "-o", color=col, ms=4)
    for xi2, mk in [(0.85, "^"), (0.95, "v")]:
        s2 = g[np.isclose(g.xi, xi2) & (g.label == lab)].sort_values("depth_mm")
        y2 = np.where(s2.df_pct_res.notna(), s2.df_pct_res, s2.df_pct_lin)
        ax4.plot(s2.depth_mm, y2, ":" + mk, color=col, ms=4)
ax4.axhspan(-0.15, 0.15, color="0.9", zorder=0)
ax4.axhline(0, color="k", lw=0.6)
ax4.set_xlabel("groove depth [mm]")
ax4.set_ylabel("rise [%]")
ax4.set_title("(d) disk-mode rise, outer band xi 0.9 o / 0.85 ^ / 0.95 v", fontsize=8, loc="left")
fig.text(0.36, 0.11, "grey band in (c), (d) = +/-0.15 %,\nthe detection threshold (3 x 0.05 % baseline scatter)", fontsize=7.5)
fig.savefig(OUT + "s2l_plan.png", dpi=170)
print("saved")
