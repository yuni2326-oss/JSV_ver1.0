"""국부 손상 시험 배치도(42번) — 2번 시편 앞판을 앞에서 본 모습: 깃 1–6, B 띠의 30° 창 12개(깃 사이 6·깃 위 6),
첫 창 L1(깃 3–4 사이), 테이프 자리 T-a·T-b·T-c, B 반지름에서 측정 h2 짝 두 성분의 배 자리.

각도 규약 = 측정 규약(깃 v의 ξ0.6 타격점 = 60°(v − 1), 깃 번호 방향이 +). 뒤에서 보면 깃 번호가 시계 방향이므로
앞에서 보면 반시계 방향이다 → 이 그림은 반시계 방향을 +로 그린다(앞에서 본 모습).
FEM과의 대응: 측정 각 = −(FEM 각 − 깃 1 ξ0.6 타격점 FEM 각 −43.1°)(s3_orient.py).
사용: python s3_local_plan_fig.py → groove-test/s3-local-plan.png"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
A, B_, RVI, WRAP = 14.89, 36.465, 15.0, 62.0
TH_HIT0 = -43.1                                   # 측정 깃 1 ξ0.6 타격점의 FEM 각 [°]


def cam_deg(r):
    u = np.log(np.maximum(r, RVI) / RVI) / math.log(B_ / RVI)
    return -WRAP * u


def meas(theta_fem):
    return -(theta_fem - TH_HIT0)


def vane_line(r, k):
    """측정 규약 깃 k(1–6)의 선 각 [°]."""
    return meas(cam_deg(r)) + 60.0 * (k - 1)


def xy(r, th_deg):
    t = np.radians(th_deg)
    return r * np.cos(t), r * np.sin(t)


xi = lambda x: A + x * (B_ - A)
rB0, rB1, rBc = xi(0.8), B_, xi(0.9)
rC0, rC1 = xi(0.2), xi(0.4)
fig, ax = plt.subplots(figsize=(8.6, 8.6))
for r_, ls in ((A, "-"), (B_, "-"), (rB0, ":"), (rC0, ":"), (rC1, ":")):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r_ * np.cos(t), r_ * np.sin(t), "k" + ls, lw=0.8 if ls == "-" else 0.6)
# B 띠 창 12개
for k in range(1, 7):
    c_vane = vane_line(rBc, k)
    c_mid = c_vane + 30.0
    ax.add_patch(Wedge((0, 0), rB1, c_vane - 15, c_vane + 15, width=rB1 - rB0, fc="#9ecae1", ec="k", lw=0.5, alpha=0.8))
    ax.add_patch(Wedge((0, 0), rB1, c_mid - 15, c_mid + 15, width=rB1 - rB0,
                       fc="#fdae6b", ec="k", lw=0.5, alpha=0.8))
c_L1 = vane_line(rBc, 3) + 30.0
ax.add_patch(Wedge((0, 0), rB1, c_L1 - 15, c_L1 + 15, width=rB1 - rB0, fc="none", ec="C3", lw=2.4, hatch="///"))
# C 띠(고리)
ax.add_patch(Wedge((0, 0), rC1, 0, 360, width=rC1 - rC0, fc="#c7e9c0", ec="none", alpha=0.5))
# 깃
rr = np.linspace(RVI, B_, 60)
for k in range(1, 7):
    x, y = xy(rr, vane_line(rr, k))
    ax.plot(x, y, "k-", lw=2.2)
    tx, ty = xy(B_ + 2.6, vane_line(B_, k))
    ax.text(tx, ty, f"깃 {k}", ha="center", va="center", fontsize=10, fontweight="bold")
# 측정 h2 짝 배 자리(B 반지름, s3_orient.py): 2134 = 30.5° + 90°j, 2142 = 76.2° + 90°j
for j in range(4):
    for th0, col, mk in ((30.5, "C0", "^"), (76.2, "C1", "s")):
        x, y = xy(rBc, th0 + 90 * j)
        ax.plot(x, y, mk, color=col, ms=8, mec="k", zorder=5)
# 테이프 자리(테두리 ξ0.9–1.0 가운데)
rT = xi(0.95)
taps = {"T-a 30 mg": vane_line(rT, 2), "T-b 30 mg": vane_line(rT, 3) - 15.0, "T-c 10 mg": vane_line(rT, 3) + 30.0}
for name, th in taps.items():
    x, y = xy(rT, th)
    ax.plot(x, y, "*", color="k", ms=15, mfc="yellow", zorder=6)
    tx, ty = xy(B_ + 7.5, th)
    ax.annotate(name, (x, y), (tx, ty), fontsize=9, ha="center", arrowprops=dict(arrowstyle="-", lw=0.6))
h = [plt.Rectangle((0, 0), 1, 1, fc="#fdae6b", ec="k", lw=0.5), plt.Rectangle((0, 0), 1, 1, fc="#9ecae1", ec="k", lw=0.5),
     plt.Rectangle((0, 0), 1, 1, fc="none", ec="C3", lw=2, hatch="///"), plt.Rectangle((0, 0), 1, 1, fc="#c7e9c0", alpha=0.5),
     plt.Line2D([], [], ls="", marker="^", color="C0", mec="k", ms=8), plt.Line2D([], [], ls="", marker="s", color="C1", mec="k", ms=8),
     plt.Line2D([], [], ls="", marker="*", mfc="yellow", color="k", ms=13)]
ax.legend(h, ["깃 사이 창 6 (L1 + L6)", "깃 위 창 6 (R: 고리 완성)", "첫 창 L1 (깃 3–4 사이)", "C 고리 ξ0.2–0.4 (계획 그대로)",
              "2134 Hz 성분의 배 (B)", "2142 Hz 성분의 배 (B)", "테이프 자리 (깎기 전)"],
          loc="lower left", fontsize=8.5, framealpha=0.95)
ax.set_aspect("equal")
lim = B_ + 11
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.axis("off")
ax.set_title("2번 시편 앞판, 앞에서 본 모습 — B 띠(ξ0.8–1.0) 30° 창 12개와 테이프 자리\n"
             "깃 선은 FEM 캠버(감김 62°)·측정 깃 번호 규약(거울상 보정). 창 경계는 B 띠 가운데(ξ0.9) 깃 선 기준 ±15°", fontsize=10)
fig.tight_layout()
fig.savefig(HERE.parents[0] / "groove-test" / "s3-local-plan.png", dpi=130)
print(HERE.parents[0] / "groove-test" / "s3-local-plan.png")
for k in range(1, 7):
    print(f"깃 {k}: 선 각(측정 규약) ξ0.9 {vane_line(rBc, k) % 360:.1f}°, 테두리 {vane_line(B_, k) % 360:.1f}°")
