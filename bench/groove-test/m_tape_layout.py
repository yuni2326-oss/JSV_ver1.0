"""고리 테이프 M75·M150 자리 설명 그림(11-10) — 40번 §3.2, mask_template.pdf 2쪽과 같은 자리.

(a) 윗판(앞판) 바깥면 전체, 앞에서 본 모습(깃 1 끝이 12시, mask_template과 같은 방향)
(b) 깃 3–4 칸 확대(칸 가운데가 위로 오게 돌림): 두 반지름 × 두 각 = 4조각, 창 경계, 각·호 길이
조각은 실제 크기(2 × 2 mm)로 그렸다. 각은 같은 반지름의 깃 선에서 깃 번호 방향(앞에서 보면 반시계)으로 잰다.
사용: python m_tape_layout.py → m-tape-layout.png"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Wedge

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
A, B = 14.89, 36.465                       # mask_template.py와 같은 기하
EXT = B - A
RB_IN = A + 0.8 * EXT                      # 32.145 (B 창 안쪽 경계)
RB_C = A + 0.9 * EXT                       # 34.31 (창 중심 반지름)
R_COLLAR = 30.82 / 2
RVI, WRAP, TH_HIT0 = 15.0, 62.0, -43.1
R_M = (35.4, 33.2)
COL = {20.0: "#1f77b4", 40.0: "#ff7f0e"}
WIN_MID, WIN_VANE, PLATE = "#fff0b3", "#e3e3e3", "#f5f5f5"
HALF = 1.0                                 # 조각 반폭[mm] (6.25 mg ≈ 2 × 2 mm)


def vane_meas(r, k):
    """측정 규약(깃 번호 방향 +) 깃 k의 선 각[°]."""
    u = math.log(max(r, RVI) / RVI) / math.log(B / RVI)
    return WRAP * u + TH_HIT0 + 60.0 * (k - 1)


ROT_A = 90.0 - vane_meas(B, 1)                      # (a) 깃 1 끝을 12시로
ROT_B = 90.0 - (vane_meas(RB_C, 3) + 30.0)          # (b) 깃 3–4 칸 가운데를 12시로
angle = lambda r, k, rel, rot: vane_meas(r, k) + rel + rot
pol = lambda r, a: (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))


def piece(ax, r, a, color, lw=0.6):
    t = math.radians(a)
    er, et = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    c = r * er
    pts = [c + HALF * (er + et), c + HALF * (er - et), c - HALF * (er + et), c - HALF * (er - et)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="k", lw=lw, zorder=6))


def vane_line(ax, k, rot, r0, r1, **kw):
    rr = np.linspace(r0, r1, 80)
    a = np.radians([angle(r, k, 0.0, rot) for r in rr])
    ax.plot(rr * np.cos(a), rr * np.sin(a), **kw)


def arc(ax, r, a0, a1, **kw):
    t = np.radians(np.linspace(a0, a1, 120))
    ax.plot(r * np.cos(t), r * np.sin(t), **kw)


def radial(ax, a, r0, r1, **kw):
    ax.plot(*zip(pol(r0, a), pol(r1, a)), **kw)


fig = plt.figure(figsize=(9, 16.5))
fig.text(0.5, 0.985, "고리 테이프 M75 → M150 자리 — 윗판(앞판) 바깥면, 앞에서 본 모습", ha="center", va="top",
         fontsize=14, weight="bold")

# ------------------------------------------------------------------ (a) 전체
ax = fig.add_axes([0.06, 0.52, 0.88, 0.44])
ax.set_aspect("equal"); ax.axis("off")
L = B + 9.0
ax.set_xlim(-L, L); ax.set_ylim(-L, L)
ax.add_patch(Circle((0, 0), B, facecolor=PLATE, edgecolor="k", lw=1.2, zorder=0))
for k in range(1, 7):
    for rel, col in ((30.0, WIN_MID), (0.0, WIN_VANE)):
        c = angle(RB_C, k, rel, ROT_A)
        ax.add_patch(Wedge((0, 0), B, c - 15, c + 15, width=B - RB_IN, facecolor=col, edgecolor="0.6", lw=0.4, zorder=1))
ax.add_patch(Circle((0, 0), R_COLLAR, facecolor="white", edgecolor="0.45", lw=0.8, ls=":", zorder=2))
ax.text(0, 0, "흡입구", ha="center", va="center", fontsize=10, color="0.45")
for r_ in R_M:
    ax.add_patch(Circle((0, 0), r_, fill=False, color="0.45", lw=0.5, zorder=3))
for k in range(1, 7):
    vane_line(ax, k, ROT_A, RVI, B, color="C3", ls="--", lw=1.0, zorder=4)
    ax.text(*pol(B + 4.6, angle(B, k, 0.0, ROT_A)), f"깃 {k}", color="C3", fontsize=11, weight="bold",
            ha="center", va="center")
    ax.text(*pol(25.0, angle(25.0, k, 30.0, ROT_A)), f"칸 {k}", color="0.3", fontsize=9.5, ha="center", va="center")
    for r_ in R_M:
        for rel, col in COL.items():
            piece(ax, r_, angle(r_, k, rel, ROT_A), col)
a3, a4 = angle(RB_C, 3, 0.0, ROT_A), angle(RB_C, 4, 0.0, ROT_A)
ax.add_patch(Wedge((0, 0), B + 1.6, a3 - 5, a4 + 5, width=B + 1.6 - (RB_IN - 2.2), fill=False, ec="k", lw=1.1,
                   ls=(0, (4, 2)), zorder=7))
ax.text(*pol(B + 6.5, 0.5 * (a3 + a4)), "(b)에서 확대", fontsize=10, ha="center", va="center", weight="bold")
ax.text(-L, L - 1, "(a) 전체 — 24조각 = 6칸 × 2반지름 × 2각", fontsize=11.5, weight="bold", va="top")
ax.text(-L, L - 6.2, "칸 k = 깃 k와 깃 k+1 사이(칸 6은 깃 6–1)", fontsize=9, va="top", color="0.3")
handles = [Patch(facecolor=COL[20.0], edgecolor="k", label="M75: 각 칸, 깃에서 20° 자리 × r 35.4·33.2 = 12조각(75 mg)"),
           Patch(facecolor=COL[40.0], edgecolor="k", label="M150: M75에 더해 깃에서 40° 자리 12조각(합 24조각, 150 mg)"),
           Patch(facecolor=WIN_MID, edgecolor="0.6", label="깃 사이 창(나중에 L1·L6로 깎는 곳)"),
           Patch(facecolor=WIN_VANE, edgecolor="0.6", label="깃 위 창(나중에 R로 깎는 곳)"),
           Line2D([], [], color="C3", ls="--", label="깃 선(앞판의 깃 용접 자국)"),
           Line2D([], [], color="0.45", lw=0.8, label="r 35.4·33.2 mm 고리(조각 중심)")]
fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.495), ncol=1, fontsize=9.5, frameon=False)

# ------------------------------------------------------------------ (b) 깃 3–4 칸 확대
ax = fig.add_axes([0.02, 0.19, 0.96, 0.24])
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-34, 34); ax.set_ylim(17.0, 47.0)
ax.add_patch(Wedge((0, 0), B, 15, 165, width=B - R_COLLAR, facecolor=PLATE, edgecolor="none", zorder=0))
for c, col in ((90.0, WIN_MID), (angle(RB_C, 3, 0.0, ROT_B), WIN_VANE), (angle(RB_C, 4, 0.0, ROT_B), WIN_VANE)):
    ax.add_patch(Wedge((0, 0), B, c - 15, c + 15, width=B - RB_IN, facecolor=col, edgecolor="0.55", lw=0.7, zorder=1))
arc(ax, B, 15, 165, color="k", lw=1.4, zorder=5)
for r_ in R_M:
    arc(ax, r_, 20, 160, color="0.45", lw=0.7, zorder=2)
for k in (3, 4):
    vane_line(ax, k, ROT_B, 21.5, B, color="C3", ls="--", lw=1.4, zorder=4)
ax.text(*pol(37.4, angle(B, 3, 0.0, ROT_B) - 8), "깃 3", color="C3", fontsize=11, weight="bold", ha="center", va="center")
ax.text(*pol(37.6, angle(B, 4, 0.0, ROT_B) + 4.5), "깃 4", color="C3", fontsize=11, weight="bold", ha="center", va="center")
for r_ in R_M:
    for rel, col in COL.items():
        piece(ax, r_, angle(r_, 3, rel, ROT_B), col, lw=0.8)
# 각·호 길이: r 35.4는 테두리 바깥, r 33.2는 안쪽에 치수선
for r_, dims in ((35.4, ((20.0, 38.1), (40.0, 41.0))), (33.2, ((20.0, 30.9), (40.0, 27.6)))):
    av = angle(r_, 3, 0.0, ROT_B)
    out = r_ > 34
    radial(ax, av, r_, (42.0 if out else 26.6), color="0.25", lw=0.6, zorder=3)
    for rel, rd in dims:
        a1 = av + rel
        radial(ax, a1, r_ + (HALF if out else -HALF), rd + (0.5 if out else -0.5), color="0.25", lw=0.6, zorder=3)
        arc(ax, rd, av, a1, color=COL[rel], lw=1.6, zorder=3)
        for a_ in (av, a1):
            ax.plot(*pol(rd, a_), "o", ms=2.6, color=COL[rel], zorder=3)
        s = f"{rel:.0f}°  (호 {math.radians(rel) * r_:.1f} mm)"
        ax.text(*pol(rd + (0.9 if out else -0.9), av + 0.55 * rel), s, fontsize=9.2, ha="center",
                va="bottom" if out else "top", rotation=av + 0.55 * rel - 90, rotation_mode="anchor",
                color="k", zorder=8)
box = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)
ax.text(*pol(35.4, 92.3), "r 35.4", fontsize=8.6, ha="center", va="center", rotation=2.3, bbox=box, zorder=7)
ax.text(*pol(33.2, 87.8), "r 33.2", fontsize=8.6, ha="center", va="center", rotation=-2.2, bbox=box, zorder=7)
lead = dict(arrowstyle="-", lw=0.6, color="0.3")
ax.annotate("테두리 끝 r 36.47", xy=pol(B, 121), xytext=(-25.5, 40.2), fontsize=9.2, ha="center", va="center",
            arrowprops=lead)
ax.annotate("B 창 안쪽 경계 r 32.15", xy=pol(RB_IN, 133.0), xytext=(-22.0, 18.6), fontsize=9.2, ha="center",
            va="center", arrowprops=lead, bbox=box)
ax.annotate("깃 사이 창(L1 자리) = 깃 3에서 15–45°", xy=pol(32.6, 81.0), xytext=(8.5, 18.6), fontsize=9.2,
            ha="center", va="center", arrowprops=lead)
ax.text(-34, 47.0, "(b) 깃 3–4 칸 확대 — 칸 가운데를 위로 돌림, 다른 다섯 칸도 같다", fontsize=11.5, weight="bold", va="top")
ax.text(-34, 44.9, "r 35.4 고리 = 테두리에서 1.1 mm, r 33.2 고리 = 3.3 mm(조각 중심)", fontsize=9.2, va="top", color="0.25")

# ------------------------------------------------------------------ 순서·자리 잡기
txt = [("붙이는 순서", True),
       ("① M75: 여섯 칸 모두, 깃에서 20° 자리에 r 35.4·33.2 두 조각씩 = 12조각(75 mg). 내려서 붙이고 → 다시 달아 10분 → 주 측정.", False),
       ("② M150: M75 조각은 그대로 두고, 깃에서 40° 자리에 두 조각씩 = 12조각 더(합 24조각, 150 mg). 같은 방식으로 주 측정.", False),
       ("③ 다 떼기 → IPA로 접착제 지우기 → 건조 → 질량(A ± 0.3 mg) → A2.", False),
       ("자리 잡기", True),
       ("· 각은 같은 반지름의 깃 선(용접 자국)에서 깃 번호 방향(앞에서 보면 반시계)으로 잰다. 호 = 그 고리를 따라 잰 길이.", False),
       ("· 깃이 휘어 있어 r 33.2 조각은 r 35.4 조각보다 약 4.5° 깃 3 쪽에 놓인다(반지름 방향으로 나란하지 않다).", False),
       ("· mask_template.pdf 2쪽(1:1)을 100 %로 투명 필름에 인쇄해 깃 선을 용접 자국에 맞추고, 조각 중심을 점으로 표시한다.", False),
       ("· 조각: 납 테이프 6.25 mg(약 2 × 2 mm), 하나씩 달아 tape_log.csv에. r 35.4 조각은 바깥 끝이 테두리 끝과 거의 맞닿는다.", False)]
y = 0.172
for s, bold in txt:
    fig.text(0.04, y, s, fontsize=9.6 if not bold else 10.5, weight="bold" if bold else None, va="top")
    y -= 0.0165 if not bold else 0.0175
out = HERE / "m-tape-layout.png"
fig.savefig(out, dpi=130)
print(out)
