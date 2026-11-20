"""흠집 시험 가공 카드(40·42번, 10-31 채택안 → 11-16 줄인 순서) — 어디를 얼마나 깎는지 한 장에.

11-16(47번 뒤): L1a·L1을 건너뛰고 A2b(가짜 가공) → L6(깃 사이 여섯 창 한 번에) → R(깃 위 여섯) → C로 간다.
(a) 앞판 정면도(앞에서 본 모습): B 띠의 창 12개(깃 사이 6 = L6, 깃 위 6 = R)와 C 고리, 깃 선(용접 자국)
(b) 반지름 단면(깊이 과장)  (c) 순서표  (d) B 띠를 둘레로 편 그림(창 순서)
치수는 geometry.ASBUILT(G3)·mask_template.py와 같다. 깃 선은 FEM 캠버를 측정 깃 번호 규약(28번 거울상)으로 옮긴 것이다:
앞에서 보면 깃 번호는 반시계 방향으로 늘고, 깃은 바깥으로 갈수록 반시계 방향으로 휜다. 깃 1 끝(테두리)을 12시에 둔다.
A4 가로로 인쇄해 작업대에 둔다.
사용: python groove_card.py → groove_card.pdf(인쇄용)·groove_card.png(미리 보기)"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["pdf.fonttype"] = 3            # CJK OTF는 Type 42로 묻으면 깨진다
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
A, B = 14.89, 36.465                        # 흡입구 반경, 외경 반지름 [mm]
EXT = B - A
RB_IN, RB_C = A + 0.8 * EXT, A + 0.9 * EXT   # 32.145, 34.31
RC_IN, RC_OUT = A + 0.2 * EXT, A + 0.4 * EXT   # 19.20, 23.52
R_COLLAR, R_BORE, R_VIN, WRAP, TH_HIT0 = 30.82 / 2, 12.62 / 2, 15.0, 62.0, -43.1
T_F, T_B, GAP, H_COL = 0.40, 0.60, 3.00, 5.0
RHO = 7.93e-3                                # g/mm³
MB = 1e3 * RHO * math.pi * (B ** 2 - RB_IN ** 2) * 0.01      # B 고리 mg / 0.01 mm
MW = MB / 12                                                   # 창 하나(30°) mg / 0.01 mm
MC = 1e3 * RHO * math.pi * (RC_OUT ** 2 - RC_IN ** 2) * 0.01
ORANGE, LBLUE, RED, BLUE = "#fdae6b", "#9ecae1", "#d6604d", "#4393c3"


def cam_deg(r):
    u = np.log(np.maximum(r, R_VIN) / R_VIN) / math.log(B / R_VIN)
    return -WRAP * u


def vane_meas(r, k):
    return -(cam_deg(r) - TH_HIT0) + 60.0 * (k - 1)


ROT = 90.0 - float(vane_meas(B, 1))


def ang(r, k, rel=0.0):
    return float(vane_meas(r, k)) + rel + ROT


fig = plt.figure(figsize=(297 / 25.4, 210 / 25.4))
fig.suptitle("흠집 시험 가공 카드 — 2번 시편 · 앞판(윗판, 흡입구 쪽) 바깥면 · B는 깃 사이 여섯(L6) → 깃 위 여섯(R), C는 고리 · 11-16 줄인 순서",
             fontsize=12, weight="bold", x=0.02, ha="left", y=0.975)

# (a) 정면도
ax = fig.add_axes([0.005, 0.27, 0.40, 0.66])
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-47, 47); ax.set_ylim(-45, 47)
ax.add_patch(Circle((0, 0), B, facecolor="0.94", edgecolor="k", lw=1.0))
for k in range(1, 7):
    c_v, c_m = ang(RB_C, k), ang(RB_C, k, 30.0)
    ax.add_patch(Wedge((0, 0), B, c_v - 15, c_v + 15, width=B - RB_IN, facecolor=LBLUE, edgecolor="k", lw=0.5))
    ax.add_patch(Wedge((0, 0), B, c_m - 15, c_m + 15, width=B - RB_IN, facecolor=ORANGE, edgecolor="k", lw=0.5))
    t = math.radians(c_m)
    ax.text(0.5 * (RB_IN + B) * math.cos(t), 0.5 * (RB_IN + B) * math.sin(t), "L6", fontsize=6.5,
            ha="center", va="center", weight="bold", rotation=c_m - 90)
    t = math.radians(c_v)
    ax.text(0.5 * (RB_IN + B) * math.cos(t), 0.5 * (RB_IN + B) * math.sin(t), "R", fontsize=6.5, ha="center", va="center",
            weight="bold", rotation=c_v - 90)
ax.add_patch(Wedge((0, 0), RC_OUT, 0, 360, width=RC_OUT - RC_IN, facecolor=BLUE, alpha=0.85, edgecolor="k", lw=0.6))
ax.add_patch(Circle((0, 0), R_COLLAR, facecolor="0.80", edgecolor="k", lw=0.8))
ax.add_patch(Circle((0, 0), A, facecolor="white", edgecolor="k", lw=0.6))
ax.add_patch(Circle((0, 0), R_BORE, facecolor="0.55", edgecolor="k", lw=0.5))
rr = np.linspace(R_VIN, B, 60)
for k in range(1, 7):
    t = np.radians([ang(r, k) for r in rr])
    ax.plot(rr * np.cos(t), rr * np.sin(t), color="0.25", lw=0.9, ls=(0, (3, 2)))
    tl = math.radians(ang(B, k))
    ax.text((B + 3.6) * math.cos(tl), (B + 3.6) * math.sin(tl), f"깃 {k}", fontsize=8.5, ha="center", va="center",
            weight="bold")
ax.text(0, 44.5, "(a) 앞판 정면(흡입구 쪽에서 본 모습) — 깃 번호 반시계", ha="center", fontsize=10, weight="bold")
ax.text(-46, -41.5, "주황 = 깃 사이 창 여섯(L6, 한 번에)\n파랑 = 깃 위 창 여섯(R)   진파랑 = C 고리\n점선 = 깃 용접 자국(깃 선)",
        fontsize=7.5, va="bottom")
ax.text(46, -41.5, f"B 창: Ø{2 * RB_IN:.1f} mm 원 바깥 ~ 테두리 끝\n창 중심 ± 15°(r {RB_C:.2f} mm의 깃 선 기준)\n"
        f"C 고리: Ø{2 * RC_IN:.1f}–{2 * RC_OUT:.1f} mm", fontsize=7.5, va="bottom", ha="right")

# (b) 반지름 단면 — 판 두께와 깊이는 같은 배율(10배)
bx = fig.add_axes([0.43, 0.56, 0.56, 0.36])
VP, VG, VC = 10.0, 2.0, 2.0
yb1 = T_B * VP; yf0 = yb1 + GAP * VG; yf1 = yf0 + T_F * VP; ytop = yf1 + H_COL * VC
bx.add_patch(Rectangle((R_BORE, 0), B - R_BORE, yb1, fc="0.75", ec="k", lw=0.7))
bx.add_patch(Rectangle((R_VIN, yb1), B - R_VIN, yf0 - yb1, fc="none", ec="0.55", lw=0.6, hatch="////"))
bx.add_patch(Rectangle((A, yf0), B - A, yf1 - yf0, fc="0.85", ec="k", lw=0.7))
bx.add_patch(Rectangle((A, yf0), R_COLLAR - A, ytop - yf0, fc="0.85", ec="k", lw=0.7))
bx.add_patch(Rectangle((RB_IN, yf1 - 0.06 * VP), B - RB_IN, 0.06 * VP, fc=ORANGE, ec="k", lw=0.4))
for i in (1, 2):
    bx.add_patch(Rectangle((RC_IN, yf1 - 0.02 * i * VP), RC_OUT - RC_IN, 0.02 * VP, fc=BLUE, alpha=0.95 - 0.3 * (i - 1),
                           ec="k", lw=0.3))
bx.plot([0, 0], [-1, ytop + 1], color="k", lw=0.6, ls="-.")
bx.text(0.4, ytop + 0.3, "회전축", fontsize=8)
bx.text(R_BORE + 0.5, -1.0, f"뒤판 {T_B:.2f} mm — 해머 치는 면, 건드리지 않음", fontsize=8, va="top")
bx.text(27.5, (yb1 + yf0) / 2, f"판 사이 통로 {GAP:.2f} mm(깃)", fontsize=8, va="center", ha="center",
        bbox=dict(fc="white", ec="none", pad=1))
bx.text((A + RC_IN) / 2 + 0.6, yf1 + 0.5, f"앞판\n{T_F:.2f} mm", fontsize=8, ha="center")
bx.text(R_COLLAR + 0.6, ytop - 1.5, "흡입구 테", fontsize=8)
bx.text((RB_IN + B) / 2, yf1 + 0.6, "B 창 12개\n각 0.06 mm(판의 15 %)\n깃 사이 여섯 → 깃 위 여섯", ha="center", fontsize=8,
        color="#8b1a1a", weight="bold")
bx.text((RC_IN + RC_OUT) / 2, yf1 + 0.6, "C 고리\n0.02 → 0.04 mm\n(5 → 10 %)", ha="center", fontsize=8, color="#08519c",
        weight="bold")
for x in (RC_IN, RC_OUT, RB_IN, B):
    bx.plot([x, x], [-3.6, yf1], color="0.6", lw=0.4, ls=":")
    bx.text(x, -3.8, f"r {x:.2f}", fontsize=7, ha="center", rotation=90, va="top")
bx.set_xlim(-1, B + 1.5); bx.set_ylim(-9.5, ytop + 2.2)
bx.axis("off")
bx.text(-1, ytop + 2.4, "(b) 반지름 방향 단면(축에서 테두리까지)", fontsize=10, weight="bold")
bx.text(B + 1.5, -9.3, f"세로 과장: 판 두께·깊이 {VP:.0f}배(같은 배율), 통로·테 높이 {VG:.0f}배.", fontsize=7.5,
        ha="right", color="0.3")

# (d) B 띠를 둘레로 편 그림(깃 1부터 깃 번호 방향)
dx = fig.add_axes([0.03, 0.05, 0.36, 0.17])
for k in range(6):
    x_v = 60 * k
    dx.add_patch(Rectangle((x_v - 15, 0), 30, 1, fc=LBLUE, ec="k", lw=0.5))
    dx.add_patch(Rectangle((x_v + 15, 0), 30, 1, fc=ORANGE, ec="k", lw=0.5))
    dx.text(x_v, 0.5, "R", ha="center", va="center", fontsize=8, weight="bold")
    dx.text(x_v + 30, 0.5, "L6", ha="center", va="center", fontsize=8, weight="bold")
    dx.plot([x_v, x_v], [-0.15, 1.15], color="0.25", lw=1.0, ls=(0, (3, 2)))
    dx.text(x_v, 1.25, f"깃 {k + 1}", ha="center", fontsize=8, weight="bold")
dx.set_xlim(-18, 348); dx.set_ylim(-0.6, 1.9); dx.axis("off")
dx.text(-18, 2.0, "(d) B 띠를 둘레로 편 그림 — 깃 1부터 깃 번호 방향, 창 하나 30°", fontsize=9, weight="bold")
dx.text(-18, -0.55, f"창 하나 0.01 mm = {MW:.2f} mg, 0.06 mm = {6 * MW:.1f} mg. 깃 사이 여섯 = 고리의 절반.", fontsize=7.5)

# (c) 순서표
cx = fig.add_axes([0.43, 0.03, 0.56, 0.50]); cx.axis("off"); cx.set_xlim(0, 1); cx.set_ylim(0, 1)
cx.text(0, 1.0, "(c) 순서 — 깎고 → 세척·건조·질량·단차·사진 → 매달아 측정 → 내렸다 다시 매달아 한 번 더", fontsize=10,
        weight="bold", va="top")
W6 = 36 * MW                                                   # 여섯 창 0.06 mm
rows = [("상태", "어디를", "깊이", "깎은 질량(누적)", "측정·비고"),
        ("A1·T·M·A2", "깎지 않음(테이프 시험)", "—", "0", "끝남(43–47번)"),
        ("A2b", "가짜 가공(전류 없이)", "—", "0 → 이 질량을 A 행에", "주 측정 + 다시 매달기, 취급 영향 확인"),
        ("L6", "깃 사이 여섯 창", "0.06 mm", f"{W6:.0f} mg(창마다 {6 * MW:.1f})",
         f"70 % = {0.7 * W6:.0f} mg에서 잼, 허용 {0.85 * W6:.0f}–{1.15 * W6:.0f}"),
        ("R", "깃 위 여섯 창", "0.06 mm", f"{2 * W6:.0f} mg(= B 고리)", f"증분 {W6:.0f} mg, 깃 사이는 막음"),
        ("C1", "C 고리(B는 덮음)", "0.02 mm", f"{2 * W6:.0f} + {2 * MC:.0f} mg", "주 측정 + 다시 매달기"),
        ("C2", "C 고리", "0.04 mm", f"{2 * W6:.0f} + {4 * MC:.0f} mg", "같음")]
xs = [0.0, 0.13, 0.36, 0.45, 0.66]
for i, row in enumerate(rows):
    y = 0.90 - 0.063 * i
    for j, (x, s) in enumerate(zip(xs, row)):
        col = "#8b1a1a" if row[0][0] in "LR" and j == 1 else "#08519c" if row[0].startswith("C") and j == 1 else "k"
        cx.text(x, y, s, fontsize=8.3, va="top", weight="bold" if i == 0 else "normal", color=col)
    if i == 0:
        cx.plot([0, 1], [y - 0.045, y - 0.045], color="k", lw=0.6, scalex=False, scaley=False)
rules = ["• 창 밖은 마스크로 가린다(mask_template.pdf 1쪽 ①: L6은 깃 사이 여섯 창을 모두 연다, ②: R, 3쪽: C). 옆면은 래커.",
         "• 마스크의 깃 선(점선)을 앞판의 깃 용접 자국에 맞춘다. 깃 번호는 뒤판 표시에서 출구 쪽 깃 끝을 따라 옮긴다.",
         f"• 깊이는 질량으로 잰다: 창 하나 0.01 mm = {MW:.2f} mg, C 고리 0.01 mm = {MC:.1f} mg. 단계마다 증분의 약 70 %에서",
         "  멈추고 잰 뒤 나머지를 깎는다(목표 ±15 %). 깎은 질량은 분석에서 예측을 비례 조정한다.",
         "• 창마다 같은 전하량을 준다(먼저 같은 판재 시험편으로 mg/C를 잰다). 질량은 합만 알려 주므로 창마다 단차를 잰다.",
         "• 단차: 창 안쪽 경계(r 32.15 mm) 가운데, 레버식 다이얼 게이지 ≤ 0.1 N. 깃 위 창은 용접부를 지난다.",
         "• 가공 직전 질량(A2b 뒤)을 mass_log.csv의 A 행에 적는다. 헹굼·IPA·건조는 매번 A2b와 똑같이 한다(47번)."]
for i, s in enumerate(rules):
    cx.text(0, 0.42 - 0.052 * i, s, fontsize=7.8, va="top")

fig.savefig(HERE / "groove_card.pdf")
fig.savefig(HERE / "groove_card.png", dpi=120)
print(HERE / "groove_card.pdf")
