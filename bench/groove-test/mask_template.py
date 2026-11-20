"""흠집 시험 마스킹 도안(42번 채택안, 10-31) — A4 1:1로 인쇄해 투명 마스크 필름(캡톤·비닐)을 오린다.

쪽 1  B 창 마스크 두 장(앞에서 본 모습): ① 깃 사이 창 6개(L6 — 11-16 줄인 B는 여섯 창을 한 번에, 11-19 1번 시편 사전 시험은
      깃 2–3 창 하나), ② 깃 위 창 6개(R). 창 = ξ0.8–1.0(r 32.15 mm–테두리),
      폭 30°(창 중심 ± 15°), 중심 = ξ0.9(r 34.31 mm)의 깃 선(깃 위) 또는 거기서 깃 번호 방향 30°(깃 사이).
쪽 2  테이프 자리 안내(1:1): T-a·T-b·T-c와 고리 테이프 M75·M150(r 33.2·35.4 mm, 칸의 1/3·2/3).
쪽 3  C 마스크 두 장(안쪽 고리 r 19.20–23.52 mm, 40번과 같다).
깃 선 = FEM 캠버(로그 나선, 감김 62°, 깃 안끝 r 15.0 mm)를 측정 깃 번호 규약(28번 거울상)으로 옮긴 것이다. 앞에서 보면 깃 번호는
반시계 방향으로 늘고, 깃은 바깥으로 갈수록 반시계 방향으로 휜다. 그림은 깃 1의 끝(테두리)이 12시에 오게 돌렸다.
사용: python mask_template.py → mask_template.pdf(3쪽, 벡터)·mask_template_p{1,2,3}.png(미리 보기)"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Wedge

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["pdf.fonttype"] = 3            # CJK OTF는 Type 42로 묻으면 깨진다
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
A, B = 14.89, 36.465
EXT = B - A
W = 0.2 * EXT
RB_IN = A + 0.8 * EXT                        # 32.145
RB_C = A + 0.9 * EXT                         # 34.31
RC_IN, RC_OUT = A + 0.2 * EXT, A + 0.4 * EXT   # 19.20, 23.52
R_COLLAR, R_HOLE, R_RIM = 30.82 / 2, 32.0 / 2, 72.93 / 2
R_OUT = R_RIM + 3.0
RVI, WRAP, TH_HIT0 = 15.0, 62.0, -43.1


def cam_deg(r):
    u = np.log(np.maximum(r, RVI) / RVI) / math.log(B / RVI)
    return -WRAP * u


def vane_meas(r, k):
    """측정 규약(깃 번호 방향 +) 깃 k의 선 각 [°]."""
    return -(cam_deg(r) - TH_HIT0) + 60.0 * (k - 1)


ROT = 90.0 - float(vane_meas(B, 1))          # 깃 1 끝을 12시로


def ang(r, k, rel=0.0):
    """종이 위 각[°] — 앞에서 본 모습(반시계 +)."""
    return float(vane_meas(r, k)) + rel + ROT


def page():
    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 210); ax.set_ylim(0, 297); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def scale_bar(ax, foot):
    ax.plot([20, 120], [22, 22], "k", lw=1.2)
    for x in range(20, 121, 10):
        ax.plot([x, x], [22, 24 if (x - 20) % 50 else 26], "k", lw=0.8)
    ax.text(20, 28, "100 mm — 자로 재서 정확히 100 mm인지 확인(인쇄 '실제 크기', 맞춤 끔)", fontsize=7.5)
    ax.text(20, 12, foot, fontsize=6.5, color="0.4")


def vanes(ax, cx, cy, lw=0.7, label=True, r_lab=None):
    rr = np.linspace(RVI, B, 50)
    for k in range(1, 7):
        t = np.radians([ang(r, k) for r in rr])
        ax.plot(cx + rr * np.cos(t), cy + rr * np.sin(t), color="C3", ls="--", lw=lw)
        if label:
            rl = r_lab or (R_OUT + 3.5)
            tl = math.radians(ang(B, k))
            ax.text(cx + rl * math.cos(tl), cy + rl * math.sin(tl), f"깃 {k}", fontsize=7, color="C3",
                    ha="center", va="center", weight="bold")


def window_mask(ax, cx, cy, rel, title, note, labels):
    """r_hole–r_out 고리에서 B 띠 창 6개를 뚫은 마스크(회색 = 남기는 부분, 흰색 = 오려 낼 창)."""
    ax.add_patch(Wedge((cx, cy), R_OUT, 0, 360, width=R_OUT - R_HOLE, facecolor="0.82", edgecolor="k", lw=0.8))
    for k in range(1, 7):
        c = ang(RB_C, k, rel)
        ax.add_patch(Wedge((cx, cy), R_OUT + 0.01, c - 15, c + 15, width=R_OUT + 0.01 - RB_IN, facecolor="white",
                           edgecolor="k", lw=0.8))
        lab = labels.get(k)
        if lab:
            t = math.radians(c)
            ax.text(cx + 0.5 * (RB_IN + B) * math.cos(t), cy + 0.5 * (RB_IN + B) * math.sin(t), lab, fontsize=6.2,
                    ha="center", va="center", rotation=c - 90, weight="bold")
    ax.add_patch(Circle((cx, cy), R_RIM, fill=False, ls="-", lw=0.5, color="0.3"))
    ax.add_patch(Circle((cx, cy), R_COLLAR, fill=False, ls=":", lw=0.5, color="0.3"))
    vanes(ax, cx, cy)
    ax.plot([cx - 3, cx + 3], [cy, cy], "k", lw=0.4); ax.plot([cx, cx], [cy - 3, cy + 3], "k", lw=0.4)
    ax.text(cx - 48, cy + R_OUT + 13, title, fontsize=10, weight="bold", va="bottom")
    ax.text(cx - 48, cy + R_OUT + 8.5, note, fontsize=6.8, va="bottom")


pdf = PdfPages(HERE / "mask_template.pdf")
# ---------------------------------------------------------------- 쪽 1: B 창 마스크
fig, ax = page()
window_mask(ax, 62, 205, 30.0, "B ① 깃 사이 창 6개 — L6(줄인 B)",
            "L6: 여섯 창을 모두 연다(11-16 줄인 B). 1번 시편 사전 시험(11-19): 깃 2–3 창(S1)만 열고 다섯 창은 막는다.",
            {k: ("L6·S1" if k == 2 else "L6") for k in range(1, 7)})
window_mask(ax, 62, 92, 0.0, "B ② 깃 위 창 6개 — R(고리 완성)", "여섯 창을 모두 연다. 깃 사이(이미 깎은 곳)는 덮인다.",
            {k: "R" for k in range(1, 7)})
txt = ["창(앞판 바깥면, 42번)",
       f"반지름: r {RB_IN:.2f} mm(Ø {2 * RB_IN:.2f}) ~ 테두리 끝",
       f"폭: 창 중심 ± 15°(30°). 중심은 r {RB_C:.2f} mm의 깃 선",
       "  (깃 위 창) 또는 거기서 깃 번호 방향 30°(깃 사이 창).",
       "깊이 0.06 mm. 창 하나 약 37 mg(0.01 mm당 6.2 mg).",
       "줄인 B(11-16): A2b 가짜 가공 → L6(깃 사이 여섯) → R(깃 위 여섯).",
       "",
       "맞추는 법",
       "· 빨강 점선 = 깃 선(앞판의 깃 용접 자국 점선과 같은 선).",
       "· 투명 필름에 옮겨 오린 뒤, 점선을 용접 자국에 겹친다.",
       "· 깃 번호는 뒤판 표시에서 출구 쪽 깃 끝을 따라 옮긴다.",
       "  앞에서 보면 번호가 반시계 방향으로 는다.",
       "· 테두리 옆면(판 두께면)은 래커로 가린다.",
       "· 창 가장자리 단차를 잴 수 있게 경계를 곧게 오린다."]
for i, s in enumerate(txt):
    ax.text(118, 165 - 5.2 * i, s, fontsize=7.0, weight="bold" if s in ("창(앞판 바깥면, 42번)", "맞추는 법") else None)
scale_bar(ax, "10-31 흠집 시험 · 40·42번 · 11-19 개정(줄인 B, 1번 시편 사전 시험) · 회색 = 남길 마스크 · 흰 부채꼴 = 오려 낼 창 · 빨강 점선 = 깃 선 · 앞에서 본 모습")
pdf.savefig(fig); fig.savefig(HERE / "mask_template_p1.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- 쪽 2: 테이프 자리
fig, ax = page()
cx, cy = 105, 190
ax.add_patch(Circle((cx, cy), R_RIM, fill=False, lw=0.9, color="k"))
ax.add_patch(Circle((cx, cy), R_COLLAR, fill=False, ls=":", lw=0.6, color="0.3"))
for r_, ls in ((RB_IN, ":"), (33.2, "-"), (35.4, "-")):
    ax.add_patch(Circle((cx, cy), r_, fill=False, ls=ls, lw=0.4, color="0.55"))
vanes(ax, cx, cy, lw=0.9, r_lab=R_RIM + 5)
r_T = A + 0.95 * EXT
TA = {"T-a 30 mg\n(깃 2 위)": ang(r_T, 2), "T-b 30 mg\n(깃 3에서 깃 2 쪽 14°)": ang(r_T, 3, -14.1),
      "T-c 10 mg\n(깃 3–4 가운데)": ang(r_T, 3, 30.0)}
for lab, a_ in TA.items():
    t = math.radians(a_)
    x, y = cx + r_T * math.cos(t), cy + r_T * math.sin(t)
    ax.plot(x, y, "*", ms=11, mfc="yellow", mec="k", zorder=5)
    xo, yo = cx + (R_RIM + 17) * math.cos(t), cy + (R_RIM + 17) * math.sin(t)
    ax.annotate(lab, (x, y), (xo, yo), fontsize=6.8, ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.5))
for r_ in (33.2, 35.4):
    for k in range(1, 7):
        for rel, mk in ((20.0, "o"), (40.0, "s")):
            t = math.radians(ang(r_, k, rel))
            ax.plot(cx + r_ * math.cos(t), cy + r_ * math.sin(t), mk, ms=3.2, mfc="C0" if rel == 20 else "C1", mec="k", mew=0.4)
ax.plot([cx - 3, cx + 3], [cy, cy], "k", lw=0.4); ax.plot([cx, cx], [cy - 3, cy + 3], "k", lw=0.4)
ax.text(18, 275, "테이프 자리 안내(1:1, 앞판 바깥면, 앞에서 본 모습)", fontsize=10, weight="bold")
txt = ["T(한 점 테이프, 깎기 전): 매단 채 붙이고, 1번 깃 ξ0.6 10–20타 → 떼고 다음 자리로.",
       f"  T-a·T-b·T-c: 별 = 붙일 각(r {r_T:.1f} mm). 조각은 바깥 끝을 테두리 끝에 맞춘다(별을 중심으로 두면 4 mm 조각이 밖으로 나간다).",
       "  T-a: 2142 Hz만 약 −0.15 %, T-b: 2134 Hz만 약 −0.21 %, T-c: 2142 Hz만 약 −0.12 %(패널 −0.3~−0.4 %) 예측.",
       "M(고리 테이프, 41번): 칸마다 r 33.2·35.4 mm 두 자리에 6.25 mg 조각.",
       "  M75 = 파란 원(칸의 1/3, 깃 k에서 깃 번호 방향 20°) 12조각.",
       "  M150 = M75 + 주황 네모(칸의 2/3, 40°) 12조각, 합 24조각 150 mg.",
       "조각마다 분석저울로 재서 tape_log.csv에 적는다. 붙이기 전 IPA로 닦고, 뗀 뒤 IPA로 접착제를 지운다.",
       "회색 실선 = r 33.2·35.4 mm, 점선 = r 32.15 mm(B 창 안쪽 경계). 빨강 점선 = 깃 선."]
for i, s in enumerate(txt):
    ax.text(18, 120 - 6.2 * i, s, fontsize=7.2)
scale_bar(ax, "10-31 흠집 시험 · 41·42번 · 테이프 자리 · 깃 1 끝이 12시")
pdf.savefig(fig); fig.savefig(HERE / "mask_template_p2.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- 쪽 3: C 마스크(40번과 같다)
fig, ax = page()


def stencil(cx, cy, r_out, r_in, title, note):
    ax.add_patch(Wedge((cx, cy), r_out, 0, 360, width=r_out - r_in, facecolor="0.85", edgecolor="k", lw=0.8))
    for r, ls in ((R_RIM, "--"), (R_COLLAR, ":")):
        ax.add_patch(Circle((cx, cy), r, fill=False, ls=ls, lw=0.5, color="C3"))
    ax.plot([cx - 3, cx + 3], [cy, cy], "k", lw=0.4); ax.plot([cx, cx], [cy - 3, cy + 3], "k", lw=0.4)
    top = cy + max(r_out, R_RIM)
    ax.text(cx - 44, top + 7, title, fontsize=10, weight="bold", va="bottom")
    ax.text(cx - 44, top + 2.5, note, fontsize=6.8, va="bottom")
    ax.text(cx + r_out + 1, cy, f"Ø {2 * r_out:.2f}", fontsize=6.5, va="center")
    ax.text(cx + r_in * 0.15, cy - r_in - 2.5, f"구멍 Ø {2 * r_in:.2f}", fontsize=6.5, va="top", ha="center")


stencil(58, 200, RC_IN, R_HOLE, "C 마스크 ① 안쪽(r 16.0–19.20)", "C 고리 안쪽 경계를 만든다.")
stencil(150, 200, R_OUT, RC_OUT, "C 마스크 ② 바깥(r 23.52–39.5)", "C 고리 바깥 경계 ~ 테두리 너머(B 창도 모두 덮는다).")
for i, s in enumerate(["C 고리(ξ0.2–0.4, 축대칭): r 19.20–23.52 mm, 폭 4.31 mm",
                       "깊이 0.02 → 0.04 mm(R 뒤), 0.02 mm당 92 mg",
                       "앞판 두께 0.40 mm, SUS304 7.93 g/cm³"]):
    ax.text(25, 130 - 6 * i, s, fontsize=7.5)
scale_bar(ax, "10-31 흠집 시험 · 40번 · 회색 = 오려 낼 마스크 · 빨강 점선 = 임펠러 외곽 참고선")
pdf.savefig(fig); fig.savefig(HERE / "mask_template_p3.png", dpi=110); plt.close(fig)
pdf.close()
print(HERE / "mask_template.pdf")
