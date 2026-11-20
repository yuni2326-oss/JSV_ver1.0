"""JSV 수정본 실험 절(58번 초안) 그림 — 본문 Fig. E1–E3, 보충 Fig. SE1–SE4.

Fig. E1 시편 배치도(앞판 정면, 실물 방향): (a) 2번 시편 납 조각 자리, (b) 3번 시편 깎은 자리와 타격점,
         (c)·(d) 같은 방향·같은 축척으로 돌린 실제 사진(2번 시편 M240, 3번 시편 두 절삭 뒤)
Fig. E2 2번 시편 질량 시험: (a) 한 점 질량의 h2 짝 성분, (b) 6회 대칭 질량의 측정 대 FEM, (c) 조각 각에 따른 FEM 예측
Fig. E3 3번 시편 깎기: (a) 12자리 평균 스펙트럼(기준·R·L), (b) 바뀐 모드의 측정 이동과 FEM, (c) 섞인 두 모드 합·h2 짝 합
Fig. SE1 3번 시편 기준 스펙트럼 대 2번 시편, Fig. SE2 R 직후 근접 사진, Fig. SE3 측정 바닥(패널 b 측정 순서·c 되돌아간 모드), Fig. SE4 깎기 두 단계의 모든 모드(12자리)
수치는 paper_bench_data.py와 같은 함수에서 온다. 평균 스펙트럼은 NAS 타격에서 한 번 내어 groove-test/s3/paper_spec_cache.npz에 둔다.
사용: python paper_bench_figs.py [--refresh-spectra]"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge, Circle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_bench_data as pdat  # noqa: E402

FIGD = pdat.FIGD
CACHE = os.path.join(pdat.S3, "paper_spec_cache.npz")
MM = 1 / 25.4
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.linewidth": 0.6,
                     "lines.linewidth": 0.9, "savefig.dpi": 300, "pdf.fonttype": 42})
C = {"base": "0.25", "R": "#c0392b", "L": "#1f6fb4", "fem": "k", "M120": "#c0392b", "M240": "#1f6fb4"}

# 기하(G3 정본, 미터 → mm) — 실물은 FEM의 거울상이다(측정 깃 v ↔ FEM 섹터 −(v − 1)). 그림은 실물 앞판을 정면에서 본 방향:
# 깃 v의 선 = 60°(v − 1) − camber(r), 깃 번호는 반시계로 늘고 깃은 바깥으로 갈수록 반시계로 휜다.
A_MM, B_MM, RV_IN = 14.89, 36.465, 15.0
WRAP = 62.0


def camber_deg(r):
    """그 반지름의 깃 선 각(FEM camber, 0 at r_vane_in … −62° at b) — FitSpec.camber와 같은 로그 나선 근사를 쓴다."""
    import s3_l1_cause as lc
    fs = lc.spec().base
    return np.degrees(fs.camber(np.asarray(r, float) * 1e-3))


def vane_angle(v, r):
    return 60.0 * (v - 1) - camber_deg(r)


def xy(r, ang):
    a = np.radians(ang)
    return r * np.cos(a), r * np.sin(a)


def plate(ax, title=None):
    ax.add_patch(Circle((0, 0), B_MM, fill=False, lw=0.8, color="0.2"))
    ax.add_patch(Circle((0, 0), A_MM, fill=False, lw=0.6, color="0.45"))
    ax.add_patch(Circle((0, 0), 6.31, fill=False, lw=0.5, color="0.6"))
    r = np.linspace(RV_IN, B_MM, 80)
    for v in range(1, 7):
        x, y = xy(r, vane_angle(v, r))
        ax.plot(x, y, color="0.35", lw=1.4, solid_capstyle="round")
        xt, yt = xy(B_MM + 3.6, vane_angle(v, B_MM))
        ax.text(xt, yt, f"{v}", ha="center", va="center", fontsize=7, color="0.2")
    ax.set_aspect("equal")
    ax.set_xlim(-43, 43); ax.set_ylim(-43, 43)
    ax.axis("off")


def drive_points(ax):
    for v in range(1, 7):
        for xi, mk in ((0.3, "o"), (0.6, "o")):
            r = A_MM + xi * (B_MM - A_MM)
            x, y = xy(r, vane_angle(v, r))
            ax.plot(x, y, mk, ms=3.2, mfc="white" if xi == 0.3 else "0.2", mec="0.2", mew=0.6, zorder=5)


def patch(ax, v, rel0, rel1, r0, color, alpha=0.55, hatch=None):
    """깃 v 선에서 rel0 → rel1(반시계 +) 사이, r0 … 테두리."""
    rr = np.linspace(r0, B_MM, 12)
    pts = []
    for rel in np.linspace(rel0, rel1, 30):
        pts.append(xy(B_MM, vane_angle(v, B_MM) + rel))
    for r in rr[::-1]:
        pts.append(xy(r, vane_angle(v, r) + rel1))
    for rel in np.linspace(rel1, rel0, 30):
        pts.append(xy(r0, vane_angle(v, r0) + rel))
    for r in rr:
        pts.append(xy(r, vane_angle(v, r) + rel0))
    P = np.array(pts)
    ax.fill(P[:, 0], P[:, 1], color=color, alpha=alpha, lw=0, hatch=hatch)


def fig_e1():
    """(a)·(b) 배치도, (c)·(d) 같은 방향·같은 mm 축척으로 놓은 실제 사진(2번 시편 M240, 3번 시편 두 절삭 뒤)."""
    W, H, S = 160.0, 146.0, 62.0          # 그림 폭·높이, 판 한 칸(mm)
    X = (10.0, 90.0)
    fig = plt.figure(figsize=(W * MM, H * MM))
    axs = [fig.add_axes([x / W, (H - 2.0 - S) / H, S / W, S / H]) for x in X]
    ph = [fig.add_axes([x / W, 2.0 / H, S / W, S / H]) for x in X]
    ax = axs[0]
    plate(ax)
    drive_points(ax)
    # T-a: 깃 2 선 위, r 34.5 / T-b: 깃 3에서 깃 2 쪽 14°(시계 방향 −)
    for (v, rel, lab, dl) in ((2, 0.0, "T-a", -7.0), (3, -14.0, "T-b", 0.0)):
        x, y = xy(34.5, vane_angle(v, 34.5) + rel)
        ax.plot(x, y, "D", ms=4.2, color="#7d3c98", mec="k", mew=0.4, zorder=6)
        xt, yt = xy(41.5, vane_angle(v, 34.5) + rel + dl)
        ax.annotate(lab, (x, y), (xt, yt), fontsize=6.5, ha="center", va="center", color="#7d3c98",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#7d3c98"))
    k_of = lambda bay: int(bay.split("-")[0])
    # 11-20(59번 §3): 조각 자리는 칸 폭으로 60°에 맞춘 각. M120은 두 사진 평균(모형 입력과 같다).
    j, m240, _ = pdat.piece_angles()
    m120 = pd.DataFrame({"bay": j.index, "r_mm": 0.5 * (j.r_mm_1 + j.r_mm_2).values, "ang": 0.5 * (j.from_s_1 + j.from_s_2).values})
    m240 = pd.DataFrame({"bay": m240.index, "r_mm": m240.r_mm.values, "ang": m240.from_s.values})
    for D, mk, col, lab in ((m120, "s", C["M120"], "M120 (6 × 20 mg)"), (m240, "^", C["M240"], "added in M240 (6 × 20 mg)")):
        for _, r in D.iterrows():
            x, y = xy(r.r_mm, vane_angle(k_of(r.bay), r.r_mm) + r.ang)
            ax.plot(x, y, mk, ms=3.6, color=col, mec="k", mew=0.3, zorder=6)
        ax.plot([], [], mk, ms=3.6, color=col, mec="k", mew=0.3, label=lab)
    ax.plot([], [], "D", ms=4.2, color="#7d3c98", mec="k", mew=0.4, label="T-a, T-b (30 mg each)")
    ax.plot([], [], "o", ms=3.2, mfc="white", mec="0.2", label="driving points, ξ = 0.3")
    ax.plot([], [], "o", ms=3.2, mfc="0.2", mec="0.2", label="driving points, ξ = 0.6")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 0.0), ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8)
    ax.text(-43, 41, "(a)", fontsize=8, fontweight="bold")
    ax = axs[1]
    plate(ax)
    drive_points(ax)
    # R: 깃 4 선 끝에서 깃 5 쪽(반시계 +). 깊은 곳 0–7°, 얕은 긁힘 7–22°(사진)
    patch(ax, 4, 0.0, 7.0, A_MM + 0.8 * (B_MM - A_MM), C["R"], alpha=0.75)
    patch(ax, 4, 7.0, 22.0, A_MM + 0.9 * (B_MM - A_MM), C["R"], alpha=0.25)
    # L: 깃 3에서 깃 4 쪽 19–36°(칸 가운데에서 깃 3 쪽 2.5°), 테두리 안쪽 3 mm
    patch(ax, 3, 19.0, 36.0, B_MM - 3.0, C["L"], alpha=0.7)
    for v, rel, lab, col in ((4, 8.0, "R", C["R"]), (3, 27.5, "L", C["L"])):
        x, y = xy(B_MM, vane_angle(v, B_MM) + rel)
        xt, yt = xy(42.0, vane_angle(v, B_MM) + rel)
        ax.annotate(lab, (x, y), (xt, yt), fontsize=7, fontweight="bold", ha="center", va="center", color=col,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=col))
    ax.fill([], [], color=C["R"], alpha=0.75, label="R, 17.9 mg: deep part beside vane 4")
    ax.fill([], [], color=C["R"], alpha=0.25, label="R: shallow part")
    ax.fill([], [], color=C["L"], alpha=0.7, label="L, 17.6 mg: between vanes 3 and 4")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.0), ncol=1, frameon=False, handletextpad=0.3)
    ax.text(-43, 41, "(b)", fontsize=8, fontweight="bold")
    # (c)·(d) 실제 사진 — 사진은 저장소에 넣지 않는다(11-20). 사진이 없으면 두 패널을 비우고 알린다.
    if not have_photos("20261112_1721_M240_on_paper.jpg", S3_PHOTO["file"]):
        for ax, lab in zip(ph, ("(c)", "(d)")):
            ax.set_aspect("equal"); ax.set_xlim(-43, 43); ax.set_ylim(-43, 43); ax.axis("off")
            ax.text(-43, 41, lab, fontsize=8, fontweight="bold")
            ax.text(0, 0, "photograph not included", ha="center", va="center", fontsize=7, color="0.4")
        print(f"Fig. E1: photographs not found in {PHOTOS}; panels (c) and (d) left empty")
        save(fig, "fig_E1_layout")
        return
    # (c) 2번 시편 M240 사진(사진 2): 테두리 바깥 기호 = 사진 속 조각의 각 자리
    P2 = s2_photo("20261112_1721_M240_on_paper")
    photo_axes(ph[0], P2, P2["vanes"], "(c)")
    piece_marks(ph[0], P2)
    # (d) 3번 시편 두 절삭 뒤: 깃 번호는 r = 34 mm에서 잰 선 각을 배치도 camber로 테두리 끝까지 옮긴 자리
    P3 = S3_PHOTO
    photo_axes(ph[1], P3, {v: a + vane_angle(v, B_MM) - vane_angle(v, 34.0) for v, a in P3["vanes34"].items()}, "(d)")
    for th, lab, col, dl in ((P3["R"], "R", C["R"], 9.0), (P3["L"], "L", C["L"], 0.0)):
        x, y = xy(B_MM - 1.0, th + P3["rot"])
        xt, yt = xy(42.0, th + P3["rot"] + dl)
        ph[1].annotate(lab, (x, y), (xt, yt), fontsize=7, fontweight="bold", ha="center", va="center", color=col,
                       arrowprops=dict(arrowstyle="-", lw=0.5, color=col))
    save(fig, "fig_E1_layout")


def save(fig, name):
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(os.path.join(FIGD, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIGD, name + ".pdf"))
    plt.close(fig)
    print("saved", name)


def fig_e2():
    T = pdat.t_series()
    M, SP, chi = pdat.m_series()
    S = pdat.angle_scan()
    fig = plt.figure(figsize=(170 * MM, 74 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.0, 1.1], wspace=0.45, left=0.085, right=0.985, top=0.93, bottom=0.33)
    # (a) T-a·T-b
    ax = fig.add_subplot(gs[0])
    xs = np.arange(2)
    for i, (st, lab) in enumerate((("T-a", "T-a"), ("T-b", "T-b"))):
        r = T[T.state == st].iloc[0]
        for j, (d, se, f, col, nm) in enumerate(((r.d_lower, r.se_lower, r.fem_lower, "0.55", "lower component"),
                                                (r.d_upper, r.se_upper, r.fem_upper, "0.15", "upper component"))):
            x = i + (j - 0.5) * 0.36
            ax.bar(x, d, 0.32, color=col, label=nm if i == 0 else None)
            ax.errorbar(x, d, yerr=2 * se, fmt="none", ecolor="k", elinewidth=0.6, capsize=1.5)
            ax.plot([x - 0.16, x + 0.16], [f, f], color=C["R"], lw=1.6, label="model" if (i == 0 and j == 0) else None)
        ax.text(i, -0.335, f"Δsplit {r.dsplit_pct:+.2f}%\n(model {r.fem_dsplit_pct:+.2f}%)", ha="center", va="top", fontsize=5.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(["T-a\n(on vane 2)", "T-b\n(14° from vane 3)"])
    ax.set_ylim(-0.45, 0.2)
    ax.set_ylabel("Change of h2 doublet component [%]")
    ax.legend(loc="upper right", ncol=1, frameon=False, fontsize=5.8, handlelength=1.2)
    ax.text(-0.36, 1.06, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold")
    # (b) 측정 대 FEM
    ax = fig.add_subplot(gs[1])
    lim = (-3.3, 0.15)
    ax.fill_between(lim, [lim[0] * 0.7, lim[1] * 0.7], [lim[0] * 1.3, lim[1] * 1.3], color="0.9", lw=0)
    ax.plot(lim, lim, color="0.4", lw=0.6)
    for step, mk, col in (("M120", "s", C["M120"]), ("M240 increment", "^", C["M240"])):
        d = M[M.step == step]
        xerr = np.vstack([d.fem_photo - d.fem_lo, d.fem_hi - d.fem_photo])      # 자리 불확도(각 ±1°, 반지름 ±0.5 mm) 범위
        ax.errorbar(d.fem_photo, d.meas, yerr=2 * d.se, xerr=xerr, fmt=mk, ms=3.6, color=col, mec="k", mew=0.3, elinewidth=0.5,
                    label=("M120" if step == "M120" else "M240 − M120") + ", photographed")
        ax.plot(d.fem_nominal, d.meas, mk, ms=3.6, mfc="none", mec=col, mew=0.6,
                label=("M120" if step == "M120" else "M240 − M120") + ", nominal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Model shift [%]"); ax.set_ylabel("Measured shift [%]")
    ax.text(-3.2, 0.0, "±30%", fontsize=6, color="0.45", va="top")
    ax.legend(loc="lower right", frameon=False, fontsize=5.8, handletextpad=0.3, title="Model at piece positions:",
              title_fontsize=5.8)
    ax.text(-0.25, 1.06, "(b)", transform=ax.transAxes, fontsize=8, fontweight="bold")
    # (c) 각 의존성
    ax = fig.add_subplot(gs[2])
    sel = (("3164", "h3:m3n0", "#c0392b"), ("3459", "h1:m1n1", "#e67e22"), ("h2", "h2:m2n0", "#1f6fb4"), ("3748", "h3:m3n1", "#27ae60"))
    m1 = M[M.step == "M120"].set_index("mode")
    a_ph = pdat.piece_angles()[2]["m120_s_all"]          # 사진 평균 각(칸 폭 환산)
    for k, lab, col in sel:
        ax.plot(S.ang_meas, S[k], color=col, lw=1.0, label=lab)
        ax.plot(a_ph, m1.loc[k, "meas"], "s", color=col, mec="k", mew=0.3, ms=3.8)
    S34 = pdat.angle_scan(34.3)
    ax.plot(S34.ang_meas, S34["3164"], color="#c0392b", lw=0.8, ls="--", label="h3:m3n0, r = 34.3 mm (nominal)")
    ax.axvline(20, color="0.5", lw=0.6, ls=":"); ax.axvline(a_ph, color="0.2", lw=0.6, ls="--")
    ax.axvspan(a_ph - 1.0, a_ph + 1.0, color="0.88", lw=0)
    ax.set_xlim(0, 60); ax.set_ylim(-4.6, 0.2)
    ax.set_xlabel("Piece angle from the vane line [deg]")
    ax.set_ylabel("Model shift, six pieces × 20 mg [%]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), frameon=False, fontsize=5.8, handlelength=1.4, ncol=2,
              columnspacing=0.8, title="r = 33.3 mm unless noted", title_fontsize=5.8)
    ax.text(-0.24, 1.06, "(c)", transform=ax.transAxes, fontsize=8, fontweight="bold")
    save(fig, "fig_E2_mass")


def mean_spectra(refresh=False):
    if os.path.exists(CACHE) and not refresh:
        Z = np.load(CACHE)
        return Z["fr"], {k: Z[k] for k in Z.files if k != "fr"}
    import groove_lib as gl
    import s1p_check as sp
    LOC12 = [f"back_vane{v}_xi{x}" for v in range(1, 7) for x in ("0.3", "0.6")]
    out, fr = {}, None
    for g, ses, date in (("baseline", [17], "20261119"), ("R", [18], "20261119"), ("L", [19], None),
                         ("s2", [12], "20261116")):
        acc = {}
        for h in gl.list_hits(sp.S2F, ses, LOC12, date):
            N = float(h["peak_force_N"])
            if N < 5 or h["status"] == "rejected":
                continue
            f, m, ti, fs = gl.load_signals(h)
            t0, _ = gl.free_decay_start(f, N, ti, fs)
            X = gl.decay_spectrum(m, ti, fs, t0)
            a = acc.setdefault(h["location"], [0.0, 0])
            a[0] = a[0] + np.abs(X) ** 2 / N ** 2
            a[1] += 1
        fr = np.fft.rfftfreq(gl.NFFT, 1 / fs)
        s = (fr > 1900) & (fr < 8700)
        out[g] = np.mean([v[0][s] / v[1] / np.median(v[0][s] / v[1]) for v in acc.values()], axis=0)
        print("spectra", g, len(acc), "points")
    fr = fr[(fr > 1900) & (fr < 8700)]
    np.savez(CACHE, fr=fr, **out)
    return fr, out


def fig_e3(refresh=False):
    fr, SPEC = mean_spectra(refresh)
    E3, FR, FL = pdat.s3_table()
    fig = plt.figure(figsize=(170 * MM, 110 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.0, 0.62], hspace=0.55, wspace=0.42,
                          left=0.075, right=0.99, top=0.95, bottom=0.12)
    # (a) 평균 스펙트럼 두 구간
    windows = ((2045, 2095, [(2066.7, "2067"), (2077.6, "2078")]),
               (3000, 3740, [(3034, "3034"), (3150, "3150"), (3296, "3296"), (3497, "3497"), (3555, "3555"), (3614, "3614"),
                             (3707, "3707")]))
    for j, (lo, hi, labs) in enumerate(windows):
        ax = fig.add_subplot(gs[0, j] if j == 0 else gs[0, 1:])
        s = (fr >= lo) & (fr <= hi)
        for i, (g, lab) in enumerate((("baseline", "baseline"), ("R", "after R"), ("L", "after L"))):
            ax.semilogy(fr[s], SPEC[g][s] * 10 ** (1.2 * (2 - i)), color=C["base"] if g == "baseline" else C[g], lw=0.8, label=lab)
        for f, t in labs:
            ax.axvline(f, color="0.75", lw=0.4, ls=":", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_yticks([]); ax.minorticks_off()
        ax.set_xlabel("Frequency [Hz]")
        top = ax.secondary_xaxis("top")
        top.set_xticks([f for f, _ in labs]); top.set_xticklabels([t for _, t in labs], fontsize=5.6, color="0.35")
        top.tick_params(length=2, pad=1)
        if j == 0:
            ax.set_ylabel("Mean |X/F|² (offset)")
            ax.text(-0.22, 1.1, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold")
            for i, (g, lab) in enumerate((("baseline", "baseline"), ("R", "after R"), ("L", "after L"))):
                k = np.argmin(np.abs(fr - 2091))
                ax.text(2093.5, SPEC[g][k] * 10 ** (1.2 * (2 - i)) * 1.6, lab, ha="right", va="bottom", fontsize=6.3,
                        color=C["base"] if g == "baseline" else C[g])
    # (b) 바뀐 모드
    ax = fig.add_subplot(gs[1, :2])
    order = [("R", "2066"), ("R", "2077"), ("R", "3033"), ("R", "3554→3562"), ("R", "3613→3595"), ("R", "3980"), ("R", "6932→6910"),
             (None, None),
             ("L", "2066"), ("L", "2077"), ("L", "3033"), ("L", "3149"), ("L", "3296"), ("L", "3980"), ("L", "6732"), ("L", "6932"),
             ("L", "8031")]
    names = {"2066": "2067", "2077": "2078", "3033": "3034", "3149": "3150", "3296": "3296", "3554→3562": "3555",
             "3613→3595": "3614", "3980": "3980", "6932→6910": "6933", "6932": "6933", "6732": "6734", "8031": "8033"}
    xs, ticks = [], []
    for i, (step, key) in enumerate(order):
        if step is None:
            continue
        r = E3[(E3.step == step) & (E3["mode"] == key)].iloc[0]
        col = C[step]
        ax.bar(i, r.meas, 0.62, color=col, alpha=0.85)
        ax.plot([i, i], [r.lo, r.hi], color="k", lw=0.8)
        if np.isfinite(r.get("fem_lo", np.nan)):
            ax.add_patch(plt.Rectangle((i - 0.36, r.fem_lo), 0.72, max(r.fem_hi - r.fem_lo, 0.012), fill=False, ec="k", lw=0.9))
        xs.append(i); ticks.append(names[key])
    ax.axhspan(-0.05, 0.05, color="0.88", zorder=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(7, color="0.6", lw=0.5)
    ax.text(3, 0.43, "R: beside vane 4 (baseline → R)", ha="center", fontsize=6.8, color=C["R"])
    ax.text(12, 0.43, "L: between vanes 3 and 4 (R → L)", ha="center", fontsize=6.8, color=C["L"])
    ax.set_xticks(xs); ax.set_xticklabels(ticks, rotation=0)
    ax.set_xlabel("Mode (baseline frequency of specimen 3) [Hz]")
    ax.set_ylabel("Frequency change [%]")
    ax.set_ylim(-0.62, 0.5); ax.set_xlim(-0.6, 16.6)
    ax.bar([], [], color="0.5", label="measured, median of vanes 2–6")
    ax.plot([], [], color="k", lw=0.8, label="range of vanes 2–6")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, fill=False, ec="k", lw=0.9, label="model, footprints consistent with the photographs"))
    ax.legend(loc="lower center", bbox_to_anchor=(0.68, 0.0), frameon=False, ncol=1, fontsize=6)
    ax.text(-0.11, 1.03, "(b)", transform=ax.transAxes, fontsize=8, fontweight="bold")
    # (c) 섞인 두 모드 합, h2 짝 합
    ax = fig.add_subplot(gs[1, 2])
    items = [("R", "3033+3149"), ("L", "3033+3149"), ("R", "h2 sum"), ("L", "h2 sum")]
    for i, (step, key) in enumerate(items):
        r = E3[(E3.step == step) & (E3["mode"] == key)].iloc[0]
        x = i + (0.4 if i >= 2 else 0)
        ax.bar(x, r.meas, 0.6, color=C[step], alpha=0.85)
        ax.add_patch(plt.Rectangle((x - 0.34, r.fem_lo), 0.68, max(r.fem_hi - r.fem_lo, 0.012), fill=False, ec="k", lw=0.9))
    ax.axhspan(-0.05, 0.05, color="0.88", zorder=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(["R", "L", "R", "L"])
    import matplotlib.transforms as mtrans
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.5, -0.13, "3034 + 3150 Hz", ha="center", va="top", fontsize=6.0, transform=tr)
    ax.text(2.9, -0.13, "h2 doublet", ha="center", va="top", fontsize=6.0, transform=tr)
    ax.set_ylim(-0.7, 0.7)
    ax.set_ylabel("Sum of changes [%]")
    ax.text(-0.42, 1.03, "(c)", transform=ax.transAxes, fontsize=8, fontweight="bold")
    save(fig, "fig_E3_removal")


PHOTOS = os.path.join(pdat.BASE, "groove-test", "photos")
# 3번 시편 두 절삭 뒤 사진(11-20, 바닥에 놓고 위에서): 판 테두리 원 맞춤(가운데·반지름 화소, 잔차 rms 8 화소),
# 그린 깃 선의 r = 34 mm 각과 깎은 자리 가운데(테두리 띠의 밝은 반사)의 사진 방향 각(반시계 +).
# rot = 그린 깃 선 6개의 r = 33–35 mm 각을 배치도 깃 선에 맞춘 원형 평균(깃별 ±3.5°).
S3_PHOTO = dict(file="20261120_S3_after_L_top.jpg", cx=1744.1, cy=2989.0, r_px=540.4, rot=-68.8, yellow_rim=True,
                vanes34={1: 126.3, 2: 184.8, 3: 246.2, 4: 300.2, 5: 4.8, 6: 68.8}, R=-56.0, L=-89.5)


def have_photos(*names):
    """사진은 저장소에 넣지 않는다(원본 JPG에 촬영 위치가 있다) — 이 컴퓨터에 사진이 있으면 True."""
    return all(os.path.exists(os.path.join(PHOTOS, n)) for n in names)


def s2_photo(name):
    """2번 시편 사진: 판 테두리 상의 가운데·반지름(화소), 배치도 방향으로 돌릴 각, 그린 깃 선의 테두리 끝과 조각의 사진 방향 각.
    자세·그린 깃 선·조각은 s3_m_photo.py가 잰 것(groove-test/out/m_photo.json, m_piece_positions.csv — 칸 폭 환산 전 각)."""
    import s3_m_photo as mp
    m = json.load(open(os.path.join(pdat.BASE, "groove-test", "out", "m_photo.json")))[name]
    pose = np.array(m["pose"])
    t = np.linspace(0, 2 * np.pi, 721)
    x, y = mp.plate_to_img(mp.R_RIM * np.cos(t), mp.R_RIM * np.sin(t), pose)
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())

    def img_ang(r, th):
        u, w = mp.plate_to_img(np.array([r * np.cos(np.radians(th))]), np.array([r * np.sin(np.radians(th))]), pose)
        return float(np.degrees(np.arctan2(cy - w[0], u[0] - cx)))

    line = lambda L, r: L["th34"] + L["slope"] * (r - 34.0)
    offs = [vane_angle(int(v), r) - img_ang(r, line(L, r)) for v, L in m["lines"].items() for r in (30.0, 34.0, 36.0)]
    T = pd.read_csv(os.path.join(pdat.BASE, "groove-test", "m_piece_positions.csv"))
    return dict(file=name + ".jpg", cx=cx, cy=cy, r_px=0.25 * (x.max() - x.min() + y.max() - y.min()),
                rot=float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(offs)))))),
                vanes={int(v): img_ang(B_MM, line(L, B_MM)) for v, L in m["lines"].items()},
                pieces=[(img_ang(r.r_mm, line(m["lines"][r.bay.split("-")[0]], r.r_mm) + r.ang_from_k),
                         "M120" if r.ang_from_k < 25 else "M240") for r in T[T.photo == name].itertuples()])


def photo_rot(P, pad=1.2, edge=1.004, n_max=1500):
    """사진을 판 테두리 상의 가운데 둘레로 P['rot']°(반시계 +) 돌려 배치도 방향에 맞추고 판 바깥을 흰색으로 지운다(배경 글씨 제거).
    P['yellow_rim']이면 테두리 띠(0.95 R 바깥)에 남은 노란 바닥 화소도 지운다. 좌표는 EXIF 회전 뒤 원본 화소.
    반환: (배열, 배열 안 판 가운데, 판 반지름 화소) — 한 변이 n_max 화소를 넘으면 줄인다."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(os.path.join(PHOTOS, P["file"]))).convert("RGB")
    h = int(round(P["r_px"] * pad))
    x0, y0 = int(round(P["cx"])) - h, int(round(P["cy"])) - h
    c = im.crop((x0, y0, x0 + 2 * h, y0 + 2 * h)).rotate(P["rot"], resample=Image.BICUBIC, center=(P["cx"] - x0, P["cy"] - y0),
                                                       fillcolor=(255, 255, 255))
    k = min(1.0, n_max / (2 * h))
    if k < 1.0:
        c = c.resize((int(round(2 * h * k)),) * 2, Image.LANCZOS)
    a = np.asarray(c).astype(float) / 255
    u0, v0, rp = (P["cx"] - x0) * k, (P["cy"] - y0) * k, P["r_px"] * k
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    rr = np.hypot(xx - u0, yy - v0)
    out = rr > rp * edge
    if P.get("yellow_rim"):
        out |= (rr > 0.95 * rp) & (a[..., 0] - a[..., 2] > 0.25) & (a[..., 1] - a[..., 2] > 0.15)
    a[out] = 1.0
    return a, (u0, v0), rp


def photo_axes(ax, P, vanes, lab):
    """판 사진을 배치도와 같은 mm 축척(테두리 = b)·같은 방향으로 놓고, 사진 속 그린 깃 선의 테두리 끝에 깃 번호를 단다."""
    a, (u0, v0), rp = photo_rot(P)
    s = B_MM / rp
    H, W = a.shape[:2]
    ax.imshow(a, extent=(-u0 * s, (W - u0) * s, -(H - v0) * s, v0 * s))
    for v, th in vanes.items():
        xt, yt = xy(B_MM + 3.6, th + P["rot"])
        ax.text(xt, yt, f"{v}", ha="center", va="center", fontsize=7, color="0.2")
    ax.set_aspect("equal")
    ax.set_xlim(-43, 43); ax.set_ylim(-43, 43)
    ax.axis("off")
    ax.text(-43, 41, lab, fontsize=8, fontweight="bold")


def piece_marks(ax, P):
    """사진 속 조각의 각 자리를 테두리 바깥에 Fig. E1(a)와 같은 기호로 찍는다."""
    for th, st in P["pieces"]:
        x, y = xy(B_MM + 2.0, th + P["rot"])
        ax.plot(x, y, "s" if st == "M120" else "^", ms=3.2, color=C[st], mec="k", mew=0.3, zorder=6)


def fig_se2():
    """보충: 3번 시편 R 직후(L 전) 비스듬히 본 근접 사진 — 깃 4 선 끝의 패인 자리와 깃 5 쪽으로 이어지는 얕은 긁힘.
    M240 사진(사진 2)과 두 절삭 뒤 사진은 본문 Fig. E1(c)·(d)에 있다. 사진 1(M120)은 매단 클립이 보여 싣지 않는다(11-20 사용자)."""
    if not have_photos("20261119_S3_after_R_oblique.jpg"):
        print(f"Fig. SE2: photograph not found in {PHOTOS}; skipped")
        return
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(os.path.join(PHOTOS, "20261119_S3_after_R_oblique.jpg"))).convert("RGB")
    x0, y0, x1, y1 = 666, 1866, 2755, 3066                 # 자른 자리(원본 화소, EXIF 회전 뒤)
    n = 1500
    c = im.crop((x0, y0, x1, y1)).resize((n, int(round(n * (y1 - y0) / (x1 - x0)))), Image.LANCZOS)
    k = n / (x1 - x0)
    fig = plt.figure(figsize=(160 * MM, 160 * (y1 - y0) / (x1 - x0) * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.asarray(c)); ax.axis("off")
    # 표시 자리(자른 사진의 화소): 패인 자리, 깃 4·5의 그린 선 위 한 점 → 글자 자리
    for (lab, (u, v), (ut, vt)) in (("R", (930, 780), (1230, 560)), ("vane 4", (300, 335), (120, 480)),
                                    ("vane 5", (1700, 185), (1770, 370))):
        col = C["R"] if lab == "R" else "0.1"
        ax.annotate(lab, (u * k, v * k), (ut * k, vt * k), fontsize=9 if lab == "R" else 7.5,
                    fontweight="bold" if lab == "R" else "normal", color=col, ha="center", va="center",
                    bbox=None if lab == "R" else dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=col))
    save(fig, "fig_SE2_photos")


def fig_se1(refresh=False):
    """보충: 3번 시편 기준 대 2번 시편(세척 뒤 A2) 12자리 평균 스펙트럼."""
    fr, SPEC = mean_spectra(refresh)
    fig, axs = plt.subplots(2, 1, figsize=(170 * MM, 95 * MM))
    for ax, (lo, hi) in zip(axs, ((1950, 4700), (4700, 8600))):
        s = (fr >= lo) & (fr <= hi)
        ax.semilogy(fr[s], SPEC["s2"][s], color="0.55", lw=0.8, label="specimen 2 (41.32 g)")
        ax.semilogy(fr[s], SPEC["baseline"][s], color="k", lw=0.8, label="specimen 3 (39.94 g)")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("Mean |X/F|² (normalized)")
    axs[0].legend(loc="upper left", frameon=False)
    axs[1].set_xlabel("Frequency [Hz]")
    for i, ax in enumerate(axs):
        ax.text(-0.075, 1.0, "(ab)"[2 * i:2 * i + 1].join("()") if False else ("(a)", "(b)")[i], transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.98, bottom=0.1, hspace=0.3)
    save(fig, "fig_SE1_spectra")


def fig_se4():
    """보충: 깎기 두 단계에서 따라간 모든 모드의 12자리 값 — 바뀐 모드, 측정 바닥 안, 되돌아간 모드를 나눈다."""
    Rc = pd.read_csv(os.path.join(pdat.S3, "R12_check.csv"), dtype={"mode": str, "xi": str})
    Lc = pd.read_csv(os.path.join(pdat.S3, "L12_check.csv"), dtype={"mode": str, "xi": str})
    changed_R = {"2066", "2077", "3033", "3554→3562", "3613→3595", "3980", "6932→6910"}
    changed_L = {"2066", "2077", "3033", "3149", "3296", "6732", "6932", "8031"}
    revers = {"5136", "7216→7228", "7749→7776", "7216", "7749"}
    fig, axs = plt.subplots(2, 1, figsize=(170 * MM, 110 * MM))
    for ax, (D, col_d, col_se, col_na, col_nb, changed, title, colr) in zip(axs, (
            (Rc, "d_pct", "se_pct", "n_a", "n_b", changed_R, "baseline → R", C["R"]),
            (Lc, "RL_d", "RL_se", "RL_na", "RL_nb", changed_L, "R → L", C["L"]))):
        order = list(dict.fromkeys(D["mode"]))
        xs = {n: i for i, n in enumerate(order)}
        ok = np.isfinite(D[col_d]) & (D[col_se] < 0.08) & (D[col_na] >= 5) & (D[col_nb] >= 5)
        for _, r in D[ok].iterrows():
            n = r["mode"]
            c = colr if n in changed else ("#7d3c98" if n in revers else "0.6")
            ax.plot(xs[n] + (r.vane - 3.5) * 0.08, r[col_d], "o" if r.xi == "0.3" else "s", ms=2.6, color=c,
                    mfc=c if r.xi == "0.6" else "white", mew=0.5)
        ax.axhspan(-0.05, 0.05, color="0.9", zorder=0); ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([n.split("→")[0] for n in order], rotation=90, fontsize=6)
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_ylabel(f"Change, {title} [%]")
        ax.plot([], [], "s", color=colr, ms=3, label="changed (Table E3)")
        ax.plot([], [], "s", color="#7d3c98", ms=3, label="reversible, handling-related")
        ax.plot([], [], "s", color="0.6", ms=3, label="within the measurement floor or unresolved")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=3)
    axs[1].set_xlabel("Mode (baseline frequency of specimen 3) [Hz]; open: ξ = 0.3, filled: ξ = 0.6")
    for i, ax in enumerate(axs):
        ax.text(-0.075, 1.02, ("(a)", "(b)")[i], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.12, hspace=0.5)
    save(fig, "fig_SE4_all_modes")


def fig_se3():
    """보충: 측정 바닥 — (a) 2번 시편 다시 매달기, (b) 깎은 직후 데워짐(측정 순서), (c) 되돌아간 세 모드."""
    V = pdat.rehang()
    RV = pdat.reversible()
    Lc = pd.read_csv(os.path.join(pdat.S3, "L12_check.csv"), dtype={"mode": str, "xi": str})
    fig = plt.figure(figsize=(170 * MM, 60 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.8], wspace=0.42, left=0.085, right=0.985, top=0.9, bottom=0.24)
    ax = fig.add_subplot(gs[0])
    V = V.copy()
    V["mode"] = V["mode"].astype(str)
    modes = list(dict.fromkeys(V["mode"]))
    for i, m in enumerate(modes):
        x = V[V["mode"] == m]
        for _, r in x.iterrows():
            j = 0 if r["loc"] == "xi0.3" else 1
            ax.errorbar(i + (j - 0.5) * 0.25, r.d_pct, yerr=2 * r.se_pct, fmt="o" if j == 0 else "s", ms=2.8,
                        color="k", mfc="white" if j == 0 else "k", elinewidth=0.5)
    ax.axhspan(-0.05, 0.05, color="0.9", zorder=0); ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(modes))); ax.set_xticklabels(modes, rotation=90, fontsize=6)
    ax.set_ylabel("Re-suspension change [%]")
    ax.set_ylim(-0.13, 0.16); ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
    ax.plot([], [], "o", mfc="white", mec="k", ms=2.8, label="ξ = 0.3"); ax.plot([], [], "s", color="k", ms=2.8, label="ξ = 0.6")
    ax.legend(frameon=False, loc="upper left", ncol=2, borderaxespad=0.2)
    ax.text(-0.16, 1.03, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax = fig.add_subplot(gs[1])
    import s3_spec3_L12 as l12
    Dr = l12.drift(Lc)
    ax.plot(Dr.t_L - Dr.t_L.min(), Dr.AL_med - np.nanmedian(Dr.AL_med[2:]), "s-", color=C["L"], ms=3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Minutes after the first point (L)")
    ax.set_ylabel("Singlet median, re-centered [%]")
    ax.text(-0.24, 1.03, "(b)", transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax = fig.add_subplot(gs[2])
    for _, r in RV.iterrows():
        ys = [0, r.R, r.cumulative]
        ax.plot([0, 1, 2], ys, "o-", ms=3, label=f"{int(r['mode'])} Hz")
    ax.axhspan(-0.05, 0.05, color="0.9", zorder=0); ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["baseline", "after R", "after L"])
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-0.07, 0.41); ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_ylabel("Change from baseline [%]")
    ax.legend(frameon=False, loc="upper left", fontsize=6, handlelength=1.2, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    ax.text(-0.3, 1.03, "(c)", transform=ax.transAxes, fontsize=8, fontweight="bold")
    save(fig, "fig_SE3_floor")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-spectra", action="store_true")
    a = ap.parse_args()
    fig_e1()
    fig_e2()
    fig_e3(a.refresh_spectra)
    fig_se1()
    fig_se2()
    fig_se3()
    fig_se4()
