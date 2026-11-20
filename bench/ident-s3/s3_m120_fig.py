"""그림 m120.png(45번): 11-12 질량 띠 M120(M75 자리, 깃 선과 나란한 납 띠 20 mg × 6 = 120 mg) 대 A1-6v(11-11 S9).
(a)(b) 깃 6곳 × ξ0.3·ξ0.6 평균 자유 감쇠 스펙트럼(위치마다 최댓값으로 나눠 더함) — h2 짝과 3.1–3.8 kHz 무리.
       회색 점선 = A1-6v 값, 빨강 눈금 = M120 측정, 파랑 눈금 = FEM 예측 위치(A1-6v × (1 + M120 예측)).
(c) 모드별 변화: 측정(깃 6곳 묶음, ±σ) 대 FEM(G3, s3_pred_poles.csv의 M120), 위 숫자 = 측정/FEM.
입력: groove-test/out/A1-6v.json·M120.json(groove_state), M120_report.csv(groove_compare), NAS 타격 자료(첫 실행에서
스펙트럼을 groove-test/out/m120_fig_cache.npz로 저장)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groove_lib as gl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "groove-test", "out")
FOLDER = "/mnt/nas/data/GY_DATA/impact_testw2s2/impact_testw2s2"
WIN = ((2110.0, 2160.0), (3040.0, 3820.0))
CACHE = os.path.join(OUT, "m120_fig_cache.npz")
A = json.load(open(os.path.join(OUT, "A1-6v.json")))
M = json.load(open(os.path.join(OUT, "M120.json")))
R = pd.read_csv(os.path.join(OUT, "M120_report.csv")).assign(mode=lambda d: d["mode"].astype(str)).set_index("mode")


def mean_spectra(S):
    """상태 JSON의 타격 목록대로 자유 감쇠 파워 스펙트럼 |X|²/N²을 위치마다 평균하고, 위치별 최댓값으로 나눠 더한다."""
    fr = np.fft.rfftfreq(gl.NFFT, 1 / gl.FS_EXPECTED)
    sel = np.zeros(fr.size, bool)
    for lo, hi in WIN:
        sel |= (fr >= lo) & (fr <= hi)
    acc = {}
    for h in S["hits"]:
        if h["N"] < 5 or h["status"] == "rejected":
            continue
        f, m, ti, fs = gl.load_signals(dict(path=os.path.join(FOLDER, h["file"])))
        t0, _ = gl.free_decay_start(f, h["N"], ti, fs)
        X = gl.decay_spectrum(m, ti, fs, t0)[sel]
        a = acc.setdefault(h["loc"], [0.0, 0])
        a[0] = a[0] + np.abs(X) ** 2 / h["N"] ** 2
        a[1] += 1
    return fr[sel], sum((p / n) / (p / n).max() for p, n in acc.values())


if os.path.exists(CACHE):
    Z = np.load(CACHE)
    fr, PA, PM = Z["fr"], Z["A"], Z["M"]
else:
    fr, PA = mean_spectra(A)
    _, PM = mean_spectra(M)
    np.savez_compressed(CACHE, fr=fr, A=PA.astype(np.float32), M=PM.astype(np.float32))

fig = plt.figure(figsize=(13, 9.4))
axs = [fig.add_axes([0.05, 0.60, 0.25, 0.33]), fig.add_axes([0.33, 0.60, 0.65, 0.33])]
cx = fig.add_axes([0.06, 0.08, 0.92, 0.38])
for ax, (lo, hi), tit in zip(axs, WIN, ("(a) h2 pair", "(b) 3.1-3.8 kHz group")):
    s = (fr >= lo) & (fr <= hi)
    ax.semilogy(fr[s], PA[s], color="k", lw=0.8, label="A1-6v (11-11 S9)")
    ax.semilogy(fr[s], PM[s], color="C3", lw=0.8, label="M120 (11-12 S10)")
    y1 = ax.get_ylim()[1]
    for n in M["modes"]:
        fa = A["modes"][n]["comb"]["f"]
        if not (lo <= fa <= hi) or n not in R.index:
            continue
        fm = M["modes"][n]["comb"]["f"]
        ax.axvline(fa, color="0.6", lw=0.6, ls=":")
        ax.plot([fm, fm], [y1 * 0.35, y1 * 0.7], color="C3", lw=1.6)
        if np.isfinite(R.loc[n, "pred_pct"]):
            fp = fa * (1 + R.loc[n, "pred_pct"] / 100)
            if lo <= fp <= hi:
                ax.plot([fp, fp], [y1 * 0.35, y1 * 0.7], color="C0", lw=1.6, ls="--")
                ax.text(fp, y1 * 0.33, n, fontsize=6.5, rotation=90, ha="center", va="top", color="C0")
        ax.text(fa, y1 * 0.9, n, fontsize=7, rotation=90, ha="center", va="top", color="0.35")
    ax.set_xlim(lo, hi)
    ax.tick_params(axis="y", which="both", labelleft=False)
    ax.set_xlabel("frequency [Hz]", fontsize=8.5)
    ax.set_title(tit, fontsize=9, loc="left")
axs[0].set_ylabel("mean free-decay power (log)\n12 locations, each normalised", fontsize=8.5)
axs[1].plot([], [], color="C3", lw=1.6, label="M120 measured")
axs[1].plot([], [], color="C0", lw=1.6, ls="--", label="FEM prediction (G3, 120 mg strips)")
axs[1].legend(fontsize=7.5, loc="lower right", ncol=2, frameon=False)
axs[0].legend(fontsize=7, loc="lower center", frameon=False)

# (c) 모드별
groups = [("disk", ["2134", "2142", "3748"], "C3"), ("panel", ["3164", "3459", "3639", "8321", "6439"], "C0"),
          ("mixed", ["3250"], "C9"), ("hub", ["3804", "4225"], "C4"), ("back", ["4967", "5617", "5745"], "C2"),
          ("candidate", ["6558", "6817", "7093", "7574"], "0.45"), ("unassigned", ["3589", "7462"], "0.7")]
x, ticks, labs = 0.0, [], []
for g, ns, col in groups:
    x0 = x
    for n in ns:
        r = R.loc[n]
        p = r.pred_pct
        if np.isfinite(p):
            cx.bar(x - 0.18, p, 0.34, color="none", edgecolor=col, hatch="//", lw=1)
        cx.errorbar(x + 0.18, r.d_pct, r.sig_pct, fmt="s", color=col, ms=5, capsize=2)
        if np.isfinite(p) and abs(p) >= 0.3:
            cx.text(x, min(p, r.d_pct) - 0.18, f"{r.d_pct / p:.2f}", ha="center", fontsize=7.5, color=col)
        ticks.append(x); labs.append(n)
        x += 1
    cx.text(0.5 * (x0 + x - 1), 0.32, g, ha="center", fontsize=8, color=col)
    x += 0.6
cx.axhline(0, color="k", lw=0.6)
cx.set_xticks(ticks)
cx.set_xticklabels(labs, fontsize=8)
cx.set_ylim(-5.4, 0.55)
cx.set_xlim(-0.8, x - 0.4)
cx.set_ylabel("change vs A1-6v [%]", fontsize=8.5)
h_ = [Patch(facecolor="none", edgecolor="0.3", hatch="//", label="FEM (G3, M120: 6 strips × 20 mg)"),
      cx.errorbar([], [], [], fmt="s", color="0.3", label="measured (6 vanes × ξ0.3/0.6, ±σ incl. 0.05 % hanging floor)")]
cx.legend(handles=h_, fontsize=7.5, loc="lower right", frameon=False)
cx.set_title("(c) measured vs FEM; number = measured/FEM. disk 3748 and 6.8-7.6 kHz modes ≈ 1, "
             "h2 pair ≈ 0.6, panel modes 0.25-0.3", fontsize=9, loc="left")
fig.suptitle("M120 (lead strips 6 × 20 mg at the M75 positions, front plate) vs A1-6v", fontsize=10)
fig.savefig(os.path.join(HERE, "..", "groove-test", "m120.png"), dpi=130)
print("saved")
