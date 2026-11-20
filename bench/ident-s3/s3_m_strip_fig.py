"""그림 m-strips.png(45번 §8): 질량 띠 두 단계 — M120(측정 20° 자리 = FEM 섹터 + 40°)과 M240(+ 측정 40° 자리 = FEM + 20°).
(a)(b) 깃 6곳 × ξ0.3·ξ0.6 평균 자유 감쇠 스펙트럼(위치마다 최댓값으로 나눠 더함): A1-6v·M120·M240, 눈금 = 상태별 측정값.
(c) 단계별 변화: 1단계 A1-6v → M120, 2단계 M120 → M240(증분). 빗금 = FEM(G3, 띠 20 mg × 6), 네모 = 측정, 숫자 = 측정/FEM.
(d) FEM 띠 고리(r 34.3) 예측의 각 의존성과 측정(1단계 = FEM 40°, 2단계 = FEM 20°) — 20° 자리(FEM 40–48°)는 패널 예측이 가파르다.
입력: groove-test/out/A1-6v·M120·M240.json(groove_state), M_report.csv·M240_inc.csv(groove_compare),
fe-nominal/s3_m_angle_scan.csv·groove-test/out/m_angle_fit.json(s3_m_angle_scan.py), NAS 타격 자료
(첫 실행에서 스펙트럼을 groove-test/out/m_strip_fig_cache.npz로 저장)."""
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
WIN = ((2100.0, 2160.0), (3040.0, 3820.0))
CACHE = os.path.join(OUT, "m_strip_fig_cache.npz")
ST = (("A1-6v", "k", "A1-6v (11-11 S9)"), ("M120", "C3", "M120 (S10, 20° strips)"), ("M240", "C0", "M240 (S11, + 40° strips)"))
J = {s: json.load(open(os.path.join(OUT, f"{s}.json"))) for s, _, _ in ST}
R1 = pd.read_csv(os.path.join(OUT, "M_report.csv")).assign(mode=lambda d: d["mode"].astype(str))
R1 = R1[R1.state == "M120"].set_index("mode")
R2 = pd.read_csv(os.path.join(OUT, "M240_inc.csv")).assign(mode=lambda d: d["mode"].astype(str)).set_index("mode")


def mean_spectra(S):
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
    fr, PS = Z["fr"], {s: Z[s] for s, _, _ in ST}
else:
    PS = {}
    for s, _, _ in ST:
        fr, PS[s] = mean_spectra(J[s])
    np.savez_compressed(CACHE, fr=fr, **{s: v.astype(np.float32) for s, v in PS.items()})

fig = plt.figure(figsize=(13, 9.6))
axs = [fig.add_axes([0.05, 0.60, 0.25, 0.33]), fig.add_axes([0.33, 0.60, 0.65, 0.33])]
cx = fig.add_axes([0.06, 0.08, 0.58, 0.38])
dx = fig.add_axes([0.70, 0.08, 0.28, 0.38])
for ax, (lo, hi), tit in zip(axs, WIN, ("(a) h2 pair", "(b) 3.0-3.8 kHz group")):
    s = (fr >= lo) & (fr <= hi)
    for st, col, lab in ST:
        ax.semilogy(fr[s], PS[st][s], color=col, lw=0.8, label=lab)
    y1 = ax.get_ylim()[1]
    for st, col, _ in ST:
        for n, v in J[st]["modes"].items():
            f = v.get("comb", {}).get("f", np.nan)
            if lo <= f <= hi and gl.MODE[n].fem:
                ax.plot([f, f], [y1 * 0.4, y1 * 0.75], color=col, lw=1.5)
                if st == "A1-6v":
                    ax.text(f, y1 * 0.95, n, fontsize=7, rotation=90, ha="center", va="top", color="0.35")
    ax.set_xlim(lo, hi)
    ax.tick_params(axis="y", which="both", labelleft=False)
    ax.set_xlabel("frequency [Hz]", fontsize=8.5)
    ax.set_title(tit, fontsize=9, loc="left")
axs[0].set_ylabel("mean free-decay power (log)\n12 locations, each normalised", fontsize=8.5)
axs[1].legend(fontsize=7.5, loc="lower right", frameon=False)

# (c) 단계별 측정 대 FEM
MODES = [("h2 pair", ["2134", "2142"]), ("3164", ["3164"]), ("3459", ["3459"]), ("3639", ["3639"]), ("8321", ["8321"]),
         ("3250", ["3250"]), ("3748", ["3748"]), ("3804", ["3804"]), ("5745", ["5745"]), ("7093", ["7093"]), ("7574", ["7574"])]
x, ticks = 0.0, []
for lab, ns in MODES:  # noqa: B007
    for j, (R, col) in enumerate(((R1, "C3"), (R2, "C0"))):
        m = float(np.mean([R.loc[n, "d_pct"] for n in ns])); p = float(np.mean([R.loc[n, "pred_pct"] for n in ns]))
        e = float(np.mean([R.loc[n, "sig_pct"] for n in ns]))
        xx = x + (j - 0.5) * 0.8
        cx.bar(xx - 0.17, p, 0.3, color="none", edgecolor=col, hatch="//", lw=1)
        cx.errorbar(xx + 0.17, m, e, fmt="s", color=col, ms=5, capsize=2)
        if abs(p) >= 0.3:
            cx.text(xx, min(p, m) - 0.17, f"{m / p:.2f}", ha="center", fontsize=6.5, color=col, rotation=90, va="top")
    ticks.append(x)
    x += 2.0
cx.axhline(0, color="k", lw=0.6)
cx.set_xticks(ticks)
cx.set_xticklabels([l for l, _ in MODES], fontsize=7.5, rotation=45)
cx.set_ylim(-4.3, 0.35)
cx.set_ylabel("change [%]", fontsize=8.5)
h_ = [Patch(facecolor="none", edgecolor="C3", hatch="//", label="step 1 FEM: A1-6v → M120 (strips at FEM sector + 40°)"),
      cx.errorbar([], [], [], fmt="s", color="C3", label="step 1 measured"),
      Patch(facecolor="none", edgecolor="C0", hatch="//", label="step 2 FEM: M120 → M240 (added strips at FEM + 20°)"),
      cx.errorbar([], [], [], fmt="s", color="C0", label="step 2 measured")]
cx.legend(handles=h_, fontsize=6.8, loc="lower right", frameon=False, ncol=1)
cx.set_title("(c) per step, measured vs FEM at the nominal strip positions (number = measured/FEM)", fontsize=9, loc="left")

# (d) 각 의존성
SC = pd.read_csv(os.path.join(HERE, "..", "fe-nominal", "s3_m_angle_scan.csv"))
FIT = json.load(open(os.path.join(OUT, "m_angle_fit.json")))
g = SC[np.isclose(SC.rc_mm, 34.3)].sort_values("theta_deg")
dfp = lambda e: 100 * (np.sqrt(1 + e) - 1)
for k, col in (("3164", "C0"), ("3639", "C9"), ("h2", "C3"), ("3748", "C2")):
    dx.plot(g.theta_deg, dfp(g[k]), color=col, lw=1.2, label=f"FEM {k}")
    dx.plot(40, FIT["M120"][k], "s", color=col, ms=6, mec="k", mew=0.5)
    dx.plot(20, FIT["INC"][k], "^", color=col, ms=6, mec="k", mew=0.5)
b34 = FIT["M120_best_d1_by_rc"]["34.3"]
dx.axvspan(40, 40 + b34["d1"], color="C3", alpha=0.12)
dx.axvline(40, color="C3", lw=0.6, ls=":"); dx.axvline(20, color="C0", lw=0.6, ls=":")
dx.set_xlim(0, 60); dx.set_ylim(-4.6, 0.2)
dx.set_xlabel("strip angle in FEM sector [deg]  (measured angle from vane k = 60 − this)", fontsize=7.5)
dx.set_ylabel("FEM change, ring of 6 strips × 20 mg, r 34.3 mm [%]", fontsize=7.5)
dx.plot([], [], "s", color="0.5", mec="k", mew=0.5, label="step 1 measured (at FEM 40°)")
dx.plot([], [], "^", color="0.5", mec="k", mew=0.5, label="step 2 measured (at FEM 20°)")
dx.legend(fontsize=6.5, loc="lower left", frameon=False, ncol=1)
dx.set_title(f"(d) FEM vs strip angle (r 34.3 mm) is steep near 40°;\nstep 1 fits at 40 + {b34['d1']:.1f}° "
             f"(χ² {FIT['nominal']['chi_M120']:.0f} → {b34['chi']:.0f}, 11 modes)", fontsize=8, loc="left")
fig.suptitle("Lead strips 6 × 20 mg per step on the front plate: M120 (20° from vane k) and M240 (+ 40°)", fontsize=10)
fig.savefig(os.path.join(HERE, "..", "groove-test", "m-strips.png"), dpi=130)
print("saved")
