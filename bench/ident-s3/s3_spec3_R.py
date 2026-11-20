"""3번 시편 R(11-19 S18, 메모 s3_R) — 깃 4 위를 17.9 mg(39941.4 → 39923.5 mg) 깎은 뒤 1번 깃 두 자리, 기준 S17(같은 두 자리)과 비교(54번).

- 대역은 두 상태의 1번 깃 평균 스펙트럼 봉우리를 모두 담게 손으로 정했다. '→'는 옮겨 간 봉우리 짝(추정), '(new)'는 R에만 있는 봉우리.
- 방법은 s3_l1_check와 같다: 타격별 극점, 상태마다 따로 둔 힘 기울기로 15 N 환산, 감쇠는 타격별 ζ 중앙값, S18 안 시간 기울기.
- FEM(s3_r_grind, 깃 4 선 위 17.9 mg)은 53번 §2 짝으로 맞댄다. 1번 깃은 깃 4와 180° 떨어져 h1·h2·h3 모두 깃 4의 배 성분을 본다.
출력: groove-test/s3/R1_vane1_check.csv·R1_vane1_check.png
사용: python s3_spec3_R.py"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1p_check as sp  # noqa: E402

OUT = os.path.join(HERE, "..", "groove-test", "s3")
FEMD = os.path.join(HERE, "..", "fe-nominal", "s3_r_grind")
V1 = ["back_vane1_xi0.3", "back_vane1_xi0.6"]
GROUPS = {"A": (sp.S2F, [17], "20261119", V1), "R": (sp.S2F, [18], "20261119", V1)}
MG = 17.9
BANDS = [("2066/2077 h2", 2050, 2092, 2), ("3033", 3005, 3060, 1), ("3149", 3120, 3180, 1), ("3296", 3265, 3325, 1),
         ("3467", 3455, 3480, 1), ("3497", 3488, 3507, 1), ("3516(new)", 3507, 3530, 1), ("3554→3560", 3540, 3575, 1),
         ("3613→3592", 3580, 3625, 1), ("3707", 3695, 3722, 1), ("3880→3872", 3860, 3892, 1), ("3980", 3950, 4000, 1),
         ("4060", 4045, 4075, 1), ("4420", 4400, 4437, 1), ("4457", 4442, 4472, 1), ("5136", 5095, 5180, 1),
         ("5262", 5235, 5290, 1), ("6110/6152", 6100, 6165, 2), ("6241→6256", 6230, 6275, 1), ("6403→6389", 6375, 6420, 1),
         ("6481→6506", 6470, 6520, 1), ("6732", 6712, 6755, 1), ("6932→6908", 6890, 6950, 1), ("7216", 7200, 7238, 1),
         ("7363→7351", 7338, 7378, 1), ("7749→7778", 7730, 7795, 1), ("8031", 8010, 8055, 1), ("8320→8303", 8285, 8340, 1)]
# FEM 짝은 확실한 것만(53번 §2): h2 짝(합), 깨끗한 h3 홑 모드 6932 ↔ h3:m3n2(7093). 나머지는 그림에 FEM 범위 글로만 적는다.
FEM_PAIR = {"2066": "2134/2142", "6932→6908": "7093"}
# 기준 12자리에 두 성분이 다 보이는 짝 — 1번 깃에서 보이는 성분이 바뀐 것일 수 있다(주파수 변화로 읽지 않는다)
VIS = {"5136": "5127 and 5137 both in baseline", "6481→6506": "6481 and 6504–6510 both in baseline",
       "7749→7778": "7777 weak in baseline (vane 3)", "3516(new)": "3497 gone, 3516 new"}
FEM_TXT = ("FEM, 17.9 mg on the vane-4 line (7 shapes), G3 names: 3164 −0.38…−0.62, 3250 −0.18…−0.39, 3459 pair −0.6…−1.4, "
           "3639 pair −0.4, 3804 −0.17…−0.27,\n6558 −0.3…−0.5, 6817 −0.4…−0.8, 7093 −0.24…−0.30, 8321 −0.08…−0.13, "
           "h2 pair sum −0.06…−0.18 %. No FEM mode rises by more than +0.07 %.")


def main():
    D, _ = sp.per_hit(GROUPS, BANDS)
    D["t_s"] = D.file.str[9:15].map(lambda s: int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]))
    rows = []
    for name, lo, hi, k in BANDS:
        for n in sp.band_modes(name, k):
            for loc in V1:
                v = sp.compare(D, "A", "R", loc, n, common_slope=False)
                if v is None:
                    cnt = {g: int(((D.group == g) & (D["loc"] == loc) & (D["mode"] == n) & np.isfinite(D.f)).sum()) for g in "AR"}
                    rows.append(dict(mode=n, loc=loc[-5:], n_a=cnt["A"], n_b=cnt["R"]))
                    continue
                x = D[(D.group == "R") & (D["loc"] == loc) & (D["mode"] == n) & np.isfinite(D.f)]
                x = x[sp.robust_keep(x.f.values)]
                tr = np.nan
                if len(x) >= 6:
                    fz = x.f * (1 - v["slope_b_pct"] / 100 * np.log2(x.N / sp.NREF))
                    tr = 100 * np.polyfit((x.t_s - x.t_s.min()) / 60, fz / fz.mean(), 1)[0]
                rows.append(dict(mode=n, loc=loc[-5:], **v, trend_pct_per_min=tr))
    R = pd.DataFrame(rows)
    fem = fem_table()
    R["fem_name"] = R["mode"].map(FEM_PAIR)
    for c in (fem.columns if fem is not None else []):
        R[f"fem_{c}"] = R.fem_name.map(fem[c])
    R.to_csv(os.path.join(OUT, "R1_vane1_check.csv"), index=False, float_format="%.5g")
    cols = [c for c in ["mode", "loc", "f_a", "f_b", "d_pct", "se_pct", "n_a", "n_b", "sd_a", "sd_b", "z_ratio", "trend_pct_per_min",
                        "fem_name"] + [c for c in R.columns if c.startswith("fem_R")] if c in R.columns]
    print(R[cols].round(3).to_string(index=False))
    g = R[(R.se_pct < 0.05) & (R.n_a >= 5) & (R.n_b >= 5)] if "se_pct" in R else R.iloc[:0]
    print(f"\n잘 정해진 {len(g)}개: 중앙 {np.median(g.d_pct):+.3f} %, 감쇠 비 중앙 {np.median(g.z_ratio):.2f}")
    T = D.groupby("group")[["T0", "T1"]].mean()
    print("기온 ai0/ai1:", T.round(2).to_dict("index"))
    figure(R, fem)


def fem_table():
    fp = os.path.join(FEMD, "key.csv")
    if not os.path.exists(fp):
        return None
    K = pd.read_csv(fp).set_index("case")
    return K.T  # 행 = 관측량 이름, 열 = 경우


def figure(R, fem):
    order = list(dict.fromkeys(R["mode"]))
    xs = {n: i for i, n in enumerate(order)}
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw=dict(hspace=0.3))
    for loc, c, mk, dx in [("xi0.3", "C3", "o", -0.12), ("xi0.6", "C1", "s", 0.12)]:
        y = R[(R["loc"] == loc) & np.isfinite(R.get("d_pct", np.nan))]
        for _, r in y.iterrows():
            good = r.se_pct < 0.05 and r.n_a >= 5 and r.n_b >= 5
            ax[0].errorbar(xs[r["mode"]] + dx, r.d_pct, yerr=2 * r.se_pct, fmt=mk, color=c, mfc=c if good else "white", ms=5.5,
                           label=f"measured vane 1 {loc}")
            ax[1].plot(xs[r["mode"]] + dx, r.z_ratio, mk, color=c, mfc=c if good else "white", ms=5.5)
    if fem is not None:
        cases = [c for c in ["RB20", "RB10", "RV25", "RW20", "RW30", "RB20p5", "RB20m5"] if c in fem.columns]
        for n, fname in FEM_PAIR.items():
            if n not in xs or fname not in fem.index:
                continue
            v = fem.loc[fname, cases].astype(float)
            ax[0].plot([xs[n]] * 2, [v.min(), v.max()], "-", color="k", lw=5, alpha=0.25, solid_capstyle="butt",
                       label="FEM on vane 4, 17.9 mg (range over 7 shapes)")
            ax[0].plot(xs[n], v["RB20"], "k_", ms=16, mew=2.2, label="FEM RB20 (rim band ±10° on vane 4)")
            if fname == "2134/2142":
                ax[0].annotate("FEM: pair sum", (xs[n], v.min()), xytext=(6, -10), textcoords="offset points", fontsize=6.5)
    for n in VIS:
        if n in xs:
            ax[0].annotate("pair: seen\ncomponent\nchanged?", (xs[n], 0.55), fontsize=6.5, ha="center", color="0.3")
    ax[0].text(0.0, -0.13, FEM_TXT, transform=ax[0].transAxes, fontsize=7.5, va="top")
    h, l = ax[0].get_legend_handles_labels(); u = dict(zip(l, h))
    ax[0].legend(u.values(), u.keys(), fontsize=8, loc="lower left")
    ax[0].axhspan(-0.05, 0.05, color="0.85", zorder=0); ax[0].axhline(0, color="k", lw=0.6); ax[0].grid(alpha=0.3)
    ax[0].set_ylabel("frequency change [%] (±2 SE)")
    ax[0].set_title(f"(a) specimen 3, vane 1: R (S18, {MG} mg ground on vane 4) − baseline (S17), 15 N. → = peak that moved, "
                    "(new) = only after R. FEM drawn only for confident pairs", fontsize=8.5, loc="left")
    ax[1].axhline(1, color="k", lw=0.6); ax[1].set_ylabel("damping ratio R / baseline"); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(0, 3)
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, fontsize=7.5, rotation=60)
    fig.savefig(os.path.join(OUT, "R1_vane1_check.png"), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
