"""1번 시편 사전 시험(48번) 2단계 — 깃 2–3 사이를 더 갈아 40.118 g(사용자, 11-19)으로 만든 뒤(S15, 메모 s1_l2) 주파수가 움직였는지.

- 주 비교: 같은 날 30분 전의 S13(약 1 mg 흠집 뒤, 메모 sample1_L1)과 S15. 둘 다 1번 깃 ξ0.3·ξ0.6이고, 사이에 내려서 갈고 다시 달았다.
- 참고: 10-28(1번 시편 마지막 측정, 손대기 전)과 S13·S15. 10-28 → S13의 차(48번 §7)가 '흠집 말고 다른 것'의 크기다.
- 방법은 s1p_check와 같다(타격별 극점, 15 N 공통 기울기 회귀, 묶음 적합 맞대기).
- 측정 중 흐름: 갈기·손으로 데워졌으면 식으며 주파수가 오른다. S15 안 시간 기울기를 본다.
출력: groove-test/s1-pretest/s1p_l2_check.csv·s1p-l2-check.png
사용: python s1p_l2_check.py"""
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

OUT = sp.OUT
GROUPS = {"10-28": sp.GROUPS["10-28"],
          "S13": (sp.S2F, [13], "20261119", sp.LOCS),
          "S15": (sp.S2F, [15], "20261119", sp.LOCS)}
PAIRS = [("S13", "S15"), ("10-28", "S15"), ("10-28", "S13")]
M0, M_S15 = 40.2059, 40.118          # 10-16 G1 질량, 11-19 사용자 보고
WIN_MG = 36.9


def main():
    D, spec = sp.per_hit(GROUPS, sp.BANDS)
    D["t_s"] = D.file.str[9:15].map(lambda s: int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]))
    rows = []
    for name, lo, hi, k in sp.BANDS:
        names = sp.band_modes(name, k)
        for loc in sp.LOCS:
            for pi, (ga, gb) in enumerate(PAIRS):
                res = {n: sp.compare(D, ga, gb, loc, n) for n in names}
                if any(v is None for v in res.values()):
                    continue
                pool = None
                if pi == 0:
                    sl = float(np.mean([v["slope_pct"] for v in res.values()])) / 100
                    pa = sp.pooled(D, spec, ga, loc, name, lo, hi, k, sl)
                    pb = sp.pooled(D, spec, gb, loc, name, lo, hi, k, sl)
                    if pa is not None and pb is not None:
                        pool = (pa, pb)
                for j, n in enumerate(names):
                    v = res[n]
                    pd_, ps_ = np.nan, np.nan
                    if pool:
                        pd_ = 100 * (pool[1][0][j] / pool[0][0][j] - 1)
                        ps_ = 100 * np.hypot(pool[0][1][j], pool[1][1][j]) / pool[0][0][j]
                    # S15 안 시간 기울기(%/분, 15 N 환산 뒤)
                    tr = np.nan
                    if pi == 0:
                        x = D[(D.group == gb) & (D["loc"] == loc) & (D["mode"] == n) & np.isfinite(D.f)]
                        if len(x) >= 6:
                            fz = x.f * (1 - v["slope_pct"] / 100 * np.log2(x.N / sp.NREF))
                            tr = 100 * np.polyfit((x.t_s - x.t_s.min()) / 60, fz / fz.mean(), 1)[0]
                    rows.append(dict(mode=n, loc=loc.replace("back_vane1_", ""), pair=f"{gb} − {ga}", **v, pool_d_pct=pd_,
                                     pool_se_pct=ps_, trend_pct_per_min=tr))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "s1p_l2_check.csv"), index=False, float_format="%.5g")

    T = D.groupby("group")[["T0", "T1"]].mean()
    print("\n기온 ai0/ai1 [°C]:", {g: f"{T.loc[g, 'T0']:.2f}/{T.loc[g, 'T1']:.2f}" for g in T.index})
    dm = 1e3 * (M0 - M_S15)
    print(f"깎은 양(10-16 질량 기준): {dm:.1f} mg = 창 하나(30°) 고르게 {dm / 6.22 * 0.01:.3f} mm, 0.06 mm 창의 {dm / WIN_MG:.2f}배")
    print("G3 FEM 창 하나(0.06 mm) 예측 × 그 배수(선형): 3164형 +%.2f, 3250형 +%.2f, 3459·3639형 +%.2f, h2 짝 평균 +%.2f(한 성분 최대 +%.2f), "
          "6817 %.2f, 7093 %.2f, 8321 %.2f %%" % tuple(v * dm / WIN_MG for v in (0.372, 0.182, 0.202, 0.12, 0.239, -0.217, -0.104, -0.116)))
    for ga, gb in PAIRS:
        x = R[(R.pair == f"{gb} − {ga}") & (R.se_pct < 0.05) & (R.n_a >= 6) & (R.n_b >= 6)]
        print(f"\n[{gb} − {ga}] 잘 정해진 모드·위치 {len(x)}개")
        cols = ["mode", "loc", "d_pct", "se_pct", "pool_d_pct", "slope_pct", "N_a", "N_b", "sd_a", "sd_b", "z_a", "z_b", "z_ratio"]
        if gb == "S15" and ga == "S13":
            cols.append("trend_pct_per_min")
        print(x[cols].round(4).to_string(index=False))
        if len(x):
            c = np.median(x.d_pct)
            print(f"  중앙 {c:+.3f} %, 중앙을 뺀 rms {np.sqrt(np.mean((x.d_pct - c) ** 2)):.3f} %, 감쇠 비 중앙 {np.median(x.z_ratio):.2f}")
    figure(R, spec, D)


def figure(R, spec, D):
    order = sp.mode_names()
    xs = {n: i for i, n in enumerate(order)}
    fig = plt.figure(figsize=(13, 9.5))
    ax = fig.add_subplot(2, 1, 1)
    sty = {("S15 − S13", "xi0.3"): ("C3", "o", -0.15), ("S15 − S13", "xi0.6"): ("C1", "s", 0.0),
           ("S13 − 10-28", "xi0.3"): ("0.45", "^", 0.12), ("S13 − 10-28", "xi0.6"): ("0.7", "v", 0.2)}
    for (pair, loc), (c, mk, dx) in sty.items():
        x = R[(R.pair == pair) & (R["loc"] == loc)]
        for _, r in x.iterrows():
            good = r.se_pct < 0.05 and r.n_a >= 6 and r.n_b >= 6
            ax.errorbar(xs[r["mode"]] + dx, r.d_pct, yerr=2 * r.se_pct, fmt=mk, color=c, mfc=c if good else "white", ms=5,
                        label=f"{pair}, {loc}")
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), fontsize=8, ncol=2, loc="lower left")
    ax.axhspan(-0.05, 0.05, color="0.85", zorder=0); ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, fontsize=8)
    ax.set_ylabel("frequency change [%] (±2 SE)"); ax.grid(alpha=0.3)
    ax.set_title("(a) specimen 1, vane 1, 15 N: S15 (40.118 g, about 88 mg ground between vanes 2–3) − S13 (about 1 mg), "
                 "and S13 − 10-28 (no cut; other changes)", fontsize=8.5, loc="left")
    # (b)–(c) h2 짝 대역의 평균 스펙트럼(감쇠 응답, 힘으로 나눔)
    for i, loc in enumerate(sp.LOCS):
        axb = fig.add_subplot(2, 2, 3 + i)
        for g, c in [("10-28", "0.6"), ("S13", "C0"), ("S15", "C3")]:
            files = D[(D.group == g) & (D["loc"] == loc)].drop_duplicates("file")
            acc = None
            for _, r in files.iterrows():
                fr, X = spec[(r.file, "2041/2063 h2")]
                p = np.abs(X) ** 2 / r.N ** 2
                acc = p if acc is None else acc + p
            if acc is not None:
                s = (fr > 2015) & (fr < 2090)
                axb.plot(fr[s], 10 * np.log10(acc[s] / len(files)), color=c, label=g)
        axb.set_title(f"({'bc'[i]}) h2 pair band, {loc.replace('back_vane1_', 'vane 1 ')}", fontsize=8.5, loc="left")
        axb.set_xlabel("Hz"); axb.grid(alpha=0.3); axb.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s1p-l2-check.png"), dpi=110)


if __name__ == "__main__":
    main()
