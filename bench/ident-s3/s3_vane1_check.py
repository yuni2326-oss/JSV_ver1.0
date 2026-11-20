"""2번 시편 1번 깃 재측정(11-19 S14, 메모 s2 — L 가공 전)을 같은 두 자리(1번 깃 ξ0.3·ξ0.6)의 A2(11-16)·A1(11-11)과 맞댄다.

- 11-16 A2 뒤 시편을 내렸다가(11-19 1번 시편 사전 시험, 48번) 다시 단 것이다. 그래서 '다시 매달기 + 사흘'의 차이다.
  A2에서 새로 보인 상태(공통 −0.19 %, 감쇠 0.45배, 연화 없음 — 47번)가 그대로인지도 본다.
- 대역은 라이브러리 모드를 A2 측정값에 맞춘 것이다(groove_state --ref A2와 같다).
- 타격별 극점(s1p_check.per_hit)을 날짜마다 따로 둔 기울기로 15 N에 맞춰 비교한다. A1은 연화가 있고 A2는 없으므로 공통 기울기를 쓰지 않는다.
- 첫 쌍(A2 → 11-19)은 묶음 적합으로도 같은 차를 내서 맞대 본다.
출력: groove-test/out/V1-1119_check.csv·V1-1119_check.png
사용: python s3_vane1_check.py"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import groove_lib as gl  # noqa: E402
import s1p_check as sp  # noqa: E402

OUT = os.path.join(HERE, "..", "groove-test", "out")
LOCS = sp.LOCS
GROUPS = {"A1 11-11": (sp.S2F, [8, 9], "20261111", LOCS, "18:25"),     # A1.json과 같은 타격
          "A2 11-16": (sp.S2F, [12], "20261116", LOCS),
          "11-19": (sp.S2F, [14], "20261119", LOCS)}
PAIRS = [("A2 11-16", "11-19"), ("A1 11-11", "11-19"), ("A1 11-11", "A2 11-16")]
NREF = 15.0
WELL = ["2142", "3164", "3459", "3589", "3639", "3748", "3804", "4225", "5745", "6558", "7093", "7462"]   # 47번의 잘 정해진 12개


def bands_from_a2():
    A2 = json.load(open(os.path.join(OUT, "A2.json")))
    base = {m.name: m.f for m in gl.MODES}
    for n, v in A2["modes"].items():
        c = v.get("comb") or {}
        if np.isfinite(c.get("f", np.nan)):
            base[n] = c["f"]
    names = [m.name for m in gl.MODES]
    B = gl.plan_bands({n: 100 * (base[n] / gl.MODE[n].f - 1) for n in names}, names)
    return [("/".join(b["modes"]) + (" pair" if len(b["modes"]) > 1 else ""), b["lo"], b["hi"], len(b["modes"])) for b in B]


def main():
    bands = bands_from_a2()
    D, spec = sp.per_hit(GROUPS, bands)
    rows = []
    for name, lo, hi, k in bands:
        names = sp.band_modes(name, k)
        for loc in LOCS:
            for pi, (ga, gb) in enumerate(PAIRS):
                res = {n: sp.compare(D, ga, gb, loc, n, nref=NREF, common_slope=False) for n in names}
                pool = None
                if pi == 0 and all(v is not None for v in res.values()):
                    sa = float(np.mean([v["slope_a_pct"] for v in res.values()])) / 100
                    sb = float(np.mean([v["slope_b_pct"] for v in res.values()])) / 100
                    pa = sp.pooled(D, spec, ga, loc, name, lo, hi, k, sa, nboot=20, nref=NREF)
                    pb = sp.pooled(D, spec, gb, loc, name, lo, hi, k, sb, nboot=20, nref=NREF)
                    if pa is not None and pb is not None:
                        pool = (pa, pb)
                for j, n in enumerate(names):
                    v = res[n]
                    if v is None:
                        continue
                    pd_, ps_ = np.nan, np.nan
                    if pool:
                        pa, pb = pool
                        pd_ = 100 * (pb[0][j] / pa[0][j] - 1); ps_ = 100 * np.hypot(pa[1][j], pb[1][j]) / pa[0][j]
                    rows.append(dict(mode=n, kind=gl.MODE[n].kind, well=n in WELL, loc=loc.replace("back_vane1_", ""),
                                     pair=f"{gb} − {ga}", **v, pool_d_pct=pd_, pool_se_pct=ps_))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "V1-1119_check.csv"), index=False, float_format="%.5g")

    T = D.groupby("group")[["T0", "T1"]].mean()
    print("\n기온 ai0/ai1 [°C]:", {g: f"{T.loc[g, 'T0']:.2f}/{T.loc[g, 'T1']:.2f}" for g in T.index})
    q = D.drop_duplicates("file").groupby(["group", "loc"]).agg(hits=("file", "size"), N_med=("N", "median"), late=("late", "sum"))
    print(q.to_string())
    for ga, gb in PAIRS:
        x = R[(R.pair == f"{gb} − {ga}") & (R.se_pct < 0.05) & (R.n_a >= 6) & (R.n_b >= 6)]
        dT = T.loc[gb] - T.loc[ga]
        print(f"\n[{gb} − {ga}] 잘 정해진 모드·위치 {len(x)}개(표준오차 < 0.05 %, 타격 6개 이상). 기온 차 {dT.T0:+.2f}/{dT.T1:+.2f} K"
              f" → 기대 {-0.02 * dT.T0:+.3f} ~ {-0.02 * dT.T1:+.3f} %")
        cols = ["mode", "kind", "loc", "d_pct", "se_pct", "pool_d_pct", "slope_a_pct", "slope_b_pct", "N_a", "N_b", "sd_a", "sd_b",
                "z_a", "z_b", "z_ratio"]
        print(x[cols].round(4).to_string(index=False))
        if len(x):
            c = np.median(x.d_pct)
            w = x[x.well]
            print(f"  전체 중앙 {c:+.3f} %, 중앙을 뺀 rms {np.sqrt(np.mean((x.d_pct - c) ** 2)):.3f} %, 최대 |차 − 중앙| "
                  f"{np.max(np.abs(x.d_pct - c)):.3f} %")
            if len(w):
                cw = np.median(w.d_pct)
                print(f"  47번 12개 모드만: {len(w)}개, 중앙 {cw:+.3f} %, 중앙을 뺀 rms {np.sqrt(np.mean((w.d_pct - cw) ** 2)):.3f} %")
            print(f"  감쇠 비 중앙 {np.median(x.z_ratio):.2f}, 연화 기울기 중앙 {np.median(x.slope_a_pct):+.3f} → {np.median(x.slope_b_pct):+.3f}"
                  f" %/힘 2배, 흩어짐 중앙 {np.median(x.sd_a):.3f} → {np.median(x.sd_b):.3f} %")
    figure(R, T)


def figure(R, T):
    order = [m.name for m in gl.MODES if m.name in set(R["mode"])]
    xs = {n: i for i, n in enumerate(order)}
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    sty = {("11-19 − A2 11-16", "xi0.3"): ("C3", "o", -0.2), ("11-19 − A2 11-16", "xi0.6"): ("C1", "s", -0.07),
           ("11-19 − A1 11-11", "xi0.3"): ("0.45", "^", 0.07), ("11-19 − A1 11-11", "xi0.6"): ("0.7", "v", 0.2)}
    for (pair, loc), (c, mk, dx) in sty.items():
        x = R[(R.pair == pair) & (R["loc"] == loc)]
        for _, r in x.iterrows():
            good = r.se_pct < 0.05 and r.n_a >= 6 and r.n_b >= 6
            ax[0].errorbar(xs[r["mode"]] + dx, r.d_pct, yerr=2 * r.se_pct, fmt=mk, color=c, mfc=c if good else "white", ms=5,
                           label=f"{pair}, {loc}")
    h, l = ax[0].get_legend_handles_labels(); u = dict(zip(l, h))
    ax[0].legend(u.values(), u.keys(), fontsize=8, ncol=2, loc="upper left")
    ax[0].axhspan(-0.05, 0.05, color="0.85", zorder=0); ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylim(-0.5, 0.5); ax[0].grid(alpha=0.3); ax[0].set_ylabel("frequency change [%] (±2 SE)")
    ax[0].set_title("(a) specimen 2, vane 1: change at 15 N (own softening slope per date). grey band = re-hang floor ±0.05 %; "
                    "open = SE ≥ 0.05 % or < 6 hits", fontsize=8.5, loc="left")
    # (b) 감쇠, (c) 연화 — 날짜별(두 위치의 평균)
    G = {"A1 11-11": ("C0", "^"), "A2 11-16": ("C2", "s"), "11-19": ("C3", "o")}
    for g, (c, mk) in G.items():
        zs, ss = {}, {}
        for _, r in R.iterrows():
            ga, gb = r.pair.split(" − ")[1], r.pair.split(" − ")[0]
            if g == ga:
                zs.setdefault(r["mode"], []).append(r.z_a); ss.setdefault(r["mode"], []).append(r.slope_a_pct)
            if g == gb:
                zs.setdefault(r["mode"], []).append(r.z_b); ss.setdefault(r["mode"], []).append(r.slope_b_pct)
        ax[1].plot([xs[n] for n in zs], [100 * np.median(v) for v in zs.values()], mk, color=c, label=g, ms=6)
        ax[2].plot([xs[n] for n in ss], [np.median(v) for v in ss.values()], mk, color=c, label=g, ms=6)
    ax[1].set_ylabel("damping ζ [%]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 0.5)
    ax[1].set_title("(b) damping (median of per-hit ζ, both points)", fontsize=8.5, loc="left")
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_ylabel("softening [% per force doubling]"); ax[2].grid(alpha=0.3)
    ax[2].set_ylim(-0.3, 0.15); ax[2].legend(fontsize=8)
    ax[2].set_title("(c) force dependence (slope of per-hit f vs log2 force)", fontsize=8.5, loc="left")
    ax[2].set_xticks(range(len(order)))
    ax[2].set_xticklabels([f"{n}\n{gl.MODE[n].kind}" for n in order], fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "V1-1119_check.png"), dpi=110)


if __name__ == "__main__":
    main()
