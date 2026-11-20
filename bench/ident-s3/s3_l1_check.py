"""2번 시편 L1(11-19 S16, 메모 s2_L34) — 깃 3–4 사이를 손으로 갈아 49.6 mg(41.3156 → 41.2660 g, 사용자 사진) 뒤 1번 깃 두 자리.

- 기준은 같은 날 40분 전의 A2-rh(S14)다. A2(11-16 S12)도 함께 본다.
- 깎은 뒤 여러 모드가 크게 옮겨 가서(3157 → 3112, 8309 → 8254 Hz 등) groove_state의 예측 대역(+쪽)으로는 맞지 않는다.
  그래서 두 상태의 봉우리를 모두 담는 대역을 손으로 정했다(평균 스펙트럼의 봉우리 목록, scratchpad l1s2/full_s16.py).
  '→'가 붙은 짝(3164 → 3112 등)은 봉우리가 사라지고 새로 생긴 자리를 짝지은 추정이다.
- 방법은 s1p_check와 같다(타격별 극점, 날짜마다 따로 둔 기울기로 15 N 환산, 감쇠는 타격별 ζ 중앙값).
- S16 안 시간 기울기(갈기·손 온기로 데워졌다면 식으며 오른다)도 본다.
출력: groove-test/out/L1_s16_check.csv·L1_s16_check.png
사용: python s3_l1_check.py"""
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
GROUPS = {"A2 11-16": (sp.S2F, [12], "20261116", LOCS),
          "A2-rh": (sp.S2F, [14], "20261119", LOCS),
          "L1": (sp.S2F, [16], "20261119", LOCS)}
PAIRS = [("A2-rh", "L1"), ("A2 11-16", "A2-rh")]
# (이름, 아래, 위, 극점 수) — 이름은 라이브러리 모드(종류는 groove_lib), '→'는 깎은 뒤 새 자리 추정
BANDS = [("2134/2142 h2", 2115, 2150, 2), ("3164→3112", 3100, 3170, 1), ("3250", 3205, 3245, 1), ("3496", 3480, 3510, 1),
         ("3589", 3565, 3595, 1), ("3748", 3715, 3750, 1), ("3804", 3755, 3800, 1), ("4225", 4190, 4225, 1),
         ("4967", 4900, 4945, 1), ("5617", 5555, 5590, 1), ("5677", 5635, 5690, 1), ("5745", 5700, 5735, 1),
         ("6817", 6770, 6805, 1), ("7093→7029", 7015, 7085, 1), ("8321→8254", 8240, 8320, 1)]
KIND = {"2134": "disk", "2142": "disk", "3164→3112": "panel", "3250": "mixed", "3496": "?", "3589": "none", "3748": "disk",
        "3804": "hub", "4225": "hub", "4967": "back", "5617": "back", "5677": "?", "5745": "back", "6817": "cand",
        "7093→7029": "cand", "8321→8254": "panel"}
# 같은 자리 L1 G3 FEM 예측(창 0.06 mm·37.4 mg, s3_pred_poles) × 49.6/37.4, R1 = (R − L6)/6 × 49.6/37.4 [%]
SC = 49.6 / 37.4
L1_PRED = {"2134": 0.001, "2142": 0.239, "3164→3112": 0.372, "3250": 0.182, "3748": -0.014, "3804": 0.068, "4225": -0.014,
           "4967": -0.014, "5617": -0.003, "5745": -0.019, "6817": -0.217, "7093→7029": -0.104, "8321→8254": -0.116}
R1_PRED = {"2134": -0.058, "2142": -0.058, "3164→3112": -0.735, "3250": -0.343, "3748": 0.046, "3804": -0.398, "4225": -0.013,
           "4967": -0.001, "5617": -0.020, "5745": -0.052, "6817": -0.551, "7093→7029": -0.586, "8321→8254": -0.226}


def main():
    D, spec = sp.per_hit(GROUPS, BANDS)
    D["t_s"] = D.file.str[9:15].map(lambda s: int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]))
    rows = []
    for name, lo, hi, k in BANDS:
        for n in sp.band_modes(name, k):
            for loc in LOCS:
                for ga, gb in PAIRS:
                    v = sp.compare(D, ga, gb, loc, n, common_slope=False)
                    if v is None:
                        continue
                    tr = np.nan
                    if gb == "L1":
                        x = D[(D.group == gb) & (D["loc"] == loc) & (D["mode"] == n) & np.isfinite(D.f)]
                        x = x[sp.robust_keep(x.f.values)]
                        if len(x) >= 6:
                            fz = x.f * (1 - v["slope_b_pct"] / 100 * np.log2(x.N / sp.NREF))
                            tr = 100 * np.polyfit((x.t_s - x.t_s.min()) / 60, fz / fz.mean(), 1)[0]
                    rows.append(dict(mode=n, kind=KIND.get(n, "?"), loc=loc.replace("back_vane1_", ""), pair=f"{gb} − {ga}", **v,
                                     trend_pct_per_min=tr, pred_L1=L1_PRED.get(n, np.nan) * SC, pred_R1=R1_PRED.get(n, np.nan) * SC))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "L1_s16_check.csv"), index=False, float_format="%.5g")
    T = D.groupby("group")[["T0", "T1"]].mean()
    print("기온 ai0/ai1:", {g: f"{T.loc[g, 'T0']:.2f}/{T.loc[g, 'T1']:.2f}" for g in T.index})
    for ga, gb in PAIRS:
        x = R[(R.pair == f"{gb} − {ga}")]
        print(f"\n[{gb} − {ga}]")
        cols = ["mode", "kind", "loc", "f_a", "f_b", "d_pct", "se_pct", "n_a", "n_b", "slope_a_pct", "slope_b_pct", "sd_a", "sd_b",
                "z_a", "z_b", "z_ratio"] + (["trend_pct_per_min", "pred_L1", "pred_R1"] if gb == "L1" else [])
        print(x[cols].round(4).to_string(index=False))
        g = x[(x.se_pct < 0.05) & (x.n_a >= 6) & (x.n_b >= 6)]
        if len(g):
            print(f"  잘 정해진 {len(g)}개: 중앙 {np.median(g.d_pct):+.3f} %, 감쇠 비 중앙 {np.median(g.z_ratio):.2f}, "
                  f"기울기 중앙 {np.median(g.slope_a_pct):+.3f} → {np.median(g.slope_b_pct):+.3f} %/힘 2배, 흩어짐 {np.median(g.sd_a):.3f} → {np.median(g.sd_b):.3f} %")
    figure(R)


def figure(R):
    x = R[R.pair == "L1 − A2-rh"]
    order = [n for n in ["2134", "2142", "3164→3112", "3250", "3496", "3589", "3748", "3804", "4225", "4967", "5617", "5677", "5745",
                         "6817", "7093→7029", "8321→8254"] if n in set(x["mode"])]
    xs = {n: i for i, n in enumerate(order)}
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for loc, c, mk, dx in [("xi0.3", "C3", "o", -0.12), ("xi0.6", "C1", "s", 0.12)]:
        y = x[x["loc"] == loc]
        for _, r in y.iterrows():
            good = r.se_pct < 0.05 and r.n_a >= 6 and r.n_b >= 6
            ax[0].errorbar(xs[r["mode"]] + dx, r.d_pct, yerr=2 * r.se_pct, fmt=mk, color=c, mfc=c if good else "white", ms=6,
                           label=f"measured, {loc}")
            ax[1].plot(xs[r["mode"]] + dx, r.z_ratio, mk, color=c, mfc=c if good else "white", ms=6, label=loc)
    for n in order:
        r = x[x["mode"] == n].iloc[0]
        if np.isfinite(r.pred_L1):
            ax[0].plot(xs[n], r.pred_L1, "k_", ms=18, mew=2.5, label="FEM: between-vane window (49.6 mg)")
            ax[0].plot(xs[n], r.pred_R1, "_", color="C2", ms=18, mew=2.5, label="FEM: on-vane window (49.6 mg, estimate)")
    h, l = ax[0].get_legend_handles_labels(); u = dict(zip(l, h))
    ax[0].legend(u.values(), u.keys(), fontsize=8, loc="upper right")
    ax[0].axhspan(-0.05, 0.05, color="0.85", zorder=0); ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("frequency change [%] (±2 SE)"); ax[0].grid(alpha=0.3)
    ax[0].set_title("(a) specimen 2, vane 1: L1 (S16, 49.6 mg ground at vanes 3–4) − A2-rh (S14, 40 min earlier), 15 N. "
                    "→ = peak that moved out of its old band", fontsize=8.5, loc="left")
    ax[1].axhline(1, color="k", lw=0.6); ax[1].set_ylabel("damping ratio L1 / A2-rh"); ax[1].grid(alpha=0.3)
    ax[1].set_title("(b) damping ratio (median of per-hit ζ)", fontsize=8.5, loc="left")
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels([f"{n}\n{KIND.get(n, '?')}" for n in order], fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "L1_s16_check.png"), dpi=110)


if __name__ == "__main__":
    main()
