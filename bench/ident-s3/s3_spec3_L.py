"""3번 시편 L(11-19 S19, 메모 s3_L) — 깃 3–4 사이 테두리를 17.6 mg(39923.5 → 39905.9 mg) 깎은 뒤 1번 깃 두 자리(56번).

비교
- L − R: R 상태의 1번 깃(S18 22:55–22:58)과 맞댄다. 두 측정 모두 깎은 직후라 데워짐 치우침(55번: 약 −0.05 %)이 서로 비슷하게 빠진다.
- L − 기준: S17의 1번 깃과 맞댄 누적(R + L) 변화. 데워짐 치우침이 그대로 남는다.
- L만 추정(홑 모드만): (L − 기준) − R 효과(55번 12자리 중 깃 2–6 중앙, 데워짐 없음).
대역은 세 상태의 1번 깃 봉우리를 모두 담게 손으로 정했다. 방법은 s3_spec3_R과 같다(타격별 극점, 상태마다 따로 둔 힘 기울기, 15 N).
FEM(s3_l_grind, 사진 모양 17.6 mg)은 55번 추정 짝(s3_spec3_R12.TENT)으로 맞댄다.
출력: groove-test/s3/L1_vane1_check.csv·L1_vane1_check.png
사용: python s3_spec3_L.py"""
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
import s3_spec3_R12 as r12  # noqa: E402

OUT = os.path.join(HERE, "..", "groove-test", "s3")
FEML = os.path.join(HERE, "..", "fe-nominal", "s3_l_grind")
V1 = ["back_vane1_xi0.3", "back_vane1_xi0.6"]
GROUPS = {"A": (sp.S2F, [17], "20261119", V1), "R": (sp.S2F, [18], "20261119", V1, None, "23:00"),
          "L": (sp.S2F, [19], "20261119", V1)}
MG = 17.6
BANDS = [("2066/2077 h2", 2050, 2092, 2), ("3033", 3005, 3060, 1), ("3149", 3120, 3185, 1), ("3296", 3265, 3325, 1),
         ("3467", 3455, 3480, 1), ("3497", 3488, 3507, 1), ("3517", 3507, 3530, 1), ("3554→3560", 3540, 3575, 1),
         ("35xx(3580–3601)", 3578, 3601, 1), ("36xx(3601–3628)", 3601, 3628, 1), ("3707", 3695, 3722, 1),
         ("3880", 3860, 3892, 1), ("3980", 3950, 4000, 1), ("4060", 4045, 4075, 1), ("4457", 4442, 4472, 1),
         ("5136", 5095, 5180, 1), ("5262", 5235, 5290, 1), ("6110/6152", 6100, 6165, 2), ("6241→6266", 6230, 6285, 1),
         ("6403", 6375, 6430, 1), ("6481", 6470, 6494, 1), ("6506", 6495, 6525, 1), ("6732", 6712, 6755, 1),
         ("6932", 6885, 6950, 1), ("7216", 7195, 7245, 1), ("7363", 7338, 7380, 1), ("7749", 7725, 7795, 1),
         ("8031", 8005, 8055, 1), ("8320", 8285, 8340, 1)]
# 55번 짝 이름(대역 이름이 조금 다르다) → s3_spec3_R12의 이름
R12_NAME = {"3554→3560": "3554→3562", "6241→6266": "6241→6266", "6403": "6411→6403", "6932": "6932→6910",
            "7216": "7216→7228", "7749": "7749→7776", "8320": "8320→8310"}
# 홑 모드 짝은 확실한 것만. 3033·3149는 h0·h3가 섞인 두 모드라(53번 공동 적합) 한 모드씩 짝짓지 않고 두 모드의 합을
# FEM 3164 + 3250 합과 맞댄다(섞여도 두 모드 합은 1차로 같다).
TENT = {"2066": ("h2:m2n0", "+"), "3980": ("h1:m1n1#2", "-"), "6932": ("h3:m3n2", None)}
FAMILY = (("3033", "3149"), ("h3:m3n0", "h0:m0n1"))


def fem_l(case):
    fp = os.path.join(FEML, "summary.csv")
    if not os.path.exists(fp):
        return {}
    S = pd.read_csv(fp)
    out = {}
    for n, (lab, comp) in TENT.items():
        r = S[(S.case == case) & (S.label == lab)]
        if r.empty:
            continue
        r = r.iloc[0]
        out[n] = r.df1 if comp is None else (max(r.df1, r.df2) if comp == "+" else min(r.df1, r.df2))
    return out


def family_text(T):
    """3033 + 3149 합: 측정(L − R, 두 자리)과 FEM(L17e·L17f·L17v·L17c의 3164 + 3250)."""
    meas = []
    for loc in ("xi0.3", "xi0.6"):
        v = [T[(T["mode"] == m) & (T["loc"] == loc)].LR_d.iloc[0] for m in FAMILY[0]]
        meas.append(sum(v))
    fp = os.path.join(FEML, "summary.csv")
    fem = {}
    if os.path.exists(fp):
        S = pd.read_csv(fp)
        for c in S.case.unique():
            fem[c] = sum(float(S[(S.case == c) & (S.label == l)].df1.iloc[0]) for l in FAMILY[1])
    return meas, fem


def main():
    D, _ = sp.per_hit(GROUPS, BANDS)
    R12 = pd.read_csv(os.path.join(OUT, "R12_summary.csv"), dtype={"mode": str}).set_index("mode")
    rows = []
    for name, lo, hi, k in BANDS:
        for n in sp.band_modes(name, k):
            for loc in V1:
                row = dict(mode=n, loc=loc[-5:])
                for ga, tag in (("R", "LR"), ("A", "LA")):
                    v = sp.compare(D, ga, "L", loc, n, common_slope=False)
                    if v:
                        row.update({f"{tag}_d": v["d_pct"], f"{tag}_se": v["se_pct"], f"{tag}_na": v["n_a"], f"{tag}_nb": v["n_b"],
                                    f"{tag}_z": v["z_ratio"]})
                        row["f_L"] = v["f_b"]
                rn = R12_NAME.get(n, n)
                row["R_v26"] = R12.loc[rn, "d_v26"] if rn in R12.index else np.nan
                rows.append(row)
    T = pd.DataFrame(rows)
    T["L_only"] = T.get("LA_d", np.nan) - T["R_v26"]
    T.to_csv(os.path.join(OUT, "L1_vane1_check.csv"), index=False, float_format="%.5g")
    cols = [c for c in ["mode", "loc", "f_L", "LR_d", "LR_se", "LR_na", "LR_nb", "LR_z", "LA_d", "LA_se", "R_v26", "L_only"] if c in T]
    print(T[cols].round(3).to_string(index=False))
    meas, fem = family_text(T)
    print("3033 + 3149 합 L − R:", [round(m, 3) for m in meas], " FEM 3164 + 3250:", {c: round(v, 3) for c, v in fem.items()})
    T0 = D.groupby("group")[["T0", "T1"]].mean()
    print("기온 ai0/ai1:", T0.round(2).to_dict("index"))
    figure(T)


def figure(T):
    order = list(dict.fromkeys(T["mode"]))
    xs = {n: i for i, n in enumerate(order)}
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw=dict(hspace=0.25))
    for loc, c, mk, dx in [("xi0.3", "C3", "o", -0.12), ("xi0.6", "C1", "s", 0.12)]:
        y = T[T["loc"] == loc]
        for _, r in y.iterrows():
            if np.isfinite(r.get("LR_d", np.nan)):
                good = r.LR_se < 0.05 and r.LR_na >= 5 and r.LR_nb >= 5
                ax[0].errorbar(xs[r["mode"]] + dx, r.LR_d, yerr=2 * r.LR_se, fmt=mk, color=c, mfc=c if good else "white", ms=5.5,
                               label=f"measured L − R, vane 1 {loc}")
            if np.isfinite(r.get("LA_d", np.nan)):
                good = r.LA_se < 0.05 and r.LA_na >= 5 and r.LA_nb >= 5
                ax[1].errorbar(xs[r["mode"]] + dx, r.LA_d, yerr=2 * r.LA_se, fmt=mk, color=c, mfc=c if good else "white", ms=5.5,
                               label=f"measured L − baseline, vane 1 {loc}")
    for case, col, lab in (("L17e", "k", "FEM L, photo shape (rim band 17°, 3° toward vane 3)"),
                           ("L17v", "0.55", "FEM L, bevel to ξ0.8")):
        for n, v in fem_l(case).items():
            if n in xs:
                ax[0].plot(xs[n], v, "_", color=col, ms=15, mew=2.4, label=lab)
    meas, fem = family_text(T)
    if fem:
        ax[0].text(0.995, 0.97, "3033 + 3149 (h0/h3 mixed pair, sum): measured " + " / ".join(f"{m:+.2f}" for m in meas)
                   + " %;  FEM 3164 + 3250: " + ", ".join(f"{c} {v:+.2f}" for c, v in fem.items()) + " %\n"
                   "FEM panel pairs rise strongly (3459 +0.4, 3639 +1.0…+1.3 %): the 3.5–3.6 kHz cluster moves up",
                   transform=ax[0].transAxes, fontsize=7.5, ha="right", va="top",
                   bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    for a, t in ((ax[0], f"(a) specimen 3, vane 1: L (S19, {MG} mg at the rim between vanes 3–4) − R (S18 22:55). Both right after "
                         "grinding. FEM on tentative pairs (doc 55)"),
                 (ax[1], "(b) cumulative L − baseline (S17), vane 1 (R 17.9 mg + L 17.6 mg; baseline was not warm)")):
        h, l = a.get_legend_handles_labels(); u = dict(zip(l, h))
        a.legend(u.values(), u.keys(), fontsize=7.5, loc="lower center")
        a.axhspan(-0.05, 0.05, color="0.85", zorder=0); a.axhline(0, color="k", lw=0.6); a.grid(alpha=0.3)
        a.set_ylabel("frequency change [%] (±2 SE)")
        a.set_title(t, fontsize=8.5, loc="left")
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, fontsize=7.5, rotation=60)
    fig.savefig(os.path.join(OUT, "L1_vane1_check.png"), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
