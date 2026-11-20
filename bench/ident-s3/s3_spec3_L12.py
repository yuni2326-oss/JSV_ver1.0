"""3번 시편 L(S19, 깃 3–4 사이 테두리 17.6 mg) 12자리 — R(S18) 12자리·기준(S17) 12자리와 비교(57번).

- S19는 1번 깃 23:55–23:57(깎은 직후), 깃 2–6 00:00–00:14(날짜가 11-20으로 바뀜 — 날짜로 거르지 않고 세션 번호로 고른다).
  R(S18)은 1번 깃이 깎은 직후, 깃 2–6이 20–35분 뒤였다. 데워짐 치우침(55번, 약 −0.05 %)을 보려고 자리별 측정 시각을 함께 남긴다.
- 대역은 세 상태의 12자리 봉우리를 모두 담게 손으로 정했다. 3.45–3.65 kHz 패널 무리는 상태마다 봉우리가 옮겨 다녀 대역을 잘게 나눴다
  (한 상태에만 있는 대역은 비교 값이 없다).
- 홑 모드 짝은 확실한 것만 FEM과 맞댄다. 3033·3149(h0·h3 섞인 두 모드)는 합으로 FEM 3164 + 3250과 맞댄다(56번).
출력: groove-test/s3/L12_check.csv·L12_summary.csv·L12_joint.csv·L12_check.png
사용: python s3_spec3_L12.py --specL <S19 npz>   (s2spec.py 출력, 없으면 공동 적합을 건너뛴다)"""
import argparse
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
LOC12 = r12.LOC12
GROUPS = {"A": (sp.S2F, [17], "20261119", LOC12), "R": (sp.S2F, [18], "20261119", LOC12), "L": (sp.S2F, [19], None, LOC12)}
BANDS = [("2066/2077 h2", 2050, 2092, 2), ("3033", 3005, 3060, 1), ("3149", 3120, 3185, 1), ("3296", 3265, 3325, 1),
         ("3467", 3455, 3482, 1), ("3497/3517", 3486, 3527, 1), ("3541", 3527, 3550, 1), ("3555/3563", 3550, 3576, 1),
         ("3592", 3580, 3601, 1), ("3612", 3601, 3626, 1), ("3707", 3695, 3722, 1), ("3880", 3860, 3892, 1),
         ("3980", 3950, 4000, 1), ("4060", 4045, 4075, 1), ("4420", 4400, 4437, 1), ("4457", 4442, 4472, 1),
         ("5136", 5095, 5180, 1), ("5262", 5235, 5290, 1), ("5312", 5290, 5340, 1), ("6110/6152", 6100, 6165, 2),
         ("6241→6266", 6230, 6285, 1), ("6403", 6375, 6430, 1), ("6481", 6470, 6494, 1), ("6506", 6495, 6525, 1),
         ("6732", 6712, 6755, 1), ("6932", 6885, 6950, 1), ("7216", 7195, 7245, 1), ("7363", 7338, 7380, 1),
         ("7749", 7725, 7795, 1), ("8031", 8005, 8055, 1), ("8195", 8175, 8215, 1), ("8320", 8285, 8340, 1)]
JFITS = [((2040, 2100), 2), ((3000, 3070), 1), ((3130, 3190), 1), ((3250, 3360), 1), ((3455, 3485), 1), ((3527, 3556), 1),
         ((3578, 3602), 1), ((3602, 3630), 1), ((3690, 3730), 1), ((3930, 4030), 1), ((6700, 6760), 1), ((6880, 6960), 1),
         ((7190, 7255), 1), ((7720, 7800), 1), ((7990, 8080), 1)]
TENT = {"2066": ("h2:m2n0", "+"), "3980": ("h1:m1n1#2", "-"), "6932": ("h3:m3n2", None)}
FAMILY = (("3033", "3149"), ("h3:m3n0", "h0:m0n1"))
PAIRS = (("R", "L"), ("A", "L"))


def per_loc_time(D):
    """상태·자리별 측정 시각(자정 넘김을 이어 붙인 분)."""
    t = D.file.str[9:15].map(lambda s: int(s[:2]) * 60 + int(s[2:4]) + int(s[4:6]) / 60)
    day = D.file.str[:8]
    t = t + np.where(day > "20261119", 24 * 60, 0)
    return D.assign(t_min=t).groupby(["group", "loc"]).t_min.median()


def compare_all(D):
    T = per_loc_time(D)
    rows = []
    for name, lo, hi, k in BANDS:
        for n in sp.band_modes(name, k):
            for loc in LOC12:
                row = dict(mode=n, loc=loc, vane=int(loc[9]), xi=loc[-3:], t_L=T.get(("L", loc), np.nan), t_R=T.get(("R", loc), np.nan))
                for ga, gb in PAIRS:
                    v = sp.compare(D, ga, gb, loc, n, common_slope=False)
                    tag = ga + gb
                    if v:
                        row.update({f"{tag}_d": v["d_pct"], f"{tag}_se": v["se_pct"], f"{tag}_na": v["n_a"], f"{tag}_nb": v["n_b"],
                                    f"{tag}_z": v["z_ratio"], "f_L": v["f_b"]})
                rows.append(row)
    return pd.DataFrame(rows)


def good(R, tag):
    return R[np.isfinite(R.get(f"{tag}_d", np.nan)) & (R[f"{tag}_se"] < 0.08) & (R[f"{tag}_na"] >= 5) & (R[f"{tag}_nb"] >= 5)]


def summarize(R):
    S = []
    for name, lo, hi, k in BANDS:
        for n in sp.band_modes(name, k):
            row = dict(mode=n)
            for tag in ("RL", "AL"):
                x = good(R[R["mode"] == n], tag)
                v1, v26 = x[x.vane == 1], x[x.vane != 1]
                row.update({f"{tag}_n": len(x), f"{tag}_med": np.median(x[f"{tag}_d"]) if len(x) else np.nan,
                            f"{tag}_v1": np.median(v1[f"{tag}_d"]) if len(v1) else np.nan,
                            f"{tag}_v26": np.median(v26[f"{tag}_d"]) if len(v26) else np.nan,
                            f"{tag}_lo": v26[f"{tag}_d"].min() if len(v26) else np.nan,
                            f"{tag}_hi": v26[f"{tag}_d"].max() if len(v26) else np.nan,
                            f"{tag}_z": np.median(x[f"{tag}_z"]) if len(x) else np.nan})
                if len(x):
                    row["f_L"] = np.median(x.f_L)
            S.append(row)
    return pd.DataFrame(S)


def drift(R):
    """자리별(측정 순서) 홑 모드 L − R·L − 기준 중앙 — 데워짐이 식는지 본다."""
    singles = ["3033", "3149", "3296", "3980", "4060", "4457", "6732", "6932", "8031", "8320", "7363"]
    out = []
    for loc in LOC12:
        x = R[(R["loc"] == loc) & R["mode"].isin(singles)]
        row = dict(loc=loc, t_L=x.t_L.iloc[0] if len(x) else np.nan)
        for tag in ("RL", "AL"):
            g = good(x, tag)
            row[f"{tag}_med"] = np.median(g[f"{tag}_d"]) if len(g) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def fem_vals(case):
    fp = os.path.join(FEML, "summary.csv")
    if not os.path.exists(fp):
        return {}, np.nan
    S = pd.read_csv(fp)
    out = {}
    for n, (lab, comp) in TENT.items():
        r = S[(S.case == case) & (S.label == lab)]
        if len(r):
            r = r.iloc[0]
            out[n] = r.df1 if comp is None else (max(r.df1, r.df2) if comp == "+" else min(r.df1, r.df2))
    fam = sum(float(S[(S.case == case) & (S.label == l)].df1.iloc[0]) for l in FAMILY[1])
    return out, fam


def figure(R, S, Dr):
    order = list(dict.fromkeys(R["mode"]))
    xs = {n: i for i, n in enumerate(order)}
    fig = plt.figure(figsize=(15, 12))
    gsp = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.55], hspace=0.38)
    cmap = plt.get_cmap("viridis")
    for i, (tag, title) in enumerate((("RL", "(a) L − R, all 12 points (17.6 mg at the rim between vanes 3–4). FEM: photo shape L17e"),
                                      ("AL", "(b) cumulative L − baseline, all 12 points (R 17.9 mg + L 17.6 mg)"))):
        ax = fig.add_subplot(gsp[i])
        for _, r in good(R, tag).iterrows():
            c = cmap((r.vane - 1) / 5)
            ax.plot(xs[r["mode"]] + (r.vane - 3.5) * 0.08, r[f"{tag}_d"], "o" if r.xi == "0.3" else "s", color=c, ms=4.5,
                    mfc=c if r.xi == "0.6" else "white")
        if tag == "RL":
            fv, fam = fem_vals("L17e")
            for n, v in fv.items():
                if n in xs:
                    ax.plot(xs[n], v, "k_", ms=15, mew=2.4, label="FEM L17e (tentative pairs)")
            fam_meas = [S.set_index("mode").loc[m, "RL_v26"] for m in FAMILY[0]]
            ax.text(0.01, 0.04, f"3033 + 3149 sum (vanes 2–6): measured {sum(fam_meas):+.2f} %;  FEM 3164 + 3250 (L17e) {fam:+.2f} %",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        for v in range(1, 7):
            ax.plot([], [], "s", color=cmap((v - 1) / 5), label=f"vane {v}")
        h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
        ax.legend(u.values(), u.keys(), fontsize=7.5, ncol=4, loc="upper center")
        ax.axhspan(-0.05, 0.05, color="0.85", zorder=0); ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3)
        ax.set_ylabel("frequency change [%]")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order if i == 1 else [], fontsize=7, rotation=60)
    ax = fig.add_subplot(gsp[2])
    ax.plot(Dr.t_L - Dr.t_L.min(), Dr.RL_med, "o-", color="C0", label="L − R (singlet median)")
    ax.plot(Dr.t_L - Dr.t_L.min(), Dr.AL_med, "s-", color="C3", label="L − baseline (singlet median)")
    for _, r in Dr.iterrows():
        ax.annotate(r["loc"][5:].replace("_xi", " ξ"), (r.t_L - Dr.t_L.min(), r.AL_med), fontsize=6, xytext=(2, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3)
    ax.set_xlabel("minutes after the first L point (vane 1 at 23:55)"); ax.set_ylabel("median [%]")
    ax.set_title("(c) per-point median over well-determined singlet modes, in measuring order (warm specimen cooling?)", fontsize=9, loc="left")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "L12_check.png"), dpi=105, bbox_inches="tight")


def main(specL):
    D, _ = sp.per_hit(GROUPS, BANDS)
    R = compare_all(D)
    R.to_csv(os.path.join(OUT, "L12_check.csv"), index=False, float_format="%.5g")
    S = summarize(R)
    S.to_csv(os.path.join(OUT, "L12_summary.csv"), index=False, float_format="%.5g")
    print(S.round(3).to_string(index=False))
    Dr = drift(R)
    print(Dr.round(3).to_string(index=False))
    T0 = D.groupby("group")[["T0", "T1"]].mean()
    print("기온 ai0/ai1:", T0.round(2).to_dict("index"))
    if specL:
        keep, r12.JFITS = r12.JFITS, JFITS          # r12.joint는 모듈의 JFITS 대역을 쓴다
        try:
            J = r12.joint(specL, "L")
        finally:
            r12.JFITS = keep
        J.to_csv(os.path.join(OUT, "L12_joint.csv"), index=False, float_format="%.4g")
        print(J.round(3).to_string(index=False))
    figure(R, S, Dr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--specL", default="")
    ap.add_argument("--fig-only", action="store_true", help="저장된 csv로 그림만 다시 그린다")
    a = ap.parse_args()
    if a.fig_only:
        R = pd.read_csv(os.path.join(OUT, "L12_check.csv"), dtype={"mode": str, "xi": str})
        S = pd.read_csv(os.path.join(OUT, "L12_summary.csv"), dtype={"mode": str})
        figure(R, S, drift(R))
    else:
        main(a.specL)
