"""3번 시편 R(11-19 S18, 깃 4 위 17.9 mg) 12자리 대 기준(S17) 12자리(55번).

- S18은 1번 깃 두 자리(22:55–22:58)와 깃 2–6(23:16–23:30)으로 나뉘어 있다. 깎은 직후 데워짐이 있었다면 1번 깃만 낮게 읽힌다.
- 대역은 두 상태의 12자리 평균 스펙트럼 봉우리를 모두 담게 손으로 정했다. '→'는 옮겨 간 봉우리 짝(추정), '(new)'는 R에만 있는 봉우리.
- 자리마다 s1p_check.compare(상태마다 따로 둔 힘 기울기, 15 N)로 차를 내고, 모드마다 12자리의 중앙·범위를 낸다.
- 원주 차수: 두 상태 모두 깨끗한 타격 H1의 깃 6곳 공동 적합(ident-w22/w22joint) — ξ0.3·ξ0.6 따로.
- FEM(s3_r_grind)은 사진 모양(RS25b: 깃 4 선 끝에서 깃 5 쪽 25°, 테두리 경사면)과 깃 선 가운데(RB20)를 추정 짝(TENT)으로 겹친다.
출력: groove-test/s3/R12_check.csv·R12_summary.csv·R12_joint.csv·R12_check.png
사용: python s3_spec3_R12.py --specA <S17 npz> --specR <S18 npz>   (s2spec.py 출력, 없으면 공동 적합을 건너뛴다)"""
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
sys.path.insert(0, os.path.join(HERE, "..", "ident-w22"))
import s1p_check as sp  # noqa: E402

OUT = os.path.join(HERE, "..", "groove-test", "s3")
FEMD = os.path.join(HERE, "..", "fe-nominal", "s3_r_grind")
LOC12 = [f"back_vane{v}_xi{x}" for v in range(1, 7) for x in ("0.3", "0.6")]
GROUPS = {"A": (sp.S2F, [17], "20261119", LOC12), "R": (sp.S2F, [18], "20261119", LOC12)}
BANDS = [("2066/2077 h2", 2050, 2092, 2), ("3033", 3005, 3060, 1), ("3149", 3120, 3180, 1), ("3296", 3265, 3325, 1),
         ("3467", 3455, 3480, 1), ("3497", 3488, 3507, 1), ("3517(new)", 3507, 3530, 1), ("3554→3562", 3540, 3575, 1),
         ("3613→3595", 3580, 3628, 1), ("3707", 3695, 3722, 1), ("3880", 3860, 3892, 1), ("3980", 3950, 4000, 1),
         ("4060", 4045, 4075, 1), ("4420", 4400, 4437, 1), ("4457", 4442, 4472, 1), ("5136", 5095, 5180, 1),
         ("5262", 5235, 5290, 1), ("5312", 5290, 5340, 1), ("6110/6152", 6100, 6165, 2), ("6241→6266", 6230, 6285, 1),
         ("6411→6403", 6380, 6430, 1), ("6481", 6470, 6494, 1), ("6506", 6495, 6525, 1), ("6732", 6712, 6755, 1),
         ("6932→6910", 6890, 6950, 1), ("7216→7228", 7200, 7245, 1), ("7363", 7338, 7380, 1), ("7749→7776", 7730, 7795, 1),
         ("8031", 8010, 8055, 1), ("8195", 8175, 8215, 1), ("8320→8310", 8285, 8340, 1)]
JFITS = [((2040, 2100), 2), ((3000, 3070), 1), ((3110, 3190), 1), ((3250, 3360), 1), ((3480, 3506), 1), ((3507, 3535), 1),
         ((3535, 3578), 1), ((3578, 3640), 1), ((3690, 3730), 1), ((3930, 4030), 1), ((4440, 4490), 1), ((5060, 5200), 1),
         ((6460, 6530), 2), ((6700, 6770), 1), ((6880, 6960), 1), ((7190, 7255), 1), ((7330, 7400), 1), ((7720, 7800), 1),
         ((7990, 8080), 1), ((8280, 8350), 1)]
# 추정 짝(3번 시편 모드 → FEM 관측량, 짝이면 '+'·'−' 성분) — 그림에 사진 모양(RS25b)과 깃 선 가운데(RB20) FEM을 겹친다
TENT = {"2066": ("h2:m2n0", "+"), "2077": ("h2:m2n0", "-"), "3033": ("h3:m3n0", None), "3149": ("h0:m0n1", None),
        "3613→3595": ("h1:m1n1", "-"), "3980": ("h1:m1n1#2", "-"), "6932→6910": ("h3:m3n2", None), "8320→8310": ("h3:m3n1#2", None)}
FEM_TXT = ("FEM, 17.9 mg near vane 4 (15 shapes: centred, widened, one-sided as in the photo): every mode falls or stays, "
           "except one h2 component (+0.04…+0.08 %) and the mixed h0:m6n0#2 (+0.15…+0.19 %).\n"
           "Measured rises of +0.17…+0.29 % (3554→3562, 5136, 7216→7228, 7749→7776) are not reproduced by material removal alone.")


def compare_all(D):
    rows = []
    for name, lo, hi, k in BANDS:
        for n in sp.band_modes(name, k):
            for loc in LOC12:
                v = sp.compare(D, "A", "R", loc, n, common_slope=False)
                cnt = {g: int(((D.group == g) & (D["loc"] == loc) & (D["mode"] == n) & np.isfinite(D.f)).sum()) for g in "AR"}
                rows.append(dict(mode=n, loc=loc, vane=int(loc[9]), xi=loc[-3:], n_ok_A=cnt["A"], n_ok_R=cnt["R"], **(v or {})))
    return pd.DataFrame(rows)


def summarize(R):
    S = []
    for name, lo, hi, k in BANDS:
        for n in sp.band_modes(name, k):
            x = R[(R["mode"] == n) & np.isfinite(R.get("d_pct", np.nan)) & (R.se_pct < 0.08) & (R.n_a >= 5) & (R.n_b >= 5)]
            y = R[R["mode"] == n]
            row = dict(mode=n, n_pos=len(x), n_A=int((y.n_ok_A >= 5).sum()), n_R=int((y.n_ok_R >= 5).sum()))
            if len(x):
                v1 = x[x.vane == 1]
                v26 = x[x.vane != 1]
                row.update(d_med=np.median(x.d_pct), d_lo=x.d_pct.min(), d_hi=x.d_pct.max(), z_ratio=np.median(x.z_ratio),
                           f_A=np.median(x.f_a), f_R=np.median(x.f_b),
                           d_v1=np.median(v1.d_pct) if len(v1) else np.nan, d_v26=np.median(v26.d_pct) if len(v26) else np.nan)
            S.append(row)
    return pd.DataFrame(S)


def joint(spec_npz, tag):
    from w22joint import joint_fit, pattern
    Z = np.load(spec_npz)
    fr, F, X = Z["fr"], Z["F"].astype(complex), Z["X"].astype(complex)
    use = ~Z["late"] & (Z["N"] >= 5)
    out = []
    for xi in (0.3, 0.6):
        Hs = []
        for v in range(1, 7):
            u = np.where(use & (Z["vane"] == v) & (Z["xi"] == xi))[0]
            Hs.append((np.conj(F[u]) * X[u]).sum(0) / (np.abs(F[u]) ** 2).sum(0))
        for (lo, hi), k in JFITS:
            fk, zk, Rr, err = joint_fit(fr, Hs, lo, hi, k)
            for j in range(k):
                mpc, a, P, th0, _ = pattern(Rr[:, j])
                hb = int(np.argmax(P))
                out.append(dict(state=tag, xi=xi, band=f"{lo}-{hi}", f=fk[j], z_pct=100 * zk[j], mpc=mpc, h=hb, h_share=P[hb],
                                th0=th0[hb] if hb in (1, 2) else np.nan, amax=float(np.abs(Rr[:, j]).max()),
                                a=" ".join(f"{x / np.abs(a).max():+.2f}" for x in a)))
    return pd.DataFrame(out)


def fem_tent(case):
    fp = os.path.join(FEMD, "summary.csv")
    if not os.path.exists(fp):
        return {}
    S = pd.read_csv(fp)
    out = {}
    for n, (lab, comp) in TENT.items():
        r = S[(S.case == case) & (S.label == lab)]
        if r.empty:
            continue
        r = r.iloc[0]
        if comp is None:
            out[n] = r.df1
        else:
            v = [r.df1, r.df2]
            out[n] = max(v) if comp == "+" else min(v)
    return out


def fit_tentative():
    """추정 짝 8개에서 FEM 경우마다 측정(깃 2–6 중앙, 1번 깃은 데워짐 때문에 뺌)과의 rms·상관 → fe-nominal/s3_r_grind/fit_tentative.csv."""
    S = pd.read_csv(os.path.join(OUT, "R12_summary.csv"), dtype={"mode": str}).set_index("mode")
    F = pd.read_csv(os.path.join(FEMD, "summary.csv"))
    meas = np.array([S.loc[n, "d_v26"] for n in TENT])
    rows = []
    for case in F.case.unique():
        v = np.array([fem_tent(case)[n] for n in TENT])
        rows.append(dict(case=case, rms=np.sqrt(np.mean((v - meas) ** 2)), corr=np.corrcoef(v, meas)[0, 1],
                         **{n: x for n, x in zip(TENT, v)}))
    T = pd.DataFrame(rows).sort_values("rms")
    T.to_csv(os.path.join(FEMD, "fit_tentative.csv"), index=False, float_format="%.4g")
    print("측정(깃 2–6 중앙):", {n: round(float(x), 3) for n, x in zip(TENT, meas)})
    print(T.round(3).to_string(index=False))


def fem_table():
    fp = os.path.join(FEMD, "key.csv")
    return pd.read_csv(fp).set_index("case").T if os.path.exists(fp) else None


def figure(R, fem):
    order = list(dict.fromkeys(R["mode"]))
    xs = {n: i for i, n in enumerate(order)}
    fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw=dict(hspace=0.3))
    cmap = plt.get_cmap("viridis")
    for _, r in R.iterrows():
        if not np.isfinite(r.get("d_pct", np.nan)) or r.se_pct >= 0.08 or r.n_a < 5 or r.n_b < 5:
            continue
        c = cmap((r.vane - 1) / 5)
        mk = "o" if r.xi == "0.3" else "s"
        ax[0].plot(xs[r["mode"]] + (r.vane - 3.5) * 0.08, r.d_pct, mk, color=c, ms=5, mfc=c if r.xi == "0.6" else "white")
        ax[1].plot(xs[r["mode"]] + (r.vane - 3.5) * 0.08, r.z_ratio, mk, color=c, ms=5, mfc=c if r.xi == "0.6" else "white")
    for v in range(1, 7):
        ax[0].plot([], [], "s", color=cmap((v - 1) / 5), label=f"vane {v}" + (" (22:55)" if v == 1 else ""))
    for case, col, lab in (("RS25b", "k", "FEM photo shape: vane-4 line → 25° toward vane 5, rim bevel (tentative pairs)"),
                           ("RB20", "0.55", "FEM centred on the vane-4 line ±10° (tentative pairs)")):
        for n, v in fem_tent(case).items():
            if n in xs:
                ax[0].plot(xs[n], v, "_", color=col, ms=15, mew=2.4, label=lab)
    h, l = ax[0].get_legend_handles_labels(); u = dict(zip(l, h))
    ax[0].legend(u.values(), u.keys(), fontsize=7.5, ncol=2, loc="lower right")
    ax[0].axhspan(-0.05, 0.05, color="0.85", zorder=0); ax[0].axhline(0, color="k", lw=0.6); ax[0].grid(alpha=0.3)
    ax[0].set_ylabel("frequency change R − baseline [%]")
    ax[0].set_title("(a) specimen 3, all 12 points (vane 1–6 × ξ0.3 open / ξ0.6 filled), 17.9 mg ground on vane 4, 15 N. "
                    "Modes with no point: peak absent in one state", fontsize=8.5, loc="left")
    ax[0].text(0.0, -0.13, FEM_TXT, transform=ax[0].transAxes, fontsize=7.5, va="top")
    ax[1].axhline(1, color="k", lw=0.6); ax[1].set_ylabel("damping ratio R / baseline"); ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 3)
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, fontsize=7.5, rotation=60)
    fig.savefig(os.path.join(OUT, "R12_check.png"), dpi=110, bbox_inches="tight")


def redraw():
    R = pd.read_csv(os.path.join(OUT, "R12_check.csv"), dtype={"mode": str, "xi": str})
    figure(R, fem_table())


def main(specA, specR):
    D, _ = sp.per_hit(GROUPS, BANDS)
    R = compare_all(D)
    R.to_csv(os.path.join(OUT, "R12_check.csv"), index=False, float_format="%.5g")
    S = summarize(R)
    S.to_csv(os.path.join(OUT, "R12_summary.csv"), index=False, float_format="%.5g")
    print(S.round(3).to_string(index=False))
    T = D.groupby("group")[["T0", "T1"]].mean()
    print("기온 ai0/ai1:", T.round(2).to_dict("index"))
    if specA and specR:
        J = pd.concat([joint(specA, "A"), joint(specR, "R")])
        J.to_csv(os.path.join(OUT, "R12_joint.csv"), index=False, float_format="%.4g")
        print(J.round(3).to_string(index=False))
    figure(R, fem_table())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--specA", default="")
    ap.add_argument("--specR", default="")
    ap.add_argument("--fig-only", action="store_true")
    a = ap.parse_args()
    if a.fig_only:
        redraw()
        fit_tentative()
    else:
        main(a.specA, a.specR)
