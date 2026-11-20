"""3번 시편 기준(11-19 S17, 메모 s3, 39941.4 mg) — 모드 목록·원주 차수·자리별 재현성(53번).

- 봉우리 대역은 12자리 평균 감쇠 스펙트럼에서 손으로 정했다(전원 120 Hz 배수 날카로운 선 5641·5761·5881·6001 등은 뺐다).
- 타격별 극점(s1p_check.per_hit, 늦은 재타격도 쓴다) → 자리·모드마다 f = f0 + b·log2(N/15 N) 회귀:
  f0(15 N), 잔차 표준편차 sd, f0 표준오차 se, 감쇠 중앙값. 이것이 다음 상태와 비교할 때의 바닥이다.
- 원주 차수: 깨끗한 타격(늦은 재타격 없음, 5 N 이상)의 깃 평균 H1을 깃 6곳 공동 적합(ident-w22/w22joint) — ξ0.3·ξ0.6 따로.
- 2번 시편 A2(11-16 S12)의 같은 방법 평균 스펙트럼을 겹쳐 그린다.
출력: groove-test/s3/S3_base_modes.csv(모드표)·S3_base_hits.csv(자리별)·S3_base.png
사용: python s3_spec3_base.py --spec <npz>   (s2spec.py로 만든 타격 스펙트럼, 없으면 공동 적합을 건너뛴다; 약 8분)
      python s3_spec3_base.py --fig-only     (저장된 csv로 그림만)"""
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
import groove_lib as gl  # noqa: E402
import s1p_check as sp  # noqa: E402

OUT = os.path.join(HERE, "..", "groove-test", "s3")
LOC12 = [f"back_vane{v}_xi{x}" for v in range(1, 7) for x in ("0.3", "0.6")]
GROUPS = {"S3-A": (sp.S2F, [17], "20261119", LOC12)}
A2 = (sp.S2F, [12], "20261116", LOC12)
MASS_MG = 39941.4
# (이름, 아래, 위, 극점 수, 2번 시편 A2에서 짝으로 보이는 봉우리 — 원주 차수·자리로 본 추정, 확실하지 않으면 '')
BANDS = [("2066/2076 h2", 2050, 2090, 2, "2134/2142 h2"), ("3033", 3000, 3070, 1, "3156·3230 무리"),
         ("3149", 3115, 3185, 1, "3156·3230 무리"), ("3296", 3260, 3335, 1, "3156·3230 무리"), ("3467", 3455, 3480, 1, ""),
         ("3496", 3485, 3510, 1, "3496"), ("3554", 3535, 3575, 1, "3455·3582·3634 무리"), ("3613", 3590, 3640, 1, "3455·3582·3634 무리"),
         ("3707", 3690, 3725, 1, "3455·3582·3634 무리"), ("3880", 3865, 3895, 1, ""), ("3979", 3940, 4020, 1, "4211 허브 h1"),
         ("4060", 4045, 4080, 1, ""), ("4420", 4405, 4437, 1, ""), ("4457", 4440, 4475, 1, ""), ("5136", 5090, 5185, 1, ""),
         ("5262", 5240, 5285, 1, ""), ("5311", 5290, 5340, 1, ""), ("6110/6152", 6095, 6175, 2, ""), ("6411", 6385, 6440, 1, ""),
         ("6732", 6705, 6760, 1, ""), ("6932", 6910, 6955, 1, "7073 h3"), ("7216", 7195, 7240, 1, ""), ("7363", 7340, 7390, 1, ""),
         ("7751", 7725, 7780, 1, ""), ("8031", 8005, 8060, 1, "8279 h3"), ("8195", 8175, 8215, 1, ""), ("8318", 8290, 8350, 1, "")]
# 공동 적합 대역(극점 수) — 같은 봉우리
JFITS = [((2040, 2100), 2), ((3000, 3080), 1), ((3110, 3190), 1), ((3250, 3360), 1), ((3450, 3480), 1), ((3480, 3520), 1),
         ((3530, 3580), 1), ((3585, 3650), 1), ((3680, 3735), 1), ((3860, 3910), 1), ((3930, 4030), 1), ((4035, 4090), 1),
         ((4395, 4440), 1), ((4440, 4490), 1), ((5060, 5200), 1), ((5240, 5290), 1), ((5290, 5360), 1), ((6095, 6190), 2),
         ((6385, 6440), 1), ((6700, 6770), 1), ((6900, 6960), 1), ((7190, 7245), 1), ((7330, 7400), 1), ((7720, 7790), 1),
         ((7990, 8080), 1), ((8170, 8220), 1), ((8290, 8360), 1)]


def per_position(D):
    rows = []
    for name, lo, hi, k, pair in BANDS:
        for n in sp.band_modes(name, k):
            for loc in LOC12:
                x = D[(D["loc"] == loc) & (D["mode"] == n) & np.isfinite(D.f)]
                if len(x) < 4:
                    rows.append(dict(mode=n, loc=loc, n=len(x)))
                    continue
                x = x[sp.robust_keep(x.f.values)]
                A = np.column_stack([np.ones(len(x)), np.log2(x.N / sp.NREF)])
                c, *_ = np.linalg.lstsq(A, x.f.values, rcond=None)
                r = x.f.values - A @ c
                cov = np.linalg.pinv(A.T @ A) * (r @ r) / max(len(x) - 2, 1)
                rows.append(dict(mode=n, loc=loc, n=len(x), f0=c[0], slope_pct=100 * c[1] / c[0], sd_pct=100 * np.std(r) / c[0],
                                 se_pct=100 * np.sqrt(cov[0, 0]) / c[0], z_pct=100 * np.median(x.z), amp=np.median(x.amp)))
    return pd.DataFrame(rows)


def joint_patterns(spec_npz):
    from w22joint import joint_fit, pattern
    Z = np.load(spec_npz)
    fr = Z["fr"]
    F, X = Z["F"].astype(complex), Z["X"].astype(complex)
    use = ~Z["late"] & (Z["N"] >= 5)
    out = []
    for xi in (0.3, 0.6):
        Hs, ncl = [], []
        for v in range(1, 7):
            u = np.where(use & (Z["vane"] == v) & (Z["xi"] == xi))[0]
            ncl.append(len(u))
            Hs.append((np.conj(F[u]) * X[u]).sum(0) / (np.abs(F[u]) ** 2).sum(0))
        for (lo, hi), k in JFITS:
            fk, zk, R, err = joint_fit(fr, Hs, lo, hi, k)
            for j in range(k):
                mpc, a, P, th0, _ = pattern(R[:, j])
                hb = int(np.argmax(P))
                out.append(dict(xi=xi, f=fk[j], mpc=mpc, h=hb, h_share=P[hb], P0=P[0], P1=P[1], P2=P[2], P3=P[3],
                                th0=th0[hb] if hb in (1, 2) else np.nan, n_clean=min(ncl), err=err,
                                a=" ".join(f"{x / np.abs(a).max():+.2f}" for x in a)))
    return pd.DataFrame(out)


def mean_spectrum(group):
    folder, ses, date, locs = group
    acc = {}
    for h in gl.list_hits(folder, ses, locs, date):
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
    s = (fr > 1900) & (fr < 8600)
    return fr[s], np.mean([v[0][s] / v[1] / np.median(v[0][s] / v[1]) for v in acc.values()], axis=0)


def main(spec_npz):
    os.makedirs(OUT, exist_ok=True)
    D, _ = sp.per_hit(GROUPS, [b[:4] for b in BANDS])
    P = per_position(D)
    P.to_csv(os.path.join(OUT, "S3_base_hits.csv"), index=False, float_format="%.5g")
    J = joint_patterns(spec_npz) if spec_npz else pd.DataFrame()
    rows = []
    for name, lo, hi, k, pair in BANDS:
        for n in sp.band_modes(name, k):
            x = P[(P["mode"] == n) & np.isfinite(P.get("f0", np.nan))]
            r = dict(mode=n, pair_S2=pair, n_pos=len(x))
            if len(x):
                r.update(f_med=np.median(x.f0), z_pct=np.median(x.z_pct), sd_med_pct=np.median(x.sd_pct),
                         se_med_pct=np.median(x.se_pct), se_max_pct=x.se_pct.max(), slope_med_pct=np.median(x.slope_pct))
            for xi in (0.3, 0.6):
                if len(J) and len(x):
                    j = J[(J.xi == xi) & (np.abs(J.f - np.median(x.f0)) < 8)]
                    if len(j):
                        j = j.iloc[0]
                        r[f"h_xi{xi}"] = f"h{int(j.h)} {j.h_share:.2f}" + (f" θ0 {j.th0:.0f}" if np.isfinite(j.th0) else "")
                        r[f"mpc_xi{xi}"] = j.mpc
            rows.append(r)
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(OUT, "S3_base_modes.csv"), index=False, float_format="%.4g")
    if len(J):
        J.to_csv(os.path.join(OUT, "S3_base_joint.csv"), index=False, float_format="%.4g")
    print(M.round(3).to_string(index=False))
    T = D.groupby("group")[["T0", "T1"]].mean()
    print("기온 ai0/ai1:", T.round(2).to_dict("index"))
    figure(M, P)


def figure(M, P):
    f3, s3 = mean_spectrum((sp.S2F, [17], "20261119", LOC12))
    f2, s2 = mean_spectrum(A2)
    fig = plt.figure(figsize=(15, 11))
    gsp = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.9], hspace=0.35)
    for i, (lo, hi) in enumerate([(1950, 4700), (4700, 8600)]):
        ax = fig.add_subplot(gsp[i])
        s = (f2 >= lo) & (f2 <= hi)
        ax.semilogy(f2[s], s2[s], color="C0", lw=0.9, label="specimen 2, A2 (11-16, 12 points)")
        s = (f3 >= lo) & (f3 <= hi)
        ax.semilogy(f3[s], s3[s], color="C3", lw=0.9, label=f"specimen 3, baseline S17 (11-19, 12 points, {MASS_MG / 1e3:.4f} g)")
        j = 0
        for _, r in M.iterrows():
            if np.isfinite(r.get("f_med", np.nan)) and lo <= r.f_med <= hi:
                y = s3[np.argmin(np.abs(f3 - r.f_med))]
                lab = r["mode"] + (f"\n{r.get('h_xi0.6', '')}" if isinstance(r.get("h_xi0.6"), str) else "")
                ax.annotate(lab, (r.f_med, y), xytext=(0, 8 + 22 * (j % 2)), textcoords="offset points", color="C3", fontsize=6.3,
                            ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color="C3", lw=0.4))
                j += 1
        ax.set_xlim(lo, hi)
        ax.set_ylim(top=ax.get_ylim()[1] * 30)
        ax.grid(alpha=0.3)
        ax.set_ylabel("mean |X/F|² (normalized)")
        if i == 0:
            ax.legend(fontsize=8, loc="lower left")
            ax.set_title("(a) 12-point mean free-decay spectra; specimen-3 labels = peak and circumferential order at ξ0.6 "
                         "(six-vane joint fit, share of the strongest harmonic)", fontsize=9, loc="left")
    ax = fig.add_subplot(gsp[2])
    order = [r["mode"] for _, r in M.iterrows() if np.isfinite(r.get("f_med", np.nan))]
    xs = {n: i for i, n in enumerate(order)}
    for _, r in P.iterrows():
        if r["mode"] in xs and np.isfinite(r.get("se_pct", np.nan)):
            xi6 = r["loc"].endswith("0.6")
            ax.plot(xs[r["mode"]] + (0.12 if xi6 else -0.12), r.se_pct, "s" if xi6 else "o", color="C1" if xi6 else "C2", ms=4,
                    mfc="C1" if xi6 else "white")
    ax.plot([], [], "o", color="C2", mfc="white", label="ξ0.3 (6 vanes)")
    ax.plot([], [], "s", color="C1", label="ξ0.6 (6 vanes)")
    ax.axhline(0.05, color="k", lw=0.8, ls="--", label="±0.05 % (specimen-2 floor)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=7, rotation=60)
    ax.set_ylabel("standard error of f0 [%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(b) per-point standard error of the 15 N frequency (one session, 10–15 hits per point)", fontsize=9, loc="left")
    fig.savefig(os.path.join(OUT, "S3_base.png"), dpi=105, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="")
    ap.add_argument("--fig-only", action="store_true", help="저장된 csv로 그림만 다시 그린다")
    a = ap.parse_args()
    if a.fig_only:
        figure(pd.read_csv(os.path.join(OUT, "S3_base_modes.csv"), dtype={"mode": str}),
               pd.read_csv(os.path.join(OUT, "S3_base_hits.csv"), dtype={"mode": str}))
    else:
        main(a.spec)
