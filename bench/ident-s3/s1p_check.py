"""1번 시편 사전 시험(48번) 점검 — 깃 2–3 사이 약 1 mg 흠집 뒤(11-19 S13, 메모 sample1_L1) 주파수가 움직였는지.

기준은 1번 시편의 마지막 측정이다.
- 10-28 16:34–16:52(impact_testw22): 1번 깃 ξ0.3 14타(S2)·ξ0.6 38타(S1). 오늘과 같은 두 자리다.
- 10-26 20:00(impact_testw22): 1번 깃 ξ0.3 51타. 10-26 → 10-28 차는 손대지 않은 1번 시편의 날짜 차다.
방법
- 타격별: groove_lib의 마지막 접촉 3 ms 뒤 자유 감쇠 스펙트럼 + 대역 극점 적합(늦은 재타격도 쓴다, groove_state와 같은 승인 기준).
- 모드·위치마다 f = f0_날짜 + b·log2(N/15 N) 회귀(기울기 b는 날짜끼리 공통). 차 = f0(11-19)/f0(10-28) − 1.
  1번 시편은 힘에 따른 연화가 커서(29번) 힘이 다른 두 날을 그대로 비교하면 안 된다.
- 묶음 적합(극점 공유, 위 기울기로 15 N 환산)으로도 같은 차를 내서 맞대 본다.
- 감쇠는 타격별 ζ의 중앙값이다.
출력: groove-test/s1-pretest/s1p_check.csv·s1p_check.json·s1p-check.png
사용: python s1p_check.py"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groove_lib as gl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "groove-test", "s1-pretest")
W22 = "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22"
S2F = "/mnt/nas/data/GY_DATA/impact_testw2s2/impact_testw2s2"
LOCS = ["back_vane1_xi0.3", "back_vane1_xi0.6"]
GROUPS = {"10-26": (W22, None, "20261026", ["back_vane1_xi0.3"]),
          "10-28": (W22, None, "20261028", LOCS),
          "11-19": (S2F, [13], "20261119", LOCS)}
NREF = 15.0
# 1번 시편 극점 대역(10-26·10-28·11-19 평균 스펙트럼에서 세 날 모두 보이는 봉우리). (이름, 아래, 위, 극점 수)
BANDS = [("2041/2063 h2", 2025, 2082, 2), ("3055", 3030, 3080, 1), ("3208", 3192, 3226, 1), ("3261", 3244, 3278, 1),
         ("3390", 3372, 3408, 1), ("3489", 3476, 3504, 1), ("3712", 3696, 3730, 1), ("3786", 3770, 3802, 1),
         ("3884", 3872, 3894, 1), ("3904", 3895, 3922, 1), ("4331", 4314, 4350, 1), ("5108", 5085, 5132, 1),
         ("5258", 5236, 5282, 1), ("6096", 6078, 6115, 1), ("6170", 6152, 6188, 1), ("6705", 6686, 6726, 1),
         ("6868", 6848, 6890, 1), ("7282", 7265, 7300, 1), ("7842", 7822, 7862, 1), ("7935", 7918, 7952, 1),
         ("8030", 8010, 8050, 1)]
# 같은 깊이(0.06 mm, 36.9 mg) 창 하나에 대한 G3 FEM 예측(s3_pred_poles L1)을 1 mg으로 줄인 값 [%] — 1번 시편 모드와의 짝은 29번 §4
CUT_MG, WIN_MG = 1.0, 36.9
L1_FEM = {"3155('홀수 깃', 3164·3250)": (0.182, 0.372), "3631·3639 패널": (0.202, 0.202), "h2 짝 평균": (0.12, 0.12)}


def mode_names():
    out = []
    for name, lo, hi, k in BANDS:
        out += band_modes(name, k)
    return out


def band_modes(name, k):
    """대역 이름 → 극점 이름들('2041/2063 h2' → ['2041', '2063'])."""
    return name.split(" ")[0].split("/") if k > 1 else [name]


def per_hit(groups=None, bands=None):
    """타격별 극점 표와 묶음 적합용 대역 스펙트럼({(파일, 대역): (f, X)}).
    groups = {이름: (폴더, 세션, 날짜, 위치들[, 시작 HH:MM[, 끝 HH:MM]])}, bands = [(이름, 아래, 위, 극점 수)]."""
    groups, bands = groups or GROUPS, bands or BANDS
    rows, spec = [], {}
    for g, spec_g in groups.items():
        folder, ses, date, locs = spec_g[:4]
        hits = gl.list_hits(folder, ses, locs, date, spec_g[4] if len(spec_g) > 4 else None,
                            spec_g[5] if len(spec_g) > 5 else None)
        bad = gl.check_channels(hits)
        if bad:
            raise SystemExit("채널 점검 실패: " + bad[0])
        for h in hits:
            N = float(h["peak_force_N"])
            if N < 5 or h["status"] == "rejected":
                continue
            f, m, ti, fs = gl.load_signals(h)
            t0, late = gl.free_decay_start(f, N, ti, fs)
            X = gl.decay_spectrum(m, ti, fs, t0)
            fr = np.fft.rfftfreq(gl.NFFT, 1 / fs)
            T = np.asarray(h.get("temperature_C") or [np.nan, np.nan], float)
            for name, lo, hi, k in bands:
                spec[(h["file"], name)] = (fr[(fr >= lo - 40) & (fr <= hi + 40)], X[(fr >= lo - 40) & (fr <= hi + 40)])
                sel = (fr >= lo - 30) & (fr <= hi + 30)
                inn = (fr[sel] >= lo - 8) & (fr[sel] <= hi + 8)
                fk, zk, amp, res, err = gl.band_fit(fr[sel][inn], X[sel][inn], lo, hi, k)
                ok = err < 0.3 and np.all((fk > lo) & (fk < hi)) and (k == 1 or np.min(np.diff(fk)) > 3)
                for j, n in enumerate(band_modes(name, k)):
                    rows.append(dict(group=g, loc=h["location"], file=h["file"], N=N, late=late, T0=T[0], T1=T[1] if T.size > 1 else np.nan,
                                     mode=n, f=float(fk[j]) if ok else np.nan, z=float(zk[j]) if ok else np.nan,
                                     amp=float(np.abs(res[j])) / N if ok else np.nan))
        print(f"[{g}] 타격 {len(hits)}개 적합 끝", flush=True)
    return pd.DataFrame(rows), spec


def robust_keep(v):
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
    return np.abs(v - med) <= max(4 * mad, 5e-4 * med)


def compare(D, ga, gb, loc, n, nref=NREF, common_slope=True):
    """날짜 둘의 f0(nref) 차 [%]와 표준오차 — 공통 기울기 회귀(common_slope=False면 날짜마다 따로 기울기)."""
    d = D[(D["loc"] == loc) & (D["mode"] == n) & D.group.isin([ga, gb]) & np.isfinite(D.f)]
    parts = []
    for g in (ga, gb):
        x = d[d.group == g]
        if len(x) < 4:
            return None
        parts.append(x[robust_keep(x.f.values)])
    za, zb = (np.median(p.z) for p in parts)
    base = dict(n_a=len(parts[0]), n_b=len(parts[1]), z_a=za, z_b=zb, z_ratio=zb / za,
                N_a=float(np.median(parts[0].N)), N_b=float(np.median(parts[1].N)))
    if not common_slope:
        cs, vs, sl, sd = [], [], [], []
        for p in parts:
            A = np.column_stack([np.ones(len(p)), np.log2(p.N / nref)])
            c, *_ = np.linalg.lstsq(A, p.f.values, rcond=None)
            r = p.f.values - A @ c
            cov = np.linalg.pinv(A.T @ A) * (r @ r) / max(len(p) - 2, 1)
            cs.append(c[0]); vs.append(cov[0, 0]); sl.append(100 * c[1] / c[0]); sd.append(100 * np.std(r) / c[0])
        d = 100 * (cs[1] / cs[0] - 1)
        se = 100 * np.sqrt(vs[1] / cs[0] ** 2 + vs[0] * (cs[1] / cs[0] ** 2) ** 2)
        return dict(f_a=cs[0], f_b=cs[1], d_pct=d, se_pct=se, slope_a_pct=sl[0], slope_b_pct=sl[1], sd_a=sd[0], sd_b=sd[1], **base)
    x = pd.concat(parts)
    A = np.column_stack([(x.group == ga).astype(float), (x.group == gb).astype(float), np.log2(x.N / nref)])
    c, *_ = np.linalg.lstsq(A, x.f.values, rcond=None)
    r = x.f.values - A @ c
    dof = max(len(x) - 3, 1)
    cov = np.linalg.inv(A.T @ A) * (r @ r) / dof
    dpct = 100 * (c[1] / c[0] - 1)
    se = 100 * np.sqrt(cov[0, 0] * (c[1] / c[0] ** 2) ** 2 + cov[1, 1] / c[0] ** 2 - 2 * cov[0, 1] * c[1] / c[0] ** 3)
    slope = 100 * c[2] / c[0]                       # 힘 2배당 %
    return dict(f_a=c[0], f_b=c[1], d_pct=dpct, se_pct=se, slope_pct=slope,
                sd_a=100 * np.std(parts[0].f) / c[0], sd_b=100 * np.std(parts[1].f) / c[1], **base)


def pooled(D, spec, g, loc, name, lo, hi, k, slope_frac, nboot=30, nref=NREF):
    """묶음 적합 극점(nref 환산) — 타격별 회귀와 맞대 보기용."""
    files = D[(D.group == g) & (D["loc"] == loc)].drop_duplicates("file")
    Ys = []
    for _, r in files.iterrows():
        frs, Xs = spec[(r.file, name)]
        gg, Y = gl.stretched(Xs, frs, r.N * 20.0 / nref, lo, hi, slope_frac)     # stretched는 20 N 기준 → N·20/nref로 nref 기준
        Ys.append(Y)
    if len(Ys) < 5:
        return None
    p, err = gl.joint_fit(gg, Ys, lo, hi, k)
    rng = np.random.default_rng(3)
    bs = [gl.joint_fit(gg, [Ys[i] for i in rng.choice(len(Ys), len(Ys))], lo, hi, k, p0=p)[0][:k] for _ in range(nboot)]
    return p[:k], np.std(bs, axis=0, ddof=1), p[k:], err


def main():
    os.makedirs(OUT, exist_ok=True)
    D, spec = per_hit()

    rows = []
    for name, lo, hi, k in BANDS:
        names = band_modes(name, k)
        for loc in LOCS:
            for ga, gb in [("10-28", "11-19"), ("10-26", "10-28")]:
                if ga == "10-26" and loc != "back_vane1_xi0.3":
                    continue
                res = {n: compare(D, ga, gb, loc, n) for n in names}
                if any(v is None for v in res.values()):
                    continue
                # 묶음 적합(같은 기울기)
                sl = float(np.mean([v["slope_pct"] for v in res.values()])) / 100
                pa, pb = (pooled(D, spec, g, loc, name, lo, hi, k, sl) for g in (ga, gb))
                for j, n in enumerate(names):
                    v = res[n]
                    pool_d = pool_se = np.nan
                    if pa is not None and pb is not None:
                        pool_d = 100 * (pb[0][j] / pa[0][j] - 1); pool_se = 100 * np.hypot(pa[1][j], pb[1][j]) / pa[0][j]
                    rows.append(dict(mode=n, loc=loc.replace("back_vane1_", ""), pair=f"{gb} − {ga}", **v, pool_d_pct=pool_d,
                                     pool_se_pct=pool_se))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "s1p_check.csv"), index=False, float_format="%.5g")

    # ---------------------------------------------------------------- 요약
    T = D.groupby("group")[["T0", "T1"]].mean()
    dT = (T.loc["11-19"] - T.loc["10-28"]).values
    print("\n기온 ai0/ai1 [°C]:", {g: f"{T.loc[g, 'T0']:.2f}/{T.loc[g, 'T1']:.2f}" for g in T.index}, "→ 11-19 − 10-28", np.round(dT, 2))
    print(f"기온으로 기대되는 이동 ≈ {-0.02 * dT[0]:+.3f} ~ {-0.02 * dT[1]:+.3f} % (−0.02 %/K)")
    print(f"흠집 {CUT_MG:g} mg로 기대되는 이동(G3 FEM 창 하나 0.06 mm 예측 × {CUT_MG:g}/{WIN_MG:g}):",
          {k: f"+{v[0] * CUT_MG / WIN_MG:.3f}~+{v[1] * CUT_MG / WIN_MG:.3f} %" for k, v in L1_FEM.items()})
    good = R[(R.se_pct < 0.05) & (R.n_a >= 6) & (R.n_b >= 6)]
    for pair in ["11-19 − 10-28", "10-28 − 10-26"]:
        x = good[good.pair == pair]
        print(f"\n[{pair}] 잘 정해진 모드·위치 {len(x)}개(표준오차 < 0.05 %, 타격 6개 이상)")
        print(x[["mode", "loc", "d_pct", "se_pct", "pool_d_pct", "pool_se_pct", "slope_pct", "N_a", "N_b", "sd_a", "sd_b", "z_a", "z_b",
                 "z_ratio"]].round(4).to_string(index=False))
        if len(x):
            c = np.median(x.d_pct)
            print(f"  중앙 {c:+.3f} %, 평균 {x.d_pct.mean():+.3f} %, 중앙을 뺀 rms {np.sqrt(np.mean((x.d_pct - c) ** 2)):.3f} %,"
                  f" 감쇠 비 중앙 {np.median(x.z_ratio):.2f}")
    json.dump(dict(T={g: [float(T.loc[g, 'T0']), float(T.loc[g, 'T1'])] for g in T.index}, dT=[float(v) for v in dT],
                   bands=BANDS, cut_mg=CUT_MG), open(os.path.join(OUT, "s1p_check.json"), "w"), ensure_ascii=False, indent=1)
    figure(R, D, dT)


def figure(R, D, dT):
    order = mode_names()
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    xs = {n: i for i, n in enumerate(order)}
    sty = {("11-19 − 10-28", "xi0.3"): ("C3", "o", -0.15), ("11-19 − 10-28", "xi0.6"): ("C1", "s", 0.0),
           ("10-28 − 10-26", "xi0.3"): ("0.55", "^", 0.15)}
    for (pair, loc), (c, mk, dx) in sty.items():
        x = R[(R.pair == pair) & (R["loc"] == loc)]
        ok = (x.se_pct < 0.05) & (x.n_a >= 6) & (x.n_b >= 6)
        for (_, r), good in zip(x.iterrows(), ok):
            ax[0].errorbar(xs[r["mode"]] + dx, r.d_pct, yerr=2 * r.se_pct, fmt=mk, color=c, mfc=c if good else "white", ms=5,
                           label=f"{pair}, {loc}")
            ax[1].plot(xs[r["mode"]] + dx, r.z_ratio, mk, color=c, mfc=c if good else "white", ms=5)
    h, l = ax[0].get_legend_handles_labels(); u = dict(zip(l, h))
    ax[0].legend(u.values(), u.keys(), fontsize=8, loc="lower left")
    ax[0].axhspan(-0.05, 0.05, color="0.85", zorder=0, label="re-hang floor")
    ax[0].axhspan(min(-0.02 * dT), max(-0.02 * dT), color="C3", alpha=0.12, zorder=0)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("frequency change [%]  (±2 SE)")
    ax[0].set_title("(a) specimen 1, vane 1: change at 15 N. grey = re-hang floor ±0.05 %, red = expected from air temperature;"
                    " 1 mg cut expected ≤ +0.01 %. open = SE ≥ 0.05 % or < 6 hits", fontsize=8.5, loc="left")
    ax[0].set_ylim(-0.6, 0.6); ax[0].grid(alpha=0.3)
    ax[1].axhline(1, color="k", lw=0.6); ax[1].set_ylabel("damping ratio (later / earlier)"); ax[1].set_ylim(0, 2.2)
    ax[1].set_title("(b) damping ratio (median of per-hit ζ)", fontsize=8.5, loc="left"); ax[1].grid(alpha=0.3)
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, rotation=45, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s1p-check.png"), dpi=110)


if __name__ == "__main__":
    main()
