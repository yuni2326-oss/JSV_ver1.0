"""JSV 수정본 실험 절(58번 초안)의 수치 — 그림·표·본문이 같은 값을 쓰도록 한곳에서 낸다.

자료(모두 이 폴더 아래, 측정은 NAS impact_testw2s2):
- 2번 시편 질량 시험: groove-test/out/T-series_report.csv(T-a·T-b·T-c), M_report.csv(A1-6v → M120·M240), M240_inc.csv(M120 → M240),
  out/m_piece_fit.json(사진 자리 FEM — 11-20부터 칸 폭 환산 각 'photo_scaled'와 자리 불확도 포락, s3_m_piece_scan.py),
  groove-test/m_piece_positions.csv, fe-nominal/s3_m_angle_scan.csv, out/V1-1119_check.csv(다시 매달기)
- 3번 시편 깎기: groove-test/s3/R12_summary.csv·R12_check.csv(기준 → R), L12_summary.csv·L12_check.csv(R → L, 기준 → L),
  S3_base_modes.csv(기준 모드표), fe-nominal/s3_r_grind·s3_l_grind/summary.csv(FEM)
사용: python paper_bench_data.py   → paper-fig/table_E*.csv 를 쓰고 요약을 찍는다."""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
OUT = os.path.join(BASE, "groove-test", "out")
S3 = os.path.join(BASE, "groove-test", "s3")
FEM = os.path.join(BASE, "fe-nominal")
FIGD = os.path.join(BASE, "paper-fig")

# 3번 시편 질량(분석저울 A&D HR-250AZ, 사용자 사진) [mg]
S3_MASS = {"baseline": 39941.4, "R": 39923.5, "L": 39905.9}
S2_MASS_G = 41.3186
MODEL_MASS_G = 37.093

# 2번 시편 질량 시험에서 비교하는 모드(측정 이름 → FEM 관측량). 측정 이름은 2번 시편 기준 주파수다.
S2_MODES = [("h2", "h2:m2n0"), ("3164", "h3:m3n0"), ("3250", "h0:m0n1"), ("3459", "h1:m1n1"), ("3639", "h2:m4n0"),
            ("3748", "h3:m3n1"), ("3804", "h0:m6n1"), ("8321", "h3:m3n1#2"), ("5745", "h1:m5n0"), ("7093", "h3:m3n2"),
            ("7574", "h2:m8n0")]


def t_series():
    """T-a·T-b: h2 짝 두 성분과 갈라짐(측정, FEM)."""
    T = pd.read_csv(os.path.join(OUT, "T-series_report.csv"))
    rows = []
    for st in ("T-a", "T-b"):
        x = T[(T.state == st) & T["mode"].astype(str).isin(["2134", "2142"])]
        x = x.set_index(x["mode"].astype(str))
        lo, up = x.loc["2134"], x.loc["2142"]
        split0 = up.f_base - lo.f_base
        split1 = up.f_state - lo.f_state
        f0 = 0.5 * (up.f_base + lo.f_base)
        rows.append(dict(state=st, d_lower=lo.d_pct, se_lower=lo.sig_pct, fem_lower=lo.pred_pct,
                         d_upper=up.d_pct, se_upper=up.sig_pct, fem_upper=up.pred_pct,
                         split0_Hz=split0, split1_Hz=split1, dsplit_pct=100 * (split1 - split0) / f0,
                         fem_dsplit_pct=up.pred_pct - lo.pred_pct, se_dsplit=np.hypot(lo.sig_pct, up.sig_pct)))
    return pd.DataFrame(rows)


def m_series():
    """M120(1단계, 기준 대비)·M240 증분(2단계, M120 대비): 측정과 FEM(사진 자리 — 맞춘 값 없음, 공칭 자리).

    11-20(59번 §3): 사진 자리는 칸 폭으로 60°에 맞춘 각('photo_scaled')을 쓰고, 모든 조각의 각 ±1°·반지름 ±0.5 mm
    공통 이동의 포락을 예측 범위(fem_lo·fem_hi)로 둔다. 예전 값('photo', 원래 각)은 비교용으로만 남긴다."""
    M = pd.read_csv(os.path.join(OUT, "M_report.csv"))
    I = pd.read_csv(os.path.join(OUT, "M240_inc.csv"))
    J = json.load(open(os.path.join(OUT, "m_piece_fit.json")))
    photo, nom, raw = J["cases"]["photo_scaled|25.0"], J["cases"]["template|25.0"], J["cases"]["photo|25.0"]
    E = J["photo_scaled_env"]
    rows = []
    for step, D, key, ek in (("M120", M[M.state == "M120"], "pred_M120", "M120"), ("M240 increment", I, "pred_inc", "inc")):
        D = D.assign(mode=D["mode"].astype(str)).set_index("mode")
        for name, lab in S2_MODES:
            if name == "h2":
                meas = 0.5 * (D.loc["2134", "d_pct"] + D.loc["2142", "d_pct"])
                se = 0.5 * np.hypot(D.loc["2134", "sig_pct"], D.loc["2142", "sig_pct"])
                split = None
            else:
                if name not in D.index:
                    continue
                meas, se = D.loc[name, "d_pct"], D.loc[name, "sig_pct"]
            fp, fn = photo[key][name], nom[key][name]
            lo, hi = E[f"{ek}_lo"][name], E[f"{ek}_hi"][name]
            rows.append(dict(step=step, mode=name, model=lab, meas=meas, se=se, fem_photo=fp, fem_lo=lo, fem_hi=hi,
                             fem_nominal=fn, fem_photo_raw=raw[key][name],
                             in_range=bool(lo - 2 * se <= meas <= hi + 2 * se),
                             ratio_photo=meas / fp if abs(fp) >= 0.3 else np.nan,       # 비는 |예측| ≥ 0.3 %인 관측량만
                             ratio_nominal=meas / fn if abs(fn) >= 0.3 else np.nan))
    R = pd.DataFrame(rows)
    # 갈라짐 증분(6회 대칭 예측 0)
    sp = []
    for step, D in (("M120", M[M.state == "M120"]), ("M240 increment", I)):
        D = D.assign(mode=D["mode"].astype(str)).set_index("mode")
        s0 = D.loc["2142", "f_base"] - D.loc["2134", "f_base"]
        s1 = D.loc["2142", "f_state"] - D.loc["2134", "f_state"]
        f0 = 0.5 * (D.loc["2142", "f_base"] + D.loc["2134", "f_base"])
        sp.append(dict(step=step, split0_Hz=s0, split1_Hz=s1, dsplit_pct=100 * (s1 - s0) / f0,
                       se=np.hypot(D.loc["2134", "sig_pct"], D.loc["2142", "sig_pct"])))
    chi = dict(photo=photo["chi_M120"] + photo["chi_inc"], nominal=nom["chi_M120"] + nom["chi_inc"],
               photo_raw=raw["chi_M120"] + raw["chi_inc"], photo_pos=E["chi_M120_pos"] + E["chi_inc_pos"],
               d_theta=E["d_theta_deg"], d_r=E["d_r_mm"])
    return R, pd.DataFrame(sp), chi


def piece_angles():
    """조각 자리 요약(59번 §3): 칸 폭 환산 각, 두 사진 차, 평균."""
    P = pieces()
    P = P.assign(from_s=P.ang_from_k * 60 / P.bay_width)
    ph = sorted(P.photo.unique())
    p1 = P[P.photo == ph[0]].set_index("bay")
    p2 = P[(P.photo == ph[1]) & (P.ang_from_k < P.bay_width / 2)].set_index("bay")
    j = p1[["ang_from_k", "from_s", "r_mm", "bay_width", "len_mm"]].join(
        p2[["ang_from_k", "from_s", "r_mm", "bay_width", "len_mm"]], lsuffix="_1", rsuffix="_2")
    m240 = P[(P.photo == ph[1]) & (P.ang_from_k >= P.bay_width / 2)].set_index("bay")
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    s = dict(d_raw_rms=rms(j.ang_from_k_2 - j.ang_from_k_1), d_raw_max=float((j.ang_from_k_2 - j.ang_from_k_1).abs().max()),
             d_s_rms=rms(j.from_s_2 - j.from_s_1), d_s_max=float((j.from_s_2 - j.from_s_1).abs().max()),
             d_r_rms=rms(j.r_mm_2 - j.r_mm_1), d_r_max=float((j.r_mm_2 - j.r_mm_1).abs().max()),
             m120_s=(float(j.from_s_1.mean()), float(j.from_s_2.mean())), m120_raw=(float(j.ang_from_k_1.mean()), float(j.ang_from_k_2.mean())),
             m120_s_all=float(0.5 * (j.from_s_1 + j.from_s_2).mean()), m120_r=float(0.5 * (j.r_mm_1 + j.r_mm_2).mean()),
             m240_s=float(m240.from_s.mean()), m240_s_sd=float(m240.from_s.std()), m240_raw_sd=float(m240.ang_from_k.std()),
             m240_r=float(m240.r_mm.mean()), width=(float(P.bay_width.min()), float(P.bay_width.max())),
             len_corr=tuple(float(np.corrcoef(P[P.photo == p].len_mm, P[P.photo == p].bay_width)[0, 1]) for p in ph),
             len_range=(float(P.len_mm.min()), float(P.len_mm.max())))
    return j, m240, s


def angle_scan(rc=33.3):
    """6조각 고리(20 mg × 6) FEM 예측의 각 의존성 — 측정 각(깃 k에서 깃 k+1 쪽) = 60 − FEM 섹터 각."""
    SC = pd.read_csv(os.path.join(FEM, "s3_m_angle_scan.csv"))
    g = SC[np.isclose(SC.rc_mm, rc)].sort_values("theta_deg")
    out = pd.DataFrame({"ang_meas": 60 - g.theta_deg.values})
    for k in [c for c in SC.columns if c not in ("rc_mm", "theta_deg")]:
        out[k] = 100 * (np.sqrt(1 + g[k].values) - 1)
    return out.sort_values("ang_meas")


def pieces():
    P = pd.read_csv(os.path.join(BASE, "groove-test", "m_piece_positions.csv"))
    return P


def rehang():
    """2번 시편 다시 매달기(A2 → A2-rh, 1번 깃 두 자리, 47번 판정 모드 12개)."""
    V = pd.read_csv(os.path.join(OUT, "V1-1119_check.csv"))
    V = V[(V.pair == "11-19 − A2 11-16") & V.well.astype(bool)].copy()     # 47번의 잘 정해진 모드, 같은 두 자리
    return V


# ---------------------------------------------------------------- 3번 시편
# 깎은 단계마다 강조하는 모드(두 단계 모두 12자리에서 한 방향, 크기 ≥ 0.1 %) — 이름은 3번 시편 기준 주파수
S3_FAMILY = ("3033", "3149")                 # h0·h3 섞인 두 모드 → 합을 FEM h3:m3n0 + h0:m0n1과 맞댄다
S3_ASSIGNED = {"2066": ("h2:m2n0", "+"), "2077": ("h2:m2n0", "-"), "3980": ("h1:m1n1#2", "-"), "6932": ("h3:m3n2", None)}
R_CASES = ["RS25b", "RS25f", "RS25w", "RS20b"]   # 사진과 맞는 한쪽(깃 4 → 깃 5 쪽) 모양
L_CASES = ["L17e", "L17f", "L17v"]          # 사진 자리(칸 가운데에서 깃 3 쪽 2.5°); L17c(정확히 가운데)는 자리 민감도로 보충에만


def s3_summaries():
    R = pd.read_csv(os.path.join(S3, "R12_summary.csv"), dtype={"mode": str}).set_index("mode")
    L = pd.read_csv(os.path.join(S3, "L12_summary.csv"), dtype={"mode": str}).set_index("mode")
    return R, L


def fem_s3(folder, cases):
    S = pd.read_csv(os.path.join(FEM, folder, "summary.csv"))
    out = {}
    for c in cases:
        x = S[S.case == c].set_index("label")
        d = {}
        for n, (lab, comp) in S3_ASSIGNED.items():
            r = x.loc[lab]
            d[n] = r.df1 if comp is None else (max(r.df1, r.df2) if comp == "+" else min(r.df1, r.df2))
        d["family"] = x.loc["h3:m3n0"].df1 + x.loc["h0:m0n1"].df1
        d["h2sum"] = x.loc["h2:m2n0"].df1 + x.loc["h2:m2n0"].df2
        d["panel_h1"] = min(x.loc["h1:m1n1"].df1, x.loc["h1:m1n1"].df2)
        d["panel_h2"] = min(x.loc["h2:m4n0"].df1, x.loc["h2:m4n0"].df2) if folder == "s3_r_grind" else \
            max(x.loc["h2:m4n0"].df1, x.loc["h2:m4n0"].df2)
        out[c] = d
    return pd.DataFrame(out).T


def s3_table():
    """깎기 두 단계의 '바뀐 모드'(본문 Table E3) — 측정(깃 2–6 중앙과 범위)과 FEM(사진과 맞는 모양의 범위)."""
    R, L = s3_summaries()
    FR, FL = fem_s3("s3_r_grind", R_CASES), fem_s3("s3_l_grind", L_CASES)
    rows = []
    # R(깃 4 옆): 기준 → R. L: R → L.
    # 11-20(59번): 본문 표와 같은 표기로 낸다(58 표는 이 출력에서 옮긴다). R의 8.32 kHz는 자리 범위가 0을 걸쳐 '바뀐 모드'가 아니라 뺐다.
    spec = [("R", R, "d_v26", "d_lo", "d_hi", FR,
             [("2066", "h2 doublet, lower (2067 Hz)", "2066"), ("2077", "h2 doublet, upper (2077 Hz)", "2077"),
              ("3033", "3.03 kHz (h0/h3)", None), ("3149", "3.15 kHz (h0/h3)", None), ("3554→3562", "3.55 kHz (panel)", None),
              ("3613→3595", "3.61 kHz (panel)", None), ("3980", "3.98 kHz (hub, h1)", "3980"), ("6932→6910", "6.93 kHz (h3)", "6932")]),
            ("L", L, "RL_v26", "RL_lo", "RL_hi", FL,
             [("2066", "h2 doublet, lower (2067 Hz)", "2066"), ("2077", "h2 doublet, upper (2077 Hz)", "2077"),
              ("3033", "3.03 kHz (h0/h3)", None), ("3149", "3.15 kHz (h0/h3)", None), ("3296", "3.30 kHz", None),
              ("3980", "3.98 kHz (hub, h1)", "3980"), ("6732", "6.73 kHz (h3)", None), ("6932", "6.93 kHz (h3)", "6932"),
              ("8031", "8.03 kHz (h3)", None)])]
    for step, S, cm, clo, chi_, F, items in spec:
        for key, label, fkey in items:
            if key not in S.index:
                continue
            r = S.loc[key]
            row = dict(step=step, mode=key, label=label, meas=r[cm], lo=r[clo], hi=r[chi_])
            if fkey:
                row.update(fem_lo=F[fkey].min(), fem_hi=F[fkey].max())
            rows.append(row)
        fam = S.loc[S3_FAMILY[0], cm] + S.loc[S3_FAMILY[1], cm]
        h2 = S.loc["2066", cm] + S.loc["2077", cm]
        rows.append(dict(step=step, mode="3033+3149", label="Sum, 3.03 and 3.15 kHz", meas=fam,
                         fem_lo=F["family"].min(), fem_hi=F["family"].max()))
        rows.append(dict(step=step, mode="h2 sum", label="h2 doublet, sum of components", meas=h2,
                         fem_lo=F["h2sum"].min(), fem_hi=F["h2sum"].max()))
    return pd.DataFrame(rows), FR, FL


def reversible():
    """R에서 오르고 L에서 되돌아간 모드(기준 대비 누적, 깃 2–6 중앙)."""
    R, L = s3_summaries()
    out = []
    for key_r, key_l, f in (("5136", "5136", 5136), ("7216→7228", "7216", 7216), ("7749→7776", "7749", 7749)):
        out.append(dict(mode=f, R=R.loc[key_r, "d_v26"], L_minus_R=L.loc[key_l, "RL_v26"], cumulative=L.loc[key_l, "AL_v26"]))
    return pd.DataFrame(out)


def s3_floor():
    """3번 시편 측정 바닥: 기준 자리별 15 N 표준오차, 깎은 직후 1번 깃 치우침."""
    B = pd.read_csv(os.path.join(S3, "S3_base_modes.csv"), dtype={"mode": str})
    R, L = s3_summaries()
    singles = ["3033", "3149", "3296", "3980", "4060", "4457", "6732", "8031", "7363"]
    offR = np.nanmedian([R.loc[m, "d_v1"] - R.loc[m, "d_v26"] for m in singles if m in R.index])
    offL = np.nanmedian([L.loc[m, "AL_v1"] - L.loc[m, "AL_v26"] for m in singles if m in L.index])
    return B, offR, offL


def fmt(v, nd=2, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    s_ = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s_.replace("-", "−")


def md_tables():
    """원고·보충 표를 자료에서 바로 마크다운으로(paper-fig/tables.md)."""
    out = []
    # Table E2 — 6회 대칭 질량
    M, SP, chi = m_series()
    lab = {"h2": "h2:m2n0 (pair mean)", "3164": "h3:m3n0", "3250": "h0:m0n1", "3459": "h1:m1n1", "3639": "h2:m4n0",
           "3748": "h3:m3n1", "3804": "h0:m6n1", "8321": "h3:m3n1#2", "5745": "h1:m5n0", "7093": "h3:m3n2", "7574": "h2:m8n0"}
    f0 = {"h2": "2137.6", "3164": "3162.8", "3250": "3242.7", "3459": "3460.1", "3639": "3640.9", "3748": "3746.3",
          "3804": "3799.1", "8321": "8307.0", "5745": "5728.6", "7093": "7087.4", "7574": "7549.9"}
    # 11-20(59번): 한 단계씩 아래로 늘어놓고, 사진 자리 예측에 자리 불확도 범위를 붙인다.
    out.append("### Table E2\n\n| Step | Model family | f [Hz] | Measured [%] | Model, photographed [%] | Range [%] | "
               "Model, nominal [%] |\n|---|---|---|---|---|---|---|")
    for step, slab in (("M120", "M120"), ("M240 increment", "M240 − M120")):
        for k in lab:
            a = M[(M.step == step) & (M["mode"] == k)].iloc[0]
            mark = "" if np.isfinite(a.ratio_photo) else "*"
            out.append(f"| {slab} | {lab[k]} | {f0[k]} | {fmt(a.meas)} | {fmt(a.fem_photo)}{mark} | "
                       f"{fmt(a.fem_lo)} to {fmt(a.fem_hi)} | {fmt(a.fem_nominal)} |")
    rp = M.ratio_photo.dropna()
    out.append(f"\nRatios (|model| ≥ 0.3%, {len(rp)} values): {int(((rp >= 0.55) & (rp <= 1.23)).sum())} in 0.55–1.23, "
               f"range {rp.min():.2f}–{rp.max():.2f}, median {rp.median():.2f}; all 22: {int(((M.meas / M.fem_photo).between(0.55, 1.23)).sum())} in 0.55–1.23. "
               f"Measured within range ± 2 SE: {int(M.in_range.sum())} of {len(M)}. "
               f"χ² over both steps (11 observables per step): photographed {chi['photo']:.0f} (with position uncertainty "
               f"{chi['photo_pos']:.0f}), unscaled photograph angles {chi['photo_raw']:.0f}, nominal {chi['nominal']:.0f}. "
               f"Splitting change of the h2 doublet: M120 {fmt(SP.dsplit_pct.iloc[0])} ± {SP.se.iloc[0]:.2f}%, "
               f"M240 − M120 {fmt(SP.dsplit_pct.iloc[1])} ± {SP.se.iloc[1]:.2f}% (model 0 for both).\n")
    # Table E3 — 깎기, 바뀐 모드
    E3, FR, FL = s3_table()
    out.append("### Table E3\n\n| Step | Mode | Measured, median [%] | Range over vanes 2–6 [%] | Model [%] |\n|---|---|---|---|---|")
    for _, r in E3.iterrows():
        rng = "—" if not np.isfinite(r.get("lo", np.nan)) else f"{fmt(r.lo)} to {fmt(r.hi)}"
        mod = "—" if not np.isfinite(r.get("fem_lo", np.nan)) else (f"{fmt(r.fem_lo)} to {fmt(r.fem_hi)}"
                                                                     if abs(r.fem_hi - r.fem_lo) >= 0.005 else fmt(r.fem_lo))
        out.append(f"| {r.step} | {r.label} | {fmt(r.meas)} | {rng} | {mod} |")
    # Table SE4 — FEM 모양 변형
    out.append("\n### Table SE4 (model footprints)\n")
    for nm, F in (("R", FR), ("L", FL)):
        out.append(f"{nm}: " + "; ".join(f"{c}: family {fmt(F.loc[c, 'family'])}, h2 sum {fmt(F.loc[c, 'h2sum'])}, "
                                          f"6.93 kHz {fmt(F.loc[c, '6932'])}, hub {fmt(F.loc[c, '3980'])}" for c in F.index))
    # 각 의존성 값(사진 평균 각 = 칸 폭 환산, 59번)
    _, _, pa = piece_angles()
    a1, a2 = round(pa["m120_s_all"], 1), round(pa["m240_s"], 1)
    for rc in (33.3, 33.8, 34.3):
        S_ = angle_scan(rc)
        angs = [a1 - 1, a1, a1 + 1, 20.0, a2, 40.0]
        vals = {k: np.interp(angs, S_.ang_meas, S_[k]) for k in ("3164", "3459", "h2", "3748", "3639", "7093")}
        out.append(f"\nRing of six 20-mg pieces at r = {rc} mm, model shift [%] at " + ", ".join(f"{x:.1f}°" for x in angs) + ": " +
                   "; ".join(f"{k}: " + ", ".join(fmt(x) for x in v) for k, v in vals.items()))
    out.append(f"\nPiece positions (scaled to the 60° pitch): M120 {pa['m120_s'][0]:.2f}° / {pa['m120_s'][1]:.2f}° (photos 1/2; raw "
               f"{pa['m120_raw'][0]:.2f}° / {pa['m120_raw'][1]:.2f}°), mean {pa['m120_s_all']:.2f}°, r {pa['m120_r']:.2f} mm; "
               f"M240 {pa['m240_s']:.2f}° (sd {pa['m240_s_sd']:.2f}, raw sd {pa['m240_raw_sd']:.2f}), r {pa['m240_r']:.2f} mm; "
               f"photo-to-photo difference scaled rms {pa['d_s_rms']:.2f}° (max {pa['d_s_max']:.2f}), raw rms {pa['d_raw_rms']:.2f}° "
               f"(max {pa['d_raw_max']:.2f}), radius rms {pa['d_r_rms']:.2f} mm (max {pa['d_r_max']:.2f}); bay widths "
               f"{pa['width'][0]:.1f}–{pa['width'][1]:.1f}°; piece length {pa['len_range'][0]:.1f}–{pa['len_range'][1]:.1f} mm, "
               f"correlation with bay width {pa['len_corr'][0]:.2f} / {pa['len_corr'][1]:.2f}.")
    txt = "\n".join(out)
    open(os.path.join(FIGD, "tables.md"), "w").write(txt + "\n")
    print(txt)


def md_supp_tables():
    """보충 표 SE1–SE5(paper-fig/tables_supp.md)."""
    out = []
    # SE1 — 한 점 질량(T-a·T-b·T-c), 판정 모드
    T = pd.read_csv(os.path.join(OUT, "T-series_report.csv"))
    T["mode"] = T["mode"].astype(str)
    keep = [m for m in dict.fromkeys(T["mode"]) if isinstance(T[T["mode"] == m].fem.iloc[0], str)]
    # 11-20(59번): 44번에서 판정에 쓰지 않은 모드 표시 — ‡ 타격별 값이 두 무리(이웃 봉우리로 건너뜀), † 후보 대응(판정 밖)
    EXCL = {"4967", "6439"}
    cand = set(T[T.kind == "cand"]["mode"])
    out.append("### Table SE1\n\n| Mode (specimen 2) [Hz] | Model family | T-a measured [%] | T-a model [%] | T-b measured [%] | "
               "T-b model [%] | T-c measured [%] | T-c model [%] |\n|---|---|---|---|---|---|---|---|")
    for m in keep:
        cells = []
        for st in ("T-a", "T-b", "T-c"):
            x = T[(T.state == st) & (T["mode"] == m)]
            if x.empty:
                cells += ["—", "—"]
                continue
            r = x.iloc[0]
            weak = " (weak)" if bool(r.weak) else ""
            cells += [f"{fmt(r.d_pct)} ± {r.sig_pct:.2f}{weak}", fmt(r.pred_pct)]
        sym = "‡" if m in EXCL else ("†" if m in cand else "")
        out.append(f"| {m}{sym} | {T[T['mode'] == m].fem.iloc[0]} | " + " | ".join(cells) + " |")
    # SE3 — 조각 자리(11-20, 59번 §3: 칸 폭으로 60°에 맞춘 각, 두 사진 나란히, 원래 각은 m_piece_positions.csv에)
    j, m240, pa = piece_angles()
    out.append("\n### Table SE3\n\n| Bay | M120 piece: r, photograph 1 / 2 [mm] | M120 piece: angle, photograph 1 / 2 [deg] | "
               "M240 piece: r [mm] | M240 piece: angle [deg] | Bay width, photograph 1 / 2 [deg] |\n|---|---|---|---|---|---|")
    order = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-1"]
    for b in order:
        r1, r2 = j.loc[b]["r_mm_1"], j.loc[b]["r_mm_2"]
        out.append(f"| {b.replace('-', '–')} | {r1:.1f} / {r2:.1f} | {j.loc[b]['from_s_1']:.1f} / {j.loc[b]['from_s_2']:.1f} | "
                   f"{m240.loc[b]['r_mm']:.1f} | {m240.loc[b]['from_s']:.1f} | {j.loc[b]['bay_width_1']:.1f} / {j.loc[b]['bay_width_2']:.1f} |")
    out.append(f"| Mean | {j.r_mm_1.mean():.1f} / {j.r_mm_2.mean():.1f} | {j.from_s_1.mean():.1f} / {j.from_s_2.mean():.1f} | "
               f"{m240.r_mm.mean():.1f} | {m240.from_s.mean():.1f} | 60 / 60 |")
    out.append(f"| Standard deviation | {j.r_mm_1.std():.1f} / {j.r_mm_2.std():.1f} | {j.from_s_1.std():.1f} / {j.from_s_2.std():.1f} | "
               f"{m240.r_mm.std():.1f} | {m240.from_s.std():.1f} | {j.bay_width_1.std():.1f} / {j.bay_width_2.std():.1f} |")
    # SE2 — 3번 시편 기준 모드
    B = pd.read_csv(os.path.join(S3, "S3_base_modes.csv"), dtype={"mode": str})
    assign = {"2066": "h2:m2n0", "2076": "h2:m2n0", "3033": "h3:m3n0 + h0:m0n1 (mixed)", "3149": "h3:m3n0 + h0:m0n1 (mixed)",
              "3979": "h1:m1n1#2 (hub)", "6932": "h3:m3n2"}
    out.append("\n### Table SE2\n\n| f [Hz] | ζ [%] | Harmonic, ξ = 0.3 | Harmonic, ξ = 0.6 | Per-point SE, median [%] | "
               "Specimen 2 counterpart | Model family |\n|---|---|---|---|---|---|---|")
    def harm(x):
        """'h2 0.96 θ0 108'(S3_base_modes.csv) → 'h2 (0.96), 108°' — 몫은 괄호, 짝 모드만 무늬 방향 θ₀."""
        if not isinstance(x, str):
            return "—"
        m = re.match(r"(h\d) ([\d.]+)(?: θ0 (\d+))?$", x.strip())
        return x if m is None else f"{m.group(1)} ({m.group(2)})" + (f", {m.group(3)}°" if m.group(3) else "")
    for _, r in B.iterrows():
        h3, h6 = harm(r.get("h_xi0.3")), harm(r.get("h_xi0.6"))
        cp = r.pair_S2 if isinstance(r.pair_S2, str) else "—"
        cp = (cp.replace("무리", "group").replace("허브", "hub"))
        out.append(f"| {r.f_med:.0f} | {r.z_pct:.2f} | {h3} | {h6} | {r.se_med_pct:.3f} | {cp} | {assign.get(r['mode'], '—')} |")
    # SE5 — 3번 시편 모든 모드
    R, L = s3_summaries()
    changed_R = {"2066", "2077", "3033", "3554→3562", "3613→3595", "3980", "6932→6910"}
    changed_L = {"2066", "2077", "3033", "3149", "3296", "6732", "6932", "8031"}
    revR = {"5136", "7216→7228", "7749→7776"}
    revL = {"5136", "7216", "7749"}
    out.append("\n### Table SE5\n\n| Mode [Hz] | R: median, vanes 2–6 [%] | R: range [%] | R: class | L: median, vanes 2–6 [%] | "
               "L: range [%] | L: class |\n|---|---|---|---|---|---|---|")
    Rc = pd.read_csv(os.path.join(S3, "R12_check.csv"), dtype={"mode": str})
    Lc = pd.read_csv(os.path.join(S3, "L12_check.csv"), dtype={"mode": str})
    nR = Rc[(Rc.vane != 1) & np.isfinite(Rc.d_pct) & (Rc.se_pct < 0.08) & (Rc.n_a >= 5) & (Rc.n_b >= 5)].groupby("mode").size()
    nL = Lc[(Lc.vane != 1) & np.isfinite(Lc.RL_d) & (Lc.RL_se < 0.08) & (Lc.RL_na >= 5) & (Lc.RL_nb >= 5)].groupby("mode").size()

    def cls(n, ch, rv, med, npts):
        if n in ch:
            return "changed"
        if n in rv:
            return "reversible"
        if not np.isfinite(med):
            return "unresolved"
        return "weak" if npts < 6 else "floor"
    Rkeys = list(R.index)
    Lmap = {"3554→3562": "3555/3563", "3613→3595": "3592", "6932→6910": "6932", "7216→7228": "7216", "7749→7776": "7749",
            "8320→8310": "8320", "6411→6403": "6403", "3497": "3497/3517", "3517(new)": None, "6241→6266": "6241→6266"}
    for k in Rkeys:
        r = R.loc[k]
        lk = Lmap.get(k, k)
        lrow = L.loc[lk] if (lk is not None and lk in L.index) else None
        rm, rlo, rhi = r.get("d_v26", np.nan), r.get("d_lo", np.nan), r.get("d_hi", np.nan)
        if lrow is not None:
            lm, llo, lhi = lrow.get("RL_v26", np.nan), lrow.get("RL_lo", np.nan), lrow.get("RL_hi", np.nan)
        else:
            lm = llo = lhi = np.nan
        rr = f"{fmt(rlo)} to {fmt(rhi)}" if np.isfinite(rlo) else "—"
        lr = f"{fmt(llo)} to {fmt(lhi)}" if np.isfinite(llo) else "—"
        name = k.split("→")[0].replace("(new)", " (new after R)")
        lcls = cls(lk if lk else '', changed_L, revL, lm, nL.get(lk, 0) if lk else 0)
        if k in ("3497", "3517(new)", "3554→3562", "3613→3595"):        # 패널 무리: L에서 봉우리가 대역을 옮겨 다녀 하나씩 따라가지 않는다
            lm = llo = lhi = np.nan
            lr, lcls = "—", "panel cluster moved"
        out.append(f"| {name} | {fmt(rm)} | {rr} | {cls(k, changed_R, revR, rm, nR.get(k, 0))} | {fmt(lm)} | {lr} | {lcls} |")
    # SE4 — FEM 모양 변형
    def fem_all(folder):
        S_ = pd.read_csv(os.path.join(FEM, folder, "summary.csv"))
        rows = []
        for c in S_.case.unique():
            x = S_[S_.case == c].set_index("label")
            h2 = sorted([x.loc["h2:m2n0"].df1, x.loc["h2:m2n0"].df2])
            rows.append((c, x.loc["h2:m2n0"].removed_mg, x.loc["h2:m2n0"].size_mm, h2[0], h2[1],
                         x.loc["h3:m3n0"].df1 + x.loc["h0:m0n1"].df1, x.loc["h3:m3n2"].df1,
                         min(x.loc["h1:m1n1#2"].df1, x.loc["h1:m1n1#2"].df2)))
        return rows
    desc = {"RB20": "band ξ ≥ 0.9, ±10° about the vane line", "RB10": "band ξ ≥ 0.9, ±5°", "RB20p5": "as RB20, centered 5° toward vane 3",
            "RB20m5": "as RB20, centered 5° toward vane 5", "RV25": "bevel to ξ = 0.8, ±12.5°", "RW20": "window ξ ≥ 0.8, ±10°",
            "RW30": "window ξ ≥ 0.8, ±15°", "RB50": "band ξ ≥ 0.9, ±25°", "RB60": "band ξ ≥ 0.9, ±30°",
            "RB30p15": "band, vane line to bay center toward vane 3", "RB30m15": "band, vane line to bay center toward vane 5",
            "RS25b": "bevel to ξ = 0.8, 0° to 25° toward vane 5 (photograph)", "RS25f": "uniform ξ ≥ 0.8, 0° to 25° toward vane 5 (photograph)",
            "RS25w": "bevel to ξ = 0.6, 0° to 25° toward vane 5 (photograph)", "RS20b": "bevel to ξ = 0.8, 0° to 20° toward vane 5 (photograph)",
            "L17e": "band ξ ≥ 0.86, 17°, deepest at the rim, 2.5° toward vane 3 (photograph)",
            "L17f": "uniform ξ ≥ 0.86, 17°, 2.5° toward vane 3 (photograph)", "L17v": "bevel to ξ = 0.8, 17°, 2.5° toward vane 3 (photograph)",
            "L17c": "as L17e, at the bay center"}
    out.append("\n### Table SE4\n\n| Case | Footprint | Removed [mg] | Peak depth [mm] | h2 components [%] | Mixed-pair sum [%] | "
               "h3:m3n2 [%] | Hub h1 [%] |\n|---|---|---|---|---|---|---|---|")
    for folder in ("s3_r_grind", "s3_l_grind"):
        for c, mg, sz, a_, b_, fam, h3, hub in fem_all(folder):
            out.append(f"| {c} | {desc.get(c, '')} | {mg:.1f} | {sz:.3f} | {fmt(a_)} / {fmt(b_)} | {fmt(fam)} | {fmt(h3)} | {fmt(hub)} |")
    txt = "\n".join(out)
    open(os.path.join(FIGD, "tables_supp.md"), "w").write(txt + "\n")
    print(txt)


def main():
    os.makedirs(FIGD, exist_ok=True)
    T = t_series()
    T.to_csv(os.path.join(FIGD, "table_E2a_single_mass.csv"), index=False, float_format="%.4g")
    print("[T-a·T-b]\n", T.round(3).to_string(index=False))
    M, SP, chi = m_series()
    M.to_csv(os.path.join(FIGD, "table_E2b_sixfold_mass.csv"), index=False, float_format="%.4g")
    SP.to_csv(os.path.join(FIGD, "table_E2c_splitting.csv"), index=False, float_format="%.4g")
    print("\n[M120·M240 증분]\n", M.round(3).to_string(index=False))
    rp = M.ratio_photo.dropna()
    print(f"  사진 자리(칸 폭 환산) 비: {rp.min():.2f}–{rp.max():.2f}, 중앙 {rp.median():.2f} (n={len(rp)}, 0.55–1.23 안 "
          f"{int(rp.between(0.55, 1.23).sum())}); 범위±2SE 안 {int(M.in_range.sum())}/{len(M)}; χ² 환산 {chi['photo']:.0f}"
          f"(자리 불확도 넣으면 {chi['photo_pos']:.0f}), 원래 각 {chi['photo_raw']:.0f}, 공칭 {chi['nominal']:.0f}")
    rn = M[M.step == "M120"].ratio_nominal.dropna()
    print(f"  공칭 자리 비(M120): {rn.min():.2f}–{rn.max():.2f}")
    print("\n[갈라짐]\n", SP.round(3).to_string(index=False))
    E3, FR, FL = s3_table()
    E3.to_csv(os.path.join(FIGD, "table_E3_removal.csv"), index=False, float_format="%.4g")
    print("\n[깎기]\n", E3.round(3).to_string(index=False))
    print("\nFEM R\n", FR.round(3).to_string(), "\nFEM L\n", FL.round(3).to_string())
    RV = reversible()
    RV.to_csv(os.path.join(FIGD, "table_SE_reversible.csv"), index=False, float_format="%.4g")
    print("\n[되돌아간 모드]\n", RV.round(3).to_string(index=False))
    B, offR, offL = s3_floor()
    n17 = int((B.se_med_pct <= 0.0175).sum())
    print(f"\n[3번 시편 바닥] 자리별 표준오차 중앙(모드별) {B.se_med_pct.min():.3f}–{B.se_med_pct.max():.3f} %, "
          f"{n17}/{len(B)} 모드 ≤ 0.017 %; 깎은 직후 1번 깃 치우침 R {offR:+.3f}, L {offL:+.3f} %")
    md_tables()
    md_supp_tables()
    V = rehang()
    print(f"\n[다시 매달기 A2 → A2-rh] 값 {len(V)}개, 중앙 {V.d_pct.median():+.3f} %, rms {np.sqrt(np.mean(V.d_pct ** 2)):.3f} %, "
          f"최대 |{V.d_pct.abs().max():.3f}| % ({V.loc[V.d_pct.abs().idxmax(), 'mode']})")


if __name__ == "__main__":
    main()
