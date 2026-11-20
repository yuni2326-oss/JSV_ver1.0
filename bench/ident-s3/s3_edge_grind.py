"""깃 사이 테두리 끝만 깎기 — G3 자유-자유 FEM(32×16)으로 깊이 모양·자리 어긋남·질량에 대한 민감도(52번).

사용자(11-19 밤, 사진): 고르게 에칭하지 못하는 상황이면 오른쪽 1번 시편처럼 깃 사이 끝만 갈아내는 편이 결과가 더 좋겠다.
경우(측정 깃 4 = FEM 섹터 3, 방위는 그 반지름의 깃 선(camber)에서 잰다, 깃 3–4 사이 가운데 = +30°):
  (가) 깃 사이 가운데 테두리 띠의 앞면 감육, 한 단계 20 mg(FEM 밀도)
      B20   띠 ξ ≥ 0.9(테두리에서 2.2 mm), 방위 20° 고르게(끝 4° 경사)
      B20c  같은 띠, 방위 cos²(폭 20°) — 가운데 깊이 = 평균의 2배
      B20e  B20 × 반지름 경사(테두리에서 가장 깊고 ξ0.9에서 0) — 끝이 둥글게 닳은 모양
      B20k  B20 × 방위 경사(깃 4 쪽 끝이 가장 깊고 반대쪽 끝 0) — 한쪽으로 쏠린 손
      B30   띠 ξ ≥ 0.9, 30°
      U20w  창 ξ ≥ 0.8(테두리에서 4.3 mm), 20° — 51번 §3 절차의 창(비교)
  (나) 자리 어긋남: 가운데를 깃 4 쪽으로 o° 옮김 — B20o5·o10·o15·o20, U20wo10·o20
  (다) 테두리 자체를 줄로 깎음(앞판 두께 전부, 반지름이 δ만큼 줄어듦; ξ0.8 바깥 절점을 비례로 당긴다)
      T20   20° 고르게(끝 4° 경사),   T20c  δ가 cos²(폭 20°),   T20o10·T20o15 깃 4 쪽으로 옮김
  (라) 세 단계 누적 60 mg: B20x3·T20x3(선형성·깊이 한계),   S1e  1번 시편 S15 비슷한 경우(띠 ξ ≥ 0.9, 30°, 87.9 mg)
  (마) 1번 시편 사진의 모양 — 테두리 모서리 경사면(ξ0.8에서 0, 테두리에서 가장 깊음), 25°: V25(20 mg)·V25o10·S1v(87.9 mg)
  (바) 창(ξ ≥ 0.8)의 고르지 않음 비교(20 mg): U20wc 방위 cos², U20we 테두리 쪽이 깊게, U20wi 안쪽(ξ0.8)이 깊게
추적: 건전 관측량마다 손상 모드의 형상 부분공간 점수(s3_l1_cause.track과 같다), 점수 0.1 이상을 모두 적는다.
출력: fe-nominal/s3_edge_grind/<경우>.csv, summary.csv, s3_edge_grind.png
사용(PYTHONPATH=저장소, env -u PAPER3_GEOM, OMP_NUM_THREADS=2; 건전 캐시는 s3_l1_cause --healthy로 만든 것):
      python s3_edge_grind.py --cases B20,B20c --cache <pkl>    경우마다 약 3분
      python s3_edge_grind.py --summary"""
import argparse
import math
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from impeller_fingerprint.rev1 import n1_assembly as n1
from impeller_fingerprint.rev1 import n6_freefree as n6

import s3_l1_cause as lc

OUTD = Path(__file__).resolve().parents[1] / "fe-nominal" / "s3_edge_grind"
STEP = 20.0                                   # 한 단계 [mg]
# 경우 → (방법, 가운데 방위[° 깃 4 선에서], 방위 모양, 폭[°], 띠 안쪽 ξ, 반지름 모양, 질량 mg)
SPEC = {"B20": ("face", 30, "trap", 20, 0.9, "flat", STEP), "B20c": ("face", 30, "cos2", 20, 0.9, "flat", STEP),
        "B20e": ("face", 30, "trap", 20, 0.9, "ramp", STEP), "B20k": ("face", 30, "skew", 20, 0.9, "flat", STEP),
        "B30": ("face", 30, "trap", 30, 0.9, "flat", STEP), "U20w": ("face", 30, "trap", 20, 0.8, "flat", STEP),
        **{f"B20o{o}": ("face", 30 - o, "trap", 20, 0.9, "flat", STEP) for o in (5, 10, 15, 20)},
        **{f"U20wo{o}": ("face", 30 - o, "trap", 20, 0.8, "flat", STEP) for o in (10, 20)},
        "T20": ("trim", 30, "trap", 20, 0.8, None, STEP), "T20c": ("trim", 30, "cos2", 20, 0.8, None, STEP),
        **{f"T20o{o}": ("trim", 30 - o, "trap", 20, 0.8, None, STEP) for o in (10, 15)},
        "B20x3": ("face", 30, "trap", 20, 0.9, "flat", 3 * STEP), "T20x3": ("trim", 30, "trap", 20, 0.8, None, 3 * STEP),
        "S1e": ("face", 30, "trap", 30, 0.9, "flat", 87.9),
        "V25": ("face", 30, "trap", 25, 0.8, "ramp", STEP), "V25o10": ("face", 20, "trap", 25, 0.8, "ramp", STEP),
        "S1v": ("face", 30, "trap", 25, 0.8, "ramp", 87.9),
        "U20wc": ("face", 30, "cos2", 20, 0.8, "flat", STEP), "U20we": ("face", 30, "trap", 20, 0.8, "ramp", STEP),
        "U20wi": ("face", 30, "trap", 20, 0.8, "inner", STEP)}
GROUPS = [("(a) bay centre, one 20 mg step: depth profile and method varied",
           ["U20w", "U20wc", "U20we", "U20wi", "B20", "B20c", "B20e", "B20k", "B30", "V25", "T20", "T20c"]),
          ("(b) placement error: centre moved toward the vane-4 line by o (20 mg)",
           ["B20", "B20o5", "B20o10", "B20o15", "B20o20", "U20w", "U20wo10", "U20wo20", "V25", "V25o10", "T20", "T20o10",
            "T20o15"]),
          ("(c) linearity: three steps in the same bay (60 mg) divided by 3, against one 20 mg step",
           ["B20", "B20x3", "T20", "T20x3"])]
CASES = list(SPEC)
# 보여 줄 관측량(FEM 이름 → 측정 이름, s3_l1_cause.MEAS와 같은 짝)
LABS = {k: v[0] for k, v in lc.MEAS.items()}
E_DEG = 4.0                                   # 고른 창 끝 경사 폭[°] (절점 간격 3.6°)


def front_sel(fm):
    ff, X = fm.spec, fm.coors
    z_in, z_top = ff.z_front
    return (X[:, 2] >= z_in - 1e-12) & (X[:, 2] <= z_top + 1e-12) & (fm.r_v >= ff.a - 1e-12)


def d_deg(fm, rel):
    """절점 방위 − (깃 4 선 + rel), 그 반지름의 깃 선에서 잰다. +는 깃 3 쪽(창 쪽)."""
    X = fm.coors
    th = np.arctan2(X[:, 1], X[:, 0]) - fm.spec.base.camber(fm.r_v)
    return np.degrees((th - math.radians(60.0 * lc.K_SEC + rel) + math.pi) % (2 * math.pi) - math.pi)


def ang_profile(fm, centre, kind, w):
    d = d_deg(fm, centre)
    trap = np.clip((w / 2 + E_DEG / 2 - np.abs(d)) / E_DEG, 0.0, 1.0)
    if kind == "trap":
        return trap
    if kind == "cos2":
        return np.where(np.abs(d) <= w / 2, np.cos(math.pi * d / w) ** 2, 0.0)
    if kind == "skew":                         # 깃 4 쪽(d < 0) 끝이 가장 깊다
        return trap * np.clip(0.5 - d / w, 0.0, 1.0)
    raise ValueError(kind)


def face_shape(fm, centre, kind, w, xi0, radial):
    xi = (fm.r_v - fm.spec.a) / fm.spec.extent
    if radial == "flat":
        rad = (xi >= xi0 - 1e-9).astype(float)
    elif radial == "ramp":                     # 테두리에서 가장 깊고 ξ0에서 0
        rad = np.clip((xi - xi0) / (1.0 - xi0), 0.0, 1.0)
    else:                                      # inner: ξ0에서 가장 깊고 테두리에서 0(안쪽으로 쏠린 손)
        rad = (xi >= xi0 - 1e-9) * np.clip((1.0 - xi) / (1.0 - xi0), 0.0, 1.0)
    return front_sel(fm) * rad * ang_profile(fm, centre, kind, w)


def morph_trim(fm, q, r0):
    """앞판 테두리를 반지름 안쪽으로 q[m](절점별)만큼 당긴다 — r0 바깥 앞판 절점을 (r − r0)/(b − r0) 비율로 옮긴다."""
    w = np.clip((fm.r_v - r0) / (fm.spec.b - r0), 0.0, 1.0)
    sel = front_sel(fm) & (w > 0) & (q > 0)
    c = fm.coors.copy()
    k = 1.0 - q[sel] * w[sel] / fm.r_v[sel]
    c[sel, 0] *= k
    c[sel, 1] *= k
    return c


def removed(fm, coors, v0):
    return fm.spec.base.rho * (v0 - lc.hex_volume(n1.physical_coors(fm, coors), fm.conn)) * 1e6


def build_case(fm, name, v0):
    meth, centre, kind, w, xi0, radial, mg = SPEC[name]
    if meth == "face":
        s = face_shape(fm, centre, kind, w, xi0, radial)
        p = s * (mg / lc.removed_mg(fm, s, v0))
        coors = n6.morph_groove(fm, p)
        info = f"가장 깊은 곳 {p.max() * fm.spec.base.t_front * 1e3:.3f} mm(두께의 {100 * p.max():.0f} %)"
        if p.max() > 0.95:
            info += " — 두께를 거의 다 뚫음(실제로는 불가)"
        return coors, info, float(p.max() * fm.spec.base.t_front * 1e3)
    s = front_sel(fm) * ang_profile(fm, centre, kind, w)
    r0 = fm.spec.a + xi0 * fm.spec.extent
    arc = fm.spec.b * math.radians(w)
    delta = mg / (fm.spec.base.rho * 1e6 * fm.spec.base.t_front * arc)        # 첫 짐작 [m]
    for _ in range(4):
        m = removed(fm, morph_trim(fm, delta * s, r0), v0)
        delta *= mg / m
    coors = morph_trim(fm, delta * s, r0)
    return coors, f"테두리를 최대 {delta * 1e3:.3f} mm 안쪽으로(ξ{xi0} 바깥 절점 비례 이동)", float(delta * 1e3)


def run_cases(names, cache):
    with open(cache, "rb") as f:
        H = pickle.load(f)
    md, obs, v0 = H["md"], H["obs"], H["v0"]
    ff = lc.spec()
    fm = n6.build(ff)
    assert np.allclose(fm.r_v, np.hypot(fm.coors[:, 0], fm.coors[:, 1]), atol=1e-9)
    OUTD.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        coors, info, size_mm = build_case(fm, name, v0)
        m = removed(fm, coors, v0)
        km = n6.assemble(fm, coors)
        mdd, _ = n6.solve(km, md.phi.shape[1] + 6, n_rigid=ff.n_rigid)
        R = pd.DataFrame(lc.track(md, obs, mdd))
        R.insert(0, "case", name)
        R["removed_mg"] = m
        R["size_mm"] = size_mm
        R.to_csv(OUTD / f"{name}.csv", index=False, float_format="%.6g")
        print(f"{name}: {info}, 깎은 질량 {m:.2f} mg ({time.time() - t0:.0f}s)", flush=True)


def load_summary():
    rows = []
    for name in CASES:
        fp = OUTD / f"{name}.csv"
        if not fp.exists():
            continue
        R = pd.read_csv(fp)
        for lab, g in R.groupby("label", sort=False):
            g = g.sort_values("score", ascending=False)
            deg = int(g.deg.iloc[0])
            top = g.head(deg)
            pair = deg == 2 and len(top) > 1
            rows.append(dict(case=name, label=lab, deg=deg, f0_Hz=g.f0_Hz.iloc[0], df1=top.df_pct.iloc[0], s1=top.score.iloc[0],
                             df2=top.df_pct.iloc[1] if pair else np.nan, s2=top.score.iloc[1] if pair else np.nan,
                             n_spread=int((g.score >= 0.1).sum()), removed_mg=g.removed_mg.iloc[0], size_mm=g.size_mm.iloc[0]))
    return pd.DataFrame(rows)


def cell(r):
    star = "*" if r.s1 < 0.6 else ""
    if r.deg == 2:
        a, b = sorted([r.df1, r.df2], key=abs)
        return f"{a:+.2f}/{b:+.2f}{star}"
    return f"{r.df1:+.2f}{star}"


def summarize():
    S = load_summary()
    S.to_csv(OUTD / "summary.csv", index=False, float_format="%.5g")
    labs = [l for l in LABS if l in set(S.label)]
    print("\n[관측량별 FEM 이동 %] 짝은 '작은/큰' 성분, * = 가장 큰 점수 < 0.6(섞임)")
    print(f"  {'case':7s} {'mg':>5s} {'size':>6s} " + " ".join(f"{LABS[l]:>11s}" for l in labs))
    for name in CASES:
        x = S[S.case == name].set_index("label")
        if x.empty:
            continue
        r0 = x.iloc[0]
        print(f"  {name:7s} {r0.removed_mg:5.1f} {r0.size_mm:6.3f} " +
              " ".join(f"{cell(x.loc[l]) if l in x.index else '--':>11s}" for l in labs))
    # 핵심 관측량 — 짝은 두 성분 합(손상 자리와 짝 방향이 어긋나도 1차로 같다)
    KEY = ["h2:m2n0", "h3:m3n0", "h0:m0n1", "h0:m6n1", "h1:m7n0", "h3:m3n2", "h3:m3n1#2"]
    print("\n[핵심 관측량 %, 짝은 두 성분 합] " + " ".join(f"{LABS[l]:>9s}" for l in KEY))
    K = []
    for name in CASES:
        x = S[S.case == name].set_index("label")
        if x.empty:
            continue
        v = {l: (x.loc[l].df1 + (x.loc[l].df2 if x.loc[l].deg == 2 else 0.0)) for l in KEY if l in x.index}
        mix = {l: x.loc[l].s1 < 0.6 for l in KEY if l in x.index}
        K.append(dict(case=name, removed_mg=x.iloc[0].removed_mg, size_mm=x.iloc[0].size_mm, **{LABS[l]: v[l] for l in v}))
        print(f"  {name:7s} " + " ".join(f"{v[l]:+8.2f}{'*' if mix[l] else ' '}" for l in KEY))
    pd.DataFrame(K).to_csv(OUTD / "key.csv", index=False, float_format="%.4g")
    # 깊이 모양 민감도: 깃 사이 가운데 20 mg 경우끼리(관측량별 큰 성분)
    big = lambda r: r.df1 if r.deg == 1 or abs(r.df1) >= abs(r.df2) else r.df2
    print("\n[같은 20 mg, 깃 사이 가운데: 경우 사이 범위(최소 ~ 최대) — 큰 성분]")
    for grp in (["B20", "B20c", "B20e", "B20k", "B30"], ["T20", "T20c"], ["U20w", "U20wc", "U20we", "U20wi"]):
        y = S[S.case.isin(grp)]
        if y.empty:
            continue
        out = []
        for l in labs:
            v = [big(r) for _, r in y[y.label == l].iterrows()]
            if v:
                out.append(f"{LABS[l]} {min(v):+.2f}~{max(v):+.2f}")
        print(f"  {'·'.join(grp)}: " + ", ".join(out))
    figure(S)


def figure(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = [l for l in LABS if l in set(S.label)]
    xs = {l: i for i, l in enumerate(labs)}
    fig = plt.figure(figsize=(15, 14))
    gsp = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.85, 1.0], hspace=0.45, wspace=0.28)
    big_axes = [fig.add_subplot(gsp[0, :]), fig.add_subplot(gsp[2, :])]
    for ax, (title, names) in zip(big_axes, [GROUPS[0], GROUPS[2]]):
        names = [n for n in names if (S.case == n).any()]
        for j, name in enumerate(names):
            x = S[S.case == name].set_index("label")
            dx = (j - (len(names) - 1) / 2) * min(0.08, 0.8 / max(len(names), 1))
            col = plt.get_cmap("tab10")(j)
            mk = MARK[SPEC[name][0] if SPEC[name][0] == "trim" else (SPEC[name][4], SPEC[name][5])]
            scale = STEP / SPEC[name][6] if name.endswith("x3") else 1.0
            for l in labs:
                if l not in x.index:
                    continue
                r = x.loc[l]
                vals = [r.df1] + ([r.df2] if r.deg == 2 and np.isfinite(r.df2) else [])
                for u in vals:
                    ax.plot(xs[l] + dx, u * scale, mk, color=col, ms=4.5, mfc=col if r.s1 >= 0.6 else "white")
            ax.plot([], [], mk, color=col, label=name + (" ÷ 3" if scale != 1 else ""))
        ax.axhline(0, color="k", lw=0.6)
        ax.axhspan(-0.05, 0.05, color="0.88", zorder=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.005, 1.0))
        ax.set_ylabel("frequency change [%]")
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([f"{LABS[l]}\n{l}" for l in labs], fontsize=7, rotation=45)
        ax.set_title(title, fontsize=9, loc="left")
    big_axes[0].text(0.0, 1.11, "o face band ξ≥0.9 (2.2 mm from rim), ◆ window ξ≥0.8 (4.3 mm), ▲ rim bevel (deepest at "
                     "the rim, 0 at ξ0.8), ■ rim filed back radially; open = mode split (top score < 0.6); grey = ±0.05 % floor",
                     transform=big_axes[0].transAxes, fontsize=8)
    # (b) 자리 어긋남: 가운데와 깃 4 선 사이 방위
    series = [("band ξ≥0.9", ["B20", "B20o5", "B20o10", "B20o15", "B20o20"], "o", "C0"),
              ("window ξ≥0.8", ["U20w", "U20wo10", "U20wo20"], "D", "C1"),
              ("rim bevel", ["V25", "V25o10"], "^", "C2"), ("rim filed", ["T20", "T20o10", "T20o15"], "s", "C3")]
    for i, (lab, title) in enumerate([("h3:m3n0", "3164"), ("h2:m2n0", "h2 pair 2134/2142 (sum of both)"),
                                      ("h0:m6n1", "3804"), ("h3:m3n2", "7093")]):
        ax = fig.add_subplot(gsp[1, i])
        for sname, names, mk, col in series:
            pts = []
            for n in names:
                r = S[(S.case == n) & (S.label == lab)]
                if r.empty:
                    continue
                r = r.iloc[0]
                v = r.df1 if r.deg == 1 else r.df1 + r.df2          # 짝은 두 성분 합(방향에 무관)
                pts.append((SPEC[n][1], v, r.s1 >= 0.6))
            if pts:
                pts.sort()
                ax.plot([q[0] for q in pts], [q[1] for q in pts], "-", color=col, lw=1)
                for q in pts:
                    ax.plot(q[0], q[1], mk, color=col, ms=5, mfc=col if q[2] else "white")
                ax.plot([], [], mk + "-", color=col, label=sname)
        ax.axhline(0, color="k", lw=0.6)
        ax.axhspan(-0.05, 0.05, color="0.88", zorder=0)
        ax.grid(alpha=0.3)
        ax.set_xlim(32, 8)
        ax.set_xlabel("patch centre from vane-4 line [deg]\n(30 = bay centre)", fontsize=8)
        ax.set_title(f"(b) {title}", fontsize=9, loc="left")
        if i == 0:
            ax.set_ylabel("frequency change [%], 20 mg")
            ax.legend(fontsize=7.5, loc="lower left")
    fig.savefig(OUTD / "s3_edge_grind.png", dpi=110, bbox_inches="tight")


MARK = {"trim": "s", (0.9, "flat"): "o", (0.8, "flat"): "D", (0.8, "ramp"): "^", (0.9, "ramp"): "o", (0.8, "inner"): "D"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "s3_l1_cause_healthy.pkl"))
    a = ap.parse_args()
    if a.cases:
        run_cases([c for c in a.cases.split(",") if c], a.cache)
    if a.summary:
        summarize()
