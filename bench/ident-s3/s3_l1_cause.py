"""2번 시편 L1(11-19 S16, 깃 3–4 사이를 손으로 49.6 mg) 원인 후보 — G3 자유-자유 FEM(32×16)에서 깎은 모양을 바꿔 다시 푼다.

측정(50번 §4, 식은 쪽 값): 3164 −1.35 · 7093 −0.58 · 3804 −0.55 · 3748 −0.33 · 4967 짝 −0.2 · 5617 −0.2 · 2142 −0.17 ·
6817 −0.10 · 4225 −0.09 · 3250 −0.05 · 5745 −0.03 %, 3459·3639 봉우리 사라짐(3514 Hz 새로), 7462 +0.22 %.
같은 자리 고른 창(0.06 mm·37.4 mg) FEM 예측은 패널 +0.2 ~ +0.4 %였다(s3_pred_windows).
후보(측정 깃 v ↔ FEM 섹터 (−(v − 1)) mod 6: 깃 4 = 섹터 3, 깃 3–4 사이 = 섹터 3 + 30°). 방위는 그 반지름의 깃 선(camber)에서 잰다.
  (가) 깃 사이 가운데가 깊은 감육 — 사용자(11-19 저녁): '깃 사이 가운데가 제일 얇아졌을 것'. 모두 49.6 mg(FEM 밀도)에 맞춘다.
      U30  바깥 띠 ξ0.8–1.0(테두리에서 4.3 mm), 30° 고르게(0.080 mm) — L1 예측을 실제 질량으로 다시 푼 것
      C30  같은 띠, 방위 cos²(폭 30°): 가운데 깊이 = 평균의 2배
      C20  cos²(폭 20°): 3배,   C16  cos²(폭 16°): 3.75배(가운데 0.30 mm = 두께의 75 %)
      C30e C30 × 반지름 경사(테두리에서 가장 깊고 ξ0.8에서 0)
      W30  안쪽으로 넓은 자국 ξ0.6–1.0(8.6 mm), 30° 고르게(50번 §0: 창보다 안쪽으로 넓어 보임)
  (나) 깃 4 선 위까지 간 자국
      R30  깃 4 위 30° 창(R1 자리) 고르게,   S50  깃 4 선 너머 5°부터 창 끝(깃 3 쪽 45°)까지 50° 고르게
      E20  깃 4 선 너머 5°부터 창 쪽 15°까지(20°) 고르게,   E10  −3° ~ +7°(10°) 고르게 — 창 쪽에서 깃 4 선까지 한쪽으로 간 경우
      HALF 반(24.8 mg)은 U30, 반은 E20
  (다) 깃 4–앞판 접합의 바깥 끝을 푼다(깎은 질량 없음) — 깃 윗층 셀(z 2.1–3.6 mm) 강성 ×10⁻³: J2·J4·J8 = 테두리에서 2·4.3·8 mm
      E20J2 E20 + J2
추적: 건전 관측량마다 손상 모드의 형상 부분공간 점수(꼭짓점 변위, s3_pred_windows.resolved와 같은 정의)를 내고, 점수 0.1 이상인
      손상 모드를 모두 적는다. 섞이면(패널 짝) 점수가 여러 모드로 나뉜다.
출력: fe-nominal/s3_l1_cause/<경우>.csv(관측량 × 손상 모드), summary.csv(관측량별 대표값·측정), s3_l1_cause.png
사용(PYTHONPATH=저장소, env -u PAPER3_GEOM, OMP_NUM_THREADS=2):
      python s3_l1_cause.py --healthy --cache <pkl>          건전 모드(약 4분)
      python s3_l1_cause.py --cases U30,C30 --cache <pkl>    경우마다 약 3분 — 여러 프로세스로 나눠 돌릴 수 있다
      python s3_l1_cause.py --summary"""
import argparse
import math
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from impeller_fingerprint import geometry as geo
from impeller_fingerprint.rev1 import n1_assembly as n1
from impeller_fingerprint.rev1 import n1_mesh as nm
from impeller_fingerprint.rev1 import n6_freefree as n6

OUTD = Path(__file__).resolve().parents[1] / "fe-nominal" / "s3_l1_cause"
assert geo.ASBUILT.gap_in is None, "G3(간격 일정)로 돌린다 — PAPER3_GEOM을 비울 것"
MASS_MG = 49.6
SOFT = 1e-3
K_SEC = 3                                   # 측정 깃 4
# 경우 → ([(깎은 모양, 질량 mg)], 푼 접합 길이 m 또는 None)
_JL = {"J2": 2.0e-3, "J4": 4.315e-3, "J8": 8.0e-3}
SPEC = {**{n: ([(n, MASS_MG)], None) for n in ("U30", "C30", "C20", "C16", "C30e", "W30", "R30", "S50", "E20", "E10")},
        "HALF": ([("U30", MASS_MG / 2), ("E20", MASS_MG / 2)], None),
        **{n: ([], L) for n, L in _JL.items()},
        "E20J2": ([("E20", MASS_MG)], _JL["J2"])}
GROUPS = [("(a) thinning between vanes 3–4, same 49.6 mg, depth profile varied", ["U30", "C30", "C20", "C16", "C30e", "W30"]),
          ("(b) ground up to or over the vane-4 line, 49.6 mg", ["R30", "S50", "E20", "E10", "HALF"]),
          ("(c) vane-4/front-plate joint released at the outer end (J: no mass removed)", ["J2", "J4", "J8", "E20J2"])]
CASES = [n for _, g in GROUPS for n in g]
# 측정(50번 §4 결과 3, 식은 쪽): FEM 관측량 → (측정 이름, 값[%] 또는 None, 메모). 짝은 두 성분 값(알면)
MEAS = {"h2:m2n0": ("2134/2142", (-0.03, -0.17), "2142 −0.17, 2134 ~0"),
        "h3:m3n0": ("3164", -1.35, ""), "h0:m0n1": ("3250", -0.05, ""),
        "h1:m1n1": ("3459", None, "봉우리 사라짐"), "h2:m4n0": ("3639", None, "봉우리 사라짐"),
        "h3:m3n1": ("3748", -0.33, ""), "h0:m6n1": ("3804", -0.55, ""), "h1:m1n1#2": ("4225", -0.09, ""),
        "h2:m4n0#2": ("4967", (-0.18, -0.24), "두 성분"), "h0:m6n0": ("5617", -0.20, "약함"),
        "h1:m5n0": ("5745", -0.03, ""), "h2:m2n1": ("6558", -0.15, "약함"), "h1:m7n0": ("6817", -0.10, ""),
        "h3:m3n2": ("7093", -0.58, ""), "h3:m3n1#2": ("8321", -0.30, "−0.1 ~ −0.6, 짝 불확실")}
# 맞음 지표는 섞이지 않는 모드만(손상 뒤에도 한 모드가 점수 0.85 이상을 갖는 관측량). 패널 무리(3164·3250·3459·3639)는
# 손상이 크면 여러 모드로 나뉘어 '대표 모드'가 정해지지 않으므로 따로 본다(아래 패널 짝 섞임 표).
FIT = ["h2:m2n0", "h3:m3n1", "h0:m6n1", "h1:m1n1#2", "h2:m4n0#2", "h0:m6n0", "h1:m5n0", "h3:m3n2", "h3:m3n1#2"]


def spec():
    return n6.FFSpec(base=nm.FitSpec(n_r=32, n_bay=16, wrap_deg=geo.ASBUILT.wrap_deg), n_hub=6, n_z_collar=4,
                     collar_h=geo.ASBUILT.collar_h, clamp_r=None)


_SG = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def hex_volume(X, conn):
    """삼선형 육면체 부피 합(2×2×2 가우스 = 정확). z만 옮기면 부피는 꼭짓점 z에 선형이다."""
    E = X[conn]
    v = np.zeros(len(conn))
    g = 1.0 / math.sqrt(3.0)
    for a in (-g, g):
        for b in (-g, g):
            for c in (-g, g):
                x = np.array([a, b, c])
                dN = np.empty((8, 3))
                for j in range(3):
                    t = _SG[:, j] / 8.0
                    for k in range(3):
                        if k != j:
                            t = t * (1 + _SG[:, k] * x[k])
                    dN[:, j] = t
                v += np.linalg.det(np.einsum("cni,nj->cij", E, dN))
    return float(np.abs(v).sum())


def removed_mg(fm, p, v0):
    X = n1.physical_coors(fm, n6.morph_groove(fm, p))
    return fm.spec.base.rho * (v0 - hex_volume(X, fm.conn)) * 1e6


def shape(fm, name):
    """경우 이름 → 앞판 꼭짓점 깊이 모양 s ∈ [0, 1](세기는 질량으로 맞춘다)."""
    ff, fs = fm.spec, fm.spec.base
    X = fm.coors
    z_in, z_top = ff.z_front
    front = (X[:, 2] >= z_in - 1e-12) & (X[:, 2] <= z_top + 1e-12) & (fm.r_v >= ff.a - 1e-12)
    xi = (fm.r_v - ff.a) / ff.extent
    th = np.arctan2(X[:, 1], X[:, 0]) - fs.camber(fm.r_v)

    def d_deg(rel):
        return np.degrees((th - math.radians(60.0 * K_SEC + rel) + math.pi) % (2 * math.pi) - math.pi)

    band = front & (xi >= 0.8 - 1e-9)
    d = d_deg(30.0)
    cos2 = lambda w: np.where(np.abs(d) <= w / 2, np.cos(math.pi * d / w) ** 2, 0.0)
    s = {"U30": band & (np.abs(d) <= 15.0 + 1e-9),
         "C30": band * cos2(30.0), "C20": band * cos2(20.0), "C16": band * cos2(16.0),
         "C30e": band * cos2(30.0) * np.clip((xi - 0.8) / 0.2, 0.0, 1.0),
         "W30": front & (xi >= 0.6 - 1e-9) & (np.abs(d) <= 15.0 + 1e-9),
         "R30": band & (np.abs(d_deg(0.0)) <= 15.0 + 1e-9),
         "S50": band & (d_deg(0.0) >= -5.0 - 1e-9) & (d_deg(0.0) <= 45.0 + 1e-9),
         "E20": band & (d_deg(0.0) >= -5.0 - 1e-9) & (d_deg(0.0) <= 15.0 + 1e-9),
         "E10": band & (d_deg(0.0) >= -3.0 - 1e-9) & (d_deg(0.0) <= 7.0 + 1e-9)}[name]
    return np.asarray(s, float)


def joint_mask(fm, length):
    """깃(섹터 K_SEC) 윗층 셀 중 테두리에서 length 안쪽까지."""
    info = fm.info
    zc = fm.coors[fm.conn][:, :, 2].mean(axis=1)
    z_mid = fm.spec.z_breaks[fm.spec.base.n_z_shroud + 1]           # 두 깃 층의 경계(2.1 mm)
    return ((info["cell_kind"] == 1) & (info["cell_vane"] == K_SEC) & (zc > z_mid)
            & (info["cell_r"] >= fm.spec.b - length))


def healthy(cache):
    t0 = time.time()
    ff = spec()
    fm = n6.build(ff)
    km = n6.assemble(fm)
    md, _ = n6.solve(km, 80, n_rigid=ff.n_rigid)
    obs = n6.classify(fm, md)
    v0 = hex_volume(n1.physical_coors(fm, fm.coors), fm.conn)
    print(f"건전 {md.freqs[0]:.1f}–{md.freqs[-1]:.1f} Hz, 관측량 {len(obs)}, 부피 {v0 * 1e6:.4f} cm³ "
          f"(sfepy {km.volume * 1e6:.4f}) ({time.time() - t0:.0f}s)", flush=True)
    with open(cache, "wb") as f:
        pickle.dump(dict(md=md, obs=obs, v0=v0), f, protocol=4)


def track(md, obs, mdd):
    Vd = mdd.vshape / np.linalg.norm(mdd.vshape, axis=1, keepdims=True)
    rows = []
    for o in obs:
        Q, _ = np.linalg.qr(md.vshape[o.members].T)
        score = np.sum((Vd @ Q) ** 2, axis=1)
        f0 = float(np.mean(md.freqs[o.members]))
        for k in np.nonzero(score >= 0.1)[0]:
            rows.append(dict(label=o.label, h=o.h, deg=o.deg, f0_Hz=f0, k=int(k), f_Hz=float(mdd.freqs[k]),
                             df_pct=100 * (float(mdd.freqs[k]) / f0 - 1), score=float(score[k])))
    return rows


def run_cases(names, cache):
    with open(cache, "rb") as f:
        H = pickle.load(f)
    md, obs, v0 = H["md"], H["obs"], H["v0"]
    ff = spec()
    fm = n6.build(ff)
    km_h = None
    OUTD.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        parts, jl = SPEC[name]
        p = np.zeros(len(fm.coors))
        for sh, mg in parts:
            s = shape(fm, sh)
            p = p + s * (mg / removed_mg(fm, s, v0))
        m = removed_mg(fm, p, v0) if parts else 0.0
        info = (f"가장 깊은 곳 {p.max() * ff.base.t_front * 1e3:.3f} mm(두께의 {100 * p.max():.0f} %), 깎은 질량 {m:.2f} mg"
                if parts else "깎지 않음")
        coors = n6.morph_groove(fm, p) if parts else None
        if parts or km_h is None:
            km = n6.assemble(fm, coors)
            if not parts:
                km_h = km
        else:
            km = km_h
        if jl is not None:
            mask = joint_mask(fm, jl)
            kr = n6.assemble(fm, coors, cell_mask=mask)
            km = n1.KM(km.K - (1.0 - SOFT) * kr.K, km.M, km.make_full, km.n_vertex, km.volume)
            info += f", 푼 셀 {int(mask.sum())}개(r ≥ {(ff.b - jl) * 1e3:.2f} mm)"
        mdd, _ = n6.solve(km, md.phi.shape[1] + 6, n_rigid=ff.n_rigid)
        R = pd.DataFrame(track(md, obs, mdd))
        R.insert(0, "case", name)
        R["removed_mg"] = m
        R.to_csv(OUTD / f"{name}.csv", index=False, float_format="%.6g")
        print(f"{name}: {info} ({time.time() - t0:.0f}s)", flush=True)


def summarize():
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
            w = g[g.score >= 0.1]
            mn = MEAS.get(lab)
            rows.append(dict(case=name, label=lab, deg=deg, f0_Hz=g.f0_Hz.iloc[0],
                             df1=top.df_pct.iloc[0], s1=top.score.iloc[0],
                             df2=top.df_pct.iloc[1] if deg == 2 and len(top) > 1 else np.nan,
                             s2=top.score.iloc[1] if deg == 2 and len(top) > 1 else np.nan,
                             df_w=float(np.sum(w.df_pct * w.score) / np.sum(w.score)), n_spread=int((g.score >= 0.1).sum()),
                             meas_name=mn[0] if mn else "", removed_mg=g.removed_mg.iloc[0]))
    S = pd.DataFrame(rows)
    S.to_csv(OUTD / "summary.csv", index=False, float_format="%.5g")
    # 측정과 맞대기: 홑은 대표 모드, 짝은 두 성분 평균(측정이 두 성분이면 평균)
    print("\n[측정 대비] 관측량별 FEM 이동 [%] (짝은 두 성분 'a/b', 점수 < 0.6이면 *)")
    lab_order = [lab for lab in MEAS]
    head = "  case  " + " ".join(f"{MEAS[l][0]:>11s}" for l in lab_order)
    print(head)
    mv = []
    for l in lab_order:
        v = MEAS[l][1]
        mv.append("     (none)" if v is None else (f"{v[0]:+.2f}/{v[1]:+.2f}" if isinstance(v, tuple) else f"{v:+11.2f}"))
    print("  측정  " + " ".join(f"{x:>11s}" for x in mv))
    out = []
    for name in CASES:
        x = S[S.case == name].set_index("label")
        if x.empty:
            continue
        cells = []
        for l in lab_order:
            if l not in x.index:
                cells.append("         --")
                continue
            r = x.loc[l]
            star = "*" if r.s1 < 0.6 else " "
            cells.append(f"{r.df1:+.2f}/{r.df2:+.2f}{star}" if r.deg == 2 else f"{r.df1:+10.2f}{star}")
        print(f"  {name:5s} " + " ".join(f"{c:>11s}" for c in cells))
        # 맞음: 측정 값이 있는 FIT 관측량(짝은 평균) — 그대로의 rms와 배율 c를 맞춘 뒤의 rms
        a, b = [], []
        for l in FIT:
            if l not in x.index or MEAS[l][1] is None:
                continue
            r = x.loc[l]
            fem = 0.5 * (r.df1 + r.df2) if r.deg == 2 else r.df1
            mv_ = np.mean(MEAS[l][1]) if isinstance(MEAS[l][1], tuple) else MEAS[l][1]
            a.append(fem)
            b.append(mv_)
        a, b = np.array(a), np.array(b)
        c = float(a @ b / (a @ a)) if a @ a > 0 else np.nan
        out.append(dict(case=name, n=len(a), rms=float(np.sqrt(np.mean((a - b) ** 2))), scale=c,
                        rms_scaled=float(np.sqrt(np.mean((c * a - b) ** 2))), corr=float(np.corrcoef(a, b)[0, 1])))
    F = pd.DataFrame(out)
    F.to_csv(OUTD / "fit.csv", index=False, float_format="%.4g")
    print("\n[모양 맞음] FIT 관측량(" + ", ".join(MEAS[l][0] for l in FIT) + ")")
    print(F.round(3).to_string(index=False))
    # 패널 짝(3459·3639)이 어디로 갔나
    print("\n[패널 짝 섞임] h1:m1n1(3459)·h2:m4n0(3639)·h3:m3n0(3164) 건전 모양을 가진 손상 모드(점수 ≥ 0.1)")
    for name in CASES:
        fp = OUTD / f"{name}.csv"
        if not fp.exists():
            continue
        R = pd.read_csv(fp)
        for lab in ("h3:m3n0", "h1:m1n1", "h2:m4n0"):
            g = R[R.label == lab].sort_values("f_Hz")
            print(f"  {name:5s} {lab:9s} " + ", ".join(f"{r.df_pct:+.2f}%({r.score:.2f})" for _, r in g.iterrows()))
    figure(S, F)


def figure(S, F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = [l for l in MEAS]
    xs = {l: i for i, l in enumerate(labs)}
    fig, axs = plt.subplots(3, 1, figsize=(14.5, 11.5), sharex=True)
    YL = (-3.2, 3.0)                     # (c)의 J8은 이 밖 — 경계에 ▼/▲로 찍고 값을 적는다
    for i, (ax, (title, names)) in enumerate(zip(axs, GROUPS)):
        for l in labs:
            v = MEAS[l][1]
            if v is None:
                ax.text(xs[l], 0.0, "gone", ha="center", va="bottom", fontsize=7, color="k")
                continue
            vv = v if isinstance(v, tuple) else (v,)
            for u in vv:
                ax.plot([xs[l] - 0.4, xs[l] + 0.4], [u, u], "k-", lw=2.5, zorder=5)
        for j, name in enumerate(names):
            x = S[S.case == name].set_index("label")
            if x.empty:
                continue
            dx = (j - (len(names) - 1) / 2) * 0.13
            for l in labs:
                if l not in x.index:
                    continue
                r = x.loc[l]
                vals = [r.df1] + ([r.df2] if r.deg == 2 and np.isfinite(r.df2) else [])
                for u in vals:
                    if i == 2 and not YL[0] < u < YL[1]:
                        ub = min(max(u, YL[0] + 0.15), YL[1] - 0.15)
                        ax.plot(xs[l] + dx, ub, "v" if u < 0 else "^", color=f"C{j}", ms=6)
                        ax.text(xs[l] + dx + 0.05, ub, f"{u:+.1f}", fontsize=6.5, color=f"C{j}", va="center")
                        continue
                    ax.plot(xs[l] + dx, u, "o", color=f"C{j}", ms=5, mfc=f"C{j}" if r.s1 >= 0.6 else "white")
            ax.plot([], [], "o", color=f"C{j}", label=name)
        ax.plot([], [], "k-", lw=2.5, label="measured (L1, cooled)")
        ax.plot([], [], "o", color="0.5", mfc="white", label="open: mode split\n(top score < 0.6)")
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0))
        if i == 2:
            ax.set_ylim(*YL)
        ax.set_ylabel("frequency change [%]")
        ax.set_title(title, fontsize=9, loc="left")
    axs[-1].set_xticks(range(len(labs)))
    axs[-1].set_xticklabels([f"{MEAS[l][0]}\n{l}" for l in labs], fontsize=7, rotation=45)
    fig.tight_layout()
    fig.savefig(OUTD / "s3_l1_cause.png", dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy", action="store_true")
    ap.add_argument("--cases", default="")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "s3_l1_cause_healthy.pkl"))
    a = ap.parse_args()
    if a.healthy:
        healthy(a.cache)
    if a.cases:
        run_cases([c for c in a.cases.split(",") if c], a.cache)
    if a.summary:
        summarize()
