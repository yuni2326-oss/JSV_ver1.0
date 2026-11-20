"""질량 띠 자리 각 훑기와 맞춤(46번, 11-12).

1) FEM 훑기: 6칸 띠 고리(20 mg × 6, 깃 선과 나란한 띠 — s3_pred_windows.py의 strip()과 같은 모델)를 FEM 각 θ(섹터 기준 0–60°,
   2° 간격)·중심 r(31.8–35.3 mm)마다 1차 질량 섭동으로 계산한다. 6회 대칭 배치라 짝은 갈리지 않으므로 고윳값 평균을 쓴다.
   → fe-nominal/s3_m_angle_scan.csv (약 4분, 있으면 다시 쓰지 않는다; --rerun으로 다시)
2) 맞춤: 측정 M120(A1-6v 대비, 측정 20° 자리 = FEM 40°)과 M240 증분(M120 대비, 측정 40° = FEM 20°)을
   (a) 공칭 자리, (b) 단계별 각 어긋남 δ1·δ2, (c) 공통 δ 하나로 맞춘다. 단계마다 공통 오프셋 c를 함께 맞춘다.
   σ = √(0.07² + (0.1·|측정|)²) %. 측정값은 groove_compare 출력(M_report.csv의 M120, M240_inc.csv).
   → groove-test/out/m_angle_fit.json, 화면 표
사용: PYTHONPATH=/home/super/project-paper3 python s3_m_angle_scan.py [--rerun]   (PAPER3_GEOM을 비울 것)"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "..", "fe-nominal", "s3_m_angle_scan.csv")
OUT = os.path.join(HERE, "..", "groove-test", "out")
LAB = {"h2": "h2:m2n0", "3164": "h3:m3n0", "3250": "h0:m0n1", "3459": "h1:m1n1", "3639": "h2:m4n0", "3748": "h3:m3n1",
       "3804": "h0:m6n1", "8321": "h3:m3n1#2", "5745": "h1:m5n0", "5617": "h0:m6n0", "7093": "h3:m3n2", "7574": "h2:m8n0",
       "6439": "h2:m2n1", "4967": "h2:m4n0#2"}
KEYS = ["h2", "3164", "3250", "3459", "3639", "3748", "3804", "8321", "5745", "7093", "7574"]
ANG = np.arange(0.0, 60.01, 2.0)
RC = (31.8, 32.3, 32.8, 33.3, 33.8, 34.3, 34.8, 35.3)


def fem_scan():
    from impeller_fingerprint import geometry as geo
    from impeller_fingerprint.rev1 import n1_mesh as nm
    from impeller_fingerprint.rev1 import n6_freefree as n6
    assert geo.ASBUILT.gap_in is None, "G3(간격 일정)로 돌린다 — PAPER3_GEOM을 비울 것"
    t0 = time.time()
    ff = n6.FFSpec(base=nm.FitSpec(n_r=32, n_bay=16, wrap_deg=geo.ASBUILT.wrap_deg), n_hub=6, n_z_collar=4,
                   collar_h=geo.ASBUILT.collar_h, clamp_r=None)
    fm = n6.build(ff)
    md, _ = n6.solve(n6.assemble(fm), 80, n_rigid=ff.n_rigid)
    obs = {o.label: o for o in n6.classify(fm, md)}
    print(f"건전 {md.freqs[0]:.1f}–{md.freqs[-1]:.1f} Hz ({time.time() - t0:.0f}s)", flush=True)
    fs, X = ff.base, fm.coors
    th_v = np.arctan2(X[:, 1], X[:, 0])
    top = np.nonzero(np.isclose(X[:, 2], ff.z_front[1], atol=1e-9) & (fm.r_v >= ff.a + ff.collar_t - 1e-9))[0]
    cam = lambda r: float(fs.camber(np.array([r]))[0])
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi

    def strip(k, rel, r_c, m=20e-6, length=6.4e-3, n=4):
        g = r_c * (cam(r_c + 1e-4) - cam(r_c - 1e-4)) / 2e-4
        cpsi = 1.0 / math.sqrt(1.0 + g * g)
        s = (np.arange(n) - (n - 1) / 2) * length / n
        return [(r_c + si * cpsi, cam(r_c + si * cpsi) + math.radians(60.0 * k + rel), m / n) for si in s]

    def eta(points, hw=1.0e-3):
        H = {lab: np.zeros((obs[lab].deg, obs[lab].deg)) for lab in LAB.values()}
        for r0, th0, m in points:
            d_r = fm.r_v[top] - r0
            d_t = r0 * wrap(th_v[top] - th0)
            sel = top[(np.abs(d_r) <= hw) & (np.abs(d_t) <= hw)]
            if sel.size == 0:
                sel = top[[int(np.argmin(d_r ** 2 + d_t ** 2))]]
            for lab in H:
                U = md.vshape[obs[lab].members].reshape(obs[lab].deg, -1, 3)[:, sel, :]
                H[lab] += -m * np.einsum("anc,bnc->ab", U, U) / sel.size
        return {k: float(np.mean(np.linalg.eigvalsh(H[lab]))) for k, lab in LAB.items()}

    rows = []
    for rc in RC:
        for a in ANG:
            rows.append(dict(rc_mm=rc, theta_deg=a, **eta([p for k in range(6) for p in strip(k, a, rc * 1e-3)])))
    pd.DataFrame(rows).to_csv(SCAN, index=False)
    print(f"훑기 끝 → {SCAN} ({time.time() - t0:.0f}s)", flush=True)


if "--rerun" in sys.argv or not os.path.exists(SCAN):
    fem_scan()
S = pd.read_csv(SCAN)
dfp = lambda e: 100 * (math.sqrt(1 + e) - 1)


def interp(rc, a):
    g = S[np.isclose(S.rc_mm, rc)].sort_values("theta_deg")
    return {k: float(np.interp(a % 60.0, g.theta_deg, g[k])) for k in LAB}


def measured():
    r1 = pd.read_csv(os.path.join(OUT, "M_report.csv")).assign(mode=lambda d: d["mode"].astype(str))
    r1 = r1[r1.state == "M120"].set_index("mode").d_pct
    r2 = pd.read_csv(os.path.join(OUT, "M240_inc.csv")).assign(mode=lambda d: d["mode"].astype(str)).set_index("mode").d_pct
    f = lambda r, k: float(0.5 * (r["2134"] + r["2142"])) if k == "h2" else float(r[k])
    return {k: f(r1, k) for k in KEYS}, {k: f(r2, k) for k in KEYS}


M120, INC = measured()


def preds(rc, d1, d2):
    e40, e20 = interp(rc, 40.0 + d1), interp(rc, 20.0 + d2)
    p1 = {k: dfp(e40[k]) for k in KEYS}
    pi = {k: 100 * (math.sqrt(1 + e40[k] + e20[k]) / math.sqrt(1 + e40[k]) - 1) for k in KEYS}
    return p1, pi


def chi(M, P):
    y = np.array([M[k] for k in KEYS]); p = np.array([P[k] for k in KEYS])
    s = np.sqrt(0.07 ** 2 + (0.1 * np.abs(y)) ** 2); w = 1 / s ** 2
    c = float((w * (y - p)).sum() / w.sum())
    return float((w * (y - p - c) ** 2).sum()), c


def total(rc, d1, d2):
    p1, pi = preds(rc, d1, d2)
    (c1, o1), (c2, o2) = chi(M120, p1), chi(INC, pi)
    return c1, c2, o1, o2


D = np.arange(-15.0, 15.01, 0.5)
nom = total(34.3, 0.0, 0.0)
b1 = min(((total(rc, d, 0.0)[0], rc, d) for rc in RC for d in D))                       # 1단계만(δ1)
b2 = min(((total(rc, b1[2], d)[1], rc, d) for rc in RC for d in D if np.isclose(rc, b1[1])))  # 같은 r에서 2단계(δ2)
bc = min(((sum(total(rc, d, d)[:2]), rc, d) for rc in RC for d in D))                    # 공통 δ
fixed_r = {rc: min(((total(rc, d, 0.0)[0], d) for d in D)) for rc in RC}
res = dict(nominal=dict(rc=34.3, d1=0.0, d2=0.0, chi_M120=nom[0], chi_inc=nom[1], c_M120=nom[2], c_inc=nom[3]),
           per_step=dict(rc=b1[1], d1=b1[2], d2=b2[2], chi_M120=b1[0], chi_inc=b2[0]),
           common=dict(rc=bc[1], d=bc[2], chi_total=bc[0]),
           M120_best_d1_by_rc={f"{rc:.1f}": dict(d1=d, chi=c) for rc, (c, d) in fixed_r.items()},
           n_modes=len(KEYS), keys=KEYS, M120=M120, INC=INC,
           note="δ는 FEM 각에 더한다(측정 각으로는 빼기: 측정 각 = 60 − FEM 각). σ = √(0.07² + (0.1·|측정|)²) %.")
for lab, (rc, d1, d2) in (("nominal", (34.3, 0.0, 0.0)), ("per_step", (b1[1], b1[2], b2[2]))):
    p1, pi = preds(rc, d1, d2)
    res[lab]["pred_M120"], res[lab]["pred_inc"] = p1, pi
json.dump(res, open(os.path.join(OUT, "m_angle_fit.json"), "w"), ensure_ascii=False, indent=1)

print(f"모드 {len(KEYS)}개: {', '.join(KEYS)}")
print(f"(a) 공칭(r 34.3, δ 0): χ² M120 {nom[0]:.1f}, 증분 {nom[1]:.1f}")
print(f"(b) 단계별: M120 δ1 {b1[2]:+.1f}°(r {b1[1]:.1f}) χ² {b1[0]:.1f}; 증분 δ2 {b2[2]:+.1f}° χ² {b2[0]:.1f}")
print("    r별 M120 최선 δ1: " + ", ".join(f"r {rc:.1f} → {d:+.1f}°(χ² {c:.1f})" for rc, (c, d) in fixed_r.items()))
print(f"(c) 공통 δ: r {bc[1]:.1f}, δ {bc[2]:+.1f}°, χ² 합 {bc[0]:.1f}")
for lab in ("nominal", "per_step"):
    r = res[lab]
    print(f"\n[{lab}] r {r['rc']:.1f}, δ1 {r['d1']:+.1f}°, δ2 {r['d2']:+.1f}°")
    print("모드 | M120 측정 / FEM | M240 증분 측정 / FEM")
    for k in KEYS:
        print(f"{k:>5} | {M120[k]:+.2f} / {r['pred_M120'][k]:+.2f} | {INC[k]:+.2f} / {r['pred_inc'][k]:+.2f}")
print(f"\n→ {os.path.join(OUT, 'm_angle_fit.json')}")
