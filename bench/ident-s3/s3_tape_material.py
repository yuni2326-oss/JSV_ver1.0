"""테이프 소재 점검(10-31 질문 '일반 접착테이프로 질량을 더해도 되나', 40번 §3.0) — 같은 질량을 면밀도 낮은 테이프 띠로
붙일 때 T-a·T-b·T-c 예측(질량점 1차, 강성은 넣지 않음). 모델·짝 성분 벡터·질량점 1차식은 s3_pred_windows.py와 같고,
발자국만 (반지름 반폭, 둘레 반길이)로 바꾼다. 납 테이프 = 40번 발자국(30 mg 4 × 4, 10 mg 2.5 × 2.5 mm),
그 밖 = 테두리 안쪽 4 mm 폭 띠, 길이 = 질량 ÷ (면밀도 × 4 mm). 면밀도[mg/mm²]: 전기(PVC) 0.22, 알루미늄 0.17, 투명(OPP) 0.05.
사용: PYTHONPATH=/home/super/project-paper3 python s3_tape_material.py > ../fe-nominal/s3_tape_material.txt   (약 6분)"""
import math
import time

import numpy as np

from impeller_fingerprint import geometry as geo
from impeller_fingerprint.rev1 import n1_mesh as nm
from impeller_fingerprint.rev1 import n6_freefree as n6

assert geo.ASBUILT.gap_in is None
THETA0 = {"2134": 117.2, "2142": 73.1}
t0 = time.time()
ff = n6.FFSpec(base=nm.FitSpec(n_r=32, n_bay=16, wrap_deg=geo.ASBUILT.wrap_deg), n_hub=6, n_z_collar=4,
               collar_h=geo.ASBUILT.collar_h, clamp_r=None)
fm = n6.build(ff)
km = n6.assemble(fm)
md, _ = n6.solve(km, 80, n_rigid=ff.n_rigid)
obs = n6.classify(fm, md)
print(f"건전 {md.freqs[0]:.1f}–{md.freqs[-1]:.1f} Hz ({time.time() - t0:.0f}s)", flush=True)
fs = ff.base
X = fm.coors
th_v = np.arctan2(X[:, 1], X[:, 0])
z_in, z_top = ff.z_front
top_v = np.isclose(X[:, 2], z_top, atol=1e-9) & (fm.r_v >= ff.a + ff.collar_t - 1e-9)
gi = fm.info["grid_idx"]
PER = fs.n_vc + fs.n_bay
loc, sec = gi[:, 1] % PER, gi[:, 1] // PER
bot = np.isclose(X[:, 2], 0.0, atol=1e-9)
cam = lambda r: float(fs.camber(np.array([r]))[0])
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
r_of = lambda xi: ff.a + xi * ff.extent
H2 = [o for o in obs if o.label == "h2:m2n0"][0]


def back_point(k, xi):
    r_t = r_of(xi)
    W = []
    for l_ in (0, 1):
        c = np.nonzero(bot & (loc == l_) & (sec == k))[0]
        c = c[np.argsort(fm.r_v[c])]
        i = int(np.searchsorted(fm.r_v[c], r_t))
        lo_, hi_ = c[i - 1], c[i]
        t = (r_t - fm.r_v[lo_]) / (fm.r_v[hi_] - fm.r_v[lo_])
        W += [(lo_, 0.5 * (1 - t)), (hi_, 0.5 * t)]
    return W


V2 = md.vshape[H2.members]
P6 = np.array([[sum(w * V2[m, 3 * n + 2] for n, w in back_point((-v) % 6, 0.6)) for m in range(2)] for v in range(6)])
TH6 = np.radians(60.0 * np.arange(6))
betas = np.radians(np.arange(0.0, 180.0, 0.25))
th0_of = np.array([(-np.degrees(np.angle(((P6 @ [math.cos(b), math.sin(b)]) * np.exp(-2j * TH6)).sum())) / 2) % 180
                   for b in betas])
E = {}
for name, t in THETA0.items():
    j = int(np.argmin(np.abs((4 * (th0_of - t) + 180) % 360 - 180)))
    E[name] = np.array([math.cos(betas[j]), math.sin(betas[j])])
idx_top = np.nonzero(top_v)[0]
r_rim = r_of(1.0)
print(f"테두리 r {1e3 * r_rim:.2f} mm, ξ0.8 r {1e3 * r_of(0.8):.2f} mm, ξ0.95 r {1e3 * r_of(0.95):.2f} mm", flush=True)


def strip_H(r0, th0, m, hw_r, hl_t):
    """발자국: |r − r0| ≤ hw_r(테두리에서 잘림), 둘레 |r0·Δθ| ≤ hl_t. 질량 m을 발자국 꼭짓점에 고르게(평균 |φ|²)."""
    d_r = fm.r_v[idx_top] - r0
    d_t = r0 * wrap(th_v[idx_top] - th0)
    sel = idx_top[(np.abs(d_r) <= hw_r) & (np.abs(d_t) <= hl_t)]
    H = {}
    for o in obs:
        U = md.vshape[o.members].reshape(o.deg, -1, 3)[:, sel, :]
        H[o.label] = -m * np.einsum("anc,bnc->ab", U, U) / sel.size
    return H, sel.size


dfp = lambda e: 100 * (math.sqrt(1 + e) - 1)
WANT = ["h2:m2n0", "h0:m0n1", "h3:m3n0", "h1:m1n1", "h2:m4n0"]
r_T = r_of(0.95)
th_sec = lambda k, rel, r: cam(r) + math.radians(60.0 * k + rel)
SPOTS = {"T-a": (5, 0.0, 30.0), "T-b": (4, 14.1, 30.0), "T-c": (3, 30.0, 10.0)}
# 면밀도[mg/mm²]: 납 테이프는 40번 발자국(30 mg 4×4, 10 mg 2.5×2.5), 그 밖은 띠(반지름 폭 4 mm, 테두리 쪽)
TAPES = {"lead": None, "PVC 0.22": 0.22, "Al 0.17": 0.17, "OPP 0.05": 0.05}
FOOT_LEAD = {30.0: 2.0e-3, 10.0: 1.25e-3}
print("상태 | 테이프 | 띠 길이[mm](둘레 각°) | 꼭짓점 | 2142 | 2134 | 그 밖 짝(평균 ± 반 갈라짐) / 홑")
for st, (k, rel, m_mg) in SPOTS.items():
    for tp, mu in TAPES.items():
        if mu is None:
            hw_r, hl_t, r0 = FOOT_LEAD[m_mg], FOOT_LEAD[m_mg], r_T
        else:
            r0 = r_rim - 2.0e-3                          # 4 mm 폭 띠: 테두리 안쪽 4 mm
            hw_r = 2.0e-3
            L = m_mg / (mu * 4.0) * 1e-3                 # 띠 길이[m]
            hl_t = 0.5 * L
        H, n = strip_H(r0, th_sec(k, rel, r0), m_mg * 1e-6, hw_r, hl_t)
        h = H["h2:m2n0"]
        a, b = dfp(float(E["2142"] @ h @ E["2142"])), dfp(float(E["2134"] @ h @ E["2134"]))
        oth = []
        for o in obs:
            if o.label not in WANT[1:]:
                continue
            hh = H[o.label]
            if o.deg == 1:
                oth.append(f"{o.label} {dfp(hh[0, 0]):+.3f}")
            else:
                w = np.linalg.eigvalsh(hh)
                oth.append(f"{o.label} {dfp(0.5 * np.trace(hh)):+.3f}±{25 * (w[1] - w[0]):.3f}")
        print(f"{st} | {tp} | {2e3 * hl_t:.1f} ({math.degrees(2 * hl_t / r0):.0f}°) | {n} | {a:+.3f} | {b:+.3f} | " + "; ".join(oth),
              flush=True)
print(f"끝 ({time.time() - t0:.0f}s)")
