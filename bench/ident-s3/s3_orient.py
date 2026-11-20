"""h2 짝(2134/2142 Hz = G3 h2:m2n0) 방향 옮기기(42번) — |배| 자리는 90° 주기(부호 무관) — 측정 θ0(뒤판 깃 선 ξ0.6, 깃 번호 기준) → 앞판 바깥면의 배 자리.

측정 θ0의 정의(`ident-w22/w22joint.py` pattern): 깃 v(1–6)의 잔차 a_v ∝ cos 2(θ_v − θ0), θ_v = 60°(v − 1).
측정 깃 번호는 FEM −θ 방향이다(28번 §4.1, 거울상): 측정 깃 v ↔ FEM 섹터 (−(v − 1)) mod 6.
FEM 점 = 뒤판 바깥면(z = 0) 깃 선(격자 loc 0·1 평균) ξ0.6의 u_z(`ident-s2/s2b_fe.py`와 같다).

짝 부분공간의 모든 조합 e(β) = cos β·φ_a + sin β·φ_b에 대해 (i) 측정 규약의 θ0, (ii) 앞판 윗면 u_z의 cos 2(θ − φ0) 적합 배 자리를
구해 θ0 → 앞판 배 자리 대응을 만든다. 앞판 자리는 그 반지름의 깃 선(캠버) 기준 각과, 측정 규약(깃 1 ξ0.6 타격점 기준, 깃 번호 방향)으로 낸다.
사용: python s3_orient.py  (저장된 G3 모드형 rev1_n6_shapes_ff32x16h6w62c50.npz를 읽는다; 1분)"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FED = Path("/home/super/project-paper3/docs/_generated/data/paper3")
KEY = "ff32x16h6w62c50"
OUT = Path(__file__).resolve().parents[1] / "fe-nominal" / "s3_local"
z = np.load(FED / f"rev1_n6_shapes_{KEY}.npz")
reg = pd.read_csv(FED / f"rev1_n6_modes_{KEY}.csv")
meta = json.load(open(FED / f"rev1_n6_modes_{KEY}.json"))
coors, gi, vsh = z["coors"], z["grid_idx"], z["vshape"].astype(float)
a, b = meta["a"], meta["b"]
PER = 1 + 16
rr = np.hypot(coors[:, 0], coors[:, 1])
th = np.degrees(np.arctan2(coors[:, 1], coors[:, 0]))
bot = np.isclose(coors[:, 2], 0.0, atol=1e-9)
ztop = coors[:, 2].max()
loc = gi[:, 1] % PER
sec = gi[:, 1] // PER
z_front_top = meta["t_back"] + meta["channel"] + meta["t_front"]
top = np.isclose(coors[:, 2], z_front_top, atol=1e-9)


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def back_point(k, xi):
    """뒤판 바깥면 깃 선(섹터 k) ξ → (가중 절점, FEM 각[°]) — s2b_fe.point와 같다."""
    r_t = a + xi * (b - a)
    W, ang = [], []
    for l_ in (0, 1):
        c = np.nonzero(bot & (loc == l_) & (sec == k))[0]
        c = c[np.argsort(rr[c])]
        i = np.searchsorted(rr[c], r_t)
        lo_, hi_ = c[i - 1], c[i]
        t = (r_t - rr[lo_]) / (rr[hi_] - rr[lo_])
        W += [(lo_, 0.5 * (1 - t)), (hi_, 0.5 * t)]
        ang.append((1 - t) * th[lo_] + t * th[hi_])
    return W, float(np.mean(ang))


def vane_angle(xi_or_r, k=0, is_r=False):
    """FEM 깃 k의 선(격자 loc 0·1 평균) 각 — 앞판 윗면 절점에서 가장 가까운 반지름 링."""
    r_t = xi_or_r if is_r else a + xi_or_r * (b - a)
    c = np.nonzero(top & (sec == k) & ((loc == 0) | (loc == 1)))[0]
    ring = np.abs(rr[c] - r_t) <= np.min(np.abs(rr[c] - r_t)) + 1e-9
    return float(np.degrees(np.angle(np.exp(1j * np.radians(th[c[ring]])).mean())))


def front_ring(xi):
    """앞판 윗면에서 ξ에 가장 가까운 링의 절점(각 순)."""
    r_t = a + xi * (b - a)
    c = np.nonzero(top)[0]
    d = np.abs(rr[c] - r_t)
    c = c[d <= d.min() + 1e-9]
    return c[np.argsort(th[c])], float(rr[c][0])


lab = "h2:m2n0"
mem = [int(i) for i in str(reg.set_index("label").loc[lab, "members"]).split("|")]
uz = lambda n: vsh[mem][:, 3 * n + 2]                        # (2,) 두 멤버의 u_z

# 측정 규약 6점(ξ0.6): 측정 깃 v → FEM 섹터 (−(v−1)) mod 6
pts = [back_point((-(v)) % 6, 0.6) for v in range(6)]
P6 = np.array([[sum(w * vsh[m, 3 * n + 2] for n, w in W) for m in mem] for W, _ in pts])   # (6, 2)
ang_hit0 = pts[0][1]                                           # 측정 깃 1 ξ0.6 타격점의 FEM 각
TH6 = np.radians(60.0 * np.arange(6))

rows = []
rings = {xi: front_ring(xi) for xi in (0.3, 0.6, 0.9)}
for beta in np.radians(np.arange(0.0, 180.0, 0.5)):
    e = np.array([math.cos(beta), math.sin(beta)])
    av = P6 @ e
    A2 = (av * np.exp(-2j * TH6)).sum()
    th0 = (-np.degrees(np.angle(A2)) / 2.0) % 180.0           # 측정 규약 θ0
    r = {"beta_deg": math.degrees(beta), "theta0_meas": th0, "h2_share6": float(abs(A2) ** 2 / (3 * (av @ av)) if av @ av > 0 else np.nan)}
    for xi, (c, r_m) in rings.items():
        u = vsh[mem][:, 3 * c + 2].T @ e                        # 링 위 u_z
        t = np.radians(th[c])
        C2 = (u * np.exp(-2j * t)).sum()
        phi0 = (-np.degrees(np.angle(C2)) / 2.0) % 180.0        # FEM 각, u ∝ cos 2(θ − φ0)
        frac2 = float(2 * abs(C2) ** 2 / (len(u) * (u @ u)))    # h2(m=2) 성분 비
        # 측정 규약 각 = −(FEM 각 − 깃 1 타격점 FEM 각) (거울)
        r[f"phi0_fem_xi{xi}"] = phi0
        r[f"phi0_meas_xi{xi}"] = (-(phi0 - ang_hit0)) % 180.0
        r[f"m2share_xi{xi}"] = frac2
    rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(OUT / "orient_h2.csv", index=False)

print(f"측정 깃 1 ξ0.6 타격점 FEM 각 {ang_hit0:.1f}°; FEM 깃 0 선: ξ0.3 {vane_angle(0.3):.1f}°, ξ0.6 {vane_angle(0.6):.1f}°, "
      f"ξ0.9 {vane_angle(0.9):.1f}°, 테두리 {vane_angle(b, is_r=True):.1f}°")
for xi, (c, r_m) in rings.items():
    print(f"앞판 링 ξ{xi}: r {r_m * 1e3:.2f} mm, 절점 {len(c)}")
# 측정 두 성분(10-30 ξ0.6): 2133.8 Hz θ0 117.2°, 2142.3 Hz θ0 73.1°
for name, t0 in (("2134 (아래)", 117.2), ("2142 (위)", 73.1)):
    # 측정 θ0는 가장 큰 잔차를 +로 둔 부호 규약이라 성분의 부호가 바뀌면 90° 옮겨진다 → |배| 자리는 90° 주기로 맞춘다
    j = int(np.argmin(np.abs(wrap180(4 * (D.theta0_meas - t0)))))
    r = D.iloc[j]
    msg = [f"{name}: θ0 {t0}° → β {r.beta_deg:.1f}°"]
    for xi in (0.3, 0.6, 0.9):
        pm = r[f"phi0_meas_xi{xi}"]
        # 측정 규약에서 그 반지름 깃 선 각: 깃 v 선 = 60(v−1) + (깃 1 선 ξ의 측정 규약 각)
        v1 = (-(vane_angle(xi) - ang_hit0)) % 360.0
        rel = (pm - v1) % 60.0
        msg.append(f"ξ{xi}: 배 {pm:.1f}°/{(pm + 90) % 180:.1f}°(+180°) 측정 규약, 깃 1 선 {v1:.1f}° → 배의 깃 기준 위치(깃 번호 방향) "
                   f"{rel:.1f}°·{(rel + 30) % 60:.1f}° = FEM θ_rel {(-rel) % 60:.1f}°·{(-rel - 30) % 60:.1f}° (m2 비 {r[f'm2share_xi{xi}']:.2f})")
    print("\n  ".join(msg))
