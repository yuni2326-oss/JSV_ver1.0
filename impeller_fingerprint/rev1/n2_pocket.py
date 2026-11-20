"""N2 — C6 조립체의 국부 포켓: 쌍평균 단독 대 + 분리, 손상 방위 θ_d 의존성 (R1-P3, R1-P2).

심사자 요구(R1-P3): 분리를 생산 역문제에 넣지 않은 이유, 방위를 모를 때 분리 크기의 득실,
조립체 수준 검증. 이 모듈은 N1과 같은 정확 C6 조립체에서 **같은 손상·같은 잡음 규약**으로
세 arm을 비교한다(받은 계획서 §6 N2):

  A  쌍평균(+singlet)만, θ_d 기지
  B  A + doublet 분리 크기 |Δη|, θ_d 기지
  C  B, θ_d를 nuisance로(Schur 보수)            — "방위를 모를 때"
  (A′ A에서 θ_d nuisance — C6 조립체에서는 쌍평균도 θ_d에 60° 주기로 의존할 수 있다)

손상: 전면 슈라우드 바깥면의 매끄러운 국부 감육
    p(r, θ) = p₀ · exp(−((r − r_d)/w_r)²) · W(θ − θ_d; Δθ),
W = 전폭 Δθ의 raised-cosine 창(콤팩트, C¹). 축대칭 레일의 포켓(Table 3b)과 같은 족이 아니다 —
3D 형상 감육이고, 결합법칙을 가정하지 않는다.

1차 관측량: 건전 doublet 기저 Φ로 H = Φᵀ(δK − λδM)Φ/λ (2×2), 쌍평균 = tr H/2,
분리 = λ₊(H) − λ₋(H) = √((H₁₁−H₂₂)² + 4H₁₂²). singlet은 1×1. 건전 분리가 정확히 0인 격자라
분리 증분 = 손상 분리다(사면체 CAD의 0.04–0.96 % 바닥이 없다). 잡음: 쌍평균 2c, 분리 4c,
singlet 2√2c(정본 §3.5, 독립 raw 오차).
"""
from __future__ import annotations

import math

import numpy as np

from . import n1_assembly as n1


def raised_cosine(dth: np.ndarray, width: float) -> np.ndarray:
    """전폭 `width`(rad)의 raised-cosine 창, 중심 0, 주기 2π로 감싼다."""
    d = np.angle(np.exp(1j * dth))
    out = np.zeros_like(d)
    inside = np.abs(d) < 0.5 * width
    out[inside] = 0.5 * (1.0 + np.cos(2 * math.pi * d[inside] / width))
    return out


def pocket_depth(am: n1.AsmMesh, xi_d: float, p0: float, w_r: float, theta_d: float,
                 dtheta: float) -> np.ndarray:
    spec = am.spec
    r_d = spec.a + xi_d * (spec.b - spec.a)
    th = np.arctan2(am.coors[:, 1], am.coors[:, 0])
    return p0 * np.exp(-((am.r_v - r_d) / w_r) ** 2) * raised_cosine(th - theta_d, dtheta)


def pair_matrices(md: n1.Modes, obs, dK, dM) -> list[np.ndarray]:
    out = []
    for o in obs:
        Phi = md.phi[:, o.members]
        lam = float(np.mean(md.lam[o.members]))
        H = (Phi.T @ (dK @ Phi) - lam * (Phi.T @ (dM @ Phi))) / lam
        out.append(0.5 * (H + H.T))
    return out


def observables_from_H(Hs, obs) -> tuple[np.ndarray, np.ndarray]:
    """(쌍평균/singlet η[n_obs], 분리 Δη[n_obs], singlet은 NaN)."""
    mean = np.empty(len(obs))
    split = np.full(len(obs), np.nan)
    for j, (H, o) in enumerate(zip(Hs, obs)):
        mean[j] = float(np.trace(H)) / H.shape[0]
        if H.shape[0] == 2:
            split[j] = float(math.sqrt((H[0, 0] - H[1, 1]) ** 2 + 4 * H[0, 1] ** 2))
    return mean, split


def first_order(am, md, obs, km0, xi_d, p0, w_r, theta_d, dtheta, p_probe=None):
    """1차 관측량(행렬 중심차분 ±p). p0에 대해 선형이므로 p_probe로 계산 후 p0로 스케일."""
    pp = p_probe or 1e-3
    p = pocket_depth(am, xi_d, pp, w_r, theta_d, dtheta)
    kp = n1.assemble(am, n1.morph(am, p))
    km_ = n1.assemble(am, n1.morph(am, -p))
    dK = (kp.K - km_.K) * 0.5
    dM = (kp.M - km_.M) * 0.5
    mean, split = observables_from_H(pair_matrices(md, obs, dK, dM), obs)
    return mean * (p0 / pp), split * (p0 / pp)


def resolved(am, md_h, obs, xi_d, p0, w_r, theta_d, dtheta, n_extra=6, min_mac=0.8):
    """유한 깊이 재해석: 손상 모드를 건전 관측량에 부분공간 MAC로 짝짓고 쌍평균·분리를 낸다.

    축대칭이 깨지므로 doublet이 갈라진다 → 후보 블록은 연속 두 모드(주파수 정렬)이며 축퇴를
    요구하지 않는다. 짝의 분리가 이웃 관측량 간격보다 크면 짝짓기가 모호해지므로 MAC를 함께 낸다.
    """
    p = pocket_depth(am, xi_d, p0, w_r, theta_d, dtheta)
    md = n1.solve(n1.assemble(am, n1.morph(am, p)), md_h.phi.shape[1] + n_extra)
    return track_split(md_h, md, obs, min_mac)


def track_split(md_h: n1.Modes, md: n1.Modes, obs, min_mac: float = 0.8):
    """대칭을 깨는 손상의 재해석 모드를 건전 관측량에 짝짓는다 — 연속 블록, 축퇴 요구 없음, 부분공간 MAC.

    반환: (쌍평균/singlet η, doublet 분리 |Δλ|/λ, MAC) — 못 찾으면 NaN.
    """
    Vh, Vd = md_h.vshape, md.vshape
    used = set()
    mean = np.full(len(obs), np.nan)
    split = np.full(len(obs), np.nan)
    mac = np.full(len(obs), np.nan)
    for j, o in enumerate(obs):
        Qa, _ = np.linalg.qr(Vh[o.members].T)
        best = (-1.0, None)
        for s in range(Vd.shape[0] - o.deg + 1):
            cand = list(range(s, s + o.deg))
            if any(c in used for c in cand):
                continue
            Qb, _ = np.linalg.qr(Vd[cand].T)
            sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
            sc = float(np.min(sv) ** 2)
            if sc > best[0]:
                best = (sc, cand)
        if best[1] is None or best[0] < min_mac:
            continue
        used.update(best[1])
        lam0 = float(np.mean(md_h.lam[o.members]))
        ld = md.lam[best[1]]
        mean[j] = float(np.mean(ld) / lam0 - 1.0)
        if o.deg == 2:
            split[j] = float(abs(ld[1] - ld[0]) / lam0)
        mac[j] = best[0]
    return mean, split, mac


def window_first_order(md: n1.Modes, obs, kmr: n1.KM, dK_frac: float,
                       dM_frac: float = 0.0):
    """구역(셀 마스크) 강성·질량 동률 손실의 1차 쌍행렬 → (쌍평균/singlet, 분리).

    δK = −dK·K_R, δM = −dM·M_R ⇒ H = Φᵀ(δK − λδM)Φ/λ. 베인 1매 뿌리창처럼 한 섹터에 갇힌
    손상이면 H가 거의 계수 1이 되어 **한 성분만** 움직인다 — 그때 분리 = 2|쌍평균|이고,
    단일부호 δK(−K_R ⪯ 0)에서는 언제나 분리 ≤ 2|쌍평균|(두 고유치가 같은 부호)이다.
    """
    mean = np.empty(len(obs))
    split = np.full(len(obs), np.nan)
    for j, o in enumerate(obs):
        Phi = md.phi[:, o.members]
        lam = float(np.mean(md.lam[o.members]))
        H = (-dK_frac * (Phi.T @ (kmr.K @ Phi)) + lam * dM_frac * (Phi.T @ (kmr.M @ Phi))) / lam
        H = 0.5 * (H + H.T)
        mean[j] = float(np.trace(H)) / H.shape[0]
        if H.shape[0] == 2:
            split[j] = float(math.sqrt((H[0, 0] - H[1, 1]) ** 2 + 4 * H[0, 1] ** 2))
    return mean, split
