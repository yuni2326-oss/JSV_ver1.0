"""심각도·손상장 파라미터화 — 설계서 §4 정의 동결.

  손상장   d(r) = δD(r)/D = (S/(w√π))·exp(−((r−r_d)/w)²),   ∫ d dr = S  [m]
  심각도   S̄_D = (1/(b−a))·∫(δD/D) dr = S/(b−a)             [무차원]  ← 정본 §3.3
  위치     ξ_d = (r_d − a)/(b − a) ∈ [0,1]

정본 v1의 "적분 심각도 S[m]"는 차원이 있어 가공포켓·모델손상이 비교 불가였다(리뷰 지적).
S̄_D는 반경구간 평균 굽힘강성 손실률이므로 실제 재료제거와 통약된다. d(r) 형태 자체는
논문1 `impeller_pinn.inverse_damage._damage_field`와 동일하게 유지해 파일럿을 재현할 수 있게 한다
(테스트 T7).
"""
from __future__ import annotations

import math

import numpy as np


def S_from_s_bar(s_bar: float, extent: float) -> float:
    """S̄_D → 적분심각도 S [m]."""
    return float(s_bar) * float(extent)


def s_bar_from_S(S: float, extent: float) -> float:
    """적분심각도 S [m] → S̄_D [무차원]."""
    return float(S) / float(extent)


def xi_to_r(xi_d: float, a: float, b: float) -> float:
    """정규화 반경위치 ξ_d ∈ [0,1] → 물리 반경 r_d [m]."""
    return float(a) + float(xi_d) * (float(b) - float(a))


def r_to_xi(r_d: float, a: float, b: float) -> float:
    """물리 반경 r_d [m] → 정규화 위치 ξ_d."""
    return (float(r_d) - float(a)) / (float(b) - float(a))


def damage_field(r: np.ndarray, r_d: float, S: float, w: float) -> np.ndarray:
    """가우시안 강성손실장 d(r) = δD/D, ∫d dr = S (논문1과 동일 형태)."""
    r = np.asarray(r, dtype=float)
    return (S / (w * math.sqrt(math.pi))) * np.exp(-((r - r_d) / w) ** 2)


def damage_field_xi(r: np.ndarray, xi_d: float, s_bar: float, w: float,
                    a: float, b: float) -> np.ndarray:
    """(ξ_d, S̄_D) 파라미터화 손상장 — 내부적으로 (r_d, S) 형태와 동일."""
    return damage_field(r, r_d=xi_to_r(xi_d, a, b),
                        S=S_from_s_bar(s_bar, b - a), w=w)


def pocket_depth_to_severity(depth_frac: float, radial_width: float,
                             extent: float) -> float:
    """가공포켓의 기하 → S̄_D 환산(1차: 굽힘강성 ∝ t³).

    깊이비 `depth_frac` = 제거깊이/두께, 반경폭 `radial_width` [m]인 직사각 포켓이
    반경방향으로 균일하게 강성을 (1−(1−depth_frac)³)만큼 잃는다고 보면
        ∫(δD/D) dr = [1 − (1−depth_frac)³] · radial_width
    → S̄_D = 그 값/(b−a). 방위방향 국소성(각폭)은 여기 들어가지 않는다(pair mean은
    방위평균이므로 `degenerate` 모듈에서 각폭 계수를 따로 곱한다).
    """
    loss = 1.0 - (1.0 - float(depth_frac)) ** 3
    return loss * float(radial_width) / float(extent)
