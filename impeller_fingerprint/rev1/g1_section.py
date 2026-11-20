"""G1 — 2026-10-16 재측정 기하: 비대칭 두 면판 단면식과 측정 기록.

측정(사용자, 디지털 캘리퍼·분석저울 A&D HR-250AZ, 2026-10-16, 사진 7장):
  앞판(흡입구 쪽 = 손상면) t_front 0.37–0.38 mm, 뒤판(허브 쪽) t_back 0.58–0.60 mm,
  판 사이 간격(출구 끝) 3.02 mm, 전체 두께(출구 근처) 4.15–4.25 mm, 깃 두께 약 1.2 mm,
  흡입구 안지름 29.91 mm · 흡입구 테 바깥지름 30.76 mm(테 두께 ≈ 0.43 mm → 앞판이 얇은 판),
  보어(별 모양) 12.67 mm, 질량 40.2059 g.
코드의 08-13 기하(양면 1.0 mm, 유로 4.1 mm, 전체 6.1 mm, 깃 1.0 mm, 모델 질량 68.4 g)를 대체할
후보다. 합 t_front + t_back + 간격 = 3.985 mm가 전체 4.15–4.25 mm보다 0.17–0.27 mm 작다 —
간격이 반경에 따라 다를 수 있다(확인 대기).

단면식은 `forward.sandwich_I`(같은 두께 두 면판)의 일반화다. 손상면(두께 t_d)을 **바깥면에서**
깊이비 p만큼 깎고(t_d → t_d(1 − p)), 안쪽면 간격 gap은 고정한다. 단위폭 단면 2차모멘트를
합성 도심 기준으로

    I(p) = t_d'³/12 + t_o³/12 + μ d²,   t_d' = t_d(1 − p),  μ = t_d' t_o/(t_d' + t_o),
    d = gap + (t_d' + t_o)/2  (두 면 중심 사이 거리)

로 둔다. t_d = t_o = t_f, gap = s − t_f이면 `forward.sandwich_I(p, t_f, s)`와 같다(테스트).
"""
from __future__ import annotations

import numpy as np

from .. import forward as fwd

#: 2026-10-16 측정 대표값 [m] — 범위의 중앙. 정본 상수는 `geometry`(GAP = 3.18 mm 평균)에 있다.
T_FRONT = 0.375e-3
T_BACK = 0.59e-3
GAP = 3.02e-3              # 출구 끝 안측정 조 직접값(전체 두께로는 3.10–3.27 mm)
T_VANE = 1.2e-3
EYE_ID = 29.91e-3
EYE_OD = 30.76e-3
BORE = 12.67e-3
MASS_G = 40.2059


def two_face_I(p, t_d: float, t_o: float, gap: float):
    """단위폭 단면 2차모멘트 [m³] — 정본 `forward.two_face_I`와 같다(측정 기록 모듈의 하위호환 이름)."""
    return fwd.two_face_I(p, t_d, t_o, gap)


def dk_from_depth(p, t_d: float, t_o: float, gap: float):
    """깊이비 p → 단면 강성손실 d_K = 1 − I(p)/I(0) (`forward.two_face_dk_from_depth`)."""
    return fwd.two_face_dk_from_depth(p, t_d, t_o, gap)


def depth_from_dk(d_K, t_d: float, t_o: float, gap: float, n_table: int = 200001):
    """d_K → 깊이비 p (`forward.two_face_depth_from_dk`)."""
    return fwd.two_face_depth_from_dk(d_K, t_d, t_o, gap, n_table)


def mass_loss(p, t_d: float, t_o: float):
    """면적질량 손실비 d_M = p·t_d/(t_d + t_o)."""
    return np.asarray(p, dtype=float) * t_d / (t_d + t_o)


def zeta0(t_d: float, t_o: float, gap: float, p: float = 1e-6) -> float:
    """작은 깊이 극한의 질량/강성 손실비 ζ = d_M/d_K (균일판 1/3, 08-13 기하 0.71)."""
    return float(mass_loss(p, t_d, t_o) / dk_from_depth(p, t_d, t_o, gap))
