"""e_pert — 1차섭동 유효성 맵 (정본 §3.4: 모든 역식별 사용에 *선행*하는 관문).

    e_pert(ξ_d, S̄_D; m) = |η̄^exact − η̄^lin| / |η̄^exact|

`η̄^exact`는 손상 D(r)로 다시 푼 Rayleigh 고유값(비섭동), `η̄^lin`은 커널 섭동식.
정본의 요구는 두 가지다: (1) e_pert가 측정 floor를 넘는 등고선을 공개하고, (2) 모든 역식별
결과에 이 맵 위 위치를 병기한다. 심각도가 선형영역을 넘으면 추정기의 순방향을 정확재해로
교체한다(`estimator`의 `exact=True`).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import forward as fwd
from . import kernels as ker


def _exact_law(mass):
    """정확재해에 넘길 질량법칙 — 샌드위치 법칙만 별도, 나머지 참값은 균일판 정확결합.

    기존 산출물(정확결합)과 비트단위 동일성을 유지하려고 `bool(mass)` 규약을 그대로 두고
    샌드위치 계열(`"sandwich"`, `"sandwich_asbuilt"`)만 통과시킨다(설계서 §5.3·F60).
    """
    return mass if mass in ("sandwich", "sandwich_asbuilt") else bool(mass)


def e_pert(plate, pool: Sequence[ker.ModeKernel], modes: Sequence[tuple[int, int]],
           xi_d: float, s_bar: float, w: float, n_trial: int = 8,
           n_grid: int = 4001, mass=None) -> np.ndarray:
    """모드별 상대 섭동오차. s_bar=0에서는 0으로 정의."""
    if s_bar == 0.0:
        return np.zeros(len(modes))
    lin = (fwd.eta_bar_linear_mass(pool, xi_d, s_bar, w, plate,
                                   coupling=fwd.resolve_coupling(mass))
           if mass else fwd.eta_bar_linear(pool, xi_d, s_bar, w, plate))
    ex = fwd.eta_bar_exact(plate, modes, xi_d, s_bar, w, n_trial=n_trial,
                           n_grid=n_grid, mass=_exact_law(mass))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(lin - ex) / np.abs(ex)
    return np.where(np.isfinite(out), out, 0.0)


def e_pert_abs(plate, pool: Sequence[ker.ModeKernel], modes: Sequence[tuple[int, int]],
               xi_d: float, s_bar: float, w: float, n_trial: int = 8,
               n_grid: int = 4001, mass=None) -> np.ndarray:
    """절대 섭동오차 |η̄^exact − η̄^lin| — 측정 floor와 직접 비교되는 양."""
    if s_bar == 0.0:
        return np.zeros(len(modes))
    lin = (fwd.eta_bar_linear_mass(pool, xi_d, s_bar, w, plate,
                                   coupling=fwd.resolve_coupling(mass))
           if mass else fwd.eta_bar_linear(pool, xi_d, s_bar, w, plate))
    ex = fwd.eta_bar_exact(plate, modes, xi_d, s_bar, w, n_trial=n_trial,
                           n_grid=n_grid, mass=_exact_law(mass))
    return np.abs(lin - ex)


def e_pert_map(plate, pool: Sequence[ker.ModeKernel], modes: Sequence[tuple[int, int]],
               xi_grid: np.ndarray, s_grid: np.ndarray, w: float,
               n_trial: int = 8, n_grid: int = 4001,
               absolute: bool = False, mass=None) -> np.ndarray:
    """(n_modes, n_xi, n_s) 유효성 맵. absolute=True면 절대오차."""
    fn = e_pert_abs if absolute else e_pert
    out = np.empty((len(modes), len(xi_grid), len(s_grid)))
    for i, xi in enumerate(np.asarray(xi_grid, dtype=float)):
        for j, s in enumerate(np.asarray(s_grid, dtype=float)):
            out[:, i, j] = fn(plate, pool, modes, float(xi), float(s), w,
                              n_trial=n_trial, n_grid=n_grid, mass=mass)
    return out


def fraction_below_floor(plate, pool: Sequence[ker.ModeKernel],
                         modes: Sequence[tuple[int, int]], xi_grid: np.ndarray,
                         s_grid: np.ndarray, w: float, sigma_rel: float,
                         n_trial: int = 8, n_grid: int = 4001, mass=None,
                         abs_map: np.ndarray | None = None) -> float:
    """섭동오차가 측정 floor 아래인 격자점 비율(모드 최대오차 기준).

    관측 노이즈는 절대 주파수 반복도 σ_f = sigma_rel·f (설계서 §4-5)이고 η̄ 단위로는
    σ_η = 2·sigma_rel. 이 값보다 섭동오차가 작으면 선형화 오차는 관측 불가 → 선형맵 사용 가능.

    `mass`는 `e_pert_map`과 같은 규약(설계서 M7) — 빠뜨리면 강성전용 맵이 나온다.
    `abs_map`을 주면 그 절대오차 맵을 재사용한다(같은 격자·같은 `mass`로 만든 것이어야
    한다 — 호출자가 이미 계산했을 때 6번 중복계산을 피하려는 용도).
    """
    M = (np.asarray(abs_map) if abs_map is not None else
         e_pert_map(plate, pool, modes, xi_grid, s_grid, w,
                    n_trial=n_trial, n_grid=n_grid, absolute=True, mass=mass))
    # floor는 **모드별**이다: m = 0은 축퇴쌍이 없어 η̄₀ = 2Δf/f가 두 추정만으로 만들어지므로
    # σ_η = 2√2·c이고, m > 0의 pair mean은 2c다(설계서 F112). 전 모드에 2c를 쓰면 m = 0의
    # 유효영역을 √2만큼 과소평가한다.
    from . import noise as _noi
    floors = np.array([2.0 * sigma_rel * (np.sqrt(2.0) if m == 0 else 1.0)
                       for m, _ in modes])
    below = (M < floors[:, None, None]).all(axis=0)
    return float(np.mean(below))


def validity_contour(plate, pool: Sequence[ker.ModeKernel],
                     modes: Sequence[tuple[int, int]], xi_grid: np.ndarray,
                     s_grid: np.ndarray, w: float, sigma_rel: float,
                     n_trial: int = 8, n_grid: int = 4001, mass=None,
                     abs_map: np.ndarray | None = None) -> np.ndarray:
    """각 ξ_d에서 '섭동오차 = 측정 floor'가 되는 S̄_D 임계값(없으면 NaN).

    정본 §3.4이 요구하는 "e_pert가 측정 floor를 넘는 등고선"의 수치판. s_grid는 증가순.
    `mass`는 `e_pert_map`과 같은 규약(설계서 M7). `abs_map`은 재사용 캐시
    (`fraction_below_floor`와 동일 규약).
    """
    M = (np.asarray(abs_map) if abs_map is not None else
         e_pert_map(plate, pool, modes, xi_grid, s_grid, w,
                    n_trial=n_trial, n_grid=n_grid, absolute=True, mass=mass))
    # 모드별 floor로 **먼저 나눈 뒤** 최댓값을 본다(F112·F126): m = 0의 σ_η는 2√2c라
    # 전 모드 2c로 재면 m = 0이 실제보다 일찍 floor를 넘는 것처럼 보인다.
    floors = np.array([2.0 * sigma_rel * (np.sqrt(2.0) if m == 0 else 1.0)
                       for m, _ in modes])
    worst = (M / floors[:, None, None]).max(axis=0)      # (n_xi, n_s), floor 단위
    floor = 1.0
    s = np.asarray(s_grid, dtype=float)
    thresh = np.full(len(xi_grid), np.nan)
    for i in range(worst.shape[0]):
        over = np.nonzero(worst[i] > floor)[0]
        if over.size == 0:
            continue
        j = int(over[0])
        if j == 0:
            thresh[i] = s[0]
        else:                                    # 로그-선형 보간
            y0, y1 = worst[i, j - 1], worst[i, j]
            t = (np.log(floor) - np.log(y0)) / (np.log(y1) - np.log(y0))
            thresh[i] = float(np.exp(np.log(s[j - 1]) + t * (np.log(s[j]) - np.log(s[j - 1]))))
    return thresh
