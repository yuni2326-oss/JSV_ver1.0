"""A5 — D-최적 모드부분집합과 모드셋 비교 (정본 §3.5·§4.3(iv)).

리뷰가 지적한 γ₀≈γ₁ 커널 중복성에 대한 **구성적 답**: 더 넓은 후보 풀(m=0…4 × 반경차수 n=0,1)에서
det F를 최대화하는 부분집합을 고른다(D-최적). 그리고 다섯 모드셋을 비교한다.

  (i)   default_m0_3        기본 m=0–3 (n=0)              — 가우시안 반경손상 모델
  (ii)  d_optimal_4         측정가능 풀에서 D-최적 4모드   — 가우시안 반경손상 모델
  (iii) all_measurable      측정가능 전체                  — 가우시안 반경손상 모델
  (iv)  freq_only           포켓 모델, pair mean만
  (v)   freq_plus_splitting 포켓 모델, pair mean + 분리 Δη

(iv)/(v)는 **포켓 모델**(E3의 물리 사례)에서 계산한다 — 분리는 방위 국소성이 있어야 정의되기
때문이다. 두 행은 같은 파라미터 (ξ_d, S̄_D)를 추정 대상으로 하므로 CRLB가 직접 비교된다.
분리 관측량의 잡음은 짝 두 성분의 독립 오차 가정에서 σ_Δη = √2·σ_η로 둔다.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from . import degenerate as deg
from . import forward as fwd
from . import identifiability as idf
from . import noise as noi
from . import severity as sev

#: (iv)/(v) 비교에 쓰는 포켓 기하 기준값 — E3 설계와 같은 규모
POCKET_RADIAL_WIDTH = 0.006      # 6 mm
POCKET_DTHETA = np.deg2rad(30.0)
POCKET_THETA0 = 0.0


def pool_table(pool: Sequence, f_max: float = 2.0e4) -> list[dict]:
    """후보 모드 풀 표 — 측정가능성(DAQ 대역) 플래그 포함."""
    return [{"label": k.label, "m": k.m, "n": k.n, "f_Hz": k.f,
             "measurable": bool(k.f <= f_max)} for k in pool]


def subset_metrics(subset: Sequence, plate, theta: tuple[float, float], w: float,
                   sigma: np.ndarray, mass=None) -> dict:
    """부분집합의 식별성 지표(가우시안 반경손상 모델, pair mean 관측)."""
    J = (fwd.jacobian_linear_mass(subset, theta[0], theta[1], w, plate,
                                  coupling=fwd.resolve_coupling(mass))
         if mass else fwd.jacobian_linear(subset, theta[0], theta[1], w, plate))
    return idf.metrics_from_J(J, sigma, plate.extent)


def d_optimal_subset(pool: Sequence, plate, theta: tuple[float, float], w: float,
                     sigma: np.ndarray, k: int, mass=None) -> dict:
    """det F를 최대화하는 k-모드 부분집합(전수탐색; 풀이 작아 가능).

    `mass`가 주어지면 질량항 포함 순방향(설계서 F12)의 야코비안으로 det F를 평가한다 —
    선택 자체가 모델에 의존하므로 반드시 함께 넘겨야 한다(설계서 M7/A5).
    """
    best = None
    for combo in combinations(range(len(pool)), k):
        subset = [pool[i] for i in combo]
        m = subset_metrics(subset, plate, theta, w, sigma, mass=mass)
        if best is None or m["det_F"] > best["det_F"]:
            best = {"labels": [s.label for s in subset], "indices": list(combo),
                    "det_F": m["det_F"], "metrics": m}
    return best


def _depth_from_severity(s_bar: float, radial_width: float, extent: float) -> float:
    """S̄_D(반경 기준) → 포켓 깊이비 (severity.pocket_depth_to_severity의 역)."""
    x = float(s_bar) * extent / radial_width
    x = min(max(x, 0.0), 0.999999)
    return 1.0 - (1.0 - x) ** (1.0 / 3.0)


def pocket_observables(plate, ms_list: Sequence[int], xi_d: float, s_bar: float,
                       *, radial_width: float = POCKET_RADIAL_WIDTH,
                       dtheta: float = POCKET_DTHETA, theta0: float = POCKET_THETA0,
                       include_splitting: bool = False, n_grid: int = 1001) -> np.ndarray:
    """포켓 모델 관측벡터: [η̄_m …] (+ [Δη_m …] if include_splitting)."""
    depth = _depth_from_severity(s_bar, radial_width, plate.extent)
    r_c = sev.xi_to_r(xi_d, plate.a, plate.b)
    r1 = max(plate.a, r_c - 0.5 * radial_width)
    r2 = min(plate.b, r1 + radial_width)
    pocket = deg.Pocket(r1=r1, r2=r2, theta0=theta0, dtheta=dtheta,
                        depth_frac=depth)
    means, splits = [], []
    for m in ms_list:
        o = deg.observables(plate, m, pocket, n_grid=n_grid, n_r=1001)
        means.append(o["eta_bar"])
        splits.append(o["delta_eta"])
    return np.array(means + splits) if include_splitting else np.array(means)


def pocket_jacobian(plate, ms_list: Sequence[int], theta: tuple[float, float],
                    *, include_splitting: bool = False,
                    h_xi: float = 5e-3, h_s: float = 1e-3, **kw) -> np.ndarray:
    """포켓 모델 야코비안 ∂관측/∂(ξ_d, S̄_D) — 중심차분."""
    xi, s_bar = theta

    def obs(x, s):
        return pocket_observables(plate, ms_list, x, s,
                                  include_splitting=include_splitting, **kw)

    d_xi = (obs(xi + h_xi, s_bar) - obs(xi - h_xi, s_bar)) / (2 * h_xi)
    d_s = (obs(xi, s_bar + h_s) - obs(xi, s_bar - h_s)) / (2 * h_s)
    return np.stack([d_xi, d_s], axis=1)


def compare_mode_sets(plate, pool: Sequence, theta: tuple[float, float], w: float,
                      sigma_rel: float, f_max: float = 2.0e4,
                      ms_pocket: Sequence[int] = (1, 2, 3), mass=None) -> list[dict]:
    """다섯 모드셋의 CRLB 비교표 — "대칭성 관측량이 무엇을 더 주는가"의 정량답.

    `mass`(예: "exact" 또는 결합비 숫자)를 주면 (i)–(iii) 가우시안 반경손상 행을 질량항
    포함 순방향으로 계산한다. (iv)/(v) 포켓 행은 `degenerate`가 이미 δK−λδM을 직접 다루므로
    (`mass_term=True` 기본) `mass` 값과 무관하게 항상 질량항을 포함한다.
    """
    measurable = [k for k in pool if k.f <= f_max]
    default = [k for k in pool if k.n == 0 and k.m <= 3]
    rows: list[dict] = []

    def add(name, subset_metrics_dict, n_obs, model, labels):
        rows.append({"set": name, "model": model, "n_obs": n_obs,
                     "labels": labels,
                     "det_F": subset_metrics_dict["det_F"],
                     "sigma_min": subset_metrics_dict["sigma_min"],
                     "cond2": subset_metrics_dict["cond2"],
                     "crlb_xi_mm": subset_metrics_dict["crlb_xi_mm"],
                     "crlb_s_pp": subset_metrics_dict["crlb_s_pp"]})

    m_def = subset_metrics(default, plate, theta, w,
                           noi.sigma_y_for_modes(default, sigma_rel), mass=mass)
    add("default_m0_3", m_def, len(default), "gaussian_radial",
        [k.label for k in default])

    k_opt = min(4, len(measurable))
    best = d_optimal_subset(measurable, plate, theta, w,
                            noi.sigma_y_for_modes(measurable[:k_opt], sigma_rel), k=k_opt, mass=mass)
    add("d_optimal_4", best["metrics"], k_opt, "gaussian_radial", best["labels"])

    m_all = subset_metrics(measurable, plate, theta, w,
                           noi.sigma_y_for_modes(measurable, sigma_rel), mass=mass)
    add("all_measurable", m_all, len(measurable), "gaussian_radial",
        [k.label for k in measurable])

    n_m = len(ms_pocket)
    J_f = pocket_jacobian(plate, ms_pocket, theta, include_splitting=False)
    add("freq_only", idf.metrics_from_J(J_f, noi.sigma_y_for_modes(ms_pocket, sigma_rel),
                                        plate.extent),
        n_m, "pocket", [f"eta_m{m}" for m in ms_pocket])

    J_fs = pocket_jacobian(plate, ms_pocket, theta, include_splitting=True)
    # 분리 분산은 pair mean의 **4배**다(σ = 4c 대 2c) — 옛 코드의 ×2는 틀렸다(F112).
    sig = noi.sigma_y_for_modes(ms_pocket, sigma_rel, include_splitting=True)
    add("freq_plus_splitting", idf.metrics_from_J(J_fs, sig, plate.extent),
        2 * n_m, "pocket",
        [f"eta_m{m}" for m in ms_pocket] + [f"dEta_m{m}" for m in ms_pocket])
    return rows


def collinearity_report(pool: Sequence) -> list[dict]:
    """커널 쌍별 코사인 유사도 — γ₀≈γ₁ 중복성의 정량판."""
    out = []
    for i, j in combinations(range(len(pool)), 2):
        gi, gj = pool[i].gamma, pool[j].gamma
        cos = float(gi @ gj / (np.linalg.norm(gi) * np.linalg.norm(gj)))
        out.append({"a": pool[i].label, "b": pool[j].label, "cos": cos})
    return out
