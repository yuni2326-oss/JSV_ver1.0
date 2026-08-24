"""노이즈 모델 Σ_y — 정본 §3.5의 "3 % of the shift" 폐기 후 대체.

리뷰의 지적: 이동량 비례 노이즈는 작은 이동에 비현실적으로 작은 오차를 준다(0.1 % 이동 →
0.003 % 오차). 대체 모델은 **절대 주파수 반복도**
    ε_m ~ N(0, σ_f²),  σ_f = c·f_m       (c = 재장착·곡선적합·열 오차의 상대 규모)
이고 관측량이 η̄ = 2Δf/f이므로 η̄ 단위 표준편차는 **2c**다(설계서 §4-5).

E1이 Σ_y를 측정하기 전까지 c는 파라미터이고, 논문은 4수준 스윕
    c ∈ {1e−4, 3e−4, 1e−3, 3e−3} (= f_m의 0.01/0.03/0.1/0.3 %)
과 상관 ρ ∈ {0, 0.3, 0.6}로 보고한다. 측정된 Σ_y가 아니라는 라벨은 산출물·논문 양쪽에 붙인다.
"""
from __future__ import annotations

import numpy as np


def sigma_y(n_obs: int, sigma_rel: float, rho: float = 0.0,
            sigma_rel_per_mode: np.ndarray | None = None) -> np.ndarray:
    """η̄ 단위 관측 공분산 Σ_y (n_obs × n_obs).

    sigma_rel: 주파수 상대 반복도 c (모드 공통). 모드별로 다르면 sigma_rel_per_mode 사용.
    rho: 등상관(equicorrelated) 계수 — 공통 온도·장착 드리프트를 모사.
    """
    if sigma_rel_per_mode is not None:
        s = 2.0 * np.asarray(sigma_rel_per_mode, dtype=float)
        if s.size != n_obs:
            raise ValueError("sigma_rel_per_mode length mismatch")
    else:
        s = np.full(n_obs, 2.0 * float(sigma_rel))
    if not (-1.0 / (n_obs - 1) < rho < 1.0) and n_obs > 1:
        raise ValueError(f"rho={rho} outside SPD range")
    corr = (1.0 - rho) * np.eye(n_obs) + rho * np.ones((n_obs, n_obs))
    return np.outer(s, s) * corr


def whitening(sigma: np.ndarray) -> np.ndarray:
    """Σ^{−1/2} (대칭 제곱근의 역) — J_w = Σ^{−1/2} J, 잔차 화이트닝에 공통 사용."""
    vals, vecs = np.linalg.eigh(np.asarray(sigma, dtype=float))
    if np.any(vals <= 0):
        raise ValueError("Sigma_y is not positive definite")
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def sample(sigma: np.ndarray, rng: np.random.Generator,
           size: int | None = None) -> np.ndarray:
    """Σ_y에서 관측오차 표본 추출. size=None이면 (n,), 아니면 (size, n)."""
    sigma = np.asarray(sigma, dtype=float)
    n = sigma.shape[0]
    L = np.linalg.cholesky(sigma)
    if size is None:
        return L @ rng.standard_normal(n)
    return rng.standard_normal((size, n)) @ L.T


def chi2(resid: np.ndarray, sigma_inv: np.ndarray) -> float:
    """χ² = rᵀ Σ⁻¹ r (모델비교·프로파일우도의 공통 통계)."""
    r = np.asarray(resid, dtype=float)
    return float(r @ sigma_inv @ r)


# ---------------------------------------------------------------------------
# 원시 주파수에서 유도한 공분산 (2026-08-15, 외부 검토 4차 #1)
# ---------------------------------------------------------------------------
#: 관측량 종류. 각각이 **원시 주파수 추정 몇 개를 어떤 계수로** 결합하는지가 분산을 정한다.
#:   pair_mean : 축퇴쌍 η̄_m = (f_{d,+}−f_{h,+})/f + (f_{d,−}−f_{h,−})/f   (4개, 계수 ±1)
#:   single    : m=0 η̄₀ = 2(f_d−f_h)/f                                   (2개, 계수 ±2)
#:   splitting : 증분 Δη_m = 2[(f_{d,+}−f_{d,−}) − (f_{h,+}−f_{h,−})]/f   (4개, 계수 ±2)
OBS_KINDS = ("pair_mean", "single", "splitting")


def observable_matrix(kinds) -> tuple[np.ndarray, list]:
    """관측량 종류 목록 → (A, 원시주파수 라벨). y = A·(δf/f).

    각 관측량이 **자기 모드의** 원시 추정만 쓰므로 A는 블록 대각이다. 라벨을 함께 돌려주어
    공통모드 상관(온도)이 어디에 걸리는지 호출자가 볼 수 있게 한다.
    """
    rows, labels = [], []
    cols: list[str] = []
    spec = []
    for j, k in enumerate(kinds):
        if k == "single":
            names = [f"h{j}", f"d{j}"]
            coeff = {f"h{j}": -2.0, f"d{j}": +2.0}
        elif k == "pair_mean":
            names = [f"h{j}+", f"h{j}-", f"d{j}+", f"d{j}-"]
            coeff = {f"h{j}+": -1.0, f"h{j}-": -1.0, f"d{j}+": +1.0, f"d{j}-": +1.0}
        elif k == "splitting":
            names = [f"h{j}+", f"h{j}-", f"d{j}+", f"d{j}-"]
            coeff = {f"h{j}+": -2.0, f"h{j}-": +2.0, f"d{j}+": +2.0, f"d{j}-": -2.0}
        else:
            raise ValueError(f"관측량 종류: {k}")
        for nm in names:
            if nm not in cols:
                cols.append(nm)
        spec.append(coeff)
    A = np.zeros((len(kinds), len(cols)))
    for i, coeff in enumerate(spec):
        for nm, v in coeff.items():
            A[i, cols.index(nm)] = v
    return A, cols


def sigma_y_from_raw(kinds, sigma_rel: float, rho: float = 0.0,
                     n_avg: int = 1) -> np.ndarray:
    """**Σ_y = A Σ_f Aᵀ** — 관측량 공분산을 원시 주파수에서 전파한다.

    검토 #1의 요구: 유도 관측량에 공분산을 *가정*하지 말고 원시 주파수에서 유도하라. 그래야
    (i) pair mean / m=0 / splitting이 서로 **다른 분산**을 갖는다는 사실이 자동으로 나오고,
    (ii) 온도 같은 **공통모드 상관이 splitting에서 상쇄되는 것**이 자동으로 반영된다.

    독립 오차(ρ=0)·n_avg 회 평균에서 결과는 σ = {2c, 2√2c, 4c}/√n_avg (pair mean / single /
    splitting)이다 — 옛 규약이 셋 모두에 2c를 쓴 것은 pair mean에만 맞았다.
    """
    A, cols = observable_matrix(kinds)
    n = len(cols)
    s = float(sigma_rel) / np.sqrt(max(int(n_avg), 1))
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho={rho} outside [0,1)")
    # **세션 블록** 상관: 온도·장착 드리프트는 한 세션(건전 측정 전체 / 손상 측정 전체) 안에서
    # 공통이고 세션 사이에서는 독립이다. 전역 등상관으로 두면 세션간 드리프트까지 상쇄되어
    # 셋 모두 같은 비율로 줄어드는데, 그것은 물리가 아니라 모델의 인공물이다.
    # 이 구조에서 (해석적으로) Var = 4σ²(1+ρ) / 8σ² / 16σ²(1−ρ)가 되어
    # **splitting만 공통모드에 면역**이고 pair mean은 오히려 나빠진다.
    sess = np.array([0 if c.startswith("h") else 1 for c in cols])
    corr = (1.0 - rho) * np.eye(n) + rho * (sess[:, None] == sess[None, :])
    return A @ (s * s * corr) @ A.T


def kinds_for_modes(pool, include_splitting: bool = False) -> list[str]:
    """모드 목록 → 관측량 종류. m = 0은 축퇴쌍이 없으므로 `single`이다.

    pool 원소는 `.m`을 가진 커널이거나 정수 방위차수 어느 쪽이어도 된다.
    """
    ms = [(k.m if hasattr(k, "m") else int(k)) for k in pool]
    kinds = ["single" if m == 0 else "pair_mean" for m in ms]
    if include_splitting:
        kinds += ["splitting"] * len(ms)
    return kinds


def sigma_y_for_modes(pool, sigma_rel: float, rho: float = 0.0, n_avg: int = 1,
                      include_splitting: bool = False) -> np.ndarray:
    """생산 규약(2026-08-15~) — 모드 목록에서 곧바로 Σ_y를 만든다.

    옛 `sigma_y(n_obs, c)`는 모든 관측량에 σ_η = 2c를 주었고 그것은 **doublet pair mean에만**
    맞았다(설계서 F112). 이 함수는 `sigma_y_from_raw`로 위임하므로 m = 0의 √2와 splitting의 2가
    자동으로 들어간다. 호출자는 관측량 종류를 몰라도 되고, 모드 풀만 넘기면 된다.
    """
    return sigma_y_from_raw(kinds_for_modes(pool, include_splitting),
                            sigma_rel, rho=rho, n_avg=n_avg)
