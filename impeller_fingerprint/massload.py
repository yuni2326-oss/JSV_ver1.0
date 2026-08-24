"""접촉센서 질량부하 한계 — 규칙이 아니라 **정확식**.

질량정규화 모드형(φᵀMφ = 1)을 갖고 있으면 부착질량의 영향은 경험규칙("구조질량의 1/10")이
아니라 Rayleigh 몫의 1차 섭동으로 정확히 나온다. 점질량 m_a가 위치 x에 강체부착되면

    δλ/λ = −φ(x)ᵀ (m_a I₃) φ(x) / (φᵀMφ) = −m_a |φ(x)|²
    **δf/f = −½ m_a |φ(x)|²**                                   ... (ML)

여기서 |φ(x)|² = φ_x² + φ_y² + φ_z²이고 단위는 **kg⁻¹**이다(φ는 kg^(−1/2)). 등가 모달질량
m_eff(x) = 1/|φ(x)|²로 읽으면 (ML)은 δf/f = −½ m_a/m_eff가 되어 해석이 명확해진다.

**왜 절점값을 그대로 쓸 수 있는가**: Lagrange 요소에서 절점 기저는 δ 성질(N_i(x_j) = δ_ij)을
가지므로, 점질량이 절점에 놓이면 ΔM = m_a N(x)ᵀN(x)가 그 절점의 3자유도에만 m_a를 얹는다.
따라서 δλ = m_a |u_절점|²이고, u_절점은 곧 변위장 값이다. 절점 사이에 붙는 경우는 형상함수
보간이 되어 |φ|²가 두 절점값 사이에 들어가므로, **절점 최댓값이 상계**다.

**정규화 규약**: `scipy.sparse.linalg.eigsh(K, M=M, sigma=…)`는 일반화 고유문제를 B-정규직교
벡터로 돌려주므로 φᵀMφ = 1이 이미 성립한다. 이 모듈은 그것을 가정하지 않고 `check_normalization`
으로 검증할 수 있게 해 두었다(M을 넘기면 최대 편차를 돌려준다). 구속자유도가 0으로 확장된
`full_shapes`도 같은 규약을 만족한다 — 구속절점은 M에 기여하지 않기 때문이다.

**해석 앵커**(테스트로 고정): EB 캔틸레버 1차의 팁 유효질량은 0.2427 m_total이므로
|φ(L)|² = 1/(0.2427 m) ≈ 4.12/m이다. 3D 레일이 이 값을 재현하면 정규화·(ML)·절점추출이
동시에 검증된다.
"""
from __future__ import annotations

import numpy as np

#: EB 캔틸레버 1차의 팁 유효질량 비 m_eff/m_total (해석값). |φ(L)|² = 1/(이 값 × m).
EB_TIP_EFFECTIVE_MASS_RATIO = 0.24267


def nodal_phi2(shape: np.ndarray, dim: int = 3) -> np.ndarray:
    """모드형 1개(평탄 자유도 벡터) → 절점별 |φ_p|² [kg⁻¹].

    shape은 (n_node·dim,)로 절점당 dim개 성분이 연속 배치된 것으로 본다(sfepy·hex 공통).
    """
    u = np.asarray(shape, dtype=float).reshape(-1, dim)
    return (u ** 2).sum(axis=1)


def check_normalization(shapes: np.ndarray, M) -> float:
    """max|φᵀMφ − 1| — 질량정규화가 실제로 성립하는지의 직접 검증.

    shapes는 (n_dof, n_modes). M은 같은 자유도 공간의 (희소) 질량행렬.
    """
    S = np.asarray(shapes, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    dev = 0.0
    for k in range(S.shape[1]):
        v = S[:, k]
        dev = max(dev, abs(float(v @ (M @ v)) - 1.0))
    return dev


def df_f_point_mass(m_a: float, phi2: float | np.ndarray) -> float | np.ndarray:
    """식 (ML) — 점질량 m_a[kg]가 |φ|² = phi2[kg⁻¹] 지점에 붙을 때의 상대 주파수 이동(음수)."""
    return -0.5 * np.asarray(m_a, dtype=float) * np.asarray(phi2, dtype=float)


def mass_limit(phi2: float | np.ndarray, df_f_budget: float) -> float | np.ndarray:
    """|δf/f| ≤ budget을 만족하는 최대 부착질량 [kg] — m_a ≤ 2·budget/|φ|².

    budget은 **양수**로 준다(예: floor 0.1 %의 절반이면 5e−4).
    """
    if df_f_budget <= 0:
        raise ValueError("df_f_budget은 양수여야 한다")
    return 2.0 * df_f_budget / np.asarray(phi2, dtype=float)


def modal_effective_mass(phi2: float | np.ndarray) -> float | np.ndarray:
    """등가 모달질량 m_eff = 1/|φ|² [kg]. (ML)을 δf/f = −½ m_a/m_eff로 읽게 해 준다."""
    return 1.0 / np.asarray(phi2, dtype=float)


def mode_phi2_max(shape: np.ndarray, coors: np.ndarray, dim: int = 3,
                  mask: np.ndarray | None = None) -> dict:
    """한 모드의 절점 |φ|² 최댓값과 그 위치.

    mask를 주면 그 절점 부분집합(예: 센서를 실제로 붙일 수 있는 외부면)에서만 찾는다.
    반환: phi2_max [kg⁻¹], m_eff [kg], node(색인), r·z [m], phi2_median.
    """
    p2 = nodal_phi2(shape, dim=dim)
    if mask is None:
        mask = np.ones(p2.size, dtype=bool)
    if p2.size != coors.shape[0]:
        raise ValueError(f"절점수 불일치: 모드형 {p2.size} vs 좌표 {coors.shape[0]}")
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        raise ValueError("mask가 빈 집합이다")
    j = idx[int(np.argmax(p2[idx]))]
    c = coors[j]
    return {"phi2_max": float(p2[j]), "m_eff_kg": float(1.0 / p2[j]), "node": int(j),
            "r_mm": float(1e3 * np.hypot(c[0], c[1])),
            "z_mm": float(1e3 * c[2]) if coors.shape[1] > 2 else 0.0,
            "phi2_median": float(np.median(p2[idx]))}


def mesh_volume(coors: np.ndarray, conn: np.ndarray) -> float:
    """격자 총부피 [m³] — 4절점 사면체 / 8절점 육면체 공통(질량 대비 % 보고용).

    육면체는 6개 사면체로 분할해 적분한다(뒤틀린 요소에서도 부호 있는 부피 합이 맞다).
    """
    c, cn = np.asarray(coors, float), np.asarray(conn)
    if cn.shape[1] == 4:
        tets = [cn]
    elif cn.shape[1] == 8:
        # 표준 6-사면체 분할 (절점 순서 0..7 = 하면 0123 / 상면 4567)
        idx = [(0, 1, 3, 7), (0, 1, 7, 4), (1, 2, 3, 7), (1, 5, 7, 4),
               (1, 2, 7, 5), (2, 6, 7, 5)]
        tets = [cn[:, list(t)] for t in idx]
    else:
        raise ValueError(f"지원하지 않는 요소(절점 {cn.shape[1]}개)")
    vol = 0.0
    for t in tets:
        p = c[t]                                   # (n_elem, 4, 3)
        vol += float(np.abs(np.einsum(
            "ei,ei->e", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
            p[:, 3] - p[:, 0])).sum() / 6.0)
    return vol


def outer_surface_mask(coors: np.ndarray, r_out: float, z_top: float,
                       tol: float = 5e-4) -> np.ndarray:
    """센서를 **실제로 붙일 수 있는** 절점 — 전면 슈라우드 상면 또는 외주면.

    폐쇄형 임펠러의 내부 유로면은 접근 불가이므로, 현실적 부착 한계는 이 부분집합에서
    잰 |φ|²로 정해진다. 최악값(전 절점 최댓값)과 함께 보고해 둘 사이의 여유를 드러낸다.
    """
    r = np.hypot(coors[:, 0], coors[:, 1])
    return ((np.abs(coors[:, 2] - z_top) <= tol) | (np.abs(r - r_out) <= tol))
