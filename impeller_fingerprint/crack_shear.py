"""A7 — Timoshenko 보 + Mode-I/II 국소유연도: 곡률-null "실명" 예측의 모델 내 검정.

정본 §4.1은 mode-2 실명(곡률 null에 균열이 놓여 거의 움직이지 않음)을 **단일베인
Euler–Bernoulli 이상화 안의 예측**으로 격하하고, §3.6-iv에서 전단·활동(Mode-II) 유연도를 넣어도
살아남는지 확인하라고 요구한다. 이 모듈이 그 해석 arm이다(3D 탄성 arm은 `rail3d`).

구성
  1. **국소유연도**: 변형에너지 방출률에서 Castigliano로 얻는다.
       K_I  = (6M/(b h²))√(πa)·F_b(ā),     K_II = (Q/(b h))√(πa)·F_II(ā)
       c_MM = (72π/(E' b h²))∫₀^ā ā' F_b(ā')² dā'      [rad/(N·m)]
       c_QQ = ( 2π/(E' b    ))∫₀^ā ā' F_II(ā')² dā'    [m/N]
     F_b는 Tada 순수굽힘 형상함수, F_II는 활동모드 모서리균열 형상함수.
     **검증**: c_MM은 논문1이 이미 쓰는 Dimarogonas 다항식 c_θ = 5.346(h/EI)J(ā)와 대조한다
     (테스트). 같은 적분기계로 얻은 c_QQ의 신뢰성은 이 대조와 `rail3d`의 3D 검정에 의존한다.
  2. **결합항 c_MQ**: 문헌마다 처리가 갈리므로 값을 단정하지 않고
     Cauchy–Schwarz 상한 |c_MQ| ≤ √(c_MM·c_QQ) 안에서 **스윕**해 결론의 민감도를 보고한다.
  3. **Timoshenko 보 FE**(선택적 감차적분, 전단잠김 억제) + 균열 = 영길이 유연연결
     (상대 [Δw, Δφ]에 C⁻¹). 세장극한에서 EB 전달행렬 해와 대조(T8).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

KAPPA_RECT = 5.0 / 6.0


def F_bending(a_bar: float) -> float:
    """Tada 순수굽힘 모서리균열 형상함수 F_b(ā)."""
    if a_bar <= 0.0:
        return 0.0
    x = min(max(a_bar, 1e-12), 0.999)
    t = math.pi * x / 2.0
    return (math.sqrt(math.tan(t) / t)
            * (0.923 + 0.199 * (1.0 - math.sin(t)) ** 4) / math.cos(t))


def F_shear(a_bar: float) -> float:
    """활동(Mode-II) 모서리균열 형상함수 F_II(ā)."""
    x = min(max(a_bar, 0.0), 0.999)
    return (1.122 - 0.561 * x + 0.085 * x ** 2 + 0.18 * x ** 3) / math.sqrt(1.0 - x)


def _int_shape(func, a_bar: float, n: int = 2001) -> float:
    """∫₀^ā ā' F(ā')² dā'."""
    if a_bar <= 0.0:
        return 0.0
    s = np.linspace(0.0, a_bar, n)
    vals = np.array([x * func(float(x)) ** 2 for x in s])
    return float(np.trapezoid(vals, s))


def compliance(a_bar: float, h: float, b: float, E: float, nu: float,
               plane_strain: bool = False, convention: str = "tada") -> dict:
    """균열단면 국소유연도 c_MM [rad/(N·m)], c_QQ [m/N]와 상한 √(c_MM c_QQ).

    **규약 문제와 그 판정(2026-08-10).** 같은 Castigliano 기계라도 핸드북 형상함수가 달라
    굽힘 유연도가 20–30 % 차이난다: Tada 적분식 c_MM_int과 논문1·정본이 인용한 Dimarogonas
    다항식 c_θ = 5.346(h/EI)J(ā)의 비가 ā에 따라 ~1.2–1.3이다.

    A11(`crack2d`, 폭 0 traction-free 슬릿의 2D 평면탄성)이 이 양자택일을 **판정했다**:
    폭 0 균열의 f₁ 강하에서 역산한 등가 회전유연도는 **Tada와 1–3 % 안에서 일치**하고
    (c_θ^2D/c_θ^Tada = 1.011–1.026, ā = 0.25–0.625) Dimarogonas보다 **27–30 % 크다**
    (1.266–1.303). 즉 Dimarogonas 다항식은 c_θ를 약 23 % **저평가**한다(설계서 F42).
    따라서 **생산 경로의 기본 규약은 `"tada"`**다.

    - `convention="tada"`(**기본**): 두 값 모두 적분식 그대로. `handbook_scale = 1`.
    - `convention="dimarogonas"`: 굽힘을 Dimarogonas 다항식으로 바꾸고 Mode-II를 같은
      정규화 인자로 환산한다 — c_QQ = c_QQ_int × (c_MM_dim / c_MM_int). **하위호환·회귀검정용**
      으로만 유지한다(논문1 코드와의 연속성, F42 이전 산출물의 재현).
    """
    Ep = E / (1.0 - nu ** 2) if plane_strain else E
    c_mm_int = (72.0 * math.pi / (Ep * b * h ** 2)) * _int_shape(F_bending, a_bar)
    c_qq_int = (2.0 * math.pi / (Ep * b)) * _int_shape(F_shear, a_bar)
    if convention == "tada" or a_bar <= 0.0:
        c_mm, c_qq = c_mm_int, c_qq_int
        scale = 1.0
    elif convention == "dimarogonas":
        c_mm = compliance_dimarogonas(a_bar, h, b, E)
        scale = c_mm / c_mm_int if c_mm_int > 0 else 1.0
        c_qq = c_qq_int * scale
    else:
        raise ValueError(f"unknown convention: {convention}")
    return {"c_MM": c_mm, "c_QQ": c_qq, "c_MQ_max": math.sqrt(c_mm * c_qq),
            "c_MM_int": c_mm_int, "c_QQ_int": c_qq_int,
            "handbook_scale": scale}


#: Ostachowicz & Krawczuk(1991) 무차원 국소 굽힘 유연도 J(ā), ā = a/h ∈ [0, ~0.6] —
#: 정본 [3]의 다항식이다. 논문1 `impeller_pinn.crack_beam.flexibility_J`와 같은 식이지만
#: **여기서 다시 구현한다**: 이 패키지는 논문1 없이도 Table 1을 재현해야 하고(코드·데이터
#: 배포본), J(ā)는 논문1의 기여가 아니라 문헌 값이다. 두 구현의 일치는
#: `test_crack_shear.TestLiteratureFlexibility`가 교차검증한다(폴더 규약: 재구현 + 교차검증).
_J_COEFFS = (1.8624, -3.95, 16.375, -37.226, 76.81, -126.9, 172.5, -143.97, 66.56)


def flexibility_J(a_bar: float) -> float:
    """무차원 국소 굽힘 유연도 J(ā) — ā²부터 ā¹⁰까지의 문헌 다항식."""
    s = float(a_bar)
    return sum(c * s ** k for k, c in enumerate(_J_COEFFS, start=2))


def compliance_dimarogonas(a_bar: float, h: float, b: float, E: float) -> float:
    """회전유연도 c_θ = 5.346 (h/EI) J(ā) — 대조 기준(A11이 23 % 저평가로 판정)."""
    I = b * h ** 3 / 12.0
    return 5.346 * (h / (E * I)) * flexibility_J(a_bar)


# ------------------------------------------------------------------ 정확 EB 전달행렬
# 정본 Table 1은 회전스프링 EB 캔틸레버의 **해석 전달행렬** 해다(논문1 `crack_beam`).
# 그 함수는 ā에서 Dimarogonas 규약의 κ를 내부에서 만들므로 다른 규약을 넣을 수 없고,
# 논문1 코드는 **수정 금지**다. 그래서 같은 8×8 특성행렬식을 논문3 안에 독립 구현하고
# 회전유연도 c_θ를 **직접** 받는다 — Table 1을 Tada 규약으로 다시 낼 수 있게 하는 경로.
# 회귀검정(`test_crack_shear.TestExactEBFromCompliance`)이 Dimarogonas c_θ를 넣으면
# 논문1 함수와 1e-9로 일치함을 고정한다.

_EB_BETA_L = (1.8751040687, 4.6940911330, 7.8547574382)


def _eb_charmat(s: float, xc_over_L: float, kappa_nd: float) -> float:
    """무차원 파수 s = βL, 균열위치 x_c/L, 무차원 유연도 κ = c_θ·EI/L의 8×8 특성행렬식.

    경계·연속 조건: 고정단 W=W'=0, 자유단 W''=W'''=0, 균열단면에서 변위·모멘트·전단 연속 +
    기울기 불연속 Δφ' = κ·L·φ''(= c_θ·M).
    """
    def f0(t): return np.array([math.cosh(t), math.sinh(t), math.cos(t), math.sin(t)])
    def f1(t): return np.array([math.sinh(t), math.cosh(t), -math.sin(t), math.cos(t)])
    def f2(t): return np.array([math.cosh(t), math.sinh(t), -math.cos(t), -math.sin(t)])
    def f3(t): return np.array([math.sinh(t), math.cosh(t), math.sin(t), -math.cos(t)])
    z = np.zeros(4)
    tc, tL = s * xc_over_L, s
    M = np.empty((8, 8))
    M[0] = np.concatenate([[1.0, 0.0, 1.0, 0.0], z])
    M[1] = np.concatenate([[0.0, 1.0, 0.0, 1.0], z])
    M[2] = np.concatenate([z, f2(tL)])
    M[3] = np.concatenate([z, f3(tL)])
    M[4] = np.concatenate([f0(tc), -f0(tc)])
    M[5] = np.concatenate([f2(tc), -f2(tc)])
    M[6] = np.concatenate([f3(tc), -f3(tc)])
    M[7] = np.concatenate([-f1(tc) - kappa_nd * s * f2(tc), f1(tc)])
    return float(np.linalg.det(M))


def exact_eb_betas(kappa_nd: float, xc_over_L: float = 0.2, n_modes: int = 3,
                   s_max: float = 11.0, n_scan: int = 2200) -> np.ndarray:
    """균열 캔틸레버의 무차원 고유값 βL (n_modes개). κ=0이면 건전보 1.8751…을 복원한다."""
    from scipy.optimize import brentq

    ss = np.linspace(0.3, s_max, n_scan)
    dv = np.array([_eb_charmat(float(s), xc_over_L, kappa_nd) for s in ss])
    roots: list[float] = []
    for i in range(len(ss) - 1):
        if dv[i] == 0.0:
            roots.append(float(ss[i]))
        elif dv[i] * dv[i + 1] < 0:
            roots.append(float(brentq(_eb_charmat, ss[i], ss[i + 1],
                                      args=(xc_over_L, kappa_nd), xtol=1e-12)))
        if len(roots) >= n_modes:
            break
    return np.array(roots[:n_modes])


def exact_eb_ratios(beam: TimoBeam, c_theta: float, xc_over_L: float = 0.2,
                    n_modes: int = 3) -> np.ndarray:
    """회전유연도 c_θ [rad/(N·m)]를 **직접** 받는 정확 EB 전달행렬의 주파수비 f_i/f_i^healthy.

    f ∝ (βL)² 이므로 비는 (βL_cracked/βL_healthy)²로, 단면·재료 상수가 상쇄된다.
    """
    kappa_nd = c_theta * beam.E * beam.I / beam.L
    b_cracked = exact_eb_betas(kappa_nd, xc_over_L=xc_over_L, n_modes=n_modes)
    b_healthy = np.array(_EB_BETA_L[:n_modes])
    if b_cracked.size < n_modes:                            # pragma: no cover
        raise RuntimeError(f"근 {n_modes}개를 못 찾았다: {b_cracked}")
    return (b_cracked / b_healthy) ** 2


@dataclass(frozen=True)
class TimoBeam:
    """등단면 Timoshenko 캔틸레버(직사각 단면)."""

    L: float
    h: float
    b: float
    E: float
    rho: float
    nu: float
    kappa: float = KAPPA_RECT

    @property
    def A(self) -> float:
        return self.b * self.h

    @property
    def I(self) -> float:
        return self.b * self.h ** 3 / 12.0

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))


def _element_matrices(beam: TimoBeam, le: float):
    """2절점 Timoshenko 요소 (w, φ) — 굽힘 2점·전단 1점(감차) 적분, 일치질량."""
    EI, GA = beam.E * beam.I, beam.kappa * beam.G * beam.A
    # 굽힘: φ' = (φ2−φ1)/le  →  K_b = EI/le * [[0,0,0,0],[0,1,0,-1],[0,0,0,0],[0,-1,0,1]]
    Kb = np.zeros((4, 4))
    Kb[1, 1] = Kb[3, 3] = EI / le
    Kb[1, 3] = Kb[3, 1] = -EI / le
    # 전단(1점 감차): γ = (w2−w1)/le − (φ1+φ2)/2
    B = np.array([-1.0 / le, -0.5, 1.0 / le, -0.5])
    Ks = GA * le * np.outer(B, B)
    # 질량(일치): 병진 ρA, 회전관성 ρI
    m_t = beam.rho * beam.A * le / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    m_r = beam.rho * beam.I * le / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    M = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            M[2 * i, 2 * j] = m_t[i, j]
            M[2 * i + 1, 2 * j + 1] = m_r[i, j]
    return Kb + Ks, M


def assemble(beam: TimoBeam, n_elem: int = 200, crack: dict | None = None,
             xc_over_L: float = 0.2):
    """(K, M, n_dof) 조립. crack이 주어지면 x_c에 유연연결(영길이)을 삽입한다.

    crack = {"c_MM":…, "c_QQ":…, "c_MQ":…}. 절점을 이중화해 상대 [Δw, Δφ]에 C⁻¹를 건다.
    """
    le = beam.L / n_elem
    ic = int(round(xc_over_L * n_elem))
    ic = min(max(ic, 1), n_elem - 1)
    n_nodes = n_elem + 1 + (1 if crack else 0)
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    Ke, Me = _element_matrices(beam, le)

    # 절점 번호: 0…ic = 균열 왼쪽면까지, ic+1 = 오른쪽 균열면(이중화), 이후 한 칸씩 밀림.
    # 균열 오른쪽 요소(e ≥ ic)는 두 절점 모두 밀린 번호를 쓴다.
    for e in range(n_elem):
        if crack and e >= ic:
            n1, n2 = e + 1, e + 2
        else:
            n1, n2 = e, e + 1
        dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += Ke[i, j]
                M[dofs[i], dofs[j]] += Me[i, j]

    if crack:
        C = np.array([[crack["c_QQ"], crack.get("c_MQ", 0.0)],
                      [crack.get("c_MQ", 0.0), crack["c_MM"]]])   # [Δw, Δφ]
        Kc = np.linalg.inv(C)
        left, right = ic, ic + 1
        T = np.zeros((2, ndof))
        T[0, 2 * right] = 1.0
        T[0, 2 * left] = -1.0
        T[1, 2 * right + 1] = 1.0
        T[1, 2 * left + 1] = -1.0
        K += T.T @ Kc @ T
    return K, M


def frequencies(beam: TimoBeam, n_modes: int = 3, n_elem: int = 200,
                crack: dict | None = None, xc_over_L: float = 0.2) -> np.ndarray:
    """캔틸레버(고정단 w=φ=0) 고유주파수 [Hz]."""
    K, M = assemble(beam, n_elem=n_elem, crack=crack, xc_over_L=xc_over_L)
    free = np.arange(2, K.shape[0])            # 절점0의 w, φ 고정
    lam = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True)
    lam = lam[lam > 0]
    return np.sqrt(lam[:n_modes]) / (2.0 * math.pi)


def signature(beam: TimoBeam, a_bars, xc_over_L: float = 0.2, n_modes: int = 3,
              n_elem: int = 200, coupling: float = 0.0, shear_flex: bool = True,
              plane_strain: bool = False, convention: str = "tada") -> list[dict]:
    """균열깊이 스윕 → 주파수비 f_i/f_i^healthy.

    coupling ∈ [0,1]: c_MQ = coupling·√(c_MM c_QQ) (Cauchy–Schwarz 상한의 비율).
    shear_flex=False면 c_QQ=0 (Mode-I 단독 = 고전 회전스프링 모델).
    convention: `compliance`와 동일 — 기본 `"tada"`(A11이 판정한 생산 규약, F42).
    """
    f0 = frequencies(beam, n_modes=n_modes, n_elem=n_elem, crack=None)
    rows = []
    for ab in a_bars:
        c = compliance(float(ab), beam.h, beam.b, beam.E, beam.nu,
                       plane_strain=plane_strain, convention=convention)
        c_qq = c["c_QQ"] if shear_flex else 0.0
        c_mq = coupling * math.sqrt(c["c_MM"] * c_qq) if c_qq > 0 else 0.0
        # c_QQ=0이면 상대 Δw를 막아야 하므로 아주 작은 유연도로 대체(수치적 강체)
        # c_QQ=0이면 상대 Δw를 막아야 하므로 작은 유연도로 대체한다. 너무 작게 잡으면
        # C⁻¹의 조건수가 폭발해 고유해가 망가지므로 1e-6 배(강성비 10⁶)로 둔다.
        c_qq_eff = c_qq if c_qq > 0 else c["c_MM"] * beam.h ** 2 * 1e-6
        crack = {"c_MM": c["c_MM"], "c_QQ": c_qq_eff, "c_MQ": c_mq}
        f = frequencies(beam, n_modes=n_modes, n_elem=n_elem, crack=crack,
                        xc_over_L=xc_over_L)
        rows.append({"a_bar": float(ab), "coupling": coupling,
                     "shear_flex": shear_flex, "convention": convention,
                     "c_MM": c["c_MM"], "c_QQ": c_qq, "c_MQ": c_mq,
                     **{f"ratio_f{i+1}": float(f[i] / f0[i]) for i in range(n_modes)}})
    return rows
