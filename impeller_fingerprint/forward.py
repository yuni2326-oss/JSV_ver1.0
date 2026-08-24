"""순방향 관측량 — 선형섭동(P 레일)과 비섭동 정확재해, 그리고 해석 야코비안.

설계서 §4 정의 동결:
    η̄_{m,n} = δλ/λ = −∫ γ_{m,n}(r) d(r) dr        (고유값 기반, 역식별이 소비하는 양)
    Δf/f     = ½ η̄                                 (주파수 표현)
정확재해는 손상 D(r)로 Rayleigh 고유값을 다시 풀어 η̄ = Λ_damaged/Λ_healthy − 1로 얻는다.
Λ는 무차원이므로 D·ρh(두께·물성)가 소거되고, 레일 기하 선택(설계서 §5.2)과 무관하게 성립한다.

야코비안은 해석적으로 준다(유한차분은 테스트에서 교차검증):
    ∂η̄/∂S̄_D = η̄/S̄_D                         (선형이므로 정확)
    ∂η̄/∂ξ_d = −∫ γ · d · [2(r−r_d)/w²] · (b−a) dr
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import kernels as ker
from . import severity as sev


def _damage_on(r: np.ndarray, xi_d: float, s_bar: float, w: float, plate) -> np.ndarray:
    return sev.damage_field_xi(r, xi_d=xi_d, s_bar=s_bar, w=w, a=plate.a, b=plate.b)


def eta_bar_linear(pool: Sequence[ker.ModeKernel], xi_d: float, s_bar: float,
                   w: float, plate) -> np.ndarray:
    """선형섭동 pair-mean η̄ (모드 풀 순서대로)."""
    out = []
    for k in pool:
        d = _damage_on(k.r, xi_d, s_bar, w, plate)
        out.append(-float(np.trapezoid(k.gamma * d, k.r)))
    return np.array(out)


#: 재료제거에서 질량손실과 강성손실의 결합비. 깊이비 p에 대해
#: d_K = 1−(1−p)³ ≈ 3p, d_M = p  ⇒  d_M = d_K/3 (1차)
#: **생산 계산은 이 값을 쓰지 않는다** — `coupling="exact"`(정확식)이 정본 규약이다(설계서 M8).
MASS_COUPLING = 1.0 / 3.0


def resolve_coupling(mass):
    """모듈 공통 `mass` 인자 → `coupling` 값 정규화 (설계서 M8).

    허용값:
      - ``"exact"``  : d_M = 1−(1−d_K)^{1/3}. **생산 규약**(A2·A3·A4·A5·B1 전부 이 값).
        균일두께판(레일 기하, 설계서 §5.2)에서 두께제거의 **정확한** 결합이다.
      - ``"sandwich"``: d_M = d_K/(1+d_K). 면판 2장 샌드위치의 **한쪽 면판**을 가공할 때의
        결합(설계서 §5.3 유도). 레일 결과를 실물 슈라우드로 이전할 때만 쓴다 —
        **생산 산출물은 이 법칙으로 계산하지 않았다**.
      - 숫자         : d_M = coupling · d_K (선형 결합비). 0이면 강성전용과 동일.
      - ``True``     : 하위호환 별칭 → ``MASS_COUPLING`` = 1/3 (p→0 선행차수 근사).
      - ``False``/``None`` : 질량항 없음(호출자가 강성전용 경로를 타야 한다).

    **왜 이 함수가 필요한가**: `True`가 모듈마다 다르게 해석되고 있었다(2026-08-05 감사).
    `identifiability`·`estimator`는 `MASS_COUPLING`=1/3으로 강등했지만
    `modeselect`·`validity`·`montecarlo`는 `coupling=True`를 그대로 넘겨
    `float(True)=1.0` → **d_M = d_K**(정확값의 3배)가 됐다. 같은 인자값이 세 가지 모델을
    뜻하는 상태였으므로(F20 ①과 동일한 사고 패턴) 정규화를 한 곳으로 모은다.
    """
    if mass is True:
        return MASS_COUPLING
    return mass


def mass_field_exact(d_K: np.ndarray) -> np.ndarray:
    """강성손실장 d_K에서 **정확한** 질량손실장 d_M을 얻는다(설계서 M8).

    깊이비 p에 대해 d_K = 1−(1−p)³, d_M = p ⇒ p = 1−(1−d_K)^{1/3}.
    1차 근사 d_M = d_K/3은 p=0.1/0.25/0.5에서 각각 11/30/71 % 과소평가하므로,
    생산 계산에는 이 정확식을 쓴다(`coupling="exact"`).
    """
    d = np.clip(np.asarray(d_K, dtype=float), 0.0, 1.0 - 1e-12)
    return 1.0 - (1.0 - d) ** (1.0 / 3.0)


def mass_field_sandwich(d_K: np.ndarray) -> np.ndarray:
    """**샌드위치**(면판 2장) 슈라우드의 한쪽 면판을 가공할 때의 질량손실장 (설계서 §5.3).

    면판 두께 t_f, 중심간 거리 s, 손상면판을 외면에서 p·t_f 제거하면 두 플랜지 단면의
    관성모멘트는 I = t₁t₂s²/(t₁+t₂)(자체관성 무시)이므로

        d_K = 1 − 2(1−p)/(2−p) = p/(2−p)      (**s에 무관**),   d_M = p/2

    즉 **d_M = d_K/(1+d_K)**. 얇은면판 극한의 1차 보정은
    d_K ≈ [p/(2−p)]·[1 + 2(1−p)·t_f/s]이지만, 결합법칙 d_M(d_K)는 그 보정을 받아도
    형태가 유지된다(둘 다 p의 함수이므로 s 의존이 상쇄되지 않는 것은 d_K(p)뿐이다).

    균일두께판(레일)의 정확결합 `mass_field_exact`와 비교하면 같은 d_K에서 질량손실이
    **2.6–3.0배 크다**(d_K = 0.1/0.01에서 2.63/2.96배) — 판 법칙 D ∝ t³ 대신
    샌드위치 법칙 D_eff ∝ t_face·s²(면판두께에 **선형**)이 성립하기 때문이다.
    """
    d = np.clip(np.asarray(d_K, dtype=float), 0.0, None)
    return d / (1.0 + d)


def sandwich_I(p, t_f: float, s: float):
    """면판 2장 단면의 **정확한** 단위폭 2차모멘트 I(p) — 면판 자체관성과 중립면 이동 포함.

    두 면(두께 t_f)이 중립면 간격 s로 떨어져 있고, 손상면을 **외면에서** p·t_f 제거한다.
    두 면적의 관성은 Σ A_i(z_i−z̄)² = (A₁A₂/(A₁+A₂))·d², d = 면 중심간 거리 = s − p·t_f/2:

        I(p) = (t_f³/12)[(1−p)³ + 1] + t_f·(1−p)/(2−p) · (s − p·t_f/2)²

    p=0에서 I₀ = t_f³/6 + t_f s²/2이고, t_f/s → 0에서 얇은면판 닫힌형 t_f s²/2로 수렴한다.
    """
    p = np.asarray(p, dtype=float)
    return ((t_f ** 3 / 12.0) * ((1.0 - p) ** 3 + 1.0)
            + t_f * (1.0 - p) / (2.0 - p) * (s - p * t_f / 2.0) ** 2)


def sandwich_dk_from_depth(p, t_f: float = None, s: float = None):
    """깊이비 p → 강성손실 d_K = 1 − I(p)/I(0) (as-built 정확 단면).

    기본 t_f·s는 **실측값**(geometry.T_SHEET, geometry.FACE_SEPARATION)이다. 실측 t_f/s = 0.196은
    얇은면판 극한이 아니므로 선행차수 d_K = p/(2−p)는 d_K를 19–39 % 과소평가한다(설계서 F60).
    """
    from . import geometry as _geo
    t_f = _geo.T_SHEET if t_f is None else t_f
    s = _geo.FACE_SEPARATION if s is None else s
    return 1.0 - sandwich_I(p, t_f, s) / float(sandwich_I(0.0, t_f, s))


def sandwich_depth_from_dk(d_K, t_f: float = None, s: float = None,
                           n_table: int = 200001):
    """d_K → 깊이비 p (위 함수의 역). 단조 표 보간 + Newton 2회로 1e-12 수준까지 조인다."""
    from . import geometry as _geo
    t_f = _geo.T_SHEET if t_f is None else t_f
    s = _geo.FACE_SEPARATION if s is None else s
    d = np.clip(np.asarray(d_K, dtype=float), 0.0, None)
    pg = np.linspace(0.0, 1.0, n_table)
    dg = sandwich_dk_from_depth(pg, t_f, s)
    p = np.interp(d, dg, pg)                       # dg는 p에 대해 단조증가
    for _ in range(2):                             # Newton 정련(해석 미분 대신 중심차분)
        h = 1e-7
        f = sandwich_dk_from_depth(p, t_f, s) - d
        fp = (sandwich_dk_from_depth(np.clip(p + h, 0, 1), t_f, s)
              - sandwich_dk_from_depth(np.clip(p - h, 0, 1), t_f, s)) / (2 * h)
        p = np.clip(p - np.where(np.abs(fp) > 0, f / fp, 0.0), 0.0, 1.0)
    return p


def mass_field_sandwich_asbuilt(d_K: np.ndarray) -> np.ndarray:
    """**as-built 샌드위치** 결합법칙 — 실측 t_f = 1.0 mm, s = 5.1 mm(설계서 F60).

    얇은면판 닫힌형 d_M = d_K/(1+d_K)는 t_f/s → 0 극한이고 **질량효과의 상계**다(정본 §3.3).
    실측 t_f/s = 0.196에서는 면판 자체관성과 중립면 이동 때문에 같은 p가 더 큰 d_K를 내므로,
    같은 d_K에서의 질량손실은 상계보다 작다: d_M = p(d_K)/2, ζ = d_M/d_K ≈ 0.63–0.70
    (얇은면판 0.75–0.97, 균일판 1/3).
    """
    return 0.5 * sandwich_depth_from_dk(d_K)


#: 문자열로 지정하는 비선형 결합법칙 — 숫자 결합비(d_M = c·d_K)와 구분한다.
#: `sandwich`는 얇은면판 극한(정본 §3.3의 공표 법칙), `sandwich_asbuilt`는 실측 t_f/s 반영.
MASS_LAWS = {"exact": mass_field_exact, "sandwich": mass_field_sandwich,
             "sandwich_asbuilt": mass_field_sandwich_asbuilt}


def eta_bar_linear_mass(pool: Sequence[ker.ModeKernel], xi_d: float, s_bar: float,
                        w: float, plate, coupling=MASS_COUPLING) -> np.ndarray:
    """**질량항을 포함한** 1차 섭동 pair mean (설계서 F12).

        η̄_m = −∫ γ^K_m d_K dr + ∫ γ^M_m d_M dr,   d_M = coupling · d_K

    가공포켓처럼 강성·질량이 같은 깊이비로 묶이면 coupling = 1/3(1차)이다. 균열처럼 질량이
    거의 안 줄면 coupling = 0 → 기존 강성전용 맵과 동일.
    **림에서는 γ^M ≫ γ^K이라 η̄가 양수가 될 수 있다** — 강성전용 맵이 표현하지 못하던 영역.
    """
    out = []
    for k in pool:
        d = _damage_on(k.r, xi_d, s_bar, w, plate)
        gm = k.gamma_mass
        if gm is None:
            raise ValueError("kernel에 gamma_mass가 없다 — mode_kernel을 다시 만들 것")
        stiff = float(np.trapezoid(k.gamma * d, k.r))
        if isinstance(coupling, str):
            mass = float(np.trapezoid(gm * MASS_LAWS[coupling](d), k.r))
        else:
            mass = float(coupling) * float(np.trapezoid(gm * d, k.r))
        out.append(-stiff + mass)
    return np.array(out)


def jacobian_linear_mass(pool: Sequence[ker.ModeKernel], xi_d: float, s_bar: float,
                         w: float, plate, coupling=MASS_COUPLING) -> np.ndarray:
    """질량항 포함 맵의 야코비안 (n_modes, 2).

    coupling이 문자열 법칙("exact"·"sandwich")이면 맵이 S̄에 비선형이므로 중심차분으로 계산한다.
    """
    if isinstance(coupling, str):
        h_xi, h_s = 1e-5, max(1e-6, 1e-3 * s_bar)
        f = lambda x, sb: eta_bar_linear_mass(pool, x, sb, w, plate, coupling=coupling)
        return np.stack([(f(xi_d + h_xi, s_bar) - f(xi_d - h_xi, s_bar)) / (2 * h_xi),
                         (f(xi_d, s_bar + h_s) - f(xi_d, s_bar - h_s)) / (2 * h_s)], axis=1)
    L = plate.b - plate.a
    r_d = sev.xi_to_r(xi_d, plate.a, plate.b)
    col_xi, col_s = [], []
    for k in pool:
        d1 = _damage_on(k.r, xi_d, 1.0, w, plate)
        gk, gm = k.gamma, k.gamma_mass
        col_s.append(-float(np.trapezoid(gk * d1, k.r))
                     + coupling * float(np.trapezoid(gm * d1, k.r)))
        dd = d1 * (2.0 * (k.r - r_d) / w ** 2)
        col_xi.append((-float(np.trapezoid(gk * dd, k.r))
                       + coupling * float(np.trapezoid(gm * dd, k.r))) * L * s_bar)
    return np.stack([np.array(col_xi), np.array(col_s)], axis=1)


def rel_freq_shift_linear(pool: Sequence[ker.ModeKernel], xi_d: float, s_bar: float,
                          w: float, plate) -> np.ndarray:
    """상대 주파수이동 Δf/f = ½η̄ (논문1 `forward_shifts`와 같은 규약)."""
    return 0.5 * eta_bar_linear(pool, xi_d, s_bar, w, plate)


def jacobian_linear(pool: Sequence[ker.ModeKernel], xi_d: float, s_bar: float,
                    w: float, plate) -> np.ndarray:
    """(n_modes, 2) 해석 야코비안 ∂η̄/∂(ξ_d, S̄_D)."""
    L = plate.b - plate.a
    r_d = sev.xi_to_r(xi_d, plate.a, plate.b)
    col_xi, col_s = [], []
    for k in pool:
        d_unit = _damage_on(k.r, xi_d, 1.0, w, plate)      # S̄=1 기준장(선형)
        col_s.append(-float(np.trapezoid(k.gamma * d_unit, k.r)))
        dd_drd = d_unit * (2.0 * (k.r - r_d) / w ** 2)
        col_xi.append(-float(np.trapezoid(k.gamma * dd_drd, k.r)) * L * s_bar)
    return np.stack([np.array(col_xi), np.array(col_s)], axis=1)


def eta_bar_exact(plate, modes: Iterable[tuple[int, int]], xi_d: float, s_bar: float,
                  w: float, n_trial: int = 8, n_grid: int = 4001,
                  mass=False) -> np.ndarray:
    """비섭동 정확재해 η̄ — 손상 D(r)로 고유값을 다시 풀어 Λ 비에서 얻는다.

    반경손상은 축대칭이므로 절점직경 m 족 안에서 모드 순서가 유지된다(반경차수 n으로 색인).
    방위 국소 손상(포켓)의 정확해는 `rail2d`/`rail3d`가 담당한다.

    `mass`: 거짓이면 강성전용. ``"sandwich"``/``"sandwich_asbuilt"``면 해당 샌드위치 결합법칙,
    그 밖의 참값은 균일판 정확결합(`mass_field_exact`) — **기존 산출물과 비트단위 동일**하게
    유지하기 위한 규약이다(숫자 결합비를 줘도 정확재해는 정확결합을 쓴다; 생산은 "exact"뿐이다).
    """
    modes = list(modes)
    if s_bar == 0.0:
        return np.zeros(len(modes))

    def damage(r: np.ndarray) -> np.ndarray:
        return _damage_on(r, xi_d, s_bar, w, plate)

    # 문자열 법칙은 이름으로 조회한다 — `mass == "sandwich"`만 걸러내던 옛 규약은
    # 새 법칙이 조용히 정확결합으로 폴백해 선형화오차를 4배로 부풀렸다(F20 계열 사고).
    law = MASS_LAWS.get(mass, mass_field_exact) if isinstance(mass, str) \
        else mass_field_exact
    dmg_m = (lambda r: law(damage(r))) if mass else None

    n_by_m: dict[int, int] = {}
    for m, n in modes:
        n_by_m[m] = max(n_by_m.get(m, 0), n)
    healthy, damaged = {}, {}
    for m, n_max in n_by_m.items():
        healthy[m] = ker.solve_eigenvalues_props(
            plate.a, plate.b, plate.nu, m, n_modes=n_max + 1,
            n_trial=n_trial, n_grid=n_grid)   # 건전
        damaged[m] = ker.solve_eigenvalues_props(
            plate.a, plate.b, plate.nu, m, n_modes=n_max + 1, damage=damage,
            n_trial=n_trial, n_grid=n_grid, damage_mass=dmg_m)
    return np.array([damaged[m][n] / healthy[m][n] - 1.0 for m, n in modes])


# ---------------------------------------------------------------------------
# 직사각 반경 밴드(= 포켓의 축대칭 극한) — 설계서 §3.6-iii의 오차 3분해에 쓰는 손상족.
#
# 왜 밴드인가: 방위국소 포켓에는 Kirchhoff **비섭동** 해가 없다(2D 솔버 미구현, F23).
# 밴드는 축대칭이므로 같은 손상형상을 (a) 섭동 선형맵, (b) Kirchhoff 정확재해,
# (c) 3D 솔리드 세 경로로 **모두** 풀 수 있어 선형화오차와 모델형식오차를 분리할 수 있다.
# ---------------------------------------------------------------------------

def band_d_K(depth_frac: float) -> float:
    """깊이비 p → 밴드 안의 굽힘강성 손실률 d_K = 1−(1−p)³ (균일두께판 법칙)."""
    return 1.0 - (1.0 - float(depth_frac)) ** 3


def band_s_bar(r1: float, r2: float, depth_frac: float, extent: float) -> float:
    """밴드 [r1,r2]·깊이비 p의 심각도 S̄_D = d_K·(r2−r1)/(b−a)."""
    return band_d_K(depth_frac) * (float(r2) - float(r1)) / float(extent)


def band_depth_for_s_bar(s_bar: float, r1: float, r2: float, extent: float) -> float:
    """목표 S̄_D를 내는 깊이비 p (역함수). 0<p<1 밖이면 ValueError."""
    d_K = float(s_bar) * float(extent) / (float(r2) - float(r1))
    if not 0.0 < d_K < 1.0:
        raise ValueError(f"밴드 폭으로 S̄={s_bar} 불가 (d_K={d_K:.3f})")
    return 1.0 - (1.0 - d_K) ** (1.0 / 3.0)


def _band_fields(r: np.ndarray, r1: float, r2: float, depth_frac: float,
                 coupling="exact"):
    d_K = band_d_K(depth_frac)
    d_M = (float(MASS_LAWS[coupling](np.array([d_K]))[0]) if isinstance(coupling, str)
           else float(coupling) * d_K)
    inb = (np.asarray(r) >= r1) & (np.asarray(r) <= r2)
    return np.where(inb, d_K, 0.0), np.where(inb, d_M, 0.0)


def eta_bar_linear_band(pool: Sequence[ker.ModeKernel], r1: float, r2: float,
                        depth_frac: float, plate, coupling="exact",
                        n_sub: int = 4001) -> np.ndarray:
    """밴드 손상의 **1차 섭동** pair mean. 계단함수 적분오차를 없애려고 [r1,r2] 위에서만
    커널을 보간해 적분한다(격자 절점과 밴드 경계가 어긋나도 O(h²))."""
    d_K = band_d_K(depth_frac)
    d_M = (float(MASS_LAWS[coupling](np.array([d_K]))[0]) if isinstance(coupling, str)
           else float(coupling) * d_K)
    rr = np.linspace(r1, r2, n_sub)
    out = []
    for k in pool:
        gk = float(np.trapezoid(np.interp(rr, k.r, k.gamma), rr))
        gm = float(np.trapezoid(np.interp(rr, k.r, k.gamma_mass), rr))
        out.append(-d_K * gk + d_M * gm)
    return np.array(out)


def eta_bar_exact_band(plate, modes: Iterable[tuple[int, int]], r1: float, r2: float,
                       depth_frac: float, coupling="exact", n_trial: int = 36,
                       n_grid: int = 8001, basis: str = "legendre") -> np.ndarray:
    """밴드 손상의 **비섭동** Kirchhoff 정확재해 η̄ = Λ_damaged/Λ_healthy − 1.

    급격한 계단형 손상이라 시행함수가 많이 필요하다. 실측 수렴(ξ=0.6·p=0.5, 최악 셀):
    n_trial 12→20에서 22 %, 20→28에서 2.9 %, 28→36에서 1.8 %, **36→44에서 0.14 %**.
    단항 기저는 n_trial ≳ 12에서 Cholesky가 깨지므로 르장드르 기저 + 전처리를 기본으로
    쓰고 n_trial 기본값을 36으로 둔다. n_trial=12로 계산하면 최악 셀에서 η̄가 24 %
    작게 나와 모델형식 오차를 두 배로 부풀린다(2026-08-09 확인).
    `basis="monomial"`로 두면 생산 규약과 같은 기저를 쓴다(테스트가 두 기저의 일치를 검정)."""
    modes = list(modes)

    def dmg(r):
        return _band_fields(r, r1, r2, depth_frac, coupling)[0]

    def dmg_m(r):
        return _band_fields(r, r1, r2, depth_frac, coupling)[1]

    n_by_m: dict[int, int] = {}
    for m, n in modes:
        n_by_m[m] = max(n_by_m.get(m, 0), n)
    healthy, damaged = {}, {}
    for m, n_max in n_by_m.items():
        healthy[m] = ker.solve_eigenvalues_props(
            plate.a, plate.b, plate.nu, m, n_modes=n_max + 1, n_trial=n_trial,
            n_grid=n_grid, precondition=True, basis=basis)
        damaged[m] = ker.solve_eigenvalues_props(
            plate.a, plate.b, plate.nu, m, n_modes=n_max + 1, damage=dmg,
            n_trial=n_trial, n_grid=n_grid, damage_mass=dmg_m, precondition=True,
            basis=basis)
    return np.array([damaged[m][n] / healthy[m][n] - 1.0 for m, n in modes])


def forward_exact_freq(plate, modes: Iterable[tuple[int, int]], xi_d: float,
                       s_bar: float, w: float, n_trial: int = 8,
                       n_grid: int = 4001) -> np.ndarray:
    """정확재해 상대 주파수이동 Δf/f = √(1+η̄) − 1 (1차에서 ½η̄와 일치)."""
    eta = eta_bar_exact(plate, modes, xi_d, s_bar, w, n_trial=n_trial, n_grid=n_grid)
    return np.sqrt(1.0 + eta) - 1.0
