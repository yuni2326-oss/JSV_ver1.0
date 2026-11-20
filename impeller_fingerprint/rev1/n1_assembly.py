"""N1 — C6 조립체 수준 민감도 커널·Fisher 정보·D-최적 모드선택 (R1-P2, R2-Q3 대응).

심사자 요구(R1-P2): "Repeat the D-optimal mode selection under the C6 constraint using
assembly-level sensitivity kernels". 레일(축대칭 균일 Kirchhoff 판)의 커널 γ_m 대신 **조립체
모드 자체**의 손상 민감도로 Fisher 정보를 다시 만든다.

모델 — 정확한 C6 구조격자(`impeller_hex.mesh`, n_θ가 6의 배수) + sfepy 2차 요소:
  · 사면체 CAD(b5)와 달리 격자 자체가 C6이므로 보호된 doublet이 **정확히 축퇴**한다(건전 분리 = 0,
    상대 1e-12). 축대칭 손상은 C6을 보존하므로 손상 후에도 축퇴가 유지된다.
  · 지지: 기본은 **내측 모서리 클램프**(r = a의 모든 절점) — 레일의 클램프-자유 환형판과 같은
    지지 이상화. 레일→조립체 이전에서 지지를 고정하고 구조(두 면판 + 베인 + C6)만 바꾼다.
  · 손상: 레일과 **같은 물리 손상**. 단면 강성손실 d_K(r)를 Gaussian(ξ_d, S̄_D, w)로 두고
    실측 2면판 정확식(`forward.sandwich_depth_from_dk`, 부록 A)으로 판재 깊이비 p(r)로 바꾼 뒤,
    전면 슈라우드 **바깥면을 실제로 깎는다**(절점 변형, 위상 불변). 질량·강성이 함께 줄고
    중립면 이동까지 3D가 스스로 계산한다 — 결합법칙을 가정하지 않는다.

관측량 — 대칭이 정한 대로(정본 §3.2): h = 1, 2 doublet은 **쌍평균**, h = 0, 3 singlet은 각각
독립 관측량(쌍으로 묶지 않는다 — 받은 계획서 §2.6). η = λ_d/λ_h − 1(고유치 기반, 정본 규약),
잡음 floor는 쌍평균 2c, singlet 2√2c.

1차 민감도: 건전 질량정규 모드 φ에 대해 δλ = φᵀ(δK − λδM)φ (Hellmann–Feynman). δK·δM은
감육 형상의 행렬 차(중심차분 ±p)로 얻으므로 **고유치 재해석 없이 모든 모드의 커널**이 나온다.
유한 손상의 비선형 야코비안은 재해석 유한차분으로 따로 확인한다.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.sparse.linalg import eigsh

from .. import forward as fwd
from .. import geometry as geo
from .. import impeller_hex as ih
from .. import rail3d as r3
from .. import severity as sev
from . import n1_mesh as nm

E_SUS, NU_SUS, RHO_ASM = 193e9, 0.29, 7930.0      # 부록 A.2: 조립체 ρ = 7930
W_GAUSS = 3.0e-3


def asm_spec(**kw) -> ih.HexImpellerSpec:
    """(구) `impeller_hex` 균등격자 스펙 — 베인이 격자에 따라 끊길 수 있다(n1_mesh 문서). 비교용."""
    base = dict(n_r=18, n_theta=72, n_z_shroud=2, n_z_channel=2, wrap_deg=60.0,
                rho=RHO_ASM, E=E_SUS, nu=NU_SUS)
    base.update(kw)
    return ih.HexImpellerSpec(**base)


def fit_spec(**kw) -> nm.FitSpec:
    """N1/N2 기본 — 캠버 정합 격자(베인 두께 1.0 mm 정확·연속·정확 C6), as-built CAD 캠버."""
    return nm.FitSpec(**kw)


# ---------------------------------------------------------------- 기하·손상
@dataclass
class AsmMesh:
    spec: object               # nm.FitSpec 또는 ih.HexImpellerSpec
    coors: np.ndarray
    conn: np.ndarray
    info: dict
    r_v: np.ndarray            # 꼭짓점 반경
    top: np.ndarray            # 전면 슈라우드(손상면판) 꼭짓점 마스크
    z_in: float                # 전면 슈라우드 안쪽면 z(베인 접합면, 고정)
    t_f: float
    hspec: ih.HexImpellerSpec = None   # 대칭 판정 함수용 색인 스펙
    t_o: float | None = None   # 반대면(뒤판) 두께 — 비대칭 기하(G1)일 때만, 아니면 None
    gap: float | None = None   # 두 면판 안쪽면 간격 — 비대칭 기하일 때만


def _t_back(spec) -> float:
    return spec.tb if hasattr(spec, "tb") else spec.t_sheet


def build(spec) -> AsmMesh:
    if isinstance(spec, nm.FitSpec):
        coors, conn, info = nm.mesh(spec)
        hspec = spec.hex_spec()
    else:
        coors, conn, info = ih.mesh(spec)
        hspec = spec
    z_in = _t_back(spec) + spec.channel
    top = coors[:, 2] >= z_in - 1e-12
    r_v = np.hypot(coors[:, 0], coors[:, 1])
    if getattr(spec, "asymmetric", False):
        return AsmMesh(spec, coors, conn, info, r_v, top, z_in, spec.tf, hspec,
                       t_o=spec.tb, gap=spec.channel)
    return AsmMesh(spec, coors, conn, info, r_v, top, z_in, spec.t_sheet, hspec)


def physical_coors(am, X: np.ndarray) -> np.ndarray:
    """평판 기준 좌표 → 조립 좌표. G4 테이퍼면 `FitSpec.taper_coors`, 아니면 그대로(옛 `impeller_hex` 스펙 포함)."""
    tc = getattr(am.spec, "taper_coors", None)
    if tc is None:                                   # N6은 `spec.base`(FitSpec)가 테이퍼를 안다
        tc = getattr(getattr(am.spec, "base", None), "taper_coors", None)
    return X if tc is None else tc(X, am.info["grid_idx"])


def depth_from_dk_local(dk: np.ndarray, r: np.ndarray, t_d: float, t_o: float, channel_at) -> np.ndarray:
    """d_K → 깊이비 p를 그 반지름의 간격 c(r)로 푼다(두 면판 식, 반지름별) — G4 테이퍼용.

    d_K = 0인 절점은 p = 0이다. 면판을 다 깎아도 닿지 않는 d_K면 `depth_profile_am`과 같은 오류를 낸다.
    """
    from . import g1_section as g1
    dk = np.asarray(dk, float)
    out = np.zeros_like(dk)
    rr = np.round(np.asarray(r, float), 9)
    for ru in np.unique(rr[dk != 0]):
        sel = (rr == ru) & (dk != 0)
        c = float(channel_at(ru))
        dk_max = float(g1.dk_from_depth(1.0, t_d, t_o, c))
        if np.any(dk[sel] >= dk_max):
            raise ValueError(f"peak d_K {dk[sel].max():.3f} ≥ {dk_max:.3f} (r = {ru * 1e3:.2f} mm): 면판 전체 제거로도 도달 불가")
        out[sel] = g1.depth_from_dk(dk[sel], t_d, t_o, c)
    return out


def depth_profile_am(am: AsmMesh, r: np.ndarray, xi_d: float, s_bar: float, w: float):
    """`am`의 기하로 d_K(r) → p(r) — 같은 두께 두 면판이면 기존 식, 비대칭이면 G1 식(G4 테이퍼면 반지름별 간격)."""
    spec = am.spec
    if getattr(spec, "tapered", False):
        dk = dK_profile(r, xi_d, s_bar, w, spec.a, spec.b)
        return depth_from_dk_local(dk, r, am.t_f, am.t_o if am.t_o is not None else spec.tb, spec.channel_at)
    if am.t_o is None:
        return depth_profile(r, xi_d, s_bar, w, spec.a, spec.b, am.t_f,
                             spec.t_sheet + spec.channel)
    from . import g1_section as g1
    dk = dK_profile(r, xi_d, s_bar, w, spec.a, spec.b)
    dk_max = float(g1.dk_from_depth(1.0, am.t_f, am.t_o, am.gap))
    if np.any(dk >= dk_max):
        raise ValueError(f"peak d_K {dk.max():.3f} ≥ {dk_max:.3f}: 면판 전체 제거로도 도달 불가")
    return g1.depth_from_dk(dk, am.t_f, am.t_o, am.gap)


def dK_profile(r: np.ndarray, xi_d: float, s_bar: float, w: float, a: float,
               b: float) -> np.ndarray:
    """정본 Gaussian 단면 강성손실(부록 A.3, 절단·비정규화)."""
    return sev.damage_field_xi(np.asarray(r, float), xi_d=xi_d, s_bar=s_bar, w=w, a=a, b=b)


def depth_profile(r: np.ndarray, xi_d: float, s_bar: float, w: float, a: float, b: float,
                  t_f: float, s: float) -> np.ndarray:
    """d_K(r) → 판재 깊이비 p(r) (실측 2면판 정확식의 역). d_K ≥ 1 − I(1)/I(0)이면 정의 불가."""
    dk = dK_profile(r, xi_d, s_bar, w, a, b)
    dk_max = float(fwd.sandwich_dk_from_depth(1.0, t_f, s))
    if np.any(dk >= dk_max):
        raise ValueError(f"peak d_K {dk.max():.3f} ≥ {dk_max:.3f}: 면판 전체 제거로도 도달 불가")
    return fwd.sandwich_depth_from_dk(dk, t_f, s)


def morph(am: AsmMesh, p_v: np.ndarray) -> np.ndarray:
    """전면 슈라우드 바깥면을 p·t_f만큼 깎은 꼭짓점 좌표. 안쪽면(베인 접합)은 고정.

    면판 안 절점은 두께방향으로 비례 압축한다: z' = z_in + (z − z_in)(1 − p(r)).
    p < 0을 주면 같은 식이 두께를 **더한다** — 중심차분 민감도에 쓴다.
    """
    c = am.coors.copy()
    sel = am.top
    c[sel, 2] = am.z_in + (c[sel, 2] - am.z_in) * (1.0 - p_v[sel])
    return c


def removed_volume_exact(am: AsmMesh, p_of_r) -> float:
    """검증용: 2π∫ p(r) t_f r dr (축대칭 감육의 해석 제거체적)."""
    r = np.linspace(am.spec.a, am.spec.b, 20001)
    return float(2 * math.pi * np.trapezoid(p_of_r(r) * am.t_f * r, r))


# ---------------------------------------------------------------- FEM
@dataclass
class KM:
    K: object                  # 축약(구속 제외) 강성
    M: object
    make_full: object          # 축약 벡터 → 전체 벡터(절점×3, 구속 0 포함)
    n_vertex: int
    volume: float


def assemble(am: AsmMesh, coors: np.ndarray | None = None, order: int = 2) -> KM:
    """sfepy 조립(rail3d.solve_modes와 같은 항·적분차수) — 행렬만 돌려준다."""
    from sfepy.base.base import output
    from sfepy.discrete import (Equation, Equations, FieldVariable, Function, Functions,
                                Integral, Material, Problem)
    from sfepy.discrete.conditions import Conditions, EssentialBC
    from sfepy.discrete.fem import FEDomain, Field, Mesh
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.terms import Term

    output.set_output(quiet=True)
    spec = am.spec
    X = physical_coors(am, am.coors if coors is None else coors)
    mesh = Mesh.from_data("asm", X, None, [am.conn], [np.zeros(len(am.conn), np.int32)],
                          [r3.elem_desc(X, am.conn)])
    domain = FEDomain("d", mesh)
    omega = domain.create_region("Omega", "all")
    clamp = spec.clamp_fn() if hasattr(spec, "clamp_fn") else r3.clamp_inner_rim(spec.a)
    fns = Functions([Function("fixed_fn", lambda coors, domain=None: clamp(coors))])
    domain.create_region("Fixed", "vertices by fixed_fn", "facet", functions=fns)
    fld = Field.from_args("fu", np.float64, "vector", omega, approx_order=order)
    u = FieldVariable("u", "unknown", fld)
    v = FieldVariable("v", "test", fld, primary_var_name="u")
    mat = Material("m", D=stiffness_from_youngpoisson(3, spec.E, spec.nu), rho=spec.rho)
    integral = Integral("i", order=2 * order)
    eq_k = Equation("K", Term.new("dw_lin_elastic(m.D, v, u)", integral, omega, m=mat, v=v, u=u))
    eq_m = Equation("M", Term.new("dw_dot(m.rho, v, u)", integral, omega, m=mat, v=v, u=u))
    pb = Problem("asm", equations=Equations([eq_k, eq_m]))
    pb.time_update(ebcs=Conditions([EssentialBC("Fixed", domain.regions["Fixed"],
                                                {"u.all": 0.0})]))
    pb.update_materials()
    K = eq_k.evaluate(mode="weak", dw_mode="matrix", asm_obj=pb.mtx_a)
    M0 = K.copy()
    M0.data[:] = 0.0
    M = eq_m.evaluate(mode="weak", dw_mode="matrix", asm_obj=M0)
    variables = pb.get_variables()
    vol = float(omega.domain.cmesh.get_volumes(3).sum())

    def make_full(vec):
        return variables.make_full_vec(vec)

    return KM(K.tocsc(), M.tocsc(), make_full, X.shape[0], vol)


def assemble_region(am: AsmMesh, cell_mask: np.ndarray, order: int = 2) -> KM:
    """`cell_mask` 셀만의 K_R·M_R — 전체 조립과 **같은 자유도 번호**(같은 구속)로 낸다.

    φᵀK_Rφ/λ = 그 구역의 모드 변형에너지 분율(국소 강성손실 δlnK에 대한 정확한 1차 감도).
    """
    from sfepy.base.base import output
    from sfepy.discrete import (Equation, Equations, FieldVariable, Function, Functions,
                                Integral, Material, Problem)
    from sfepy.discrete.conditions import Conditions, EssentialBC
    from sfepy.discrete.fem import FEDomain, Field, Mesh
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.terms import Term

    output.set_output(quiet=True)
    spec = am.spec
    mask = np.asarray(cell_mask, bool)
    X = physical_coors(am, am.coors)
    mesh = Mesh.from_data("asmR", X, None, [am.conn], [mask.astype(np.int32)],
                          [r3.elem_desc(X, am.conn)])
    domain = FEDomain("d", mesh)
    omega = domain.create_region("Omega", "all")
    rgn = domain.create_region("Rgn", "cells of group 1", "cell")
    clamp = spec.clamp_fn() if hasattr(spec, "clamp_fn") else r3.clamp_inner_rim(spec.a)
    fns = Functions([Function("fixed_fn", lambda coors, domain=None: clamp(coors))])
    domain.create_region("Fixed", "vertices by fixed_fn", "facet", functions=fns)
    fld = Field.from_args("fu", np.float64, "vector", omega, approx_order=order)
    u = FieldVariable("u", "unknown", fld)
    v = FieldVariable("v", "test", fld, primary_var_name="u")
    mat = Material("m", D=stiffness_from_youngpoisson(3, spec.E, spec.nu), rho=spec.rho)
    integral = Integral("i", order=2 * order)
    eq_k = Equation("K", Term.new("dw_lin_elastic(m.D, v, u)", integral, omega, m=mat, v=v, u=u))
    eq_kr = Equation("Kr", Term.new("dw_lin_elastic(m.D, v, u)", integral, rgn, m=mat, v=v, u=u))
    eq_mr = Equation("Mr", Term.new("dw_dot(m.rho, v, u)", integral, rgn, m=mat, v=v, u=u))
    pb = Problem("asmR", equations=Equations([eq_k, eq_kr, eq_mr]))
    pb.time_update(ebcs=Conditions([EssentialBC("Fixed", domain.regions["Fixed"],
                                                {"u.all": 0.0})]))
    pb.update_materials()
    K = eq_k.evaluate(mode="weak", dw_mode="matrix", asm_obj=pb.mtx_a)
    KR = K.copy()
    KR.data[:] = 0.0
    KR = eq_kr.evaluate(mode="weak", dw_mode="matrix", asm_obj=KR)
    MR = K.copy()
    MR.data[:] = 0.0
    MR = eq_mr.evaluate(mode="weak", dw_mode="matrix", asm_obj=MR)
    variables = pb.get_variables()
    return KM(KR.tocsc(), MR.tocsc(), lambda vec: variables.make_full_vec(vec),
              am.coors.shape[0], float("nan"))


def vane_root_mask(am: AsmMesh, k: int = 0, span=(0.0, 0.25)) -> np.ndarray:
    """베인 k의 뿌리(입구) 창 셀 — b5 `impeller_cad.damaged_cell_mask`와 같은 정의(스팬 0–25 %)."""
    spec = am.spec
    if "cell_vane" in am.info:          # 캠버 정합 격자: 셀 속성으로 정확히 고른다
        cv, cs = am.info["cell_vane"], am.info["cell_span"]
        return (cv == k) & (cs >= span[0]) & (cs <= span[1])
    c = am.coors[am.conn].mean(axis=1)
    rr = np.hypot(c[:, 0], c[:, 1])
    zc = c[:, 2]
    f_r = (rr - spec.a) / (spec.b - spec.a)
    in_chan = (zc > spec.t_sheet) & (zc < spec.t_sheet + spec.channel)
    th_k = 2 * math.pi * k / spec.n_vane + math.radians(spec.wrap_deg) * f_r
    d = np.angle(np.exp(1j * (np.arctan2(c[:, 1], c[:, 0]) - th_k)))
    near = np.abs(d) < math.pi / spec.n_vane          # 베인 k의 섹터(다른 베인 배제)
    return in_chan & near & (f_r >= span[0]) & (f_r <= span[1])


def energy_fractions(md: "Modes", obs: list, kmr: KM) -> tuple[np.ndarray, np.ndarray]:
    """관측량별 구역 변형에너지·운동에너지 분율(축퇴군은 자취 평균)."""
    ef = np.empty(len(obs))
    kf = np.empty(len(obs))
    for j, o in enumerate(obs):
        es, ks = [], []
        for i in o.members:
            ph = md.phi[:, i]
            es.append(float(ph @ (kmr.K @ ph)) / md.lam[i])
            ks.append(float(ph @ (kmr.M @ ph)))           # φ는 질량정규
        ef[j], kf[j] = float(np.mean(es)), float(np.mean(ks))
    return ef, kf


@dataclass
class Modes:
    lam: np.ndarray            # 고유치 (rad/s)²
    freqs: np.ndarray
    phi: np.ndarray            # 축약 질량정규 고유벡터 (ndof_red, n)
    vshape: np.ndarray         # 꼭짓점 변위 (n, n_vertex*3)
    ndof: int
    seconds: float


def solve(km: KM, n_modes: int, v0_seed: int = 20261006,
          permc_spec: str | None = "MMD_AT_PLUS_A") -> Modes:
    """shift-invert(σ = 0) Lanczos. K는 클램프로 양정치이므로 σ = 0에서 바로 인수분해한다.

    `permc_spec="MMD_AT_PLUS_A"`(+SymmetricMode)는 대칭 FEM 행렬에서 scipy 기본(COLAMD)보다
    채움이 적어 인수분해가 빠르다(12×72에서 측정). None이면 scipy 기본 경로.
    """
    t0 = time.time()
    v0 = np.random.default_rng(v0_seed).standard_normal(km.K.shape[0])
    if permc_spec:
        from scipy.sparse.linalg import LinearOperator, splu
        opts = {"SymmetricMode": True} if permc_spec == "MMD_AT_PLUS_A" else {}
        lu = splu(km.K.tocsc(), permc_spec=permc_spec, options=opts)
        op = LinearOperator(km.K.shape, matvec=lu.solve, dtype=float)
        vals, vecs = eigsh(km.K, k=n_modes, M=km.M, sigma=0.0, which="LM", v0=v0, OPinv=op)
    else:
        vals, vecs = eigsh(km.K, k=n_modes, M=km.M, sigma=0.0, which="LM", v0=v0)
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]
    # 질량정규화(ARPACK은 M-정규를 보장하지만 명시적으로 맞춘다)
    mn = np.sqrt(np.einsum("ij,ij->j", vecs, km.M @ vecs))
    vecs = vecs / mn
    nv = km.n_vertex
    vsh = np.stack([km.make_full(vecs[:, k]).reshape(-1, 3)[:nv].ravel()
                    for k in range(vecs.shape[1])])
    return Modes(vals, np.sqrt(np.maximum(vals, 0)) / (2 * math.pi), vecs, vsh,
                 km.K.shape[0], time.time() - t0)


# ---------------------------------------------------------------- 분류·관측량
@dataclass
class Obs:
    """대칭 관측량 하나 = 축퇴군 하나(doublet 쌍평균 또는 singlet)."""
    label: str
    h: int
    deg: int
    members: list
    f_Hz: float
    m_dom: int
    n_rad: int
    vane_share: float
    sigma_factor: float        # 잡음 floor / c  (쌍평균 2, singlet 2√2)
    resolvable: bool = True
    cluster: int = -1


def classify(am: AsmMesh, md: Modes, rel_tol: float = 1e-7, m_max: int = 9,
             res_gap: float = 2e-3) -> list[Obs]:
    """건전 모드 → 대칭 관측량 목록(주파수가 아니라 형상으로 h·m·n을 판정).

    `res_gap`: 서로 다른 관측량의 상대 주파수 간격이 이보다 작으면 **분해 불가 군**으로 표시한다
    (예: 베인 6매의 국부모드는 h = 0,1,2,3이 1e-4 안에 몰린다). 측정 풀에서 제외하고 보고한다.
    """
    spec = am.spec
    hs = am.hspec
    res = r3.ModalResult(freqs=md.freqs, shapes=None, coors=am.coors, ndof=md.ndof,
                         field_coors=am.coors, full_shapes=md.vshape)
    h, c, deg, gid = ih.harmonic_indices_grouped(hs, res, am.info["grid_idx"], None,
                                                 rel_tol=rel_tol)
    # 지배 방위차수·반경차수: 전면 슈라우드 u_z의 링별 이산 푸리에
    prof = {m: ih.radial_harmonic_profiles(hs, res, am.info["grid_idx"], m, z_frac=0.3)
            for m in range(m_max + 1)}
    zc = am.coors[:, 2]
    tb = _t_back(spec)
    vane_nodes = (zc > tb + 1e-9) & (zc < tb + spec.channel - 1e-9)
    out: list[Obs] = []
    for g in np.unique(gid):
        idx = [int(i) for i in np.nonzero(gid == g)[0]]
        hh = int(round(float(h[idx[0]])))
        en = np.array([sum(np.sum(np.abs(prof[m][i]) ** 2) for i in idx)
                       for m in range(m_max + 1)])
        mdom = int(np.argmax(en))
        n_rad = ih.radial_order(prof[mdom][idx[0]])
        U = md.vshape[idx].reshape(len(idx), -1, 3)
        tot = float(np.sum(U ** 2))
        vs = float(np.sum(U[:, vane_nodes] ** 2) / tot) if tot > 0 else 0.0
        d = len(idx)
        if d not in (1, 2):
            raise RuntimeError(f"축퇴군 크기 {d} (f={md.freqs[idx]}) — rel_tol 확인")
        if d == 1 and hh not in (0, spec.n_vane // 2):
            # 보호 doublet인데 짝이 없다 — 계산한 스펙트럼의 **맨 위**에서 짝이 잘린 경우만 허용
            if max(idx) == len(md.freqs) - 1:
                continue
            raise RuntimeError(f"h={hh} 단일 모드가 스펙트럼 중간에 있다 (f={md.freqs[idx]})")
        if d == 2 and hh in (0, spec.n_vane // 2):
            raise RuntimeError(f"h={hh}인데 2중축퇴 — 우연축퇴 또는 분류 오류 (f={md.freqs[idx]})")
        sf = 2.0 if d == 2 else 2.0 * math.sqrt(2.0)
        out.append(Obs(label=f"h{hh}:m{mdom}n{n_rad}", h=hh, deg=d, members=idx,
                       f_Hz=float(np.mean(md.freqs[idx])), m_dom=mdom, n_rad=n_rad,
                       vane_share=vs, sigma_factor=sf))
    # 같은 라벨 중복 → 순번 접미사(주파수 순)
    seen: dict = {}
    for o in out:
        k = seen.get(o.label, 0)
        seen[o.label] = k + 1
        if k:
            o.label = f"{o.label}#{k + 1}"
    # 분해 불가 군
    fs = np.array([o.f_Hz for o in out])
    order = np.argsort(fs)
    cl = 0
    for a_i, b_i in zip(order[:-1], order[1:]):
        if abs(fs[b_i] - fs[a_i]) / fs[a_i] < res_gap:
            ca = out[a_i].cluster if out[a_i].cluster >= 0 else cl
            if out[a_i].cluster < 0:
                cl += 1
            out[a_i].cluster = out[b_i].cluster = ca
            out[a_i].resolvable = out[b_i].resolvable = False
    return out


def first_order_eta(md: Modes, obs: list[Obs], dK, dM) -> np.ndarray:
    """1차 섭동 η = tr(φᵀ(δK − λδM)φ)/(deg·λ) — 관측량별(쌍은 자취 평균)."""
    out = np.empty(len(obs))
    for j, o in enumerate(obs):
        vals = []
        for i in o.members:
            ph = md.phi[:, i]
            vals.append(float(ph @ (dK @ ph) - md.lam[i] * (ph @ (dM @ ph))) / md.lam[i])
        out[j] = float(np.mean(vals))
    return out


def track(md_h: Modes, md_d: Modes, obs: list[Obs], min_mac: float = 0.8):
    """손상 모드를 건전 관측량에 형상(꼭짓점 변위의 부분공간 MAC)으로 짝짓는다.

    반환: (η_resolved[n_obs], mac[n_obs]) — 짝을 못 찾으면 NaN.
    """
    Vh = md_h.vshape
    Vd = md_d.vshape
    nd = Vd.shape[0]
    used = set()
    eta = np.full(len(obs), np.nan)
    mac = np.full(len(obs), np.nan)
    for j, o in enumerate(obs):
        A = Vh[o.members].T                              # (n, deg)
        Qa, _ = np.linalg.qr(A)
        best = (-1.0, None)
        # 후보: 같은 크기의 연속 블록(주파수 정렬된 손상 모드)
        for s in range(nd - o.deg + 1):
            cand = list(range(s, s + o.deg))
            if any(c in used for c in cand):
                continue
            if o.deg == 2 and abs(md_d.freqs[cand[1]] - md_d.freqs[cand[0]]) > 1e-6 * md_d.freqs[cand[0]]:
                continue
            Qb, _ = np.linalg.qr(Vd[cand].T)
            sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
            score = float(np.min(sv) ** 2)
            if score > best[0]:
                best = (score, cand)
        if best[1] is not None and best[0] >= min_mac:
            used.update(best[1])
            eta[j] = float(np.mean(md_d.lam[best[1]]) / np.mean(md_h.lam[o.members]) - 1.0)
            mac[j] = best[0]
    return eta, mac


# ---------------------------------------------------------------- 커널·야코비안
@dataclass
class KernelSet:
    xi: np.ndarray
    k: np.ndarray              # (n_xi, n_obs): η_lin / S̄  (S̄ → 0 극한)
    dk: np.ndarray             # (n_xi, n_obs): ∂k/∂ξ
    s_probe: float
    labels: list


def kernel_scan(am: AsmMesh, md: Modes, obs: list[Obs], xi_grid, w: float = W_GAUSS,
                s_probe: float = 1e-3, dxi: float = 0.01, km0: KM | None = None,
                verbose: bool = False) -> KernelSet:
    """S̄ → 0 극한의 조립체 커널 k_j(ξ) = η_j/S̄와 ∂k/∂ξ.

    p(r)는 d_K에 대해 원점 근방 선형이므로, ±s_probe 감육(두께 가산/감산)의 중심차분이
    2차 정확도로 1차 미분을 준다.
    """
    km0 = km0 or assemble(am)

    def eta_lin(xi):
        p = depth_profile_am(am, am.r_v, xi, s_probe, w)   # 같은 두께면 기존 식(s = b₂ + t)
        kp = assemble(am, morph(am, p))
        km_ = assemble(am, morph(am, -p))
        dK = (kp.K - km_.K) * 0.5
        dM = (kp.M - km_.M) * 0.5
        return first_order_eta(md, obs, dK, dM) / s_probe

    xi_grid = np.asarray(xi_grid, float)
    K = np.empty((xi_grid.size, len(obs)))
    D = np.empty_like(K)
    for i, xi in enumerate(xi_grid):
        t0 = time.time()
        K[i] = eta_lin(xi)
        lo, hi = max(xi - dxi, 0.0), min(xi + dxi, 1.0)
        D[i] = (eta_lin(hi) - eta_lin(lo)) / (hi - lo)
        if verbose:
            print(f"  kernel ξ={xi:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return KernelSet(xi_grid, K, D, s_probe, [o.label for o in obs])


def resolved_eta(am: AsmMesh, md_h: Modes, obs, xi, s_bar, w=W_GAUSS, n_modes=None):
    """유한 손상 재해석 η(관측량별) + 짝짓기 MAC."""
    p = depth_profile_am(am, am.r_v, xi, s_bar, w)
    km = assemble(am, morph(am, p))
    md = solve(km, n_modes or md_h.phi.shape[1] + 6)
    return track(md_h, md, obs)


def fisher_metrics(J: np.ndarray, sig: np.ndarray, extent: float) -> dict:
    """정본 identifiability.metrics_from_J와 같은 규약(위치 mm, 심각도 %p)."""
    from .. import identifiability as idf
    return idf.metrics_from_J(J, np.diag(sig ** 2) if sig.ndim == 1 else sig, extent)
