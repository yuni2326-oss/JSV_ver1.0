"""N6 — 자유-자유 실측 형상 조립체: E2-I 1단계 모드 지도와 2단계 홈 예측 (2026-10-16 사용자 결정).

E2-I 1단계 지지는 자유-자유(매달기)로 정했다(`G2-recompute-plan.md`, P5). N1 캠버 정합 구조격자(정확 C6, 베인
두께 정확, 연속)에 실물에만 있는 두 부분을 붙이고 **경계조건 없이** 푼다.

  (i)  허브: 뒤판(G3 0.60 mm; G1 0.59)을 흡입구 반경 a(G3 14.89 mm; G1 14.955) 안쪽으로 보어 r_bore(G3 6.31 mm; G1 6.335)까지 연장한다(앞판·베인은
       없다). 실물 허브의 압출 고리(돌출 비드)는 평판으로 이상화한다.
  (ii) 흡입구 테: 앞판 안쪽 가장자리 위에 두께 collar_t(G3 0.52 mm 테 끝 직접; G1 0.425 = (30.76 − 29.91)/2), 높이 h_c의 원통 벽. 첫 반경 링의 폭을
       테 두께와 같게 잡아 테가 그 링 위에 정확히 선다. h_c는 미측정이라 두 값(3.5·5.0 mm)으로 괄호친다.

강체 6모드는 음의 이동(σ = −(2π·σ_Hz)²) shift-invert로 피해 풀고 주파수로 걸러낸다(정확히 6개여야 한다).
대칭 판정(h)은 `impeller_hex.harmonic_indices_grouped`(축퇴군 자취, 기저 무관)를 쓰고, 방위차수 m과 반경차수 n은
**앞판 윗면 절점**의 u_z 이산 푸리에로 정한다 — 흡입구 테가 위로 솟아 있어 N1의 "위쪽 30 % 절점" 규칙은 테를 잡는다.

홈(2단계): 앞판 바깥면의 축대칭 홈(중심 ξ, 폭 0.2(b − a), 깊이 δ)을 절점 변형으로 깎는다(N1 `morph`와 같은 규약:
안쪽면은 고정, 판 안 절점은 두께방향 비례 압축). 1차(δK − λδM)와 재해석(+ 부분공간 MAC 짝짓기)을 함께 낸다.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from .. import geometry as geo
from .. import impeller_hex as ih
from .. import rail3d as r3
from . import n1_assembly as n1
from . import n1_mesh as nm

_AB = geo.ASBUILT


@dataclass(frozen=True)
class FFSpec:
    """자유-자유 조립체 스펙 [m]. `base`는 판·베인·격자(N1 `FitSpec`, 지지 필드는 쓰지 않는다)."""

    base: nm.FitSpec = field(default_factory=nm.FitSpec)
    r_bore: float = _AB.r_bore
    n_hub: int = 6                       # 허브(뒤판 r_bore–a) 반경 셀 수
    collar_t: float = _AB.collar_t
    collar_h: float = _AB.collar_h       # 흡입구 테 높이 — G3 5.0 mm(사진 추정, 4.5·5.5 mm로 괄호). G1 산출은 4.0 mm
    n_z_collar: int = 4
    clamp_r: float | None = None         # None = 자유-자유. 값 = 설치 상태(축 슬리브가 허브를 잡는다): r ≤ clamp_r 고정

    @property
    def n_rigid(self) -> int:
        return 6 if self.clamp_r is None else 0

    @property
    def a(self) -> float:
        return self.base.a

    @property
    def b(self) -> float:
        return self.base.b

    @property
    def extent(self) -> float:
        return self.base.b - self.base.a

    @property
    def n_cells_r(self) -> int:
        return self.n_hub + self.base.n_r

    @property
    def r_nodes(self) -> np.ndarray:
        """반경 절점: 허브 [r_bore, a] n_hub칸, 첫 주 링 [a, a + t_c](테 두께), 나머지 [a + t_c, b] n_r − 1칸."""
        fs = self.base
        hub = np.linspace(self.r_bore, fs.a, self.n_hub + 1)[:-1]
        main = np.concatenate([[fs.a], np.linspace(fs.a + self.collar_t, fs.b, fs.n_r)])
        return np.concatenate([hub, main])

    @property
    def z_breaks(self) -> np.ndarray:
        zb = self.base.z_breaks
        zc = zb[-1] + np.linspace(0.0, self.collar_h, self.n_z_collar + 1)[1:]
        return np.concatenate([zb, zc])

    @property
    def z_front(self) -> tuple[float, float]:
        """앞판의 (안쪽면, 바깥면) z."""
        fs = self.base
        return fs.t_back + fs.channel, fs.t_back + fs.channel + fs.t_front

    def hex_spec(self) -> ih.HexImpellerSpec:
        """대칭 판정 색인용(반경 칸 수만 전체 링으로 바꾼다)."""
        return replace(self.base.hex_spec(), n_r=self.n_cells_r)


def mesh(ff: FFSpec):
    """→ (coors, conn, info). info: grid_idx(꼭짓점 (ir, it, iz)), cell_kind(0 뒤판·1 베인·2 앞판·3 테),
    cell_hub(뒤판 허브 셀), cell_vane, cell_r, cell_span."""
    fs = ff.base
    nth = fs.n_theta
    per = fs.n_vc + fs.n_bay
    r_nodes = ff.r_nodes
    nrc = len(r_nodes) - 1
    i_a = ff.n_hub
    zb = ff.z_breaks
    nz = len(zb) - 1
    kinds = (["back"] * fs.n_z_shroud + ["channel"] * fs.n_z_channel
             + ["front"] * fs.n_z_shroud + ["collar"] * ff.n_z_collar)
    phi = np.zeros((nrc + 1, per))
    for i, r in enumerate(r_nodes):
        hv = fs.t_vane / (2 * r * math.cos(fs.blade_angle(r)))
        vane = np.linspace(-hv, hv, fs.n_vc + 1)[:-1]
        bay = np.linspace(hv, 2 * math.pi / fs.n_vane - hv, fs.n_bay + 1)[:-1]
        phi[i] = np.concatenate([vane, bay])
    cam = fs.camber(r_nodes)
    theta = np.zeros((nrc + 1, nth))
    for k in range(fs.n_vane):
        theta[:, k * per:(k + 1) * per] = 2 * math.pi * k / fs.n_vane + cam[:, None] + phi

    def nid(ir, it, iz):
        return (iz * nth * (nrc + 1)) + ((it % nth) * (nrc + 1)) + ir

    conn, ckind, chub, cvane, cr, cspan = [], [], [], [], [], []
    for iz in range(nz):
        kd = kinds[iz]
        for it in range(nth):
            is_vane = (it % per) < fs.n_vc
            for ir in range(nrc):
                rc = 0.5 * (r_nodes[ir] + r_nodes[ir + 1])
                hub = ir < i_a
                if kd == "channel" and (hub or not (is_vane and rc >= fs.r_vane_in)):
                    continue
                if kd == "front" and hub:
                    continue
                if kd == "collar" and ir != i_a:
                    continue
                conn.append([nid(ir, it, iz), nid(ir + 1, it, iz), nid(ir + 1, it + 1, iz),
                             nid(ir, it + 1, iz), nid(ir, it, iz + 1), nid(ir + 1, it, iz + 1),
                             nid(ir + 1, it + 1, iz + 1), nid(ir, it + 1, iz + 1)])
                ckind.append({"back": 0, "channel": 1, "front": 2, "collar": 3}[kd])
                chub.append(hub)
                cvane.append(it // per if kd == "channel" else -1)
                cr.append(rc)
                cspan.append((rc - fs.r_vane_in) / (fs.b - fs.r_vane_in))
    conn = np.array(conn, dtype=np.int64)
    used = np.unique(conn)
    remap = -np.ones(nth * (nrc + 1) * (nz + 1), dtype=np.int64)
    remap[used] = np.arange(used.size)
    conn = remap[conn]
    iz_g = used // (nth * (nrc + 1))
    rem = used % (nth * (nrc + 1))
    it_g = rem // (nrc + 1)
    ir_g = rem % (nrc + 1)
    R, TH, Z = r_nodes[ir_g], theta[ir_g, it_g], zb[iz_g]
    coors = np.stack([R * np.cos(TH), R * np.sin(TH), Z], axis=1)
    e = coors[conn]
    v = np.einsum("ij,ij->i", np.cross(e[:, 1] - e[:, 0], e[:, 3] - e[:, 0]), e[:, 4] - e[:, 0])
    if np.any(v < 0):
        if np.all(v < 0):
            conn = conn[:, [0, 3, 2, 1, 4, 7, 6, 5]]
        else:
            raise RuntimeError("요소 방향이 섞였다 — 격자 기울기 점검")
    info = {"n_elem": int(conn.shape[0]), "n_node": int(coors.shape[0]),
            "grid_idx": np.stack([ir_g, it_g, iz_g], axis=1),
            "cell_kind": np.array(ckind), "cell_hub": np.array(chub), "cell_vane": np.array(cvane),
            "cell_r": np.array(cr), "cell_span": np.array(cspan)}
    return coors, conn, info


@dataclass
class FFMesh:
    spec: FFSpec
    coors: np.ndarray
    conn: np.ndarray
    info: dict
    r_v: np.ndarray
    hspec: ih.HexImpellerSpec


def build(ff: FFSpec) -> FFMesh:
    coors, conn, info = mesh(ff)
    return FFMesh(ff, coors, conn, info, np.hypot(coors[:, 0], coors[:, 1]), ff.hex_spec())


def assemble(fm: FFMesh, coors: np.ndarray | None = None, cell_mask=None, order: int = 2):
    """K·M — 자유-자유면 경계조건 없음, `spec.clamp_r`가 있으면 r ≤ clamp_r(허브) 절점 고정.
    `cell_mask`를 주면 그 셀만의 K_R·M_R(같은 자유도 번호)."""
    from sfepy.base.base import output
    from sfepy.discrete import Equation, Equations, FieldVariable, Integral, Material, Problem
    from sfepy.discrete.conditions import Conditions
    from sfepy.discrete.fem import FEDomain, Field, Mesh
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.terms import Term

    output.set_output(quiet=True)
    fs = fm.spec.base
    X = n1.physical_coors(fm, fm.coors if coors is None else coors)     # G4 테이퍼면 절점 높이를 옮긴다
    grp = (np.zeros(len(fm.conn), np.int32) if cell_mask is None
           else np.asarray(cell_mask, bool).astype(np.int32))
    mesh_ = Mesh.from_data("ff", X, None, [fm.conn], [grp], [r3.elem_desc(X, fm.conn)])
    domain = FEDomain("d", mesh_)
    omega = domain.create_region("Omega", "all")
    rgn = omega if cell_mask is None else domain.create_region("Rgn", "cells of group 1", "cell")
    fld = Field.from_args("fu", np.float64, "vector", omega, approx_order=order)
    u = FieldVariable("u", "unknown", fld)
    v = FieldVariable("v", "test", fld, primary_var_name="u")
    mat = Material("m", D=stiffness_from_youngpoisson(3, fs.E, fs.nu), rho=fs.rho)
    integral = Integral("i", order=2 * order)
    eq_k = Equation("K", Term.new("dw_lin_elastic(m.D, v, u)", integral, rgn, m=mat, v=v, u=u))
    eq_m = Equation("M", Term.new("dw_dot(m.rho, v, u)", integral, rgn, m=mat, v=v, u=u))
    pb = Problem("ff", equations=Equations([eq_k, eq_m]))
    ebcs = []
    if fm.spec.clamp_r is not None:
        from sfepy.discrete import Function, Functions
        from sfepy.discrete.conditions import EssentialBC
        r_c = fm.spec.clamp_r + 1e-7
        fns = Functions([Function("hub_fn", lambda coors, domain=None:
                                  np.nonzero(np.hypot(coors[:, 0], coors[:, 1]) <= r_c)[0])])
        domain.create_region("Hub", "vertices by hub_fn", "vertex", functions=fns)
        ebcs = [EssentialBC("HubFix", domain.regions["Hub"], {"u.all": 0.0})]
    pb.time_update(ebcs=Conditions(ebcs))
    pb.update_materials()
    K = eq_k.evaluate(mode="weak", dw_mode="matrix", asm_obj=pb.mtx_a)
    M0 = K.copy()
    M0.data[:] = 0.0
    M = eq_m.evaluate(mode="weak", dw_mode="matrix", asm_obj=M0)
    variables = pb.get_variables()
    vol = float(omega.domain.cmesh.get_volumes(3).sum()) if cell_mask is None else float("nan")
    return n1.KM(K.tocsc(), M.tocsc(), lambda vec: variables.make_full_vec(vec), X.shape[0], vol)


def solve(km: n1.KM, n_modes: int, sigma_hz: float = 200.0, rigid_hz: float = 50.0,
          v0_seed: int = 20261016, n_rigid: int = 6) -> tuple[n1.Modes, np.ndarray]:
    """shift-invert(σ < 0). 반환: (탄성 모드만의 Modes, 강체 모드 주파수[n_rigid]).
    자유-자유는 n_rigid = 6, 허브 고정(설치 상태)은 0."""
    t0 = time.time()
    sig = -(2 * math.pi * sigma_hz) ** 2
    A = (km.K - sig * km.M).tocsc()
    lu = splu(A, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
    op = LinearOperator(A.shape, matvec=lu.solve, dtype=float)
    v0 = np.random.default_rng(v0_seed).standard_normal(A.shape[0])
    vals, vecs = eigsh(km.K, k=n_modes + n_rigid, M=km.M, sigma=sig, which="LM", v0=v0, OPinv=op)
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]
    f = np.sqrt(np.maximum(vals, 0.0)) / (2 * math.pi)
    rigid = f < rigid_hz
    if int(rigid.sum()) != n_rigid:
        raise RuntimeError(f"강체 모드가 {n_rigid}개가 아니다: {np.round(f[:10], 2)} Hz — 연결 끊김 또는 σ 점검")
    vals, vecs = vals[~rigid], vecs[:, ~rigid]
    mn = np.sqrt(np.einsum("ij,ij->j", vecs, km.M @ vecs))
    vecs = vecs / mn
    nv = km.n_vertex
    vsh = np.stack([km.make_full(vecs[:, k]).reshape(-1, 3)[:nv].ravel() for k in range(vecs.shape[1])])
    md = n1.Modes(vals, np.sqrt(vals) / (2 * math.pi), vecs, vsh, km.K.shape[0], time.time() - t0)
    return md, f[rigid]


def _front_top_nodes(fm: FFMesh) -> np.ndarray:
    z_in, z_top = fm.spec.z_front
    rr = fm.r_v
    return np.nonzero(np.isclose(fm.coors[:, 2], z_top, atol=1e-9)
                      & (rr >= fm.spec.a + fm.spec.collar_t - 1e-9))[0]


def m_profiles(fm: FFMesh, md: n1.Modes, m: int) -> np.ndarray:
    """앞판 윗면 u_z의 링별 방위 푸리에 계수 c_m(r) — (n_modes, 링 수)."""
    sel = _front_top_nodes(fm)
    ir = fm.info["grid_idx"][sel, 0]
    rings = np.unique(ir)
    pos = np.searchsorted(rings, ir)
    th = np.arctan2(fm.coors[sel, 1], fm.coors[sel, 0])
    ph = np.exp(-1j * m * th)
    out = np.zeros((md.vshape.shape[0], rings.size), dtype=complex)
    for k in range(md.vshape.shape[0]):
        uz = md.vshape[k].reshape(-1, 3)[sel, 2]
        np.add.at(out[k], pos, uz * ph)
    return out


def classify(fm: FFMesh, md: n1.Modes, rel_tol: float = 1e-7, m_max: int = 9,
             res_gap: float = 2e-3) -> list[n1.Obs]:
    """건전 탄성 모드 → 대칭 관측량(형상으로 h·m·n 판정). N1 `classify`와 같은 규약."""
    hs = fm.hspec
    res = r3.ModalResult(freqs=md.freqs, shapes=None, coors=fm.coors, ndof=md.ndof,
                         field_coors=fm.coors, full_shapes=md.vshape)
    h, c, deg, gid = ih.harmonic_indices_grouped(hs, res, fm.info["grid_idx"], None, rel_tol=rel_tol)
    prof = {m: m_profiles(fm, md, m) for m in range(m_max + 1)}
    vane_nodes = np.zeros(fm.coors.shape[0], bool)
    vc = fm.conn[fm.info["cell_kind"] == 1]
    vane_nodes[np.unique(vc)] = True
    out: list[n1.Obs] = []
    nv = fm.spec.base.n_vane
    for g in np.unique(gid):
        idx = [int(i) for i in np.nonzero(gid == g)[0]]
        hh = int(round(float(h[idx[0]])))
        en = np.array([sum(np.sum(np.abs(prof[m][i]) ** 2) for i in idx) for m in range(m_max + 1)])
        mdom = int(np.argmax(en))
        n_rad = ih.radial_order(prof[mdom][idx[0]])
        U = md.vshape[idx].reshape(len(idx), -1, 3)
        tot = float(np.sum(U ** 2))
        vs = float(np.sum(U[:, vane_nodes] ** 2) / tot) if tot > 0 else 0.0
        d = len(idx)
        if d not in (1, 2):
            raise RuntimeError(f"축퇴군 크기 {d} (f={md.freqs[idx]}) — rel_tol 확인")
        if d == 1 and hh not in (0, nv // 2):
            if max(idx) == len(md.freqs) - 1:
                continue
            raise RuntimeError(f"h={hh} 단일 모드가 스펙트럼 중간에 있다 (f={md.freqs[idx]})")
        if d == 2 and hh in (0, nv // 2):
            raise RuntimeError(f"h={hh}인데 2중축퇴 (f={md.freqs[idx]})")
        sf = 2.0 if d == 2 else 2.0 * math.sqrt(2.0)
        out.append(n1.Obs(label=f"h{hh}:m{mdom}n{n_rad}", h=hh, deg=d, members=idx,
                          f_Hz=float(np.mean(md.freqs[idx])), m_dom=mdom, n_rad=n_rad,
                          vane_share=vs, sigma_factor=sf))
    seen: dict = {}
    for o in out:
        k = seen.get(o.label, 0)
        seen[o.label] = k + 1
        if k:
            o.label = f"{o.label}#{k + 1}"
    fs_ = np.array([o.f_Hz for o in out])
    order = np.argsort(fs_)
    cl = 0
    for a_i, b_i in zip(order[:-1], order[1:]):
        if abs(fs_[b_i] - fs_[a_i]) / fs_[a_i] < res_gap:
            ca = out[a_i].cluster if out[a_i].cluster >= 0 else cl
            if out[a_i].cluster < 0:
                cl += 1
            out[a_i].cluster = out[b_i].cluster = ca
            out[a_i].resolvable = out[b_i].resolvable = False
    return out


def region_masks(fm: FFMesh) -> dict:
    ck, hub = fm.info["cell_kind"], fm.info["cell_hub"]
    return {"front": ck == 2, "back_main": (ck == 0) & ~hub, "hub": (ck == 0) & hub,
            "vanes": ck == 1, "collar": ck == 3}


def energy_fractions(fm: FFMesh, md: n1.Modes, obs: list, masks: dict | None = None) -> dict:
    """관측량별 구역 변형에너지 분율(축퇴군 자취 평균)."""
    masks = masks or region_masks(fm)
    out = {}
    for name, mk in masks.items():
        if not mk.any():
            out[name] = np.zeros(len(obs))
            continue
        kmr = assemble(fm, cell_mask=mk)
        out[name] = n1.energy_fractions(md, obs, kmr)[0]
    return out


# ---------------------------------------------------------------- 홈(2단계)
def groove_depth_ratio(r: np.ndarray, ff: FFSpec, xi: float, depth: float,
                       width: float | None = None) -> np.ndarray:
    """앞판 바깥면 축대칭 홈의 깊이비 p(r) = depth/t_front (|r − r_c| ≤ width/2), 아니면 0."""
    width = 0.2 * ff.extent if width is None else width
    r_c = ff.a + xi * ff.extent
    return np.where(np.abs(np.asarray(r, float) - r_c) <= 0.5 * width + 1e-12,
                    depth / ff.base.t_front, 0.0)


def morph_groove(fm: FFMesh, p_v: np.ndarray) -> np.ndarray:
    """앞판 절점만 두께방향 비례 압축(안쪽면 고정). 흡입구 테 절점은 움직이지 않는다."""
    z_in, z_top = fm.spec.z_front
    c = fm.coors.copy()
    sel = (c[:, 2] >= z_in - 1e-12) & (c[:, 2] <= z_top + 1e-12) & (fm.r_v >= fm.spec.a - 1e-12)
    c[sel, 2] = z_in + (c[sel, 2] - z_in) * (1.0 - p_v[sel])
    return c


def groove_removed_volume(ff: FFSpec, xi: float, depth: float, width: float | None = None) -> float:
    width = 0.2 * ff.extent if width is None else width
    r_c = ff.a + xi * ff.extent
    r1, r2 = r_c - 0.5 * width, r_c + 0.5 * width
    return math.pi * (r2 ** 2 - r1 ** 2) * depth


def groove_first_order(fm: FFMesh, md: n1.Modes, obs: list, xi: float, depth: float,
                       width: float | None = None, probe: float = 0.05) -> np.ndarray:
    """1차 η(관측량별) — 깊이에 선형으로 외삽(중심차분 ±probe·depth)."""
    p = groove_depth_ratio(fm.r_v, fm.spec, xi, depth * probe, width)
    kp = assemble(fm, morph_groove(fm, p))
    km_ = assemble(fm, morph_groove(fm, -p))
    dK = (kp.K - km_.K) * 0.5
    dM = (kp.M - km_.M) * 0.5
    return n1.first_order_eta(md, obs, dK, dM) / probe


def groove_resolved(fm: FFMesh, md_h: n1.Modes, obs: list, xi: float, depth: float,
                    width: float | None = None, n_modes: int | None = None):
    """유한 홈 재해석 η + 부분공간 MAC 짝짓기."""
    p = groove_depth_ratio(fm.r_v, fm.spec, xi, depth, width)
    km = assemble(fm, morph_groove(fm, p))
    md, _ = solve(km, n_modes or md_h.phi.shape[1] + 6, n_rigid=fm.spec.n_rigid)
    return n1.track(md_h, md, obs)


# ---------------------------------------------------------------- 가우시안 감육 커널(N1과 같은 규약)
def gaussian_depth_ratio(fm: FFMesh, xi: float, s_bar: float, w: float = n1.W_GAUSS) -> np.ndarray:
    """정본 가우시안 단면 강성손실 d_K(r)(부록 A.3) → 앞판 깊이비 p(r) — 실측 2면판 비대칭 식(G1).

    앞판(r ≥ a) 절점만 값이 있다. 흡입구 테·허브는 0.
    """
    from . import g1_section as g1
    fs, ff = fm.spec.base, fm.spec
    r = fm.r_v
    dk = np.clip(n1.dK_profile(r, xi, s_bar, w, fs.a, fs.b), 0.0, None)
    if fs.tapered:                                   # G4: 그 반지름의 간격으로
        dk = np.where(r >= fs.a - 1e-12, dk, 0.0)
        return n1.depth_from_dk_local(dk, r, fs.t_front, fs.t_back, fs.channel_at)
    p = g1.depth_from_dk(dk, fs.t_front, fs.t_back, fs.channel)
    return np.where(r >= fs.a - 1e-12, p, 0.0)


def kernel_scan(fm: FFMesh, md: n1.Modes, obs: list, xi_grid, w: float = n1.W_GAUSS,
                s_probe: float = 1e-3, dxi: float = 0.01, verbose: bool = False) -> n1.KernelSet:
    """S̄ → 0 극한의 조립체 커널 k_j(ξ) = η_j/S̄와 ∂k/∂ξ (N1 `kernel_scan`과 같은 중심차분 규약).

    손상 = 앞판 바깥면 감육(안쪽면·흡입구 테 고정, `morph_groove`). ξ는 앞판 반경 [a, b] 기준.
    """
    def eta_lin(xi):
        p = gaussian_depth_ratio(fm, xi, s_probe, w)
        kp = assemble(fm, morph_groove(fm, p))
        km_ = assemble(fm, morph_groove(fm, -p))
        return n1.first_order_eta(md, obs, (kp.K - km_.K) * 0.5, (kp.M - km_.M) * 0.5) / s_probe

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
    return n1.KernelSet(xi_grid, K, D, s_probe, [o.label for o in obs])


def resolved_gaussian(fm: FFMesh, md_h: n1.Modes, obs: list, xi: float, s_bar: float,
                      w: float = n1.W_GAUSS, n_modes: int | None = None):
    """유한 가우시안 감육 재해석 η + 부분공간 MAC 짝짓기."""
    km = assemble(fm, morph_groove(fm, gaussian_depth_ratio(fm, xi, s_bar, w)))
    md, _ = solve(km, n_modes or md_h.phi.shape[1] + 6, n_rigid=fm.spec.n_rigid)
    return n1.track(md_h, md, obs)
