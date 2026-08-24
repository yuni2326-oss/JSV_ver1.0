"""Fig 1 — 파라메트릭 폐쇄형 임펠러 **구조격자 육면체** 메시 + 자유-자유 건조 모달.

정본 §3.1의 Fig 1(임펠러 기하 + 모드형)을 만들고, 그 그림이 드러낸 **순환대칭(C_N) 제약**을
수치로 규명하는 모듈이다. `impeller_cad`(gmsh OCC + 사면체)와 역할이 다르다:

  `impeller_cad`  후곡 캠버·필렛까지 담는 **as-built 대리 CAD**(사면체) — §3.6-ii(F16/F62)의
                  감육 스윕용. 사면체 격자는 C6 대칭을 **정확히 보존하지 못해** 건전 상태에서도
                  0.04–0.96 % of f의 수치 미스튜닝이 나온다(F62 — 그래서 분리 배수를 철회했다).
  `impeller_hex`  방위 균등분할 구조격자(육면체). `n_theta % n_vane == 0`이면 격자 자체가
                  **정확히 C_{n_theta}** 대칭이므로 m ≤ n_theta/2 − 1 조화에는 격자 유래
                  미스튜닝이 원리적으로 들어가지 않는다. 순환대칭 성질(어느 m이 doublet이고
                  어느 m이 singlet인가)을 **인공물과 분리해** 측정할 수 있는 유일한 격자다.

**illustration-grade**: 이 메시는 그림과 대칭성 판정을 위한 것이고, 제출용 수치(균열 지문·
식별성·model-form penalty)는 정본 §3.6의 canonical 레일(`geometry.DISK` + `rail3d`,
`impeller_cad`의 b5)에서 나온다. 사용한 이산화는 1차 육면체 + lumped mass이므로 얇은 굽힘에서
전단잠김이 있고 **절대주파수는 위로 편향**된다(자기검증으로 강성 6 rigid mode와 총질량만 본다).

기하 = 정본 §3.1의 **실측 단면**:
    판두께 t = 1.0 mm(전·후면 슈라우드·베인 동일), 유로폭 b₂ = 4.1 mm ⇒ 림 6.1 mm, s = 5.1 mm,
    환형 a = 15.4 mm, b = 36.56 mm, 베인 6매, SUS304 (E = 193 GPa, ρ = 8000, ν = 0.29).
베인 wrap 각은 **미확정 스윕 파라미터**(정본 §3.6-ii)이므로 그림은 60°를 쓰고 노출한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import rail3d as r3

#: 8절점 육면체 모절점(ξ,η,ζ) 순서 — sfepy `3_8`과 같은 규약(`rail3d._HEX_OFFSETS`).
_CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)


@dataclass(frozen=True)
class HexImpellerSpec:
    """Fig 1 임펠러 파라미터 (SI, m). 기하는 실측, 메시는 노출된 파라미터."""

    # --- 기하(실측 단면) ---
    a: float = 0.01540              # 환형 내경 R₁ (흡입구)
    b: float = 0.03656              # 환형 외경 R₂
    t_sheet: float = 0.0010         # 판두께(양 슈라우드)  **실측**
    channel: float = 0.0041         # 유로폭 b₂           **실측**
    t_vane: float = 0.0010          # 베인 판두께          **실측**
    n_vane: int = 6                 # 베인 수 (스폿용접 궤적으로 확인)
    wrap_deg: float = 60.0          # 베인 wrap 각 (**미확정**, 스윕)
    # --- 물성 ---
    E: float = 193e9
    nu: float = 0.29
    rho: float = 8000.0
    # --- 메시(노출) ---
    n_r: int = 18                   # 반경 요소수
    n_theta: int = 108              # 방위 요소수(주기) — n_vane의 배수여야 한다
    n_z_shroud: int = 2             # 슈라우드 1장의 z 요소수
    n_z_channel: int = 2            # 유로층 z 요소수
    #: "discrete" = 베인 n_vane매(C_n_vane) / "web" = 방위 균일 웹(축대칭 대조군)
    vane_mode: str = "discrete"
    #: 베인 1매가 차지하는 방위 **셀 수**(None이면 물리 t_vane으로 판정). C6 섭동 세기를
    #: 연속적으로 줄이는 스윕 손잡이 — `n_theta/n_vane`이면 웹(축대칭)과 정확히 같아진다.
    vane_arc_cells: int | None = None
    #: **형상동결 정제**(F78) — 베인 발자국을 이 기준격자에서 한 번 판정하고, 실제 격자는
    #: 그것을 정수배로 **세분**만 한다. `None`이면 자기 격자에서 판정한다(기존 동작).
    #:
    #: 왜 필요한가: 물리 t_vane 판정(`vane_arc_cells=None`)은 셀 방위폭 r·dθ와 비교하므로
    #: n_theta를 올리면 발자국 셀수가 달라지고 **형상 자체가 변한다** — F11′이 3D 포켓
    #: 레일에서 겪은 격자 스냅과 같은 함정이고, 그 상태로 잰 수렴은 이산화가 아니라
    #: 기하 변화를 재는 것이다. n_r·n_theta가 기준격자의 정수배면 기준격자의 모든 셀면이
    #: 정제격자의 셀면이기도 하므로, 기준 발자국은 정제격자에서 **정확히** 실현된다.
    footprint_n_r: int | None = None
    footprint_n_theta: int | None = None

    def check(self) -> None:
        if self.n_theta % max(self.n_vane, 1) != 0:
            raise ValueError(
                f"n_theta({self.n_theta})가 n_vane({self.n_vane})의 배수가 아니다 — "
                "격자가 C_n_vane 대칭을 정확히 보존하지 못해 대칭성 판정이 오염된다")
        if self.vane_mode not in ("discrete", "web"):
            raise ValueError(f"vane_mode: {self.vane_mode}")
        if self.channel <= 0 or self.t_sheet <= 0:
            raise ValueError("두께/유로폭이 양수여야 한다")
        if self.vane_arc_cells is not None:
            lim = self.n_theta // max(self.n_vane, 1)
            if not 1 <= self.vane_arc_cells <= lim:
                raise ValueError(f"vane_arc_cells는 1..{lim} 범위여야 한다")
        if (self.footprint_n_r is None) != (self.footprint_n_theta is None):
            raise ValueError("footprint_n_r·footprint_n_theta는 함께 지정해야 한다")
        if self.footprint_n_r is not None:
            for nm, fine, ref in (("n_r", self.n_r, self.footprint_n_r),
                                  ("n_theta", self.n_theta, self.footprint_n_theta)):
                if ref < 1 or fine % ref != 0:
                    raise ValueError(
                        f"{nm}({fine})이 기준격자({ref})의 정수배가 아니다 — 기준 "
                        "발자국이 셀면에 정확히 놓이지 않아 형상이 달라진다(F11′)")
            if self.footprint_n_theta % max(self.n_vane, 1) != 0:
                raise ValueError("footprint_n_theta도 n_vane의 배수여야 한다")

    @property
    def footprint_spec(self) -> "HexImpellerSpec":
        """발자국을 판정하는 기준격자 스펙(형상동결이 아니면 자기 자신)."""
        if self.footprint_n_r is None:
            return self
        return replace(self, n_r=self.footprint_n_r, n_theta=self.footprint_n_theta,
                       footprint_n_r=None, footprint_n_theta=None)

    @property
    def refine_factors(self) -> tuple[int, int]:
        """(반경, 방위) 세분 배수 — 형상동결이 아니면 (1, 1)."""
        if self.footprint_n_r is None:
            return 1, 1
        return (self.n_r // self.footprint_n_r,
                self.n_theta // self.footprint_n_theta)

    @property
    def cells_per_sector(self) -> int:
        return self.n_theta // max(self.n_vane, 1)

    @property
    def modulation_depth(self) -> float:
        """C_N 방위 변조 깊이 = 1 − (베인 방위 점유율). 0이면 축대칭."""
        if self.vane_mode == "web":
            return 0.0
        if self.vane_arc_cells is None:
            return 1.0                       # 물리 t_vane은 1셀 수준(최대 변조)
        return 1.0 - self.vane_arc_cells / self.cells_per_sector

    @property
    def total_thickness(self) -> float:
        """림 전체두께 = b₂ + 2t (파생값)."""
        return self.channel + 2 * self.t_sheet

    @property
    def face_separation(self) -> float:
        """면판 중립면 간격 s = b₂ + t."""
        return self.channel + self.t_sheet

    @property
    def z_breaks(self) -> np.ndarray:
        """z 절점 좌표 — [후면 슈라우드 | 유로 | 전면 슈라우드]."""
        t, c = self.t_sheet, self.channel
        back = np.linspace(0.0, t, self.n_z_shroud + 1)
        chan = np.linspace(t, t + c, self.n_z_channel + 1)[1:]
        front = np.linspace(t + c, 2 * t + c, self.n_z_shroud + 1)[1:]
        return np.concatenate([back, chan, front])

    @property
    def layer_kind(self) -> list[str]:
        return (["shroud"] * self.n_z_shroud + ["channel"] * self.n_z_channel
                + ["shroud"] * self.n_z_shroud)


def vane_footprint(spec: HexImpellerSpec, r_c: np.ndarray,
                   th_c: np.ndarray) -> np.ndarray:
    """요소중심 (r_c, th_c)가 베인 발자국 안인가 — 선형 wrap 캠버.

    캠버선 θ_k(r) = 2πk/N + wrap·(r−a)/(b−a). `impeller_cad`의 로그나선 후곡과 법칙이
    다르지만(그림용 단순 스윕), **wrap 각을 노출**하므로 §3.6-ii의 미확정 파라미터 스윕과
    같은 역할을 한다. 대칭성 판정은 캠버 법칙이 아니라 **N매라는 사실**에만 의존한다.
    """
    if spec.vane_mode == "web":
        return np.ones_like(r_c, dtype=bool)
    u = (r_c - spec.a) / (spec.b - spec.a)
    wrap = math.radians(spec.wrap_deg)
    dth = 2 * math.pi / spec.n_theta
    inside = np.zeros_like(r_c, dtype=bool)
    for k in range(spec.n_vane):
        th_k = 2 * math.pi * k / spec.n_vane + wrap * u
        d = np.angle(np.exp(1j * (th_c - th_k)))
        if spec.vane_arc_cells is None:
            inside |= (np.abs(d) * r_c <= 0.5 * spec.t_vane)
        else:
            inside |= (np.abs(d) <= 0.5 * spec.vane_arc_cells * dth + 1e-12)
    return inside


def footprint_grid(spec: HexImpellerSpec) -> np.ndarray:
    """(n_r, n_theta) 요소중심의 베인 발자국 불리언 — **형상동결 정제**를 처리한다(F78).

    `footprint_n_*`가 없으면 자기 격자의 요소중심에서 `vane_footprint`를 그대로 부른다
    (기존 동작과 비트단위 동일). 있으면 기준격자에서 한 번 판정하고 `np.repeat`으로
    세분한다 — 기준 셀 (p, q)는 정제 셀 [p·k_r, (p+1)·k_r) × [q·k_θ, (q+1)·k_θ)에
    정확히 대응하므로(r 절점은 등간격 linspace, θ 절점은 0에서 시작하는 등간격) 고체
    영역이 **정확히** 같다. 즉 정제가 바꾸는 것은 이산화뿐이고 형상은 아니다.
    """
    ref = spec.footprint_spec
    dth = 2 * math.pi / ref.n_theta
    r_nodes = np.linspace(ref.a, ref.b, ref.n_r + 1)
    r_c = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    th_c = np.arange(ref.n_theta) * dth + 0.5 * dth
    RC, TC = np.meshgrid(r_c, th_c, indexing="ij")
    base = vane_footprint(ref, RC.ravel(), TC.ravel()).reshape(ref.n_r, ref.n_theta)
    kr, kt = spec.refine_factors
    if (kr, kt) == (1, 1):
        return base
    return np.repeat(np.repeat(base, kr, axis=0), kt, axis=1)


def mesh(spec: HexImpellerSpec):
    """구조격자 육면체 메시 → (coors, conn, info). 방위는 주기(wrap)."""
    spec.check()
    nr, nth = spec.n_r, spec.n_theta
    zb = spec.z_breaks
    nz = len(zb) - 1
    kinds = spec.layer_kind

    r_nodes = np.linspace(spec.a, spec.b, nr + 1)
    dth = 2 * math.pi / nth
    th_nodes = np.arange(nth) * dth

    def nid(ir, it, iz):
        return (iz * nth * (nr + 1)) + ((it % nth) * (nr + 1)) + ir

    in_vane = footprint_grid(spec)                           # (nr, nth), F78

    conn = []
    for iz in range(nz):
        full = kinds[iz] == "shroud"
        for it in range(nth):
            for ir in range(nr):
                if not (full or in_vane[ir, it]):
                    continue
                conn.append([nid(ir, it, iz), nid(ir + 1, it, iz),
                             nid(ir + 1, it + 1, iz), nid(ir, it + 1, iz),
                             nid(ir, it, iz + 1), nid(ir + 1, it, iz + 1),
                             nid(ir + 1, it + 1, iz + 1), nid(ir, it + 1, iz + 1)])
    conn = np.array(conn, dtype=np.int64)

    used = np.unique(conn)
    remap = -np.ones(nth * (nr + 1) * (nz + 1), dtype=np.int64)
    remap[used] = np.arange(used.size)
    conn = remap[conn]

    iz_g = used // (nth * (nr + 1))
    rem = used % (nth * (nr + 1))
    it_g = rem // (nr + 1)
    ir_g = rem % (nr + 1)
    R, TH, Z = r_nodes[ir_g], th_nodes[it_g], zb[iz_g]
    coors = np.stack([R * np.cos(TH), R * np.sin(TH), Z], axis=1)

    info = {"n_elem": int(conn.shape[0]), "n_node": int(coors.shape[0]),
            "n_dof": int(3 * coors.shape[0]),
            "n_vane_elem": int(in_vane.sum() * spec.n_z_channel),
            "elem_desc": r3.elem_desc(coors, conn),   # 요소형 **추론**(하드코딩 금지 규약)
            "vane_cells_per_vane": float(in_vane.sum() / max(spec.n_vane, 1)),
            # 형상동결 정제(F78)의 불변량 — 발자국 방위점유율은 정제해도 같아야 한다.
            "vane_area_frac": float(in_vane.mean()),
            "footprint_key": (f"{spec.footprint_spec.n_r}x"
                              f"{spec.footprint_spec.n_theta}"),
            # 대칭연산(방위 셀 이동)을 만들기 위한 격자 색인 — (n_node, 3) = (ir, it, iz)
            "grid_idx": np.stack([ir_g, it_g, iz_g], axis=1)}
    return coors, conn, info


def assemble(spec: HexImpellerSpec, coors, conn):
    """등방 선형탄성 K(sparse) + lumped M(diag) + 체적. 2×2×2 Gauss, 벡터화."""
    E, nu, rho = spec.E, spec.nu, spec.rho
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.array([[lam + 2 * mu, lam, lam, 0, 0, 0],
                  [lam, lam + 2 * mu, lam, 0, 0, 0],
                  [lam, lam, lam + 2 * mu, 0, 0, 0],
                  [0, 0, 0, mu, 0, 0], [0, 0, 0, 0, mu, 0], [0, 0, 0, 0, 0, mu]])
    g = 1.0 / math.sqrt(3.0)
    gp = np.array([[sx * g, sy * g, sz * g]
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

    ne = conn.shape[0]
    Xe = coors[conn]                                     # (ne, 8, 3)
    Ke = np.zeros((ne, 24, 24))
    Ve = np.zeros(ne)
    for (xi, eta, ze) in gp:
        dN = np.empty((8, 3))
        for i, (ci, cj, ck) in enumerate(_CORNERS):
            dN[i, 0] = 0.125 * ci * (1 + cj * eta) * (1 + ck * ze)
            dN[i, 1] = 0.125 * cj * (1 + ci * xi) * (1 + ck * ze)
            dN[i, 2] = 0.125 * ck * (1 + ci * xi) * (1 + cj * eta)
        J = np.einsum("ia,eib->eab", dN, Xe)
        detJ = np.linalg.det(J)
        dNx = np.einsum("eab,ib->eia", np.linalg.inv(J), dN)
        B = np.zeros((ne, 6, 24))
        B[:, 0, 0::3] = dNx[:, :, 0]
        B[:, 1, 1::3] = dNx[:, :, 1]
        B[:, 2, 2::3] = dNx[:, :, 2]
        B[:, 3, 0::3] = dNx[:, :, 1]; B[:, 3, 1::3] = dNx[:, :, 0]
        B[:, 4, 1::3] = dNx[:, :, 2]; B[:, 4, 2::3] = dNx[:, :, 1]
        B[:, 5, 0::3] = dNx[:, :, 2]; B[:, 5, 2::3] = dNx[:, :, 0]
        Ke += np.einsum("eip,ij,ejq,e->epq", B, D, B, detJ, optimize=True)
        Ve += detJ

    nn = coors.shape[0]
    dofmap = np.empty((ne, 24), dtype=np.int64)
    dofmap[:, 0::3] = 3 * conn
    dofmap[:, 1::3] = 3 * conn + 1
    dofmap[:, 2::3] = 3 * conn + 2
    rows = np.repeat(dofmap, 24, axis=1).ravel()
    cols = np.tile(dofmap, (1, 24)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(3 * nn, 3 * nn)).tocsc()

    mlump = np.zeros(3 * nn)
    for c in range(8):
        for d in range(3):
            np.add.at(mlump, 3 * conn[:, c] + d, rho * Ve / 8.0)
    M = sp.diags(mlump).tocsc()
    return K, M, float(Ve.sum())


def analytic_mass(spec: HexImpellerSpec) -> float:
    """총질량 해석 추정 — 슈라우드 2장 + 베인 N매(캠버선 길이 적분). 자기검증용."""
    A_ann = math.pi * (spec.b ** 2 - spec.a ** 2)
    m_shroud = spec.rho * 2 * A_ann * spec.t_sheet
    if spec.vane_mode == "web":
        return m_shroud + spec.rho * A_ann * spec.channel
    r = np.linspace(spec.a, spec.b, 400)
    wrap = math.radians(spec.wrap_deg)
    ds = np.sqrt(1.0 + (wrap / (spec.b - spec.a)) ** 2 * r ** 2)
    L_v = float(np.trapezoid(ds, r))
    return m_shroud + spec.rho * spec.n_vane * L_v * spec.t_vane * spec.channel


def solve_free_free(spec: HexImpellerSpec, coors=None, conn=None, n_modes: int = 20,
                    sigma: float = -1.0e5, rigid_tol_hz: float = 1.0,
                    mesh_info: dict | None = None):
    """자유-자유 건조 모달 → (`rail3d.ModalResult`, info).

    자유-자유는 K가 특이하므로 shift-invert의 sigma를 **음수**로 둔다(sigma=0이면 분해 실패).
    반환은 `rail3d.ModalResult`로 맞춰 `rail3d.azimuthal_orders`·`group_pairs`를 그대로 쓴다.
    """
    if coors is None or conn is None:
        coors, conn, minfo = mesh(spec)
    elif mesh_info is not None:
        minfo = mesh_info
    else:                                  # 격자 색인이 없으면 대칭연산을 만들 수 없다
        _, _, minfo = mesh(spec)
    K, M, vol = assemble(spec, coors, conn)
    vals, vecs = spla.eigsh(K, k=n_modes, M=M, sigma=sigma, which="LM")
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]
    freqs = np.sqrt(np.clip(vals, 0.0, None)) / (2 * math.pi)
    n_rigid = int(np.sum(freqs < rigid_tol_hz))
    res = r3.ModalResult(freqs=freqs, shapes=vecs, coors=coors, ndof=K.shape[0],
                         field_coors=coors, full_shapes=vecs.T.copy())
    info = dict(minfo)
    info.update({"n_rigid": n_rigid, "mass_fem_g": spec.rho * vol * 1e3,
                 "mass_analytic_g": analytic_mass(spec) * 1e3,
                 "f_first_elastic_Hz": float(freqs[n_rigid]) if n_rigid < len(freqs)
                 else float("nan")})
    info["_M_diag"] = M.diagonal()
    return res, info


# ------------------------------------------------------------------ 순환대칭 판정
def sector_permutation(spec: HexImpellerSpec, grid_idx: np.ndarray,
                       n_sector: int = 1) -> np.ndarray:
    """대칭연산 R(2π n_sector / N)의 **절점 순열** — `nxt[p]` = θ_p + α 위치의 절점.

    `n_theta % n_vane == 0`이므로 회전이 방위 셀 경계에 정확히 떨어지고, 구조가 정확히
    C_N 대칭이면 회전된 절점집합이 원래 집합과 **일치**한다(일치하지 않으면 예외).
    """
    shift = (spec.n_theta // max(spec.n_vane, 1)) * n_sector
    ir, it, iz = grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]
    key = (iz.astype(np.int64) * spec.n_theta + it) * (spec.n_r + 1) + ir
    key_rot = (iz.astype(np.int64) * spec.n_theta
               + (it + shift) % spec.n_theta) * (spec.n_r + 1) + ir
    order = np.argsort(key)
    pos = np.searchsorted(key[order], key_rot)
    if pos.max() >= key.size or not np.array_equal(key[order][pos], key_rot):
        raise RuntimeError("회전된 절점집합이 원래 집합과 다르다 — 구조가 C_N 대칭이 아니다")
    return order[pos]


def harmonic_indices(spec: HexImpellerSpec, res: "r3.ModalResult",
                     grid_idx: np.ndarray, m_diag: np.ndarray | None = None):
    """각 모드의 **순환조화지수 h**를 대칭연산 하나로 판정한다 — 주파수를 쓰지 않는다.

    C_N 구조의 고유공간은 조화지수 h의 기약표현이다. 회전 R = R(2π/N)이 그 공간에서 각
    2πh/N의 회전으로 작용하므로, 질량정규 내적으로

        c_k = ⟨u_k, M R u_k⟩ / ⟨u_k, M u_k⟩ = cos(2π h / N)

    이고 이 값은 **고유공간 안의 기저 선택과 무관**하다. 따라서

        c = +1  → h = 0        (1차원 표현 = **singlet**)
        c = −1  → h = N/2      (N 짝수일 때만; 1차원 표현 = **singlet**)
        |c| < 1 → 0 < h < N/2  (2차원 표현 = **doublet**), h = (N/2π)·arccos(c)

    이것이 정본 §3.6의 "형상으로 판정, 주파수 순서로 판정 금지"를 대칭성 판정에 적용한 형태다.
    아울러 h는 **공간 조화 m의 엄침(aliasing) 류(class)** 이기도 하다: m ≡ ±h (mod N)인 모든 m이
    같은 h를 가지므로, N=6에서 m=2와 m=4는 같은 h=2 표현에 속한다.

    반환: (h_hat[n_modes], c[n_modes], degeneracy[n_modes]) — degeneracy는 1 또는 2.
    """
    if res.full_shapes is None:
        raise RuntimeError("full_shapes 필요")
    nxt = sector_permutation(spec, grid_idx, 1)
    ang = 2 * math.pi / spec.n_vane
    Q = np.array([[math.cos(ang), -math.sin(ang), 0.0],
                  [math.sin(ang), math.cos(ang), 0.0], [0.0, 0.0, 1.0]])
    w = (np.ones(grid_idx.shape[0]) if m_diag is None
         else np.asarray(m_diag).reshape(-1, 3)[:, 0])
    hs, cs, degs = [], [], []
    for k in range(res.full_shapes.shape[0]):
        u = res.full_shapes[k].reshape(-1, 3)
        ru = np.empty_like(u)
        ru[nxt] = u @ Q.T                      # (Ru)[θ+α] = Q u[θ]
        num = float(np.sum(w[:, None] * u * ru))
        den = float(np.sum(w[:, None] * u * u))
        c = num / den if den > 0 else 0.0
        c = min(max(c, -1.0), 1.0)
        h = (spec.n_vane / (2 * math.pi)) * math.acos(c)
        hs.append(h)
        cs.append(c)
        # **대칭이 보호하는** 다중도. h = 0 또는 N/2는 1차원 표현이므로 C_N은 축퇴를
        # 보호하지 않는다 — 축대칭 구조에서 그 조화가 겹쳐 보이는 것은 O(2)가 보호하는
        # 우연축퇴이고, C_N 섭동이 정확히 그것을 갈라놓는다. 관측된 주파수 일치와
        # 구별해서 써야 한다(그래서 이름이 protected).
        degs.append(1 if min(abs(c - 1.0), abs(c + 1.0)) < 1e-6 else 2)
    return np.array(hs), np.array(cs), np.array(degs, dtype=int)


def radial_harmonic_profiles(spec: HexImpellerSpec, res: "r3.ModalResult",
                             grid_idx: np.ndarray, m: int, z_frac: float = 0.3):
    """모드별 **복소 반경 프로파일** c_m(r) = Σ_θ u_z(r,θ) e^{−imθ} (전면 슈라우드 절점).

    구조격자이므로 반경 링이 정확히 (n_r+1)개이고 각 링의 θ가 균등하다 — 링별 방위 투영이
    정확한 이산 푸리에 계수가 된다. 반경 형상과 방위 위상을 **분리**해서 볼 수 있으므로,
    같은 축대칭 doublet에서 갈라진 두 singlet을 형상으로 짝지을 수 있다(주파수 무사용).
    """
    c = res.field_coors
    zmax, zmin = c[:, 2].max(), c[:, 2].min()
    sel = np.nonzero(c[:, 2] >= zmin + (1.0 - z_frac) * (zmax - zmin))[0]
    ir = grid_idx[sel, 0]
    th = np.arctan2(c[sel, 1], c[sel, 0])
    ph = np.exp(-1j * m * th)
    out = np.zeros((res.full_shapes.shape[0], spec.n_r + 1), dtype=complex)
    for k in range(res.full_shapes.shape[0]):
        uz = res.full_shapes[k].reshape(-1, 3)[sel, 2]
        np.add.at(out[k], ir, uz * ph)
    return out


def partner_overlap(prof: np.ndarray, k: int, l: int, m: int):
    """두 모드가 **같은 축대칭 축퇴쌍에서 갈라진 짝**인가 — 형상 기준 두 지표.

    같은 doublet의 두 성분은 반경 프로파일이 (거의) 같고 방위 위상만 π/(2m) 어긋난다:
        c_k(r) ≈ A(r) e^{i m ψ_k},  c_l(r) ≈ A(r) e^{i m ψ_l},  |ψ_k − ψ_l| = π/(2m)
    따라서 정규화 내적 |⟨c_k, c_l⟩|이 1에 가깝고 그 위상이 m·Δψ = π/2가 된다.
    반경차수가 다른 두 모드는 A(r)이 다르므로 내적이 1보다 뚜렷히 작다.

    반환: (overlap, dpsi_deg) — overlap ∈ [0,1], dpsi_deg = 방위 위상차[deg].
    """
    a, b = prof[k], prof[l]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0, float("nan")
    ip = complex(np.vdot(a, b) / (na * nb))
    return float(abs(ip)), float(math.degrees(np.angle(ip) / m))


def cyclic_symmetry_rows(spec: HexImpellerSpec, res: "r3.ModalResult",
                         info: dict, m_max: int = 10) -> list[dict]:
    """A13 산출 행 — 모드별 조화지수·축퇴수·지배 공간조화·엄침 동반차수."""
    grid_idx = info["grid_idx"]
    h, c, deg = harmonic_indices(spec, res, grid_idx, info.get("_M_diag"))
    A, _ = r3.order_amplitudes(res, m_max=m_max)
    orders, psi, purity = r3.azimuthal_orders(res, m_max=m_max)
    nrig = info["n_rigid"]
    rows = []
    for k in range(len(res.freqs)):
        share = A[k] / max(A[k].sum(), 1e-300)
        h_int = int(round(float(h[k])))
        # 이 조화지수에 엄침되는 공간차수: m ≡ ±h (mod N)
        alias = sorted({m for m in range(m_max + 1)
                        if (m - h_int) % spec.n_vane == 0
                        or (m + h_int) % spec.n_vane == 0})
        rows.append({
            "n_vane": spec.n_vane, "vane_mode": spec.vane_mode,
            "vane_arc_cells": (-1 if spec.vane_arc_cells is None
                               else spec.vane_arc_cells),
            "modulation_depth": spec.modulation_depth,
            "wrap_deg": spec.wrap_deg,
            "n_theta": spec.n_theta, "n_r": spec.n_r, "n_dof": info["n_dof"],
            "mode": k, "rigid": bool(k < nrig), "f_Hz": float(res.freqs[k]),
            "c_rotation": float(c[k]), "h_index": h_int,
            "h_index_raw": float(h[k]), "deg_protected": int(deg[k]),
            "m_dominant": int(orders[k]), "m_share": float(share[orders[k]]),
            "alias_orders": "|".join(str(m) for m in alias),
            "alias_share": float(sum(share[m] for m in alias)),
            "psi_deg": float(math.degrees(psi[k])),
            "purity": float(purity[k]),
        })
    return rows


def splitting_summary(spec: HexImpellerSpec, res: "r3.ModalResult", info: dict,
                      m_max: int = 10) -> dict:
    """C_N 섭동이 **보호되지 않는** 조화(h = N/2)를 얼마나 갈라놓는가 + 인공물 floor.

    두 양을 함께 낸다 —
      `split_hN2_*`   h = N/2 짝(형상으로 짝지음: `partner_overlap`이 최대인 두 singlet)의
                      주파수 분리. 이것이 **물리적** 분리다.
      `floor_*`       h가 1..N/2−1인 **대칭보호 doublet**의 관측 분리 = 이 격자·고유해의
                      수치 인공물 상한. 물리 분리를 이 floor와 대조해야 판정이 성립한다
                      (F62의 교훈: 사면체 격자에서는 floor가 0.04–0.96 % of f였다).
    """
    grid_idx = info["grid_idx"]
    h, c, deg = harmonic_indices(spec, res, grid_idx, info.get("_M_diag"))
    f = res.freqs
    nrig = info["n_rigid"]
    el = np.arange(nrig, len(f))
    h_int = np.rint(h).astype(int)

    # --- 대칭보호 doublet의 관측 분리 = 인공물 floor
    floors = []
    for hh in range(1, spec.n_vane // 2 + (spec.n_vane % 2)):
        grp = [k for k in el if h_int[k] == hh and deg[k] == 2]
        for i in range(0, len(grp) - 1, 2):
            a, b = grp[i], grp[i + 1]
            fm = 0.5 * (f[a] + f[b])
            floors.append(abs(f[b] - f[a]) / fm)
    out = {"n_protected_pairs": len(floors),
           "floor_split_rel_max": float(max(floors)) if floors else float("nan"),
           "floor_split_rel_median": float(np.median(floors)) if floors
           else float("nan")}

    # --- h = N/2 짝(N 짝수일 때만 존재)
    out.update({"h_half": -1, "split_hN2_lo_Hz": float("nan"),
                "split_hN2_hi_Hz": float("nan"), "split_hN2_Hz": float("nan"),
                "split_hN2_rel": float("nan"), "partner_overlap": float("nan"),
                "partner_dpsi_deg": float("nan")})
    if spec.n_vane % 2 == 0:
        hh = spec.n_vane // 2
        grp = [k for k in el if h_int[k] == hh]
        out["h_half"] = hh
        if len(grp) >= 2:
            prof = radial_harmonic_profiles(spec, res, grid_idx, hh)
            best = None
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    ov, dps = partner_overlap(prof, grp[i], grp[j], hh)
                    if best is None or ov > best[0]:
                        best = (ov, dps, grp[i], grp[j])
            ov, dps, a, b = best
            fm = 0.5 * (f[a] + f[b])
            out.update({"split_hN2_lo_Hz": float(f[a]),
                        "split_hN2_hi_Hz": float(f[b]),
                        "split_hN2_Hz": float(f[b] - f[a]),
                        "split_hN2_rel": float((f[b] - f[a]) / fm),
                        "partner_overlap": ov, "partner_dpsi_deg": dps})
    return out


def radial_order(prof_k: np.ndarray, edge_trim: int = 1) -> int:
    """반경 프로파일 → **nodal circle 수 n**. 방위차수 m과 함께 모드족을 정한다.

    복소 프로파일 c_m(r)의 위상을 최대진폭 링에 맞춰 정렬한 뒤 실부의 **내부 부호변화**를 센다.
    가장자리 링은 이산화 잡음이 크므로 `edge_trim`개씩 버린다. 진폭이 최대의 5 % 미만인 링은
    부호가 의미 없으므로 제외한다 — 그러지 않으면 절점 근방의 수치잡음이 가짜 교차를 만든다.

    **왜 필요한가**(설계서 F107): 같은 순환조화지수 h를 공유한다고 두 모드가 같은 정보를 주지
    않는다. h는 모드가 *어떻게 변환하는가*만 말하고, 실제로 다른 관측량인지는 (m, n) 족이
    다른지가 정한다. m = 4 우세모드가 m = 2와 다른 족이면 관측량에서 지울 수 없다.
    """
    a = np.abs(prof_k)
    if a.max() <= 0:
        return -1
    k0 = int(np.argmax(a))
    v = np.real(prof_k * np.exp(-1j * np.angle(prof_k[k0])))
    if edge_trim:
        v, a = v[edge_trim:-edge_trim or None], a[edge_trim:-edge_trim or None]
    keep = a > 0.05 * a.max()
    v = v[keep]
    if v.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(np.sign(v)) != 0))


def harmonic_indices_grouped(spec: HexImpellerSpec, res: "r3.ModalResult",
                             grid_idx: np.ndarray, m_diag=None,
                             rel_tol: float = 1e-4):
    """축퇴군 단위로 h를 판정한다 — **기저 무관** 판정 (외부 검토 6차).

    `harmonic_indices`는 벡터 하나의 Rayleigh 몫 ⟨u, M R u⟩/⟨u, M u⟩을 쓴다. 축퇴 부분공간이
    정확히 2차원이고 그 위에서 R이 순수 회전이면 그 값은 기저와 무관하지만, 실제로는 (i) 주파수가
    거의 겹치는 다른 표현이 섞이거나 (ii) 병렬 BLAS 리덕션의 비결합성으로 고유해가 돌려주는
    기저가 실행마다 달라지면(F111) 값이 흔들린다. 실제로 `a17`의 m2n0 쌍에서 두 멤버가 h = 2와
    h = 1로 다르게 읽혔다.

    이 함수는 **군 전체의 자취**를 쓴다: 축퇴군 G에 대해 C_ij = ⟨u_i, M R u_j⟩를 만들고
    c = tr(C)/|G|를 취한다. 자취는 부분공간 안의 기저변환에 불변이므로 판정이 실행에 의존하지
    않는다 — 2차원 표현에서 tr(C)/2 = cos(2πh/N)이 정확히 성립한다.

    반환: (h[n_modes], c[n_modes], degeneracy[n_modes], group_id[n_modes]).
    """
    if res.full_shapes is None:
        raise RuntimeError("full_shapes 필요")
    nxt = sector_permutation(spec, grid_idx, 1)
    ang = 2 * math.pi / spec.n_vane
    Q = np.array([[math.cos(ang), -math.sin(ang), 0.0],
                  [math.sin(ang), math.cos(ang), 0.0], [0.0, 0.0, 1.0]])
    w = (np.ones(grid_idx.shape[0]) if m_diag is None
         else np.asarray(m_diag).reshape(-1, 3)[:, 0])
    U = [res.full_shapes[k].reshape(-1, 3) for k in range(res.full_shapes.shape[0])]
    RU = []
    for u in U:
        ru = np.empty_like(u)
        ru[nxt] = u @ Q.T
        RU.append(ru)
    f = np.asarray(res.freqs, dtype=float)
    # 주파수로 축퇴군을 만든다(판정이 아니라 **묶기**에만 쓴다 — 군 안에서는 형상으로 본다)
    gid = np.zeros(len(f), dtype=int)
    g = 0
    for k in range(1, len(f)):
        gid[k] = gid[k - 1] if abs(f[k] - f[k - 1]) <= rel_tol * max(f[k], 1.0) else (g := g + 1)
    hs = np.zeros(len(f))
    cs = np.zeros(len(f))
    degs = np.ones(len(f), dtype=int)
    for gg in np.unique(gid):
        idx = np.nonzero(gid == gg)[0]
        num = sum(float(np.sum(w[:, None] * U[i] * RU[i])) for i in idx)
        den = sum(float(np.sum(w[:, None] * U[i] * U[i])) for i in idx)
        c = min(max(num / den if den > 0 else 0.0, -1.0), 1.0)
        h = (spec.n_vane / (2 * math.pi)) * math.acos(c)
        for i in idx:
            hs[i], cs[i], degs[i] = h, c, len(idx)
    return hs, cs, degs, gid
