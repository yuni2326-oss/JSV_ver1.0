"""B2·B3 — sfepy 3D 솔리드 레일 (정본 §3.6, inverse-crime 회피의 1차 레일).

역식별에 쓰는 모델(P: Kirchhoff + 1차섭동)과 **모델형식·이산화가 모두 다른** 3D 선형탄성으로
같은 물리대상을 푼다. 손상은 modulus 패치가 아니라 **실제 재료제거**(요소 삭제)로 넣는다 —
리뷰가 명시적으로 요구한 as-built 조건.

  B2  베인 직육면체 + EDM 커프 슬롯  → 곡률-null 실명의 3D 탄성 검정, 노치→c_θ 등가
  B3  환형판 + 포켓                  → 쌍 평균·분리·배향의 ground truth, model-form penalty

메시는 구조격자 육면체(3_8)를 numpy로 직접 만든다(gmsh 불필요, 형상이 단순하고 해석해가 있음).
2차 근사(order=2)를 기본으로 쓴다 — 얇은 굽힘에서 1차 육면체는 전단잠김이 심하다.
고유치는 shift-invert(`eigsh(K, M, sigma=0)`)로 푼다: sfepy 기본 `which='SM'`은 수렴하지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh

_HEX_OFFSETS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))


def _grid_nodes(xs, ys, zs):
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _hex_conn(nx, ny, nz, keep=None, wrap_y: bool = False):
    """구조격자 육면체 연결 — (nx,ny,nz)는 **요소 수**. keep(i,j,k)->bool로 요소 제거.

    wrap_y=True면 y 인덱스가 주기적(환형의 방위방향).
    """
    ny_nodes = ny if wrap_y else ny + 1

    def nid(i, j, k):
        jj = j % ny if wrap_y else j
        return (i * ny_nodes + jj) * (nz + 1) + k

    conn = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if keep is not None and not keep(i, j, k):
                    continue
                conn.append([nid(i + di, j + dj, k + dk)
                             for di, dj, dk in _HEX_OFFSETS])
    return np.array(conn, dtype=np.int32)


def _compact(coors, conn):
    """고아 절점 제거 + 재번호(제거된 요소 때문에 생긴 특이 질량행렬 방지)."""
    used = np.unique(conn)
    remap = -np.ones(coors.shape[0], dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return coors[used], remap[conn]


def vane_mesh(L=0.030, w=0.0024, h=0.0012, nx=60, ny=4, nz=8,
              kerf: dict | None = None):
    """베인 직육면체 메시 (+ 선택적 EDM 커프 슬롯).

    kerf = {"xc_over_L":0.2, "width":0.0003, "depth_frac":0.5} — x_c 근처 요소를
    **깊이비만큼 위쪽부터** 제거한다(슬롯). 폭은 요소 크기로 이산화된다.
    """
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(-0.5 * w, 0.5 * w, ny + 1)
    zs = np.linspace(-0.5 * h, 0.5 * h, nz + 1)
    coors = _grid_nodes(xs, ys, zs)

    keep = None
    kerf_info = {}
    if kerf:
        xc = kerf["xc_over_L"] * L
        half = 0.5 * kerf["width"]
        n_rm = max(1, int(round(kerf["depth_frac"] * nz)))
        xmid = 0.5 * (xs[:-1] + xs[1:])
        in_kerf = np.abs(xmid - xc) <= half + 1e-12
        if not in_kerf.any():                      # 폭이 요소보다 작으면 최근접 1열
            in_kerf[np.argmin(np.abs(xmid - xc))] = True

        def keep(i, j, k):                          # noqa: F811
            return not (in_kerf[i] and k < n_rm)    # 아래쪽(z 최소)부터 제거 = 인장면

        kerf_info = {"n_layers_removed": n_rm, "depth_actual": n_rm / nz,
                     "width_actual": float(in_kerf.sum() * (L / nx))}
    conn = _hex_conn(nx, ny, nz, keep=keep)
    coors, conn = _compact(coors, conn)
    return coors, conn, kerf_info


def disk_mesh(a=0.0154, b=0.03656, t=0.0016, nr=10, ntheta=64, nz=4,
              pocket: dict | None = None):
    """환형판 메시(방위 주기) (+ 선택적 포켓 = 재료제거).

    pocket = {"r1","r2","theta0","dtheta","depth_frac"} — 위쪽 z층부터 깊이비만큼 제거.
    """
    rs = np.linspace(a, b, nr + 1)
    ths = np.linspace(0.0, 2 * math.pi, ntheta, endpoint=False)
    zs = np.linspace(-0.5 * t, 0.5 * t, nz + 1)
    R, TH, Z = np.meshgrid(rs, ths, zs, indexing="ij")
    coors = np.stack([(R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel(),
                      Z.ravel()], axis=1)

    keep = None
    info = {}
    if pocket:
        rmid = 0.5 * (rs[:-1] + rs[1:])
        dth = 2 * math.pi / ntheta
        thmid = ths + 0.5 * dth
        in_r = (rmid >= pocket["r1"] - 1e-12) & (rmid <= pocket["r2"] + 1e-12)
        dd = np.angle(np.exp(1j * (thmid - pocket["theta0"])))
        in_t = np.abs(dd) <= 0.5 * pocket["dtheta"] + 1e-12
        n_rm = max(1, int(round(pocket["depth_frac"] * nz)))

        def keep(i, j, k):                          # noqa: F811
            return not (in_r[i] and in_t[j] and k >= nz - n_rm)

        r1_act = float(rs[:-1][in_r].min()) if in_r.any() else float("nan")
        r2_act = float(rs[1:][in_r].max()) if in_r.any() else float("nan")
        # 방위 실현중심: 선택된 셀들의 중심각 편차 범위에서 되찾는다(0/2π 감김 안전).
        if in_t.any():
            dsel = dd[in_t]
            th0_act = float(pocket["theta0"] + 0.5 * (dsel.min() + dsel.max()))
        else:
            th0_act = float("nan")
        tol_r = 1e-9
        info = {"n_layers_removed": n_rm, "depth_actual": n_rm / nz,
                "n_r_cells": int(in_r.sum()), "n_theta_cells": int(in_t.sum()),
                "dtheta_actual": float(in_t.sum() * dth),
                "theta0_actual": th0_act,
                "r1_actual": r1_act, "r2_actual": r2_act,
                # 형상보존 진단(설계서 F11′): 요청한 포켓이 셀면에 정확히 실현됐는가.
                "snap_r1_mm": (r1_act - pocket["r1"]) * 1e3,
                "snap_r2_mm": (r2_act - pocket["r2"]) * 1e3,
                "snap_dtheta_deg": math.degrees(in_t.sum() * dth - pocket["dtheta"]),
                "snap_theta0_deg": math.degrees(th0_act - pocket["theta0"]),
                "snap_depth": n_rm / nz - pocket["depth_frac"]}
        info["shape_exact"] = bool(
            abs(r1_act - pocket["r1"]) < tol_r and abs(r2_act - pocket["r2"]) < tol_r
            and abs(in_t.sum() * dth - pocket["dtheta"]) < 1e-9
            and abs(th0_act - pocket["theta0"]) < 1e-9
            and abs(n_rm / nz - pocket["depth_frac"]) < 1e-12)
    conn = _hex_conn(nr, ntheta, nz, keep=keep, wrap_y=True)
    coors, conn = _compact(coors, conn)
    return coors, conn, info


@dataclass
class ModalResult:
    freqs: np.ndarray
    shapes: np.ndarray            # (n_dof_reduced, n_modes)
    coors: np.ndarray
    ndof: int
    field_coors: np.ndarray | None = None   # 장(field) 절점 좌표 (고차 절점 포함)
    full_shapes: np.ndarray | None = None   # (n_modes, n_dof_full) 구속 포함 확장
    #: 모드별 **구역 변형에너지 분율** uᵀK_R u / uᵀK u (`solve_modes(region_mask=…)`)
    region_energy_frac: np.ndarray | None = None
    #: 모드별 **구역 운동에너지 분율** uᵀM_R u / uᵀM u — 감육의 질량항 감도
    region_kinetic_frac: np.ndarray | None = None


def order_amplitudes(res: "ModalResult", m_max: int = 6, z_frac: float = 0.3):
    """모드별 방위 푸리에 진폭행렬 `A[k, m]`과 위상 `PHI[k, m]`.

    상부면 근처 절점의 면외변위 u_z를 방위 조화 exp(−i m θ)로 투영한다. m=0은 켤레쌍이
    없어 2배 과대평가되므로 ½을 곱해 다른 차수와 통약한다.
    """
    if res.full_shapes is None or res.field_coors is None:
        raise RuntimeError("solve_modes(..., keep_shapes=True)로 풀어야 한다")
    c = res.field_coors
    zmax, zmin = c[:, 2].max(), c[:, 2].min()
    sel = c[:, 2] >= zmin + (1.0 - z_frac) * (zmax - zmin)
    th = np.arctan2(c[sel, 1], c[sel, 0])
    n_modes = res.full_shapes.shape[0]
    A = np.empty((n_modes, m_max + 1))
    PHI = np.empty((n_modes, m_max + 1))
    for k in range(n_modes):
        uz = res.full_shapes[k].reshape(-1, 3)[:, 2][sel]
        coefs = np.array([np.sum(uz * np.exp(-1j * m * th)) for m in range(m_max + 1)])
        amps = np.abs(coefs)
        amps[0] *= 0.5
        A[k] = amps
        PHI[k] = np.angle(coefs)
    return A, PHI


def azimuthal_orders(res: "ModalResult", m_max: int = 6, z_frac: float = 0.3):
    """각 모드의 **방위차수 m과 배향**을 모드형에서 직접 판정한다.

    정본 §3.6은 "주파수 순서로 모드를 짝지어서는 안 된다(분리·veering·mixing으로 순서가
    뒤바뀐다)"고 못박는다. 따라서 3D 결과의 쌍 매칭은 형상 기반이어야 한다.

    반환: (orders[n_modes], psi[n_modes], purity[n_modes]) — purity는 지배 성분 비율.
    """
    A, PHI = order_amplitudes(res, m_max=m_max, z_frac=z_frac)
    orders, psis, purity = [], [], []
    for k in range(A.shape[0]):
        m_hat = int(np.argmax(A[k]))
        psi = (-PHI[k, m_hat] / m_hat) % (math.pi / m_hat) if m_hat > 0 else 0.0
        orders.append(m_hat)
        psis.append(float(psi))
        purity.append(float(A[k, m_hat] / max(A[k].sum(), 1e-300)))
    return np.array(orders), np.array(psis), np.array(purity)


def match_order(res: "ModalResult", m: int, n_take: int = 1, m_max: int = 6,
                z_frac: float = 0.3, purity_min: float = 0.5) -> dict:
    """방위차수 m의 지배 모드를 **형상으로** 고른다 — 주파수 근접 폴백 금지(정본 §3.6).

    `azimuthal_orders`의 argmax 분류(모드→차수)를 뒤집어 **차수→모드**로 고른다:
    m차 성분 진폭 `A[:, m]`이 큰 순으로 `n_take`개를 취하고, 그 모드들이 실제로 m차
    지배인지(자기 argmax = m)와 순도(purity = A[k,m]/ΣA[k])를 함께 돌려준다.
    포켓이 축대칭을 강하게 깨면 순수 m=0 모드가 아예 없을 수 있고, 그때는
    `matched=False`가 되어 **호출자가 그 셀을 버려야 한다**(억지 짝짓기 금지, F21).

    반환 dict: idx(주파수 오름차순), purity, amp, is_dominant, matched.
    """
    A, PHI = order_amplitudes(res, m_max=m_max, z_frac=z_frac)
    col = A[:, m]
    take = np.argsort(col)[::-1][:n_take]
    idx = sorted(int(i) for i in take)              # freqs가 오름차순이므로 인덱스순=주파수순
    pur = [float(A[i, m] / max(A[i].sum(), 1e-300)) for i in idx]
    dom = [bool(int(np.argmax(A[i])) == m) for i in idx]
    if m > 0:
        period = math.pi / m
        psi = [float((-PHI[i, m] / m) % period) for i in idx]
    else:
        psi = [0.0] * len(idx)
    return {"idx": idx, "purity": pur, "amp": [float(col[i]) for i in idx],
            "psi": psi, "is_dominant": dom,
            "matched": bool(len(idx) == n_take and all(dom)
                            and min(pur) >= purity_min)}


def field_node_map(res_h: "ModalResult", res_d: "ModalResult", tol: float = 1e-9):
    """손상 메시의 장절점 → 건전 메시의 장절점 색인. 요소제거는 절점만 없애므로
    남은 절점은 좌표가 **정확히** 일치한다(같은 구조격자에서 생성)."""
    from scipy.spatial import cKDTree
    if res_h.field_coors is None or res_d.field_coors is None:
        raise RuntimeError("solve_modes(..., keep_shapes=True) 필요")
    dist, idx = cKDTree(res_h.field_coors).query(res_d.field_coors)
    return idx, dist < tol


def subspace_mac_match(res_h: "ModalResult", idx_h, res_d: "ModalResult",
                       n_take: int = 2, mac_min: float = 0.8) -> dict:
    """**subspace MAC**로 손상 모드를 건전 (축퇴)부분공간에 짝짓는다 — 정본 §3.6의 규약.

    정본은 "모드 매칭은 MAC / subspace MAC / 변형에너지 중첩으로 하고 **절대 주파수
    순서로 하지 않는다**"고 못박는다. 방위차수 투영(`match_order`)은 방위 성분만 보므로
    포켓에 국소화된 모드가 낮은 m 성분을 크게 가질 때 진짜 짝을 밀어낼 수 있다
    (실측: ξ=0.2·Δθ=30°에서 η̄_{m=1}이 이웃 셀의 9.5배로 튀었다). MAC은 반경형상까지
    보므로 그 오검출을 걸러낸다.

        MAC_sub(k) = ‖P_H u_k‖² / ‖u_k‖²,  P_H = 건전 부분공간(공유절점 제한) 정사영

    반환 dict: idx(주파수순), mac, matched. `matched=False`면 **호출자가 그 셀을 버려야
    한다** — 억지 짝짓기 금지(설계서 F21).
    """
    idx, ok = field_node_map(res_h, res_d)
    hid = idx[ok]
    H = np.stack([res_h.full_shapes[j].reshape(-1, 3)[hid].ravel() for j in idx_h])
    Q, _ = np.linalg.qr(H.T)                       # 직교정규 기저 (N, p)
    macs = []
    for k in range(res_d.full_shapes.shape[0]):
        u = res_d.full_shapes[k].reshape(-1, 3)[ok].ravel()
        nn = float(u @ u)
        pr = Q.T @ u
        macs.append(float((pr @ pr) / nn) if nn > 0 else 0.0)
    macs = np.array(macs)
    take = np.argsort(macs)[::-1][:n_take]
    sel = sorted(int(i) for i in take)
    return {"idx": sel, "mac": [float(macs[i]) for i in sel],
            "mac_all": macs,
            "matched": bool(len(sel) == n_take and min(macs[i] for i in sel) >= mac_min)}


def beam_mode_kinds(res: "ModalResult") -> list[str]:
    """보 모드를 형상으로 분류: "flap"(면외 z) / "edge"(폭방향 y) / "axial".

    3D 캔틸레버의 주파수 순서는 보 모델의 모드 순서와 **다르다**(예: 30×2.4×1.2 mm에서
    2번째 3D 모드는 폭방향 굽힘 2110 Hz이고 면외 2차는 6596 Hz). 정본 §3.6이 금지한
    "주파수 순서 매칭"을 피하려면 반드시 형상으로 골라야 한다.

    2D 평면 단면(x=길이, y=두께)에서는 폭방향 모드가 없으므로 "axial"/"flap"만 쓴다.
    """
    if res.full_shapes is None:
        raise RuntimeError("solve_modes(..., keep_shapes=True) 필요")
    dim = int(res.coors.shape[1])
    labels = {2: ["axial", "flap"], 3: ["axial", "edge", "flap"]}[dim]
    kinds = []
    for k in range(res.full_shapes.shape[0]):
        u = res.full_shapes[k].reshape(-1, dim)
        amp = np.sqrt((u ** 2).sum(axis=0))          # 성분별 RMS
        kinds.append(labels[int(np.argmax(amp))])
    return kinds


def group_pairs(freqs: np.ndarray, orders: np.ndarray) -> dict:
    """방위차수별 모드 인덱스 묶음 — m>0은 (낮은쪽, 높은쪽) 쌍, m=0은 단일."""
    out: dict[int, list[int]] = {}
    for i, m in enumerate(orders):
        out.setdefault(int(m), []).append(i)
    for m in out:
        out[m] = sorted(out[m], key=lambda i: freqs[i])
    return out


#: (공간차원, 요소 절점수) → sfepy 요소 서술자. 하드코딩("3_8")하면 사면체·2D 메시에서
#: sfepy가 8절점을 읽으려다 버퍼를 넘어 **세그폴트**한다(2026-08-02 확인) — 반드시 추론한다.
_ELEM_DESC = {(2, 3): "2_3", (2, 4): "2_4", (3, 4): "3_4", (3, 8): "3_8"}


def elem_desc(coors, conn) -> str:
    """좌표 차원과 연결도 폭에서 요소형을 추론한다(2D 삼각/사각, 3D 사면/육면)."""
    key = (int(coors.shape[1]), int(conn.shape[1]))
    desc = _ELEM_DESC.get(key)
    if desc is None:
        raise ValueError(f"지원하지 않는 (차원, 절점수): {key}")
    return desc


def solve_modes(coors, conn, E, nu, rho, fixed_select, n_modes=8,
                order=2, sigma=0.0, keep_shapes: bool = False,
                plane: str = "strain",
                region_mask: np.ndarray | None = None,
                v0_seed: int | None = 20260815) -> ModalResult:
    """선형탄성 고유해 — sfepy 조립 + scipy shift-invert. 2D·3D 공통.

    fixed_select: sfepy 영역 셀렉터 문자열(예: "vertices in (x < 1e-9)").
    **facet 영역**으로 만들어야 2차 요소의 면 중간절점 자유도까지 구속된다 —
    정점 영역만 쓰면 클램프가 물러져 f₁이 10 % 낮게 나온다(개발 중 확인).
    order=2 권장(얇은 굽힘에서 1차 육면체는 전단잠김).

    `plane`("stress"|"strain")은 **2D에서만** 의미가 있다(3D에서는 무시된다):
    좁은 보 단면은 평면응력(유효계수 E), 넓은 판 스트립은 평면변형(E/(1−ν²))에 대응한다.

    `region_mask`(요소별 bool)를 주면 그 구역에 대한 **모드 변형에너지 분율**
    uᵀK_R u / uᵀK u를 `region_energy_frac`에 함께 돌려준다. 국소 강성손실 δlnK에 대한
    1차 주파수감도가 정확히 이 분율이므로(§3.2), 서로 다른 구조에서 잰 주파수강하를
    **통약 가능한 통화**로 바꿔준다 — a19 사다리가 비율을 기전으로 바꾸는 수단.
    """
    from sfepy.base.base import output
    from sfepy.discrete import (Equation, Equations, FieldVariable, Integral,
                                Material, Problem)
    from sfepy.discrete.conditions import Conditions, EssentialBC
    from sfepy.discrete.fem import FEDomain, Field, Mesh
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.terms import Term

    from sfepy.discrete import Function, Functions

    output.set_output(quiet=True)
    mat_ids = np.zeros(conn.shape[0], dtype=np.int32)
    if region_mask is not None:
        mask = np.asarray(region_mask, dtype=bool)
        if mask.shape != (conn.shape[0],):
            raise ValueError(
                f"region_mask 길이 {mask.shape}가 요소수 {conn.shape[0]}와 다르다")
        if not mask.any():
            raise ValueError("region_mask가 비어 있다 — 구역 에너지 분율이 정의되지 않는다")
        mat_ids = mask.astype(np.int32)      # 셀 그룹 1 = 구역
    dim = int(coors.shape[1])
    desc = elem_desc(coors, conn)
    mesh = Mesh.from_data("m", coors, None, [conn], [mat_ids], [desc])
    domain = FEDomain("d", mesh)
    omega = domain.create_region("Omega", "all")
    if callable(fixed_select):
        # 셀렉터 문법은 x**2+y**2 같은 식을 못 받으므로 함수 선택자를 쓴다.
        fns = Functions([Function("fixed_fn",
                                  lambda coors, domain=None: fixed_select(coors))])
        domain.create_region("Fixed", "vertices by fixed_fn", "facet", functions=fns)
    elif fixed_select:
        domain.create_region("Fixed", fixed_select, "facet")

    field = Field.from_args("fu", np.float64, "vector", omega, approx_order=order)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")
    m = Material("m", D=stiffness_from_youngpoisson(dim, E, nu, plane=plane),
                 rho=rho)
    integral = Integral("i", order=2 * order)
    eq_k = Equation("K", Term.new("dw_lin_elastic(m.D, v, u)", integral, omega,
                                  m=m, v=v, u=u))
    eq_m = Equation("M", Term.new("dw_dot(m.rho, v, u)", integral, omega,
                                  m=m, v=v, u=u))
    eqs = [eq_k, eq_m]
    eq_kr = eq_mr = None
    if region_mask is not None:
        rgn = domain.create_region("Rgn", "cells of group 1", "cell")
        eq_kr = Equation("Kr", Term.new("dw_lin_elastic(m.D, v, u)", integral, rgn,
                                       m=m, v=v, u=u))
        eq_mr = Equation("Mr", Term.new("dw_dot(m.rho, v, u)", integral, rgn,
                                        m=m, v=v, u=u))
        eqs += [eq_kr, eq_mr]
    pb = Problem("modal", equations=Equations(eqs))
    if fixed_select:
        pb.time_update(ebcs=Conditions([
            EssentialBC("Fixed", domain.regions["Fixed"], {"u.all": 0.0})]))
    else:
        pb.time_update()
    pb.update_materials()

    mtx_k = eq_k.evaluate(mode="weak", dw_mode="matrix", asm_obj=pb.mtx_a)
    mtx_m = mtx_k.copy()
    mtx_m.data[:] = 0.0
    mtx_m = eq_m.evaluate(mode="weak", dw_mode="matrix", asm_obj=mtx_m)
    mtx_kr = mtx_mr = None
    if eq_kr is not None:
        mtx_kr = mtx_k.copy()
        mtx_kr.data[:] = 0.0
        mtx_kr = eq_kr.evaluate(mode="weak", dw_mode="matrix", asm_obj=mtx_kr)
        mtx_mr = mtx_k.copy()
        mtx_mr.data[:] = 0.0
        mtx_mr = eq_mr.evaluate(mode="weak", dw_mode="matrix", asm_obj=mtx_mr)
    K, M = mtx_k.tocsc(), mtx_m.tocsc()
    # ARPACK 시작벡터를 **명시적으로** 준다: 기본값은 난수이고 그 난수는 전역 시드로도
    # 완전히 고정되지 않아, 같은 입력이 실행마다 마지막 자리가 다른 고유값을 준다. 1차량은
    # 1e-9 수준으로 같지만 **작은 차의 비**(예: 곡률-null 모드의 강하비)는 1e-3까지 갈린다 —
    # `cli a11`의 커밋 산출물이 재현되지 않던 원인이다(2026-08-24). `v0_seed=None`이면 옛 거동.
    v0 = (np.random.default_rng(v0_seed).standard_normal(K.shape[0])
          if v0_seed is not None else None)
    vals, vecs = eigsh(K, k=n_modes, M=M, sigma=sigma, which="LM", v0=v0)
    order_idx = np.argsort(vals)
    vals, vecs = vals[order_idx], vecs[:, order_idx]
    freqs = np.sqrt(np.maximum(vals, 0.0)) / (2 * math.pi)

    region_frac = region_kin = None
    if mtx_kr is not None:
        def _frac(mtx_r, mtx_all):
            return (np.einsum("ij,ij->j", vecs, mtx_r.tocsc() @ vecs)
                    / np.einsum("ij,ij->j", vecs, mtx_all @ vecs))
        region_frac = _frac(mtx_kr, K)
        region_kin = _frac(mtx_mr, M)

    field_coors = full = None
    if keep_shapes:
        variables = pb.get_variables()
        field_coors = variables["u"].field.get_coor()
        full = np.stack([variables.make_full_vec(vecs[:, k])
                         for k in range(vecs.shape[1])])
    return ModalResult(freqs=freqs, shapes=vecs, coors=coors, ndof=K.shape[0],
                       field_coors=field_coors, full_shapes=full,
                       region_energy_frac=region_frac,
                       region_kinetic_frac=region_kin)


def clamp_inner_rim(a: float, tol: float = 1e-7):
    """환형판 내경 클램프 — 함수 선택자(반경 조건은 셀렉터 문법으로 표현 불가)."""
    def fn(coors):
        rr = np.hypot(coors[:, 0], coors[:, 1])
        return np.nonzero(rr <= a + tol)[0]
    return fn


def clamp_root(x0: float = 0.0, tol: float = 1e-9) -> str:
    """캔틸레버 고정단(x=x0) 셀렉터."""
    return f"vertices in (x < {x0 + tol:.12e})"
