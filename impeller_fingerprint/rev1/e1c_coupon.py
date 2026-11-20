"""E1-C — 베인 쿠폰 open-notch 시험의 사전 설계와 수치 예측 (v2 계획서 §7.3; R1-P4, R2-Q3 부품 수준).

시편: SUS304 스트립 30 × 3.0 × 1.0 mm(유효길이 L = 30 mm, 폭 b = 3.0 mm = 깃 높이, 두께 h = 1.0 mm — G3 정본
2026-10-30, 2번 시편 깃), 고정–자유. (G1 30 × 3.2 × 1.2 mm, 08-13 30 × 4.1 × 1.0 mm는 r4.1·r3.9 산출 재현용.)
재료는 정본 `geometry.VANE`(E = 193 GPa, ν = 0.29, ρ = 8000 kg m⁻³)와 같다 — 원고 f₁ = 881.6 Hz(G3; G1 1057.9 Hz).
노치: x_c/L ≈ 0.2(또는 mode-2 곡률 null), 폭 전체를 가로지르는 노치, 깊이 ā = 0.3·0.5(G3 0.30·0.50 mm),
같은 쿠폰에서 ā 0.3 → 0.5로 깊혀 가는 중첩 시퀀스.

이 모듈은 **실험 전에** 다음을 고정한다(시험 결과를 보고 바꾸지 않는다):

  healthy3d  건전 쿠폰 3D 솔리드(sfepy 2차 육면체) 모드지도 — 면외굽힘(flap)·면내굽힘(edge)·비틀림·축을
             **형상으로** 분류(주파수 순서 매칭 금지) + 격자수렴 + EB/Timoshenko/2D/비틀림 해석해 대조
  notch      노치 예측 — Timoshenko 전달(Tada·Dimarogonas), 2D 평면응력(폭 0 슬릿, 평바닥 커프,
             둥근 선단 U-노치 = 와이어 EDM), 3D 솔리드 평바닥 슬롯(폭 전체). 각 경우 Δf₁₋₃/f,
             D₂₁, 공통모드 제거 관측량 r₂₁ = δ₂ − δ₁, r₃₁ = δ₃ − δ₁, Δf₁에 다시 맞춘 매끄러운 비교장과 대비(N3)
  tolerance  노치 위치 오차(±0.3·±0.6 mm), 클램프 가장자리(유효 뿌리) 이동, 뿌리 회전강성 k̃ 민감도
  lookup     건전 측정비 f₂/f₁·f₃/f₁ → k̃ → mode-2 null 위치(노치를 측정한 null에 둔다) 조회표
  measure    측정 요건 — 재장착 산포 시나리오별 반복수, 감쇠 정현파 주파수 추정 CRLB, 센서·테이프 질량,
             온도(공통모드) 처리
  invert     깊이 역산 앵커 — 2D/3D 노치 '진실'을 Tada·Dimarogonas 날카로운-균열 역맵으로 읽은 등가깊이
  nodes      해석 EB 변위 마디·곡률 영점(측정점·노치 배치)
  negctl     음성 대조 x_c/L = 0.35(지문이 없어야 하는 위치)의 같은 모델 예측
분석기 `fit_damped_sine`(한 번 타격의 감쇠 정현파 최소제곱)도 시험 전에 고정한다.

정본 모듈(`crack_shear`·`crack2d`·`rail3d`·`eb_reference`·`geometry`)과 N3(`n3_crack`)는 **읽기만** 한다.
δ_m = Δf_m/f_m(부호 포함, 음수 = 강하). 잡음 규약은 N3와 같다(기준선 대비 단일모드 이동의 표준편차).

실행:
    PYTHONPATH=. OMP_NUM_THREADS=2 python -m impeller_fingerprint.rev1.e1c_coupon <item> [--workers 6]
      (저장소 루트에서; 출력 루트는 PAPER3_OUT — G3 정본 산출은 별도 루트에 내고 docs/_generated로 옮겼다)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from .. import crack2d as c2
from .. import crack_shear as cs
from .. import geometry as geo
from .. import rail3d as r3
from . import n3_crack as n3
from . import out_csv, out_fig

# ====================================================================== 시편 상수
VANE = geo.VANE
L_C = 0.030                 # 유효(자유) 길이 [m]
B_C = geo.VANE_WIDTH          # 폭 [m] (깃 높이 = 판 사이 간격, G3 3.0 mm)
H_C = geo.T_VANE              # 두께 [m] (깃 두께, G3 1.0 mm)
E_C, NU_C, RHO_C = VANE.E, VANE.nu, VANE.rho
#: 실제 폭(3.2 mm)의 Timoshenko 보. 주파수**비**는 b에 무관하다(N3 BEAM의 b = 2h와 같은 비).
COUPON_BEAM = cs.TimoBeam(L=L_C, h=H_C, b=B_C, E=E_C, rho=RHO_C, nu=NU_C)
EI_C = COUPON_BEAM.E * COUPON_BEAM.I
DEPTHS_ABAR = (0.3, 0.5)
KERFS_MM = (0.15, 0.25, 0.35)
SIGMA_R = (3e-4, 1e-3, 3e-3)            # 재장착 산포 시나리오(모드별 상대 표준편차)
#: 온도 계수 d ln f/dT [1/K] — **가정값**(v2 계획서 §2.15의 산술 "±2 K → ±0.044 %"와 같은 기울기).
#: 실제 시편의 값이 아니다. 공통모드이므로 r₂₁·r₃₁에서 1차로 상쇄된다.
DLNF_DT_ASSUMED = -2.2e-4
#: LDV 반사테이프 질량 시나리오 [mg] — **가정값**(2 × 2 mm 조각의 규모).
TAPE_MG_ASSUMED = 0.5


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[2],
                                       text=True).strip()
    except Exception:                                   # pragma: no cover
        return "unknown"


def mode2_null_tm(k_root: float | None = None, beam: cs.TimoBeam = COUPON_BEAM,
                  n_elem: int = 1000) -> float:
    """Timoshenko FE 건전 쿠폰의 mode-2 곡률 null x/L (뿌리 회전스프링 선택)."""
    nul = n3.TimoFE(beam=beam, n_elem=n_elem, k_root=k_root).curvature_nulls(2)
    return float(nul[0]) if nul.size else float("nan")


# ====================================================================== 해석 기준
def torsion_J(b: float, h: float, n_terms: int = 50) -> float:
    """직사각 단면(b ≥ h) St-Venant 비틀림상수 — 급수해 J = (b h³/3)[1 − (192/π⁵)(h/b)Σ tanh(kπb/2h)/k⁵]."""
    if b < h:
        b, h = h, b
    s = sum(math.tanh(k * math.pi * b / (2 * h)) / k ** 5 for k in range(1, 2 * n_terms, 2))
    return b * h ** 3 / 3.0 * (1.0 - 192.0 / math.pi ** 5 * (h / b) * s)


def analytic_reference() -> pd.DataFrame:
    """해석 기준: EB flap·edge, Timoshenko flap·edge(FE n = 1000), St-Venant 비틀림(뒤틀림 구속 무시)."""
    rows = []
    fb = geo.Beam(L=L_C, h=H_C, E=E_C, rho=RHO_C, nu=NU_C).eb_frequencies(4)
    fe_ = geo.Beam(L=L_C, h=B_C, E=E_C, rho=RHO_C, nu=NU_C).eb_frequencies(4)
    tf = n3.TimoFE(beam=COUPON_BEAM, n_elem=1000).frequencies(4)
    edge_beam = cs.TimoBeam(L=L_C, h=B_C, b=H_C, E=E_C, rho=RHO_C, nu=NU_C)
    te = n3.TimoFE(beam=edge_beam, n_elem=1000).frequencies(4)
    for k in range(4):
        rows.append({"kind": "flap", "order": k + 1, "model": "EB_analytic", "f_Hz": fb[k]})
        rows.append({"kind": "flap", "order": k + 1, "model": "Timoshenko_FE1000", "f_Hz": tf[k]})
        rows.append({"kind": "edge", "order": k + 1, "model": "EB_analytic", "f_Hz": fe_[k]})
        rows.append({"kind": "edge", "order": k + 1, "model": "Timoshenko_FE1000", "f_Hz": te[k]})
    G = E_C / (2 * (1 + NU_C))
    J = torsion_J(B_C, H_C)
    Ip = B_C * H_C * (B_C ** 2 + H_C ** 2) / 12.0
    ct = math.sqrt(G * J / (RHO_C * Ip))
    for k in range(1, 4):
        rows.append({"kind": "torsion", "order": k, "model": "StVenant_no_warping",
                     "f_Hz": (2 * k - 1) * ct / (4 * L_C), "J_m4": J, "Ip_m4": Ip})
    return pd.DataFrame(rows)


def eb_nodes() -> pd.DataFrame:
    """해석 EB 캔틸레버 flap 모드의 변위 마디(측정점 배치)와 곡률 영점(노치 배치) x/L."""
    rows = []
    for m, bl in enumerate(geo.BETA_L[:3], start=1):
        sig = (math.cosh(bl) + math.cos(bl)) / (math.sinh(bl) + math.sin(bl))

        def w(x, bl=bl, sig=sig):
            t = bl * x
            return math.cosh(t) - math.cos(t) - sig * (math.sinh(t) - math.sin(t))

        def k(x, bl=bl, sig=sig):
            t = bl * x
            return math.cosh(t) + math.cos(t) - sig * (math.sinh(t) + math.sin(t))
        xs = np.linspace(0.02, 0.999, 4000)
        for name, fn in (("displacement_node", w), ("curvature_null", k)):
            v = np.array([fn(x) for x in xs])
            for i in range(len(xs) - 1):
                if v[i] * v[i + 1] < 0:
                    r = brentq(fn, xs[i], xs[i + 1], xtol=1e-12)
                    rows.append({"mode": f"flap{m}", "kind": name, "x_over_L": r,
                                 "x_mm": r * L_C * 1e3})
    return pd.DataFrame(rows)


# ====================================================================== 3D 솔리드
def coupon_mesh3d(nx: int = 60, ny: int = 8, nz: int = 4, *, xc_over_L: float | None = None,
                  kerf: float = 0.0, depth_abar: float = 0.0, notch: bool = False,
                  n_left: int = 20, n_kerf: int = 2, n_right: int = 48, n_below: int = 4,
                  n_above: int = 6, bias_x: float = 8.0, bias_z: float = 4.0):
    """쿠폰 육면체 격자. x = 길이(0 = 고정단), y = 폭(±b/2), z = 두께(±h/2).

    xc_over_L이 None이면 균일격자(nx, ny, nz)다. 주어지면 노치 면(x_c ± kerf/2)과 노치 바닥
    (z = −h/2 + ā h)에 절점을 **정확히** 두는 등급격자를 만들고, notch=True면 커프 밴드의 바닥 아래
    요소를 지운다(폭 전체를 가로지르는 평바닥 슬롯 — 싱커 EDM). notch=False는 **같은 절점배치**의 건전
    격자다(주파수비에서 이산화 오차가 상쇄된다).
    """
    ys = np.linspace(-0.5 * B_C, 0.5 * B_C, ny + 1)
    info = {"kerf_m": 0.0, "depth_m": 0.0, "xc_m": float("nan")}
    if xc_over_L is None:
        xs = np.linspace(0.0, L_C, nx + 1)
        zs = np.linspace(-0.5 * H_C, 0.5 * H_C, nz + 1)
        conn = r3._hex_conn(nx, ny, nz)
        coors = r3._grid_nodes(xs, ys, zs)
        coors, conn = r3._compact(coors, conn)
        return coors, conn, info
    if kerf <= 0 or not 0 < depth_abar < 1:
        raise ValueError("3D 노치는 유한폭(kerf > 0)과 0 < ā < 1이 필요하다")
    xc = xc_over_L * L_C
    half = 0.5 * kerf
    z_tip = -0.5 * H_C + depth_abar * H_C
    xs = np.concatenate([c2.graded_nodes(0.0, xc - half, n_left, "hi", bias_x),
                         np.linspace(xc - half, xc + half, n_kerf + 1)[1:],
                         c2.graded_nodes(xc + half, L_C, n_right, "lo", bias_x)[1:]])
    zs = np.concatenate([c2.graded_nodes(-0.5 * H_C, z_tip, n_below, "hi", bias_z),
                         c2.graded_nodes(z_tip, 0.5 * H_C, n_above, "lo", bias_z)[1:]])
    nx_, nz_ = xs.size - 1, zs.size - 1
    i0, i1 = n_left, n_left + n_kerf

    def keep(i, j, k):
        return not (notch and i0 <= i < i1 and k < n_below)

    conn = r3._hex_conn(nx_, ny, nz_, keep=keep)
    coors = r3._grid_nodes(xs, ys, zs)
    coors, conn = r3._compact(coors, conn)
    info = {"kerf_m": float(xs[i1] - xs[i0]), "depth_m": float(z_tip + 0.5 * H_C),
            "xc_m": float(0.5 * (xs[i0] + xs[i1])),
            "dx_min_m": float(np.min(np.diff(xs))), "dz_min_m": float(np.min(np.diff(zs))),
            "n_elem": int(conn.shape[0])}
    return coors, conn, info


def hex_volume(coors: np.ndarray, conn: np.ndarray) -> float:
    """축정렬 직육면체 요소의 부피 합(구조격자 전용)."""
    c = coors[conn]
    ext = c.max(axis=1) - c.min(axis=1)
    return float(np.sum(np.prod(ext, axis=1)))


@dataclass
class Modes3D:
    freqs: np.ndarray
    field_coors: np.ndarray
    shapes: np.ndarray           # (n_modes, n_field_nodes, 3), 질량정규화 [1/√kg]
    ndof: int
    seconds: float


def solve3d(coors: np.ndarray, conn: np.ndarray, n_modes: int = 8, order: int = 2,
            v0_seed: int = 20261006) -> Modes3D:
    """sfepy 조립(rail3d.solve_modes와 같은 항·적분차수, x = 0 면 클램프) + MMD shift-invert Lanczos."""
    from sfepy.base.base import output
    from sfepy.discrete import (Equation, Equations, FieldVariable, Integral, Material,
                                Problem)
    from sfepy.discrete.conditions import Conditions, EssentialBC
    from sfepy.discrete.fem import FEDomain, Field, Mesh
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.terms import Term

    t0 = time.time()
    output.set_output(quiet=True)
    mesh = Mesh.from_data("coupon", coors, None, [conn],
                          [np.zeros(len(conn), np.int32)], [r3.elem_desc(coors, conn)])
    domain = FEDomain("d", mesh)
    omega = domain.create_region("Omega", "all")
    domain.create_region("Fixed", r3.clamp_root(0.0), "facet")
    fld = Field.from_args("fu", np.float64, "vector", omega, approx_order=order)
    u = FieldVariable("u", "unknown", fld)
    v = FieldVariable("v", "test", fld, primary_var_name="u")
    mat = Material("m", D=stiffness_from_youngpoisson(3, E_C, NU_C), rho=RHO_C)
    integral = Integral("i", order=2 * order)
    eq_k = Equation("K", Term.new("dw_lin_elastic(m.D, v, u)", integral, omega, m=mat, v=v, u=u))
    eq_m = Equation("M", Term.new("dw_dot(m.rho, v, u)", integral, omega, m=mat, v=v, u=u))
    pb = Problem("coupon", equations=Equations([eq_k, eq_m]))
    pb.time_update(ebcs=Conditions([EssentialBC("Fixed", domain.regions["Fixed"],
                                                {"u.all": 0.0})]))
    pb.update_materials()
    K = eq_k.evaluate(mode="weak", dw_mode="matrix", asm_obj=pb.mtx_a)
    M0 = K.copy()
    M0.data[:] = 0.0
    M = eq_m.evaluate(mode="weak", dw_mode="matrix", asm_obj=M0)
    K, M = K.tocsc(), M.tocsc()
    lu = splu(K, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
    op = LinearOperator(K.shape, matvec=lu.solve, dtype=float)
    v0 = np.random.default_rng(v0_seed).standard_normal(K.shape[0])
    vals, vecs = eigsh(K, k=n_modes, M=M, sigma=0.0, which="LM", v0=v0, OPinv=op)
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]
    vecs = vecs / np.sqrt(np.einsum("ij,ij->j", vecs, M @ vecs))
    variables = pb.get_variables()
    fc = variables["u"].field.get_coor()
    shapes = np.stack([variables.make_full_vec(vecs[:, k]).reshape(-1, 3)
                       for k in range(vecs.shape[1])])
    return Modes3D(np.sqrt(np.maximum(vals, 0.0)) / (2 * math.pi), fc, shapes, K.shape[0],
                   time.time() - t0)


def classify3d(md: Modes3D, n_bin: int = 60) -> list[dict]:
    """모드를 형상으로 분류한다: flap(단면 평균 u_z) / torsion(u_z의 y-기울기 × b/2) /
    edge(단면 평균 u_y) / axial(단면 평균 u_x). 차수 = 지배 단면함수의 부호교차 수 + 1.

    정본 `rail3d.beam_mode_kinds`는 지배 변위성분만 보므로 폭 4.1 mm 쿠폰의 1차 비틀림(u_z 지배)을
    "flap"으로 표시한다 — 여기서는 u_z를 y에 대해 대칭(굽힘)·반대칭(비틀림) 부분으로 나눈다.
    """
    x, y = md.field_coors[:, 0], md.field_coors[:, 1]
    bins = np.clip((x / L_C * n_bin).astype(int), 0, n_bin - 1)
    cnt = np.bincount(bins, minlength=n_bin).astype(float)
    cnt[cnt == 0] = np.nan
    ymean = np.bincount(bins, weights=y, minlength=n_bin) / cnt
    yc = y - ymean[bins]
    syy = np.bincount(bins, weights=yc * yc, minlength=n_bin)
    out = []
    for k in range(md.shapes.shape[0]):
        u = md.shapes[k]
        W = np.bincount(bins, weights=u[:, 2], minlength=n_bin) / cnt
        V = np.bincount(bins, weights=u[:, 1], minlength=n_bin) / cnt
        U = np.bincount(bins, weights=u[:, 0], minlength=n_bin) / cnt
        Th = np.bincount(bins, weights=u[:, 2] * yc, minlength=n_bin) / np.where(syy > 0, syy, np.nan)
        T = Th * 0.5 * B_C
        fields = {"flap": W, "torsion": T, "edge": V, "axial": U}
        en = {kk: float(np.nansum(vv ** 2)) for kk, vv in fields.items()}
        tot = sum(en.values())
        kind = max(en, key=en.get)
        F = fields[kind]
        F = F[np.isfinite(F)]
        sig = F[np.abs(F) > 0.02 * np.max(np.abs(F))]
        order = int(np.sum(np.diff(np.sign(sig)) != 0)) + 1
        out.append({"mode_index": k, "f_Hz": float(md.freqs[k]), "kind": kind, "order": order,
                    **{f"frac_{kk}": en[kk] / tot for kk in en}})
    return out


def find_mode(cls: list[dict], kind: str, order: int) -> int | None:
    for c in cls:
        if c["kind"] == kind and c["order"] == order:
            return c["mode_index"]
    return None


def null_from_3d(md: Modes3D, mode_index: int, window=(0.05, 0.45), deg: int = 8) -> float:
    """3D flap 모드의 굽힘 회전 θ(x) ≈ (u_x(z 최소) − u_x(z 최대))/h(폭 평균)의 극값 = 곡률 영점.

    절점 θ(x)에 창 [0.05, 0.45]L에서 다항식(차수 8, x/L 정규화)을 맞추고 θ′ = 0의 근 중
    0.1L–0.4L에 있는 것을 낸다 — 선형보간 기울기의 부호교차는 절점간격(격자)에 묶여 ±0.004L 흔들린다.
    """
    c = md.field_coors
    u = md.shapes[mode_index]
    zt, zb = c[:, 2].max(), c[:, 2].min()
    top = np.abs(c[:, 2] - zt) < 1e-12
    bot = np.abs(c[:, 2] - zb) < 1e-12

    def prof(sel):
        s = pd.Series(u[sel, 0]).groupby(np.round(c[sel, 0], 12)).mean()
        return s.index.values, s.values
    xb, ub = prof(bot)
    xt, ut = prof(top)
    xx = np.intersect1d(xb, xt)
    th = (np.interp(xx, xb, ub) - np.interp(xx, xt, ut)) / H_C
    s = xx / L_C
    sel = (s >= window[0]) & (s <= window[1])
    p = np.polynomial.Polynomial.fit(s[sel], th[sel], deg)
    roots = [float(r.real) for r in p.deriv().roots()
             if abs(r.imag) < 1e-9 and 0.1 <= r.real <= 0.4]
    return min(roots) if roots else float("nan")


def mass_limits(md: Modes3D, mode_index: int, budgets=(5e-4, 1e-3)) -> dict:
    """점질량 1차 감도 Δf/f ≈ −½ m |φ(x_a)|² (질량정규화 φ) — 표면 최대 진폭점과 끝단 중앙에서."""
    c = md.field_coors
    u = md.shapes[mode_index]
    phi2 = np.sum(u ** 2, axis=1)
    surf = (np.abs(np.abs(c[:, 2]) - 0.5 * H_C) < 1e-12)
    tip = (np.abs(c[:, 0] - L_C) < 1e-12) & (np.abs(c[:, 1]) < 1e-12) & surf
    p_surf = float(phi2[surf].max())
    p_tip = float(phi2[tip].max()) if tip.any() else float("nan")
    out = {"phi2_surface_max_per_kg": p_surf, "phi2_tip_center_per_kg": p_tip,
           "m_eff_surface_g": 1e3 / p_surf}
    for bgt in budgets:
        out[f"m_limit_surface_mg_dff{100 * bgt:g}pct"] = 2 * bgt / p_surf * 1e6
    out["dff_pct_per_mg_tip"] = -0.5 * 1e-6 * p_tip * 100
    out["dff_pct_tape_assumed_tip"] = -0.5 * TAPE_MG_ASSUMED * 1e-6 * p_tip * 100
    for g in (0.2, 0.5, 2.0):
        out[f"dff_pct_first_order_{g:g}g_surface_max"] = -0.5 * g * 1e-3 * p_surf * 100
    return out


# ====================================================================== 2D 평면응력
def _unotch_geo(path: Path, xc: float, a: float, w: float, lc: float, lc_tip: float,
                notch: bool) -> None:
    """gmsh .geo — 쿠폰 단면(길이 L × 두께 h) + 아래면에서 깊이 a의 둥근 선단 U-노치(폭 w = 2r).

    노치 = 슬롯(폭 w, 바닥 높이 −h/2 + a − r) ∪ 반원(중심 (x_c, −h/2 + a − r), 반경 r) — 와이어 EDM의
    단면. 깊이는 가장 깊은 점까지. notch=False면 같은 크기장의 건전 단면(비교용).
    """
    h, L = H_C, L_C
    r = 0.5 * w
    yc = -0.5 * h + a - r
    g = ['SetFactory("OpenCASCADE");', "Geometry.Tolerance = 1e-12;",
         "Geometry.ToleranceBoolean = 1e-12;",
         f"Rectangle(1) = {{0, {-h / 2!r}, 0, {L!r}, {h!r}}};"]
    if notch:
        eps = 0.05 * h
        g += [f"Rectangle(2) = {{{xc - r!r}, {-h / 2 - eps!r}, 0, {w!r}, {yc + h / 2 + eps!r}}};",
              f"Disk(3) = {{{xc!r}, {yc!r}, 0, {r!r}, {r!r}}};",
              "BooleanUnion(4) = { Surface{2}; Delete; }{ Surface{3}; Delete; };",
              "BooleanDifference(5) = { Surface{1}; Delete; }{ Surface{4}; Delete; };",
              "Physical Surface(1) = {5};",
              f"nc() = Curve In BoundingBox{{{xc - r - 1e-7!r}, {-h / 2 - 1e-7!r}, -1, "
              f"{xc + r + 1e-7!r}, {-h / 2 + a + 1e-7!r}, 1}};",
              "Field[1] = Distance; Field[1].CurvesList = {nc()}; Field[1].Sampling = 400;"]
    else:
        g += ["Physical Surface(1) = {1};",
              f"Point(901) = {{{xc!r}, {-h / 2 + a!r}, 0}};",
              "Field[1] = Distance; Field[1].PointsList = {901};"]
    g += [f"Field[2] = Threshold; Field[2].InField = 1; Field[2].SizeMin = {lc_tip!r};",
          f"Field[2].SizeMax = {lc!r}; Field[2].DistMin = {0.25 * w!r}; "
          f"Field[2].DistMax = {0.6 * h!r};",
          "Background Field = 2;",
          "Mesh.MeshSizeExtendFromBoundary = 0; Mesh.MeshSizeFromPoints = 0;",
          "Mesh.MeshSizeFromCurvature = 0;", "Mesh.Algorithm = 6;", "Mesh.ElementOrder = 1;"]
    path.write_text("\n".join(g) + "\n")


def _gmsh_tri(geo_path: Path, out: Path):
    subprocess.run(["gmsh", str(geo_path), "-2", "-format", "msh22", "-o", str(out)],
                   check=True, capture_output=True)
    return n3._read_tri(out)


def tri_area(co: np.ndarray, tri: np.ndarray) -> float:
    p = co[tri]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return float(0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())


def flap2d(co, conn, n_modes: int = 6, plane: str = "stress") -> np.ndarray:
    """2D 면내굽힘 모드 주파수(형상 분류, 시드 고정) — crack2d.flap_modes와 같은 경로."""
    f, kinds, nd = c2.flap_modes(co, conn, E_C, NU_C, RHO_C, plane=plane, n_modes=n_modes)
    return f, nd


def unotch_2d(xc_over_L: float, a_bar: float, kerf: float, level: int = 1,
              workdir: Path | None = None) -> dict:
    """둥근 선단 U-노치(2D 평면응력)와 같은 크기장의 건전 단면 — 주파수비."""
    workdir = Path(workdir or tempfile.mkdtemp(prefix="e1c_unotch_"))
    lc = H_C / (8 * level)
    lc_tip = min(kerf / 12.0, H_C / 40.0) / level
    xc, a = xc_over_L * L_C, a_bar * H_C
    res = {}
    for tag, notch in (("h", False), ("d", True)):
        gp = workdir / f"u_{tag}_{xc_over_L:.4f}_{a_bar}_{kerf * 1e3:.2f}_{level}.geo"
        _unotch_geo(gp, xc, a, kerf, lc, lc_tip, notch)
        co, tri = _gmsh_tri(gp, gp.with_suffix(".msh"))
        f, nd = flap2d(co, tri)
        res[tag] = (f[:3], nd, tri_area(co, tri))
    removed = H_C * L_C - res["d"][2]
    exact = kerf * (a - 0.5 * kerf) + 0.5 * math.pi * (0.5 * kerf) ** 2
    return {"ratio": res["d"][0] / res["h"][0], "f_healthy": res["h"][0],
            "ndof": res["d"][1], "removed_area_rel_err": removed / exact - 1.0}


def slit_or_flat_2d(xc_over_L: float, a_bar: float, kerf: float, ref: int = 3,
                    plane: str = "stress") -> dict:
    """crack2d 구조격자 — kerf = 0이면 폭 0 슬릿, > 0이면 평바닥 커프(요소 제거). 같은 절점의 건전 대조."""
    kw = n3._a11_grid(ref)
    n_k = max(2, int(round(kerf / (0.05e-3 / ref)))) if kerf > 0 else 2
    co0, cn0, _ = c2.slit_mesh(L_C, H_C, a_bar, xc_over_L=xc_over_L, crack=False,
                               kerf_width=kerf, n_kerf=n_k, **kw)
    co1, cn1, info = c2.slit_mesh(L_C, H_C, a_bar, xc_over_L=xc_over_L, crack=True,
                                  kerf_width=kerf, n_kerf=n_k, **kw)
    f0, _ = flap2d(co0, cn0, plane=plane)
    f1, nd = flap2d(co1, cn1, plane=plane)
    return {"ratio": f1[:3] / f0[:3], "f_healthy": f0[:3], "ndof": nd}


# ====================================================================== 판별 지표
def fingerprint_cols(ratio: np.ndarray, xc_over_L: float) -> dict:
    """N3 규약의 판별 지표 + 공통모드 제거 관측량. 매끄러운 비교장은 이 경우의 Δf₁에 다시 맞춘다."""
    ratio = np.asarray(ratio, dtype=float)
    drops, dmax = n3.matched_smooth(100 * (1 - ratio[0]), centers=(xc_over_L,))
    d = ratio - 1.0
    ds = -drops / 100.0
    out = {"d1_pct": 100 * d[0], "d2_pct": 100 * d[1], "d3_pct": 100 * d[2],
           "r21_pct": 100 * (d[1] - d[0]), "r31_pct": 100 * (d[2] - d[0]),
           "smooth_dmax": dmax, "smooth_d2_pct": 100 * ds[1], "smooth_d3_pct": 100 * ds[2],
           "smooth_r21_pct": 100 * (ds[1] - ds[0]), "smooth_r31_pct": 100 * (ds[2] - ds[0]),
           # 균열 대 비교장 대비 = |δ₂ˢ| − |δ₂ᶜ| (r₂₁ 차이와 같다: r₂₁ᶜ − r₂₁ˢ = δ₂ᶜ − δ₂ˢ)
           "contrast2_pct": 100 * (abs(ds[1]) - abs(d[1])),
           "contrast3_pct": 100 * (abs(ds[2]) - abs(d[2])),
           # C1의 비율 조건 여유: ¼·D₂₁ˢ·|δ₁| − |δ₂ᶜ|
           "c1_margin_pct": 100 * (0.25 * abs(ds[1]) - abs(d[1]))}
    out.update({k: v for k, v in n3.contrast_cols(ratio, drops).items()
                if k in ("D21_crack_pct", "D21_smooth_pct", "D31_crack_pct", "D31_smooth_pct",
                         "contrast_m2_c0.001", "C1_nullfp", "C2_multimode_c0.001")})
    return out


def tm_ratio(xc_over_L: float, a_bar: float, convention: str = "tada",
             k_root: float | None = None, beam: cs.TimoBeam = COUPON_BEAM,
             n_elem: int = 1000) -> np.ndarray:
    C = n3.crack_compliance(a_bar, convention, True, beam=beam)
    f = n3.TimoFE(beam=beam, n_elem=n_elem, cracks=[(xc_over_L, C)], k_root=k_root).frequencies(3)
    f0 = n3.TimoFE(beam=beam, n_elem=n_elem, extra_nodes=[xc_over_L], k_root=k_root).frequencies(3)
    return f / f0


# ====================================================================== (1) 건전 3D 모드지도
MESHES_3D = {"M1": (60, 8, 4), "M2": (90, 12, 6), "M3": (120, 12, 8)}


def _healthy_job(name: str) -> dict:
    nx, ny, nz = MESHES_3D[name]
    co, cn, _ = coupon_mesh3d(nx, ny, nz)
    md = solve3d(co, cn, n_modes=10)
    cls = classify3d(md)
    i2 = find_mode(cls, "flap", 2)
    extra = {"mode2_null_xL_3d": null_from_3d(md, i2) if i2 is not None else float("nan")}
    ml = []
    for kind, order in (("flap", 1), ("flap", 2), ("flap", 3), ("edge", 1), ("torsion", 1)):
        i = find_mode(cls, kind, order)
        if i is not None:
            ml.append({"kind": kind, "order": order, "f_Hz": float(md.freqs[i]),
                       **mass_limits(md, i)})
    return {"mesh": name, "nx": nx, "ny": ny, "nz": nz, "ndof": md.ndof,
            "seconds": md.seconds, "modes": cls, "mass": ml, **extra}


def run_healthy3d(workers: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = _map(_healthy_job, list(MESHES_3D), workers)
    rows, mrows = [], []
    for r in res:
        for m in r["modes"]:
            rows.append({"mesh": r["mesh"], "nx": r["nx"], "ny": r["ny"], "nz": r["nz"],
                         "ndof": r["ndof"], "seconds": r["seconds"],
                         "mode2_null_xL_3d": r["mode2_null_xL_3d"], **m})
        for m in r["mass"]:
            mrows.append({"mesh": r["mesh"], "ndof": r["ndof"], **m})
    df = pd.DataFrame(rows)
    ref = analytic_reference()
    tm = ref[ref.model.str.startswith("Timoshenko")]
    eb = ref[ref.model == "EB_analytic"]
    tor = ref[ref.model.str.startswith("StVenant")]
    df["f_EB_Hz"] = [eb[(eb.kind == k) & (eb.order == o)].f_Hz.iloc[0]
                     if ((eb.kind == k) & (eb.order == o)).any() else np.nan
                     for k, o in zip(df.kind, df.order)]
    df["f_Timoshenko_Hz"] = [tm[(tm.kind == k) & (tm.order == o)].f_Hz.iloc[0]
                             if ((tm.kind == k) & (tm.order == o)).any() else np.nan
                             for k, o in zip(df.kind, df.order)]
    df["f_StVenant_Hz"] = [tor[tor.order == o].f_Hz.iloc[0]
                           if k == "torsion" and (tor.order == o).any() else np.nan
                           for k, o in zip(df.kind, df.order)]
    # 2D 평면응력·평면변형 건전 flap(정본 a11 격자 ref 3)
    co, cn, _ = c2.slit_mesh(L_C, H_C, 0.5, xc_over_L=0.2, crack=False, **n3._a11_grid(3))
    for plane in ("stress", "strain"):
        f2d, _ = flap2d(co, cn, plane=plane)
        df[f"f_2D_{plane}_Hz"] = [f2d[o - 1] if k == "flap" and o <= 3 else np.nan
                                  for k, o in zip(df.kind, df.order)]
    return df, pd.DataFrame(mrows)


# ====================================================================== (2) 노치 예측
def _xnull() -> float:
    return round(mode2_null_tm(), 4)


def notch_cases() -> list[dict]:
    xn = _xnull()
    cases = []
    for xc in (0.2, xn):
        for ab in DEPTHS_ABAR:
            cases.append({"model": "TM_tada", "xc_over_L": xc, "a_bar": ab, "kerf_mm": 0.0})
            cases.append({"model": "TM_dimarogonas", "xc_over_L": xc, "a_bar": ab, "kerf_mm": 0.0})
            cases.append({"model": "2D_stress_slit0", "xc_over_L": xc, "a_bar": ab, "kerf_mm": 0.0})
            cases.append({"model": "2D_strain_slit0", "xc_over_L": xc, "a_bar": ab, "kerf_mm": 0.0})
            for k in KERFS_MM:
                cases.append({"model": "2D_stress_flat", "xc_over_L": xc, "a_bar": ab, "kerf_mm": k})
                cases.append({"model": "2D_stress_Unotch", "xc_over_L": xc, "a_bar": ab,
                              "kerf_mm": k})
            cases.append({"model": "3D_flat_slot", "xc_over_L": xc, "a_bar": ab, "kerf_mm": 0.25})
    for k in (0.15, 0.35):
        for ab in DEPTHS_ABAR:
            cases.append({"model": "3D_flat_slot", "xc_over_L": 0.2, "a_bar": ab, "kerf_mm": k})
    # 격자수렴 점검(대표 경우): 2D U-노치 level 2, 3D 슬롯 세밀격자
    cases.append({"model": "2D_stress_Unotch", "xc_over_L": 0.2, "a_bar": 0.5, "kerf_mm": 0.25,
                  "level": 2})
    cases.append({"model": "2D_stress_flat", "xc_over_L": 0.2, "a_bar": 0.5, "kerf_mm": 0.25,
                  "level": 4})
    cases.append({"model": "3D_flat_slot", "xc_over_L": 0.2, "a_bar": 0.5, "kerf_mm": 0.25,
                  "level": 2})
    return cases


#: 3D 슬롯 격자: level 1 = 작업격자, level 2 = 노치 근방·두께·폭 세분(수렴 점검; 자유단 쪽은 같다)
GRID3D = {1: dict(n_left=20, n_kerf=2, n_right=48, n_below=4, n_above=6, ny=6),
          2: dict(n_left=30, n_kerf=4, n_right=48, n_below=6, n_above=9, ny=8)}


def _notch3d(xc: float, ab: float, kerf: float, level: int = 1) -> dict:
    kw = dict(xc_over_L=xc, kerf=kerf, depth_abar=ab, **GRID3D[level])
    co0, cn0, _ = coupon_mesh3d(**kw, notch=False)
    co1, cn1, info = coupon_mesh3d(**kw, notch=True)
    m0 = solve3d(co0, cn0, n_modes=8)
    m1 = solve3d(co1, cn1, n_modes=8)
    c0, c1 = classify3d(m0), classify3d(m1)
    out = {"ndof": m1.ndof, "seconds": m0.seconds + m1.seconds,
           "vol_removed_rel_err": (hex_volume(co0, cn0) - hex_volume(co1, cn1))
           / (kerf * ab * H_C * B_C) - 1.0, **{f"mesh_{k}": v for k, v in info.items()}}
    ratios = []
    for kind, order in (("flap", 1), ("flap", 2), ("flap", 3), ("edge", 1), ("torsion", 1)):
        i0, i1 = find_mode(c0, kind, order), find_mode(c1, kind, order)
        if i0 is None or i1 is None:
            out[f"ratio_{kind}{order}"] = float("nan")
            continue
        out[f"f_healthy_{kind}{order}_Hz"] = float(m0.freqs[i0])
        out[f"ratio_{kind}{order}"] = float(m1.freqs[i1] / m0.freqs[i0])
        # 끝단 중앙 테이프(고정 질량)의 2차 잔차: 건전·손상 φ² 차이 (모드별, per mg)
        tip0 = mass_limits(m0, i0)["phi2_tip_center_per_kg"]
        tip1 = mass_limits(m1, i1)["phi2_tip_center_per_kg"]
        out[f"tape_resid_pct_per_mg_{kind}{order}"] = -0.5 * 1e-6 * (tip1 - tip0) * 100
    out["ratio"] = np.array([out["ratio_flap1"], out["ratio_flap2"], out["ratio_flap3"]])
    out["mode2_null_xL_3d_healthy"] = null_from_3d(m0, find_mode(c0, "flap", 2))
    return out


def _notch_job(case: dict) -> dict:
    m, xc, ab, k = case["model"], case["xc_over_L"], case["a_bar"], case["kerf_mm"] * 1e-3
    lev = case.get("level", 1)
    t0 = time.time()
    extra = {}
    if m == "TM_tada":
        r = tm_ratio(xc, ab, "tada")
    elif m == "TM_dimarogonas":
        r = tm_ratio(xc, ab, "dimarogonas")
    elif m in ("2D_stress_slit0", "2D_strain_slit0"):
        res = slit_or_flat_2d(xc, ab, 0.0, ref=3, plane=m.split("_")[1])
        r, extra = res["ratio"], {"ndof": res["ndof"]}
    elif m == "2D_stress_flat":
        res = slit_or_flat_2d(xc, ab, k, ref=3 if lev == 1 else lev)
        r, extra = res["ratio"], {"ndof": res["ndof"]}
    elif m == "2D_stress_Unotch":
        res = unotch_2d(xc, ab, k, level=lev)
        r, extra = res["ratio"], {"ndof": res["ndof"],
                                  "removed_area_rel_err": res["removed_area_rel_err"]}
    elif m == "3D_flat_slot":
        res = _notch3d(xc, ab, k, level=lev)
        r = res.pop("ratio")
        extra = res
    else:
        raise ValueError(m)
    out = {**case, "level": lev, "seconds_total": time.time() - t0,
           **{f"ratio_f{i + 1}": float(r[i]) for i in range(3)}, **extra}
    out.update(fingerprint_cols(r, xc))
    return out


def run_notch(workers: int = 6) -> pd.DataFrame:
    cases = notch_cases()
    # 무거운 3D를 먼저 던진다
    cases.sort(key=lambda c: (0 if c["model"].startswith("3D") else 1, -c.get("level", 1)))
    rows = _map(_notch_job, cases, workers)
    df = pd.DataFrame(rows)
    df["x_null_tm"] = _xnull()
    df["position"] = np.where(np.isclose(df.xc_over_L, 0.2), "x0.20", "at_null")
    df["depth_mm"] = df.a_bar * H_C * 1e3
    return df.sort_values(["position", "a_bar", "model", "kerf_mm", "level"]).reset_index(drop=True)


#: 음성 대조 위치 — N3에서 균열과 비교장의 D₂₁이 같아지는 곳(지문이 없어야 한다)
X_NEG = 0.35


def run_negative_control(workers: int = 4) -> pd.DataFrame:
    """음성 대조 쿠폰(x_c/L = 0.35): 같은 모델들로 '지문 없음'을 사전 예측한다."""
    cases = []
    for ab in DEPTHS_ABAR:
        cases += [{"model": "TM_tada", "xc_over_L": X_NEG, "a_bar": ab, "kerf_mm": 0.0},
                  {"model": "2D_stress_slit0", "xc_over_L": X_NEG, "a_bar": ab, "kerf_mm": 0.0},
                  {"model": "2D_stress_Unotch", "xc_over_L": X_NEG, "a_bar": ab, "kerf_mm": 0.25},
                  {"model": "3D_flat_slot", "xc_over_L": X_NEG, "a_bar": ab, "kerf_mm": 0.25}]
    cases.sort(key=lambda c: 0 if c["model"].startswith("3D") else 1)
    df = pd.DataFrame(_map(_notch_job, cases, workers))
    df["position"] = "negative_control_x0.35"
    df["depth_mm"] = df.a_bar * H_C * 1e3
    return df


# ====================================================================== (3) 허용오차·뿌리
#: 와이어 EDM이 없을 때(2026-10-17) — 톱(슬리팅 소·실톱) 노치는 평바닥 슬롯으로 보고 커프 0.4–0.6 mm로 괄호를 친다
SAW_KERFS_MM = (0.4, 0.5, 0.6)


def run_saw(workers: int = 4) -> pd.DataFrame:
    """톱 노치의 사전 예측 — 3D 평바닥 슬롯·2D 평바닥 커프, 주 위치 0.2 L와 음성 대조 0.35 L, ā 0.3·0.5."""
    cases = [{"model": m, "xc_over_L": xc, "a_bar": ab, "kerf_mm": k}
             for xc in (0.2, X_NEG) for ab in DEPTHS_ABAR for k in SAW_KERFS_MM
             for m in ("3D_flat_slot", "2D_stress_flat")]
    cases.sort(key=lambda c: 0 if c["model"].startswith("3D") else 1)
    df = pd.DataFrame(_map(_notch_job, cases, workers))
    df["position"] = np.where(np.isclose(df.xc_over_L, 0.2), "x0.20", "negative_control_x0.35")
    df["depth_mm"] = df.a_bar * H_C * 1e3
    return df.sort_values(["position", "a_bar", "model", "kerf_mm"]).reset_index(drop=True)


def run_tolerance() -> pd.DataFrame:
    """노치 위치 오차와 클램프 가장자리(유효 뿌리) 이동 — Timoshenko·Tada(폭 0 균열)."""
    xn = _xnull()
    rows = []
    for base_tag, x0 in (("x0.20", 0.2), ("at_null", xn)):
        for ab in DEPTHS_ABAR:
            for d_mm in (-0.6, -0.3, 0.0, 0.3, 0.6):
                xc = x0 + d_mm * 1e-3 / L_C
                r = tm_ratio(xc, ab)
                rows.append({"perturbation": "notch_station", "base": base_tag, "a_bar": ab,
                             "shift_mm": d_mm, "xc_over_L_eff": xc, "L_eff_mm": L_C * 1e3,
                             **{f"ratio_f{i + 1}": float(r[i]) for i in range(3)},
                             **fingerprint_cols(r, xc)})
            for s_mm in (-0.6, -0.3, 0.3, 0.6):
                # 유효 뿌리가 s만큼 클램프 안(+)·밖(−)으로: 자유길이 L + s, 노치는 뿌리에서 x_c + s
                Lp = L_C + s_mm * 1e-3
                xcp = (x0 * L_C + s_mm * 1e-3) / Lp
                beam = cs.TimoBeam(L=Lp, h=H_C, b=B_C, E=E_C, rho=RHO_C, nu=NU_C)
                r = tm_ratio(xcp, ab, beam=beam)
                rows.append({"perturbation": "clamp_edge", "base": base_tag, "a_bar": ab,
                             "shift_mm": s_mm, "xc_over_L_eff": xcp, "L_eff_mm": Lp * 1e3,
                             **{f"ratio_f{i + 1}": float(r[i]) for i in range(3)},
                             **fingerprint_cols(r, xcp)})
    return pd.DataFrame(rows)


K_ROOT_GRID = (np.inf, 10000.0, 3000.0, 1000.0, 300.0, 100.0, 50.0, 30.0, 20.0, 10.0)


def run_rootflex() -> pd.DataFrame:
    """뿌리 회전강성 k̃ = k_root L/EI: 건전 주파수비·null 이동, 지문(x_c = 0.2·null)과 같은 뿌리의 비교장."""
    rows = []
    for knd in K_ROOT_GRID:
        kr = None if not np.isfinite(knd) else float(knd) * EI_C / L_C
        fh = n3.TimoFE(beam=COUPON_BEAM, n_elem=1000, k_root=kr).frequencies(3)
        xn = mode2_null_tm(kr)
        for where, xc in (("x0.20", 0.2), ("at_null", round(xn, 4))):
            for ab in DEPTHS_ABAR:
                r = tm_ratio(xc, ab, k_root=kr)
                drops, s = n3.matched_smooth_fe(100 * (1 - r[0]), centers=(xc,), k_root=kr,
                                                n_elem=600)
                d = r - 1
                ds = -drops / 100
                rows.append({"k_root_nd": knd, "f1_Hz": fh[0], "f2_Hz": fh[1], "f3_Hz": fh[2],
                             "R21": fh[1] / fh[0], "R31": fh[2] / fh[0], "mode2_null_xL": xn,
                             "crack_at": where, "xc_over_L": xc, "a_bar": ab,
                             **{f"ratio_f{i + 1}": float(r[i]) for i in range(3)},
                             "d1_pct": 100 * d[0], "d2_pct": 100 * d[1],
                             "smooth_d2_pct": 100 * ds[1], "smooth_dmax": s,
                             "contrast2_pct": 100 * (abs(ds[1]) - abs(d[1])),
                             "c1_margin_pct": 100 * (0.25 * abs(ds[1]) - abs(d[1])),
                             **{k: v for k, v in n3.contrast_cols(r, drops).items()
                                if k in ("D21_crack_pct", "D21_smooth_pct", "C1_nullfp")}})
    return pd.DataFrame(rows)


def run_lookup(n: int = 41) -> pd.DataFrame:
    """측정 건전비 → k̃ → null 조회표. ρ₂₁ = (f₂/f₁)/(f₂/f₁)_고정, ρ₃₁ 같은 규약(모델 고유값으로 정규화)."""
    ideal = n3.TimoFE(beam=COUPON_BEAM, n_elem=1000).frequencies(3)
    R21_0, R31_0 = ideal[1] / ideal[0], ideal[2] / ideal[0]
    rows = []
    for knd in np.logspace(np.log10(3.0), np.log10(3e4), n):
        kr = float(knd) * EI_C / L_C
        f = n3.TimoFE(beam=COUPON_BEAM, n_elem=1000, k_root=kr).frequencies(3)
        rows.append({"k_root_nd": knd, "rho21": (f[1] / f[0]) / R21_0,
                     "rho31": (f[2] / f[0]) / R31_0, "f1_over_clamped": f[0] / ideal[0],
                     "mode2_null_xL": mode2_null_tm(kr),
                     "mode2_null_mm": mode2_null_tm(kr) * L_C * 1e3})
    df = pd.DataFrame(rows)
    df["R21_clamped_TM"] = R21_0
    df["R31_clamped_TM"] = R31_0
    # 두께 공차가 고정단 비 R₂₁·R₃₁에 주는 영향(전단보정 h/L) — k̃ 식별의 교란 규모
    for hh in (0.95 * H_C, 1.05 * H_C):              # 공칭 두께 ±5 %
        bm = cs.TimoBeam(L=L_C, h=hh, b=B_C, E=E_C, rho=RHO_C, nu=NU_C)
        f = n3.TimoFE(beam=bm, n_elem=1000).frequencies(3)
        df[f"rho21_clamped_h{hh * 1e3:.2f}mm"] = (f[1] / f[0]) / R21_0
        df[f"rho31_clamped_h{hh * 1e3:.2f}mm"] = (f[2] / f[0]) / R31_0
    return df


# ====================================================================== (4) 측정 요건
def required_repeats(effect: float, sigma_r: float, z: float, var_factor: float) -> int:
    """σ(통계량) = σ_r √(var_factor/N) 일 때 z·σ ≤ effect 를 만족하는 최소 N (상태당 장착 수)."""
    if not effect > 0:
        return -1                                          # 효과 없음 = 어떤 N으로도 불가
    return int(math.ceil(var_factor * (z * sigma_r / effect) ** 2))


def crlb_damped_sine(f: float, zeta: float, fs: float, T: float, snr: float) -> float:
    """감쇠 정현파 s = A e^{−ζωt} cos(ω_d t + φ) + 백색잡음(σ_n = A/snr)의 주파수 CRLB σ_f/f (A, f, ζ, φ 미지)."""
    t = np.arange(int(round(fs * T))) / fs
    w = 2 * math.pi * f
    sq = math.sqrt(1 - zeta ** 2)
    a, wd = zeta * w, w * sq
    A, ph = 1.0, 0.3
    e = np.exp(-a * t)
    c, s = np.cos(wd * t + ph), np.sin(wd * t + ph)
    J = np.stack([e * c,
                  A * e * (-t * 2 * math.pi * zeta * c - t * 2 * math.pi * sq * s),
                  A * e * (-t * w * c + t * w * zeta / sq * s),
                  -A * e * s], axis=1)
    sig_n = A / snr
    F = J.T @ J / sig_n ** 2
    return float(math.sqrt(np.linalg.inv(F)[1, 1]) / f)


def fit_damped_sine(x: np.ndarray, fs: float, f_guess: float, rel_band: float = 0.2,
                    t_skip: float | None = None, floor_mult: float = 5.0,
                    dyn_range: float = 1e-4) -> dict:
    """사전등록 분석기: 한 번의 타격 자유감쇠에서 한 모드의 주파수·감쇠를 낸다.

    1) f_guess ± rel_band 영위상 FFT 대역통과, 2) 해석신호 포락선이 꼬리 잡음의 floor_mult배와
    최대값의 dyn_range배를 모두 넘는 구간을 창으로(타격 직후 t_skip 제외 — 기본은 대역통과 울림이
    가라앉는 max(5 ms, 3/(rel_band·f_guess))), 3) A e^{−αt} cos(2π f_d t + φ) 비선형 최소제곱.
    잡음 없는 합성신호에서 대역 절단에 의한 편향은 상대 1e-6 규모다(검정).
    반환 f_Hz(비감쇠 f = f_d/√(1−ζ²)), zeta, f_d_Hz, 창 길이, 정규화 잔차.
    """
    from scipy.optimize import least_squares
    from scipy.signal import hilbert
    x = np.asarray(x, dtype=float)
    n = x.size
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    X = np.fft.rfft(x - x.mean())
    X[(fr < f_guess * (1 - rel_band)) | (fr > f_guess * (1 + rel_band))] = 0.0
    y = np.fft.irfft(X, n)
    t = np.arange(n) / fs
    env = np.abs(hilbert(y))
    noise = float(np.median(env[int(0.9 * n):]))
    if t_skip is None:
        t_skip = max(0.005, 3.0 / (rel_band * f_guess))
    i0 = int(round(t_skip * fs))
    thr = max(floor_mult * noise, dyn_range * float(env[i0:].max()))
    above = np.nonzero(env[i0:] > thr)[0]
    i1 = i0 + (int(above[-1]) if above.size else n - i0 - 1)
    tt, yy = t[i0:i1] - t[i0], y[i0:i1]
    # 초기값: 영패딩 FFT 피크 + 포락선 로그 기울기
    nz = 1 << int(np.ceil(np.log2(max(len(yy), 2) * 8)))
    Z = np.abs(np.fft.rfft(yy, nz))           # 감쇠신호 자체가 창 — 한닝은 신호가 사는 앞부분을 죽인다
    fz = np.fft.rfftfreq(nz, 1.0 / fs)
    fd0 = float(fz[np.argmax(Z)])
    ee = env[i0:i1]
    alpha0 = max(-np.polyfit(tt, np.log(np.maximum(ee, 1e-300)), 1)[0], 1e-3)
    A0 = float(ee[0])
    c, s = np.cos(2 * math.pi * fd0 * tt), np.sin(2 * math.pi * fd0 * tt)
    ab = np.linalg.lstsq(np.stack([np.exp(-alpha0 * tt) * c, np.exp(-alpha0 * tt) * s], 1),
                         yy, rcond=None)[0]
    ph0 = float(math.atan2(-ab[1], ab[0]))

    def resid(p):
        A, fd, al, ph = p
        return (A * np.exp(-al * tt) * np.cos(2 * math.pi * fd * tt + ph) - yy) / max(A0, 1e-300)

    sol = least_squares(resid, [A0, fd0, alpha0, ph0], x_scale=[A0, fd0 * 1e-4, alpha0, 1.0],
                        xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
    A, fd, al, ph = sol.x
    zeta = al / math.sqrt((2 * math.pi * fd) ** 2 + al ** 2)
    return {"f_Hz": float(fd / math.sqrt(1 - zeta ** 2)), "f_d_Hz": float(fd),
            "zeta": float(zeta), "amp": float(abs(A)), "window_s": float(tt[-1]),
            "rms_resid_rel": float(np.sqrt(np.mean(sol.fun ** 2)))}


def run_measure(notch: pd.DataFrame, healthy: pd.DataFrame) -> pd.DataFrame:
    """재장착 산포별 반복수(3σ 대비, C1 여유, 노치 검출, 모드-2 이동 자체) + 추정 CRLB + 온도·질량."""
    rows = []
    ref = notch[(notch.model.isin(["2D_stress_slit0", "3D_flat_slot", "2D_stress_Unotch"]))
                & (notch.level == 1) & (notch.kerf_mm.isin([0.0, 0.25]))]
    for _, c in ref.iterrows():
        for sr in SIGMA_R:
            base = {"block": "repeats", "model": c.model, "position": c.position,
                    "a_bar": c.a_bar, "kerf_mm": c.kerf_mm, "sigma_r_pct": 100 * sr,
                    "d1_pct": c.d1_pct, "d2_pct": c.d2_pct, "smooth_d2_pct": c.smooth_d2_pct,
                    "contrast2_pct": c.contrast2_pct, "c1_margin_pct": c.c1_margin_pct}
            rows.append({**base,
                         # F1: 균열 대 비교장(공통모드 제거 r₂₁, 모드 독립 가정 → var 4/N), 3σ
                         "N_F1_r21_3sigma": required_repeats(c.contrast2_pct / 100, sr, 3.0, 4.0),
                         # 같은 대비를 δ₂ 단독으로(온도 보정 전제, var 2/N)
                         "N_F1_d2_3sigma": required_repeats(c.contrast2_pct / 100, sr, 3.0, 2.0),
                         # F2: D₂₁ 상한(단측 95 %) ≤ ¼ D₂₁ˢ
                         "N_F2_c1_onesided95": required_repeats(c.c1_margin_pct / 100, sr, 1.645, 2.0),
                         # 노치 검출(|δ₁| ≥ 5σ)
                         "N_detect_d1_5sigma": required_repeats(abs(c.d1_pct) / 100, sr, 5.0, 2.0),
                         # 모드-2 이동 자체를 0과 구별(3σ) — 지문이 '거의 0'이라 대개 비현실적
                         "N_resolve_d2_3sigma": required_repeats(abs(c.d2_pct) / 100, sr, 3.0, 2.0)})
    # 추정 CRLB (단일 타격 자유감쇠, 가정 ζ·SNR)
    fl = healthy[(healthy.mesh == "M3") & (healthy.kind == "flap") & (healthy.order <= 3)]
    for _, m in fl.iterrows():
        for zeta in (1e-3, 3e-3):
            for snr in (30.0, 100.0, 300.0):
                for fs in (51200.0, 204800.0):
                    T = min(1.0, max(0.2, 8.0 / (zeta * 2 * math.pi * m.f_Hz)))
                    rows.append({"block": "crlb_single_hit", "mode": f"flap{int(m.order)}",
                                 "f_Hz": m.f_Hz, "zeta_assumed": zeta, "snr_amp_assumed": snr,
                                 "fs_Hz": fs, "record_s": T,
                                 "crlb_sigma_f_over_f_pct": 100 * crlb_damped_sine(m.f_Hz, zeta,
                                                                                   fs, T, snr)})
    rows.append({"block": "temperature", "dlnf_dT_per_K_assumed": DLNF_DT_ASSUMED,
                 "dT_for_0.03pct_K": 3e-4 / abs(DLNF_DT_ASSUMED),
                 "dT_for_0.1pct_K": 1e-3 / abs(DLNF_DT_ASSUMED),
                 "note": "uniform E drift: common-mode, cancels in r21=d2-d1 and r31=d3-d1"})
    return pd.DataFrame(rows)


# ====================================================================== (5) 깊이 역산 앵커
def run_invert(notch: pd.DataFrame) -> pd.DataFrame:
    """2D/3D 노치의 주파수비를 날카로운-균열(TM) 역맵으로 읽은 등가깊이 — Tada·Dimarogonas, f1·lsq."""
    rows = []
    sel = notch[notch.model.isin(["2D_stress_slit0", "2D_stress_flat", "2D_stress_Unotch",
                                  "3D_flat_slot"]) & (notch.level == 1)]
    for _, c in sel.iterrows():
        r = np.array([c.ratio_f1, c.ratio_f2, c.ratio_f3])
        for conv in ("tada", "dimarogonas"):
            for meth in ("f1", "lsq"):
                a_hat, chi = n3.invert_depth(r, conv, meth, xc=float(c.xc_over_L))
                rows.append({"generator": c.model, "position": c.position,
                             "xc_over_L": c.xc_over_L, "a_bar_phys": c.a_bar,
                             "kerf_mm": c.kerf_mm, "inverse_convention": conv, "method": meth,
                             "a_bar_hat": a_hat, "a_eq_over_phys": a_hat / c.a_bar,
                             "depth_eq_mm": a_hat * H_C * 1e3,
                             "resid_noise_units_c0.001": chi})
    df = pd.DataFrame(rows)
    # 깊이 측정 불확도 → δ₁ 예측 불확도 변환용 감도(TM Tada, x_c = 0.2): dδ₁/dā [%/0.01]
    sens = []
    for ab in DEPTHS_ABAR:
        dp = tm_ratio(0.2, ab + 0.01)[0] - tm_ratio(0.2, ab - 0.01)[0]
        sens.append({"generator": "sensitivity_TM_tada", "a_bar_phys": ab,
                     "dd1_pct_per_0.01abar": 100 * dp / 2.0})
    return pd.concat([df, pd.DataFrame(sens)], ignore_index=True)


# ====================================================================== 그림
def figure() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tol = pd.read_csv(out_csv("e1c_tolerance"))
    lk = pd.read_csv(out_csv("e1c_root_lookup"))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ab, mk in ((0.3, "^"), (0.5, "o")):
        for base, ls in (("x0.20", "-"), ("at_null", "--")):
            g = tol[(tol.perturbation == "notch_station") & np.isclose(tol.a_bar, ab)
                    & (tol.base == base)].sort_values("shift_mm")
            ln, = ax[0].plot(g.shift_mm, g.d2_pct, mk + ls, ms=4,
                             label=rf"crack $\bar a$={ab}, {base}")
            ax[0].plot(g.shift_mm, g.smooth_d2_pct, mk + ls, ms=3, alpha=0.35,
                       color=ln.get_color())
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("Notch-station error [mm]")
    ax[0].set_ylabel(r"$\Delta f_2/f_2$ [%] (faint: matched smooth field)")
    ax[0].legend(fontsize=6.5)
    ax[0].grid(alpha=0.3)
    ax[0].set_title("(a)")
    ax[1].semilogx(lk.k_root_nd, lk.rho21, label=r"$\rho_{21}$")
    ax[1].semilogx(lk.k_root_nd, lk.rho31, label=r"$\rho_{31}$")
    ax2 = ax[1].twinx()
    ax2.semilogx(lk.k_root_nd, lk.mode2_null_mm, "k:", label="mode-2 null [mm]")
    ax2.set_ylabel("Mode-2 curvature null from clamp edge [mm]")
    ax[1].set_xlabel(r"Root rotational stiffness $\tilde k = k_\theta L/EI$")
    ax[1].set_ylabel("Healthy ratio / clamped value")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].set_title("(b)")
    fig.tight_layout()
    path = out_fig("e1c_coupon_design")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"[saved] {path}", flush=True)
    return path


# ====================================================================== CLI
def _map(fn, jobs, workers: int):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with get_context("fork").Pool(workers) as p:
        return p.map(fn, jobs, chunksize=1)


def _save(df: pd.DataFrame, name: str, meta: dict | None = None) -> Path:
    p = out_csv(name)
    df = df.copy()
    df["git_head"] = _git_head()
    for k, v in (meta or {}).items():
        df[k] = v
    df.to_csv(p, index=False)
    print(f"[saved] {p} ({len(df)} rows)", flush=True)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m impeller_fingerprint.rev1.e1c_coupon")
    ap.add_argument("item", choices=["healthy3d", "notch", "tolerance", "rootflex", "lookup",
                                     "measure", "invert", "nodes", "negctl", "figure", "saw", "all"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    t0 = time.time()
    todo = (["healthy3d", "notch", "tolerance", "rootflex", "lookup", "measure", "invert",
             "nodes", "negctl", "figure"] if a.item == "all" else [a.item])
    meta = {"coupon_L_mm": L_C * 1e3, "coupon_b_mm": B_C * 1e3, "coupon_h_mm": H_C * 1e3, "E_Pa": E_C,
            "nu": NU_C, "rho": RHO_C}
    for it in todo:
        if it == "healthy3d":
            df, ml = run_healthy3d(workers=min(a.workers, 3))
            _save(df, "e1c_healthy_modes", meta)
            _save(ml, "e1c_mass_limits", {**meta, "tape_mg_assumed": TAPE_MG_ASSUMED})
            _save(analytic_reference(), "e1c_analytic_reference", meta)
        elif it == "notch":
            _save(run_notch(a.workers), "e1c_notch_predictions", meta)
        elif it == "tolerance":
            _save(run_tolerance(), "e1c_tolerance", meta)
        elif it == "rootflex":
            _save(run_rootflex(), "e1c_root_flexibility", meta)
        elif it == "lookup":
            _save(run_lookup(), "e1c_root_lookup", meta)
        elif it == "measure":
            _save(run_measure(pd.read_csv(out_csv("e1c_notch_predictions")),
                              pd.read_csv(out_csv("e1c_healthy_modes"))),
                  "e1c_measurement_requirements", meta)
        elif it == "invert":
            _save(run_invert(pd.read_csv(out_csv("e1c_notch_predictions"))),
                  "e1c_depth_inversion", meta)
        elif it == "negctl":
            _save(run_negative_control(min(a.workers, 4)), "e1c_negative_control", meta)
        elif it == "saw":
            _save(run_saw(min(a.workers, 4)), "e1c_saw_predictions", meta)
        elif it == "nodes":
            _save(eb_nodes(), "e1c_eb_nodes", meta)
        elif it == "figure":
            figure()
        print(f"[e1c] {it} done at {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
