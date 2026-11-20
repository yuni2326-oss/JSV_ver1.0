"""N3 — 베인 균열 지문(곡률-null 무감도)의 강건성 (심사 R1-P4, R2-Q3 일부).

정본 §4.1의 지문은 **x_c/L = 0.2의 단일·완전개구 균열**에서 mode 2가 거의 움직이지 않는다는
예측이다(2D 폭 0 슬릿 ā = 0.5: Δf₁ −10.28 %, Δf₂ 0.0986 %, f₁을 맞춘 매끄러운 가우시안 강성장은
Δf₂ 2.20 % → 판별비 D₂₁ = |Δf₂/f₂|/|Δf₁/f₁| 0.96 % 대 21.4 %). 이 모듈은 그 예측이
**어디까지** 판별력을 유지하는지 다섯 축으로 잰다 — 유지 구간과 소멸 구간을 함께 보고한다.

  sweep      균열 위치 스윕(Timoshenko Mode-I+II, Tada) + 매 위치에서 매끄러운 비교장 재정합
  sweep2d    2D 평면응력 폭 0 슬릿 스폿 검정(x_c/L = 0.1, 0.2, 0.3, 0.5)
  multi      재해(再解) 2균열 모델(중첩 아님)
  breathing  개폐: (i) 조화평균 이선형 추정(축약 근사) (ii) 단측 힌지 시간영역 모델
  root       뿌리 회전유연도 k_root 스윕(곡률-null 이동) [+ 2D 필렛: `fillet`]
  invert     유연도 규약(Tada/Dimarogonas/2D) → 역산 균열깊이 오차

정본 모듈은 읽기만 한다. `crack_shear.assemble`은 균열을 **요소 격자에 스냅**하므로
(ic = round(x_c/L · n_elem)) 위치 스윕에는 쓸 수 없다 — 균열 위치를 절점으로 넣는
다중균열 Timoshenko FE를 여기서 재구현하고 단일균열에서 `crack_shear.frequencies`와
1e-10으로 교차검증한다(`tests/test_rev1_n3.py`).

판정 기준(사전 선언, 결과를 보고 바꾸지 않는다):
  * 잡음 바닥: 기준선 대비 단일모드 상대주파수 이동의 표준편차 √2·c (c = σ_f/f).
  * C1 "곡률-null 지문"   D₂₁^crack ≤ ¼·D₂₁^smooth 이고 mode-2 대비
                          (|δ₂|_smooth − |δ₂|_crack)/(√2 c) ≥ 3   (c = 0.1 %)
  * C2 "다중모드 판별"     χ²₂₃ = Σ_{m=2,3} ((δ_m^smooth − δ_m^crack)/(√2 c))² ≥ χ²₂(0.99) = 9.21
    (δ_m = Δf_m/f_m 부호 포함, f₁은 정합으로 같으므로 자유도 2)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import eigh, lu_factor, lu_solve
from scipy.optimize import brentq, least_squares

from .. import crack_shear as cs
from .. import eb_reference as eb
from .. import geometry as geo
from . import DATA, out_csv, out_fig

VANE = geo.VANE
#: N3 재현 기준(깃 두께 [0.1 mm] → a11 정합 매끄러운 비교장의 f₂ 강하 %, D21 %).
_SMOOTH_REF = {10: (2.20397, 21.4408), 12: (2.549081, 21.236563)}
#: 단면 폭 b는 주파수**비**에서 정확히 상쇄된다(c_MM ∝ 1/b, c_QQ ∝ 1/b, EI·GA·ρA·ρI ∝ b).
#: 정본 a7·a11과 같은 규약(b = 2h)을 그대로 쓴다.
BEAM = cs.TimoBeam(L=VANE.L, h=VANE.h, b=2 * VANE.h, E=VANE.E, rho=VANE.rho,
                   nu=VANE.nu)
#: 매끄러운 경쟁장 폭(x̃ 단위) — 정본 `cli.A11_GAUSS_WIDTH`와 동일.
GAUSS_WIDTH = 0.12
#: 판정 잡음수준 c = σ_f/f.
C_LEVELS = (1e-3, 3e-4)
CHI2_2_99 = 9.2103
SWEEP_XC = (0.05, 0.10, 0.15, 0.18, 0.19, 0.20, 0.21, 0.216, 0.22, 0.23, 0.25,
            0.30, 0.35, 0.40, 0.50)
SWEEP_ABAR = (0.3, 0.5, 0.6)
N_ELEM_SWEEP = 1000      # 모든 스윕 위치가 0.001 L 배수 → 균일격자의 절점에 정확히 놓인다


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[2],
                                       text=True).strip()
    except Exception:                                   # pragma: no cover
        return "unknown"


# ====================================================================== 균열 유연도
def crack_compliance(a_bar: float, convention: str = "tada", shear: bool = True,
                     coupling: float = 0.0, beam: cs.TimoBeam = BEAM) -> dict:
    """`crack_shear.signature`와 **같은 규칙**으로 만든 균열 유연도 행렬 성분.

    shear=False면 c_QQ를 c_MM·h²·1e-6(수치 강체)로 둔다 — 정본과 동일.
    """
    c = cs.compliance(float(a_bar), beam.h, beam.b, beam.E, beam.nu,
                      convention=convention)
    c_qq = c["c_QQ"] if shear else 0.0
    c_mq = coupling * math.sqrt(c["c_MM"] * c_qq) if c_qq > 0 else 0.0
    c_qq_eff = c_qq if c_qq > 0 else c["c_MM"] * beam.h ** 2 * 1e-6
    return {"c_MM": c["c_MM"], "c_QQ": c_qq_eff, "c_MQ": c_mq}


# ====================================================================== Timoshenko FE
@dataclass
class TimoFE:
    """다중균열·뿌리스프링·매끄러운 강성손실을 받는 Timoshenko 캔틸레버 FE.

    요소는 `crack_shear._element_matrices`와 **같은** 정식화(굽힘 2점, 전단 1점 감차,
    일치질량 + 회전관성)이고, 균열은 절점 이중화 + [Δw, Δφ]에 C⁻¹ 유연연결이다.

    cracks:  [(x_c/L, {"c_MM","c_QQ","c_MQ"}), ...] — 위치가 격자 절점에 추가된다.
    k_root:  None이면 고정단(φ₀ = 0), 값이면 회전스프링 [N·m/rad].
    damage:  x̃ → d(x̃) 콜러블(굽힘강성 EI(1−d)), 요소 내 3점 Gauss 평균.
    """

    beam: cs.TimoBeam = BEAM
    n_elem: int = 300
    cracks: Sequence = ()
    k_root: float | None = None
    damage: Callable | None = None
    extra_nodes: Sequence[float] = ()
    _K: np.ndarray = field(init=False, repr=False, default=None)
    _M: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self):
        L = self.beam.L
        xs = np.linspace(0.0, 1.0, self.n_elem + 1)
        pts = [float(x) for x, _ in self.cracks] + [float(x) for x in self.extra_nodes]
        xs = np.unique(np.round(np.concatenate([xs, pts]), 12))
        self.xt = xs                                   # 정규화 절점 좌표
        self.x = xs * L
        ne = xs.size - 1
        crack_at = {}
        for xc, C in self.cracks:
            i = int(np.argmin(np.abs(xs - float(xc))))
            if not 0 < i < ne:
                raise ValueError(f"균열 위치 {xc}는 내부 절점이어야 한다")
            if i in crack_at:
                raise ValueError(f"같은 절점에 균열 두 개: {xc}")
            crack_at[i] = C
        self.crack_nodes = sorted(crack_at)
        # 절점번호: 원래 절점 0..ne, 균열 오른쪽면은 ne+1부터
        right_id = {i: ne + 1 + k for k, i in enumerate(self.crack_nodes)}
        n_nodes = ne + 1 + len(self.crack_nodes)
        ndof = 2 * n_nodes
        K = np.zeros((ndof, ndof))
        M = np.zeros((ndof, ndof))
        b = self.beam
        EI0, GA = b.E * b.I, b.kappa * b.G * b.A
        g3 = np.array([-math.sqrt(0.6), 0.0, math.sqrt(0.6)])
        w3 = np.array([5.0, 8.0, 5.0]) / 9.0
        for e in range(ne):
            le = self.x[e + 1] - self.x[e]
            if self.damage is not None:
                xm = 0.5 * (xs[e] + xs[e + 1]) + 0.5 * (xs[e + 1] - xs[e]) * g3
                dbar = float(np.sum(w3 * np.asarray(self.damage(xm), dtype=float)) / 2.0)
            else:
                dbar = 0.0
            EI = EI0 * (1.0 - dbar)
            Kb = np.zeros((4, 4))
            Kb[1, 1] = Kb[3, 3] = EI / le
            Kb[1, 3] = Kb[3, 1] = -EI / le
            B = np.array([-1.0 / le, -0.5, 1.0 / le, -0.5])
            Ks = GA * le * np.outer(B, B)
            m_t = b.rho * b.A * le / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
            m_r = b.rho * b.I * le / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
            Me = np.zeros((4, 4))
            for i in range(2):
                for j in range(2):
                    Me[2 * i, 2 * j] = m_t[i, j]
                    Me[2 * i + 1, 2 * j + 1] = m_r[i, j]
            n1 = right_id.get(e, e)          # 균열 절점의 오른쪽 요소는 이중화 절점을 쓴다
            n2 = e + 1
            dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
            Ke = Kb + Ks
            K[np.ix_(dofs, dofs)] += Ke
            M[np.ix_(dofs, dofs)] += Me
        for i in self.crack_nodes:
            C = crack_at[i]
            Cm = np.array([[C["c_QQ"], C.get("c_MQ", 0.0)],
                           [C.get("c_MQ", 0.0), C["c_MM"]]])
            Kc = np.linalg.inv(Cm)
            T = np.zeros((2, ndof))
            r = right_id[i]
            T[0, 2 * r] = 1.0
            T[0, 2 * i] = -1.0
            T[1, 2 * r + 1] = 1.0
            T[1, 2 * i + 1] = -1.0
            K += T.T @ Kc @ T
        if self.k_root is not None:
            K[1, 1] += float(self.k_root)
            self.fixed = [0]
        else:
            self.fixed = [0, 1]
        self._K, self._M = K, M
        self.ndof = ndof
        self.right_id = right_id

    @property
    def free(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.ndof), self.fixed)

    def modes(self, n_modes: int = 3):
        # 정본(`crack_shear.frequencies`)과 같은 **전체** 고유분해를 쓴다. 부분집합 드라이버는
        # 수치강체 연결(c_QQ = c_MM·h²·1e-6, Mode-I 단독)이나 강한 뿌리스프링처럼 K의 조건수가
        # 큰 경우 1e-5 규모로 어긋났다(검정 실패로 확인).
        fr = self.free
        lam, vec = eigh(self._K[np.ix_(fr, fr)], self._M[np.ix_(fr, fr)])
        full = np.zeros((self.ndof, n_modes))
        full[fr] = vec[:, :n_modes]
        return np.sqrt(np.maximum(lam[:n_modes], 0.0)) / (2 * math.pi), full

    def frequencies(self, n_modes: int = 3) -> np.ndarray:
        fr = self.free
        lam = eigh(self._K[np.ix_(fr, fr)], self._M[np.ix_(fr, fr)], eigvals_only=True)
        lam = lam[lam > 0]
        return np.sqrt(lam[:n_modes]) / (2 * math.pi)

    def curvature_nulls(self, mode: int, n_modes: int = 3) -> np.ndarray:
        """mode(1-based)의 굽힘모멘트 EIφ′ 영점 위치 x̃ (요소 중점 선형보간, 뿌리·끝 제외)."""
        _, V = self.modes(max(n_modes, mode))
        v = V[:, mode - 1]
        ne = self.xt.size - 1
        kap = np.empty(ne)
        xm = np.empty(ne)
        for e in range(ne):
            n1 = self.right_id.get(e, e)
            kap[e] = (v[2 * (e + 1) + 1] - v[2 * n1 + 1]) / (self.x[e + 1] - self.x[e])
            xm[e] = 0.5 * (self.xt[e] + self.xt[e + 1])
        out = []
        for e in range(ne - 1):
            if kap[e] == 0.0:
                out.append(xm[e])
            elif kap[e] * kap[e + 1] < 0 and 0.02 < xm[e] < 0.98:
                t = kap[e] / (kap[e] - kap[e + 1])
                out.append(xm[e] + t * (xm[e + 1] - xm[e]))
        return np.array(out)


def crack_ratios(cracks: Sequence, n_modes: int = 3, n_elem: int = N_ELEM_SWEEP,
                 k_root: float | None = None) -> np.ndarray:
    """균열 상태/건전 상태 주파수비 f_m/f_m^h (같은 격자·같은 뿌리조건)."""
    xs_extra = [x for x, _ in cracks]
    f0 = TimoFE(n_elem=n_elem, k_root=k_root, extra_nodes=xs_extra).frequencies(n_modes)
    f = TimoFE(n_elem=n_elem, cracks=cracks, k_root=k_root).frequencies(n_modes)
    return f / f0


# ====================================================================== 매끄러운 비교장
def _gauss_field(centers: Sequence[float], amps: Sequence[float], width: float = GAUSS_WIDTH):
    centers = [float(c) for c in centers]
    amps = [float(a) for a in amps]

    def d(xt):
        xt = np.asarray(xt, dtype=float)
        out = np.zeros_like(xt)
        for c, a in zip(centers, amps):
            out = out + a * np.exp(-((xt - c) ** 2) / (width ** 2))
        return out
    return d


_FAN = geo.Beam(L=VANE.L, h=VANE.h, E=VANE.E, rho=VANE.rho, nu=VANE.nu).eb_frequencies(3)


def smooth_drops(centers, amps, width: float = GAUSS_WIDTH) -> np.ndarray:
    """매끄러운 가우시안 강성장의 (Δf₁,Δf₂,Δf₃) 강하[%] — 정본 `_a11_matched_gaussian`과
    **같은 구성**(EB Ritz, 시행함수 7개, 해석 EB 건전값 대비)."""
    r = eb.solve_ritz(VANE.L, VANE.h, VANE.E, VANE.rho, n_modes=3,
                      damage=_gauss_field(centers, amps, width), n_trial=7)
    return np.array([100 * (1 - r[k]["f"] / _FAN[k]) for k in range(3)])


def matched_smooth(target_drop1_pct: float, centers=(0.2,), shape=(1.0,),
                   width: float = GAUSS_WIDTH, d_hi: float = 0.99):
    """Δf₁ 강하를 맞춘 매끄러운 장 — 진폭 비율 `shape`를 고정하고 공통 배율을 푼다.

    반환 (drops[3] %, peak_amp). 최대 진폭 d_hi(< 1, 물리적 강성손실)로 맞출 수 없으면 NaN.
    """
    shape = np.asarray(shape, dtype=float)
    shape = shape / shape.max()

    def g(s):
        return smooth_drops(centers, s * shape, width)[0] - target_drop1_pct

    lo = 1e-4
    if g(lo) * g(d_hi) >= 0:
        return np.full(3, np.nan), float("nan")
    s = float(brentq(g, lo, d_hi, xtol=1e-7))
    return smooth_drops(centers, s * shape, width), s


def matched_smooth_fe(target_drop1_pct: float, centers=(0.2,), shape=(1.0,),
                      k_root: float | None = None, n_elem: int = 600,
                      width: float = GAUSS_WIDTH, d_hi: float = 0.99):
    """같은 Timoshenko FE(뿌리조건 포함) 안의 매끄러운 비교장 — 뿌리 유연도 스윕용."""
    shape = np.asarray(shape, dtype=float)
    shape = shape / shape.max()
    f0 = TimoFE(n_elem=n_elem, k_root=k_root).frequencies(3)

    def drops(s):
        fe = TimoFE(n_elem=n_elem, k_root=k_root,
                    damage=_gauss_field(centers, s * shape, width))
        return 100 * (1 - fe.frequencies(3) / f0)

    lo = 1e-4
    if (drops(lo)[0] - target_drop1_pct) * (drops(d_hi)[0] - target_drop1_pct) >= 0:
        return np.full(3, np.nan), float("nan")
    s = float(brentq(lambda s: drops(s)[0] - target_drop1_pct, lo, d_hi, xtol=1e-7))
    return drops(s), s


# ====================================================================== 판별 지표
def contrast_cols(ratio_crack: np.ndarray, drops_smooth: np.ndarray) -> dict:
    """크랙 주파수비와 매끄러운장 강하[%]로 판별 지표를 만든다(부호 포함 δ = Δf/f)."""
    dc = np.asarray(ratio_crack, dtype=float) - 1.0          # 부호 있는 δ (음수 = 강하)
    ds = -np.asarray(drops_smooth, dtype=float) / 100.0
    out = {f"dcrack_f{i+1}_pct": 100 * dc[i] for i in range(3)}
    out.update({f"dsmooth_f{i+1}_pct": 100 * ds[i] for i in range(3)})
    d21c = abs(dc[1]) / abs(dc[0]) if dc[0] != 0 else np.nan
    d21s = abs(ds[1]) / abs(ds[0]) if np.isfinite(ds[0]) and ds[0] != 0 else np.nan
    d31c = abs(dc[2]) / abs(dc[0]) if dc[0] != 0 else np.nan
    d31s = abs(ds[2]) / abs(ds[0]) if np.isfinite(ds[0]) and ds[0] != 0 else np.nan
    out.update({"D21_crack_pct": 100 * d21c, "D21_smooth_pct": 100 * d21s,
                "D31_crack_pct": 100 * d31c, "D31_smooth_pct": 100 * d31s,
                "D21_ratio_crack_over_smooth": d21c / d21s if d21s else np.nan})
    for c in C_LEVELS:
        sig = math.sqrt(2.0) * c
        tag = f"c{c:g}"
        out[f"contrast_m2_{tag}"] = (abs(ds[1]) - abs(dc[1])) / sig
        out[f"contrast_m3_{tag}"] = (abs(ds[2]) - abs(dc[2])) / sig
        out[f"chi2_23_{tag}"] = float(((ds[1] - dc[1]) / sig) ** 2
                                      + ((ds[2] - dc[2]) / sig) ** 2)
    out["C1_nullfp"] = bool(np.isfinite(d21s) and d21c <= 0.25 * d21s
                            and out["contrast_m2_c0.001"] >= 3.0)
    out["C2_multimode_c0.001"] = bool(out["chi2_23_c0.001"] >= CHI2_2_99)
    out["C2_multimode_c0.0003"] = bool(out["chi2_23_c0.0003"] >= CHI2_2_99)
    return out


# ====================================================================== (0) 재현
def reproduce() -> pd.DataFrame:
    """정본 §4.1 수치 재현 — 진행 전제(재현 실패 시 중단)."""
    rows = []
    # Timoshenko Mode-I+II Tada, n_elem = 300(정본 a11 arm과 같은 격자)
    for ab in (0.3, 0.5, 0.6):
        C = crack_compliance(ab, "tada", shear=True)
        r_mine = crack_ratios([(0.2, C)], n_elem=300)
        r_can = cs.signature(BEAM, [ab], xc_over_L=0.2, n_modes=3, n_elem=300,
                             shear_flex=True, convention="tada")[0]
        rows.append({"check": "timoshenko_I+II_tada_n300", "a_bar": ab,
                     **{f"ratio_f{i+1}": float(r_mine[i]) for i in range(3)},
                     **{f"canon_ratio_f{i+1}": r_can[f"ratio_f{i+1}"] for i in range(3)},
                     "max_rel_diff": float(max(abs(r_mine[i] / r_can[f"ratio_f{i+1}"] - 1)
                                               for i in range(3)))})
    # 매끄러운 비교장: 2D 폭 0 슬릿(ā=0.5, 평면응력)의 f₁ 강하에 정합
    a11 = pd.read_csv(DATA / "a11_crack2d.csv")
    s = a11[(a11.plane == "stress") & np.isclose(a11.a_bar, 0.5)].iloc[0]
    r2d = np.array([s.ratio_f1, s.ratio_f2, s.ratio_f3])
    drops, dmax = matched_smooth(100 * (1 - r2d[0]), centers=(0.2,))
    cc = contrast_cols(r2d, drops)
    rows.append({"check": "2d_stress_slit_a0.5_vs_matched_gauss", "a_bar": 0.5,
                 "ratio_f1": r2d[0], "ratio_f2": r2d[1], "ratio_f3": r2d[2],
                 "gauss_dmax": dmax, "gauss_drop_f1_pct": drops[0],
                 "gauss_drop_f2_pct": drops[1], "gauss_drop_f3_pct": drops[2],
                 "D21_crack_pct": cc["D21_crack_pct"],
                 "D21_smooth_pct": cc["D21_smooth_pct"]})
    return pd.DataFrame(rows)


# ====================================================================== (a) 위치 스윕
def sweep(xcs=SWEEP_XC, abars=SWEEP_ABAR, n_elem: int = N_ELEM_SWEEP) -> pd.DataFrame:
    rows = []
    f0 = TimoFE(n_elem=n_elem).frequencies(3)
    for ab in abars:
        C = crack_compliance(ab, "tada", shear=True)
        for xc in xcs:
            f = TimoFE(n_elem=n_elem, cracks=[(xc, C)]).frequencies(3)
            r = f / f0
            drops, dmax = matched_smooth(100 * (1 - r[0]), centers=(xc,))
            rows.append({"model": "timoshenko_I+II_tada", "n_elem": n_elem,
                         "xc_over_L": xc, "a_bar": ab, "gauss_dmax": dmax,
                         "gauss_width_xt": GAUSS_WIDTH,
                         **{f"ratio_f{i+1}": float(r[i]) for i in range(3)},
                         **contrast_cols(r, drops)})
            print(f"[n3 sweep] ā={ab} x_c/L={xc:.3f} Δf1={100*(r[0]-1):+.3f}% "
                  f"Δf2={100*(r[1]-1):+.4f}% D21 {rows[-1]['D21_crack_pct']:.2f}% vs "
                  f"{rows[-1]['D21_smooth_pct']:.2f}%  χ²23={rows[-1]['chi2_23_c0.001']:.1f}",
                  flush=True)
    return pd.DataFrame(rows)


def null_positions(n_elem: int = 1000) -> dict:
    """건전 캔틸레버 mode 2·3의 곡률(굽힘모멘트) 영점 x/L — FE와 해석 EB를 함께."""
    fe = TimoFE(n_elem=n_elem)
    out = {"fe_mode2": fe.curvature_nulls(2).tolist(),
           "fe_mode3": fe.curvature_nulls(3).tolist()}
    # 해석 EB: φ″(x) = cosh βx + cos βx − σ (sinh βx + sin βx)
    for m, bl in ((2, geo.BETA_L[1]), (3, geo.BETA_L[2])):
        sig = (math.cosh(bl) + math.cos(bl)) / (math.sinh(bl) + math.sin(bl))

        def phi2(x, bl=bl, sig=sig):
            t = bl * x
            return math.cosh(t) + math.cos(t) - sig * (math.sinh(t) + math.sin(t))
        xs = np.linspace(0.01, 0.99, 2000)
        v = np.array([phi2(x) for x in xs])
        roots = [brentq(phi2, xs[i], xs[i + 1]) for i in range(len(xs) - 1)
                 if v[i] * v[i + 1] < 0]
        out[f"eb_mode{m}"] = roots
    return out


# ====================================================================== (a') 2D 스폿 검정
SWEEP2D_XC = (0.10, 0.20, 0.30, 0.50)
A11_REF_WORK = 3          # 정본 a11 작업격자 = ref 3 (23 568 DOF)


def _a11_grid(ref: int) -> dict:
    return {"nx_left": 8 * ref, "nx_right": 32 * ref, "nz_below": 4 * ref,
            "nz_above": 4 * ref, "bias": 12.0}


def sweep2d(xcs=SWEEP2D_XC, a_bar: float = 0.5, plane: str = "stress",
            ref: int = A11_REF_WORK) -> pd.DataFrame:
    """2D 평면응력 폭 0 슬릿(정본 a11과 같은 격자 규약) — 위치만 바꾼다."""
    from .. import crack2d as c2
    rows = []
    kw = _a11_grid(ref)
    for xc in xcs:
        co0, cn0, _ = c2.slit_mesh(VANE.L, VANE.h, a_bar, xc_over_L=xc, crack=False, **kw)
        f0, _, nd0 = c2.flap_modes(co0, cn0, VANE.E, VANE.nu, VANE.rho, plane=plane)
        co, cn, info = c2.slit_mesh(VANE.L, VANE.h, a_bar, xc_over_L=xc, **kw)
        f, kinds, nd = c2.flap_modes(co, cn, VANE.E, VANE.nu, VANE.rho, plane=plane)
        r = f[:3] / f0[:3]
        drops, dmax = matched_smooth(100 * (1 - r[0]), centers=(xc,))
        tm = crack_ratios([(xc, crack_compliance(a_bar, "tada", True))])
        rows.append({"model": f"2d_{plane}_slit", "ref": ref, "ndof": nd,
                     "xc_over_L": xc, "a_bar": a_bar, "gauss_dmax": dmax,
                     **{f"ratio_f{i+1}": float(r[i]) for i in range(3)},
                     **{f"tm_ratio_f{i+1}": float(tm[i]) for i in range(3)},
                     "flap_kinds": ",".join(kinds[:4]),
                     **contrast_cols(r, drops)})
        print(f"[n3 2d] x_c/L={xc} Δf1={100*(r[0]-1):+.3f}% (TM {100*(tm[0]-1):+.3f}%) "
              f"Δf2={100*(r[1]-1):+.4f}% (TM {100*(tm[1]-1):+.4f}%) "
              f"D21 {rows[-1]['D21_crack_pct']:.2f}% vs {rows[-1]['D21_smooth_pct']:.2f}%",
              flush=True)
    return pd.DataFrame(rows)


# ====================================================================== (b) 다중균열
MULTI_CASES = (
    # (설명, [(x/L, ā), ...])
    ("base_single", [(0.20, 0.5)]),
    ("x2_0.35_a0.3", [(0.20, 0.5), (0.35, 0.3)]),
    ("x2_0.50_a0.3", [(0.20, 0.5), (0.50, 0.3)]),
    ("x2_0.65_a0.3", [(0.20, 0.5), (0.65, 0.3)]),
    ("x2_0.35_a0.5", [(0.20, 0.5), (0.35, 0.5)]),
    ("x2_0.50_a0.5", [(0.20, 0.5), (0.50, 0.5)]),
    ("x2_0.65_a0.5", [(0.20, 0.5), (0.65, 0.5)]),
    ("near_null_0.20+0.23_a0.3", [(0.20, 0.5), (0.23, 0.3)]),
    ("near_null_0.19+0.23_a0.5", [(0.19, 0.5), (0.23, 0.5)]),
    ("near_null_0.20+0.216_a0.3", [(0.20, 0.3), (0.216, 0.3)]),
)


def multi(cases=MULTI_CASES, n_elem: int = N_ELEM_SWEEP) -> pd.DataFrame:
    """재해 다중균열(유연연결 여러 개를 한 번에 푼다) + 단일균열 중첩과의 차이."""
    rows = []
    f0 = TimoFE(n_elem=n_elem).frequencies(3)
    singles = {}
    for name, cr in cases:
        C = [(x, crack_compliance(a, "tada", True)) for x, a in cr]
        r = TimoFE(n_elem=n_elem, cracks=C).frequencies(3) / f0
        # 중첩(비교용): 각 균열 단독 이동의 합
        sup = np.zeros(3)
        ind_drop1 = []
        for x, a in cr:
            key = (x, a)
            if key not in singles:
                singles[key] = TimoFE(n_elem=n_elem, cracks=[
                    (x, crack_compliance(a, "tada", True))]).frequencies(3) / f0
            sup += singles[key] - 1.0
            ind_drop1.append(100 * (1 - singles[key][0]))
        target = 100 * (1 - r[0])
        # 비교장 (i): 주균열 위치의 단일 가우시안
        d1, s1 = matched_smooth(target, centers=(cr[0][0],))
        # 비교장 (ii): 균열마다 가우시안 — 각자의 단독 Δf₁에 맞춘 진폭비를 고정하고 총 Δf₁에 재정합
        amps = []
        for (x, a), dd in zip(cr, ind_drop1):
            amps.append(matched_smooth(dd, centers=(x,))[1])
        d2, s2 = matched_smooth(target, centers=[x for x, _ in cr], shape=amps)
        base = {"case": name, "cracks": json.dumps(cr), "n_cracks": len(cr),
                **{f"ratio_f{i+1}": float(r[i]) for i in range(3)},
                **{f"superpos_dpct_f{i+1}": float(100 * sup[i]) for i in range(3)}}
        for tag, dr, s in (("single_gauss_at_x1", d1, s1), ("gauss_per_crack", d2, s2)):
            rows.append({**base, "comparator": tag, "gauss_peak": s,
                         **contrast_cols(r, dr)})
        print(f"[n3 multi] {name}: Δf1={100*(r[0]-1):+.3f}% Δf2={100*(r[1]-1):+.4f}% "
              f"(superpos {100*sup[1]:+.4f}%) D21={rows[-1]['D21_crack_pct']:.2f}% vs "
              f"{rows[-2]['D21_smooth_pct']:.2f}/{rows[-1]['D21_smooth_pct']:.2f}%", flush=True)
    return pd.DataFrame(rows)


# ====================================================================== (c) 개폐(breathing)
def _bandpass_zc_freq(t: np.ndarray, q: np.ndarray, f_c: float, rel: float = 0.25,
                      trim: float = 0.05) -> float:
    """f_c ± rel 대역만 남긴(영위상 FFT 마스크) 뒤 상향 영교차 평균주기로 주파수를 낸다.

    단측 균열의 이선형 결합은 모드 사이로 에너지를 옮긴다. 열림 상태에서 건전 mode-2 사영
    q₂에 f₁ 성분이 섞이면 날 영교차가 오염되므로 대역통과로 걷어낸다. 마스크 가장자리 효과를
    피해 앞뒤 trim 비율의 교차는 버린다.
    """
    x = q - q.mean()
    n = x.size
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, t[1] - t[0])
    X[(f < (1 - rel) * f_c) | (f > (1 + rel) * f_c)] = 0.0
    y = np.fft.irfft(X, n)
    s_ = np.sign(y)
    idx = np.nonzero((s_[:-1] < 0) & (s_[1:] >= 0))[0]
    k0, k1 = int(trim * idx.size), int((1 - trim) * idx.size)
    idx = idx[k0:k1]
    if idx.size < 5:
        return float("nan")
    tc = t[idx] - y[idx] * (t[idx + 1] - t[idx]) / (y[idx + 1] - y[idx])
    return float((tc.size - 1) / (tc[-1] - tc[0]))


def _fft_peak(t: np.ndarray, q: np.ndarray, f_guess: float, rel_win: float = 0.30,
              pad: int = 8) -> float:
    """Hann 창 + 영채움 FFT의 f_guess ± rel_win 안 최대봉(포물선 보간).

    창은 ±30 %: 깊은 균열의 열림 이동(ā = 0.6에서 mode 1 −24 %)을 담고 이웃 모드(f₂/f₁ ≈ 6.3,
    f₃/f₂ ≈ 2.8)는 배제한다. ±5 %이던 초판은 ā = 0.5 열림(−10 %)을 창 밖에 둬 가장자리 값을
    돌려줬다(검정 실패로 확인).
    """
    dt = t[1] - t[0]
    x = (q - q.mean()) * np.hanning(q.size)
    n = int(2 ** math.ceil(math.log2(q.size * pad)))
    X = np.abs(np.fft.rfft(x, n))
    f = np.fft.rfftfreq(n, dt)
    sel = np.nonzero(np.abs(f - f_guess) <= rel_win * f_guess)[0]
    k = sel[np.argmax(X[sel])]
    if 0 < k < X.size - 1:
        a, b_, c = np.log(X[k - 1] + 1e-300), np.log(X[k] + 1e-300), np.log(X[k + 1] + 1e-300)
        p = 0.5 * (a - c) / (a - 2 * b_ + c)
        return float(f[k] + p * (f[1] - f[0]))
    return float(f[k])


@dataclass
class MasslessHinge:
    """**질량 없는 힌지** 단측 균열 모델 — EB Hermite 보, 집중(병진)질량, 회전자유도 정적축약.

    회전자유도에 질량이 없으므로 힌지 상대회전 Δθ는 순간 모멘트 평형으로 정해진다:
        열림(m > 0):  m = k_o·Δθ,     닫힘(m ≤ 0):  Δθ = 0 (구속, 건전 보).
    두 상태의 축약 강성은 m = 0인 전환점에서 **같은 힘**을 준다(이선형 법칙의 연속성) —
    그래서 전환 때 속도·가속도가 연속이고 사영·충격·떨림이 생기지 않는다. 상태 지시량은
    처짐 w에 대한 선형식이다:
        열림:  Δθ(w) = g_o·w   (열림 유지 ⇔ Δθ > 0)
        닫힘:  λ(w)  = g_c·w   (닫힘 유지 ⇔ λ ≤ 0; λ는 구속 모멘트, 열림 스프링 모멘트와 같은 부호)
    병진 질량만 쓰므로 건전 고유값은 해석 EB보다 O(h²) 낮다 — 비(균열/건전)는 같은 모델 안에서 낸다.

    **왜 이 모델인가(시행착오, 2026-10-06)**: 회전관성이 있는(일치질량) 힌지에서는 상대회전이
    관성을 가져 닫힐 때 속도 처리가 필요했다. (1) 닫힘 = 큰 벌칙 회전스프링: 국소 힌지 모드가
    전환마다 들떠 20주기에 1.2만 회 전환, 평균가속도·HHT-α 모두 발산. (2) 닫힘 = 정확 구속 +
    상대속도 소성사영: 전환시각 오차의 사영이 변형에너지를 주입해 0.02 s 뒤 발산, 이를 막아도
    예하중이 있으면 미소 접촉이 끝없이 반복(Zeno 떨림, 6주기에 1 064회). 힌지를 질량 없게 두면
    상대회전이 순간 평형으로 정해져 두 문제가 원리적으로 사라진다.
    """

    c_theta: float
    xc: float = 0.2
    n_elem: int = 240
    beam: cs.TimoBeam = BEAM

    def __post_init__(self):
        b = self.beam
        n = self.n_elem
        ic = self.xc * n
        if abs(ic - round(ic)) > 1e-9 or not 0 < round(ic) < n:
            raise ValueError("x_c·n_elem는 내부 정수 절점이어야 한다")
        self.ic = ic = int(round(ic))
        le = b.L / n
        EI, rA = b.E * b.I, b.rho * b.A
        ke = EI / le ** 3 * np.array([[12, 6 * le, -12, 6 * le],
                                      [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                      [-12, -6 * le, 12, -6 * le],
                                      [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
        # 자유도: w_1..w_n (병진), θ_1..θ_n, θ_R (회전) — 고정단 w₀ = θ₀ = 0
        nw = n
        nth = n + 1
        iw = {i: i - 1 for i in range(1, n + 1)}
        ith = {i: nw + (i - 1) for i in range(1, n + 1)}
        thR = nw + n
        nd = nw + nth
        K = np.zeros((nd, nd))
        for e in range(n):
            dofs = []
            for node, side in ((e, "L"), (e + 1, "R")):
                dofs.append(iw.get(node, -1))
                if node == ic and side == "L":         # 균열 오른쪽 요소의 왼쪽 회전 = θ_R
                    dofs.append(thR)
                else:
                    dofs.append(ith.get(node, -1))
            for a_, da in enumerate(dofs):
                if da < 0:
                    continue
                for b_, db in enumerate(dofs):
                    if db < 0:
                        continue
                    K[da, db] += ke[a_, b_]
        T = np.zeros(nth)
        T[thR - nw] = 1.0
        T[ith[ic] - nw] = -1.0
        Kww, Kwt, Ktt = K[:nw, :nw], K[:nw, nw:], K[nw:, nw:]
        self.k_o = 1.0 / self.c_theta
        Ktt_o = Ktt + self.k_o * np.outer(T, T)
        # 열림 축약
        Xo = np.linalg.solve(Ktt_o, Kwt.T)              # θ = −Xo·w
        self.Ko = Kww - Kwt @ Xo
        self.g_o = -(T @ Xo)                            # Δθ = g_o·w
        # 닫힘(구속 T·θ = 0) 축약과 구속 모멘트
        Xt = np.linalg.solve(Ktt, Kwt.T)
        y = np.linalg.solve(Ktt, T)
        TKT = float(T @ y)
        # λ = −(T Ktt⁻¹ Ktw w)/(T Ktt⁻¹ T),  θ = −Ktt⁻¹(Ktw w + λ T)
        self.g_c = -(T @ Xt) / TKT
        Xc = Xt - np.outer(y, T @ Xt) / TKT             # θ = −Xc·w (구속 만족)
        self.Kc = Kww - Kwt @ Xc
        self.Kc = 0.5 * (self.Kc + self.Kc.T)
        self.Ko = 0.5 * (self.Ko + self.Ko.T)
        m = np.full(nw, rA * le)
        m[-1] = 0.5 * rA * le                           # 끝 절점
        self.M = np.diag(m)
        self.tip = nw - 1
        self.EI, self.le = EI, le

    def K(self, state: int) -> np.ndarray:
        return self.Ko if state > 0 else self.Kc

    def modes(self, state: int, n_modes: int = 3):
        lam, V = eigh(self.K(state), self.M)
        return np.sqrt(lam[:n_modes]) / (2 * math.pi), V[:, :n_modes]

    def indicator(self, state: int, w: np.ndarray) -> float:
        """현 상태 유지 ⇔ 지시량 > 0. 열림: Δθ, 닫힘: −λ."""
        return float(self.g_o @ w) if state > 0 else -float(self.g_c @ w)


def simulate_exact(model: MasslessHinge, w0: np.ndarray, v0: np.ndarray, t_end: float,
                   dt: float, F: np.ndarray | None = None, record=()):
    """**사건 구동 정확해** — 각 상태 안의 선형 운동을 모드 중첩으로 정확히 쓰고, 상태 전환은
    해석적 지시량의 근(brentq)으로 잡는다. 기록·근 탐색 표본 간격은 dt.

    각 구간에서 에너지는 정확히 보존되고 전환점(지시량 0)에서 퍼텐셜이 연속이므로, 스침
    잔물결로 표본 사이의 짧은 반대부호 구간을 놓쳐도 **에너지 오류가 생기지 않는다**(놓친
    구간 동안 현 상태의 운동을 따를 뿐). 시간적분 오차·주기연장도 없다.

    **왜 이 방식인가(시행착오, 2026-10-06)**: 같은 질량 없는 힌지 모델을 Newmark로 적분하면
    균열이 mode-2 null 근방일 때 힌지 지시량이 dt로 분해되지 않는 고주파 성분에 지배되는
    스침 궤적이 되어, 한 스텝 안의 이중 교차를 놓친 채 틀린 상태로 적분해 에너지가 쌓였다
    (mode-2 초기조건 200주기 뒤, 열림 예하중에서는 수십 주기 안에 발산). 상태 교정·공통
    Ritz 축소(꺾임 방향이 닫힘에서 수 MHz 강성 모드가 됨)로도 막지 못했다.
    """
    F = np.zeros_like(w0) if F is None else F
    Mm = model.M
    sysd = {}
    for st_ in (1, -1):
        lam, Phi = eigh(model.K(st_), Mm)
        om = np.sqrt(np.maximum(lam, 0.0))
        ws = np.linalg.solve(model.K(st_), F)
        g = model.g_o if st_ > 0 else -model.g_c
        sysd[st_] = {"om": om, "Phi": Phi, "ws": ws, "g": g, "gP": g @ Phi, "gws": float(g @ ws),
                     "RP": np.array([r @ Phi for r in record]) if record else None,
                     "Rws": np.array([float(r @ ws) for r in record]) if record else None}

    def coeffs(st_, w, v):
        S = sysd[st_]
        c = S["Phi"].T @ (Mm @ (w - S["ws"]))
        d = S["Phi"].T @ (Mm @ v) / np.where(S["om"] > 0, S["om"], 1.0)
        return c, d

    def state_at(st_, c, d, tau):
        S = sysd[st_]
        co, si = np.cos(S["om"] * tau), np.sin(S["om"] * tau)
        w = S["ws"] + S["Phi"] @ (c * co + d * si)
        v = S["Phi"] @ (S["om"] * (-c * si + d * co))
        return w, v

    def ind(st_, c, d, tau):
        S = sysd[st_]
        tau = np.atleast_1d(tau)
        A = np.cos(np.outer(tau, S["om"])) * c + np.sin(np.outer(tau, S["om"])) * d
        return S["gws"] + A @ S["gP"]

    def energy(st_, w, v):
        return (0.5 * float(v @ (Mm @ v)) + 0.5 * float(w @ (model.K(st_) @ w))
                - float(F @ w))

    st = 1 if float(model.g_o @ w0) > 0 else -1
    if st < 0 and -float(model.g_c @ w0) < 0:
        st = 1
    n = int(round(t_end / dt))
    grid = np.arange(n + 1) * dt
    out_s = np.empty(n + 1)
    rec = np.empty((len(record), n + 1))
    t0 = 0.0
    c, d = coeffs(st, w0, v0)
    e0 = energy(st, w0, v0)
    e_scale = max(abs(e0), 1e-300)
    e_dev, n_switch, n_graze = 0.0, 0, 0
    delta = 1e-4 * dt                                # 스침 판정용 앞보기 간격
    t_from = 0.0                                     # 이 구간에서 근을 찾기 시작할 상대시각
    j = 0
    block = 64                                       # 전환 직후에는 작게, 이후 기하급수로 키운다
    while j <= n:
        jb = min(j + block, n + 1)
        taus = grid[j:jb] - t0
        I = ind(st, c, d, taus)
        bad = np.nonzero((I < 0) & (taus > t_from))[0]
        k_end = jb if bad.size == 0 else j + int(bad[0])
        if k_end > j:                                # [j, k_end) 기록
            S = sysd[st]
            tt = grid[j:k_end] - t0
            A = np.cos(np.outer(tt, S["om"])) * c + np.sin(np.outer(tt, S["om"])) * d
            if record:
                rec[:, j:k_end] = S["Rws"][:, None] + (A @ S["RP"].T).T
            out_s[j:k_end] = st
        if bad.size == 0:
            j = jb
            block = min(2 * block, 8192)
            continue
        block = 64
        ta = max(grid[k_end - 1] - t0, t_from) if k_end > j else t_from
        tb = grid[k_end] - t0
        fa = float(ind(st, c, d, ta)[0])
        if fa >= 0:
            tr = brentq(lambda x: float(ind(st, c, d, x)[0]), ta, tb, xtol=1e-15 * dt + 1e-300,
                        rtol=1e-15, maxiter=300)
            w, v = state_at(st, c, d, tr)
            e_dev = max(e_dev, abs(energy(st, w, v) - e0) / e_scale)
            t0 += tr
            st = -st
            c, d = coeffs(st, w, v)
            e_dev = max(e_dev, abs(energy(st, w, v) - e0) / e_scale)
            n_switch += 1
            t_from = 0.0
        else:
            # 구간 시작(직전 전환점, 지시량 ≈ 0)에서 이미 음 — 스침. 두 상태의 지시량을 같은
            # 출발점에서 δ 앞서 보고 양인 쪽으로 간다(전환점에서 힘·속도가 연속이라 정확).
            w, v = state_at(st, c, d, ta)
            cand = []
            for s_ in (st, -st):
                cs_, ds_ = coeffs(s_, w, v)
                cand.append((float(ind(s_, cs_, ds_, delta)[0]), s_, cs_, ds_))
            best = max(cand, key=lambda z: z[0])
            t0 += ta
            if best[1] != st:
                n_switch += 1
            st, c, d = best[1], best[2], best[3]
            e_dev = max(e_dev, abs(energy(st, w, v) - e0) / e_scale)
            t_from = delta
            n_graze += 1
        # j는 그대로: 현재 표본을 새 구간에서 다시 평가한다
    return grid, out_s, list(rec), {"n_switch": n_switch, "n_state_fix": 0,
                                    "n_switch_cap": 0, "n_graze_skip": n_graze,
                                    "energy_dev_max": e_dev}


def breathing_case(xc: float, a_bar: float, n_cycles1: float = 60.0,
                   dt_frac: float = 400.0, n_elem: int = 240, preload: str = "none",
                   amp_ratio: float = float("inf"), ic_mode: int = 1,
                   n_cycles2: float = 300.0) -> dict:
    """질량 없는 힌지 단측 균열의 시간영역 결과. preload ∈ {none, open, closed}.

    amp_ratio = (초기 동적 지시량 진폭)/(정적 예하중 지시량) — 열림 예하중은 Δθ,
    닫힘 예하중은 구속 모멘트로 잰다. preload가 있을 때만 의미가 있다.
    """
    c_th = cs.compliance(a_bar, BEAM.h, BEAM.b, BEAM.E, BEAM.nu, convention="tada")["c_MM"]
    mdl = MasslessHinge(c_theta=c_th, xc=xc, n_elem=n_elem)
    fh, Vh = mdl.modes(-1, 3)                     # 닫힘 = 건전
    fo, Vo = mdl.modes(+1, 3)                     # 열림
    proj = [mdl.M @ Vh[:, k] for k in range(3)]   # 질량정규 건전 모드의 모달좌표
    dt = 1.0 / fh[1] / dt_frac
    # mode 2는 짧게: 균열이 mode-2 null 근방이면 힌지 지시량이 미분해 고주파 성분에 지배되는
    # 스침 궤적이 되어, 긴 기록에서 사건 탐지 실패가 누적되고 에너지가 쌓였다(200주기 이후 발산,
    # 확인). 유효성은 에너지 편차·상태교정 횟수로 판정해 함께 낸다.
    t_end = n_cycles1 / fh[0] if ic_mode == 1 else n_cycles2 / fh[1]
    F = np.zeros(mdl.M.shape[0])
    w_s = np.zeros_like(F)
    if preload == "open":
        F[mdl.tip] = 1.0                          # 위로 당김 → 아래면 균열이 열린다
        w_s = np.linalg.solve(mdl.Ko, F)
        if float(mdl.g_o @ w_s) <= 0:
            raise RuntimeError("열림 예하중인데 Δθ ≤ 0")
        phi = Vo[:, ic_mode - 1].copy()
        ind_s, ind_phi = abs(float(mdl.g_o @ w_s)), abs(float(mdl.g_o @ phi))
    elif preload == "closed":
        F[mdl.tip] = -1.0
        w_s = np.linalg.solve(mdl.Kc, F)
        if float(mdl.g_c @ w_s) >= 0:
            raise RuntimeError("닫힘 예하중인데 구속 모멘트 ≥ 0")
        phi = Vh[:, ic_mode - 1].copy()
        ind_s, ind_phi = abs(float(mdl.g_c @ w_s)), abs(float(mdl.g_c @ phi))
    else:
        phi = Vo[:, ic_mode - 1].copy()
        if float(mdl.g_o @ phi) < 0:
            phi = -phi
        ind_s, ind_phi = 1.0, 1.0
    A = (amp_ratio * ind_s / max(ind_phi, 1e-300) if preload != "none"
         else 1e-6 / max(abs(float(mdl.g_o @ phi)), 1e-300))
    w0 = w_s + A * phi
    t, sta, (q1, q2, q3), info = simulate_exact(mdl, w0, np.zeros_like(w0), t_end, dt,
                                                F=F, record=proj)
    qs = [q1, q2, q3]
    fz = [_bandpass_zc_freq(t, q, fh[k]) for k, q in enumerate(qs)]
    ff = [_fft_peak(t, q, fh[k]) for k, q in enumerate(qs)]
    # 모드 간 에너지 이동 진단: 초기조건 모드 외 성분의 rms 비(앞 10 %와 뒤 10 %)
    nq = qs[0].size
    rms_head = [float(np.sqrt(np.mean(q[: nq // 10] ** 2))) for q in qs]
    rms_tail = [float(np.sqrt(np.mean(q[-nq // 10:] ** 2))) for q in qs]
    # 정확해이므로 주기연장 보정이 없다(열 이름은 하위호환으로 유지)
    fzc = list(fz)
    return {"xc_over_L": xc, "a_bar": a_bar, "preload": preload, "amp_ratio": amp_ratio,
            "ic_mode": ic_mode, "dt_frac_T2": dt_frac, "n_elem": n_elem,
            "t_end_s": t_end, "n_switch": info["n_switch"],
            "n_state_fix": info["n_state_fix"], "n_switch_cap": info["n_switch_cap"],
            "energy_dev_max": info["energy_dev_max"],
            "n_graze_skip": info["n_graze_skip"],
            "td_valid": bool(info["energy_dev_max"] < 1e-5),
            "frac_time_open": float(np.mean(sta > 0)),
            **{f"f{k+1}_healthy": float(fh[k]) for k in range(3)},
            **{f"f{k+1}_open": float(fo[k]) for k in range(3)},
            **{f"f{k+1}_td_zc": fz[k] for k in range(3)},
            **{f"f{k+1}_td_zc_corr": fzc[k] for k in range(3)},
            **{f"f{k+1}_td_fft": ff[k] for k in range(3)},
            **{f"q{k+1}_rms_head": rms_head[k] for k in range(3)},
            **{f"q{k+1}_rms_tail": rms_tail[k] for k in range(3)}}


def breathing(xcs=(0.20, 0.35), abars=(0.3, 0.5), n_elem: int = 240,
              dt_frac: float = 400.0) -> tuple:
    """(i) 조화평균 이선형 추정(축약 근사) (ii) 질량 없는 단측 힌지 시간영역 — 예하중 3체제 + dt 수렴.

    깊이 역산: 시간영역 Δf₁을 **같은 모델의 열린균열 맵** f₁(ā)으로 되읽어 ā_inv/ā를 낸다
    (작은 깊이에서 유연도 ∝ ā²이면 √(이동비)에 가깝다 — "깊이 절반"이 아니다).
    """
    rows, conv = [], []

    def d_of(f, fh):
        return f / fh - 1.0

    for xc in xcs:
        for ab in abars:
            base = breathing_case(xc, ab, n_elem=n_elem, preload="none", ic_mode=1,
                                  dt_frac=dt_frac)
            m2 = breathing_case(xc, ab, n_elem=n_elem, preload="none", ic_mode=2,
                                dt_frac=dt_frac)
            fh = np.array([base[f"f{k}_healthy"] for k in (1, 2, 3)])
            fo = np.array([base[f"f{k}_open"] for k in (1, 2, 3)])
            fhm = 2 * fo * fh / (fo + fh)                       # 조화평균(축약 근사)
            # 주 추정량 = FFT 피크(Hann·영채움·포물선). 대역통과 영교차는 순수 열림 검정에서
            # +0.008 %p 편향이 있어(확인) 진단용으로만 남긴다.
            ftd = np.array([base["f1_td_fft"], m2["f2_td_fft"], np.nan])

            def f1_open(a, xc=xc):
                c = cs.compliance(a, BEAM.h, BEAM.b, BEAM.E, BEAM.nu, convention="tada")["c_MM"]
                return MasslessHinge(c_theta=c, xc=xc, n_elem=n_elem).modes(+1, 1)[0][0]
            a_inv_td = brentq(lambda a: f1_open(a) / fh[0] - ftd[0] / fh[0], 0.01, 0.9,
                              xtol=1e-7)
            a_inv_hm = brentq(lambda a: f1_open(a) / fh[0] - fhm[0] / fh[0], 0.01, 0.9,
                              xtol=1e-7)
            for tag, f, a_inv, info in (("open", fo, ab, None),
                                        ("harmonic_mean", fhm, a_inv_hm, None),
                                        ("td_unilateral", ftd, a_inv_td, base)):
                d = d_of(f, fh)
                do = d_of(fo, fh)
                rows.append({"xc_over_L": xc, "a_bar": ab, "model": tag, "preload": "none",
                             "amp_ratio": np.inf, "dt_frac_T2": dt_frac,
                             **{f"d_f{k+1}_pct": 100 * d[k] for k in range(3)},
                             "D21_pct": 100 * abs(d[1]) / abs(d[0]),
                             "shift1_over_open": d[0] / do[0],
                             "shift2_over_open": d[1] / do[1],
                             "a_inv_open_map": a_inv, "a_inv_over_true": a_inv / ab,
                             "sqrt_shift1_ratio": math.sqrt(abs(d[0] / do[0])),
                             "frac_time_open": base["frac_time_open"] if info else np.nan,
                             "n_switch_ic1": base["n_switch"] if info else np.nan,
                             "n_switch_ic2": m2["n_switch"] if info else np.nan,
                             "td_valid": (bool(base["td_valid"] and m2["td_valid"]) if info
                                          else np.nan),
                             "energy_dev_max": (max(base["energy_dev_max"], m2["energy_dev_max"])
                                                if info else np.nan)})
            print(f"[n3 breath] x={xc} ā={ab}: open Δf1 {100*(fo[0]/fh[0]-1):+.3f}% Δf2 "
                  f"{100*(fo[1]/fh[1]-1):+.4f}% | HM {100*(fhm[0]/fh[0]-1):+.3f}/"
                  f"{100*(fhm[1]/fh[1]-1):+.4f}% | TD {100*(ftd[0]/fh[0]-1):+.3f}/"
                  f"{100*(ftd[1]/fh[1]-1):+.4f}% | ā_inv/ā {a_inv_td/ab:.3f} | sw "
                  f"{base['n_switch']}/{m2['n_switch']}", flush=True)
            for pre in ("open", "closed"):
                for ar in (0.5, 2.0, 8.0):
                    c1 = breathing_case(xc, ab, n_elem=n_elem, preload=pre, amp_ratio=ar,
                                        ic_mode=1, dt_frac=dt_frac)
                    c2_ = breathing_case(xc, ab, n_elem=n_elem, preload=pre, amp_ratio=ar,
                                         ic_mode=2, dt_frac=dt_frac)
                    f = np.array([c1["f1_td_fft"], c2_["f2_td_fft"], np.nan])
                    d = d_of(f, fh)
                    do = d_of(fo, fh)
                    rows.append({"xc_over_L": xc, "a_bar": ab, "model": "td_unilateral",
                                 "preload": pre, "amp_ratio": ar, "dt_frac_T2": dt_frac,
                                 **{f"d_f{k+1}_pct": 100 * d[k] for k in range(3)},
                                 "D21_pct": (100 * abs(d[1]) / abs(d[0])
                                             if abs(d[0]) > 1e-7 else np.nan),
                                 "shift1_over_open": d[0] / do[0],
                                 "shift2_over_open": d[1] / do[1],
                                 "frac_time_open": c1["frac_time_open"],
                                 "n_switch_ic1": c1["n_switch"], "n_switch_ic2": c2_["n_switch"],
                                 "td_valid": bool(c1["td_valid"] and c2_["td_valid"]),
                                 "energy_dev_max": max(c1["energy_dev_max"],
                                                       c2_["energy_dev_max"])})
                    print(f"[n3 breath]   preload={pre} amp={ar}: Δf1 {100*d[0]:+.4f}% "
                          f"Δf2 {100*d[1]:+.5f}% open-frac {c1['frac_time_open']:.2f} "
                          f"sw {c1['n_switch']}/{c2_['n_switch']}", flush=True)
    # 정확해이므로 시간적분 오차는 없다 — 표본간격(근 탐색·기록 해상도)과 기록 길이에 대한
    # 수렴을 본다(x = 0.2, ā = 0.5, 예하중 없음).
    for frac, n1, n2 in ((200.0, 60.0, 300.0), (400.0, 60.0, 300.0), (800.0, 60.0, 300.0),
                         (400.0, 120.0, 600.0)):
        c1 = breathing_case(0.2, 0.5, n_elem=n_elem, dt_frac=frac, ic_mode=1, n_cycles1=n1)
        c2_ = breathing_case(0.2, 0.5, n_elem=n_elem, dt_frac=frac, ic_mode=2, n_cycles2=n2)
        conv.append({"dt_frac_T2": frac, "n_cycles1": n1, "n_cycles2": n2,
                     "f1_td_fft": c1["f1_td_fft"], "f2_td_fft": c2_["f2_td_fft"],
                     "f1_healthy": c1["f1_healthy"], "f2_healthy": c1["f2_healthy"],
                     "d_f1_pct": 100 * (c1["f1_td_fft"] / c1["f1_healthy"] - 1),
                     "d_f2_pct": 100 * (c2_["f2_td_fft"] / c1["f2_healthy"] - 1),
                     "energy_dev_max": max(c1["energy_dev_max"], c2_["energy_dev_max"]),
                     "n_switch_ic1": c1["n_switch"], "n_switch_ic2": c2_["n_switch"]})
        print(f"[n3 breath] dt=T2/{frac:g} rec {n1:g}/{n2:g}: Δf1 {conv[-1]['d_f1_pct']:+.5f}% "
              f"Δf2 {conv[-1]['d_f2_pct']:+.6f}%", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(conv)


def breathing_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """개폐 결과(열림·시간영역, 예하중 없음)에 **그 Δf₁에 다시 맞춘** 매끄러운 비교장과 판별지표를 붙인다.

    개폐는 이동 패턴(D₂₁)을 거의 보존하지만 절대 이동을 약 절반으로 줄이므로, 잡음단위 대비
    (C1의 두 번째 조건)는 따로 확인해야 한다.
    """
    rows = []
    sel = df[df.model.isin(["open", "harmonic_mean", "td_unilateral"]) & (df.preload == "none")]
    for _, r in sel.iterrows():
        drops, s_ = matched_smooth(-float(r.d_f1_pct), centers=(float(r.xc_over_L),))
        rc = np.array([1 + r.d_f1_pct / 100, 1 + r.d_f2_pct / 100, np.nan])
        cc = contrast_cols(np.nan_to_num(rc, nan=1.0), drops)
        rows.append({"xc_over_L": r.xc_over_L, "a_bar": r.a_bar, "model": r.model,
                     "d_f1_pct": r.d_f1_pct, "d_f2_pct": r.d_f2_pct, "gauss_dmax": s_,
                     "D21_crack_pct": cc["D21_crack_pct"], "D21_smooth_pct": cc["D21_smooth_pct"],
                     "D21_ratio_crack_over_smooth": cc["D21_ratio_crack_over_smooth"],
                     "contrast_m2_c0.001": cc["contrast_m2_c0.001"],
                     "contrast_m2_c0.0003": cc["contrast_m2_c0.0003"],
                     "C1_nullfp": cc["C1_nullfp"]})
    return pd.DataFrame(rows)


def breathing_validation(n_elem: int = 240) -> pd.DataFrame:
    """질량 없는 힌지 모델 ↔ 정확 EB 전달행렬(Tada c_θ) + 닫힘 ↔ 해석 건전값 + 전환점 힘 연속."""
    rows = []
    fan = np.array(_FAN)
    for ab in (0.3, 0.5):
        c = cs.compliance(ab, BEAM.h, BEAM.b, BEAM.E, BEAM.nu, convention="tada")["c_MM"]
        m = MasslessHinge(c_theta=c, xc=0.2, n_elem=n_elem)
        fo = m.modes(+1, 3)[0]
        fc, Vc = m.modes(-1, 3)
        ex = cs.exact_eb_ratios(BEAM, c, xc_over_L=0.2, n_modes=3)
        w = Vc[:, 0] - (m.g_c @ Vc[:, 0]) / (m.g_c @ m.g_c) * m.g_c   # λ = 0인 처짐
        rows.append({"a_bar": ab, "n_elem": n_elem,
                     **{f"fe_ratio_open_f{k+1}": fo[k] / fc[k] for k in range(3)},
                     **{f"exact_ratio_f{k+1}": float(ex[k]) for k in range(3)},
                     **{f"closed_over_analytic_f{k+1}": fc[k] / fan[k] for k in range(3)},
                     "force_jump_rel_at_switch": float(np.linalg.norm(m.Ko @ w - m.Kc @ w)
                                                       / np.linalg.norm(m.Kc @ w))})
    return pd.DataFrame(rows)


# ====================================================================== (d) 뿌리
K_ROOT_ND = (np.inf, 300.0, 100.0, 30.0, 10.0)


def root_flex(k_nds=K_ROOT_ND, abars=(0.3, 0.5), n_elem: int = 1000) -> pd.DataFrame:
    """뿌리 회전유연도 k_root·L/EI 스윕 — 곡률-null 이동과 x_c/L = 0.2 지문의 변화.

    비교장은 **같은 FE·같은 뿌리조건** 안의 매끄러운 강성손실(`matched_smooth_fe`)이다.
    아울러 균열을 **이동한 null 위치**에 둔 경우를 함께 낸다(지문이 null을 따라가는가).
    """
    EI = BEAM.E * BEAM.I
    rows = []
    for knd in k_nds:
        kr = None if not np.isfinite(knd) else float(knd) * EI / BEAM.L
        fe_n = TimoFE(n_elem=1000, k_root=kr)
        nulls2 = fe_n.curvature_nulls(2)
        nulls3 = fe_n.curvature_nulls(3)
        x_null = float(nulls2[0]) if nulls2.size else np.nan
        fh = TimoFE(n_elem=n_elem, k_root=kr).frequencies(3)
        for ab in abars:
            C = crack_compliance(ab, "tada", True)
            for where, xc in (("x0.20", 0.20), ("at_null", round(x_null, 3))):
                r = TimoFE(n_elem=n_elem, cracks=[(xc, C)], k_root=kr).frequencies(3) / \
                    TimoFE(n_elem=n_elem, k_root=kr, extra_nodes=[xc]).frequencies(3)
                drops, s = matched_smooth_fe(100 * (1 - r[0]), centers=(xc,), k_root=kr)
                rows.append({"k_root_nd": knd, "mode2_null_xL": x_null,
                             "mode3_nulls_xL": json.dumps([round(float(v), 4) for v in nulls3]),
                             "f1_healthy_Hz": fh[0], "f2_healthy_Hz": fh[1],
                             "a_bar": ab, "crack_at": where, "xc_over_L": xc,
                             "gauss_dmax": s,
                             **{f"ratio_f{i+1}": float(r[i]) for i in range(3)},
                             **contrast_cols(r, drops)})
                print(f"[n3 root] K={knd} null2={x_null:.4f} ā={ab} {where}: D21 "
                      f"{rows[-1]['D21_crack_pct']:.2f}% vs {rows[-1]['D21_smooth_pct']:.2f}% "
                      f"C1={rows[-1]['C1_nullfp']}", flush=True)
    return pd.DataFrame(rows)


def smooth_fe_validation() -> dict:
    """FE 매끄러운 비교장(고정단) ↔ 정본 EB Ritz 비교장 — 같은 f₁ 강하에서 Δf₂·Δf₃."""
    target = 10.279361
    d_ritz, s_ritz = matched_smooth(target, centers=(0.2,))
    d_fe, s_fe = matched_smooth_fe(target, centers=(0.2,), k_root=None)
    return {"target_drop1_pct": target, "ritz_dmax": s_ritz, "fe_dmax": s_fe,
            **{f"ritz_drop_f{k+1}": float(d_ritz[k]) for k in range(3)},
            **{f"fe_drop_f{k+1}": float(d_fe[k]) for k in range(3)}}


# ====================================================================== (e) 깊이 역산
INV_ABARS = (0.10, 0.25, 0.30, 0.50, 0.60, 0.625)


def tm_ratios(a_bar: float, convention: str, xc: float = 0.2, n_elem: int = 300) -> np.ndarray:
    return crack_ratios([(xc, crack_compliance(a_bar, convention, True))], n_elem=n_elem)


def invert_depth(target: np.ndarray, convention: str, method: str, xc: float = 0.2,
                 c: float = 1e-3) -> tuple[float, float]:
    """ā 역산. method = 'f1'(Δf₁ 근찾기) | 'lsq'(f₁–f₃ 최소제곱, 같은 σ = √2c).

    반환 (ā̂, 잔차 χ = ‖r(ā̂) − r_t‖ / (√2c)).
    """
    from scipy.optimize import minimize_scalar
    target = np.asarray(target, dtype=float)
    lo, hi = 0.005, 0.95
    if method == "f1":
        g = lambda a: tm_ratios(a, convention, xc)[0] - target[0]
        if g(lo) * g(hi) > 0:
            return float("nan"), float("nan")
        a = float(brentq(g, lo, hi, xtol=1e-7))
    elif method == "lsq":
        obj = lambda a: float(np.sum((tm_ratios(a, convention, xc)[:3] - target[:3]) ** 2))
        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-7})
        a = float(res.x)
    else:
        raise ValueError(method)
    chi = float(np.linalg.norm(tm_ratios(a, convention, xc)[:3] - target[:3])
                / (math.sqrt(2.0) * c))
    return a, chi


def inversion(abars=INV_ABARS) -> pd.DataFrame:
    a11 = pd.read_csv(DATA / "a11_crack2d.csv")
    rows = []
    for ab in abars:
        gens = {"tada_tm": tm_ratios(ab, "tada"),
                "dimarogonas_tm": tm_ratios(ab, "dimarogonas")}
        s = a11[(a11.plane == "stress") & np.isclose(a11.a_bar, ab)]
        if len(s):
            s = s.iloc[0]
            gens["2d_stress_slit"] = np.array([s.ratio_f1, s.ratio_f2, s.ratio_f3])
        for gname, r in gens.items():
            for conv in ("tada", "dimarogonas"):
                for meth in ("f1", "lsq"):
                    a_hat, chi = invert_depth(r, conv, meth)
                    rows.append({"a_bar_true": ab, "generator": gname,
                                 "inverse_convention": conv, "method": meth,
                                 "a_bar_hat": a_hat, "a_err": a_hat - ab,
                                 "rel_depth_err_pct": 100 * (a_hat / ab - 1),
                                 "resid_noise_units_c0.001": chi,
                                 **{f"gen_ratio_f{k+1}": float(r[k]) for k in range(3)}})
                    print(f"[n3 inv] ā={ab} gen={gname} inv={conv}/{meth}: "
                          f"ā̂={a_hat:.4f} ({100*(a_hat/ab-1):+.1f}%) χ={chi:.2f}", flush=True)
    return pd.DataFrame(rows)


def comparator_width(widths=(0.06, 0.12, 0.24), a_bar: float = 0.5,
                     xcs=SWEEP_XC, n_elem: int = N_ELEM_SWEEP) -> pd.DataFrame:
    """비교장 폭 민감도 — C1·C2 판정이 경쟁장 폭(정본 0.12)에 얼마나 의존하는가."""
    rows = []
    f0 = TimoFE(n_elem=n_elem).frequencies(3)
    C = crack_compliance(a_bar, "tada", True)
    for xc in xcs:
        r = TimoFE(n_elem=n_elem, cracks=[(xc, C)]).frequencies(3) / f0
        for w in widths:
            drops, s = matched_smooth(100 * (1 - r[0]), centers=(xc,), width=w)
            rows.append({"xc_over_L": xc, "a_bar": a_bar, "gauss_width_xt": w,
                         "gauss_dmax": s, **{f"ratio_f{i+1}": float(r[i]) for i in range(3)},
                         **contrast_cols(r, drops)})
    return pd.DataFrame(rows)


# ====================================================================== (d-ii) 2D 필렛
def _fillet_geo(path: Path, L: float, h: float, a_bar: float, xc_over_L: float,
                r_f: float, t_base: float, side: float, with_base: bool,
                lc: float, lc_tip: float, lc_fillet: float) -> None:
    """gmsh .geo — 베인 스트립 + (선택) 받침 블록 + 오목 필렛 2개 + 폭 0 균열선(매립).

    좌표 [m]. 받침 블록은 x ∈ [−t_base, 0], y ∈ [−H/2, H/2] (H = h + 2·side)이고
    고정조건은 블록 뒷면 x = −t_base에 건다. 필렛은 (0, ±h/2) 오목 모서리의 반경 r_f 원호.
    균열선은 x_c에서 아래면부터 ā·h까지 — 표면을 **나누도록** BooleanFragments로 매립한다.
    """
    xc = xc_over_L * L
    yt = -0.5 * h + a_bar * h
    H = h + 2 * side
    g = ['SetFactory("OpenCASCADE");', "Geometry.Tolerance = 1e-12;",
         "Geometry.ToleranceBoolean = 1e-12;",
         f"Rectangle(1) = {{0, {-h/2!r}, 0, {L!r}, {h!r}}};"]
    surfs = [1]
    if with_base:
        g.append(f"Rectangle(2) = {{{-t_base!r}, {-H/2!r}, 0, {t_base!r}, {H!r}}};")
        surfs.append(2)
        if r_f > 0:
            g += [f"Rectangle(3) = {{0, {h/2!r}, 0, {r_f!r}, {r_f!r}}};",
                  f"Disk(4) = {{{r_f!r}, {h/2 + r_f!r}, 0, {r_f!r}, {r_f!r}}};",
                  "BooleanDifference(5) = { Surface{3}; Delete; }{ Surface{4}; Delete; };",
                  f"Rectangle(6) = {{0, {-h/2 - r_f!r}, 0, {r_f!r}, {r_f!r}}};",
                  f"Disk(7) = {{{r_f!r}, {-h/2 - r_f!r}, 0, {r_f!r}, {r_f!r}}};",
                  "BooleanDifference(8) = { Surface{6}; Delete; }{ Surface{7}; Delete; };"]
            surfs += [5, 8]
    g += [f"Point(1001) = {{{xc!r}, {-h/2!r}, 0}};",
          f"Point(1002) = {{{xc!r}, {yt!r}, 0}};",
          "Line(1001) = {1001, 1002};",
          ("s() = BooleanUnion{ Surface{%d}; Delete; }{ Surface{%s}; Delete; };"
           % (surfs[0], ",".join(map(str, surfs[1:])))) if len(surfs) > 1 else "s() = {1};",
          "f() = BooleanFragments{ Surface{s()}; Delete; }{ Curve{1001}; Delete; };",
          "Physical Surface(1) = {f()};",
          # 균열선과 입구점(아래면 위의 점)을 좌표로 다시 찾는다(불리언 후 번호가 바뀐다)
          f"cr() = Curve In BoundingBox{{{xc - 1e-6!r}, {-h/2 - 1e-6!r}, -1, "
          f"{xc + 1e-6!r}, {yt + 1e-6!r}, 1}};",
          "Physical Curve(101) = {cr()};",
          f"mo() = Point In BoundingBox{{{xc - 1e-6!r}, {-h/2 - 1e-6!r}, -1, "
          f"{xc + 1e-6!r}, {-h/2 + 1e-6!r}, 1}};",
          "Physical Point(102) = {mo()};",
          f"tp() = Point In BoundingBox{{{xc - 1e-6!r}, {yt - 1e-6!r}, -1, "
          f"{xc + 1e-6!r}, {yt + 1e-6!r}, 1}};",
          # 크기장: 균열선단 근방 + 필렛 근방 세밀
          "Field[1] = Distance; Field[1].PointsList = {tp()};",
          f"Field[2] = Threshold; Field[2].InField = 1; Field[2].SizeMin = {lc_tip!r};",
          f"Field[2].SizeMax = {lc!r}; Field[2].DistMin = {0.05 * h!r}; "
          f"Field[2].DistMax = {1.5 * h!r};",
          f"Field[3] = Box; Field[3].VIn = {lc_fillet!r}; Field[3].VOut = {lc!r};",
          f"Field[3].XMin = {-t_base!r}; Field[3].XMax = {max(r_f, 0.0) + 0.5 * h!r};",
          f"Field[3].YMin = {-H/2!r}; Field[3].YMax = {H/2!r};",
          "Field[4] = Min; Field[4].FieldsList = {2, 3};",
          "Background Field = 4;",
          "Mesh.MeshSizeExtendFromBoundary = 0; Mesh.MeshSizeFromPoints = 0;",
          "Mesh.MeshSizeFromCurvature = 0;",
          "Mesh.Algorithm = 6;", "Mesh.ElementOrder = 1;"]
    path.write_text("\n".join(g) + "\n")


def _gmsh_mesh(geo: Path, out: Path, crack: bool) -> None:
    cmd = ["gmsh", str(geo), "-2", "-format", "msh22", "-o", str(out)]
    if crack:
        # Crack 플러그인: 물리곡선 101의 절점을 이중화(입구점 102도 연다) — 폭 0 슬릿
        post = out.with_suffix(".crack.geo")
        post.write_text(f'Merge "{out}";\nPlugin(Crack).Dimension = 1;\n'
                        "Plugin(Crack).PhysicalGroup = 101;\n"
                        "Plugin(Crack).OpenBoundaryPhysicalGroup = 102;\n"
                        "Plugin(Crack).Run;\n"
                        f'Save "{out}";\n')
        subprocess.run(cmd, check=True, capture_output=True)
        subprocess.run(["gmsh", str(post), "-0", "-format", "msh22"], check=True,
                       capture_output=True)
    else:
        subprocess.run(cmd, check=True, capture_output=True)


def _read_tri(path: Path):
    import meshio
    from .. import rail3d as r3
    m = meshio.read(str(path))
    tri = np.concatenate([c.data for c in m.cells if c.type == "triangle"])
    co = m.points[:, :2].astype(float)
    co, tri = r3._compact(co, tri.astype(np.int64))
    return co, tri


def _flap_2d(co, tri, x_clamp: float, n_modes: int = 8):
    from .. import rail3d as r3
    res = r3.solve_modes(co, tri, VANE.E, VANE.nu, VANE.rho, r3.clamp_root(x_clamp),
                         n_modes=n_modes, order=2, keep_shapes=True, plane="stress")
    kinds = r3.beam_mode_kinds(res)
    flap = [i for i, k in enumerate(kinds) if k == "flap"]
    return res, flap


def _null_from_2d(res, flap_idx: int, h: float, L: float) -> float:
    """2D 모드의 단면회전 θ(x) ≈ (u_x(아래면) − u_x(윗면))/h → 곡률 dθ/dx의 영점(0.1L–0.4L)."""
    c = res.field_coors
    u = res.full_shapes[flap_idx].reshape(-1, 2)
    tol = 1e-9
    top = np.nonzero((np.abs(c[:, 1] - h / 2) < tol) & (c[:, 0] >= 0) & (c[:, 0] <= L))[0]
    bot = np.nonzero((np.abs(c[:, 1] + h / 2) < tol) & (c[:, 0] >= 0) & (c[:, 0] <= L))[0]
    xs = np.linspace(0.05 * L, 0.5 * L, 2001)

    def interp(idx):
        o = np.argsort(c[idx, 0])
        xx, uu = c[idx, 0][o], u[idx, 0][o]
        xx, k = np.unique(xx, return_index=True)
        return np.interp(xs, xx, uu[k])
    th = (interp(bot) - interp(top)) / h
    kap = np.gradient(th, xs)
    # 균열(아래면 x_c)의 불연속을 피해 x_c 근방 ±0.25h 구간은 잡음이 크다 — 부호교차만 본다
    sgn = np.sign(kap)
    idx = np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]
    if idx.size == 0:
        return float("nan")
    i = idx[0]
    return float((xs[i] - kap[i] * (xs[i + 1] - xs[i]) / (kap[i + 1] - kap[i])) / L)


def fillet(r_fs=(0.0, 0.5, 1.0), a_bar: float = 0.5, xc_over_L: float = 0.2,
           t_base: float = 1.0e-3, side: float = 3.0e-3, lc: float = 0.10e-3,
           lc_tip: float = 0.01e-3, lc_fillet: float = 0.04e-3,
           workdir: Path | None = None) -> pd.DataFrame:
    """2D 평면응력 뿌리 필렛 스윕 — 이상 고정단(받침 없음) + 받침 블록(r_f/h = 0, 0.5, 1.0).

    고정은 받침 뒷면(x = −t_base), 유효 스팬·균열 위치(x_c = 0.2 L from x = 0)는 고정이다.
    건전/균열 메시는 같은 .geo에서 Crack 플러그인 유무만 바꿔 만든다(같은 절점배치).
    비교장은 (i) 정본 고정단 EB Ritz 비교장, (ii) 2D 건전 f₁에 맞춘 등가 뿌리스프링 FE 비교장.
    """
    import tempfile
    workdir = Path(workdir or tempfile.mkdtemp(prefix="n3_fillet_"))
    h, L = VANE.h, VANE.L
    EI = BEAM.E * BEAM.I
    rows = []
    cases = [("ideal_clamp", None)] + [(f"base_rf{rf/h:g}h", rf) for rf in
                                         [r * h for r in r_fs]]
    for name, rf in cases:
        with_base = rf is not None
        geo_p = workdir / f"{name}.geo"
        _fillet_geo(geo_p, L, h, a_bar, xc_over_L, rf or 0.0, t_base, side, with_base,
                    lc, lc_tip, lc_fillet)
        out0 = workdir / f"{name}_healthy.msh"
        out1 = workdir / f"{name}_crack.msh"
        _gmsh_mesh(geo_p, out0, crack=False)
        _gmsh_mesh(geo_p, out1, crack=True)
        co0, tr0 = _read_tri(out0)
        co1, tr1 = _read_tri(out1)
        xcl = -t_base if with_base else 0.0
        res0, fl0 = _flap_2d(co0, tr0, xcl)
        res1, fl1 = _flap_2d(co1, tr1, xcl)
        f0 = res0.freqs[fl0][:3]
        f1 = res1.freqs[fl1][:3]
        r = f1 / f0
        x_null = _null_from_2d(res0, fl0[1], h, L)
        # (i) 정본 고정단 비교장
        d_i, s_i = matched_smooth(100 * (1 - r[0]), centers=(xc_over_L,))
        # (ii) 2D 건전 f₁에 맞춘 등가 뿌리스프링(EI·K/L) FE 비교장
        f_clamp = TimoFE(n_elem=600).frequencies(1)[0]
        if with_base and f0[0] < f_clamp:
            g = lambda lk: TimoFE(n_elem=600, k_root=10 ** lk * EI / L).frequencies(1)[0] - f0[0]
            lk = brentq(g, -1.0, 8.0, xtol=1e-6)
            k_eq = 10 ** lk
            d_ii, s_ii = matched_smooth_fe(100 * (1 - r[0]), centers=(xc_over_L,),
                                           k_root=k_eq * EI / L)
        else:
            k_eq = np.inf
            d_ii, s_ii = matched_smooth_fe(100 * (1 - r[0]), centers=(xc_over_L,))
        n_dup = int(co1.shape[0] - co0.shape[0])
        for tag, dr, s in (("clamped_EB_ritz", d_i, s_i), ("equiv_root_spring_FE", d_ii, s_ii)):
            rows.append({"case": name, "r_f_over_h": (rf / h if rf is not None else np.nan),
                         "with_base": with_base, "t_base_mm": 1e3 * t_base if with_base else 0,
                         "side_mm": 1e3 * side if with_base else 0,
                         "a_bar": a_bar, "xc_over_L": xc_over_L,
                         "ndof_crack": res1.ndof, "n_dup_nodes": n_dup,
                         "f1_healthy_Hz": f0[0], "f2_healthy_Hz": f0[1], "f3_healthy_Hz": f0[2],
                         "mode2_null_xL_2d": x_null, "k_root_nd_equiv": k_eq,
                         "comparator": tag, "gauss_dmax": s,
                         **{f"ratio_f{i+1}": float(r[i]) for i in range(3)},
                         **contrast_cols(r, dr)})
        print(f"[n3 fillet] {name}: f0={np.round(f0,1)} r={np.round(r,6)} null2={x_null:.4f} "
              f"K_eq={k_eq:.3g} D21 {rows[-1]['D21_crack_pct']:.3f}% vs "
              f"{rows[-2]['D21_smooth_pct']:.2f}/{rows[-1]['D21_smooth_pct']:.2f}% dup={n_dup}",
              flush=True)
    return pd.DataFrame(rows)


def figure() -> Path:
    """(a) 위치 스윕 D₂₁(균열 대 비교장, 세 깊이) + 2D 스폿 + C1 창, (b) 다중균열 D₂₁."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sw = pd.read_csv(out_csv("n3_location_sweep"))
    s2 = pd.read_csv(out_csv("n3_location_2d"))
    mc = pd.read_csv(out_csv("n3_multicrack"))
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ab, mk in ((0.3, "^"), (0.5, "o"), (0.6, "s")):
        g = sw[np.isclose(sw.a_bar, ab)].sort_values("xc_over_L")
        ln, = ax[0].plot(g.xc_over_L, g.D21_crack_pct, mk + "-", ms=4,
                         label=rf"crack, $\bar a$ = {ab}")
        ax[0].plot(g.xc_over_L, g.D21_smooth_pct, mk + "--", ms=3, color=ln.get_color(),
                   alpha=0.6, label=rf"matched smooth field, $\bar a$ = {ab}")
    ax[0].plot(s2.xc_over_L, s2.D21_crack_pct, "kx", ms=8, mew=2,
               label=r"2D zero-width slit, $\bar a$ = 0.5")
    ok = sw[sw.C1_nullfp].xc_over_L
    ax[0].axvspan(ok.min(), ok.max(), color="0.85", zorder=0)
    ax[0].axvline(0.2162, color="k", lw=0.8, ls=":")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"Crack position $x_c/L$")
    ax[0].set_ylabel(r"$|\Delta f_2/f_2|\,/\,|\Delta f_1/f_1|$ [%]")
    ax[0].set_title("(a)")
    ax[0].legend(fontsize=6.5, ncol=2, loc="lower right")
    ax[0].grid(alpha=0.3, which="both")
    m1 = mc[mc.comparator == "gauss_per_crack"].reset_index(drop=True)
    x = np.arange(len(m1))
    ax[1].bar(x - 0.2, m1.D21_crack_pct, 0.4, label="cracks")
    ax[1].bar(x + 0.2, m1.D21_smooth_pct, 0.4, label="matched smooth field (per crack)")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(m1.case, rotation=60, ha="right", fontsize=6.5)
    ax[1].set_ylabel(r"$|\Delta f_2/f_2|\,/\,|\Delta f_1/f_1|$ [%]")
    ax[1].set_title("(b)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both", axis="y")
    fig.tight_layout()
    path = out_fig("n3_crack_robustness")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"[saved] {path}", flush=True)
    return path


# ====================================================================== CLI
def _save(df: pd.DataFrame, name: str, meta: dict | None = None) -> Path:
    p = out_csv(name)
    df = df.copy()
    df["git_head"] = _git_head()
    if meta:
        for k, v in meta.items():
            df[k] = v
    df.to_csv(p, index=False)
    print(f"[saved] {p} ({len(df)} rows)", flush=True)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m impeller_fingerprint.rev1.n3_crack")
    ap.add_argument("item", choices=["reproduce", "sweep", "width", "sweep2d", "multi",
                                     "breathing", "breathing_contrast", "root", "fillet",
                                     "invert", "figure", "all"])
    a = ap.parse_args(argv)
    t0 = time.time()
    todo = (["reproduce", "sweep", "width", "sweep2d", "multi", "breathing", "root",
             "fillet", "invert"] if a.item == "all" else [a.item])
    for it in todo:
        if it == "reproduce":
            df = reproduce()
            bad = df[df.check.str.startswith("timoshenko")].max_rel_diff.max()
            if bad > 1e-6:
                raise SystemExit(f"정본 재현 실패: Timoshenko 상대차 {bad:.2e}")
            s = df[df.check.str.startswith("2d")].iloc[0]
            # 정본 기준값(a11 ā = 0.5 평면응력) — 깃 두께에 따라: h = 1.0 mm(G3 정본·08-13) 2.20397·21.4408,
            # h = 1.2 mm(G1, r4.1) 2.549081·21.236563. 2D 평면응력이라 쿠폰 폭과는 무관하다.
            ref = _SMOOTH_REF.get(round(VANE.h * 1e4))
            if ref is None:
                raise SystemExit(f"정본 재현 기준값 없음: 깃 두께 {VANE.h * 1e3:.2f} mm")
            if abs(s.gauss_drop_f2_pct - ref[0]) > 1e-3 or abs(s.D21_smooth_pct - ref[1]) > 1e-2:
                raise SystemExit("정본 재현 실패: 매끄러운 비교장")
            nul = null_positions()
            _save(df, "n3_reproduce", {"mode2_null_fe": nul["fe_mode2"][0],
                                       "mode2_null_eb": nul["eb_mode2"][0],
                                       "mode3_nulls_eb": json.dumps(nul["eb_mode3"])})
        elif it == "sweep":
            _save(sweep(), "n3_location_sweep",
                  {"criterion_C1": "D21c<=0.25*D21s & contrast_m2(c=0.1%)>=3",
                   "criterion_C2": "chi2_23>=9.21"})
        elif it == "width":
            _save(comparator_width(), "n3_comparator_width")
        elif it == "sweep2d":
            _save(sweep2d(), "n3_location_2d")
        elif it == "multi":
            _save(multi(), "n3_multicrack")
        elif it == "breathing":
            val = breathing_validation()
            _save(val, "n3_breathing_validation")
            df, conv = breathing()
            _save(df, "n3_breathing",
                  {"model_note": "massless unilateral hinge, EB Mode-I (Tada), lumped mass, "
                                 "event-driven exact modal solution; TD freq = FFT peak"})
            _save(conv, "n3_breathing_dt_convergence")
            _save(breathing_contrast(df), "n3_breathing_contrast")
        elif it == "breathing_contrast":
            _save(breathing_contrast(pd.read_csv(out_csv("n3_breathing"))), "n3_breathing_contrast")
        elif it == "figure":
            figure()
        elif it == "root":
            _save(pd.DataFrame([smooth_fe_validation()]), "n3_smooth_fe_validation")
            _save(root_flex(), "n3_root_flexibility")
        elif it == "fillet":
            _save(fillet(), "n3_root_fillet_2d")
        elif it == "invert":
            _save(inversion(), "n3_depth_inversion")
        print(f"[n3] {it} done at {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
