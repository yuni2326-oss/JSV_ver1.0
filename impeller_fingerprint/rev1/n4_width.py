"""N4 — 손상 폭·발자국 불일치와 폭 조건부 식별성 (R1-P5, R1-P1·R2-Q1 보조).

질문: 생산 역맵은 **폭 w = 3 mm로 고정된** 절단 가우시안 강성손실장을 가정한다(정본 §3.3, A.3).
실제 손상의 폭·모양이 다르면 (ξ_d, S̄_D) 추정이 얼마나 편향되고, 그 편향을 잔차가 드러내는가?
폭을 미지수로 풀면 무엇을 잃는가?

분리 원칙(받은 계획서 §6 N4): **같은 Kirchhoff 레일 안에서** 손상장 모양만 바꾼다 — 3D 모델형식
불일치(N5)와 섞지 않는다. 진실과 정확 역맵은 같은 비섭동 Ritz 솔버를 쓰므로, 진실 폭이 3 mm일 때
(같은 모양 대조군) 정확 역맵은 진실을 그대로 복원해야 한다.

솔버 선택(수렴 근거는 보고서 §2): 생산 `forward.eta_bar_exact`(단항식 8개)와 정본 르장드르 경로
(`kernels._basis(basis="legendre")`, 단항식 계수로 바꿔 평가)는 좁거나 깊은 손상·n = 1 모드에서
미수렴이다. N4는 **반경 Hermite 3차 FEM**(`FemRail`, 요소 100개, 밴드는 경계 절점)을 쓴다 —
N = 480 대비 ≤ 0.0015 floor. 옛 `RitzRail`은 정본 솔버와의 동치 테스트용으로만 남긴다.

관측량은 생산과 같다: 고유값 기반 η̄ = Λ_d/Λ_h − 1, m = 0은 단일모드(σ = 2√2c), m > 0은 축퇴쌍
평균(σ = 2c) — `noise.sigma_y_for_modes`.

손상장(모두 **강성손실 d_K(r)** 를 정의하고 질량은 균일판 정확결합 d_M = 1 − (1 − d_K)^{1/3}):
  gauss  d_K = (S/(w√π))·exp(−((r − r_d)/w)²),  S = S̄_D (b − a),  w = **1/e 반폭**(HWHM = w√ln2)
  band   d_K = S̄_D (b − a)/W  (r_d − W/2 ≤ r ≤ r_d + W/2, [a, b]로 절단),  W = 전폭
절단 후 재정규화하지 않는다(정본 A.3과 같은 규약) — 명목 S̄_D와 영역내 적분손실 S̄_D·Q가 다르다.
물리 상한: peak d_K < 1. 가우시안 S̄_D < w√π/(b − a)(w = 1.5 mm → 12.57 %), 밴드 S̄_D < W/(b − a).

실행:  PAPER3_OUT=… python -m impeller_fingerprint.rev1.n4_width all --workers 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import least_squares

from .. import forward as fwd
from .. import geometry as geo
from .. import identifiability as idf
from .. import kernels as ker
from .. import modeselect as msel
from .. import noise as noi
from .. import severity as sev
from . import out_csv, out_fig

PLATE = geo.DISK
EXT = PLATE.extent                       # b − a = 21.16 mm
W_REF = 0.003                            # 생산 고정 폭
C_REF = 1e-3                             # σ_f/f = 0.1 %
MODES_DEF = ((0, 0), (1, 0), (2, 0), (3, 0))
#: 25.6 kHz 후보풀(Table 3a "all eight available modes")
MODES_EXT = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1))
NT, NG, NG_BAND = 36, 4001, 8001
W_BOUNDS = (0.0010, 0.0100)              # 폭 자유 역산의 탐색 구간 [m]
W_GRID = np.round(np.arange(1.0, 10.0001, 0.25), 4) * 1e-3
XI_GRID = np.round(np.linspace(0.05, 0.95, 20), 6)
S_TARGETS = (0.01, 0.05, 0.15)
SEED0 = 20261006
_PEAK_MAX = 1.0 - 1e-9
#: 유한차분 상대간격 — 르장드르 Gram 행렬(cond ~1e8)의 반올림이 η에 ~1e-7을 남기므로
#: scipy 기본(√ε ≈ 1.5e-8)은 야코비안을 잡음으로 만든다(최적화가 시작점에 멈춘다).
_DIFF_STEP = 1e-3


# ============================================================ 손상장(프로파일)
@dataclass(frozen=True)
class Profile:
    """손상 발자국. kind = "gauss"(w = 1/e 반폭) | "band"(w = 전폭)."""

    kind: str
    w: float

    @property
    def label(self) -> str:
        return (f"gauss_w{self.w * 1e3:g}mm" if self.kind == "gauss"
                else f"band_W{self.w * 1e3:g}mm")

    @property
    def hwhm_mm(self) -> float:
        """반치반폭 — 가우시안은 w√ln2, 밴드는 W/2."""
        return (self.w * math.sqrt(math.log(2.0)) if self.kind == "gauss"
                else 0.5 * self.w) * 1e3

    def peak(self, s_bar: float) -> float:
        """국부 강성손실의 최댓값 peak d_K (절단 전 진폭)."""
        if self.kind == "gauss":
            return float(s_bar) * EXT / (self.w * math.sqrt(math.pi))
        return float(s_bar) * EXT / self.w

    def ceiling(self) -> float:
        """peak d_K = 1이 되는 명목 S̄_D — 이 이상은 순방향이 정의되지 않는다."""
        return (self.w * math.sqrt(math.pi) / EXT if self.kind == "gauss"
                else self.w / EXT)

    def s_from_peak(self, peak: float) -> float:
        return float(peak) * self.ceiling()

    def bounds_r(self, xi: float):
        r_c = sev.xi_to_r(xi, PLATE.a, PLATE.b)
        return max(PLATE.a, r_c - 0.5 * self.w), min(PLATE.b, r_c + 0.5 * self.w)

    def d_K(self, r: np.ndarray, xi: float, s_bar: float) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        if self.kind == "gauss":
            return sev.damage_field_xi(r, xi, s_bar, self.w, PLATE.a, PLATE.b)
        r1, r2 = self.bounds_r(xi)
        return np.where((r >= r1) & (r <= r2), self.peak(s_bar), 0.0)

    def in_domain(self, xi: float, s_bar: float) -> float:
        """영역내 적분 강성손실 (1/(b−a))∫_a^b d_K dr — 명목 S̄_D·Q."""
        if self.kind == "gauss":
            return float(s_bar) * q_factor(xi, self.w)
        r1, r2 = self.bounds_r(xi)
        return self.peak(s_bar) * (r2 - r1) / EXT


def q_factor(xi: float, w: float) -> float:
    """절단분율 Q(ξ_d) = ½[erf((b − r_d)/w) + erf((r_d − a)/w)] (정본 Eq. 8)."""
    r_d = sev.xi_to_r(xi, PLATE.a, PLATE.b)
    return 0.5 * (math.erf((PLATE.b - r_d) / w) + math.erf((r_d - PLATE.a) / w))


GAUSS_REF = Profile("gauss", W_REF)
TRUTHS = (Profile("gauss", 0.0015), Profile("gauss", 0.003), Profile("gauss", 0.006),
          Profile("gauss", 0.009), Profile("band", 0.006))


def matched_s_true(prof: Profile, xi: float, s_target: float, matching: str) -> float:
    """진실 손상의 명목 S̄_D — 세 가지 대응 규약(받은 계획서 §6 N4).

    nominal  : 명목 S̄_D가 같다(주 분석)
    in_domain: 영역내 적분손실이 기준 가우시안(3 mm, 같은 명목 S̄)과 같다
    peak     : peak d_K가 기준 가우시안과 같다
    """
    if matching == "nominal":
        return float(s_target)
    if matching == "in_domain":
        return GAUSS_REF.in_domain(xi, s_target) / (prof.in_domain(xi, 1.0))
    if matching == "peak":
        return prof.s_from_peak(GAUSS_REF.peak(s_target))
    raise ValueError(matching)


# ============================================================ 비섭동 Ritz 레일
class _RitzM:
    """방위차수 m 하나의 Rayleigh–Ritz — 적분 행렬을 미리 만든다.

    K_ij(d_K) = ∫ (1 − d_K) q_ij r dr,  M_ij(d_M) = ∫ (1 − d_M) ψ_i ψ_j r dr  (D·ρh 인자 제외)
    q_ij는 `kernels._matrices`와 같은 굽힘에너지 이선형형식이다. 사다리꼴 가중치까지 미리
    곱해 두면 행렬 조립이 (n_pair × n_grid) 행렬–벡터 곱 하나가 된다.
    """

    def __init__(self, m: int, n_trial: int, n_grid: int, basis: str, nu: float):
        r, psi, dps, d2p = ker._basis(PLATE.a, PLATE.b, n_trial, n_grid, basis=basis)
        krr, kth, krt = ker._curvatures(m, r, psi, dps, d2p)
        A = krr + kth
        iu = np.triu_indices(n_trial)
        i, j = iu
        tw = np.full(r.size, r[1] - r[0])
        tw[0] *= 0.5
        tw[-1] *= 0.5
        wr = tw * r
        qk = (A[i] * A[j] - (1 - nu) * (krr[i] * kth[j] + kth[i] * krr[j])
              + 2 * (1 - nu) * krt[i] * krt[j])
        qm = psi[i] * psi[j]
        self.r, self.iu, self.n = r, iu, n_trial
        self.QK = qk * wr[None, :]
        self.QM = qm * wr[None, :]
        Kh, Mh = self._mats(np.ones(r.size), np.ones(r.size))
        self.scale = 1.0 / np.sqrt(np.diag(Mh))         # 대각 전처리(고유값 불변)
        self.lam_h = self._eig(Kh, Mh)

    def _mats(self, wk, wm):
        n = self.n
        K = np.zeros((n, n))
        M = np.zeros((n, n))
        K[self.iu] = self.QK @ wk
        M[self.iu] = self.QM @ wm
        K = K + np.triu(K, 1).T
        M = M + np.triu(M, 1).T
        return K, M

    def _eig(self, K, M, n_modes: int = 3):
        s = getattr(self, "scale", None)
        if s is None:
            s = 1.0 / np.sqrt(np.diag(M))
        K = K * s[:, None] * s[None, :]
        M = M * s[:, None] * s[None, :]
        return eigh(K, M, eigvals_only=True, subset_by_index=[0, n_modes - 1])

    def lam(self, d_K: np.ndarray, d_M: np.ndarray, n_modes: int = 3):
        K, M = self._mats(1.0 - d_K, 1.0 - d_M)
        return self._eig(K, M, n_modes)


_RM: dict = {}


def _ritz_m(m: int, n_trial: int, n_grid: int, basis: str) -> _RitzM:
    """방위차수별 적분행렬 캐시 — 모드 부분집합이 달라도 같은 m은 한 번만 만든다."""
    key = (m, n_trial, n_grid, basis)
    if key not in _RM:
        _RM[key] = _RitzM(m, n_trial, n_grid, basis, PLATE.nu)
    return _RM[key]


class RitzRail:
    """모드 목록 전체의 비섭동 재해 η̄ = Λ_d/Λ_h − 1 (균일판 정확결합)."""

    def __init__(self, modes=MODES_DEF, n_trial: int = NT, n_grid: int = NG,
                 basis: str = "legendre"):
        self.modes = tuple(tuple(x) for x in modes)
        self.n_trial, self.n_grid, self.basis = n_trial, n_grid, basis
        self._m = {m: _ritz_m(m, n_trial, n_grid, basis)
                   for m in sorted({m for m, _ in self.modes})}
        self.r = next(iter(self._m.values())).r

    def eta_fields(self, d_K: np.ndarray) -> np.ndarray:
        d_K = np.asarray(d_K, dtype=float)
        if d_K.max(initial=0.0) >= 1.0 or d_K.min(initial=0.0) < 0.0:
            raise ValueError(f"d_K 범위 밖(max {d_K.max():.4f}) — 순방향 미정의 영역 평가 시도")
        d_M = 1.0 - (1.0 - d_K) ** (1.0 / 3.0)
        lam_d = {m: rm.lam(d_K, d_M) for m, rm in self._m.items()}
        return np.array([lam_d[m][n] / self._m[m].lam_h[n] - 1.0 for m, n in self.modes])

    def eta(self, prof: Profile, xi: float, s_bar: float) -> np.ndarray:
        if s_bar <= 0.0:
            return np.zeros(len(self.modes))
        if prof.peak(s_bar) >= 1.0:
            raise ValueError(f"{prof.label}: S̄={s_bar:.4f} ≥ 상한 {prof.ceiling():.4f}")
        return self.eta_fields(prof.d_K(self.r, xi, s_bar))


# ============================================================ 반경 Hermite-3차 FEM 레일(생산 솔버)
class _FemM:
    """방위차수 m의 반경 방향 Hermite 3차 유한요소 — 내경 클램프·외경 자유 환형 Kirchhoff 판.

    **왜 Ritz를 버렸는가**(보고서 §2): 전역 다항식 Ritz는 국소 손상에서 수렴이 느리고, 정본
    `kernels._basis(basis="legendre")`는 르장드르를 단항식 계수로 바꿔 평가해 고차에서 자릿수를
    잃는다(n = 1 모드가 S̄ = 15 %에서 4–12 floor, 폭 1.5 mm·S̄ = 12 %에서 n = 0이 최대 26 floor).
    국소 기저(요소)는 좁은 가우시안과 밴드 계단(절점을 밴드 경계에 둔다)을 모두 정확히 따라간다.
    회전 자유도의 질량이 h³ 규모라 고유해 전에 질량 대각으로 Jacobi 스케일한다(고유값 불변).
    고유값은 고유벡터로 **요소 곡률 에너지의 Rayleigh 몫**을 다시 계산해 얻는다 — 조립 행렬의
    최소 고유값은 상대오차가 ε·λ_max/λ_min(∝N⁴)이라 N = 160에서 1e-5까지 흔들렸다.
    수렴: N = 100 대 480이 가우시안(w = 1–3 mm, peak < 0.99) 전 격자에서 ≤ 0.0015 floor,
    밴드(경계 절점)에서 ~1e-5 floor.
    """

    def __init__(self, m: int, r_nodes, n_gauss: int = 5, nu: float = PLATE.nu):
        import scipy.sparse as _sp
        r_nodes = np.asarray(r_nodes, dtype=float)
        E = r_nodes.size - 1
        xg, wg = np.polynomial.legendre.leggauss(n_gauss)
        s = 0.5 * (xg + 1.0)
        ws = 0.5 * wg
        h = np.diff(r_nodes)
        rq = r_nodes[:-1, None] + h[:, None] * s[None, :]
        S = np.broadcast_to(s[None, :], (E, n_gauss))
        hh = h[:, None]
        H = np.stack([1 - 3 * S ** 2 + 2 * S ** 3, hh * (S - 2 * S ** 2 + S ** 3),
                      3 * S ** 2 - 2 * S ** 3, hh * (-S ** 2 + S ** 3)], -1)
        dH = np.stack([(-6 * S + 6 * S ** 2) / hh, 1 - 4 * S + 3 * S ** 2,
                       (6 * S - 6 * S ** 2) / hh, -2 * S + 3 * S ** 2], -1)
        d2H = np.stack([(-6 + 12 * S) / hh ** 2, (-4 + 6 * S) / hh,
                        (6 - 12 * S) / hh ** 2, (-2 + 6 * S) / hh], -1)
        r3 = rq[..., None]
        krr, kth, krt = d2H, dH / r3 - m ** 2 * H / r3 ** 2, m * (dH / r3 - H / r3 ** 2)
        A = krr + kth
        wq = ws[None, :] * hh * rq
        kq = (A[..., :, None] * A[..., None, :]
              - (1 - nu) * (krr[..., :, None] * kth[..., None, :]
                            + kth[..., :, None] * krr[..., None, :])
              + 2 * (1 - nu) * krt[..., :, None] * krt[..., None, :]) * wq[..., None, None]
        mq = H[..., :, None] * H[..., None, :] * wq[..., None, None]
        ndof = 2 * (E + 1)
        free = np.arange(2, ndof)                      # R(a) = R'(a) = 0
        nf = free.size
        pos = -np.ones(ndof, dtype=np.int64)
        pos[free] = np.arange(nf)
        dofs = 2 * np.arange(E)[:, None] + np.arange(4)[None, :]
        I = np.broadcast_to(dofs[:, None, :, None], (E, n_gauss, 4, 4))
        J = np.broadcast_to(dofs[:, None, None, :], (E, n_gauss, 4, 4))
        q = np.broadcast_to(np.arange(E * n_gauss).reshape(E, n_gauss)[:, :, None, None],
                            (E, n_gauss, 4, 4))
        pi, pj = pos[I], pos[J]
        keep = (pi >= 0) & (pj >= 0)
        rows = (pi * nf + pj)[keep]
        cols = q[keep]
        self.AK = _sp.csr_matrix((kq[keep], (rows, cols)), shape=(nf * nf, E * n_gauss))
        self.AM = _sp.csr_matrix((mq[keep], (rows, cols)), shape=(nf * nf, E * n_gauss))
        self.nf, self.rq = nf, rq.ravel()
        # 요소별 곡률 에너지로 Rayleigh 몫을 다시 계산하기 위한 형상함수(점별)
        self._H, self._dH, self._d2H = H, dH, d2H
        self._r3, self._wq, self._dofs, self._free = rq, wq, dofs, free
        self._m, self._nu, self._ndof = m, nu, ndof
        self.lam_h = self.lam(np.zeros(self.rq.size), np.zeros(self.rq.size), k=2)

    def _rayleigh(self, x: np.ndarray, wk: np.ndarray, wm: np.ndarray) -> float:
        """λ = U/T를 **요소 곡률에서 직접** 적분한다.

        조립된 K로 xᵀKx를 만들면 큰 항들의 상쇄 때문에 상대오차가 ε·λ_max/λ(4계 연산자에서
        ~N⁴ε)가 되지만, 곡률을 형상함수로 바로 만들면 ε·(L/h)² 수준이다. 고유벡터 오차는
        Rayleigh 몫에 2차로만 들어간다.
        """
        u = np.zeros(self._ndof)
        u[self._free] = x
        ue = u[self._dofs]                                     # (E,4)
        R = np.einsum("egi,ei->eg", self._H, ue)
        dR = np.einsum("egi,ei->eg", self._dH, ue)
        d2R = np.einsum("egi,ei->eg", self._d2H, ue)
        r, m, nu = self._r3, self._m, self._nu
        krr, kth, krt = d2R, dR / r - m ** 2 * R / r ** 2, m * (dR / r - R / r ** 2)
        q = (krr + kth) ** 2 - 2 * (1 - nu) * (krr * kth - krt ** 2)
        U = float(np.sum(wk.reshape(r.shape) * q * self._wq))
        T = float(np.sum(wm.reshape(r.shape) * R ** 2 * self._wq))
        return U / T

    def lam(self, d_K: np.ndarray, d_M: np.ndarray, k: int = 2) -> np.ndarray:
        wk, wm = 1.0 - d_K, 1.0 - d_M
        K = (self.AK @ wk).reshape(self.nf, self.nf)
        M = (self.AM @ wm).reshape(self.nf, self.nf)
        d = 1.0 / np.sqrt(np.diag(M))
        _, V = eigh(K * d[:, None] * d[None, :], M * d[:, None] * d[None, :],
                    subset_by_index=[0, k - 1])
        return np.array([self._rayleigh(d * V[:, j], wk, wm) for j in range(k)])


#: FEM 요소 수(균일 격자 + 밴드 경계 절점). N = 100 대 480 ≤ 0.0015 floor(보고서 §2).
FEM_N = 100
_FM: dict = {}


def _fem_nodes(n_elem: int = FEM_N, cuts=()) -> np.ndarray:
    """균일 격자 + 밴드 경계 절점. **요소 길이 ≥ 0.05h를 보장한다**(2026-10-06 정정).

    처음 구현은 경계를 그냥 끼워 넣어, 경계가 기준 절점과 µm 이내(~1e-9 h)로 붙으면 그 요소의 강성이
    h⁻³로 폭주해 고유해가 망가졌다(ξ = 0.73, p = 0.097에서 η̄ = 104). 0.01h 요소까지는 무해함을
    확인했다(옛 결과 대비 ≤ 1e-6 floor). 그래서 경계는 그대로 끼워 넣되, 기존 절점과 0.05h 안이면
    그 **내부** 절점을 경계로 옮긴다. 끝점 a·b와 0.05h 안이면 경계를 절점으로 두지 않는다(계단이
    0.05h 이내에서만 가우스 적분에 맡겨진다).
    """
    nodes = list(np.linspace(PLATE.a, PLATE.b, n_elem + 1))
    h = (PLATE.b - PLATE.a) / n_elem
    fixed = {0, len(nodes) - 1}                       # 끝점 + 이미 경계가 된 절점
    for c in sorted(float(c) for c in cuts):
        if not PLATE.a < c < PLATE.b:
            continue
        arr = np.asarray(nodes)
        j = int(np.argmin(np.abs(arr - c)))
        if abs(arr[j] - c) < 0.05 * h:
            if j not in fixed:
                nodes[j] = c
                fixed.add(j)
            continue
        k = int(np.searchsorted(arr, c))
        nodes.insert(k, c)
        fixed = {f + 1 if f >= k else f for f in fixed} | {k}
    return np.asarray(nodes)


class FemRail:
    """모드 목록 전체의 비섭동 재해 η̄ = Λ_d/Λ_h − 1 (균일판 정확결합) — N4 생산 솔버."""

    def __init__(self, modes=MODES_DEF, n_elem: int = FEM_N, cuts=()):
        self.modes = tuple(tuple(x) for x in modes)
        self.n_elem, self.cuts = n_elem, tuple(float(c) for c in cuts)
        nodes = _fem_nodes(n_elem, self.cuts)
        self._m = {}
        for m in sorted({m for m, _ in self.modes}):
            key = (m, n_elem, self.cuts)
            if not self.cuts:
                if key not in _FM:
                    _FM[key] = _FemM(m, nodes)
                self._m[m] = _FM[key]
            else:                                         # 밴드 진실 전용(셀마다 한 번)
                self._m[m] = _FemM(m, nodes)
        self.r = next(iter(self._m.values())).rq
        #: m별로 필요한 고유값 개수(반경차수 n의 최댓값 + 1)
        self._k = {m: 1 + max(n for mm, n in self.modes if mm == m) for m in self._m}

    def eta_fields(self, d_K: np.ndarray) -> np.ndarray:
        d_K = np.asarray(d_K, dtype=float)
        if d_K.max(initial=0.0) >= 1.0 or d_K.min(initial=0.0) < 0.0:
            raise ValueError(f"d_K 범위 밖(max {d_K.max():.4f}) — 순방향 미정의 영역 평가 시도")
        d_M = 1.0 - (1.0 - d_K) ** (1.0 / 3.0)
        lam_d = {m: fm.lam(d_K, d_M, k=self._k[m]) for m, fm in self._m.items()}
        return np.array([lam_d[m][n] / self._m[m].lam_h[n] - 1.0 for m, n in self.modes])

    def eta(self, prof: Profile, xi: float, s_bar: float) -> np.ndarray:
        if s_bar <= 0.0:
            return np.zeros(len(self.modes))
        if prof.peak(s_bar) >= 1.0:
            raise ValueError(f"{prof.label}: S̄={s_bar:.4f} ≥ 상한 {prof.ceiling():.4f}")
        return self.eta_fields(prof.d_K(self.r, xi, s_bar))


_RAILS: dict = {}


def rail(modes=MODES_DEF) -> "FemRail":
    """생산 솔버(반경 FEM, 균일 격자) — 가우시안 진실과 모든 역맵이 같은 솔버를 쓴다."""
    key = tuple(tuple(x) for x in modes)
    if key not in _RAILS:
        _RAILS[key] = FemRail(modes)
    return _RAILS[key]


_POOLS: dict = {}


def linear_pool(modes=MODES_DEF):
    """생산 1차 맵의 커널 풀(n_grid 2001 — `cli._pool`과 같다)."""
    key = tuple(tuple(x) for x in modes)
    if key not in _POOLS:
        _POOLS[key] = [ker.mode_kernel(PLATE, m=m, n=n, n_grid=2001) for m, n in modes]
    return _POOLS[key]


def eta_linear(modes, xi: float, s_bar: float, w: float) -> np.ndarray:
    """생산 1차 맵(가우시안, 질량항 정확결합) — `forward.eta_bar_linear_mass`."""
    return fwd.eta_bar_linear_mass(linear_pool(modes), xi, s_bar, w, PLATE,
                                   coupling="exact")


def truth_eta(prof: Profile, xi: float, s_bar: float, modes=MODES_DEF) -> np.ndarray:
    """진실 관측량 — 가우시안은 역맵과 같은 솔버, 밴드는 경계에 절점을 둔 FEM(계단을 정확히)."""
    if prof.kind == "band":
        return FemRail(modes, cuts=prof.bounds_r(xi)).eta(prof, xi, s_bar)
    return rail(modes).eta(prof, xi, s_bar)


def sigma_obs(modes, c: float = C_REF) -> np.ndarray:
    return noi.sigma_y_for_modes([m for m, _ in modes], c)


# ============================================================ 역산
def _whiten(sigma):
    return noi.whitening(sigma)


def fit_fixed(y, sigma, w: float = W_REF, *, model: str = "exact", modes=MODES_DEF,
              starts=(0.2, 0.5, 0.8), s0: float = 0.03) -> dict:
    """폭 고정(가우시안) 역산 θ = (ξ_d, S̄_D). S̄ 상한 = 그 폭의 물리 상한(탐색점 보호).

    model = "exact"(비섭동 Ritz, 같은 솔버) | "linear"(생산 1차 맵).
    """
    Wh = _whiten(sigma)
    y = np.asarray(y, dtype=float)
    prof = Profile("gauss", w)
    s_hi = prof.ceiling() * _PEAK_MAX
    fwd_fn = ((lambda x, s: rail(modes).eta(prof, x, s)) if model == "exact"
              else (lambda x, s: eta_linear(modes, x, s, w)))
    best = None
    for x0 in starts:
        p0 = [float(x0), min(float(s0), 0.5 * s_hi)]
        sol = least_squares(lambda p: Wh @ (fwd_fn(p[0], p[1]) - y), p0,
                            bounds=([0.0, 0.0], [1.0, s_hi]), x_scale=[0.1, 0.01],
                            jac="3-point", diff_step=_DIFF_STEP,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        c2 = float(sol.fun @ sol.fun)
        if best is None or c2 < best["chi2"]:
            best = {"xi": float(sol.x[0]), "s_bar": float(sol.x[1]), "w": w,
                    "chi2": c2, "nfev": int(sol.nfev)}
    tol = 1e-6
    best["boundary_hit"] = bool(best["xi"] < tol or best["xi"] > 1 - tol
                                or best["s_bar"] < tol or best["s_bar"] > s_hi * (1 - 1e-6))
    return best


def fit_free(y, sigma, *, modes=MODES_DEF, starts=((0.2, 3e-3), (0.5, 3e-3), (0.8, 3e-3),
                                                     (0.5, 1.5e-3), (0.5, 7e-3)),
             a0: float = 0.15, w_bounds=W_BOUNDS) -> dict:
    """폭 자유 역산 — (ξ_d, peak A ∈ [0,1), w)로 매개화해 물리영역이 상자가 되게 한다.

    S̄_D = A·w√π/(b − a). 탐색 중 어떤 점도 peak d_K ≥ 1을 평가하지 않는다.
    """
    Wh = _whiten(sigma)
    y = np.asarray(y, dtype=float)
    R = rail(modes)

    def eta3(p):
        prof = Profile("gauss", float(p[2]))
        return R.eta(prof, float(p[0]), prof.s_from_peak(float(p[1])))

    lo = [0.0, 0.0, w_bounds[0]]
    hi = [1.0, _PEAK_MAX, w_bounds[1]]
    best = None
    for x0, w0 in starts:
        sol = least_squares(lambda p: Wh @ (eta3(p) - y), [x0, a0, w0], bounds=(lo, hi),
                            x_scale=[0.1, 0.1, 1e-3], jac="3-point",
                            diff_step=_DIFF_STEP, xtol=1e-12, ftol=1e-12,
                            gtol=1e-12, max_nfev=600)
        c2 = float(sol.fun @ sol.fun)
        if best is None or c2 < best["chi2"]:
            prof = Profile("gauss", float(sol.x[2]))
            best = {"xi": float(sol.x[0]), "peak": float(sol.x[1]), "w": float(sol.x[2]),
                    "s_bar": prof.s_from_peak(float(sol.x[1])), "chi2": c2,
                    "nfev": int(sol.nfev)}
    tol = 1e-6
    best["boundary_hit"] = bool(best["xi"] < tol or best["xi"] > 1 - tol
                                or best["peak"] < tol or best["peak"] > _PEAK_MAX - 1e-6
                                or best["w"] < w_bounds[0] * (1 + 1e-4)
                                or best["w"] > w_bounds[1] * (1 - 1e-4))
    best["w_at_bound"] = ("lo" if best["w"] < w_bounds[0] * (1 + 1e-4) else
                          "hi" if best["w"] > w_bounds[1] * (1 - 1e-4) else "")
    return best


def profile_over_w(y, sigma, w_grid=W_GRID, *, modes=MODES_DEF, starts=(0.2, 0.5, 0.8),
                   xi_hint: float | None = None) -> np.ndarray:
    """χ²_p(w) = min_{ξ,S̄} χ² — 폭 격자 위의 프로파일(정확 역맵)."""
    out = np.empty(len(w_grid))
    st = tuple(starts) + ((xi_hint,) if xi_hint is not None else ())
    for k, w in enumerate(w_grid):
        out[k] = fit_fixed(y, sigma, float(w), model="exact", modes=modes,
                           starts=st)["chi2"]
    return out


def profile_interval(w_grid, prof_chi2, level: float = 3.841458820694124) -> dict:
    """{w : χ²_p − min ≤ level}의 범위 — 한쪽이 격자 끝에 닿으면 그대로 표시한다."""
    d = np.asarray(prof_chi2) - np.min(prof_chi2)
    inside = d <= level
    idx = np.nonzero(inside)[0]
    w = np.asarray(w_grid)
    runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if idx.size else []
    return {"w_min_mm": float(w[np.argmin(prof_chi2)] * 1e3),
            "w_lo_mm": float(w[idx[0]] * 1e3) if idx.size else float("nan"),
            "w_hi_mm": float(w[idx[-1]] * 1e3) if idx.size else float("nan"),
            "lo_at_grid_edge": bool(idx.size and idx[0] == 0),
            "hi_at_grid_edge": bool(idx.size and idx[-1] == len(w) - 1),
            "n_runs": len(runs)}


# ============================================================ 식별성(1차 맵 야코비안)
def jac_linear(modes, xi: float, s_bar: float, w: float, with_w: bool = True) -> np.ndarray:
    """생산 1차 맵의 야코비안 ∂η̄/∂(ξ, S̄[, w]) — 중심차분(`jacobian_linear_mass`와 같은 간격)."""
    J = fwd.jacobian_linear_mass(linear_pool(modes), xi, s_bar, w, PLATE, coupling="exact")
    if not with_w:
        return J
    h = 1e-3 * w
    cw = (eta_linear(modes, xi, s_bar, w + h) - eta_linear(modes, xi, s_bar, w - h)) / (2 * h)
    return np.column_stack([J, cw])


def jac_exact(modes, xi: float, s_bar: float, w: float) -> np.ndarray:
    """비섭동 Ritz 맵의 야코비안 ∂η̄/∂(ξ, S̄, w) — 중심차분."""
    R = rail(modes)
    f = lambda x, s, ww: R.eta(Profile("gauss", ww), x, s)
    hx, hs, hw = 1e-4, max(1e-6, 1e-3 * s_bar), 1e-3 * w
    return np.column_stack([(f(xi + hx, s_bar, w) - f(xi - hx, s_bar, w)) / (2 * hx),
                            (f(xi, s_bar + hs, w) - f(xi, s_bar - hs, w)) / (2 * hs),
                            (f(xi, s_bar, w + hw) - f(xi, s_bar, w - hw)) / (2 * hw)])


def crlb_fixed_free(J3: np.ndarray, sigma: np.ndarray) -> dict:
    """폭 고정 대 폭 nuisance(슈어 보수, 받은 계획서 Eq. M.4)의 (ξ, S̄) CRLB.

    J3 열 = (ξ, S̄, w[m]). 관심모수 (ξ, S̄)를 그대로 두고 nuisance만 바꾸므로 슈어 보수는
    nuisance 재매개화에 불변이다.
    """
    Wh = _whiten(sigma)
    Jw = Wh @ J3
    F = Jw.T @ Jw
    Ftt, Ftw, Fww = F[:2, :2], F[:2, 2:], F[2:, 2:]
    Feff = Ftt - Ftw @ np.linalg.solve(Fww, Ftw.T)
    inv_fix = np.linalg.inv(Ftt)
    inv_eff = np.linalg.inv(Feff)
    try:
        inv_full = np.linalg.inv(F)
        crlb_w = float(math.sqrt(inv_full[2, 2]) * 1e3)
    except np.linalg.LinAlgError:
        crlb_w = float("inf")
    # 물리단위 열 스케일(ξ→mm, S̄→%p, w→mm)에서의 특이값
    scale = np.array([1.0 / (EXT * 1e3), 1.0 / 100.0, 1e-3])
    sv = np.linalg.svd(Jw * scale[None, :], compute_uv=False)
    sv2 = np.linalg.svd(Jw[:, :2] * scale[None, :2], compute_uv=False)
    return {"crlb_xi_fixed_mm": float(math.sqrt(inv_fix[0, 0]) * EXT * 1e3),
            "crlb_s_fixed_pp": float(math.sqrt(inv_fix[1, 1]) * 100),
            "crlb_xi_free_mm": float(math.sqrt(inv_eff[0, 0]) * EXT * 1e3),
            "crlb_s_free_pp": float(math.sqrt(inv_eff[1, 1]) * 100),
            "crlb_w_mm": crlb_w,
            "infl_xi": float(math.sqrt(inv_eff[0, 0] / inv_fix[0, 0])),
            "infl_s": float(math.sqrt(inv_eff[1, 1] / inv_fix[1, 1])),
            "sv3_max": float(sv[0]), "sv3_min": float(sv[-1]),
            "cond3": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
            "sv2_min": float(sv2[-1]),
            "rank3_tol1e-8": int(np.sum(sv > 1e-8 * sv[0]))}


def crlb_subset(modes, xi, s_bar, w, c=C_REF) -> dict:
    """부분집합 CRLB — **부분집합 자기의** 관측량 종류로 Σ_y를 만든다."""
    return crlb_fixed_free(jac_linear(modes, xi, s_bar, w), sigma_obs(modes, c))


def d_optimal(pool_modes, k: int, xi, s_bar, w, c=C_REF, nuisance_w: bool = False):
    """det F(관심모수 2×2; nuisance_w면 슈어 보수) 최대 k-부분집합 — 잡음은 부분집합별."""
    best = None
    for combo in combinations(pool_modes, k):
        J3 = jac_linear(combo, xi, s_bar, w)
        Wh = _whiten(sigma_obs(combo, c))
        Jw = Wh @ J3
        F = Jw.T @ Jw
        Ftt = F[:2, :2]
        if nuisance_w:
            Ftt = Ftt - F[:2, 2:] @ np.linalg.solve(F[2:, 2:], F[2:, :2])
        det = float(np.linalg.det(Ftt))
        if best is None or det > best["det"]:
            best = {"modes": combo, "det": det}
    return best


# ============================================================ 부호전환 반경
def sign_reversal_linear(w: float, s_bar: float = 0.03, n: int = 2000,
                         interpolate: bool = False) -> np.ndarray:
    """1차 맵의 모드별 ξ* — 생산 정의(`figures.null_loci`: 격자 첫 교차점)를 폭 일반화."""
    xs = np.linspace(0.02, 0.9999, n)
    E = np.array([eta_linear(MODES_DEF, x, s_bar, w) for x in xs])
    out = []
    for j in range(E.shape[1]):
        idx = np.nonzero(np.diff(np.sign(E[:, j])) != 0)[0]
        if not idx.size:
            out.append(np.nan)
            continue
        i = int(idx[0])
        if interpolate:
            out.append(float(xs[i] - E[i, j] * (xs[i + 1] - xs[i]) / (E[i + 1, j] - E[i, j])))
        else:
            out.append(float(xs[i]))
    return np.array(out)


def sign_reversal_exact(w: float, s_bar: float = 0.03, n: int = 400) -> np.ndarray:
    """비섭동 Ritz 맵의 ξ*(선형 보간). 상한 밖이면 NaN."""
    prof = Profile("gauss", w)
    if prof.peak(s_bar) >= 1:
        return np.full(len(MODES_DEF), np.nan)
    xs = np.linspace(0.02, 0.9999, n)
    E = np.array([rail().eta(prof, x, s_bar) for x in xs])
    out = []
    for j in range(E.shape[1]):
        idx = np.nonzero(np.diff(np.sign(E[:, j])) != 0)[0]
        if not idx.size:
            out.append(np.nan)
            continue
        i = int(idx[0])
        out.append(float(xs[i] - E[i, j] * (xs[i + 1] - xs[i]) / (E[i + 1, j] - E[i, j])))
    return np.array(out)


# ============================================================ 작업 단위(셀)
def _dist_chi2_power(lam: float, dof: int, alpha: float = 0.05) -> float:
    """비중심 χ² 검정력 P(χ²_dof(λ) > χ²_dof,1−α) — 잔차가 불일치를 드러낼 확률."""
    from scipy.stats import chi2, ncx2
    crit = chi2.ppf(1 - alpha, dof)
    if lam <= 0:
        return float(alpha)
    return float(ncx2.sf(crit, dof, lam))


def bias_cell(args) -> dict:
    """한 진실 셀의 무잡음 역산 3종 — (A) 정확·폭고정, (A′) 생산 1차·폭고정, (B) 폭자유."""
    prof_kw, xi, s_target, matching, modes = args
    prof = Profile(**prof_kw)
    s_true = matched_s_true(prof, xi, s_target, matching)
    row = {"profile": prof.label, "kind": prof.kind, "w_true_mm": prof.w * 1e3,
           "hwhm_true_mm": prof.hwhm_mm, "matching": matching, "xi_true": xi,
           "s_target": s_target, "s_true_nominal": s_true,
           "peak_true": prof.peak(s_true),
           "s_true_in_domain": prof.in_domain(xi, s_true) if s_true < prof.ceiling() else np.nan,
           "ceiling_true": prof.ceiling(), "modes": "|".join(f"m{m}n{n}" for m, n in modes)}
    if prof.peak(s_true) >= 1.0:
        row.update({"admissible": False})
        return row
    row["admissible"] = True
    y = truth_eta(prof, xi, s_true, modes)
    sig = sigma_obs(modes)
    floors = np.sqrt(np.diag(sig))
    row["y_max_over_floor"] = float(np.max(np.abs(y) / floors))
    dof_fixed = len(modes) - 2
    dof_free = len(modes) - 3
    starts = tuple(sorted({0.1, 0.3, 0.5, 0.7, 0.9, round(xi, 6)}))
    for tag, model in (("A_exact", "exact"), ("Ap_linear", "linear")):
        f = fit_fixed(y, sig, W_REF, model=model, modes=modes, starts=starts)
        row.update({f"{tag}_xi": f["xi"], f"{tag}_s": f["s_bar"],
                    f"{tag}_bias_xi_mm": (f["xi"] - xi) * EXT * 1e3,
                    f"{tag}_bias_s_nom_pp": (f["s_bar"] - s_true) * 100,
                    f"{tag}_bias_s_dom_pp": (GAUSS_REF.in_domain(f["xi"], f["s_bar"])
                                             - row["s_true_in_domain"]) * 100,
                    f"{tag}_chi2": f["chi2"],
                    f"{tag}_power": _dist_chi2_power(f["chi2"], dof_fixed),
                    f"{tag}_boundary": f["boundary_hit"]})
    fb = fit_free(y, sig, modes=modes,
                  starts=((xi, 3e-3), (0.2, 3e-3), (0.5, 3e-3), (0.8, 3e-3),
                          (xi, 1.5e-3), (xi, 7e-3)))
    gfit = Profile("gauss", fb["w"])
    row.update({"B_xi": fb["xi"], "B_s": fb["s_bar"], "B_w_mm": fb["w"] * 1e3,
                "B_peak": fb["peak"],
                "B_bias_xi_mm": (fb["xi"] - xi) * EXT * 1e3,
                "B_bias_s_nom_pp": (fb["s_bar"] - s_true) * 100,
                "B_bias_s_dom_pp": (gfit.in_domain(fb["xi"], fb["s_bar"])
                                    - row["s_true_in_domain"]) * 100,
                "B_chi2": fb["chi2"],
                "B_power": (_dist_chi2_power(fb["chi2"], dof_free) if dof_free > 0
                            else float("nan")),
                "B_boundary": fb["boundary_hit"], "B_w_at_bound": fb["w_at_bound"]})
    return row


def profile_cell(args) -> dict:
    prof_kw, xi, s_target, modes = args
    prof = Profile(**prof_kw)
    row = {"profile": prof.label, "w_true_mm": prof.w * 1e3, "xi_true": xi,
           "s_target": s_target, "modes": "|".join(f"m{m}n{n}" for m, n in modes)}
    if prof.peak(s_target) >= 1.0:
        row["admissible"] = False
        return row
    row["admissible"] = True
    y = truth_eta(prof, xi, s_target, modes)
    sig = sigma_obs(modes)
    chi = profile_over_w(y, sig, W_GRID, modes=modes, xi_hint=xi)
    row.update(profile_interval(W_GRID, chi))
    row["chi2_min"] = float(np.min(chi))
    row["chi2_at_3mm"] = float(chi[np.argmin(np.abs(W_GRID - W_REF))])
    row["chi2_profile"] = json.dumps([round(float(v), 6) for v in chi])
    return row


def mc_cell(args) -> dict:
    """잡음 MC 한 셀 — 폭 고정(정확 맵) 대 폭 자유. 구간은 의사진실 θ*의 국소 CRLB."""
    prof_kw, xi, s_true, n_real, seed, c = args
    prof = Profile(**prof_kw)
    modes = MODES_DEF
    y0 = truth_eta(prof, xi, s_true, modes)
    sig = sigma_obs(modes, c)
    # 의사진실(무잡음 최적점)과 그 점의 CRLB
    f0 = fit_fixed(y0, sig, W_REF, model="exact", starts=(0.1, 0.3, 0.5, 0.7, 0.9, xi))
    g0 = fit_free(y0, sig, starts=((xi, 3e-3), (0.2, 3e-3), (0.5, 3e-3), (0.8, 3e-3),
                                    (xi, 1.5e-3), (xi, 7e-3)))
    Jf = jac_exact(modes, f0["xi"], f0["s_bar"], W_REF)
    cf = crlb_fixed_free(Jf, sig)
    Jg = jac_exact(modes, g0["xi"], g0["s_bar"], g0["w"])
    cg = crlb_fixed_free(Jg, sig)
    hw_fix = 1.96 * cf["crlb_xi_fixed_mm"] / (EXT * 1e3)
    hw_free = 1.96 * cg["crlb_xi_free_mm"] / (EXT * 1e3)
    rng = np.random.default_rng(seed)
    eps = noi.sample(sig, rng, size=n_real)
    xf, sf, xg, sg, wg = (np.empty(n_real) for _ in range(5))
    hit_f = hit_g = 0
    for k in range(n_real):
        y = y0 + eps[k]
        a = fit_fixed(y, sig, W_REF, model="exact", starts=(f0["xi"],), s0=f0["s_bar"])
        b = fit_free(y, sig, starts=((g0["xi"], g0["w"]),), a0=g0["peak"])
        xf[k], sf[k] = a["xi"], a["s_bar"]
        xg[k], sg[k], wg[k] = b["xi"], b["s_bar"], b["w"]
        hit_f += int(a["boundary_hit"])
        hit_g += int(b["boundary_hit"])
    to_mm = EXT * 1e3

    def stats(x, hw, tag):
        return {f"{tag}_bias_xi_mm": float(np.mean(x - xi) * to_mm),
                f"{tag}_std_xi_mm": float(np.std(x, ddof=1) * to_mm),
                f"{tag}_rmse_xi_mm": float(np.sqrt(np.mean((x - xi) ** 2)) * to_mm),
                f"{tag}_cov95_xi": float(np.mean(np.abs(x - xi) <= hw)),
                f"{tag}_halfwidth95_mm": float(hw * to_mm)}

    return {"profile": prof.label, "w_true_mm": prof.w * 1e3, "xi_true": xi,
            "s_true": s_true, "c": c, "n_real": n_real, "seed": seed,
            "pseudo_fixed_xi": f0["xi"], "pseudo_fixed_s": f0["s_bar"],
            "pseudo_free_xi": g0["xi"], "pseudo_free_s": g0["s_bar"],
            "pseudo_free_w_mm": g0["w"] * 1e3,
            **stats(xf, hw_fix, "fixed"), **stats(xg, hw_free, "free"),
            "fixed_crlb_xi_mm": cf["crlb_xi_fixed_mm"],
            "free_crlb_xi_mm": cg["crlb_xi_free_mm"],
            "free_crlb_w_mm": cg["crlb_w_mm"],
            "free_w_median_mm": float(np.median(wg) * 1e3),
            "free_w_p05_mm": float(np.quantile(wg, 0.05) * 1e3),
            "free_w_p95_mm": float(np.quantile(wg, 0.95) * 1e3),
            "free_w_at_bound_frac": float(np.mean((wg <= W_BOUNDS[0] * (1 + 1e-4))
                                                  | (wg >= W_BOUNDS[1] * (1 - 1e-4)))),
            "fixed_bias_s_pp": float(np.mean(sf - s_true) * 100),
            "free_bias_s_pp": float(np.mean(sg - s_true) * 100),
            "fixed_boundary_frac": hit_f / n_real, "free_boundary_frac": hit_g / n_real}


# ============================================================ 실행기
def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:
        return "unknown"


def _map(fn, jobs, workers: int):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with get_context("fork").Pool(workers) as p:
        return p.map(fn, jobs, chunksize=1)


def run_anchors() -> pd.DataFrame:
    """r3.9 앵커 재현 — Table 3a 기본 CRLB, ξ*(3 mm)."""
    rows = []
    c = crlb_subset(MODES_DEF, 0.5, 0.05, W_REF)
    rows.append({"item": "table3a_default_crlb_xi_mm", "value": c["crlb_xi_fixed_mm"],
                 "r39": 0.279})
    rows.append({"item": "table3a_default_crlb_s_pp", "value": c["crlb_s_fixed_pp"],
                 "r39": 0.326})
    xs = sign_reversal_linear(W_REF, 0.03)
    for (m, _), x, ref in zip(MODES_DEF, xs, (0.629, 0.633, 0.706, 0.966)):
        rows.append({"item": f"xi_star_m{m}_s3pct", "value": x, "r39": ref})
    return pd.DataFrame(rows)


def run_identifiability() -> pd.DataFrame:
    """CRLB 고정폭 대 폭 nuisance — 기본·확장(25.6 kHz) 집합, 폭·위치·심각도 격자."""
    rows = []
    sets = {"default_m0_3": MODES_DEF, "ext25k_all8": MODES_EXT}
    # 확장풀 D-최적 4(관심모수 det, 부분집합별 잡음) — 고정폭 기준과 폭 nuisance 기준
    for xi in (0.2, 0.5, 0.8, 0.95):
        for s in (0.01, 0.05, 0.15):
            for w in (0.0015, 0.003, 0.006, 0.009):
                if Profile("gauss", w).peak(s) >= 1:
                    rows.append({"set": "-", "xi": xi, "s_bar": s, "w_mm": w * 1e3,
                                 "admissible": False})
                    continue
                dsets = dict(sets)
                dopt = d_optimal(MODES_EXT, 4, xi, s, w)
                dsets["ext25k_dopt4_fixedw"] = dopt["modes"]
                dopt_n = d_optimal(MODES_EXT, 4, xi, s, w, nuisance_w=True)
                dsets["ext25k_dopt4_nuisancew"] = dopt_n["modes"]
                for name, modes in dsets.items():
                    c = crlb_subset(modes, xi, s, w)
                    Je = jac_exact(modes, xi, s, w)
                    ce = crlb_fixed_free(Je, sigma_obs(modes))
                    rows.append({"set": name, "xi": xi, "s_bar": s, "w_mm": w * 1e3,
                                 "admissible": True,
                                 "labels": "|".join(f"m{m}n{n}" for m, n in modes),
                                 **c,
                                 "exactJ_crlb_xi_fixed_mm": ce["crlb_xi_fixed_mm"],
                                 "exactJ_crlb_xi_free_mm": ce["crlb_xi_free_mm"],
                                 "exactJ_crlb_w_mm": ce["crlb_w_mm"]})
    return pd.DataFrame(rows)


def run_canonical_dopt_check() -> pd.DataFrame:
    """정본 `compare_mode_sets`의 D-최적(위치별 잡음) 대 부분집합별 잡음 — 25.6 kHz, ξ=0.5."""
    pool = ker.mode_pool(PLATE, ms=(0, 1, 2, 3, 4), ns=(0, 1), n_grid=2001)
    rows = msel.compare_mode_sets(PLATE, pool, (0.5, 0.05), W_REF, C_REF, f_max=25.6e3,
                                  mass="exact")
    out = []
    for r in rows:
        if r["set"] != "d_optimal_4":
            continue
        modes = tuple((int(l[1]), int(l[3])) for l in r["labels"])
        c = crlb_subset(modes, 0.5, 0.05, W_REF)
        out.append({"set": "d_optimal_4 (25.6 kHz)", "labels": "|".join(r["labels"]),
                    "canonical_crlb_xi_mm": r["crlb_xi_mm"],
                    "subset_noise_crlb_xi_mm": c["crlb_xi_fixed_mm"],
                    "canonical_crlb_s_pp": r["crlb_s_pp"],
                    "subset_noise_crlb_s_pp": c["crlb_s_fixed_pp"]})
    best = d_optimal(MODES_EXT, 4, 0.5, 0.05, W_REF)
    c = crlb_subset(best["modes"], 0.5, 0.05, W_REF)
    out.append({"set": "d_optimal_4 re-selected with subset noise",
                "labels": "|".join(f"m{m}n{n}" for m, n in best["modes"]),
                "canonical_crlb_xi_mm": np.nan,
                "subset_noise_crlb_xi_mm": c["crlb_xi_fixed_mm"],
                "canonical_crlb_s_pp": np.nan,
                "subset_noise_crlb_s_pp": c["crlb_s_fixed_pp"]})
    base = crlb_subset(MODES_DEF, 0.5, 0.05, W_REF)["crlb_xi_fixed_mm"]
    for r in out:
        r["default_crlb_xi_mm"] = base
        r["reduction_vs_default_subset_noise"] = 1 - r["subset_noise_crlb_xi_mm"] / base
    return pd.DataFrame(out)


def run_sign_reversal() -> pd.DataFrame:
    rows = []
    for w in (0.0015, 0.002, 0.003, 0.004, 0.006, 0.009):
        lin = sign_reversal_linear(w, 0.03)
        lin_i = sign_reversal_linear(w, 0.03, interpolate=True)
        ex = sign_reversal_exact(w, 0.03)
        for (m, _), a, b, c in zip(MODES_DEF, lin, lin_i, ex):
            rows.append({"w_mm": w * 1e3, "m": m, "xi_star_linear_grid": a,
                         "xi_star_linear_interp": b, "xi_star_exact_interp": c,
                         "r_star_linear_mm": (PLATE.a + b * EXT) * 1e3
                         if np.isfinite(b) else np.nan})
    return pd.DataFrame(rows)


def run_pocket_width() -> pd.DataFrame:
    """Table 3b 두 행(쌍평균만 / +분리)의 포켓 반경폭 감도 — 정본 `pocket_jacobian` 사용."""
    rows = []
    ms_p = (1, 2, 3)
    for rw in (0.003, 0.0045, 0.006, 0.009, 0.012):
        for xi in (0.2, 0.5, 0.8):
            for s in (0.01, 0.05):
                x = s * EXT / rw
                if x >= 1.0:
                    rows.append({"radial_width_mm": rw * 1e3, "xi": xi, "s_bar": s,
                                 "admissible": False})
                    continue
                Jf = msel.pocket_jacobian(PLATE, ms_p, (xi, s), include_splitting=False,
                                          radial_width=rw)
                Jfs = msel.pocket_jacobian(PLATE, ms_p, (xi, s), include_splitting=True,
                                           radial_width=rw)
                a = idf.metrics_from_J(Jf, noi.sigma_y_for_modes(list(ms_p), C_REF), EXT)
                b = idf.metrics_from_J(Jfs, noi.sigma_y_for_modes(list(ms_p), C_REF,
                                                                   include_splitting=True),
                                       EXT)
                r_c = sev.xi_to_r(xi, PLATE.a, PLATE.b)
                r1 = max(PLATE.a, r_c - 0.5 * rw)
                # 정본 `pocket_observables`는 포켓이 경계를 넘으면 폭을 지키고 **위치를 민다**
                # (r1 = max(a, r_c − W/2), r2 = r1 + W). 그 영역에서는 ξ가 관측량에 들어가지
                # 않아 F가 특이하다 — 매개화의 산물이지 물리적 비식별이 아니므로 표시만 한다.
                pinned = bool(r_c - 0.5 * rw < PLATE.a - 1e-12
                              or r_c + 0.5 * rw > PLATE.b + 1e-12)
                ok = (a["det_F"] > 0 and b["det_F"] > 0 and np.isfinite(a["crlb_xi_mm"])
                      and np.isfinite(b["crlb_xi_mm"]))
                rows.append({"radial_width_mm": rw * 1e3, "xi": xi, "s_bar": s,
                             "admissible": True, "pinned_at_edge": pinned,
                             "pocket_r1_mm": r1 * 1e3, "pocket_r2_mm": min(PLATE.b, r1 + rw) * 1e3,
                             "depth_frac": 1 - (1 - x) ** (1 / 3),
                             "crlb_xi_means_mm": a["crlb_xi_mm"],
                             "crlb_xi_means_split_mm": b["crlb_xi_mm"],
                             "crlb_s_means_pp": a["crlb_s_pp"],
                             "crlb_s_means_split_pp": b["crlb_s_pp"],
                             "gain_xi": (1 - b["crlb_xi_mm"] / a["crlb_xi_mm"]) if ok else np.nan,
                             "detF_ratio": (b["det_F"] / a["det_F"]) if ok else np.nan})
    return pd.DataFrame(rows)


def b7_decomp_cell(args) -> dict:
    """r3.9 대리 MC(b7)와 같은 밴드 셀을 **Kirchhoff 안에서만** 역산 — 발자국·선형화의 몫.

    b7 진실 = 3D 앵커 보정 밴드(반폭 0.10(b−a), 전폭 4.232 mm), 역맵 = 생산 1차 가우시안(3 mm).
    여기서는 3D 불일치 δ를 빼고 같은 밴드의 Kirchhoff 재해만 진실로 둔다. 따라서
      (A′) 1차 맵 편향 = 발자국 + 선형화,   (A) 정확 맵 편향 = 발자국만
    이고, b7 편향 − (A′)가 3D↔Kirchhoff 모델형식의 몫이다(같은 셀·같은 맵).
    """
    xi, s_bar = args
    prof = Profile("band", 2 * 0.10 * EXT)
    y = truth_eta(prof, xi, s_bar)
    sig = sigma_obs(MODES_DEF)
    starts = tuple(sorted({0.1, 0.3, 0.5, 0.7, 0.9, round(xi, 6)}))
    a = fit_fixed(y, sig, W_REF, model="exact", starts=starts)
    b = fit_fixed(y, sig, W_REF, model="linear", starts=starts)
    return {"xi_d": xi, "s_bar": s_bar, "band_full_width_mm": prof.w * 1e3,
            "A_exact_bias_xi_mm": (a["xi"] - xi) * EXT * 1e3, "A_exact_chi2": a["chi2"],
            "A_exact_boundary": a["boundary_hit"],
            "Ap_linear_bias_xi_mm": (b["xi"] - xi) * EXT * 1e3, "Ap_linear_chi2": b["chi2"],
            "Ap_linear_boundary": b["boundary_hit"]}


def run_b7_decomp(workers: int) -> pd.DataFrame:
    from .. import cli as _cli          # 경로 상수만 읽는다(b7 산출물 위치)
    b7 = pd.read_csv(_cli.DATA / "b7_mc_surrogate.csv")
    b7 = b7[b7.truth_model == "3d_rail_surrogate"]
    cells = sorted({(round(float(x), 6), round(float(s), 6)) for x, s in zip(b7.xi_d, b7.s_bar)})
    rows = pd.DataFrame(_map(b7_decomp_cell, cells, workers))
    lo = b7[np.isclose(b7.sigma_rel, 1e-4)][["xi_d", "s_bar", "bias_xi_mm", "boundary_hit_prob"]]
    lo = lo.assign(xi_d=lo.xi_d.round(6), s_bar=lo.s_bar.round(6)).rename(
        columns={"bias_xi_mm": "b7_bias_xi_mm_c1e-4", "boundary_hit_prob": "b7_boundary_hit_c1e-4"})
    out = rows.merge(lo, on=["xi_d", "s_bar"], how="left")
    out["modelform_share_mm"] = out["b7_bias_xi_mm_c1e-4"] - out["Ap_linear_bias_xi_mm"]
    return out


# ============================================================ N5 대조 — 리드 후속 지시(2026-10-06)
#: b7 대리모델 손상족 — 반폭 0.10(b − a)의 축대칭 밴드(전폭 4.232 mm)
HALF_B7 = 0.10
W_B7 = 2 * HALF_B7 * EXT


def band_fem_eta(r1: float, r2: float, d_k: float, modes=MODES_DEF) -> np.ndarray:
    """[r1, r2]에 균일 d_K가 있는 밴드의 FEM 재해 — 경계에 절점을 둔다(계단을 정확히)."""
    R = FemRail(modes, cuts=(r1, r2))
    dK = np.where((R.r >= r1) & (R.r <= r2), float(d_k), 0.0)
    return R.eta_fields(dK)


def _b7_band(xi: float, half: float = HALF_B7):
    """b7 `surrogate_truth`와 같은 밴드 기하 — [a, b]로 절단."""
    r_c = sev.xi_to_r(xi, PLATE.a, PLATE.b)
    return max(PLATE.a, r_c - half * EXT), min(PLATE.b, r_c + half * EXT)


def b6_anchor_fem() -> pd.DataFrame:
    """b6 앵커 전부(두 폭족)의 Kirchhoff 밴드 재해를 FEM으로 — 정본(르장드르 36) 열과 나란히."""
    from .. import cli as _cli
    d = pd.read_csv(_cli.DATA / "b6_epert_fem.csv")
    pool = linear_pool(MODES_DEF)
    ms = [m for m, _ in MODES_DEF]
    rows = []
    for (h, x, pdep), g in d.groupby(["half_xi", "xi_c", "depth_frac"]):
        r1, r2 = float(g.r1_mm.iloc[0]) / 1e3, float(g.r2_mm.iloc[0]) / 1e3
        dk = 1.0 - (1.0 - float(pdep)) ** 3
        kf = band_fem_eta(r1, r2, dk)
        lin = fwd.eta_bar_linear_band(pool, r1, r2, float(pdep), PLATE, coupling="exact")
        for _, row in g.iterrows():
            i = ms.index(int(row["m"]))
            rec = {k: row[k] for k in ("half_xi", "xi_c", "depth_frac", "r1_mm", "r2_mm",
                                       "s_bar_pct", "m", "matched", "eta_3d",
                                       "eta_lin", "eta_kirchhoff_exact", "delta_modelform")}
            rec.update({"eta_lin_recomputed": lin[i], "eta_K_fem": kf[i],
                        "delta_fem": float(row["eta_3d"]) - kf[i],
                        "K_fem_minus_canon": kf[i] - float(row["eta_kirchhoff_exact"])})
            rows.append(rec)
    return pd.DataFrame(rows)


def b7_delta_spec(anchors: pd.DataFrame, val: str, half: float = HALF_B7) -> dict:
    """b7의 δ 격자 조립 규칙 그대로(kgrid → ξ 방향 → S̄ 방향 채움, 외삽은 클램프)."""
    fam = anchors[np.isclose(anchors.half_xi, half) & anchors.matched.astype(bool)].copy()
    fam["s_key"] = fam["s_bar_pct"].round(2)
    xis = [float(v) for v in sorted(fam.xi_c.unique())]
    ss = [float(v) for v in sorted(fam.s_key.unique())]
    ms = [m for m, _ in MODES_DEF]
    K = np.full((len(ms), len(xis), len(ss)), np.nan)
    for i, m in enumerate(ms):
        for j, x in enumerate(xis):
            for k, sk in enumerate(ss):
                sel = fam[(fam.m == m) & np.isclose(fam.xi_c, x) & np.isclose(fam.s_key, sk)]
                if len(sel):
                    K[i, j, k] = float(sel[val].iloc[0])
    n_missing = int(np.isnan(K).sum())
    for i in range(K.shape[0]):
        for k in range(K.shape[2]):
            col, ok = K[i, :, k], np.isfinite(K[i, :, k])
            if 0 < ok.sum() < len(xis) and ok.sum() >= 2:
                K[i, :, k] = np.interp(xis, np.array(xis)[ok], col[ok])
        for j in range(K.shape[1]):
            row, ok = K[i, j, :], np.isfinite(K[i, j, :])
            if 0 < ok.sum() < len(ss) and ok.sum() >= 1:
                K[i, j, :] = (np.interp(ss, np.array(ss)[ok], row[ok])
                              if ok.sum() >= 2 else row[ok][0])
    return {"xi": xis, "s_bar": [v / 100 for v in ss], "delta": K.tolist(),
            "half_xi": half, "n_missing": n_missing}


def _grid_then_lsq(y, sig, fn, hi2: float, xi_true: float, n_xi: int = 41, n_2: int = 13):
    """(ξ, θ₂) 역산 — 거친 격자의 상위 3점 + 다중시작 → least_squares(중심차분)."""
    Wh = _whiten(sig)
    y = np.asarray(y, dtype=float)
    xs = np.linspace(0.0, 1.0, n_xi)
    t2 = np.linspace(0.02 * hi2, 0.98 * hi2, n_2)
    cand = []
    for x in xs:
        for t in t2:
            r = Wh @ (fn(x, t) - y)
            cand.append((float(r @ r), x, t))
    cand.sort(key=lambda c: c[0])
    starts = [(c[1], c[2]) for c in cand[:3]] + [(xi_true, cand[0][2])]
    best = None
    for x0, t0 in starts:
        sol = least_squares(lambda p_: Wh @ (fn(p_[0], p_[1]) - y), [x0, t0],
                            bounds=([0.0, 0.0], [1.0, hi2]), x_scale=[0.1, 0.1 * hi2],
                            jac="3-point", diff_step=_DIFF_STEP,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        c2 = float(sol.fun @ sol.fun)
        if best is None or c2 < best[0]:
            best = (c2, float(sol.x[0]), float(sol.x[1]))
    tol = 1e-6
    bnd = bool(best[1] < tol or best[1] > 1 - tol or best[2] < tol * hi2
               or best[2] > hi2 * (1 - 1e-6))
    return {"chi2": best[0], "xi": best[1], "t2": best[2], "boundary_hit": bnd}


#: 2×2 역맵 — (선형화: "lin" | "ex") × (발자국: "G" = 가우시안 3 mm | "B" = 참 밴드)
MAPS = ("lin_G", "ex_G", "lin_B", "ex_B")


def _map_fn(name: str):
    """역맵 η̄(ξ, θ₂). 가우시안 θ₂ = 명목 S̄, 밴드 θ₂ = 밴드 안 d_K(peak) — 둘 다 상자 영역."""
    if name == "lin_G":
        return (lambda x, s: eta_linear(MODES_DEF, x, s, W_REF)), GAUSS_REF.ceiling() * _PEAK_MAX
    if name == "ex_G":
        return (lambda x, s: rail(MODES_DEF).eta(GAUSS_REF, x, s)), GAUSS_REF.ceiling() * _PEAK_MAX
    if name == "lin_B":
        pool = linear_pool(MODES_DEF)

        def f(x, A):
            r1, r2 = _b7_band(x)
            return fwd.eta_bar_linear_band(pool, r1, r2, 1.0 - (1.0 - A) ** (1.0 / 3.0),
                                           PLATE, coupling="exact")
        return f, _PEAK_MAX
    if name == "ex_B":
        def f(x, A):
            r1, r2 = _b7_band(x)
            return band_fem_eta(r1, r2, A)
        return f, _PEAK_MAX
    raise ValueError(name)


def reconcile_cell(args) -> list:
    """한 b7 셀 × 세 진실 × 2×2 역맵의 무잡음 적합(가중 c = 0.1 %)."""
    xi, s_bar, spec_fem, spec_can = args
    sig = sigma_obs(MODES_DEF)
    r1, r2 = _b7_band(xi)
    dk = s_bar * EXT / (r2 - r1)              # b7: band_depth_for_s_bar(절단 폭 기준)
    y_k = band_fem_eta(r1, r2, dk)
    from .. import montecarlo as _mc
    truths = {"K_band_fem": y_k,
              "y3D_fem": y_k + _mc.kappa_eval(spec_fem, xi, s_bar, key="delta")}
    if spec_can is not None:                  # b7가 실제로 쓴 대리 진실(정본 르장드르 + δ_정본)
        truths["y3D_canon_b7"] = _mc.surrogate_truth(PLATE, MODES_DEF, xi, s_bar, spec_can)
    out = []
    for tname, y in truths.items():
        for mname in MAPS:
            fn, hi2 = _map_fn(mname)
            f = _grid_then_lsq(y, sig, fn, hi2, xi)
            s_hat = (f["t2"] if mname.endswith("G") else f["t2"] * W_B7 / EXT)
            out.append({"xi_d": xi, "s_bar": s_bar, "truth": tname, "map": mname,
                        "xi_hat": f["xi"], "bias_xi_mm": (f["xi"] - xi) * EXT * 1e3,
                        "s_hat": s_hat, "bias_s_pp": (s_hat - s_bar) * 100,
                        "chi2_c1e-3": f["chi2"], "boundary_hit": f["boundary_hit"],
                        "truth_minus_canonb7_over_floor_max": np.nan})
        if spec_can is not None and tname == "y3D_fem":
            dlt = (truths["y3D_fem"] - truths["y3D_canon_b7"]) / np.sqrt(np.diag(sig))
            for r in out:
                if r["truth"] == "y3D_fem":
                    r["truth_minus_canonb7_over_floor_max"] = float(np.max(np.abs(dlt)))
    return out


def _decompose(cells: pd.DataFrame) -> pd.DataFrame:
    """셀별 2×2 분해 — 경로 1(a→b→c)·경로 2(a→a′→c)·두 요인 Shapley·상호작용·모델형식."""
    piv = cells.pivot_table(index=["truth", "xi_d", "s_bar"], columns="map",
                            values="bias_xi_mm").reset_index()
    a, b, ap, c = piv["lin_G"], piv["ex_G"], piv["lin_B"], piv["ex_B"]
    piv["path1_lin"], piv["path1_foot"] = a - b, b - c
    piv["path2_foot"], piv["path2_lin"] = a - ap, ap - c
    piv["shapley_lin"] = 0.5 * ((a - b) + (ap - c))
    piv["shapley_foot"] = 0.5 * ((a - ap) + (b - c))
    piv["interaction"] = a - b - ap + c
    piv["modelform"] = c
    piv["closure_resid"] = a - (piv.shapley_lin + piv.shapley_foot + piv.modelform)
    return piv


def run_reconcile(workers: int) -> dict:
    from .. import cli as _cli
    anc = b6_anchor_fem()
    anc.to_csv(out_csv("n4_reconcile_b6_anchors_fem"), index=False)
    spec_fem = b7_delta_spec(anc, "delta_fem")
    spec_can = b7_delta_spec(anc, "delta_modelform")
    b7 = pd.read_csv(_cli.DATA / "b7_mc_surrogate.csv")
    b7 = b7[b7.truth_model == "3d_rail_surrogate"]
    cells = sorted({(round(float(x), 6), round(float(s), 6)) for x, s in zip(b7.xi_d, b7.s_bar)})
    rows = []
    for r in _map(reconcile_cell, [(x, s, spec_fem, spec_can) for x, s in cells], workers):
        rows += r
    cdf = pd.DataFrame(rows)
    # 부분집합 표시: 43 = b7 c=1e-4에서 경계접촉 없음, 36 = 어느 c에서도 없음(N5 규약)
    hit = b7.assign(xi_d=b7.xi_d.round(6), s_bar=b7.s_bar.round(6))
    lo = hit[np.isclose(hit.sigma_rel, 1e-4)].set_index(["xi_d", "s_bar"]).boundary_hit_prob
    anyc = hit.groupby(["xi_d", "s_bar"]).boundary_hit_prob.max()
    key = list(zip(cdf.xi_d, cdf.s_bar))
    cdf["in43"] = [float(lo.loc[k]) == 0.0 for k in key]
    cdf["in36"] = [float(anyc.loc[k]) == 0.0 for k in key]
    b7c = hit[np.isclose(hit.sigma_rel, 1e-4)].set_index(["xi_d", "s_bar"]).bias_xi_mm
    cdf["b7_bias_xi_mm_c1e-4"] = [float(b7c.loc[k]) for k in key]
    cdf.to_csv(out_csv("n4_reconcile_cells"), index=False)
    dec = _decompose(cdf)
    flags = cdf.drop_duplicates(["xi_d", "s_bar"]).set_index(["xi_d", "s_bar"])[
        ["in43", "in36", "b7_bias_xi_mm_c1e-4"]]
    dec = dec.join(flags, on=["xi_d", "s_bar"])
    dec.to_csv(out_csv("n4_reconcile_decomposition"), index=False)
    return {"anchors": anc, "cells": cdf, "dec": dec, "spec_fem": spec_fem}


def run_b6_recheck() -> pd.DataFrame:
    """r3.9 §4.3 b6 수치(선형화 14.9·모델형식 10.2 floor, 104셀 중 4셀 부호 반대)를 FEM으로 재확인."""
    anc = pd.read_csv(out_csv("n4_reconcile_b6_anchors_fem"))
    m = anc[anc.matched.astype(bool)].copy()
    rows = []
    for floor_name, fl in (("2c_all(r3.9)", np.full(len(m), 2 * C_REF)),
                           ("mode_specific", np.where(m.m == 0, 2 * math.sqrt(2) * C_REF,
                                                      2 * C_REF))):
        for kname, kcol in (("canonical_legendre36", "eta_kirchhoff_exact"),
                            ("fem", "eta_K_fem")):
            e_lin = np.abs(m.eta_lin - m[kcol]) / fl
            e_mf = np.abs(m[kcol] - m.eta_3d) / fl
            e_tot = np.abs(m.eta_lin - m.eta_3d) / fl
            opp = np.sign(m[kcol]) != np.sign(m.eta_3d)
            rows.append({"floor": floor_name, "kirchhoff": kname, "n_cells": len(m),
                         "median_e_lin_floors": float(np.median(e_lin)),
                         "median_e_modelform_floors": float(np.median(e_mf)),
                         "median_e_total_floors": float(np.median(e_tot)),
                         "ratio_lin_over_mf": float(np.median(e_lin) / np.median(e_mf)),
                         "n_opposite_sign_K_vs_3D": int(opp.sum()),
                         "opposite_cells": ";".join(
                             f"h{r.half_xi}/xi{r.xi_c}/p{r.depth_frac}/m{int(r.m)}"
                             for r in m[opp].itertuples())})
    return pd.DataFrame(rows)


# ============================================================ 요청 7 — 건전 Ritz 기저(단항 8개)의 정확도
def _hermite_eval(nodes: np.ndarray, u_full: np.ndarray, r: np.ndarray):
    """Hermite 3차 보간으로 R, R′, R″를 임의 r에서 — u_full = (R, R′) 절점 자유도."""
    r = np.asarray(r, dtype=float)
    e = np.clip(np.searchsorted(nodes, r, side="right") - 1, 0, nodes.size - 2)
    h = nodes[e + 1] - nodes[e]
    s_ = (r - nodes[e]) / h
    ue = np.stack([u_full[2 * e], u_full[2 * e + 1], u_full[2 * e + 2], u_full[2 * e + 3]], 1)
    H = np.stack([1 - 3 * s_ ** 2 + 2 * s_ ** 3, h * (s_ - 2 * s_ ** 2 + s_ ** 3),
                  3 * s_ ** 2 - 2 * s_ ** 3, h * (-s_ ** 2 + s_ ** 3)], 1)
    dH = np.stack([(-6 * s_ + 6 * s_ ** 2) / h, 1 - 4 * s_ + 3 * s_ ** 2,
                   (6 * s_ - 6 * s_ ** 2) / h, -2 * s_ + 3 * s_ ** 2], 1)
    d2H = np.stack([(-6 + 12 * s_) / h ** 2, (-4 + 6 * s_) / h,
                    (6 - 12 * s_) / h ** 2, (-2 + 6 * s_) / h], 1)
    return (np.sum(H * ue, 1), np.sum(dH * ue, 1), np.sum(d2H * ue, 1))


def fem_mode_kernel(m: int, n: int, n_elem: int = FEM_N, r_eval=None):
    """FEM 건전 모드 → `kernels.ModeKernel`(같은 정의의 γ^K·γ^M) — 정본 1차 맵에 그대로 꽂힌다."""
    nodes = _fem_nodes(n_elem)
    fm = _FemM(m, nodes)
    K = (fm.AK @ np.ones(fm.rq.size)).reshape(fm.nf, fm.nf)
    M = (fm.AM @ np.ones(fm.rq.size)).reshape(fm.nf, fm.nf)
    d = 1.0 / np.sqrt(np.diag(M))
    _, V = eigh(K * d[:, None] * d[None, :], M * d[:, None] * d[None, :],
                subset_by_index=[0, n])
    x = d * V[:, n]
    lam = fm._rayleigh(x, np.ones(fm.rq.size), np.ones(fm.rq.size))
    u = np.zeros(fm._ndof)
    u[fm._free] = x
    r = (np.linspace(PLATE.a, PLATE.b, 2001) if r_eval is None else np.asarray(r_eval))
    R, dR, d2R = _hermite_eval(nodes, u, r)
    nu = PLATE.nu
    kr, kt, kx = d2R, dR / r - m ** 2 * R / r ** 2, m * (dR / r - R / r ** 2)
    e = (kr + kt) ** 2 - 2 * (1 - nu) * (kr * kt - kx ** 2)
    g = e * r
    g = g / np.trapezoid(g, r)
    gm = R ** 2 * r
    gm = gm / np.trapezoid(gm, r)
    f = math.sqrt(max((PLATE.D / PLATE.rhoh) * lam, 0.0)) / (2 * math.pi)
    return ker.ModeKernel(m=m, n=n, f=f, Lambda=lam, r=r, gamma=g, gamma_mass=gm,
                          R=R, dR=dR, d2R=d2R)


HEALTHY_MODES = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1))


def run_healthy_check() -> tuple:
    """(1) Λ·커널 대조, (2) FEM 커널로 Table 3a·Table R-A 레일 열 재계산."""
    rows = []
    canon = {mn: ker.mode_kernel(PLATE, m=mn[0], n=mn[1], n_grid=2001) for mn in HEALTHY_MODES}
    fem100 = {mn: fem_mode_kernel(*mn, n_elem=100) for mn in HEALTHY_MODES}
    fem400 = {mn: fem_mode_kernel(*mn, n_elem=400) for mn in HEALTHY_MODES}
    for mn in HEALTHY_MODES:
        c, a, b = canon[mn], fem100[mn], fem400[mn]
        rows.append({"mode": f"m{mn[0]}n{mn[1]}", "f_canon_Hz": c.f, "f_fem400_Hz": b.f,
                     "Lambda_rel_canon_vs_fem400": c.Lambda / b.Lambda - 1,
                     "Lambda_rel_fem100_vs_fem400": a.Lambda / b.Lambda - 1,
                     "gK_L1_canon_vs_fem400": float(np.trapezoid(np.abs(c.gamma - b.gamma), c.r)),
                     "gM_L1_canon_vs_fem400": float(np.trapezoid(np.abs(c.gamma_mass - b.gamma_mass), c.r)),
                     "gK_maxrel_canon_vs_fem400": float(np.max(np.abs(c.gamma - b.gamma)) / np.max(b.gamma)),
                     "gM_maxrel_canon_vs_fem400": float(np.max(np.abs(c.gamma_mass - b.gamma_mass)) / np.max(b.gamma_mass)),
                     "gK_L1_fem100_vs_fem400": float(np.trapezoid(np.abs(a.gamma - b.gamma), b.r))})
    kdf = pd.DataFrame(rows)

    def crlb_with(pool_map, modes, xi, s_bar):
        pool = [pool_map[mn] for mn in modes]
        J = fwd.jacobian_linear_mass(pool, xi, s_bar, W_REF, PLATE, coupling="exact")
        met = idf.metrics_from_J(J, noi.sigma_y_for_modes([m for m, _ in modes], C_REF), EXT)
        return met["crlb_xi_mm"], met["crlb_s_pp"], met["det_F"]

    def dopt(pool_map, cand, xi, s_bar):
        best = None
        for combo in combinations(cand, 4):
            cx, cs, det = crlb_with(pool_map, combo, xi, s_bar)
            if best is None or det > best[3]:
                best = (combo, cx, cs, det)
        return best

    out = []
    for xi in (0.2, 0.5, 0.8, 0.94):
        for tag, pmap in (("canonical_monomial8", canon), ("fem_n400", fem400)):
            pool20 = [mn for mn in HEALTHY_MODES if pmap[mn].f <= 2.0e4]
            pool25 = [mn for mn in HEALTHY_MODES if pmap[mn].f <= 2.56e4]
            d_cx, d_cs, _ = crlb_with(pmap, MODES_DEF, xi, 0.05)
            b20 = dopt(pmap, pool20, xi, 0.05)
            b25 = dopt(pmap, pool25, xi, 0.05)
            a20 = crlb_with(pmap, tuple(pool20), xi, 0.05)
            a25 = crlb_with(pmap, tuple(pool25), xi, 0.05)
            out.append({"xi": xi, "kernels": tag,
                        "default_crlb_xi_mm": d_cx, "default_crlb_s_pp": d_cs,
                        "dopt20_set": "|".join(f"m{m}n{n}" for m, n in b20[0]),
                        "dopt20_crlb_xi_mm": b20[1], "dopt20_crlb_s_pp": b20[2],
                        "dopt25_set": "|".join(f"m{m}n{n}" for m, n in b25[0]),
                        "dopt25_crlb_xi_mm": b25[1], "dopt25_crlb_s_pp": b25[2],
                        "all20_n": len(pool20), "all20_crlb_xi_mm": a20[0], "all20_crlb_s_pp": a20[1],
                        "all25_n": len(pool25), "all25_crlb_xi_mm": a25[0], "all25_crlb_s_pp": a25[1]})
    tdf = pd.DataFrame(out)
    # 같은 1차 맵에서 관측량 자체의 차(η̄, 바닥 단위) — 대표 셀
    eta_rows = []
    for xi in (0.2, 0.5, 0.8, 0.94):
        for s_bar in (0.01, 0.05, 0.15):
            ec = fwd.eta_bar_linear_mass([canon[mn] for mn in HEALTHY_MODES], xi, s_bar, W_REF,
                                         PLATE, coupling="exact")
            ef = fwd.eta_bar_linear_mass([fem400[mn] for mn in HEALTHY_MODES], xi, s_bar, W_REF,
                                         PLATE, coupling="exact")
            fl = np.sqrt(np.diag(noi.sigma_y_for_modes([m for m, _ in HEALTHY_MODES], C_REF)))
            for k, mn in enumerate(HEALTHY_MODES):
                eta_rows.append({"xi": xi, "s_bar": s_bar, "mode": f"m{mn[0]}n{mn[1]}",
                                 "eta_canon": ec[k], "eta_fem400": ef[k],
                                 "diff_over_floor": (ec[k] - ef[k]) / fl[k]})
    return kdf, tdf, pd.DataFrame(eta_rows)


# ============================================================ 요청 4–6 + 보충 — §3.4·§4.3 수치의 FEM 재확인
def gauss_exact_fem(xi: float, s_bar: float, w: float = W_REF, modes=MODES_DEF) -> np.ndarray:
    """가우시안 정확재해(FEM). 물리 상한(peak d_K ≥ 1) 위는 NaN."""
    prof = Profile("gauss", w)
    if prof.peak(s_bar) >= 1.0:
        return np.full(len(modes), np.nan)
    return rail(modes).eta(prof, xi, s_bar)


def run_validity_fem(n_xi: int = 41, n_s: int = 31) -> dict:
    """요청 6 — Fig. 2(§3.4 유효성 지도)를 정본 `cmd_a2`와 같은 격자·같은 정의로, 정확맵만 FEM."""
    from .. import validity as val
    from .. import cli as _cli
    pool = linear_pool(MODES_DEF)
    xi_grid = np.linspace(0.05, 0.95, n_xi)
    s_grid = np.geomspace(0.001, 0.30, n_s)
    Mabs = np.full((len(MODES_DEF), n_xi, n_s), np.nan)
    Mrel = np.full_like(Mabs, np.nan)
    for i, x in enumerate(xi_grid):
        for j, sv in enumerate(s_grid):
            ex = gauss_exact_fem(float(x), float(sv))
            lin = fwd.eta_bar_linear_mass(pool, float(x), float(sv), W_REF, PLATE,
                                          coupling="exact")
            Mabs[:, i, j] = np.abs(ex - lin)
            Mrel[:, i, j] = np.abs(ex - lin) / np.abs(ex)
    npz = DATA_PATH("rev1_a2_epert_map_mass_fem.npz")
    np.savez(npz, xi=xi_grid, s=s_grid, rel=Mrel, abs=Mabs, modes=np.array(MODES_DEF))
    rows = []
    canon = pd.read_csv(_cli.DATA / "a2_validity_contours_mass.csv")
    for c in (1e-4, 3e-4, 1e-3, 3e-3):
        thr = val.validity_contour(PLATE, pool, list(MODES_DEF), xi_grid, s_grid, W_REF, c,
                                   mass="exact", abs_map=np.nan_to_num(Mabs, nan=np.inf))
        cc = canon[np.isclose(canon.sigma_rel, c)].sort_values("xi_d")
        for x, t, tc in zip(xi_grid, thr, cc.s_bar_threshold.to_numpy()):
            rows.append({"sigma_rel": c, "xi_d": float(x), "s_bar_threshold_fem": t,
                         "s_bar_threshold_canon": tc})
    cont = pd.DataFrame(rows)
    cont.to_csv(out_csv("a2_validity_contours_mass_fem"), index=False)
    # 그림 영역(S̄ < 25.1 %)에서 정본 맵 대비 차(바닥 단위, c = 0.1 %)
    z = np.load(_cli.DATA / "a2_epert_map_mass.npz")
    keep = s_grid < Profile("gauss", W_REF).ceiling()
    fl = np.array([2e-3 * (math.sqrt(2.0) if m == 0 else 1.0) for m, _ in MODES_DEF])
    dfloor = np.abs(Mabs - z["abs"])[:, :, keep] / fl[:, None, None]
    return {"npz": str(npz), "contours": cont,
            "max_abs_map_diff_floor_c1e-3": float(np.nanmax(dfloor)),
            "median_abs_map_diff_floor_c1e-3": float(np.nanmedian(dfloor))}


def DATA_PATH(name: str):
    from . import DATA as _D
    _D.mkdir(parents=True, exist_ok=True)
    return _D / name


def run_f36_recheck() -> pd.DataFrame:
    """요청 4 — r3.9 §4.3 "Gaussian validity map understates … by a median factor of 2.2"(F36).

    정의(정본 b6 `_decomp`): 같은 (ξ_c, S̄)·같은 모드에서 밴드 e_lin,rel = |η̄^lin − η̄^K|/|η̄^K| 대
    A2 가우시안 e_pert,rel = |η̄^lin_G − η̄^ex_G|/|η̄^ex_G| (`validity.e_pert`, mass="exact")의 비, 104셀.
    """
    anc = pd.read_csv(out_csv("n4_reconcile_b6_anchors_fem"))
    m = anc[anc.matched.astype(bool)].copy()
    from .. import cli as _cli
    b6 = pd.read_csv(_cli.DATA / "b6_epert_fem.csv")
    b6 = b6[b6.matched == True]                                      # noqa: E712
    key = ["half_xi", "xi_c", "depth_frac", "m"]
    m = m.merge(b6[key + ["e_lin_rel", "e_pert_a2_gaussian_rel"]], on=key, how="left")
    pool = linear_pool(MODES_DEF)
    ms = [mm for mm, _ in MODES_DEF]
    cache = {}
    eg_fem = []
    for r in m.itertuples():
        k = (round(r.xi_c, 6), round(r.s_bar_pct, 6))
        if k not in cache:
            ex = gauss_exact_fem(float(r.xi_c), float(r.s_bar_pct) / 100)
            lin = fwd.eta_bar_linear_mass(pool, float(r.xi_c), float(r.s_bar_pct) / 100,
                                          W_REF, PLATE, coupling="exact")
            cache[k] = np.abs(ex - lin) / np.abs(ex)
        eg_fem.append(cache[k][ms.index(int(r.m))])
    m["e_pert_gauss_rel_fem"] = eg_fem
    m["e_lin_band_rel_fem"] = np.abs(m.eta_lin - m.eta_K_fem) / np.abs(m.eta_K_fem)
    m["ratio_canon"] = m.e_lin_rel / m.e_pert_a2_gaussian_rel
    m["ratio_fem"] = m.e_lin_band_rel_fem / m.e_pert_gauss_rel_fem
    m.to_csv(out_csv("n4_f36_ratio_recheck"), index=False)
    return m


def run_loo_recheck() -> pd.DataFrame:
    """요청 5 — b7 V1(ξ-LOO) "median leave-one-out error 1.5 %, 95th percentile 16 %"를 δ_FEM으로.

    정의(정본 `cmd_b7._predict_rows`): ξ 노드 하나를 빼고 나머지 앵커로 δ 격자(같은 규칙)를 만들어
    빠진 앵커를 η̄_pred = η̄_K + δ̂ 로 예측, 상대잔차 |η̄_pred − η̄_3D|/|η̄_3D|. 반폭 0.10족, 정합 모드.
    """
    from .. import montecarlo as _mc
    anc = pd.read_csv(out_csv("n4_reconcile_b6_anchors_fem"))
    fam = anc[np.isclose(anc.half_xi, HALF_B7) & anc.matched.astype(bool)].copy()
    xs = sorted(fam.xi_c.unique())
    rows = []
    for x in xs:
        keep = [v for v in xs if v != x]
        if len(keep) < 2:
            continue
        tr = fam[~np.isclose(fam.xi_c, x)]
        for tag, kcol, dcol in (("fem", "eta_K_fem", "delta_fem"),
                                ("canon", "eta_kirchhoff_exact", "delta_modelform")):
            sp = b7_delta_spec(tr, dcol)
            for (_, sp_pct), g in fam[np.isclose(fam.xi_c, x)].groupby(["xi_c", "s_bar_pct"]):
                dh = _mc.kappa_eval(sp, float(x), float(sp_pct) / 100, key="delta")
                for r in g.itertuples():
                    i = int(r.m)
                    pred = getattr(r, kcol) + dh[i]
                    rows.append({"solver": tag, "held_xi": float(x), "s_bar_pct": float(sp_pct),
                                 "m": i, "eta_pred": pred, "eta_3d": r.eta_3d,
                                 "resid_rel": abs(pred - r.eta_3d) / abs(r.eta_3d),
                                 "resid_over_floor": abs(pred - r.eta_3d) / (
                                     2e-3 * (math.sqrt(2.0) if i == 0 else 1.0))})
    d = pd.DataFrame(rows)
    d.to_csv(out_csv("n4_b7_loo_recheck"), index=False)
    return d


def run_modelform_crlb_counts() -> pd.DataFrame:
    """보충 — 정확 밴드 맵의 잔여 모델형식 편향(T_3D 진실) 대 그 맵의 국소 CRLB(진실점), 36셀.

    N5 `add_matched_ratios`와 같은 정의: CRLB는 역산 맵 자신의 야코비안(중심차분, hx = hs = 2e-3)과
    관측량별 바닥 Σ_y(c)로 진실점에서. 맵은 FEM 밴드 표(N5 BandMap 규약, (ξ, S̄) 매개화).
    """
    from . import n5_fem as nf
    from . import n5_heldout as n5
    band, _ = nf.load_band_fem()
    dec = pd.read_csv(out_csv("n4_reconcile_decomposition"))
    rows = []
    for r in dec[dec.truth == "y3D_fem"].itertuples():
        J = n5.jac_fd(band, float(r.xi_d), float(r.s_bar))
        rec = {"xi_d": r.xi_d, "s_bar": r.s_bar, "in36": bool(r.in36), "in43": bool(r.in43),
               "modelform_bias_mm": r.ex_B}
        for c in (1e-4, 3e-4, 1e-3, 3e-3):
            cx, cs = n5.crlb(J, n5.sigma_y(c))
            rec[f"crlb_xi_mm_c{c:g}"] = cx
            rec[f"ratio_c{c:g}"] = abs(r.ex_B) / cx
        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(out_csv("n4_modelform_vs_crlb_fem"), index=False)
    return d


# ============================================================ 요청 3·보충 — 원고 분해 문단의 부분집합과 수치
def nosat_fem_cells(b7: pd.DataFrame) -> pd.Series:
    """FEM b7(`rev1_n4_b7_fem.csv`)에서 어느 c에서도 경계에 닿지 않은 셀 — (ξ, S̄) → bool.

    원고가 인용하는 b7 통계가 FEM b7의 포화 판정을 쓰므로, 분해·잡음 대비 셀 수도 같은 부분집합으로 센다
    (N4 §10.2: 08-13 기하에서 35셀. 정본 b7 기준 36셀·c = 10⁻⁴만 본 43셀과 다르다).
    """
    s = b7[b7.truth_model == "3d_rail_surrogate"]
    return s.groupby([s.xi_d.round(6), s.s_bar.round(6)]).boundary_hit_prob.max() == 0


def _nosat_flag(df: pd.DataFrame, nosat: pd.Series) -> np.ndarray:
    return np.array([bool(nosat.loc[(round(float(x), 6), round(float(s), 6))])
                     for x, s in zip(df.xi_d, df.s_bar)])


def decomp_summary(dec: pd.DataFrame, mf: pd.DataFrame, nosat: pd.Series) -> dict:
    """§4.3 분해 문단의 수치 — T_3D 진실(FEM) 2×2 역맵 무잡음 편향, 비포화 부분집합.

    몫은 셀별 Shapley 값 ÷ 생산 편향(a = 선형화 + 발자국 + 모델형식)의 중앙값, 경로 몫은 부호 있는 중앙값.
    잡음 대비는 `run_modelform_crlb_counts`의 비(|모델형식 편향| / 국소 CRLB) > 1인 셀 수.
    """
    t = dec[(dec.truth == "y3D_fem") & _nosat_flag(dec, nosat)]
    k = dec[(dec.truth == "K_band_fem") & _nosat_flag(dec, nosat)]
    m = mf[_nosat_flag(mf, nosat)]
    a = t.lin_G
    cs = (1e-4, 3e-4, 1e-3, 3e-3)
    return {"n_cells": int(len(t)), "n_cells_crlb": int(len(m)),
            "abs_bias_median_mm": {c: float(t[c].abs().median()) for c in MAPS},
            "path1_median_mm": {"lin": float(t.path1_lin.median()), "foot": float(t.path1_foot.median())},
            "path2_median_mm": {"foot": float(t.path2_foot.median()), "lin": float(t.path2_lin.median())},
            "interaction_median_mm": float(t.interaction.median()),
            "shapley_share_median": {"lin": float((t.shapley_lin / a).median()),
                                     "foot": float((t.shapley_foot / a).median()),
                                     "modelform": float((t.modelform / a).median())},
            "n_modelform_below_production": int((t.ex_B.abs() < a.abs()).sum()),
            "kirchhoff_only_production_abs_median_mm": float(k.lin_G.abs().median()),
            "modelform_gt_crlb": {f"{c:g}": int((m[f"ratio_c{c:g}"] > 1).sum()) for c in cs},
            "modelform_over_crlb_median": {f"{c:g}": float(m[f"ratio_c{c:g}"].median()) for c in cs}}


def run_decomp_summary() -> dict:
    """`reconcile`·`n5_fem b7`·`mfcrlb` 뒤 — 두 CSV에 `in_nosat_fem` 열을 붙이고 요약 JSON을 쓴다."""
    nosat = nosat_fem_cells(pd.read_csv(out_csv("n4_b7_fem")))
    frames = {}
    for name in ("n4_reconcile_decomposition", "n4_modelform_vs_crlb_fem"):
        df = pd.read_csv(out_csv(name))
        df["in_nosat_fem"] = _nosat_flag(df, nosat)
        df.to_csv(out_csv(name), index=False)
        frames[name] = df
    out = decomp_summary(frames["n4_reconcile_decomposition"], frames["n4_modelform_vs_crlb_fem"], nosat)
    with open(out_csv("n4_decomposition_summary").with_suffix(".json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def make_figure() -> str:
    """N4 요약 그림 — (a) 폭고정 위치편향(정확 맵, S̄=5 %) 대 ξ, (b) 같은 셀의 잔차 χ²,
    (c) 폭 프로파일 χ²_p(w)(ξ=0.5, S̄=5 %), (d) ξ*(w). 라벨은 영문(논문용)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = pd.read_csv(out_csv("n4_bias_noiseless"))
    b = b[(b.matching == "nominal") & (b.modes == "m0n0|m1n0|m2n0|m3n0") & b.admissible]
    pr = pd.read_csv(out_csv("n4_width_profile"))
    sr = pd.read_csv(out_csv("n4_sign_reversal_vs_w"))
    idt = pd.read_csv(out_csv("n4_identifiability"))
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    for k, (lab, g) in enumerate(b[b.s_target == 0.05].groupby("profile", sort=False)):
        ax[0].plot(g.xi_true, g.A_exact_bias_xi_mm, "o-", ms=3, label=lab, color=f"C{k}")
        ax[1].semilogy(g.xi_true, np.maximum(g.A_exact_chi2, 1e-8), "o-", ms=3,
                       color=f"C{k}")
    crl = idt[(idt.set == "default_m0_3") & (idt.s_bar == 0.05) & (idt.w_mm == 3.0)]
    if len(crl):
        ax[0].plot(crl.xi, 1.96 * crl.crlb_xi_fixed_mm, "k:", lw=1)
        ax[0].plot(crl.xi, -1.96 * crl.crlb_xi_fixed_mm, "k:", lw=1)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel(r"$\xi_d$"); ax[0].set_ylabel("location bias [mm]")
    ax[0].set_title("(a)"); ax[0].legend(fontsize=7)
    ax[1].axhline(5.991, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel(r"$\xi_d$"); ax[1].set_ylabel(r"residual $\chi^2$")
    ax[1].set_title("(b)")
    g = pr[(pr.xi_true == 0.5) & (pr.s_target == 0.05) & pr.admissible
           & (pr.modes == "m0n0|m1n0|m2n0|m3n0")]
    for k, (_, row) in enumerate(g.iterrows()):
        chi = np.array(json.loads(row.chi2_profile))
        ax[2].semilogy(W_GRID * 1e3, np.maximum(chi - chi.min(), 1e-6), "-",
                       label=row.profile, color=f"C{k}")
    ax[2].axhline(3.841, color="k", ls="--", lw=0.8)
    ax[2].set_xlabel("assumed width $w$ [mm]")
    ax[2].set_ylabel(r"$\Delta\chi^2_p(w)$"); ax[2].set_title("(c)")
    for m, gg in sr.groupby("m"):
        ax[3].plot(gg.w_mm, gg.xi_star_exact_interp, "o-", ms=3, label=f"m={m}")
    ax[3].set_xlabel("width $w$ [mm]"); ax[3].set_ylabel(r"$\xi^*$")
    ax[3].set_title("(d)"); ax[3].legend(fontsize=7)
    fig.tight_layout()
    path = out_fig("n4_width_mismatch")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("item", choices=["anchors", "ident", "sign", "pocket", "bias",
                                     "profile", "mc", "all", "figure", "b7decomp",
                                     "reconcile", "b6recheck", "healthy", "validity",
                                     "f36", "loo", "mfcrlb", "decompsum"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-real", type=int, default=300)
    ap.add_argument("--matchings", nargs="+", default=["nominal", "in_domain", "peak"])
    args = ap.parse_args(argv)
    meta = {"git_head": _git_head(), "solver": "radial Hermite-cubic FEM + element-energy Rayleigh",
            "fem_n_elem": FEM_N, "c": C_REF, "w_ref_mm": W_REF * 1e3,
            "seed0": SEED0, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    items = (["anchors", "ident", "sign", "pocket", "bias", "profile", "mc"]
             if args.item == "all" else [args.item])
    for it in items:
        t0 = time.time()
        if it == "anchors":
            df = run_anchors()
            df.to_csv(out_csv("n4_anchors"), index=False)
            dc = run_canonical_dopt_check()
            dc.to_csv(out_csv("n4_canonical_dopt_noise_check"), index=False)
            print(df.to_string(), "\n", dc.to_string())
        elif it == "ident":
            run_identifiability().to_csv(out_csv("n4_identifiability"), index=False)
        elif it == "sign":
            run_sign_reversal().to_csv(out_csv("n4_sign_reversal_vs_w"), index=False)
        elif it == "pocket":
            run_pocket_width().to_csv(out_csv("n4_pocket_width_table3b"), index=False)
        elif it == "bias":
            jobs = [(asdict(p), float(x), s, mt, MODES_DEF) for mt in args.matchings
                    for p in TRUTHS for x in XI_GRID for s in S_TARGETS]
            # 확장풀(25.6 kHz)은 주 대응규약·대표 셀만
            jobs += [(asdict(p), float(x), s, "nominal", MODES_EXT) for p in TRUTHS
                     for x in (0.2, 0.5, 0.8) for s in S_TARGETS]
            rows = _map(bias_cell, jobs, args.workers)
            pd.DataFrame(rows).to_csv(out_csv("n4_bias_noiseless"), index=False)
        elif it == "profile":
            jobs = [(asdict(p), x, s, MODES_DEF) for p in TRUTHS for x in (0.2, 0.5, 0.8)
                    for s in S_TARGETS]
            jobs += [(asdict(p), x, 0.05, MODES_EXT) for p in TRUTHS for x in (0.2, 0.5, 0.8)]
            rows = _map(profile_cell, jobs, args.workers)
            pd.DataFrame(rows).to_csv(out_csv("n4_width_profile"), index=False)
        elif it == "figure":
            print(make_figure())
        elif it == "validity":
            res = run_validity_fem()
            from .. import figures as _figs
            _figs.fig_a2_epert(Path(res["npz"]), out_fig("fig2_validity_fem"), coupling="exact")
            print({k: v for k, v in res.items() if k != "contours"})
        elif it == "f36":
            run_f36_recheck()
        elif it == "loo":
            run_loo_recheck()
        elif it == "mfcrlb":
            run_modelform_crlb_counts()
        elif it == "decompsum":
            print(json.dumps(run_decomp_summary(), indent=1))
        elif it == "healthy":
            kdf, tdf, edf = run_healthy_check()
            kdf.to_csv(out_csv("n4_healthy_basis_kernels"), index=False)
            tdf.to_csv(out_csv("n4_healthy_basis_tables"), index=False)
            edf.to_csv(out_csv("n4_healthy_basis_eta"), index=False)
            print(kdf.round(6).to_string()); print(tdf.round(4).to_string())
        elif it == "reconcile":
            run_reconcile(args.workers)
        elif it == "b6recheck":
            run_b6_recheck().to_csv(out_csv("n4_reconcile_b6_recheck"), index=False)
        elif it == "b7decomp":
            run_b7_decomp(args.workers).to_csv(out_csv("n4_b7_footprint_decomposition"),
                                               index=False)
        elif it == "mc":
            cells = [Profile("gauss", 0.0015), Profile("gauss", 0.006),
                     Profile("band", 0.006), Profile("gauss", 0.003)]
            jobs = []
            k = 0
            for p in cells:
                for x in (0.2, 0.5, 0.8):
                    jobs.append((asdict(p), x, 0.05, args.n_real, SEED0 + k, C_REF))
                    k += 1
            rows = _map(mc_cell, jobs, args.workers)
            pd.DataFrame(rows).to_csv(out_csv("n4_mc_fixed_vs_free"), index=False)
        print(f"[n4] {it} done in {time.time() - t0:.0f}s", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out_csv("n4_meta").with_suffix(".json"), "w") as fh:
        json.dump({**meta, "items": items, "argv": vars(args)}, fh, indent=1)


if __name__ == "__main__":
    main()
