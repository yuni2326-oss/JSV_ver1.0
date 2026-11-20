"""N5 — b7 모델형식 편향의 분해 + held-out 불일치 보정 역산 (R2-Q4·R2-Q1·R1-P1 대응).

**무엇을 묻는가.** r3.9 §4.3의 대리 MC는 "3D 레일 대리모델"(축대칭 밴드, 반폭 0.10(b−a))이
만든 관측량을 **생산 1차 Gaussian(w = 3 mm) 맵**으로 역산해 편향 중앙 1.73 mm를 냈다(N0 §3.2).
그 편향에는 세 불일치가 섞여 있다 —

  (i)   3D ↔ Kirchhoff 모델형식          δ_m = η̄^3D − η̄^{K,exact}(같은 밴드)
  (ii)  밴드 ↔ Gaussian 발자국(형상·폭)
  (iii) 1차 섭동 선형화(S̄ = 12–15 %는 1차 유효영역 3–7 %를 크게 넘는다)

이 모듈은 같은 진실을 네 맵으로 역산해 (iii)→(ii)→(i)을 차례로 걷어낸다(과제 1), 그리고
**훈련 앵커에서만** 배운 가법 보정 g_corr = g_{K,band} + δ̂_train을 held-out 3D 앵커에
적용해 보정의 실익과 한계를 잰다(과제 2·3). 마지막으로 어느 앵커와도 겹치지 않는 3D 시험격자를
새로 풀어 외부 시험으로 쓴다(과제 4).

**정본과 다르게 구현한 것(이유를 남긴다).**
- `estimator.fit(exact=True)`는 `eta_bar_exact`를 **질량항 없이** 부른다(강성전용 재해) —
  질량 포함 정확맵이 필요하므로 여기서 WLS를 따로 쓴다.
- `forward.eta_bar_exact`의 기본 Ritz(단항 8개)는 S̄ ≥ 12 %의 국소 Gaussian에서 수렴하지 않는다
  (ξ = 0.5, S̄ = 15 %에서 m = 0 η̄가 −0.0882 대 Legendre 24개 −0.0924, c = 0.1 % 바닥 2개 분량).
  Gaussian 정확맵은 Legendre 24개(16과 1e-5 일치)로 다시 푼다.
- b7의 δ 격자 조립(`kgrid`·`fill`)은 `cmd_b7` 안의 닫힌 함수라 가져올 수 없어 같은 규칙으로
  다시 썼다 — 테스트가 b7 산출과 대조한다.

산출: `rev1_n5_*.csv`(+ 캐시 `rev1_n5_*_table.npz`), 보고서 `revision-r1/N5-heldout-correction.md`.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
from scipy.spatial import Delaunay

from .. import estimator as est
from .. import forward as fwd
from .. import geometry as geo
from .. import kernels as ker
from .. import montecarlo as mc
from .. import noise as noi
from .. import severity as sev
from . import DATA, out_csv, out_fig

PLATE = geo.DISK
MODES = [(0, 0), (1, 0), (2, 0), (3, 0)]
MS = [m for m, _ in MODES]
W_GAUSS = 0.003
HALF_REF = 0.10                      # b7 기준 손상족(밴드 반폭, ξ 단위)
EXT = PLATE.extent                   # b − a
B7_XI = np.linspace(0.2, 0.9, 15)    # b7 MC 위치(`--n-loc 15`, 앵커 ξ 범위)
B7_S = (0.12, 0.135, 0.15)           # b7 MC 심각도(앵커 범위 11.56–17.5 % 안)
SIGMA_RELS = (1e-4, 3e-4, 1e-3, 3e-3)
SEED0 = 20261006
D_MAX = 0.99                         # 밴드 d_K 상한(넘으면 벌점; d_K → 1이면 Ritz가 특이해진다)
S_MAX_GAUSS = 0.24                   # Gaussian(w = 3 mm) 정확맵 표 상한 — peak d_K = 0.955

# Gaussian 정확맵의 Ritz 설정(정본 단항 8개는 이 심각도에서 미수렴 — 모듈 문서열)
G_NTRIAL, G_NGRID = 24, 8001
# 밴드 정확맵 — 정본(`eta_bar_exact_band`) 기본값과 같다(b7 대리모델이 쓴 설정)
B_NTRIAL, B_NGRID = 36, 8001


# =============================================================== 잡음
_POOL_CACHE: dict = {}


def pool_kernels(n_grid: int = 1001):
    """생산 1차 맵의 커널 풀 — b7 MC가 쓴 n_grid = 1001과 같게 둔다."""
    if n_grid not in _POOL_CACHE:
        _POOL_CACHE[n_grid] = [ker.mode_kernel(PLATE, m=m, n=n, n_grid=n_grid)
                               for m, n in MODES]
    return _POOL_CACHE[n_grid]


def sigma_y(c: float) -> np.ndarray:
    """생산 규약 Σ_y(관측량별 바닥: m > 0 쌍평균 2c, m = 0 단일 2√2c, ρ = 0)."""
    return noi.sigma_y_for_modes(pool_kernels(), c)


def sigma_y_legacy(c: float) -> np.ndarray:
    """**옛 규약**(2026-08-15 이전) Σ_y = (2c)² I — 모든 관측량에 2c(m = 0도).

    b7(`b7_mc_surrogate.csv`, 2026-08-14 커밋)은 이 가중으로 계산됐다 — 같은 진실을 이 Σ로
    잡음 없이 적합하면 b7의 c = 0.01 % 편향이 ≤ 0.004 mm 안에서 재현된다(테스트). r3.9가
    인용한 1.73 mm·0.94·172/180·19·1.2 %는 따라서 원고 §3.5가 명시한 관측량별 바닥이 아니라
    이 가중의 결과다(설계서 §11.41 미결 ①의 실체).
    """
    return noi.sigma_y(len(MODES), c)


def floors(c: float) -> np.ndarray:
    return np.sqrt(np.diag(sigma_y(c)))


# =============================================================== 밴드 기하
def band_edges(xi: float, half: float = HALF_REF) -> tuple[float, float]:
    """밴드 [r1, r2] — 환형 경계에서 자른다(`montecarlo.surrogate_truth`와 같은 규칙)."""
    r1 = PLATE.a + (xi - half) * EXT
    r2 = PLATE.a + (xi + half) * EXT
    return max(r1, PLATE.a), min(r2, PLATE.b)


def band_width(xi: float, half: float = HALF_REF) -> float:
    r1, r2 = band_edges(xi, half)
    return (r2 - r1) / EXT


def band_dk(xi: float, s_bar: float, half: float = HALF_REF) -> float:
    """밴드 안의 강성손실 d_K = S̄ (b−a)/(r2−r1)."""
    return float(s_bar) / band_width(xi, half)


def depth_from_dk(d_k):
    """균일판 법칙 d_K = 1−(1−p)³의 역."""
    return 1.0 - (1.0 - np.asarray(d_k, dtype=float)) ** (1.0 / 3.0)


# =============================================================== 정확맵(직접)
def eta_band_direct(xi: float, p: float, half: float = HALF_REF,
                    n_trial: int = B_NTRIAL, n_grid: int = B_NGRID) -> np.ndarray:
    """밴드 정확재해(정본 `eta_bar_exact_band`) — 깊이비 p로 지정."""
    if p <= 0.0:
        return np.zeros(len(MODES))
    r1, r2 = band_edges(xi, half)
    return fwd.eta_bar_exact_band(PLATE, MODES, r1, r2, p, coupling="exact",
                                  n_trial=n_trial, n_grid=n_grid)


def eta_gauss_direct(xi: float, s_bar: float, w: float = W_GAUSS,
                     n_trial: int = G_NTRIAL, n_grid: int = G_NGRID) -> np.ndarray:
    """Gaussian 정확재해, 질량 정확결합 — Legendre 기저(정본 단항 8개 미수렴, 문서열)."""
    if s_bar <= 0.0:
        return np.zeros(len(MODES))

    def dmg(r):
        return sev.damage_field_xi(r, xi_d=xi, s_bar=s_bar, w=w, a=PLATE.a, b=PLATE.b)

    def dmg_m(r):
        return fwd.mass_field_exact(dmg(r))

    out = []
    for m, n in MODES:
        h = ker.solve_eigenvalues_props(PLATE.a, PLATE.b, PLATE.nu, m, n_modes=n + 1,
                                        n_trial=n_trial, n_grid=n_grid,
                                        precondition=True, basis="legendre")
        d = ker.solve_eigenvalues_props(PLATE.a, PLATE.b, PLATE.nu, m, n_modes=n + 1,
                                        damage=dmg, damage_mass=dmg_m,
                                        n_trial=n_trial, n_grid=n_grid,
                                        precondition=True, basis="legendre")
        out.append(d[n] / h[n] - 1.0)
    return np.array(out)


# =============================================================== 표(스플라인) 맵
def _band_worker(args):
    xi, p, half = args
    return eta_band_direct(xi, p, half)


def _gauss_worker(args):
    xi, s = args
    return eta_gauss_direct(xi, s)


class PiecewiseTable:
    """(x, y) 격자 위 모드별 3차 스플라인 — x 방향으로 **꺾임점에서 조각을 나눈다**.

    밴드는 ξ = half와 1 − half에서 환형 경계에 닿아 폭이 꺾이므로 η̄(ξ)가 그 점에서 C¹이 아니다.
    한 스플라인으로 꺾임을 건너면 그 근방에서 진동한다 — 조각마다 따로 맞춘다.
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray, vals: np.ndarray,
                 breaks: tuple = ()):
        self.xs, self.ys, self.vals = np.asarray(xs), np.asarray(ys), np.asarray(vals)
        edges = [float(self.xs[0]), *[float(b) for b in breaks], float(self.xs[-1])]
        self.pieces = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (self.xs >= lo - 1e-12) & (self.xs <= hi + 1e-12)
            xp = self.xs[sel]
            kx = min(3, len(xp) - 1)
            sp = [RectBivariateSpline(xp, self.ys, self.vals[sel, :, k], kx=kx, ky=3)
                  for k in range(self.vals.shape[2])]
            self.pieces.append((lo, hi, sp))

    def __call__(self, x: float, y: float) -> np.ndarray:
        x = float(np.clip(x, self.xs[0], self.xs[-1]))
        y = float(np.clip(y, self.ys[0], self.ys[-1]))
        for lo, hi, sp in self.pieces:
            if x <= hi + 1e-12:
                return np.array([float(s(x, y)[0, 0]) for s in sp])
        lo, hi, sp = self.pieces[-1]
        return np.array([float(s(x, y)[0, 0]) for s in sp])

    def save(self, path: Path, meta: dict):
        np.savez(path, xs=self.xs, ys=self.ys, vals=self.vals,
                 meta=json.dumps(meta))


def _pmap(fn, jobs, workers: int):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with Pool(workers) as pl:
        return pl.map(fn, jobs, chunksize=8)


def build_band_table(half: float = HALF_REF, dxi: float = 0.01, dp: float = 0.02,
                     p_max: float = 0.80, workers: int = 4,
                     save: bool = True) -> PiecewiseTable:
    """밴드 정확맵 표 η̄(ξ, p). ξ 격자는 꺾임점 half·1−half를 **노드로 포함**해야 한다."""
    xs = np.round(np.arange(0.0, 1.0 + 1e-9, dxi), 10)
    for b in (half, 1.0 - half):
        if not np.isclose(xs, b).any():
            raise ValueError(f"ξ 격자가 꺾임점 {b}를 포함하지 않는다(dxi={dxi})")
    ps = np.round(np.arange(0.0, p_max + 1e-9, dp), 10)
    jobs = [(float(x), float(p), half) for x in xs for p in ps]
    t0 = time.time()
    vals = np.array(_pmap(_band_worker, jobs, workers)).reshape(len(xs), len(ps), len(MODES))
    tab = PiecewiseTable(xs, ps, vals, breaks=(half, 1.0 - half))
    meta = {"kind": "band", "half": half, "n_trial": B_NTRIAL, "n_grid": B_NGRID,
            "dxi": dxi, "dp": dp, "p_max": p_max, "build_s": time.time() - t0}
    if save:
        tab.save(DATA / f"rev1_n5_band_table_h{half:.2f}.npz", meta)
    return tab


def build_gauss_table(dxi: float = 0.01, ds: float = 0.005,
                      s_max: float = S_MAX_GAUSS, workers: int = 4,
                      save: bool = True) -> PiecewiseTable:
    xs = np.round(np.arange(0.0, 1.0 + 1e-9, dxi), 10)
    ss = np.round(np.arange(0.0, s_max + 1e-9, ds), 10)
    jobs = [(float(x), float(s)) for x in xs for s in ss]
    t0 = time.time()
    vals = np.array(_pmap(_gauss_worker, jobs, workers)).reshape(len(xs), len(ss), len(MODES))
    tab = PiecewiseTable(xs, ss, vals)
    meta = {"kind": "gauss", "w": W_GAUSS, "n_trial": G_NTRIAL, "n_grid": G_NGRID,
            "basis": "legendre", "dxi": dxi, "ds": ds, "s_max": s_max,
            "build_s": time.time() - t0}
    if save:
        tab.save(DATA / "rev1_n5_gauss_table.npz", meta)
    return tab


def load_table(kind: str, half: float = HALF_REF) -> PiecewiseTable:
    path = (DATA / f"rev1_n5_band_table_h{half:.2f}.npz" if kind == "band"
            else DATA / "rev1_n5_gauss_table.npz")
    if not path.exists():
        raise SystemExit(f"{path} 없음 — 먼저 `table` 하위명령을 실행할 것")
    z = np.load(path, allow_pickle=False)
    breaks = (half, 1.0 - half) if kind == "band" else ()
    return PiecewiseTable(z["xs"], z["ys"], z["vals"], breaks=breaks)


# =============================================================== 맵(θ = (ξ, S̄))
class BandMap:
    """밴드(반폭 `half`) Kirchhoff 정확맵 g(ξ, S̄) — 표 보간. 부적격(d_K > D_MAX)은 벌점."""

    def __init__(self, table: PiecewiseTable, half: float = HALF_REF):
        self.t, self.half = table, half

    def penalty(self, xi: float, s: float) -> float:
        return max(0.0, band_dk(xi, s, self.half) - D_MAX)

    def __call__(self, xi: float, s: float) -> np.ndarray:
        dk = min(band_dk(xi, max(s, 0.0), self.half), D_MAX)
        return self.t(xi, float(depth_from_dk(dk)))


class CorrectedMap:
    """g_corr = g_{K,band} + δ̂_train(ξ, S̄) — δ̂는 훈련 앵커 격자의 이중선형 보간(밖은 클램프)."""

    def __init__(self, band: BandMap, spec: dict):
        self.band, self.spec = band, spec

    def penalty(self, xi, s):
        return self.band.penalty(xi, s)

    def __call__(self, xi, s):
        return self.band(xi, s) + mc.kappa_eval(self.spec, xi, max(s, 0.0), key="delta")


class GaussExactMap:
    def __init__(self, table: PiecewiseTable):
        self.t = table

    def penalty(self, xi, s):
        return max(0.0, s - S_MAX_GAUSS)

    def __call__(self, xi, s):
        return self.t(xi, min(max(s, 0.0), S_MAX_GAUSS))


class LinearGaussMap:
    """생산 1차 Gaussian 맵(질량 정확결합) — `estimator.fit(mass="exact")`가 쓰는 것과 같다."""

    def __init__(self, pool=None):
        self.pool = pool or pool_kernels()

    def penalty(self, xi, s):
        return 0.0

    def __call__(self, xi, s):
        return fwd.eta_bar_linear_mass(self.pool, xi, s, W_GAUSS, PLATE, coupling="exact")


class LinearBandMap:
    """1차 섭동 밴드 맵(참 발자국) — 분해 경로 2(선형화를 발자국보다 나중에 걷는 순서)."""

    def __init__(self, pool=None, half: float = HALF_REF):
        self.pool, self.half = pool or pool_kernels(), half

    def penalty(self, xi, s):
        return max(0.0, band_dk(xi, s, self.half) - D_MAX)

    def __call__(self, xi, s):
        r1, r2 = band_edges(xi, self.half)
        dk = min(band_dk(xi, max(s, 0.0), self.half), D_MAX)
        return fwd.eta_bar_linear_band(self.pool, r1, r2, float(depth_from_dk(dk)), PLATE,
                                       coupling="exact")


# =============================================================== WLS
DEFAULT_STARTS = [(x, s) for x in (0.05, 0.35, 0.65, 0.95) for s in (0.05, 0.15)]


def _grid_starts(resid, s_hi: float, n_xi: int = 101, n_s: int = 61, keep: int = 3):
    """(ξ, S̄) 조밀격자에서 목적함수가 가장 작은 `keep`개 점 — 전역 탐색용 출발점.

    보정맵은 δ̂가 구간선형이라 꺾임이 있고, 미보정맵은 표현할 수 없는 자료에서 멀리 떨어진
    극소점을 여럿 가질 수 있다. 국소 다중시작만으로는 그 중 하나에 멈출 수 있으므로(실측:
    자기일관 대조군에서 0.083 mm), 잡음 없는 결정론 적합에는 이 격자를 먼저 훑는다.
    """
    xs = np.linspace(0.0, 1.0, n_xi)
    ss = np.linspace(1e-4, s_hi, n_s)
    vals = np.array([[float(np.sum(resid((x, sv)) ** 2)) for sv in ss] for x in xs])
    flat = np.argsort(vals, axis=None)[:keep]
    return [(float(xs[i // n_s]), float(ss[i % n_s])) for i in flat]


def wls_fit(y: np.ndarray, fmap, sigma: np.ndarray, starts=None,
            s_hi: float = 0.30, pen_weight: float = 1e4, grid: bool = False) -> dict:
    """가중 비선형 최소제곱 θ̂ = argmin ‖Σ^{-1/2}(g(θ) − y)‖² (+ 부적격 벌점), 다중시작.

    결과에 `boundary_xi`(ξ̂가 0/1), `boundary_s`(S̄ 경계), `penalty`(최종점의 부적격 정도)를
    그대로 남긴다 — 경계 포화나 부적격 최종점을 성공으로 섞지 않기 위해서다(계획서 N0).
    `grid=True`면 조밀격자의 최선 3점을 출발점에 더한다(결정론 적합의 전역성 보강).
    """
    W = noi.whitening(sigma)
    y = np.asarray(y, dtype=float)

    def resid(p):
        g = fmap(float(p[0]), float(p[1]))
        return np.concatenate([W @ (g - y), [pen_weight * fmap.penalty(float(p[0]),
                                                                        float(p[1]))]])

    starts = list(starts or DEFAULT_STARTS)
    if grid:
        starts += _grid_starts(resid, s_hi)
    best = None
    for x0 in starts:
        sol = least_squares(resid, np.array(x0, dtype=float),
                            bounds=([0.0, 0.0], [1.0, s_hi]), xtol=1e-12, ftol=1e-12,
                            gtol=1e-12, max_nfev=400)
        r = resid(sol.x)
        c2 = float(r[:-1] @ r[:-1])
        tot = float(r @ r)
        if best is None or tot < best["_tot"]:
            best = {"xi": float(sol.x[0]), "s": float(sol.x[1]), "chi2": c2, "_tot": tot,
                    "penalty": float(fmap.penalty(float(sol.x[0]), float(sol.x[1])))}
    tol = 1e-6
    best["boundary_xi"] = bool(best["xi"] < tol or best["xi"] > 1.0 - tol)
    best["boundary_s"] = bool(best["s"] < tol or best["s"] > s_hi - tol)
    best["n_starts"] = len(starts)
    best.pop("_tot")
    return best


def jac_fd(fmap, xi: float, s: float, hx: float = 2e-3, hs: float = 2e-3) -> np.ndarray:
    """중심차분 야코비안 (n_obs, 2) — 표 보간은 매끄럽고, δ̂는 구간선형이라 보폭을 크게 둔다."""
    xa, xb = max(xi - hx, 0.0), min(xi + hx, 1.0)
    sa, sb = max(s - hs, 1e-6), s + hs
    cx = (fmap(xb, s) - fmap(xa, s)) / (xb - xa)
    cs = (fmap(xi, sb) - fmap(xi, sa)) / (sb - sa)
    return np.stack([cx, cs], axis=1)


def crlb(J: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """국소 CRLB 표준편차 (위치 mm, 심각도 %p)."""
    F = J.T @ np.linalg.inv(sigma) @ J
    Fi = np.linalg.inv(F)
    return (float(math.sqrt(max(Fi[0, 0], 0.0)) * EXT * 1e3),
            float(math.sqrt(max(Fi[1, 1], 0.0)) * 100.0))


# =============================================================== 앵커·δ 격자
def load_anchors(half: float | None = HALF_REF) -> pd.DataFrame:
    """b6 3D 밴드 앵커(MAC 통과분) — 앵커 1개 = (반폭, ξ_c, 깊이) 한 줄, 모드별 열."""
    d = pd.read_csv(DATA / "b6_epert_fem.csv")
    d = d[d["matched"] == True].copy()                         # noqa: E712
    d["s_key"] = d["s_bar_pct"].round(2)
    if half is not None:
        d = d[np.isclose(d.half_xi, half)]
    rows = []
    for (h, x, p), g in d.groupby(["half_xi", "xi_c", "depth_frac"]):
        rec = {"half_xi": float(h), "xi_c": float(x), "depth_frac": float(p),
               "s_bar": float(g["s_bar_pct"].iloc[0]) / 100.0,
               "s_key": float(g["s_key"].iloc[0]),
               "r1_mm": float(g["r1_mm"].iloc[0]), "r2_mm": float(g["r2_mm"].iloc[0]),
               "n_modes_matched": int(len(g))}
        for m in MS:
            gm = g[g.m == m]
            rec[f"eta3d_m{m}"] = float(gm["eta_3d"].iloc[0]) if len(gm) else np.nan
            rec[f"etak_m{m}"] = (float(gm["eta_kirchhoff_exact"].iloc[0]) if len(gm)
                                 else np.nan)
            rec[f"delta_m{m}"] = (float(gm["delta_modelform"].iloc[0]) if len(gm)
                                  else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["half_xi", "s_key", "xi_c"]).reset_index(drop=True)


def _grid_from(train: pd.DataFrame, xi_nodes, s_nodes) -> np.ndarray:
    """(n_modes, n_xi, n_s) δ 격자 — 훈련 앵커만, 빈 칸은 NaN(`cmd_b7.kgrid`와 같은 규칙)."""
    K = np.full((len(MS), len(xi_nodes), len(s_nodes)), np.nan)
    for _, r in train.iterrows():
        j = int(np.argmin(np.abs(np.asarray(xi_nodes) - r.xi_c)))
        k = int(np.argmin(np.abs(np.asarray(s_nodes) - r.s_key)))
        if not (np.isclose(xi_nodes[j], r.xi_c) and np.isclose(s_nodes[k], r.s_key)):
            continue
        for i, m in enumerate(MS):
            K[i, j, k] = r[f"delta_m{m}"]
    return K


def _fill(K: np.ndarray, xi_nodes, s_nodes) -> np.ndarray:
    """결측을 ξ 방향 → S̄ 방향 순으로 메운다(밖은 클램프) — `cmd_b7.fill`과 같은 규칙."""
    K = K.copy()
    xs, ss = np.asarray(xi_nodes, float), np.asarray(s_nodes, float)
    for i in range(K.shape[0]):
        for k in range(K.shape[2]):
            col, ok = K[i, :, k], np.isfinite(K[i, :, k])
            if 0 < ok.sum() < len(xs) and ok.sum() >= 2:
                K[i, :, k] = np.interp(xs, xs[ok], col[ok])
        for j in range(K.shape[1]):
            row, ok = K[i, j, :], np.isfinite(K[i, j, :])
            if 0 < ok.sum() < len(ss) and ok.sum() >= 1:
                K[i, j, :] = np.interp(ss, ss[ok], row[ok]) if ok.sum() >= 2 else row[ok][0]
    return K


def delta_spec(train: pd.DataFrame, half: float = HALF_REF) -> dict:
    """훈련 앵커 → `montecarlo.kappa_eval`용 δ 사양. 노드는 **훈련 앵커의** ξ·S̄만 쓴다."""
    xi_nodes = sorted(float(x) for x in train["xi_c"].unique())
    s_nodes = sorted(float(s) for s in train["s_key"].unique())
    K = _fill(_grid_from(train, xi_nodes, s_nodes), xi_nodes, s_nodes)
    if np.isnan(K).any():
        raise ValueError("훈련 격자에 메울 수 없는 빈 칸이 있다")
    return {"xi": xi_nodes, "s_bar": [s / 100 for s in s_nodes], "delta": K.tolist(),
            "half_xi": half}


def inside_hull(train: pd.DataFrame, xi: float, s_key: float) -> bool:
    """held-out 점이 훈련 (ξ, S̄) 점들의 **닫힌 볼록껍질** 안인가(경계 포함)."""
    pts = train[["xi_c", "s_key"]].to_numpy(float)
    pts = pts / np.array([1.0, 100.0])                 # S̄를 분율로(척도만 맞춘다)
    q = np.array([[xi, s_key / 100.0]])
    if len(np.unique(pts[:, 1])) < 2 or len(np.unique(pts[:, 0])) < 2:
        return False
    try:
        tri = Delaunay(pts)
        if tri.find_simplex(q, tol=1e-9)[0] >= 0:
            return True
    except Exception:
        pass
    # 직사각 격자의 한 행 위 점은 Delaunay가 경계에서 놓칠 수 있다 — 같은 행 사이인지 직접 본다
    same = pts[np.isclose(pts[:, 1], q[0, 1])]
    return bool(len(same) >= 2 and same[:, 0].min() - 1e-12 <= xi <= same[:, 0].max() + 1e-12)


def make_folds(anchors: pd.DataFrame) -> list[dict]:
    """(i) 앵커 1개 제외(모드 벡터 통째로), (ii) 반경 위치 1개 제외(그 ξ의 모든 심각도)."""
    folds = []
    for idx, r in anchors.iterrows():
        tr = anchors.drop(index=idx)
        folds.append({"fold": "leave_one_anchor", "held": [idx],
                      "desc": f"xi={r.xi_c:.2f},S={r.s_key:.2f}", "train": tr})
    for x in sorted(anchors.xi_c.unique()):
        held = anchors.index[np.isclose(anchors.xi_c, x)].tolist()
        tr = anchors.drop(index=held)
        folds.append({"fold": "leave_one_xi", "held": held, "desc": f"xi={x:.2f}",
                      "train": tr})
    return folds


# =============================================================== 과제 1: 분해
def run_decomposition(band: BandMap, gauss: GaussExactMap, spec_all: dict) -> pd.DataFrame:
    """b7 셀(15 ξ × 3 S̄)의 **잡음 없는** 진실 y = g_{K,band} + δ̂_all을 네 맵으로 역산.

    Σ_y ∝ c²Σ₀이므로 잡음 없는 WLS 최적점은 c와 무관하다(c = 0.1 % 가중을 쓴다).
    경로 1: (a) 생산 1차 Gaussian → (b) 정확 Gaussian → (c) 정확 밴드 = 순수 모델형식.
    경로 2: (a) → (a′) 1차 밴드 → (c) — 순서를 바꾸면 몫이 달라지는가(경로의존성)를 본다.
    대조군: (d) y = g_{K,band}(θ)를 (c)로, (e) y = 1차 Gaussian(θ)를 (a)로 — 자기일관.
    """
    lin, linb = LinearGaussMap(), LinearBandMap()
    sig = sigma_y(1e-3)
    sig_old = sigma_y_legacy(1e-3)
    pool = pool_kernels()
    rows = []
    for xi in B7_XI:
        for s in B7_S:
            y_k = band(xi, s)
            y = y_k + mc.kappa_eval(spec_all, xi, s, key="delta")
            rec = {"xi_true": float(xi), "s_true": float(s)}
            # (a) 생산 경로 그대로(`estimator.fit`, b7과 같은 n_starts = 4) — 현행 Σ_y
            fa = est.fit(y, pool, PLATE, sig, w=W_GAUSS, mass="exact", n_starts=4)
            rec.update(_fit_cols("a_lin_gauss", fa["xi_d"], fa["s_bar"], xi, s,
                                 lin, y, sig, fa["boundary_hit"]))
            # (a_legacy) 같은 적합을 b7이 실제로 쓴 옛 균일 2c 가중으로 — r3.9 수치와의 연결
            fo = est.fit(y, pool, PLATE, sig_old, w=W_GAUSS, mass="exact", n_starts=4)
            rec.update(_fit_cols("a_legacy", fo["xi_d"], fo["s_bar"], xi, s,
                                 lin, y, sig_old, fo["boundary_hit"]))
            for tag, fmap in (("ap_lin_band", linb), ("b_exact_gauss", gauss),
                              ("c_exact_band", band)):
                f = wls_fit(y, fmap, sig, s_hi=(S_MAX_GAUSS if tag.startswith("b_")
                                                else 0.30), grid=True)
                rec.update(_fit_cols(tag, f["xi"], f["s"], xi, s, fmap, y, sig,
                                     f["boundary_xi"] or f["boundary_s"],
                                     penalty=f["penalty"]))
            fd = wls_fit(y_k, band, sig, grid=True)
            rec.update(_fit_cols("d_ctrl_band", fd["xi"], fd["s"], xi, s, band, y_k, sig,
                                 fd["boundary_xi"] or fd["boundary_s"]))
            y_l = lin(xi, s)
            fe = est.fit(y_l, pool, PLATE, sig, w=W_GAUSS, mass="exact", n_starts=4)
            rec.update(_fit_cols("e_ctrl_lin", fe["xi_d"], fe["s_bar"], xi, s, lin, y_l,
                                 sig, fe["boundary_hit"]))
            rows.append(rec)
    d = pd.DataFrame(rows)
    # 단계별 몫(부호 있는 mm) — 경로 1과 경로 2
    d["lin_share_mm"] = d["a_lin_gauss_bias_xi_mm"] - d["b_exact_gauss_bias_xi_mm"]
    d["footprint_share_mm"] = d["b_exact_gauss_bias_xi_mm"] - d["c_exact_band_bias_xi_mm"]
    d["modelform_share_mm"] = d["c_exact_band_bias_xi_mm"]
    d["footprint_share_path2_mm"] = d["a_lin_gauss_bias_xi_mm"] - d["ap_lin_band_bias_xi_mm"]
    d["lin_share_path2_mm"] = d["ap_lin_band_bias_xi_mm"] - d["c_exact_band_bias_xi_mm"]
    # 두 경로의 평균 = 두 요인 Shapley 값 — 순서 의존을 없앤 몫. 셀마다
    # lin_shapley + footprint_shapley + modelform = (a) 편향이 정확히 성립한다.
    d["lin_shapley_mm"] = 0.5 * (d["lin_share_mm"] + d["lin_share_path2_mm"])
    d["footprint_shapley_mm"] = 0.5 * (d["footprint_share_mm"] + d["footprint_share_path2_mm"])
    return d


def add_matched_ratios(d: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """같은 셀·같은 맵의 |결정론 편향| / 국소 CRLB(진실점, 그 맵의 야코비안) — c마다.

    받은 계획서 §2.1의 대체 방법: 서로 다른 셀·손상족의 중앙값끼리 나누지 않고, 셀별 비를
    먼저 만든 뒤 그 분포를 본다. CRLB는 역산에 쓰는 맵 자신의 정보로 계산한다(그 맵이
    옳다고 가정할 때의 통계적 하한 — 모델형식 오차의 상한이 아니다).
    """
    d = d.copy()
    for tag, fmap in maps.items():
        for c in SIGMA_RELS:
            sig = sigma_y(c)
            cr = []
            for _, r in d.iterrows():
                J = jac_fd(fmap, float(r.xi_true), float(r.s_true))
                cr.append(crlb(J, sig)[0])
            d[f"{tag}_crlb_xi_mm_c{c:g}"] = cr
            d[f"{tag}_ratio_c{c:g}"] = d[f"{tag}_bias_xi_mm"].abs() / d[f"{tag}_crlb_xi_mm_c{c:g}"]
    return d


def _fit_cols(tag, xi_hat, s_hat, xi, s, fmap, y, sig, boundary, penalty=0.0) -> dict:
    r = noi.whitening(sig) @ (fmap(xi_hat, s_hat) - y)
    return {f"{tag}_xi_hat": xi_hat, f"{tag}_s_hat": s_hat,
            f"{tag}_bias_xi_mm": (xi_hat - xi) * EXT * 1e3,
            f"{tag}_bias_s_pp": (s_hat - s) * 100.0,
            f"{tag}_chi2_c1e3": float(r @ r), f"{tag}_boundary": bool(boundary),
            f"{tag}_penalty": float(penalty)}


def join_b7(d: pd.DataFrame) -> pd.DataFrame:
    """b7 MC(대리모델 진실, 생산 맵)의 셀별 편향을 붙인다 — 잡음 없는 (a)가 그것을 재현하는가."""
    b7 = pd.read_csv(DATA / "b7_mc_surrogate.csv")
    b7 = b7[b7.truth_model == "3d_rail_surrogate"].copy()
    b7["kx"], b7["ks"] = b7.xi_d.round(6), b7.s_bar.round(6)
    d = d.copy()
    d["kx"], d["ks"] = d.xi_true.round(6), d.s_true.round(6)
    sat = (b7.groupby(["kx", "ks"])["boundary_hit_prob"].max() > 0).rename("b7_saturated")
    for c in SIGMA_RELS:
        g = b7[np.isclose(b7.sigma_rel, c)].set_index(["kx", "ks"])
        d = d.merge(g[["bias_xi_mm"]].rename(columns={"bias_xi_mm": f"b7_bias_xi_mm_c{c:g}"}),
                    left_on=["kx", "ks"], right_index=True, how="left")
    d = d.merge(sat, left_on=["kx", "ks"], right_index=True, how="left")
    if d[[f"b7_bias_xi_mm_c{c:g}" for c in SIGMA_RELS]].isna().any().any():
        raise RuntimeError("b7 셀과 맞물리지 않는 행이 있다")
    return d.drop(columns=["kx", "ks"])


# =============================================================== 과제 2: held-out(잡음 없음)
def heldout_noiseless(anchors: pd.DataFrame, band: BandMap, c_list=(1e-3, 3e-4)) -> pd.DataFrame:
    """held-out 3D 앵커 자체(참 3D 값)를 역산 — 미보정(c)·보정·생산(a)·자기일관 대조."""
    lin = LinearGaussMap()
    pool = pool_kernels()
    sig = sigma_y(1e-3)
    rows = []
    for fo in make_folds(anchors):
        spec = delta_spec(fo["train"])
        corr = CorrectedMap(band, spec)
        for idx in fo["held"]:
            r = anchors.loc[idx]
            xi, s = float(r.xi_c), float(r.s_bar)
            y3d = np.array([r[f"eta3d_m{m}"] for m in MS])
            if np.isnan(y3d).any():
                continue
            rec = {"fold": fo["fold"], "held_desc": fo["desc"], "xi_true": xi,
                   "s_true": s, "s_key": float(r.s_key), "depth_frac": float(r.depth_frac),
                   "inside_hull": inside_hull(fo["train"], xi, float(r.s_key)),
                   "n_train": int(len(fo["train"]))}
            # 순방향 예측오차(절대·바닥 정규화) — 진실 위치에서
            e_unc = y3d - band(xi, s)
            e_cor = y3d - corr(xi, s)
            for c in c_list:
                fl = floors(c)
                rec[f"fwd_unc_floor_max_c{c:g}"] = float(np.max(np.abs(e_unc) / fl))
                rec[f"fwd_cor_floor_max_c{c:g}"] = float(np.max(np.abs(e_cor) / fl))
            for i, m in enumerate(MS):
                rec[f"fwd_unc_abs_m{m}"] = float(e_unc[i])
                rec[f"fwd_cor_abs_m{m}"] = float(e_cor[i])
            # 역산
            fa = est.fit(y3d, pool, PLATE, sig, w=W_GAUSS, mass="exact", n_starts=4)
            rec.update(_fit_cols("prod_lin_gauss", fa["xi_d"], fa["s_bar"], xi, s, lin, y3d,
                                 sig, fa["boundary_hit"]))
            for tag, fmap in (("unc_band", band), ("cor_band", corr)):
                f = wls_fit(y3d, fmap, sig, grid=True)
                rec.update(_fit_cols(tag, f["xi"], f["s"], xi, s, fmap, y3d, sig,
                                     f["boundary_xi"] or f["boundary_s"], f["penalty"]))
            # 자기일관 대조: 보정맵이 만든 자료를 보정맵으로
            yc = corr(xi, s)
            fc = wls_fit(yc, corr, sig, grid=True)
            rec.update(_fit_cols("ctrl_cor", fc["xi"], fc["s"], xi, s, corr, yc, sig,
                                 fc["boundary_xi"] or fc["boundary_s"], fc["penalty"]))
            rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================== 과제 3: held-out MC
def nested_loo_cov(train: pd.DataFrame, band: BandMap) -> np.ndarray:
    """훈련 앵커 **안에서만** 한 개씩 빼 δ를 예측한 잔차 공분산 Σ_δ(모드×모드).

    held-out 앵커는 쓰지 않는다(누설 금지). 훈련 껍질 **안**의 점만 쓴다 — 외삽 잔차는
    보간 불확도가 아니다. 가정: 잔차가 평균 0이고 측정잡음과 독립(계획서 N5 uncertainty).
    """
    res = []
    for idx, r in train.iterrows():
        tr2 = train.drop(index=idx)
        if not inside_hull(tr2, float(r.xi_c), float(r.s_key)):
            continue
        try:
            sp = delta_spec(tr2)
        except ValueError:
            continue
        dh = mc.kappa_eval(sp, float(r.xi_c), float(r.s_bar), key="delta")
        res.append(np.array([r[f"delta_m{m}"] for m in MS]) - dh)
    R = np.array(res)
    if len(R) < 3:
        return np.zeros((len(MS), len(MS)))
    return np.cov(R.T, ddof=1)


def _mc_worker(job):
    (key, y3d, xi, s, c, seed, n_real, spec, sig_delta, half, band_npz) = job
    tab = PiecewiseTable(band_npz["xs"], band_npz["ys"], band_npz["vals"],
                         breaks=(half, 1.0 - half))
    band = BandMap(tab, half)
    corr = CorrectedMap(band, spec)
    sig = sigma_y(c)
    rng = np.random.default_rng(seed)
    eps = noi.sample(sig, rng, size=n_real)
    out = []
    for tag, fmap, sg in (("unc_band", band, sig), ("cor_band", corr, sig),
                          ("cor_band_inflated", corr, sig + sig_delta)):
        # 잡음 없는 최적점에서 출발(같은 시드·같은 출발점으로 짝지은 비교)
        f0 = wls_fit(y3d, fmap, sg, grid=True)
        st = [(f0["xi"], f0["s"]), (xi, s)]
        J = jac_fd(fmap, xi, s)
        cr_xi, cr_s = crlb(J, sg)
        xs, ss, hits = np.empty(n_real), np.empty(n_real), 0
        for k in range(n_real):
            f = wls_fit(y3d + eps[k], fmap, sg, starts=st)
            xs[k], ss[k] = f["xi"], f["s"]
            hits += int(f["boundary_xi"])
        err_x = (xs - xi) * EXT * 1e3
        err_s = (ss - s) * 100.0
        out.append({**key, "c": c, "map": tag, "seed": seed, "n_real": n_real,
                    "det_bias_xi_mm": (f0["xi"] - xi) * EXT * 1e3,
                    "det_bias_s_pp": (f0["s"] - s) * 100.0,
                    "bias_xi_mm": float(err_x.mean()), "std_xi_mm": float(err_x.std(ddof=1)),
                    "rmse_xi_mm": float(np.sqrt(np.mean(err_x ** 2))),
                    "bias_s_pp": float(err_s.mean()), "std_s_pp": float(err_s.std(ddof=1)),
                    "rmse_s_pp": float(np.sqrt(np.mean(err_s ** 2))),
                    "crlb_xi_mm": cr_xi, "crlb_s_pp": cr_s,
                    "coverage95_xi": float(np.mean(np.abs(err_x) <= 1.96 * cr_xi)),
                    "coverage95_s": float(np.mean(np.abs(err_s) <= 1.96 * cr_s)),
                    "boundary_hit_prob": hits / n_real})
    return out


def heldout_mc(anchors: pd.DataFrame, band: BandMap, half: float = HALF_REF,
               c_list=(1e-3, 3e-4), n_real: int = 300, workers: int = 4,
               interior_only: bool = True) -> pd.DataFrame:
    z = np.load(DATA / f"rev1_n5_band_table_h{half:.2f}.npz")
    band_npz = {"xs": z["xs"], "ys": z["ys"], "vals": z["vals"]}
    jobs = []
    k = 0
    for fo in make_folds(anchors):
        spec = delta_spec(fo["train"])
        sig_delta = nested_loo_cov(fo["train"], band)
        for idx in fo["held"]:
            r = anchors.loc[idx]
            xi, s = float(r.xi_c), float(r.s_bar)
            hull = inside_hull(fo["train"], xi, float(r.s_key))
            if interior_only and not hull:
                continue
            y3d = np.array([r[f"eta3d_m{m}"] for m in MS])
            key = {"fold": fo["fold"], "held_desc": fo["desc"], "xi_true": xi,
                   "s_true": s, "inside_hull": hull,
                   "sig_delta_diag": "|".join(f"{v:.3e}" for v in np.diag(sig_delta))}
            for ci, c in enumerate(c_list):
                # 같은 앵커·같은 c에는 fold 종류와 무관하게 같은 시드(짝지은 비교)
                seed = SEED0 + 1000 * int(round(xi * 100)) + 10 * int(round(s * 1000)) + ci
                jobs.append((key, y3d, xi, s, c, seed, n_real, spec, sig_delta, half,
                             band_npz))
                k += 1
    with Pool(workers) as pl:
        res = pl.map(_mc_worker, jobs, chunksize=1)
    return pd.DataFrame([r for rr in res for r in rr])


# =============================================================== 과제 4: 외부 3D 시험격자
EXT_XI = (0.25, 0.45, 0.65, 0.85)
EXT_DEPTH = (0.25, 0.5)


def run_ext3d(nr: int = 20, nt: int = 48, nz: int = 4) -> pd.DataFrame:
    """b6와 **같은 설정**(nr 20 · nθ 48 · nz 4, 2차, 내측 클램프)으로 앵커와 겹치지 않는 ξ의
    3D 밴드를 새로 푼다. 밴드 경계 ξ_c ± 0.10이 셀면(0.05 간격)에 정확히 놓인다.

    정본 `cli._band_anchor_rows`를 **읽기 전용으로** 재사용한다(같은 짝짓기·같은 Kirchhoff 열).
    """
    from .. import cli
    from .. import rail3d as r3
    pool = cli._pool()
    c0, cn0, _ = r3.disk_mesh(a=PLATE.a, b=PLATE.b, t=PLATE.t, nr=nr, ntheta=nt, nz=nz)
    res0 = r3.solve_modes(c0, cn0, PLATE.E, PLATE.nu, PLATE.rho,
                          r3.clamp_inner_rim(PLATE.a), n_modes=12, order=2,
                          keep_shapes=True)
    lam0 = (2 * np.pi * res0.freqs) ** 2
    h0 = {m: r3.match_order(res0, m, n_take=(1 if m == 0 else 2)) for m, _ in MODES}
    print(f"[n5 ext3d] healthy ndof={res0.ndof} f={np.round(res0.freqs[:6], 1)}", flush=True)
    rows = []
    for xi_c in EXT_XI:
        for depth in EXT_DEPTH:
            t0 = time.time()
            rs, eta3d, kappa, s_bar = cli._band_anchor_rows(
                res0, lam0, h0, r3, nr, nt, nz, xi_c, HALF_REF, depth, pool, 44,
                n_trial=36)
            for r in rs:
                r["source"] = "rev1_n5_ext3d"
            rows += rs
            print(f"[n5 ext3d] ξ_c={xi_c} p={depth} S̄={s_bar*100:.3f}% "
                  f"η3D={np.round(eta3d, 5)} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_csv("n5_ext3d_anchors"), index=False)
    return pd.DataFrame(rows)


def external_test(anchors: pd.DataFrame, band: BandMap, c_list=(1e-3, 3e-4),
                  n_real: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전 기준 앵커(16개)로 훈련한 보정맵을 외부 3D 시험격자에 — 누설이 원리적으로 없다."""
    src = out_csv("n5_ext3d_anchors")
    if not src.exists():
        raise SystemExit(f"{src} 없음 — 먼저 `ext3d`를 실행할 것")
    e = pd.read_csv(src)
    e = e[e["matched"] == True].copy()                          # noqa: E712
    e["s_key"] = e["s_bar_pct"].round(2)
    ext_rows = []
    for (x, p), g in e.groupby(["xi_c", "depth_frac"]):
        rec = {"xi_c": float(x), "depth_frac": float(p),
               "s_bar": float(g["s_bar_pct"].iloc[0]) / 100.0,
               "s_key": float(g["s_key"].iloc[0]), "n_modes_matched": int(len(g))}
        for m in MS:
            gm = g[g.m == m]
            rec[f"eta3d_m{m}"] = float(gm["eta_3d"].iloc[0]) if len(gm) else np.nan
            rec[f"delta_m{m}"] = (float(gm["delta_modelform"].iloc[0]) if len(gm)
                                  else np.nan)
        ext_rows.append(rec)
    ext = pd.DataFrame(ext_rows)
    spec = delta_spec(anchors)
    corr = CorrectedMap(band, spec)
    lin = LinearGaussMap()
    pool = pool_kernels()
    sig = sigma_y(1e-3)
    sig_delta = nested_loo_cov(anchors, band)
    det, mcr = [], []
    for _, r in ext.iterrows():
        xi, s = float(r.xi_c), float(r.s_bar)
        y3d = np.array([r[f"eta3d_m{m}"] for m in MS])
        if np.isnan(y3d).any():
            continue
        rec = {"xi_true": xi, "s_true": s, "depth_frac": float(r.depth_frac),
               "inside_hull": inside_hull(anchors, xi, float(r.s_key))}
        e_unc, e_cor = y3d - band(xi, s), y3d - corr(xi, s)
        for c in c_list:
            fl = floors(c)
            rec[f"fwd_unc_floor_max_c{c:g}"] = float(np.max(np.abs(e_unc) / fl))
            rec[f"fwd_cor_floor_max_c{c:g}"] = float(np.max(np.abs(e_cor) / fl))
        for i, m in enumerate(MS):
            rec[f"delta_true_m{m}"] = float(r[f"delta_m{m}"])
            rec[f"delta_hat_m{m}"] = float(mc.kappa_eval(spec, xi, s, key="delta")[i])
        fa = est.fit(y3d, pool, PLATE, sig, w=W_GAUSS, mass="exact", n_starts=4)
        rec.update(_fit_cols("prod_lin_gauss", fa["xi_d"], fa["s_bar"], xi, s, lin, y3d, sig,
                             fa["boundary_hit"]))
        for tag, fmap in (("unc_band", band), ("cor_band", corr)):
            f = wls_fit(y3d, fmap, sig, grid=True)
            rec.update(_fit_cols(tag, f["xi"], f["s"], xi, s, fmap, y3d, sig,
                                 f["boundary_xi"] or f["boundary_s"], f["penalty"]))
        det.append(rec)
        # 잡음 MC
        z = np.load(DATA / f"rev1_n5_band_table_h{HALF_REF:.2f}.npz")
        band_npz = {"xs": z["xs"], "ys": z["ys"], "vals": z["vals"]}
        for ci, c in enumerate(c_list):
            seed = SEED0 + 500000 + 1000 * int(round(xi * 100)) + 10 * int(round(s * 1000)) + ci
            key = {"fold": "external_3d", "held_desc": f"xi={xi:.2f},p={r.depth_frac}",
                   "xi_true": xi, "s_true": s, "inside_hull": rec["inside_hull"],
                   "sig_delta_diag": "|".join(f"{v:.3e}" for v in np.diag(sig_delta))}
            mcr += _mc_worker((key, y3d, xi, s, c, seed, n_real, spec, sig_delta, HALF_REF,
                               band_npz))
    return pd.DataFrame(det), pd.DataFrame(mcr)


# =============================================================== 과제 0: b7 재실행(현행 Σ_y)
def rerun_b7(spec: dict, n_real: int = 1000, workers: int = 4) -> pd.DataFrame:
    """b7 MC를 **원고 §3.5가 명시한 관측량별 바닥**으로 다시 돈다 — 정본 b7 파일은 건드리지 않는다.

    진실 = 3D 대리모델(같은 δ 사양), 역맵 = 생산 1차 Gaussian(`estimator.fit`, mass="exact",
    n_starts = 4), 셀·시드·실현수 = b7과 같다(seed0 = 20260808, 15 ξ × 3 S̄ × 4 c). 달라지는 것은
    `montecarlo.run_cell`이 쓰는 `noi.sigma_y_for_modes`가 이제 m = 0에 2√2c를 준다는 것뿐이다
    (b7은 2026-08-14, 그 규약 도입 하루 전에 계산됐다).
    """
    rows = mc.run_production(PLATE, MODES, B7_XI, list(B7_S), SIGMA_RELS, w=W_GAUSS,
                             n_real=n_real, n_workers=workers, n_grid=1001, mass="exact",
                             surrogate=spec, n_starts=4, seed0=20260808)
    base = mc.run_production(PLATE, MODES, B7_XI, list(B7_S), SIGMA_RELS, w=W_GAUSS,
                             n_real=n_real, n_workers=workers, n_grid=1001, mass="exact",
                             n_starts=4, seed0=20260808)
    return pd.DataFrame(rows + base)


def b7_stats(d: pd.DataFrame) -> dict:
    """r3.9 §4.3이 인용한 통계를 같은 규칙으로(경계포화 셀 = boundary_hit_prob > 0 제외)."""
    s = d[d.truth_model == "3d_rail_surrogate"]
    sat = s.boundary_hit_prob > 0
    ns = s[~sat]
    ab = ns.bias_xi_mm.abs()
    r = s.bias_xi_mm.abs() / s.crlb_xi_mm
    p = s.pivot_table(index=["xi_d", "s_bar"], columns="sigma_rel", values="bias_xi_mm")
    ch = ((p[1e-4] - p[3e-3]).abs() / p[3e-3].abs())
    return {"n_cells": int(len(s)), "n_saturated": int(sat.sum()),
            "nonsat_abs_bias_median_mm": float(ab.median()),
            "nonsat_abs_bias_range_mm": [float(ab.min()), float(ab.max())],
            "std_over_crlb_median": float(s.ratio_std_over_crlb_xi.median()),
            "cells_bias_gt_crlb": int((r > 1).sum()),
            "bias_over_crlb_median": float(r.median()),
            "bias_change_30x_median": float(ch.median()),
            "coverage95_xi_median": float(s.coverage95_xi.median())}


# =============================================================== 요약
def summarize(dec: pd.DataFrame | None, ho: pd.DataFrame | None,
              hmc: pd.DataFrame | None) -> dict:
    out = {}
    if dec is not None:
        ns = dec[~dec["b7_saturated"].fillna(False)]
        a = ns["a_lin_gauss_bias_xi_mm"]
        out["decomp_n_cells_nonsat"] = int(len(ns))
        out["decomp_a_abs_median_mm"] = float(a.abs().median())
        out["decomp_a_abs_range_mm"] = [float(a.abs().min()), float(a.abs().max())]
        out["decomp_b7_c1e-4_abs_median_mm"] = float(ns["b7_bias_xi_mm_c0.0001"].abs().median())
        out["decomp_a_vs_b7_c1e-4_maxdiff_mm"] = float(
            (ns["a_lin_gauss_bias_xi_mm"] - ns["b7_bias_xi_mm_c0.0001"]).abs().max())
        for tag in ("a_legacy", "b_exact_gauss", "c_exact_band", "ap_lin_band"):
            col = ns[f"{tag}_bias_xi_mm"].abs()
            out[f"decomp_{tag}_abs_median_mm"] = float(col.median())
            out[f"decomp_{tag}_abs_range_mm"] = [float(col.min()), float(col.max())]
            out[f"decomp_{tag}_boundary_cells"] = int(ns[f"{tag}_boundary"].sum())
            out[f"decomp_{tag}_abs_bias_s_median_pp"] = float(
                ns[f"{tag}_bias_s_pp"].abs().median())
            out[f"decomp_{tag}_chi2_c1e3_median"] = float(ns[f"{tag}_chi2_c1e3"].median())
        out["decomp_a_legacy_vs_b7_c1e-4_maxdiff_mm"] = float(
            (ns["a_legacy_bias_xi_mm"] - ns["b7_bias_xi_mm_c0.0001"]).abs().max())
        out["decomp_a_vs_a_legacy_absdiff_max_mm"] = float(
            (ns["a_lin_gauss_bias_xi_mm"] - ns["a_legacy_bias_xi_mm"]).abs().max())
        out["decomp_a_abs_bias_s_median_pp"] = float(ns["a_lin_gauss_bias_s_pp"].abs().median())
        out["decomp_a_chi2_c1e3_median"] = float(ns["a_lin_gauss_chi2_c1e3"].median())
        for tag in ("a_lin_gauss", "b_exact_gauss", "c_exact_band"):
            for c in SIGMA_RELS:
                col = f"{tag}_ratio_c{c:g}"
                if col in ns:
                    out[f"decomp_{tag}_ratio_median_c{c:g}"] = float(ns[col].median())
                    out[f"decomp_{tag}_cells_ratio_gt1_c{c:g}"] = int((ns[col] > 1).sum())
                    out[f"decomp_{tag}_crlb_median_mm_c{c:g}"] = float(
                        ns[f"{tag}_crlb_xi_mm_c{c:g}"].median())
        out["decomp_saturated_cells"] = [
            [float(x), float(y)] for x, y in
            dec.loc[dec["b7_saturated"].fillna(False), ["xi_true", "s_true"]].to_numpy()]
        for col in ("lin_share_mm", "footprint_share_mm", "modelform_share_mm",
                    "footprint_share_path2_mm", "lin_share_path2_mm", "lin_shapley_mm",
                    "footprint_shapley_mm"):
            out[f"decomp_{col}_median"] = float(ns[col].median())
            out[f"decomp_{col}_abs_median"] = float(ns[col].abs().median())
        tot = ns["a_lin_gauss_bias_xi_mm"]
        for col in ("lin_shapley_mm", "footprint_shapley_mm", "modelform_share_mm"):
            share = ns[col] / tot                        # 셀별 부호 있는 몫(합 = 1)
            out[f"decomp_signedfrac_{col}_median"] = float(share.median())
            out[f"decomp_signedfrac_{col}_iqr"] = [float(share.quantile(.25)),
                                                  float(share.quantile(.75))]
        for num, tag in (("c_exact_band_bias_xi_mm", "modelform"),
                         ("b_exact_gauss_bias_xi_mm", "after_lin_removed"),
                         ("ap_lin_band_bias_xi_mm", "after_footprint_removed_path2")):
            frac = ns[num].abs() / ns["a_lin_gauss_bias_xi_mm"].abs()
            out[f"decomp_cellratio_{tag}_over_total_median"] = float(frac.median())
            out[f"decomp_cellratio_{tag}_over_total_iqr"] = [float(frac.quantile(.25)),
                                                             float(frac.quantile(.75))]
            out[f"decomp_cells_{tag}_smaller_than_total"] = int((frac < 1).sum())
    if ho is not None:
        for (fold, hull), g in ho.groupby(["fold", "inside_hull"]):
            k = f"ho_{fold}_{'interp' if hull else 'extrap'}"
            out[k + "_n"] = int(len(g))
            for tag in ("prod_lin_gauss", "unc_band", "cor_band"):
                out[k + f"_{tag}_abs_err_xi_median_mm"] = float(
                    g[f"{tag}_bias_xi_mm"].abs().median())
                out[k + f"_{tag}_abs_err_xi_max_mm"] = float(
                    g[f"{tag}_bias_xi_mm"].abs().max())
                out[k + f"_{tag}_abs_err_s_median_pp"] = float(
                    g[f"{tag}_bias_s_pp"].abs().median())
            out[k + "_ctrl_cor_abs_err_xi_max_mm"] = float(g["ctrl_cor_bias_xi_mm"].abs().max())
    if hmc is not None:
        for (fold, c, mp), g in hmc.groupby(["fold", "c", "map"]):
            k = f"mc_{fold}_c{c:g}_{mp}"
            out[k + "_n_cells"] = int(len(g))
            out[k + "_rmse_xi_median_mm"] = float(g["rmse_xi_mm"].median())
            out[k + "_abs_bias_xi_median_mm"] = float(g["bias_xi_mm"].abs().median())
            out[k + "_coverage95_xi_median"] = float(g["coverage95_xi"].median())
            out[k + "_coverage95_s_median"] = float(g["coverage95_s"].median())
    return out


# =============================================================== 그림
def make_figure(ho: pd.DataFrame, hmc: pd.DataFrame, path=None):
    """(a) held-out 3D 앵커의 잡음 없는 위치오차, (b) c = 0.1 % 잡음 MC의 95 % 포함률.

    캡션 문구는 그림에 넣지 않는다(정본 규약) — 축·범례만.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = ho[ho.fold == "leave_one_anchor"]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.6))
    styles = {"prod_lin_gauss": ("First-order Gaussian (production)", "C7", "s"),
              "unc_band": ("Re-solved band, uncorrected", "C3", "o"),
              "cor_band": ("Re-solved band + held-out correction", "C0", "D")}
    for tag, (lab, col, mk) in styles.items():
        for sk, fill in ((11.56, True), (17.5, False)):
            gg = g[np.isclose(g.s_key, sk)].sort_values("xi_true")
            ax[0].semilogy(gg.xi_true, np.maximum(gg[f"{tag}_bias_xi_mm"].abs(), 1e-3),
                           marker=mk, color=col, ls="-" if fill else "--",
                           mfc=col if fill else "white",
                           label=f"{lab}, S = {sk:.1f}%")
    for x in (0.2, 0.9):
        ax[0].axvspan(x - 0.03, x + 0.03, color="0.9", zorder=0)
    ax[0].set_xlabel("Held-out anchor location ξ_d")
    ax[0].set_ylabel("Absolute location error [mm]")
    ax[0].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    m = hmc[(hmc.fold == "leave_one_anchor") & np.isclose(hmc.c, 1e-3)]
    styles2 = {"unc_band": ("Uncorrected", "C3", "o"), "cor_band": ("Corrected", "C0", "D"),
               "cor_band_inflated": ("Corrected, Σ_y + Σ_δ", "C2", "^")}
    for tag, (lab, col, mk) in styles2.items():
        for sk, fill in ((0.115625, True), (0.175, False)):
            gg = m[(m["map"] == tag) & np.isclose(m.s_true, sk)].sort_values("xi_true")
            ax[1].plot(gg.xi_true, gg.coverage95_xi, marker=mk, color=col,
                       ls="-" if fill else "--", mfc=col if fill else "white",
                       label=f"{lab}, S = {100*sk:.1f}%")
    ax[1].axhline(0.95, color="k", lw=0.8, ls=":")
    ax[1].set_ylim(-0.03, 1.03)
    ax[1].set_xlabel("Held-out anchor location ξ_d")
    ax[1].set_ylabel("Empirical 95% coverage (location), σ_f/f = 0.1%")
    ax[1].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    for a, t in zip(ax, ("(a)", "(b)")):
        a.text(0.01, 1.02, t, transform=a.transAxes, fontsize=10)
    fig.tight_layout()
    out = path or out_fig("n5_heldout")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# =============================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m impeller_fingerprint.rev1.n5_heldout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("table")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--kinds", nargs="+", default=["band", "gauss"])
    sub.add_parser("decomp")
    sub.add_parser("heldout")
    sp = sub.add_parser("mc")
    sp.add_argument("--n-real", type=int, default=300)
    sp.add_argument("--workers", type=int, default=4)
    sub.add_parser("ext3d")
    sp = sub.add_parser("exttest")
    sp.add_argument("--n-real", type=int, default=300)
    sub.add_parser("summary")
    sp = sub.add_parser("b7rerun")
    sp.add_argument("--n-real", type=int, default=1000)
    sp.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    if a.cmd == "table":
        if "band" in a.kinds:
            build_band_table(workers=a.workers)
            print("[n5] band table saved", flush=True)
        if "gauss" in a.kinds:
            build_gauss_table(workers=a.workers)
            print("[n5] gauss table saved", flush=True)
        return
    if a.cmd == "ext3d":
        run_ext3d()
        return
    if a.cmd == "b7rerun":
        spec = delta_spec(load_anchors(HALF_REF))
        t0 = time.time()
        d = rerun_b7(spec, n_real=a.n_real, workers=a.workers)
        d.to_csv(out_csv("n5_b7_rerun_stated_cov"), index=False)
        old = pd.read_csv(DATA / "b7_mc_surrogate.csv")
        st = {"stated_cov_rerun": b7_stats(d), "b7_as_committed_legacy_cov": b7_stats(old),
              "elapsed_s": time.time() - t0}
        (DATA / "rev1_n5_b7_rerun_stats.json").write_text(json.dumps(st, indent=1))
        print(json.dumps(st, indent=1))
        return
    band = BandMap(load_table("band"))
    anchors = load_anchors(HALF_REF)
    if a.cmd == "decomp":
        gauss = GaussExactMap(load_table("gauss"))
        d = join_b7(run_decomposition(band, gauss, delta_spec(anchors)))
        d = add_matched_ratios(d, {"a_lin_gauss": LinearGaussMap(),
                                   "b_exact_gauss": gauss, "c_exact_band": band})
        d.to_csv(out_csv("n5_decomposition"), index=False)
        print(json.dumps(summarize(d, None, None), indent=1))
    elif a.cmd == "heldout":
        ho = heldout_noiseless(anchors, band)
        ho.to_csv(out_csv("n5_heldout_noiseless"), index=False)
        print(json.dumps(summarize(None, ho, None), indent=1))
    elif a.cmd == "mc":
        t0 = time.time()
        hmc = heldout_mc(anchors, band, n_real=a.n_real, workers=a.workers)
        hmc.to_csv(out_csv("n5_heldout_mc"), index=False)
        print(f"[n5 mc] {len(hmc)} rows in {time.time()-t0:.0f}s")
        print(json.dumps(summarize(None, None, hmc), indent=1))
    elif a.cmd == "exttest":
        det, mcr = external_test(anchors, band, n_real=a.n_real)
        det.to_csv(out_csv("n5_external_test"), index=False)
        mcr.to_csv(out_csv("n5_external_mc"), index=False)
        print(det.filter(regex="xi_true|s_true|bias_xi_mm|floor_max").to_string())
        print(json.dumps(summarize(None, None, mcr), indent=1))
    elif a.cmd == "summary":
        dec = pd.read_csv(out_csv("n5_decomposition"))
        ho = pd.read_csv(out_csv("n5_heldout_noiseless"))
        hm = pd.read_csv(out_csv("n5_heldout_mc"))
        s = summarize(dec, ho, hm)
        ext_mc = out_csv("n5_external_mc")
        if ext_mc.exists():
            s.update(summarize(None, None, pd.read_csv(ext_mc)))
        ext_det = out_csv("n5_external_test")
        if ext_det.exists():
            e = pd.read_csv(ext_det)
            for tag in ("prod_lin_gauss", "unc_band", "cor_band"):
                s[f"ext_{tag}_abs_err_xi_median_mm"] = float(e[f"{tag}_bias_xi_mm"].abs().median())
                s[f"ext_{tag}_abs_err_xi_max_mm"] = float(e[f"{tag}_bias_xi_mm"].abs().max())
                s[f"ext_{tag}_chi2_c1e3_median"] = float(e[f"{tag}_chi2_c1e3"].median())
            for c in ("0.001", "0.0003"):
                s[f"ext_fwd_unc_floor_max_median_c{c}"] = float(e[f"fwd_unc_floor_max_c{c}"].median())
                s[f"ext_fwd_cor_floor_max_median_c{c}"] = float(e[f"fwd_cor_floor_max_c{c}"].median())
        st = DATA / "rev1_n5_b7_rerun_stats.json"
        if st.exists():
            s["b7_rerun"] = json.loads(st.read_text())
        s["figure"] = str(make_figure(ho, hm))
        (DATA / "rev1_n5_summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
