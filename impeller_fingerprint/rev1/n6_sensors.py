"""N6 — LDV 없이: 부착식 가속도계의 질량부하와 근거리 마이크 가시도 (E2-I, 2026-10-17).

가속도계는 한 면(뒤판 바깥면 z = 0 또는 앞판 윗면)의 한 점 둘레 발자국 반경 안 꼭짓점에 고르게 나눈 집중질량
(병진 3성분)이다. 회전관성·접착층·국부 보강은 넣지 않는다. 얇은 판의 패널 모드에서는 실제 영향이 이보다
크므로 강한 자리(허브·베인 용접점 위)만 추천한다. 자유-자유만 다룬다 — 구속이 없어 축약 자유도가 전체 자유도와
같고, 꼭짓점 i·성분 c의 자유도는 3i + c다(`n6_freefree.solve`가 쓰는 배열 규약).

센서 하나는 C6 대칭을 깬다 → doublet이 갈라진다. 짝짓기는 `n2_pocket.track_split`와 같은 규칙(연속 블록,
축퇴 요구 없음, 부분공간 MAC)이고, 짝 번호도 함께 돌려준다(센서를 단 채 홈 1차 응답을 내기 위해).

마이크는 앞판 윗면(또는 뒤판 바깥면) 법선변위의 Rayleigh 적분(배플 가정)이다:
    p(x) = (ω² ρ₀ / 2π) Σ_j w_j A_j e^{−ikR_j} / R_j
근거리(d ≪ λ)에서는 마이크 아래 몇 cm² 넓이의 평균 변위를 본다. 반대편 면·가장자리 회절은 넣지 않는다.
모드형은 두 판 바깥면의 최대 법선변위를 w_ref(기본 1 µm)로 맞춘다 → |p|는 "그 모드가 1 µm로 떨 때의 음압"이다.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy import sparse

from . import n1_assembly as n1

RHO_AIR = 1.204          # kg/m³ (20 °C)
C_AIR = 343.0            # m/s


# ---------------------------------------------------------------- 면·발자국
def face_masks(fm) -> dict:
    """바깥면 꼭짓점: front = 앞판 윗면(흡입구 테 아래 제외), back = 뒤판 바깥면(z = 0, 허브 포함)."""
    c, r = fm.coors, fm.r_v
    _, z_top = fm.spec.z_front
    return {"front": np.isclose(c[:, 2], z_top, atol=1e-9) & (r >= fm.spec.a + fm.spec.collar_t - 1e-9),
            "back": np.isclose(c[:, 2], 0.0, atol=1e-9)}


def footprint(fm, face: str, r_c: float, theta_c: float, radius: float) -> np.ndarray:
    """면 위 (r_c, θ_c) 둘레, 평면 거리 ≤ radius인 꼭짓점. 하나도 없으면 가장 가까운 한 점."""
    m = face_masks(fm)[face]
    c = fm.coors
    d = np.hypot(c[:, 0] - r_c * math.cos(theta_c), c[:, 1] - r_c * math.sin(theta_c))
    idx = np.nonzero(m & (d <= radius + 1e-12))[0]
    if idx.size == 0:
        cand = np.nonzero(m)[0]
        idx = cand[[int(np.argmin(d[cand]))]]
    return idx


def footprint_radius_mm(mass_g: float) -> float:
    """센서 질량 → 바닥 반경 [mm] 가정(소형 IEPE 가속도계: 0.5–1 g Ø 5 mm, 2 g Ø 7 mm, 5–6 g Ø 10–11 mm)."""
    return 2.5 if mass_g <= 1.0 else (3.5 if mass_g <= 2.5 else 5.0)


def with_mass(km: n1.KM, nodes, mass: float) -> n1.KM:
    """M + 집중질량(발자국 꼭짓점의 병진 3성분에 고르게) — 자유-자유 전용."""
    n = km.M.shape[0]
    if n < 3 * km.n_vertex:
        raise ValueError("구속이 있는 모형은 자유도 대응이 다르다 — 자유-자유만 지원")
    nodes = np.asarray(nodes, int)
    dofs = (3 * nodes[:, None] + np.arange(3)[None, :]).ravel()
    dM = sparse.coo_matrix((np.full(dofs.size, mass / nodes.size), (dofs, dofs)), shape=(n, n))
    return replace(km, M=(km.M + dM).tocsc())


def first_order_mass_eta(md: n1.Modes, obs, nodes, mass: float) -> np.ndarray:
    """1차 η = −φᵀδMφ(질량정규 φ)의 성분 평균 — 작은 질량의 기준값."""
    nodes = np.asarray(nodes, int)
    out = np.empty(len(obs))
    for j, o in enumerate(obs):
        vals = [-(mass / nodes.size) * float((md.vshape[i].reshape(-1, 3)[nodes] ** 2).sum()) for i in o.members]
        out[j] = float(np.mean(vals))
    return out


def match(md_h: n1.Modes, md: n1.Modes, obs, min_mac: float = 0.8):
    """대칭이 깨진 모드를 건전 관측량에 짝짓는다 → (쌍평균·singlet η, doublet 분리 |Δλ|/λ, MAC, 짝 번호)."""
    Vh, Vd = md_h.vshape, md.vshape
    used: set = set()
    mean = np.full(len(obs), np.nan)
    split = np.full(len(obs), np.nan)
    mac = np.full(len(obs), np.nan)
    idx: list = [[] for _ in obs]
    for j, o in enumerate(obs):
        Qa, _ = np.linalg.qr(Vh[o.members].T)
        best = (-1.0, None)
        for s in range(Vd.shape[0] - o.deg + 1):
            cand = list(range(s, s + o.deg))
            if any(c in used for c in cand):
                continue
            Qb, _ = np.linalg.qr(Vd[cand].T)
            sc = float(np.min(np.linalg.svd(Qa.T @ Qb, compute_uv=False)) ** 2)
            if sc > best[0]:
                best = (sc, cand)
        if best[1] is None or best[0] < min_mac:
            continue
        used.update(best[1])
        lam0 = float(np.mean(md_h.lam[o.members]))
        ld = md.lam[best[1]]
        mean[j] = float(np.mean(ld) / lam0 - 1.0)
        if o.deg == 2:
            split[j] = float(abs(ld[1] - ld[0]) / lam0)
        mac[j] = best[0]
        idx[j] = list(best[1])
    return mean, split, mac, idx


def accel_visibility(fm, md: n1.Modes, members, node: int) -> float:
    """가속도계(축방향 z) 가시도 = 그 점 √Σφ_z² / 두 판 바깥면 최대 √Σφ_z² (doublet은 성분 합)."""
    if not members:
        return float("nan")
    ref = face_masks(fm)
    ref = ref["front"] | ref["back"]
    uz2 = sum(md.vshape[i].reshape(-1, 3)[:, 2] ** 2 for i in members)
    return float(math.sqrt(uz2[node] / uz2[ref].max()))


# ---------------------------------------------------------------- 마이크
def node_areas(fm, face: str) -> np.ndarray:
    """면 꼭짓점의 넓이 가중 — 그 면에 놓인 육면체 면(꼭짓점 4개)의 넓이를 1/4씩 나눈다."""
    m = face_masks(fm)[face]
    A = np.zeros(len(fm.coors))
    conn = fm.conn
    on = m[conn]
    for e in np.nonzero(on.sum(axis=1) == 4)[0]:
        q = conn[e][on[e]]
        xy = fm.coors[q, :2]
        cen = xy.mean(axis=0)
        order = np.argsort(np.arctan2(xy[:, 1] - cen[1], xy[:, 0] - cen[0]))
        x, y = xy[order, 0], xy[order, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        A[q] += 0.25 * area
    return A


def rayleigh_pressure(xs: np.ndarray, areas: np.ndarray, w: np.ndarray, f: float, mic: np.ndarray) -> complex:
    """배플 Rayleigh 적분 p = (ω²ρ₀/2π) Σ w A e^{−ikR}/R [Pa] (w = 법선변위 진폭 [m])."""
    om = 2 * math.pi * f
    R = np.linalg.norm(np.asarray(mic)[None, :] - xs, axis=1)
    return complex(om ** 2 * RHO_AIR / (2 * math.pi) * np.sum(w * areas * np.exp(-1j * om / C_AIR * R) / R))


def mic_pressure(fm, md: n1.Modes, obs, mic: np.ndarray, face: str = "front", w_ref: float = 1e-6,
                 members_list=None) -> np.ndarray:
    """관측량별 |p| [Pa] — 두 판 바깥면 최대 법선변위를 w_ref로 맞춘 모드형(doublet은 두 성분 RSS)."""
    A = node_areas(fm, face)
    idx = np.nonzero(A > 0)[0]
    masks = face_masks(fm)
    ref = masks["front"] | masks["back"]
    xs = fm.coors[idx]
    out = np.full(len(obs), np.nan)
    for j, o in enumerate(obs):
        mem = o.members if members_list is None else members_list[j]
        if not mem:
            continue
        W = np.stack([md.vshape[i].reshape(-1, 3)[:, 2] for i in mem])
        scale = w_ref / float(np.sqrt((W[:, ref] ** 2).sum(axis=0)).max())
        f = float(np.mean(np.sqrt(md.lam[mem])) / (2 * math.pi))
        ps = [rayleigh_pressure(xs, A[idx], W[k, idx] * scale, f, mic) for k in range(len(mem))]
        out[j] = float(math.sqrt(sum(abs(p) ** 2 for p in ps)))
    return out


# ---------------------------------------------------------------- 후보 위치
def placements(fm) -> dict:
    """가속도계 후보 부착점 이름 → (면, 중심 꼭짓점). 베인선 = 섹터 격자 열 0, 베이 가운데 = 열 n_vc + n_bay//2."""
    fs = fm.spec.base
    per = fs.n_vc + fs.n_bay
    loc = fm.info["grid_idx"][:, 1] % per
    masks = face_masks(fm)
    a, ext = fm.spec.a, fm.spec.extent

    def pick(face, r_t, col):
        c = np.nonzero(masks[face] & (loc == col))[0]
        return int(c[np.argmin(np.abs(fm.r_v[c] - r_t))])

    out = {"hub_back_r10mm": ("back", pick("back", 10e-3, 0))}
    for xi in (0.3, 0.6, 0.9):
        out[f"back_vane_xi{xi}"] = ("back", pick("back", a + xi * ext, 0))
    out["front_vane_xi0.6"] = ("front", pick("front", a + 0.6 * ext, 0))
    out["front_centre_xi0.9"] = ("front", pick("front", a + 0.9 * ext, fs.n_vc + fs.n_bay // 2))
    return out


def mic_positions(fm) -> dict:
    """근거리 마이크 후보 이름 → (적분면, 위치 xyz). 베이 가운데·베인선 위 높이 d, 축 위, 뒤판 아래."""
    fs = fm.spec.base
    per = fs.n_vc + fs.n_bay
    loc = fm.info["grid_idx"][:, 1] % per
    masks = face_masks(fm)
    _, z_top = fm.spec.z_front
    a, ext = fm.spec.a, fm.spec.extent

    def xy(face, xi, col):
        c = np.nonzero(masks[face] & (loc == col))[0]
        j = int(c[np.argmin(np.abs(fm.r_v[c] - (a + xi * ext)))])
        return fm.coors[j, 0], fm.coors[j, 1]

    cen = fs.n_vc + fs.n_bay // 2
    out = {}
    for xi in (0.5, 0.9):
        x, y = xy("front", xi, cen)
        for d in (5e-3, 10e-3):
            out[f"front_centre_xi{xi}_d{d * 1e3:.0f}mm"] = ("front", np.array([x, y, z_top + d]))
    x, y = xy("front", 0.9, 0)
    out["front_vane_xi0.9_d5mm"] = ("front", np.array([x, y, z_top + 5e-3]))
    out["axis_front_d20mm"] = ("front", np.array([0.0, 0.0, z_top + 20e-3]))
    x, y = xy("back", 0.9, cen)
    out["back_centre_xi0.9_d5mm"] = ("back", np.array([x, y, -5e-3]))
    return out
