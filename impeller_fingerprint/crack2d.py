"""A11 — 2D 평면탄성 + **폭 0 균열**: 정본 §3.6-iv의 남은 arm.

정본 §3.6-iv는 "Timoshenko **and 2D-elasticity** crack models with Mode-I and Mode-II
local flexibility"를 요구한다. A7(Timoshenko + 국소유연도)과 B2(3D 솔리드 + EDM 커프)는
끝냈으나 **2D는 미실행**이었고, "3D가 2D를 포함한다"는 근거는 틀렸다:

  B2는 손상을 **요소 제거**로 넣으므로 커프 폭이 격자 크기로 이산화되고 최소 폭이 0.25 mm였다.
  그 결과 등가 균열깊이가 물리 깊이보다 13–70 % 깊게 나왔다(설계서 F10). 즉 **3D 레일이 푼 것은
  노치**이고, 회전스프링 모델이 이상화하는 **폭 0의 날카로운 균열**은 어느 arm도 풀지 않았다.

2D에서는 균열선상의 절점을 **이중화**해(양쪽 요소가 그 절점을 공유하지 않게) 폭이 정확히 0인
traction-free 슬릿을 만들 수 있다. 이 모듈이 그것을 만들고, 같은 격자에서 유한폭 노치
(요소 제거)도 만들어 **폭 → 0 극한**을 한 모델 안에서 검정한다.

구성
  1. `slit_mesh` — 등급화 구조격자 사각요소(2_4). 균열선 x_c와 균열선단 z_tip에 절점을
     **정확히** 놓으므로 균열깊이가 격자에 스냅되지 않는다(설계서 F11′의 형상보존 규약).
     `kerf_width=0`이면 절점 이중화(폭 0 슬릿), >0이면 요소 제거(유한폭 노치).
  2. `flap_modes` — `rail3d.solve_modes`(shift-invert)를 2D 요소로 재사용하고 면내 굽힘
     모드를 **형상으로** 고른다(정본 §3.6의 "주파수 순서 매칭 금지").
  3. `平面응력·평면변형 둘 다` — 좁은 보 단면은 평면응력(유효계수 E), 넓은 판 스트립은
     평면변형(E/(1−ν²)). 정본 §3.1이 이미 이 구분을 caveat로 달고 있다.
  4. `invert_c_theta` — 2D의 f₁ 강하와 일치하는 **등가 회전유연도** c_θ를 보 모델에서 역산한다.
     파괴역학 c_θ(ā)의 핸드북 규약차(Tada 적분 vs Dimarogonas 다항, 설계서 F7의 27 %) 중
     2D 탄성이 **어느 쪽에 가까운가**를 판정하는 것이 이 arm의 핵심 실용 가치다.
"""
from __future__ import annotations

import math

import numpy as np

from . import crack_shear as cs
from . import rail3d as r3

#: 등가 회전유연도 역산에서 쓰는 무차원 유연도 = c_θ·EI/L (E·I·L 규약차를 상쇄).
#: 핸드북(Tada/Dimarogonas)은 평면응력/평면변형 구분이 이 무차원 값에서 **상쇄**된다:
#: c_θ ∝ 1/E' 이고 보 강성도 E'I 이므로 c_θ·E'I/L 는 E' 에 무관하다.


def graded_nodes(x0: float, x1: float, n: int, toward: str = "hi",
                 bias: float = 1.0) -> np.ndarray:
    """[x0,x1]을 n개 구간으로 나눈 절점 좌표 — 한쪽 끝으로 등비 등급화.

    bias = (가장 큰 구간)/(가장 작은 구간). toward="hi"면 x1 쪽이 촘촘하다.
    균열선단은 응력 특이성이 있어 균일격자로는 수렴이 느리므로 등급화가 필요하다.
    """
    if n < 1:
        raise ValueError("n >= 1")
    if n == 1 or bias == 1.0:
        return np.linspace(x0, x1, n + 1)
    r = bias ** (1.0 / (n - 1))
    lens = r ** np.arange(n)                    # 증가하는 구간 길이
    if toward == "hi":
        lens = lens[::-1]                       # x1 쪽이 작아진다
    elif toward != "lo":
        raise ValueError("toward ∈ {'lo','hi'}")
    pos = np.concatenate([[0.0], np.cumsum(lens)])
    return x0 + (x1 - x0) * pos / pos[-1]


def slit_mesh(L: float, h: float, a_bar: float, xc_over_L: float = 0.2,
              nx_left: int = 12, nx_right: int = 48, nz_below: int = 6,
              nz_above: int = 6, bias: float = 12.0, kerf_width: float = 0.0,
              n_kerf: int = 2, crack: bool = True):
    """베인 단면(길이 L × 두께 h) 평면격자 + 균열/노치.

    좌표는 (x=길이, y=두께). 균열은 x_c에서 **아래면(y=−h/2)**부터 깊이 ā·h까지 —
    B2(3D)가 z 최소면부터 제거한 것과 같은 인장면이다.

    kerf_width=0  → 균열선상의 절점 이중화 = **폭 0 traction-free 슬릿**
    kerf_width>0  → 폭 밴드의 요소 제거 = 유한폭 노치(B2와 같은 손상법이지만 2D)
    crack=False   → 같은 절점배치의 **건전** 메시(주파수비에서 이산화오차 상쇄).

    반환 (coors, conn, info). info["tip"]=(x_c, y_tip), info["depth_exact"]=True/False.
    """
    if not 0.0 < a_bar < 1.0:
        raise ValueError("0 < a_bar < 1")
    xc = xc_over_L * L
    y_tip = -0.5 * h + a_bar * h
    half = 0.5 * kerf_width

    if half > 0.0:
        xs = np.concatenate([
            graded_nodes(0.0, xc - half, nx_left, "hi", bias),
            np.linspace(xc - half, xc + half, n_kerf + 1)[1:],
            graded_nodes(xc + half, L, nx_right, "lo", bias)[1:]])
        i_c0, i_c1 = nx_left, nx_left + n_kerf     # 커프 밴드의 좌·우 열 인덱스
    else:
        xs = np.concatenate([graded_nodes(0.0, xc, nx_left, "hi", bias),
                             graded_nodes(xc, L, nx_right, "lo", bias)[1:]])
        i_c0 = i_c1 = nx_left
    ys = np.concatenate([graded_nodes(-0.5 * h, y_tip, nz_below, "hi", bias),
                         graded_nodes(y_tip, 0.5 * h, nz_above, "lo", bias)[1:]])
    k_tip = nz_below                              # ys[k_tip] == y_tip (정확)

    nx, ny = xs.size - 1, ys.size - 1
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    coors = np.stack([X.ravel(), Y.ravel()], axis=1)
    nnode = coors.shape[0]

    def nid(i, k):
        return i * (ny + 1) + k

    # 폭 0 슬릿: 균열선(i_c0) 위의 k < k_tip 절점을 이중화한다. 선단절점(k_tip)은
    # 두 면이 만나는 점이므로 **공유**한다 — 이것이 날카로운 균열의 정의.
    dup = {}
    if crack and half == 0.0:
        extra = []
        for k in range(k_tip):
            dup[k] = nnode + len(extra)
            extra.append(coors[nid(i_c0, k)])
        coors = np.concatenate([coors, np.array(extra).reshape(-1, 2)], axis=0)

    conn = []
    for i in range(nx):
        # 유한폭 노치: 커프 밴드 안 + 선단 아래 요소를 제거한다.
        if crack and half > 0.0 and i_c0 <= i < i_c1:
            k_from = k_tip
        else:
            k_from = 0
        for k in range(k_from, ny):
            n00, n10 = nid(i, k), nid(i + 1, k)
            n11, n01 = nid(i + 1, k + 1), nid(i, k + 1)
            # 슬릿에 **왼쪽 변이 닿는** 요소(i == i_c0)만 이중절점을 쓴다.
            # `i >= i_c0`으로 쓰면 오른쪽 모든 열이 이중절점을 참조해 격자가 찢어진다.
            if dup and i == i_c0:
                if k in dup:
                    n00 = dup[k]
                if k + 1 in dup:
                    n01 = dup[k + 1]
            conn.append([n00, n10, n11, n01])
    conn = np.array(conn, dtype=np.int32)
    coors, conn = r3._compact(coors, conn)

    dx_tip = float(min(xs[i_c0] - xs[i_c0 - 1], xs[i_c1 + 1] - xs[i_c1]))
    dy_tip = float(min(ys[k_tip] - ys[k_tip - 1], ys[k_tip + 1] - ys[k_tip]))
    info = {"a_bar": float(a_bar), "kerf_width": float(kerf_width),
            "n_elem": int(conn.shape[0]), "n_node": int(coors.shape[0]),
            "n_dup_nodes": len(dup), "tip_x": float(xc), "tip_y": float(y_tip),
            "dx_tip": dx_tip, "dy_tip": dy_tip,
            "tip_elem_over_h": float(min(dx_tip, dy_tip) / h),
            "depth_exact": True, "crack": bool(crack)}
    return coors, conn, info


def flap_modes(coors, conn, E: float, nu: float, rho: float,
               plane: str = "stress", n_modes: int = 6, order: int = 2,
               seed: int | None = 20260815):
    """면내 굽힘(flap) 모드 주파수 [Hz] — 형상으로 골라낸다.

    2D 평면 단면에는 폭방향 굽힘이 없으므로 축방향(axial) 모드만 걸러내면 된다.

    **시드를 고정한다**(F102): `solve_modes`의 ARPACK는 시작벡터를 전역 RNG에서 뽑으므로
    고정하지 않으면 같은 입력이 실행마다 다른 마지막 자리를 주고, `cli a11`의 커밋 산출물이
    재현되지 않는다(2026-08-24에 두 실행이 파생열에서 상대 1e-4까지 갈렸다). 호출자가 다른
    시드를 쓰고 싶으면 `seed=None`으로 끌 수 있다.
    """
    if seed is not None:
        np.random.seed(seed)
    res = r3.solve_modes(coors, conn, E, nu, rho, r3.clamp_root(),
                         n_modes=n_modes, order=order, keep_shapes=True,
                         plane=plane)
    kinds = r3.beam_mode_kinds(res)
    flap = [i for i, k in enumerate(kinds) if k == "flap"]
    return res.freqs[flap], kinds, res.ndof


def beam_ratios_from_cmm(beam: cs.TimoBeam, c_mm: float, xc_over_L: float = 0.2,
                         n_elem: int = 300, n_modes: int = 3) -> np.ndarray:
    """회전스프링(Mode-I 단독) 보 모델의 주파수비 — c_MM을 **직접** 받는다.

    `cs.signature`는 ā에서 유연도를 만들지만, 등가 c_θ 역산에는 유연도를 자유변수로
    받는 함수가 필요하다. c_QQ는 상대 병진을 막기 위한 수치적 강체값(강성비 10⁶).
    """
    f0 = cs.frequencies(beam, n_modes=n_modes, n_elem=n_elem, crack=None)
    crack = {"c_MM": c_mm, "c_QQ": c_mm * beam.h ** 2 * 1e-6, "c_MQ": 0.0}
    f = cs.frequencies(beam, n_modes=n_modes, n_elem=n_elem, crack=crack,
                       xc_over_L=xc_over_L)
    return f / f0


def invert_c_theta(beam: cs.TimoBeam, ratio_f1: float, xc_over_L: float = 0.2,
                   n_elem: int = 300, lo: float = 1e-6, hi: float = 1e3) -> float:
    """주어진 f₁ 강하와 일치하는 등가 회전유연도 c_θ [rad/(N·m)] (없으면 NaN).

    lo·hi는 c_MM의 브래킷. 단조(유연도↑ → f₁↓)이므로 brentq로 안전하게 푼다.
    """
    from scipy.optimize import brentq

    def g(log_c):
        return beam_ratios_from_cmm(beam, 10.0 ** log_c, xc_over_L=xc_over_L,
                                   n_elem=n_elem, n_modes=1)[0] - ratio_f1

    a, b = math.log10(lo), math.log10(hi)
    if g(a) * g(b) > 0:
        return float("nan")
    return float(10.0 ** brentq(g, a, b, xtol=1e-6))


def dimensionless_c_theta(c_theta: float, beam: cs.TimoBeam) -> float:
    """무차원 회전유연도 c_θ·E I / L — 평면응력/평면변형의 E' 규약이 상쇄된다."""
    return c_theta * beam.E * beam.I / beam.L
