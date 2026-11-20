"""N1/N2 조립체 격자 — 캠버를 따라가는 구조격자 육면체(베인 두께 정확, 연속, 정확 C6).

`impeller_hex.mesh`는 균등 방위격자에서 요소중심이 캠버선 ±t/2 안이면 베인으로 판정한다.
그 판정은 격자에 따라 **베인이 끊긴다**(2026-10-06 확인: 18×72에서 반경행 18개 중 7개,
24×72에서 24개 중 15개에 베인 셀이 없다; 12×72는 연속이나 호 폭 1.4–3.1 mm로 실제 1.0 mm의
1.4–3.1배). Fig. 1·a13은 형상을 동결한 18×108(연속)이라 영향이 없지만, 정량 커널에는 쓸 수 없다.

이 격자는 각 반경 링에서 섹터의 방위 절점을 **베인 캠버에 고정된 오프셋**으로 둔다:

    θ(r) = θ_k(r) + φ(r),  θ_k(r) = 2πk/N − wrap · ln(r/r_vin)/ln(b/r_vin)   (r ≥ r_vin)

φ는 베인 면 ±t_v/(2r cos β)(캠버에 **수직인** 두께가 정확히 t_v; β = 블레이드각)과 그 사이
베이를 n_bay 등분한 값이다. 섹터마다
위상이 같으므로 n_θ = N(n_vc + n_bay)이고 회전 R(2π/N)이 절점집합을 정확히 자기 자신으로
보낸다(`impeller_hex.sector_permutation`이 그대로 동작). 캠버 법칙은 as-built CAD
(`impeller_cad.vane_camber`)와 같은 로그나선 후곡이다 — 블레이드각이 일정해 요소 기울기도
반경에 무관하게 일정하다(wrap 90°, r_vin 17 mm에서 약 64°).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import geometry as geo
from .. import impeller_hex as ih
from .. import rail3d as r3

_AB = geo.ASBUILT


@dataclass(frozen=True)
class FitSpec:
    """캠버 정합 조립체 스펙 [m]. 기본은 정본(`geometry.ASBUILT`, G3 표준 두께; G1은 `ASBUILT_G1`).

    08-13 기하(양면 1.0 mm, 유로 4.1 mm, 깃 1.0 mm, wrap 90°, r_vin 17 mm, a 15.40, b 36.56)는
    `FitSpec.geom0813()`으로 만든다 — r4.0 산출물(`rev1_n1_*_fit24x12v1z2c2w90` 등)이 그대로 재현된다.
    """

    a: float = _AB.a
    b: float = _AB.b
    t_front: float = _AB.t_front    # 앞판(z 위쪽, 흡입구 쪽 = 손상면)
    t_back: float = _AB.t_back      # 뒤판(z 아래쪽, 허브 쪽)
    channel: float = _AB.gap        # 두 판 안쪽면 간격(= 깃 높이)
    t_vane: float = _AB.t_vane
    n_vane: int = 6
    wrap_deg: float = _AB.wrap_deg  # 앞판 용접 자국 베인각 50.5°(G1 §3)
    r_vane_in: float = _AB.r_vane_in
    n_r: int = 24
    n_bay: int = 12                 # 베이(베인 사이) 방위 셀 수
    n_vc: int = 1                   # 베인 두께방향 셀 수
    n_z_shroud: int = 2
    n_z_channel: int = 2
    E: float = 193e9
    nu: float = 0.29
    rho: float = 7930.0
    #: 지지: "inner_edge" = r = a의 모든 절점 클램프(레일의 클램프-자유 환형판과 같은 이상화),
    #: "back_sheet" = 후면 슈라우드 안쪽 모서리만 클램프(전면 슈라우드 흡입구 모서리는 자유 —
    #: 실물은 후면 판이 허브로 이어지고 전면 판은 흡입구에서 열려 있다). 지지 민감도용.
    support: str = "inner_edge"
    #: 흡입구 쪽(r = a) 간격 — None이면 일정(G3). 값이면 r = b의 `channel`에서 r = a의 이 값까지 선형(G4 테이퍼).
    #: 격자는 평판 기준으로 만들고, 조립할 때만 `taper_coors`로 절점 높이를 옮긴다.
    channel_in: float | None = _AB.gap_in

    @classmethod
    def geom0813(cls, **kw) -> "FitSpec":
        """08-13 기하(폐기)의 스펙 — r4.0 N1·N2 산출물 재현용."""
        g = geo.GEOM_0813
        base = dict(a=g.a, b=g.b, t_front=g.t_sheet, t_back=g.t_sheet, channel=g.b2_channel,
                    t_vane=g.t_vane, wrap_deg=g.wrap_deg, r_vane_in=g.r_vane_in, channel_in=None)
        base.update(kw)
        return cls(**base)

    @property
    def tf(self) -> float:
        return self.t_front

    @property
    def tb(self) -> float:
        return self.t_back

    @property
    def t_sheet(self) -> float:
        """대칭 판정 색인용(`hex_spec`)·하위호환 — 뒤판 두께."""
        return self.t_back

    @property
    def asymmetric(self) -> bool:
        return self.t_front != self.t_back

    @property
    def extent(self) -> float:
        return self.b - self.a

    @property
    def tapered(self) -> bool:
        """간격이 반지름에 따라 바뀌는가(G4)."""
        return self.channel_in is not None and self.channel_in != self.channel

    def channel_at(self, r) -> np.ndarray:
        """반지름 r의 두 판 안쪽면 간격 [m] — 일정(G3) 또는 r = b의 channel에서 r = a의 channel_in까지 선형(G4).
        r < a(허브)는 channel_in, r > b는 channel로 자른다."""
        r = np.asarray(r, float)
        if not self.tapered:
            return np.full_like(r, self.channel)
        return self.channel + (self.channel_in - self.channel) * np.clip((self.b - r) / (self.b - self.a), 0.0, 1.0)

    def taper_coors(self, coors: np.ndarray, grid_idx: np.ndarray) -> np.ndarray:
        """평판 기준 격자 좌표 → 조립 좌표(G4 테이퍼). 일정 간격이면 입력을 그대로 돌려준다.

        뒤판(허브 포함, 층 iz ≤ n_z_shroud)은 그대로 두고, 채널 절점(깃)은 높이 비율 j/n_z_channel만큼, 앞판·흡입구 테
        절점은 Δc(r) = c(r) − channel만큼 z를 올린다 — 뒤판은 평판, 앞판은 얕은 원뿔이 되고 판 두께는 그대로다.
        손상(감육·홈·포켓·깃 뿌리)의 절점 이동은 평판 좌표에서 하고 이 사상을 마지막에 얹는다(층 번호로 고르므로
        이동한 절점에도 같은 Δz가 붙는다). N6 격자(`n6_freefree`)도 층 번호가 같다(테 층은 앞판 위).
        """
        if not self.tapered:
            return coors
        nzs, nzc = self.n_z_shroud, self.n_z_channel
        iz = np.asarray(grid_idx)[:, 2]
        frac = np.where(iz <= nzs, 0.0, np.where(iz <= nzs + nzc, (iz - nzs) / nzc, 1.0))
        out = np.array(coors, dtype=float, copy=True)
        out[:, 2] += frac * (self.channel_at(np.hypot(coors[:, 0], coors[:, 1])) - self.channel)
        return out

    def clamp_fn(self, tol: float = 1e-7):
        a, zmax = self.a, self.tb
        if self.support == "inner_edge":
            return lambda c: np.nonzero(np.hypot(c[:, 0], c[:, 1]) <= a + tol)[0]
        if self.support == "back_sheet":
            return lambda c: np.nonzero((np.hypot(c[:, 0], c[:, 1]) <= a + tol)
                                        & (c[:, 2] <= zmax + tol))[0]
        raise ValueError(f"support: {self.support}")

    @property
    def n_theta(self) -> int:
        return self.n_vane * (self.n_vc + self.n_bay)

    def hex_spec(self) -> ih.HexImpellerSpec:
        """`impeller_hex`의 대칭 판정 함수에 넘길 색인용 스펙(기하는 쓰지 않는다)."""
        return ih.HexImpellerSpec(a=self.a, b=self.b, t_front=self.t_front, t_back=self.t_back,
                                  channel=self.channel, t_vane=self.t_vane,
                                  n_vane=self.n_vane, wrap_deg=self.wrap_deg,
                                  E=self.E, nu=self.nu, rho=self.rho, n_r=self.n_r,
                                  n_theta=self.n_theta, n_z_shroud=self.n_z_shroud,
                                  n_z_channel=self.n_z_channel)

    def camber(self, r) -> np.ndarray:
        r = np.asarray(r, float)
        u = np.log(np.maximum(r, self.r_vane_in) / self.r_vane_in) / math.log(self.b / self.r_vane_in)
        return -math.radians(self.wrap_deg) * u

    def blade_angle(self, r) -> float:
        """캠버선과 반경방향 사이 각 β(r) [rad] — 로그나선이면 r ≥ r_vin에서 일정."""
        if r < self.r_vane_in:
            return 0.0
        return math.atan(math.radians(self.wrap_deg) / math.log(self.b / self.r_vane_in))

    @property
    def z_breaks(self) -> np.ndarray:
        c = self.channel
        if not self.asymmetric:                     # 같은 두께면 08-13 식 그대로(부동소수 비트 보존)
            t = self.t_front
            back = np.linspace(0.0, t, self.n_z_shroud + 1)
            chan = np.linspace(t, t + c, self.n_z_channel + 1)[1:]
            front = np.linspace(t + c, 2 * t + c, self.n_z_shroud + 1)[1:]
            return np.concatenate([back, chan, front])
        tb, tf = self.tb, self.tf
        back = np.linspace(0.0, tb, self.n_z_shroud + 1)
        chan = np.linspace(tb, tb + c, self.n_z_channel + 1)[1:]
        front = np.linspace(tb + c, tb + c + tf, self.n_z_shroud + 1)[1:]
        return np.concatenate([back, chan, front])


def mesh(fs: FitSpec):
    """→ (coors, conn, info). info: grid_idx(꼭짓점 (ir,it,iz)), cell_kind(0 후면·1 베인·2 전면),
    cell_vane(베인 셀이면 섹터 번호, 아니면 −1), cell_r(셀 중심 반경), cell_span(베인 스팬분율)."""
    nr, nth = fs.n_r, fs.n_theta
    per = fs.n_vc + fs.n_bay
    zb = fs.z_breaks
    nz = len(zb) - 1
    kinds = (["back"] * fs.n_z_shroud + ["channel"] * fs.n_z_channel
             + ["front"] * fs.n_z_shroud)
    r_nodes = np.linspace(fs.a, fs.b, nr + 1)
    # 섹터 국소 방위 오프셋(링별): 베인 셀 n_vc개(−t/2r … +t/2r), 베이 n_bay개
    phi = np.zeros((nr + 1, per))
    for i, r in enumerate(r_nodes):
        # 캠버에 **수직인** 두께가 t_v가 되도록: 방위폭 = t_v/(r·cos β), tan β = r·|dθ_k/dr|
        hv = fs.t_vane / (2 * r * math.cos(fs.blade_angle(r)))
        vane = np.linspace(-hv, hv, fs.n_vc + 1)[:-1]
        bay = np.linspace(hv, 2 * math.pi / fs.n_vane - hv, fs.n_bay + 1)[:-1]
        phi[i] = np.concatenate([vane, bay])
    cam = fs.camber(r_nodes)
    theta = np.zeros((nr + 1, nth))
    for k in range(fs.n_vane):
        theta[:, k * per:(k + 1) * per] = (2 * math.pi * k / fs.n_vane + cam[:, None] + phi)

    def nid(ir, it, iz):
        return (iz * nth * (nr + 1)) + ((it % nth) * (nr + 1)) + ir

    conn, ckind, cvane, cr, cspan = [], [], [], [], []
    for iz in range(nz):
        kd = kinds[iz]
        for it in range(nth):
            loc = it % per
            is_vane = loc < fs.n_vc
            for ir in range(nr):
                rc = 0.5 * (r_nodes[ir] + r_nodes[ir + 1])
                if kd == "channel" and not (is_vane and rc >= fs.r_vane_in):
                    continue
                conn.append([nid(ir, it, iz), nid(ir + 1, it, iz), nid(ir + 1, it + 1, iz),
                             nid(ir, it + 1, iz), nid(ir, it, iz + 1), nid(ir + 1, it, iz + 1),
                             nid(ir + 1, it + 1, iz + 1), nid(ir, it + 1, iz + 1)])
                ckind.append({"back": 0, "channel": 1, "front": 2}[kd])
                cvane.append(it // per if kd == "channel" else -1)
                cr.append(rc)
                cspan.append((rc - fs.r_vane_in) / (fs.b - fs.r_vane_in))
    conn = np.array(conn, dtype=np.int64)
    used = np.unique(conn)
    remap = -np.ones(nth * (nr + 1) * (nz + 1), dtype=np.int64)
    remap[used] = np.arange(used.size)
    conn = remap[conn]
    iz_g = used // (nth * (nr + 1))
    rem = used % (nth * (nr + 1))
    it_g = rem // (nr + 1)
    ir_g = rem % (nr + 1)
    R, TH, Z = r_nodes[ir_g], theta[ir_g, it_g], zb[iz_g]
    coors = np.stack([R * np.cos(TH), R * np.sin(TH), Z], axis=1)
    # 육면체 방향(오른손) 확인 — 음의 야코비안이면 꼭짓점 순서를 뒤집는다
    e = coors[conn]
    v = np.einsum("ij,ij->i", np.cross(e[:, 1] - e[:, 0], e[:, 3] - e[:, 0]), e[:, 4] - e[:, 0])
    if np.any(v < 0):
        if np.all(v < 0):
            conn = conn[:, [0, 3, 2, 1, 4, 7, 6, 5]]
        else:
            raise RuntimeError("요소 방향이 섞였다 — 격자 기울기 점검")
    info = {"n_elem": int(conn.shape[0]), "n_node": int(coors.shape[0]),
            "elem_desc": r3.elem_desc(coors, conn),
            "grid_idx": np.stack([ir_g, it_g, iz_g], axis=1),
            "cell_kind": np.array(ckind), "cell_vane": np.array(cvane),
            "cell_r": np.array(cr), "cell_span": np.array(cspan)}
    return coors, conn, info


def skew_deg(fs: FitSpec) -> float:
    """로그나선 캠버의 일정 블레이드 기울기(요소 기울기) [deg]."""
    return math.degrees(math.atan(math.radians(fs.wrap_deg) / math.log(fs.b / fs.r_vane_in)))


def vane_thin_coors(coors: np.ndarray, grid_idx: np.ndarray, fs: FitSpec,
                    vane: int = 0, span=(0.0, 0.25), frac: float = 0.6) -> np.ndarray:
    """베인 `vane`의 스팬 창(입구 = 0, (r − r_vin)/(b − r_vin))에서 두께를 (1 − frac)배로 — 캠버 기준 대칭.

    b5 `impeller_cad.profile_polygon`과 같은 손상 이상화(창 안 반두께 × (1 − frac)). 베인 발자국의 방위 절점
    (섹터 국소 번호 0 … n_vc)을 그 링의 캠버 방위 쪽으로 당긴다 — 모든 z층을 함께 옮기므로 판 두께·판 형상은
    그대로이고 판 셀은 방위로만 늘거나 준다(접합 폭도 같은 비율로 준다). 다른 베인·베이 절점은 움직이지 않는다.
    """
    if frac == 0.0:
        return coors.copy()
    per = fs.n_vc + fs.n_bay
    it = grid_idx[:, 1]
    r = np.hypot(coors[:, 0], coors[:, 1])
    s_r = (r - fs.r_vane_in) / (fs.b - fs.r_vane_in)
    sel = ((it // per) == vane) & ((it % per) <= fs.n_vc) & (r >= fs.r_vane_in - 1e-12) \
        & (s_r >= span[0] - 1e-9) & (s_r <= span[1] + 1e-9)
    c = coors.copy()
    th = np.arctan2(c[sel, 1], c[sel, 0])
    th_c = 2 * math.pi * vane / fs.n_vane + fs.camber(r[sel])
    phi = np.angle(np.exp(1j * (th - th_c)))
    th_new = th_c + (1.0 - frac) * phi
    rr = np.hypot(c[sel, 0], c[sel, 1])
    c[sel, 0] = rr * np.cos(th_new)
    c[sel, 1] = rr * np.sin(th_new)
    return c
