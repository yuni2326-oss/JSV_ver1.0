"""§3.6-ii — 폐쇄형 임펠러 파라메트릭 CAD(gmsh OCC) + 3D 모달.

**형상 근거**(2026-08-01 사용자 제공 실측 사진·사양 + 2026-08-13 캘리퍼 2치수):
  D₂ = 73.12 mm (R₂ = 36.56), 흡입구 D₁ = 30.80 mm (R₁ = 15.40), 흡입 넥 높이 8.95 mm,
  베인 **6매**(전면 슈라우드의 6줄 스폿용접 궤적으로 확인), 후곡형(backward-curved),
  **프레스 판재라 두께 일정 t = 1.0 mm(실측)**, 내부 **유로폭 b₂ = 4.1 mm(실측)**
  ⇒ 림 전체두께 = 6.1 mm, 6-spline 보어 Φ12–14.

**해석 정정(2026-08-13)**: 이전 세션은 사용자 지시로 "4.1 mm = 림 **전체** 두께"로 가정해
유로높이를 1.65 mm(= 4.1 − 1.1 − 1.35)로 잡았다. 실측으로 4.1 mm는 **유로폭**임이 확정돼
유로높이가 **4.1 mm(2.5배)**, 전체두께가 6.1 mm가 됐다. 파라미터화도 `gap`을 **직접**
받도록 바꿨다(전체두께에서 판두께를 빼는 방식은 두 실측치수 중 하나를 파생값으로 만들어
같은 오해를 재발시킨다).

**라벨 규칙**: 확정 치수(외경·흡입구경·판두께·유로폭·베인 수)는 고정하고, 여전히 미확정인
것(베인 입구반경·wrap 각·필렛·스폿용접 이산성)은 **스윕**한다. 논문에는
"as-built 치수 + 미확정 형상 파라미터 스윕"으로 명시한다(설계서 §11.17 F62).

판재 구조라 3D 모델은 **얇은 판 3장(전면 슈라우드 + 베인 + 후면 슈라우드)** 이다.
베인은 테이퍼 없이 등두께 곡면이므로, r–θ 평면의 곡선 프로파일을 z로 압출하면 실제와 맞는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImpellerSpec:
    """폐쇄형 임펠러 대리형상 파라미터 (SI, m)."""

    r_out: float = 0.03656          # R₂ (실측 D₂/2)
    r_eye: float = 0.01540          # R₁ 흡입구 (실측 D₁/2)
    channel_gap: float = 0.0041     # 유로폭 b₂ (**실측**) = 전·후면 슈라우드 사이 높이
    t_front: float = 0.0010         # 전면 슈라우드 판 두께 (**실측** 1.0)
    t_back: float = 0.0010          # 후면 슈라우드 판 두께 (**실측** 1.0)
    n_vane: int = 6                 # 스폿용접 궤적으로 확인
    t_vane: float = 0.0010          # 베인 판 두께 (**실측** 1.0, 같은 판재)
    r_vane_in: float = 0.0170       # 베인 입구 반경 (**미확정**, 스윕 16–19 mm)
    wrap_deg: float = 90.0          # 후곡 wrap 각 (**미확정**, 스윕 60–120°)
    damage_vane: int = -1           # 손상 베인 인덱스(-1 = 건전)
    damage_frac: float = 0.0        # 국소 두께 감소율 (0.6 = 60 % 얇아짐)
    damage_span: tuple = (0.0, 0.25)  # 손상 구간(베인 스팬 분율, 0=입구/뿌리)
    E: float = 193e9
    rho: float = 7930.0
    nu: float = 0.29

    @property
    def gap(self) -> float:
        """전·후면 슈라우드 사이 유로 높이 = 실측 유로폭 b₂ (직접 파라미터)."""
        return self.channel_gap

    @property
    def total_thickness(self) -> float:
        """림 끝단 전체 두께 = b₂ + t_front + t_back (**파생값**)."""
        return self.channel_gap + self.t_front + self.t_back

    @property
    def face_separation(self) -> float:
        """면판 중립면 간격 s = b₂ + (t_front + t_back)/2 — 샌드위치 D_eff ∝ t_f s²의 s."""
        return self.channel_gap + 0.5 * (self.t_front + self.t_back)

    def check(self) -> None:
        if self.gap <= 0:
            raise ValueError(f"gap<=0: 유로폭 {self.channel_gap*1e3:.2f} mm")


def vane_camber(spec: ImpellerSpec, k: int, n_pts: int = 24):
    """k번째 베인의 후곡 캠버선 (로그나선). 반환 (x[n], y[n])."""
    th0 = 2 * math.pi * k / spec.n_vane
    r = np.linspace(spec.r_vane_in, spec.r_out, n_pts)
    # 로그나선: θ = θ0 + wrap · ln(r/r_in)/ln(r_out/r_in)  (후곡 = 회전 반대방향)
    frac = np.log(r / spec.r_vane_in) / math.log(spec.r_out / spec.r_vane_in)
    th = th0 - math.radians(spec.wrap_deg) * frac
    return r * np.cos(th), r * np.sin(th)


def profile_polygon(spec: ImpellerSpec, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k번째 베인의 **단면 다각형**(z = t_back 평면) — 캠버선 ± 반두께 오프셋.

    손상 베인이면 `damage_span` 창 안의 반두께를 (1 − `damage_frac`)배로 줄인다. 조립체와
    고립 베인이 **이 함수 하나**를 공유하므로 두 모델의 손상 이상화가 구조적으로 동일하다
    (a19 정합-기하 사다리의 전제 — 재구현하면 그 전제가 깨진다).

    반환 다각형은 [뿌리→팁 (+오프셋), 팁→뿌리 (−오프셋)] 순서라 i번째 점의 짝은 2n−1−i다.
    """
    xs, ys = vane_camber(spec, k)
    tx, ty = np.gradient(xs), np.gradient(ys)
    nn = np.hypot(tx, ty)
    nx, ny = -ty / nn, tx / nn
    h = np.full(len(xs), 0.5 * spec.t_vane)
    if k == spec.damage_vane and spec.damage_frac > 0:
        # 국소 두께감소 = 프레스 판재 베인의 균열/침식 대리. 뿌리쪽 구간에 적용.
        f = np.linspace(0.0, 1.0, len(xs))
        win = (f >= spec.damage_span[0]) & (f <= spec.damage_span[1])
        h[win] *= (1.0 - spec.damage_frac)
    px = np.concatenate([xs + h * nx, (xs - h * nx)[::-1]])
    py = np.concatenate([ys + h * ny, (ys - h * ny)[::-1]])
    return px, py


def clamp_vane_root(spec: ImpellerSpec, k: int = 0, tol: float = 1e-6):
    """고립 베인의 **내경단 면** 클램프 — 캠버 시작점의 접선평면(함수 선택자).

    반경 조건으로는 고를 수 없다: 끝면의 두 꼭점은 캠버 법선 방향으로 벌어져 있어
    반경이 최대 t_vane만큼 다르다. 끝면은 접선에 수직한 평면이므로 부호거리로 판정한다.
    """
    xs, ys = vane_camber(spec, k)
    t = np.array([xs[1] - xs[0], ys[1] - ys[0]])
    t /= np.linalg.norm(t)
    p0 = np.array([xs[0], ys[0]])

    def fn(coors):
        return np.nonzero(np.abs((coors[:, :2] - p0) @ t) <= tol)[0]
    return fn


def clamp_vane_shroud_faces(spec: ImpellerSpec, tol: float = 1e-9):
    """고립 베인의 **양 슈라우드 접합면**(z = t_back, t_back + b₂) 클램프.

    슈라우드를 **강체로 둔 극한** — 조립체와의 차이가 슈라우드 컴플라이언스와 하중분담뿐이
    되도록 기하·손상을 고정한 채 경계만 바꾸는 사다리 한 단이다.
    """
    lo, hi = spec.t_back, spec.t_back + spec.gap

    def fn(coors):
        z = coors[:, 2]
        return np.nonzero((np.abs(z - lo) <= tol) | (np.abs(z - hi) <= tol))[0]
    return fn


#: 육면체(`rail3d._HEX_OFFSETS` 순서)를 절점 0–6 대각으로 가르는 6-사면체 분해.
#: 평행육면체·상자에서는 **정확**하고, 겹면이 비평면인 셀에서는 O(h²) 근사다.
_HEX_TETS = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
             (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))


def cell_volumes(coors, conn) -> np.ndarray:
    """셀별 체적(사면체 4절점·육면체 8절점) — 구역 마스크가 고른 것이 같은 물리 구역인지
    대조하는 데 쓴다. 다른 요소형은 조용히 틀린 값을 주지 않고 거절한다."""
    def _tet(p):
        return np.abs(np.einsum("ij,ij->i",
                                np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                                p[:, 3] - p[:, 0])) / 6.0

    p = coors[conn]
    if conn.shape[1] == 4:
        return _tet(p)
    if conn.shape[1] == 8:
        return sum(_tet(p[:, list(t)]) for t in _HEX_TETS)
    raise ValueError(f"지원하지 않는 요소형: 절점 {conn.shape[1]}개(4 또는 8만)")


def camber_frame(spec: ImpellerSpec, x, y, k: int = 0):
    """(x, y)에서의 캠버 **국소 정규직교 기저** (t̂ 접선, n̂ 법선) — 로그나선의 닫힌형.

    p(r) = r(cos θ, sin θ), θ(r) = θ_k − wrap·ln(r/r_in)/ln(r_out/r_in) 이므로
    dp/dr = r̂ + c θ̂ 이고 **c = −wrap_rad / ln(r_out/r_in)은 상수**다(반경 무관).
    따라서 t̂ = (r̂ + c θ̂)/√(1+c²), n̂ = (−c r̂ + θ̂)/√(1+c²).
    """
    c = -math.radians(spec.wrap_deg) / math.log(spec.r_out / spec.r_vane_in)
    ph = np.arctan2(y, x)
    rhat = np.stack([np.cos(ph), np.sin(ph), np.zeros_like(ph)], axis=-1)
    that = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], axis=-1)
    s = 1.0 / math.sqrt(1.0 + c * c)
    return (rhat + c * that) * s, (-c * rhat + that) * s


def vane_mode_kinds(spec: ImpellerSpec, res, k: int = 0):
    """곡면 베인 모드를 **캠버 국소좌표**로 분류 — 반환 (kinds, part[n_modes, 3]).

    라벨은 "chord"(접선 t̂) / "flap"(법선 n̂) / "span"(z, 슈라우드 사이). 직선 레일의
    `rail3d.beam_mode_kinds`는 데카르트 성분으로 골랐지만 후곡 베인은 굽힘 방향이
    반경마다 회전하므로 그대로 쓰면 오분류한다. 정본 §3.6이 금지한 "주파수 순서 매칭"을
    피하기 위한 사다리 쪽 장치다(설계서 3D 함정 ③).
    """
    if res.full_shapes is None or res.field_coors is None:
        raise RuntimeError("solve_modes(..., keep_shapes=True) 필요")
    c = res.field_coors
    that, nhat = camber_frame(spec, c[:, 0], c[:, 1], k)
    labels = ("chord", "flap", "span")
    kinds, part = [], []
    for j in range(res.full_shapes.shape[0]):
        u = res.full_shapes[j].reshape(-1, 3)
        e = np.array([((u * that).sum(axis=1) ** 2).sum(),
                      ((u * nhat).sum(axis=1) ** 2).sum(),
                      (u[:, 2] ** 2).sum()])
        e = e / e.sum()
        part.append(e)
        kinds.append(labels[int(np.argmax(e))])
    return kinds, np.array(part)


def damaged_cell_mask(spec: ImpellerSpec, coors, conn, k: int = 0) -> np.ndarray:
    """`damage_span` 창에 드는 **베인 k의 셀**(요소중심 판정) — 구역 에너지의 마스크.

    손상 크기(`damage_frac`)와 **무관하게 기하로만** 판정한다: 1차 감도는 *건전* 구조에서
    그 구역이 지탱하는 모드 에너지 분율이므로 건전 메시에 같은 창을 씌워야 한다.
    조립체 메시에도 그대로 쓸 수 있도록 세 조건을 모두 본다 — (i) 유로층 z 밴드(슈라우드
    셀 배제), (ii) 베인 k 캠버선 근접(다른 베인 배제), (iii) 반경분율 창.
    """
    c = coors[conn].mean(axis=1)
    rr = np.maximum(np.hypot(c[:, 0], c[:, 1]), 1e-12)
    frac = np.log(rr / spec.r_vane_in) / math.log(spec.r_out / spec.r_vane_in)
    th_k = 2 * math.pi * k / spec.n_vane - math.radians(spec.wrap_deg) * frac
    dth = (np.arctan2(c[:, 1], c[:, 0]) - th_k + math.pi) % (2 * math.pi) - math.pi
    f_r = (rr - spec.r_vane_in) / (spec.r_out - spec.r_vane_in)
    return ((np.abs(dth) * rr <= spec.t_vane)
            & (c[:, 2] > spec.t_back) & (c[:, 2] < spec.t_back + spec.gap)
            & (f_r >= spec.damage_span[0]) & (f_r <= spec.damage_span[1]))


def write_geo(spec: ImpellerSpec, path: str, mesh_size: float = 0.0012, *,
              include_shrouds: bool = True, vanes=None) -> str:
    """OpenCASCADE `.geo` 스크립트를 쓴다 (gmsh **CLI** 경로 — python3-gmsh 불필요).

    `include_shrouds=False`, `vanes=(k,)`로 부르면 **베인 k 하나만** 있는 고립 형상이 된다
    (a19 R1·R2). 기하·손상은 조립체와 같은 코드에서 나온다.
    """
    spec.check()
    idx = tuple(range(spec.n_vane)) if vanes is None else tuple(vanes)
    L = [f'SetFactory("OpenCASCADE");',
         f"Mesh.MeshSizeMax = {mesh_size};",
         f"Mesh.MeshSizeMin = {mesh_size*0.35};",
         "Mesh.Algorithm3D = 1;"]
    tb, tf, gap, ro, re = spec.t_back, spec.t_front, spec.gap, spec.r_out, spec.r_eye
    zf = tb + gap
    if include_shrouds:
        # 후면 원판(중심 보어 제거) / 전면 환형판(흡입구 개구)
        L += [f"Cylinder(1) = {{0,0,0, 0,0,{tb}, {ro}}};",
              f"Cylinder(2) = {{0,0,{-1e-3}, 0,0,{tb+2e-3}, 0.0065}};",
              "BooleanDifference(10) = { Volume{1}; Delete; }{ Volume{2}; Delete; };",
              f"Cylinder(3) = {{0,0,{zf}, 0,0,{tf}, {ro}}};",
              f"Cylinder(4) = {{0,0,{zf-1e-3}, 0,0,{tf+2e-3}, {re}}};",
              "BooleanDifference(11) = { Volume{3}; Delete; }{ Volume{4}; Delete; };"]
    pid, cid = 100000, 200000   # OCC 자동태그와 충돌 방지용 고대역
    # Extrude는 신규 엔티티 태그를 자동할당해 다음 베인 대역을 잠식하므로,
    # **점·곡선·면을 전부 먼저 만들고 압출은 맨 끝에 모아서** 수행한다.
    ext = []
    for k in idx:
        px, py = profile_polygon(spec, k)
        p0 = pid
        for a, b in zip(px, py):
            L.append(f"Point({pid}) = {{{a:.7f}, {b:.7f}, {tb}, {mesh_size}}};")
            pid += 1
        L += [f"Spline({cid}) = {{{p0}:{pid-1}}};",
              f"Line({cid+1}) = {{{pid-1}, {p0}}};",
              f"Curve Loop({cid+2}) = {{{cid}, {cid+1}}};",
              f"Plane Surface({cid+3}) = {{{cid+2}}};"]
        ext.append(cid + 3)
        cid += 100
    for k, sid in enumerate(ext):
        L.append(f"ext{k}[] = Extrude {{0,0,{gap}}} {{ Surface{{{sid}}}; }};")
    # 용접 조립체 = 접합면을 공유하는 conformal 분할 (체적이 하나면 분할할 것이 없다)
    if 2 * include_shrouds + len(idx) > 1:
        L += ["BooleanFragments{ Volume{:}; Delete; }{}"]
    L += ['Physical Volume("all") = { Volume{:} };']
    open(path, "w").write("\n".join(L) + "\n")
    return path


def build_geometry(spec: ImpellerSpec, mesh_size: float = 0.0012,
                   workdir: str = "/tmp", tag: str = "impeller", *,
                   include_shrouds: bool = True, vanes=None):
    """`.geo` → gmsh CLI → `.msh` → meshio 로 (coors, tet conn) 반환."""
    import subprocess

    import meshio
    geo = f"{workdir}/{tag}.geo"
    msh = f"{workdir}/{tag}.msh"
    write_geo(spec, geo, mesh_size=mesh_size, include_shrouds=include_shrouds,
              vanes=vanes)
    r = subprocess.run(["gmsh", "-3", geo, "-format", "msh2", "-o", msh],
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0 or not __import__("os").path.exists(msh):
        raise RuntimeError(f"gmsh 실패: {r.stdout[-800:]}\n{r.stderr[-800:]}")
    m = meshio.read(msh)
    tet = None
    for cb in m.cells:
        if cb.type == "tetra":
            tet = cb.data.astype(np.int32)
    if tet is None:
        raise RuntimeError("사면체 요소 없음")
    used = np.unique(tet)
    remap = -np.ones(len(m.points), dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return np.asarray(m.points, dtype=float)[used], remap[tet]


def solve_modal(spec: ImpellerSpec, coors, conn, n_modes: int = 12, order: int = 2,
                clamp_radius: float | None = None,
                region_mask: np.ndarray | None = None):
    """보어(내경) 클램프 조건의 3D 모달. clamp_radius 이내 절점을 고정.

    `region_mask`는 `rail3d.solve_modes`로 그대로 넘어간다 — 조립체 단의 구역 에너지
    분율(a19 R3)이 이 경로로만 나오므로 배선을 테스트로 고정한다.
    """
    from . import rail3d as r3
    rc = clamp_radius if clamp_radius is not None else 0.0075

    def fixed(c):
        return np.nonzero(np.hypot(c[:, 0], c[:, 1]) <= rc)[0]

    return r3.solve_modes(coors, conn, spec.E, spec.nu, spec.rho, fixed,
                          n_modes=n_modes, order=order, keep_shapes=True,
                          region_mask=region_mask)


def vane_localization(res, coors, spec: ImpellerSpec) -> np.ndarray:
    """모드별 **베인 국재화 지표** — 정본 §3.6-ii의 핵심 산출물.

    각 모드에서 변위 에너지가 몇 개의 베인 섹터에 몰려 있는가를 참여비의 역참여수
    (participation ratio)로 잰다. 값 ≈ n_vane이면 전(全)베인 균등(순환대칭), ≈1이면 완전 국재화.
    단일베인 지문이 조립체에서 살아남으려면 국재화가 충분히 커야 한다.
    """
    if res.full_shapes is None:
        raise RuntimeError("keep_shapes=True 필요")
    c = res.field_coors
    th = np.arctan2(c[:, 1], c[:, 0]) % (2 * math.pi)
    sector = np.floor(th / (2 * math.pi / spec.n_vane)).astype(int)
    out = []
    for k in range(res.full_shapes.shape[0]):
        u = res.full_shapes[k].reshape(-1, 3)
        e = (u ** 2).sum(axis=1)
        w = np.array([e[sector == j].sum() for j in range(spec.n_vane)])
        w = w / max(w.sum(), 1e-300)
        out.append(1.0 / np.sum(w ** 2))          # 역참여수
    return np.array(out)
