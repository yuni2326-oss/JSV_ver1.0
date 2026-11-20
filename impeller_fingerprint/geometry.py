"""기하·물성 상수와 유효물성 — 논문3.JSV 전 모듈의 단일 출처.

정본 §3.1의 이상화를 상수화한다.
  (i)  베인 → Euler–Bernoulli 캔틸레버 L = 30 mm, h = **1.0 mm**(깃 두께), 폭 3.0 mm(깃 높이 = 판 사이 간격),
       E = 193 GPa, ρ = 8000 kg/m³ (해석 f₁ = 881.6 Hz)
  (ii) 슈라우드 → 내경클램프·외경자유 환형 Kirchhoff 판(레일), a = 15.4 mm, b = 36.56 mm (b − a ≈ 21.2 mm)

**2026-10-30 정본(G3) = 표준 판재 두께**(사용자 결정, `docs/paper3-jsv/revision-r1/31-…`·`32-…`·`G3-recompute-plan.md`):
    앞판 t_front = 0.40 mm, 뒤판 t_back = 0.60 mm(표준 두께; 2번 시편 캘리퍼 0.40·0.58 mm, 해머 시험 주파수가 가리키는
    두께 앞 0.41 ± 0.01·뒤 0.59–0.62 mm), 판 사이 간격 GAP = 3.00 mm(2번 시편 출구 끝 직접값 2.95–3.00), 깃 1.0 mm,
    외경 72.93, 흡입구 테 바깥지름 30.82·벽 0.52(→ 흡입구 반경 14.89), 보어 12.62 mm, 테 높이 5.0 mm(사진 추정), 질량 41.32 g.
    시험 시편이 2번 시편(1번 시편은 앞판 테가 찌그러져 은퇴)이라 2번 시편 치수로 맞췄다.

**G1(2026-10-16, 1번 시편 재측정 — r4.1 정본)**은 `ASBUILT_G1`·`VANE_WIDTH_G1`에 남긴다
(캘리퍼·분석저울, `docs/paper3-jsv/revision-r1/G1-geometry-remeasure.md`):
    앞판(흡입구 쪽 = 손상면) t_front = 0.375 mm, 뒤판(허브 쪽) t_back = 0.59 mm
    판 사이 간격 3.10(출구)–3.27 mm(흡입구 쪽) → GAP = 3.18 mm(평균; 일정 간격 이상화)
    깃 두께 1.2 mm(높이 = 간격), 외경 73.03, 흡입구 안지름 29.91, 보어 12.67 mm, 질량 40.21 g
    앞판 용접 자국의 베인각 50.5° → wrap ≈ 62°(r_vin 15 mm)

**08-13 기하의 폐기**: 양면 1.0 mm·유로 4.1 mm·깃 1.0 mm는 부품 질량 68.7 g을 내 실측 40.21 g과 맞지 않는다.
제출본(r3.9)·r4.0 산출물 재현용으로 `GEOM_0813`에 남긴다(`GEOM_0813.disk()` 등).

**레일 기하**(설계서 §5.2): 섭동맵(P)·2D 완전해(R2)·3D 솔리드(R3)가 같은 물리대상을 서술하도록 공통 기하를
균일두께 환형판 `DISK`(t = t_front + t_back → 면적질량이 실물 두 면판과 같다)로 고정한다. 균일 D·ρh에서 모드형과
γ_m(r)은 두께에 무관하므로(테스트 T3) 식별성 결론은 두께 정정에 영향을 받지 않는다. 레일 반경 a·b는 사진 실측
15.40·36.56 mm를 유지한다 — 캘리퍼 실측 14.955·36.515 mm로 바꿔도 CRLB가 −4.1 ~ +0.03 %, ξ*가 +0.001–0.004
움직일 뿐이다(G2 결정 D1). 샌드위치 유효물성 `SANDWICH`는 **절대주파수 사상**에만 쓴다(정본 §3.7).
실측 형상 조립체(N1·N2·자유-자유·b5·a14·a19·Fig. 1)의 치수는 `ASBUILT`다.

**G4(2026-10-31, 계산 비교용 — 사용자 지시)** = G3 + 깃 높이(판 사이 간격) 테이퍼: 출구 끝(r = b) 3.00 mm에서
흡입구 쪽(r = a) 3.19 mm까지 선형(2번 시편 전체 두께 출구 끝 4.00·깊은 곳 4.19 mm, 1번 시편 4.07/4.12/4.23 mm,
판 두께는 표준 그대로; `docs/paper3-jsv/revision-r1/38-gap-taper-2026-10-31.md`). `ASBUILT_G4`에 두고, 환경변수
`PAPER3_GEOM=g4`인 프로세스에서만 `ASBUILT`가 된다(출력 루트 `PAPER3_OUT`과 함께 쓴다). 테이퍼는 3D 조립체(N1·N2·N6)의
조립 좌표에만 들어간다(`rev1.n1_mesh.FitSpec.taper_coors`) — 레일·두 면판 식·샌드위치 사상·깃 쿠폰은 출구 끝 값 `GAP`을 쓴다.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

# 캔틸레버 무차원 고유값 (βL) 1~4차
BETA_L = (1.8751041, 4.6940911, 7.8547574, 10.9955407)


@dataclass(frozen=True)
class Beam:
    """등단면 Euler–Bernoulli 캔틸레버(베인 이상화)."""

    L: float
    h: float
    E: float
    rho: float
    nu: float

    @property
    def omega_b(self) -> float:
        """주파수 스케일 ω_b = (h/√12)·√(E/ρ)/L²  (f_n = (βL_n)²·ω_b/2π)."""
        return (self.h / math.sqrt(12.0)) * math.sqrt(self.E / self.rho) / self.L ** 2

    def eb_frequencies(self, n: int = 3) -> list[float]:
        """해석 EB 캔틸레버 고유주파수[Hz] n개."""
        return [(b ** 2) * self.omega_b / (2 * math.pi) for b in BETA_L[:n]]


@dataclass(frozen=True)
class Plate:
    """균일두께 환형 Kirchhoff 판 — 내경 클램프, 외경 자유."""

    a: float
    b: float
    t: float
    E: float
    rho: float
    nu: float

    @property
    def extent(self) -> float:
        return self.b - self.a

    @property
    def D(self) -> float:
        return self.E * self.t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    @property
    def rhoh(self) -> float:
        return self.rho * self.t


@dataclass(frozen=True)
class Sandwich:
    """샌드위치 환형판 유효물성 — 두 면판(t_front, t_back)이 안쪽 간격 gap으로 떨어져 전단연결된 이상화.

    얇은면판 닫힌형: D_eff = E μ d²/(1−ν²), μ = t_f t_b/(t_f + t_b), d = gap + (t_f + t_b)/2(면 중심 간격),
    ρh_eff = ρ(t_f + t_b). 같은 두께(t_f = t_b = t)면 D_eff = E t s²/[2(1−ν²)], s = gap + t
    (논문1 §2·08-13 정의)와 같다.
    """

    a: float
    b: float
    t_front: float
    t_back: float
    gap: float
    E: float
    rho: float
    nu: float

    @property
    def extent(self) -> float:
        return self.b - self.a

    @property
    def sep(self) -> float:
        """두 면판 중심 사이 거리 d = gap + (t_f + t_b)/2."""
        return self.gap + 0.5 * (self.t_front + self.t_back)

    @property
    def mu(self) -> float:
        """두 면판의 환산 두께 μ = t_f t_b/(t_f + t_b) (같은 두께면 t/2)."""
        return self.t_front * self.t_back / (self.t_front + self.t_back)

    @property
    def D(self) -> float:
        return self.E * self.mu * self.sep ** 2 / (1.0 - self.nu ** 2)

    @property
    def rhoh(self) -> float:
        return self.rho * (self.t_front + self.t_back)


#: G3 정본 [m] — 앞판(흡입구 쪽, 손상면)·뒤판(허브 쪽) 표준 두께, 판 사이 간격, 깃 두께.
T_FRONT = 0.40e-3
T_BACK = 0.60e-3
GAP = 3.00e-3
T_VANE = 1.0e-3
#: 베인 쿠폰 폭 [m] — 깃 높이(= 판 사이 간격).
VANE_WIDTH = 3.0e-3
#: 면판 중심 간격 d [m] (파생값).
FACE_SEPARATION = GAP + 0.5 * (T_FRONT + T_BACK)
#: 전체 두께 [m] (파생값 4.00 mm; 2번 시편 전체 두께 4.19 mm는 간격이 더 넓은 안쪽 값으로 본다).
RIM_TOTAL = GAP + T_FRONT + T_BACK

#: 베인(정본 §3.1-i). ρ = 8000은 정본 표기값.
VANE = Beam(L=0.030, h=T_VANE, E=193e9, rho=8000.0, nu=0.29)

#: 레일 공통 환형판(설계서 §5.2). a·b는 논문1 disk GEO와 동일(사진 실측), ρ = 7930(SUS304).
DISK = Plate(a=0.0154, b=0.03656, t=T_FRONT + T_BACK, E=193e9, rho=7930.0, nu=0.29)

#: 절대주파수 사상용 샌드위치(정본 §3.7).
SANDWICH = Sandwich(a=0.0154, b=0.03656, t_front=T_FRONT, t_back=T_BACK, gap=GAP,
                    E=193e9, rho=7930.0, nu=0.29)


@dataclass(frozen=True)
class AsBuilt:
    """실측 형상 조립체의 정본 치수 [m] (G3: 표준 두께 + 2번 시편 치수). 베인–판 접합은 연속으로 이상화한다."""

    a: float = 0.01489             # 흡입구 반경(테 바깥지름 30.82/2 − 벽 0.52)
    b: float = 0.036465            # 외경 반지름(72.93/2)
    t_front: float = T_FRONT
    t_back: float = T_BACK
    gap: float = GAP
    t_vane: float = T_VANE
    n_vane: int = 6
    wrap_deg: float = 62.0         # 앞판 용접 자국 베인각 50.5°, r_vin 15 mm
    r_vane_in: float = 0.0150
    r_bore: float = 0.00631        # 보어(6홈 별 모양) 12.62/2
    collar_t: float = 0.00052      # 흡입구 테 벽(테 끝에서 직접)
    collar_h: float | None = 0.0050  # 흡입구 테 높이(앞판 바깥면에서) — 사진 추정 4.6–5.0 mm
    mass_g: float = 41.3186        # 분석저울 실측(2번 시편)
    gap_in: float | None = None    # 흡입구 쪽(r = a) 간격 — None이면 일정(G3), 값이면 r = b의 gap에서 선형(G4)


ASBUILT = AsBuilt()
ASBUILT_G3 = ASBUILT
#: G4 — G3 + 깃 높이 테이퍼(출구 끝 3.00 → 흡입구 쪽 3.19 mm, 선형; 뒤판 평판·앞판 얕은 원뿔 0.5°). 모듈 문서 참고.
ASBUILT_G4 = replace(ASBUILT_G3, gap_in=3.19e-3)
if os.environ.get("PAPER3_GEOM", "g3").strip().lower() == "g4":
    ASBUILT = ASBUILT_G4

#: G1(2026-10-16, 1번 시편) — r4.1 산출물 재현용. 테 높이는 미측정이었고 N6은 4.0 mm(`FFSpec(collar_h=0.004)`)로 풀었다.
ASBUILT_G1 = AsBuilt(a=0.014955, b=0.036515, t_front=0.375e-3, t_back=0.59e-3, gap=3.18e-3, t_vane=1.2e-3,
                     r_bore=0.006335, collar_t=0.000425, collar_h=None, mass_g=40.2059)
VANE_WIDTH_G1 = 3.2e-3


@dataclass(frozen=True)
class Geom0813:
    """08-13 기하(폐기 — 질량 68.7 g 대 실측 40.21 g). r3.9·r4.0 산출물 재현용."""

    t_sheet: float = 0.0010
    b2_channel: float = 0.0041
    t_vane: float = 0.0010
    vane_width: float = 0.0041
    a: float = 0.0154
    b: float = 0.03656
    wrap_deg: float = 90.0
    r_vane_in: float = 0.0170

    @property
    def face_separation(self) -> float:
        return self.b2_channel + self.t_sheet

    @property
    def rim_total(self) -> float:
        return self.b2_channel + 2 * self.t_sheet

    def disk(self) -> Plate:
        return Plate(a=self.a, b=self.b, t=2 * self.t_sheet, E=193e9, rho=7930.0, nu=0.29)

    def vane(self) -> Beam:
        return Beam(L=0.030, h=self.t_vane, E=193e9, rho=8000.0, nu=0.29)

    def sandwich(self) -> Sandwich:
        return Sandwich(a=self.a, b=self.b, t_front=self.t_sheet, t_back=self.t_sheet,
                        gap=self.b2_channel, E=193e9, rho=7930.0, nu=0.29)


GEOM_0813 = Geom0813()
