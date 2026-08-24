"""기하·물성 상수와 유효물성 — 논문3.JSV 전 모듈의 단일 출처.

정본 §3.1의 이상화를 그대로 상수화한다.
  (i)  베인 → Euler–Bernoulli 캔틸레버 L=30 mm, h=**1.0 mm**(실측 판두께), E=193 GPa,
       ρ=8000 kg/m³ (해석 f₁=881.6 Hz)
  (ii) 슈라우드 → 내경클램프·외경자유 환형 Kirchhoff 판, a=15.4 mm, b=36.56 mm (b−a≈21.2 mm)

**실측 2치수 확정(2026-08-13, 사용자 제공 캘리퍼값)**
    판두께  t_f = t_back = t_vane = 1.0 mm  (프레스 판재, 전 구간 동일)
    유로폭  b₂ = 4.1 mm                    (전·후면 슈라우드 사이 내부 유로)
  ⇒ 림 전체두께 = b₂ + 2 t_f = **6.1 mm**,  중립면 간격 s = b₂ + t_f = **5.1 mm**

**이전 해석의 폐기**: 2026-08-01 세션은 사용자 지시로 "4.1 mm = 림 **전체** 두께"로 가정했고
(t_face 0.8 / s 4.1 / 베인 h 1.2 / 레일 t 1.6 mm) 그 해석은 이제 폐기됐다 — 4.1 mm는 전체가
아니라 **유로폭**이었다. 두께에 걸린 결론(절대주파수·측정가능 대역·베인 h/L)은 전부 갱신했고,
두께에 무관한 결론(커널 γ^K·γ^M, 부호전환 반경, 식별성, 샌드위치 결합법칙)은 불변임을
수치로 확인했다(설계서 §11.17 F58–F62).

**레일 기하**(설계서 §5.2): 섭동맵(P)·2D 완전해(R2)·3D 솔리드(R3)가 같은 물리대상을 서술해야
model-form/이산화 차이를 분리할 수 있으므로, 공통 기하를 **균일두께 환형판** `DISK`
(t = 2 t_face → 면적질량이 샌드위치와 동일)로 고정한다. 균일 D·ρh에서 모드형과 γ_m(r)은
두께에 무관하므로(테스트 T3) 식별성 결론은 샌드위치로 그대로 이전된다. 샌드위치 유효물성
`SANDWICH`는 **절대주파수 사상**에만 쓴다(정본 §3.7).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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
    """샌드위치 환형판 유효물성 — D_eff = E t_f s²/[2(1−ν²)], ρh_eff = 2ρ t_f.

    두 면판(두께 t_face)이 간격 sep으로 떨어져 전단연결된 이상화(논문1 §2와 동일 정의).
    """

    a: float
    b: float
    t_face: float
    sep: float
    E: float
    rho: float
    nu: float

    @property
    def extent(self) -> float:
        return self.b - self.a

    @property
    def D(self) -> float:
        return self.E * self.t_face * self.sep ** 2 / (2.0 * (1.0 - self.nu ** 2))

    @property
    def rhoh(self) -> float:
        return 2.0 * self.rho * self.t_face


#: 실측 판두께 t_f = t_back = t_vane [m] — 프레스 판재라 전 구간 동일(2026-08-13 실측).
T_SHEET = 0.0010
#: 실측 유로폭 b₂ [m] — 전·후면 슈라우드 사이 내부 유로(2026-08-13 실측).
B2_CHANNEL = 0.0041
#: 림 전체두께 [m] = b₂ + 2 t_f (파생값, 실측 아님).
RIM_TOTAL = B2_CHANNEL + 2 * T_SHEET
#: 면판 중립면 간격 [m] = b₂ + t_f (파생값). 샌드위치 D_eff ∝ t_f·s²의 s.
FACE_SEPARATION = B2_CHANNEL + T_SHEET

#: 베인(정본 §3.1-i). 베인도 같은 판재이므로 h = t_f. ρ=8000은 정본 표기값.
VANE = Beam(L=0.030, h=T_SHEET, E=193e9, rho=8000.0, nu=0.29)

#: 레일 공통 환형판(설계서 §5.2). a·b는 논문1 disk GEO와 동일, ρ=7930(SUS304).
DISK = Plate(a=0.0154, b=0.03656, t=2 * T_SHEET, E=193e9, rho=7930.0, nu=0.29)

#: 절대주파수 사상용 샌드위치(정본 §3.7). t_face·sep은 이제 **실측 파생값**이다.
SANDWICH = Sandwich(a=0.0154, b=0.03656, t_face=T_SHEET, sep=FACE_SEPARATION,
                    E=193e9, rho=7930.0, nu=0.29)
