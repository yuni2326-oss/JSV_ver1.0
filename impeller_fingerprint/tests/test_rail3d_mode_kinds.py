"""F181 회귀 — 폭이 넓은 쿠폰의 1차 비틀림을 굽힘으로 표시하지 않는다(`rail3d.beam_mode_kinds`)."""
import numpy as np
import pytest

from impeller_fingerprint import geometry as geo
from impeller_fingerprint import rail3d as r3

VANE = geo.VANE
#: F181이 발견된 08-13 쿠폰(30 × 4.1 × 1.0 mm) — 분류기 회귀는 그 형상에서 고정한다.
VANE_0813 = geo.GEOM_0813.vane()


@pytest.mark.slow
def test_wide_coupon_torsion_is_not_flap():
    V = VANE_0813
    c, cn, _ = r3.vane_mesh(L=V.L, w=geo.GEOM_0813.vane_width, h=V.h, nx=60, ny=4, nz=8)
    res = r3.solve_modes(c, cn, V.E, V.nu, V.rho, r3.clamp_root(),
                         n_modes=10, order=2, keep_shapes=True)
    kinds = r3.beam_mode_kinds(res)
    assert "torsion" in kinds
    flap = res.freqs[[i for i, k in enumerate(kinds) if k == "flap"]]
    # 면외 굽힘 1–3차는 Euler–Bernoulli 881.6 / 5525 / 15470 Hz 근처(3D 판효과 몇 %)
    assert abs(flap[0] / 881.6 - 1) < 0.03
    assert abs(flap[1] / 5525.0 - 1) < 0.05
    assert abs(flap[2] / 15470.0 - 1) < 0.06
    tors = res.freqs[[i for i, k in enumerate(kinds) if k == "torsion"]]
    assert 10e3 < tors[0] < 13e3          # 1차 비틀림 ≈ 11.4 kHz


@pytest.mark.slow
def test_measured_coupon_mode_kinds():
    """정본 깃 쿠폰 30 × 3.0 × 1.0 mm(G3): 면외 굽힘 1–2차 ≈ EB 881.6/5525 Hz, 1차 비틀림 ≈ 14.6 kHz.
    (G1 30 × 3.2 × 1.2 mm에서는 1057.9/6630 Hz, 비틀림 약 16 kHz였다.)"""
    c, cn, _ = r3.vane_mesh(L=VANE.L, w=geo.VANE_WIDTH, h=VANE.h, nx=60, ny=4, nz=8)
    res = r3.solve_modes(c, cn, VANE.E, VANE.nu, VANE.rho, r3.clamp_root(),
                         n_modes=10, order=2, keep_shapes=True)
    kinds = r3.beam_mode_kinds(res)
    flap = res.freqs[[i for i, k in enumerate(kinds) if k == "flap"]]
    assert abs(flap[0] / 881.6 - 1) < 0.03
    assert abs(flap[1] / 5525.0 - 1) < 0.05
    tors = res.freqs[[i for i, k in enumerate(kinds) if k == "torsion"]]
    assert len(tors) and 13e3 < tors[0] < 17e3


@pytest.mark.slow
def test_narrow_coupon_unchanged_order():
    """폭 2h 쿠폰: 면외 굽힘 세 개가 그대로 잡힌다(이전 분류와 같은 결과)."""
    c, cn, _ = r3.vane_mesh(L=VANE.L, w=2 * VANE.h, h=VANE.h, nx=60, ny=4, nz=8)
    res = r3.solve_modes(c, cn, VANE.E, VANE.nu, VANE.rho, r3.clamp_root(),
                         n_modes=10, order=2, keep_shapes=True)
    kinds = r3.beam_mode_kinds(res)
    flap = res.freqs[[i for i, k in enumerate(kinds) if k == "flap"]]
    assert len(flap) >= 3
    assert abs(flap[2] / (15470.0 * VANE.h / 1.0e-3) - 1) < 0.06     # f ∝ h
