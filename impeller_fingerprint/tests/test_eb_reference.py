"""EB 캔틸레버 기준해 재구현 — 논문1 없이도 a11·a8을 재생성할 수 있어야 한다.

배포본(코드·데이터)에는 논문1 패키지가 없으므로 `cli a11`(Table 1·arm 비교·수렴·폭한계)과
`cli a8`(습윤보정)이 돌지 않았다. 이 모듈이 그 의존을 없앤다. 담는 것은 전부 **표준 정식화**
이며 논문1의 기여가 아니다 — 회전스프링 균열 캔틸레버의 8×8 특성행렬, (1−d) 가중 Ritz,
등가 저강성 구간, 부가질량 습윤비.

이 파일이 고정하는 주장
  (T-X1) 네 함수 모두 논문1 구현과 **rel 1e-12**로 일치한다(논문1이 없으면 건너뛴다).
  (T-X2) ā = 0에서 전달행렬이 해석 캔틸레버 고유값(βL)을 되돌려준다.
  (T-X3) κ·d0는 균열이 깊어질수록 단조증가하고 ā = 0에서 0이다.
  (T-X4) 습윤비는 β = 0에서 1이고 β에 단조감소한다.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from impeller_fingerprint import eb_reference as eb
from impeller_fingerprint import geometry as geo

VANE = geo.VANE
GEO = dict(L=VANE.L, h=VANE.h, E=VANE.E, rho=VANE.rho)


class TestAgainstCompanionPackage:
    """T-X1 — 재구현이 논문1과 같은 값을 준다(폴더 규약: 재구현 + 교차검증)."""

    def test_kappa_matches(self):
        p1 = pytest.importorskip("impeller_pinn.crack_beam")
        for ab in (0.1, 0.3, 0.5, 0.6):
            assert eb.kappa(ab, VANE.h, VANE.L) == pytest.approx(
                p1.kappa(ab, VANE.h, VANE.L), rel=1e-12)

    def test_cracked_frequencies_match(self):
        p1 = pytest.importorskip("impeller_pinn.crack_beam")
        for ab in (0.0, 0.3, 0.5, 0.6):
            got = eb.cracked_cantilever_frequencies(a_bar=ab, xc_over_L=0.2,
                                                    n_modes=3, **GEO)
            exp = p1.cracked_cantilever_frequencies(VANE.L, VANE.h, VANE.E, VANE.rho,
                                                    ab, 0.2, n_modes=3)
            assert got == pytest.approx(exp, rel=1e-12)

    def test_knockdown_matches(self):
        p1 = pytest.importorskip("impeller_pinn.crack_beam")
        for ab in (0.1, 0.3, 0.5, 0.6):
            g = eb.crack_knockdown(ab, VANE.h, VANE.L, 0.2)
            e = p1.crack_knockdown(ab, VANE.h, VANE.L, 0.2)
            assert g.d0 == pytest.approx(e.d0, rel=1e-12)
            assert g.width_xt == pytest.approx(e.width_xt, rel=1e-12)
            xs = np.linspace(0.0, 1.0, 101)
            assert np.allclose(g(xs), e(xs), rtol=1e-12, atol=0.0)

    def test_ritz_matches(self):
        p1 = pytest.importorskip("impeller_pinn.classical_ritz")
        def dfun(xt):
            return 0.3 * np.exp(-((np.asarray(xt) - 0.2) ** 2) / (0.05 ** 2))
        got = eb.solve_ritz(n_modes=3, damage=dfun, n_trial=7, **GEO)
        exp = p1.solve_ritz(n_modes=3, damage=dfun, n_trial=7, **GEO)
        for g, e in zip(got, exp):
            assert g["f"] == pytest.approx(e["f"], rel=1e-12)
            assert g["Lambda"] == pytest.approx(e["Lambda"], rel=1e-12)
            assert g["nodes"] == e["nodes"]

    def test_wet_correction_matches(self):
        p1 = pytest.importorskip("impeller_pinn.fluid_loading")
        for bh in (1.0, 4.1, 10.0):
            b = eb.beta_beam(bh, VANE.rho)
            assert b == pytest.approx(p1.beta_beam(bh, VANE.rho), rel=1e-12)
            assert eb.wet_ratio(b) == pytest.approx(p1.wet_ratio(b), rel=1e-12)


class TestPhysicalAnchors:
    def test_healthy_transfer_matrix_recovers_the_analytic_cantilever(self):
        """T-X2 — ā = 0이면 (βL)₁₋₃의 해석해와 같아야 한다."""
        got = eb.cracked_cantilever_frequencies(a_bar=0.0, xc_over_L=0.2,
                                                n_modes=3, **GEO)
        exp = geo.Beam(**GEO, nu=VANE.nu).eb_frequencies(3)
        assert got == pytest.approx(exp, rel=1e-6)

    def test_crack_parameters_are_monotone_and_vanish_when_healthy(self):
        """T-X3 — κ·d0는 ā에 단조증가하고 건전 상태에서 0이다."""
        assert eb.kappa(0.0, VANE.h, VANE.L) == 0.0
        assert eb.crack_knockdown(0.0, VANE.h, VANE.L, 0.2).d0 == 0.0
        ks = [eb.kappa(ab, VANE.h, VANE.L) for ab in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
        ds = [eb.crack_knockdown(ab, VANE.h, VANE.L, 0.2).d0
              for ab in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
        assert all(b > a for a, b in zip(ks, ks[1:])), ks
        assert all(b > a for a, b in zip(ds, ds[1:])), ds
        assert all(0.0 < d < 1.0 for d in ds)

    def test_cracked_fundamental_falls_with_depth(self):
        f = [eb.cracked_cantilever_frequencies(a_bar=ab, xc_over_L=0.2, n_modes=1,
                                               **GEO)[0]
             for ab in (0.0, 0.3, 0.5, 0.6)]
        assert all(b < a for a, b in zip(f, f[1:])), f

    def test_wet_ratio_anchors(self):
        """T-X4 — β = 0에서 1, β에 단조감소, β = 3에서 정확히 ½."""
        assert eb.wet_ratio(0.0) == 1.0
        assert eb.wet_ratio(3.0) == pytest.approx(0.5, rel=1e-12)
        r = [eb.wet_ratio(b) for b in (0.5, 1.0, 2.0, 4.0)]
        assert all(b < a for a, b in zip(r, r[1:])), r

    def test_ritz_healthy_matches_the_analytic_cantilever(self):
        """건전 Ritz가 해석 캔틸레버로 수렴한다 — 단항 7항 기저의 3차 모드가 가장 느리다.

        관측 편차: 1·2차는 1e-6 이하, 3차가 **+0.029 %**(좁은 시행공간의 정상 거동).
        허용오차를 그 규모로 두고 값을 기록한다 — 더 좁히면 기저 차수를 올려야 하고,
        생산 산출은 n_trial = 7로 고정돼 있다. 부호까지 상계로 주장하지는 않는다:
        K·M을 사다리꼴로 적분하므로 그 오차가 Rayleigh 상계보다 클 수 있다(1차가 실제로
        해석값보다 1e-9 낮다).
        """
        r = eb.solve_ritz(n_modes=3, damage=None, n_trial=7, **GEO)
        exp = geo.Beam(**GEO, nu=VANE.nu).eb_frequencies(3)
        got = [x["f"] for x in r]
        assert got == pytest.approx(exp, rel=5e-4)
        assert 100 * (got[2] / exp[2] - 1) == pytest.approx(0.029, abs=0.003)
        assert [x["nodes"] for x in r] == [0, 1, 2]
