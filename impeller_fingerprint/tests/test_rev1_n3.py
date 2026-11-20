"""N3(균열 지문 강건성) 회귀검정 — 재구현의 정본 교차검증과 시간적분의 극한 거동."""
import math

import numpy as np
import pytest
from scipy.linalg import eigh

from impeller_fingerprint import crack_shear as cs
from impeller_fingerprint.rev1 import n3_crack as n3

B = n3.BEAM


class TestTimoFEAgainstCanon:
    @pytest.mark.parametrize("a_bar", [0.3, 0.5, 0.6])
    @pytest.mark.parametrize("n_elem", [200, 300])
    @pytest.mark.parametrize("shear", [True, False])
    def test_single_crack_matches_crack_shear(self, a_bar, n_elem, shear):
        C = n3.crack_compliance(a_bar, "tada", shear=shear)
        mine = n3.crack_ratios([(0.2, C)], n_elem=n_elem)
        can = cs.signature(B, [a_bar], xc_over_L=0.2, n_modes=3, n_elem=n_elem,
                           shear_flex=shear, convention="tada")[0]
        for i in range(3):
            # 고유해 드라이버 차(부분집합 vs 전체)로 1e-8 규모 차가 난다 — 지시 기준 1e-6보다 엄격하게
            assert mine[i] == pytest.approx(can[f"ratio_f{i+1}"], rel=1e-7)

    def test_second_crack_vanishing_depth_recovers_single(self):
        C1 = n3.crack_compliance(0.5, "tada", True)
        C2 = n3.crack_compliance(0.01, "tada", True)
        one = n3.crack_ratios([(0.2, C1)], n_elem=1000)
        two = n3.crack_ratios([(0.2, C1), (0.5, C2)], n_elem=1000)
        assert np.allclose(one, two, rtol=1e-4)

    def test_stiff_root_spring_is_clamped(self):
        EI = B.E * B.I
        f_cl = n3.TimoFE(n_elem=300).frequencies(3)
        f_sp = n3.TimoFE(n_elem=300, k_root=1e9 * EI / B.L).frequencies(3)
        assert np.allclose(f_cl, f_sp, rtol=1e-6)

    def test_mode2_curvature_null_location(self):
        nul = n3.null_positions(n_elem=1000)
        assert 0.214 < nul["fe_mode2"][0] < 0.218
        assert nul["eb_mode2"][0] == pytest.approx(0.2166, abs=2e-4)
        assert nul["eb_mode3"][0] == pytest.approx(0.1323, abs=2e-4)


class TestSmoothComparator:
    def test_reproduces_canonical_matched_gaussian(self):
        # 정본 a11: 2D 폭 0 슬릿(ā=0.5, 평면응력)의 f₁ 강하 10.279361 %에 정합
        drops, dmax = n3.matched_smooth(10.279361, centers=(0.2,))
        assert dmax == pytest.approx(0.391724, abs=2e-6)
        assert drops[1] == pytest.approx(2.20397, abs=1e-4)
        assert 100 * drops[1] / drops[0] == pytest.approx(21.4408, abs=1e-3)

    def test_fe_comparator_close_to_ritz_comparator(self):
        v = n3.smooth_fe_validation()
        assert v["fe_drop_f2"] == pytest.approx(v["ritz_drop_f2"], rel=0.02)

    def test_contrast_sign_and_criteria(self):
        r_crack = np.array([0.9, 0.999, 0.97])            # mode 2가 거의 안 움직이는 균열
        drops_smooth = np.array([10.0, 2.2, 4.0])         # 매끄러운장
        c = n3.contrast_cols(r_crack, drops_smooth)
        assert c["D21_crack_pct"] == pytest.approx(1.0, rel=1e-9)
        assert c["D21_smooth_pct"] == pytest.approx(22.0, rel=1e-9)
        assert c["contrast_m2_c0.001"] == pytest.approx((0.022 - 0.001) / (math.sqrt(2) * 1e-3))
        assert c["C1_nullfp"] and c["C2_multimode_c0.001"]


class TestBreathingModel:
    def _model(self, a_bar=0.5, xc=0.2):
        c = cs.compliance(a_bar, B.h, B.b, B.E, B.nu, convention="tada")["c_MM"]
        return n3.MasslessHinge(c_theta=c, xc=xc, n_elem=240)

    def test_open_hinge_matches_exact_eb(self):
        df = n3.breathing_validation(n_elem=240)
        for k in (1, 2):
            assert np.allclose(df[f"fe_ratio_open_f{k}"], df[f"exact_ratio_f{k}"], rtol=1e-6)
        assert np.allclose(df["fe_ratio_open_f3"], df["exact_ratio_f3"], rtol=1e-5)

    def test_closed_is_healthy(self):
        f = self._model().modes(-1, 3)[0]
        assert np.allclose(f, n3._FAN, rtol=1e-4)          # 집중질량 O(h²)

    def test_force_continuous_at_switch(self):
        df = n3.breathing_validation(n_elem=240)
        assert (df["force_jump_rel_at_switch"] < 1e-9).all()

    def test_open_preload_small_amplitude_stays_open(self):
        c = n3.breathing_case(0.2, 0.5, n_cycles1=30, preload="open", amp_ratio=0.5,
                              ic_mode=1)
        assert c["n_switch"] == 0 and c["frac_time_open"] == 1.0
        assert c["f1_td_fft"] == pytest.approx(c["f1_open"], rel=1e-6)

    def test_closed_preload_small_amplitude_is_healthy(self):
        c = n3.breathing_case(0.2, 0.5, n_cycles1=30, preload="closed", amp_ratio=0.5,
                              ic_mode=1)
        assert c["n_switch"] == 0 and c["frac_time_open"] == 0.0
        assert c["f1_td_fft"] == pytest.approx(c["f1_healthy"], rel=1e-6)

    def test_no_preload_breathing_equals_harmonic_mean_mode1(self):
        c = n3.breathing_case(0.2, 0.5, n_cycles1=30, ic_mode=1)
        hm = 2 * c["f1_open"] * c["f1_healthy"] / (c["f1_open"] + c["f1_healthy"])
        assert c["f1_open"] < c["f1_td_fft"] < c["f1_healthy"]
        assert c["f1_td_fft"] == pytest.approx(hm, rel=2e-5)
        assert c["td_valid"] and c["energy_dev_max"] < 1e-5

    def test_grazing_mode2_near_null_conserves_energy(self):
        # 균열이 mode-2 null 근방 — 힌지 지시량이 미분해 성분에 지배되는 스침 궤적(과거 발산)
        c = n3.breathing_case(0.35, 0.5, n_cycles2=150, ic_mode=2)
        assert c["td_valid"] and c["energy_dev_max"] < 1e-5
        assert c["f2_open"] < c["f2_td_fft"] < c["f2_healthy"]


class TestDepthInversion:
    @pytest.mark.parametrize("a_bar", [0.3, 0.5])
    def test_roundtrip_same_convention(self, a_bar):
        r = n3.tm_ratios(a_bar, "tada")
        for meth in ("f1", "lsq"):
            a_hat, chi = n3.invert_depth(r, "tada", meth)
            assert a_hat == pytest.approx(a_bar, abs=1e-5)
            assert chi < 1e-3

    def test_dimarogonas_reads_tada_data_deeper(self):
        # Dimarogonas는 c_θ를 저평가하므로 같은 Δf₁을 설명하려면 더 깊은 균열이 필요하다
        r = n3.tm_ratios(0.5, "tada")
        a_hat, _ = n3.invert_depth(r, "dimarogonas", "f1")
        assert a_hat > 0.5
