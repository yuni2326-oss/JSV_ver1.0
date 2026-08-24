"""crack_shear 테스트 — A7: Timoshenko + Mode-I/II로 곡률-null 실명 예측 검정.

T8(설계서 §9): 세장극한에서 EB 전달행렬 해 복원.
추가: 유연도 적분식(Tada)이 논문1의 Dimarogonas 다항식과 일치하는지 — Mode-II 값의 신뢰 근거.
"""
import math

import numpy as np
import pytest

from impeller_fingerprint import crack_shear as cs
from impeller_fingerprint import geometry as geo

VANE = geo.VANE
BEAM = cs.TimoBeam(L=VANE.L, h=VANE.h, b=2 * VANE.h, E=VANE.E, rho=VANE.rho,
                   nu=VANE.nu)


class TestCompliance:
    @pytest.mark.parametrize("ab", [0.1, 0.3, 0.5, 0.6])
    def test_default_convention_is_tada(self, ab):
        """**의도된 변경(2026-08-10, 설계서 F42/F48)**: 기본 규약이 Tada 적분식이다.

        이전 기본값은 Dimarogonas 다항식이었다. 근거는 A11(2D 평면탄성 + 폭 0 슬릿)이
        폭 0 균열의 f₁ 강하에서 역산한 등가 회전유연도다 — Tada와 1–3 % 일치,
        Dimarogonas보다 27–30 % 큼. 즉 규약차는 양방향 불확실도가 아니라 **한쪽의 편향**이고,
        Dimarogonas는 c_θ를 약 23 % 저평가한다. 생산 경로는 그래서 Tada다.
        """
        got = cs.compliance(ab, BEAM.h, BEAM.b, BEAM.E, BEAM.nu)
        tada = cs.compliance(ab, BEAM.h, BEAM.b, BEAM.E, BEAM.nu, convention="tada")
        assert got["c_MM"] == pytest.approx(tada["c_MM"], rel=1e-14)
        assert got["c_QQ"] == pytest.approx(tada["c_QQ"], rel=1e-14)
        assert got["handbook_scale"] == pytest.approx(1.0)

    @pytest.mark.parametrize("ab", [0.1, 0.3, 0.5, 0.6])
    def test_dimarogonas_path_is_preserved_for_backward_compat(self, ab):
        """`convention="dimarogonas"`는 유지된다 — 정본 Table 1·논문1 코드와의 연속성."""
        pytest.importorskip("impeller_pinn.crack_beam")
        mine = cs.compliance(ab, BEAM.h, BEAM.b, BEAM.E, BEAM.nu,
                             convention="dimarogonas")["c_MM"]
        ref = cs.compliance_dimarogonas(ab, BEAM.h, BEAM.b, BEAM.E)
        assert mine == pytest.approx(ref, rel=1e-12)

    @pytest.mark.parametrize("ab,ratio_expected", [(0.1, 1.240), (0.3, 1.253),
                                                   (0.5, 1.268), (0.6, 1.273)])
    def test_handbook_convention_gap_is_recorded(self, ab, ratio_expected):
        """규약차를 **수치로** 고정한다: c_θ^Tada / c_θ^Dimarogonas = 1.24–1.27.

        A11 이전에는 이 값이 E2 예측구간의 양방향 불확실도(±27 %)였다. A11이 2D 탄성으로
        Tada 편임을 판정한 뒤로는 **일방 편향**이고, 예측구간 항목에서 빠진다(F42·F7).
        """
        c = cs.compliance(ab, BEAM.h, BEAM.b, BEAM.E, BEAM.nu, convention="tada")
        ref = cs.compliance_dimarogonas(ab, BEAM.h, BEAM.b, BEAM.E)
        assert c["c_MM"] / ref == pytest.approx(ratio_expected, abs=0.003), (ab,)

    def test_compliance_grows_with_depth(self):
        vals = [cs.compliance(ab, BEAM.h, BEAM.b, BEAM.E, BEAM.nu)
                for ab in (0.1, 0.3, 0.5, 0.6)]
        assert all(b["c_MM"] > a["c_MM"] for a, b in zip(vals, vals[1:]))
        assert all(b["c_QQ"] > a["c_QQ"] for a, b in zip(vals, vals[1:]))

    def test_shear_compliance_is_much_smaller_in_slender_beams(self):
        """세장보에서 활동유연도의 영향은 굽힘유연도보다 훨씬 작다(무차원 비교)."""
        c = cs.compliance(0.5, BEAM.h, BEAM.b, BEAM.E, BEAM.nu)
        # 무차원화: c_QQ / (c_MM · h²) — 1보다 훨씬 작아야 한다
        assert c["c_QQ"] / (c["c_MM"] * BEAM.h ** 2) < 0.5


class TestHealthyBeam:
    def test_healthy_matches_euler_bernoulli_in_slender_limit(self):
        f = cs.frequencies(BEAM, n_modes=3, n_elem=300)
        eb = VANE.eb_frequencies(3)
        assert f[0] / eb[0] == pytest.approx(1.0, rel=0.01)
        assert f[1] / eb[1] == pytest.approx(1.0, rel=0.02)

    def test_shear_lowers_higher_modes(self):
        """Timoshenko 보정은 고차 모드를 EB보다 낮춘다."""
        f = cs.frequencies(BEAM, n_modes=3, n_elem=300)
        eb = VANE.eb_frequencies(3)
        assert f[2] / eb[2] < f[0] / eb[0]

    def test_mesh_convergence(self):
        f1 = cs.frequencies(BEAM, n_modes=3, n_elem=150)
        f2 = cs.frequencies(BEAM, n_modes=3, n_elem=300)
        assert np.allclose(f1, f2, rtol=2e-3)


class TestT8EBRecovery:
    @pytest.mark.parametrize("ab", [0.3, 0.5])
    def test_mode_I_only_matches_transfer_matrix(self, ab):
        """Mode-I 단독 + 세장 → 논문1의 EB 전달행렬 해와 일치(T8).

        **규약을 맞춰야 성립한다**: 논문1의 전달행렬은 Dimarogonas κ를 내부에서 만들므로
        같은 규약을 명시해 비교한다(기본값은 2026-08-10부터 tada).
        """
        cb = pytest.importorskip("impeller_pinn.crack_beam")
        ref = cb.cracked_cantilever_frequencies(VANE.L, VANE.h, VANE.E, VANE.rho,
                                                ab, 0.2, n_modes=3)
        f0_ref = cb.cracked_cantilever_frequencies(VANE.L, VANE.h, VANE.E, VANE.rho,
                                                   0.0, 0.2, n_modes=3)
        rows = cs.signature(BEAM, [ab], xc_over_L=0.2, n_modes=3, n_elem=300,
                            shear_flex=False, convention="dimarogonas")
        for i in range(3):
            mine = rows[0][f"ratio_f{i+1}"]
            theirs = ref[i] / f0_ref[i]
            assert mine == pytest.approx(theirs, abs=0.02), (i, mine, theirs)


class TestExactEBFromCompliance:
    """`exact_eb_ratios` — 회전유연도를 **직접** 받는 논문3 소유 전달행렬(정본 Table 1 경로).

    논문1 `crack_beam`은 수정 금지이고 ā에서 Dimarogonas κ를 내부 생성하므로 다른 규약을
    넣을 수 없다. 이 구현이 규약 교체를 가능하게 하며, Dimarogonas c_θ를 넣으면 논문1
    함수와 일치해야 한다 — 그것이 이 arm의 자기검증이다.
    """

    def test_healthy_limit_recovers_cantilever_eigenvalues(self):
        b = cs.exact_eb_betas(0.0, n_modes=3)
        assert b == pytest.approx([1.8751041, 4.6940911, 7.8547574], abs=1e-6)

    @pytest.mark.parametrize("ab", [0.1, 0.3, 0.5, 0.6])
    def test_matches_paper1_transfer_matrix_in_dimarogonas_convention(self, ab):
        cb = pytest.importorskip("impeller_pinn.crack_beam")
        f0 = np.array(cb.cracked_cantilever_frequencies(VANE.L, VANE.h, VANE.E,
                                                        VANE.rho, 0.0, 0.2, n_modes=3))
        ref = np.array(cb.cracked_cantilever_frequencies(VANE.L, VANE.h, VANE.E,
                                                         VANE.rho, ab, 0.2,
                                                         n_modes=3)) / f0
        cdim = cs.compliance_dimarogonas(ab, BEAM.h, BEAM.b, BEAM.E)
        assert cs.exact_eb_ratios(BEAM, cdim) == pytest.approx(ref, abs=1e-9)

    @pytest.mark.parametrize("ab,f1_tada", [(0.1, 0.99630), (0.3, 0.96856),
                                            (0.5, 0.89904), (0.6, 0.83277)])
    def test_tada_convention_table1_values(self, ab, f1_tada):
        """정본 Table 1의 **Tada 열**을 고정한다(§4.1·§5가 인용하는 물리 예측의 스프링 짝).

        2026-08-13 갱신: 실측 판두께로 h/L = 0.04 → 1/30이 됐다. 무차원 스프링 유연도
        c_θEI/L ∝ h/L이므로 균열이 상대적으로 **덜** 유연해져 강하가 작아진다 —
        옛 값 0.99556/0.96262/0.88227/0.80835는 h = 1.2 mm의 것이다.
        """
        ctad = cs.compliance(ab, BEAM.h, BEAM.b, BEAM.E, BEAM.nu)["c_MM"]
        assert cs.exact_eb_ratios(BEAM, ctad)[0] == pytest.approx(f1_tada, abs=5e-5)

    def test_fundamental_gap_to_2d_is_almost_all_convention(self):
        """**F48**: f₁ 격차는 거의 전부 규약 인자다 — 힌지 이상화 인자는 ā=0.5에서 1.02.

        같은 관측량을 mode 2에서 보면 힌지 인자가 1.12–1.20으로 커진다(F44). 즉 점 힌지
        환원은 균열이 크게 움직이는 모드에서 정확하고 곡률-null 모드에서 부정확하다.
        """
        ab, ratio_2d = 0.5, 0.897206        # a11_crack2d.csv, plane stress, ref=3
        cdim = cs.compliance_dimarogonas(ab, BEAM.h, BEAM.b, BEAM.E)
        ctad = cs.compliance(ab, BEAM.h, BEAM.b, BEAM.E, BEAM.nu)["c_MM"]
        s_dim = 1 - cs.exact_eb_ratios(BEAM, cdim)[0]
        s_tad = 1 - cs.exact_eb_ratios(BEAM, ctad)[0]
        s_2d = 1 - ratio_2d
        assert s_tad / s_dim == pytest.approx(1.229, abs=0.01)      # 규약
        assert s_2d / s_tad == pytest.approx(1.018, abs=0.01)       # 힌지
        assert s_2d / s_dim == pytest.approx(1.251, abs=0.01)       # 합


class TestBlindnessPrediction:
    def test_mode2_nearly_blind_with_mode_I_only(self):
        """정본 §4.1: x_c/L=0.2가 mode 2의 곡률 null 근처 → 거의 안 움직인다."""
        rows = cs.signature(BEAM, [0.5], xc_over_L=0.2, n_elem=300, shear_flex=False)
        assert abs(rows[0]["ratio_f2"] - 1.0) < 0.01
        assert rows[0]["ratio_f1"] < 0.95        # 기본모드는 크게 떨어진다

    def test_shear_flexibility_effect_is_reported(self):
        """Mode-II를 켰을 때 mode 2 이동이 얼마나 커지는지 — 결론이 아니라 수치 산출."""
        a = cs.signature(BEAM, [0.5], xc_over_L=0.2, n_elem=300, shear_flex=False)[0]
        b = cs.signature(BEAM, [0.5], xc_over_L=0.2, n_elem=300, shear_flex=True)[0]
        assert b["c_QQ"] > 0.0
        assert abs(b["ratio_f2"] - 1.0) >= abs(a["ratio_f2"] - 1.0) - 1e-6

    def test_coupling_sweep_bounded_by_cauchy_schwarz(self):
        c = cs.compliance(0.5, BEAM.h, BEAM.b, BEAM.E, BEAM.nu)
        assert c["c_MQ_max"] == pytest.approx(math.sqrt(c["c_MM"] * c["c_QQ"]))
        rows = [cs.signature(BEAM, [0.5], xc_over_L=0.2, n_elem=200,
                             coupling=k)[0] for k in (0.0, 0.5, 0.9)]
        assert all(abs(r["c_MQ"]) <= c["c_MQ_max"] + 1e-18 for r in rows)
        # 결합이 커질수록 mode 2 이동이 단조 증가하는지(방향성 기록)
        shifts = [abs(r["ratio_f2"] - 1.0) for r in rows]
        assert shifts == sorted(shifts) or shifts == sorted(shifts, reverse=True)

    def test_deep_beam_sweep_runs(self):
        """h/L 스윕 — 정본이 인용한 깊은보 경향[13,14]의 자체 재현 경로."""
        out = {}
        for hl in (0.04, 0.1, 0.2):
            beam = cs.TimoBeam(L=VANE.L, h=hl * VANE.L, b=2 * hl * VANE.L,
                               E=VANE.E, rho=VANE.rho, nu=VANE.nu)
            out[hl] = cs.signature(beam, [0.5], xc_over_L=0.2, n_elem=200,
                                   shear_flex=True)[0]["ratio_f2"]
        assert all(np.isfinite(v) for v in out.values())
        # 깊은보일수록 mode 2가 더 움직인다(전단 결합 강화)
        assert abs(out[0.2] - 1.0) > abs(out[0.04] - 1.0)


class TestLiteratureFlexibility:
    """J(ā) 재구현 — 논문1 패키지 없이도 Table 1을 재현할 수 있어야 한다.

    `compliance_dimarogonas`가 논문1의 `flexibility_J`를 import하고 있어, 코드·데이터만
    배포한 트리에서 Table 1의 Dimarogonas 열이 재현되지 않았다. J(ā)는 논문1의 기여가
    아니라 **문헌 다항식**(Ostachowicz & Krawczuk 1991, 정본 [3])이므로 폴더 규약이
    허용하는 "재구현 + 교차검증"으로 자립시킨다.
    """

    def test_matches_paper1_when_available(self):
        """교차검증 — 두 구현이 같은 값을 준다(논문1이 없으면 건너뛴다)."""
        p1 = pytest.importorskip("impeller_pinn.crack_beam")
        for ab in (0.05, 0.1, 0.25, 0.3, 0.5, 0.6):
            assert cs.flexibility_J(ab) == pytest.approx(p1.flexibility_J(ab), rel=1e-12)

    def test_literature_shape(self):
        """J(0) = 0, (0, 0.6]에서 단조증가, 그리고 표에 실린 값과 맞는다."""
        assert cs.flexibility_J(0.0) == 0.0
        xs = [0.05 * k for k in range(1, 13)]
        vals = [cs.flexibility_J(x) for x in xs]
        assert all(b > a for a, b in zip(vals, vals[1:])), vals
        for ab, exp in ((0.1, 0.016), (0.3, 0.140), (0.5, 0.498), (0.6, 0.925)):
            assert cs.flexibility_J(ab) == pytest.approx(exp, abs=6e-4)

    def test_dimarogonas_compliance_needs_no_companion_package(self):
        """`compliance_dimarogonas`가 자립 구현을 쓴다 — c_θ = 5.346 (h/EI) J(ā)."""
        h, b, E = 0.001, 0.0041, 193e9
        I = b * h ** 3 / 12.0
        got = cs.compliance_dimarogonas(0.5, h, b, E)
        assert got == pytest.approx(5.346 * (h / (E * I)) * cs.flexibility_J(0.5), rel=1e-12)
