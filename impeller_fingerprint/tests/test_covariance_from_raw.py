"""Σ_y = A Σ_f Aᵀ — 유도 관측량의 공분산을 **원시 주파수에서** 전파한다 (검토 4차 #1).

옛 규약은 pair mean·m=0·splitting 셋 모두에 σ_η = 2c를 썼다. 그것은 **doublet pair mean에만**
맞는다. 이 검정은 세 관측량이 서로 다른 분산을 갖는다는 사실과, 세션 공통모드(온도)가
splitting에서만 상쇄된다는 사실을 해석식에 대해 고정한다.
"""
import numpy as np
import pytest

from impeller_fingerprint import noise as noi

C = 1e-3


def sd(kind, rho=0.0, n_avg=1):
    return float(np.sqrt(noi.sigma_y_from_raw([kind], C, rho, n_avg)[0, 0]))


class TestIndependentErrors:
    """ρ = 0에서 σ = {2c, 2√2c, 4c} — 검토가 유도한 값."""

    def test_pair_mean_is_two_c(self):
        assert sd("pair_mean") == pytest.approx(2 * C, rel=1e-12)

    def test_single_mode_is_root_two_larger(self):
        assert sd("single") == pytest.approx(2 * np.sqrt(2) * C, rel=1e-12)

    def test_splitting_is_twice_the_pair_mean(self):
        assert sd("splitting") == pytest.approx(4 * C, rel=1e-12)

    def test_ordering_is_strict(self):
        """이 순서가 뒤집히면 Fisher 가중이 뒤집힌다 — D-optimal 선택이 바뀐다."""
        assert sd("pair_mean") < sd("single") < sd("splitting")

    def test_averaging_scales_as_one_over_root_n(self):
        for k in noi.OBS_KINDS:
            assert sd(k, n_avg=9) == pytest.approx(sd(k) / 3.0, rel=1e-12)


class TestSessionCommonMode:
    """온도 같은 **세션 공통** 오차의 효과 — splitting만 면역이다.

    해석: Var = 4σ²(1+ρ) / 8σ² / 16σ²(1−ρ). 즉 pair mean은 **나빠지고**, m=0은 불변,
    splitting은 좋아진다. 전역 등상관으로 모델링하면 셋 다 같은 비율로 줄어드는데 그것은
    세션간 드리프트까지 상쇄시키는 모델의 인공물이다.
    """

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
    def test_matches_closed_form(self, rho):
        assert sd("pair_mean", rho) == pytest.approx(2 * C * np.sqrt(1 + rho), rel=1e-12)
        assert sd("single", rho) == pytest.approx(2 * np.sqrt(2) * C, rel=1e-12)
        assert sd("splitting", rho) == pytest.approx(4 * C * np.sqrt(1 - rho), rel=1e-12)

    def test_splitting_overtakes_pair_mean_at_high_correlation(self):
        """ρ가 크면 splitting이 pair mean보다 **조용해진다** — 조립체에서 splitting을
        주 관측량으로 쓰는 §4.3-vii의 결론과 같은 방향이다."""
        assert sd("splitting", 0.9) < sd("pair_mean", 0.9)
        assert sd("splitting", 0.0) > sd("pair_mean", 0.0)

    def test_rho_is_a_correlation(self):
        with pytest.raises(ValueError):
            noi.sigma_y_from_raw(["pair_mean"], C, rho=1.0)


class TestMatrixStructure:
    def test_observables_are_block_diagonal_across_modes(self):
        S = noi.sigma_y_from_raw(["pair_mean", "pair_mean"], C)
        assert S[0, 1] == pytest.approx(0.0, abs=1e-24), "다른 모드끼리 상관이 생기면 안 된다"

    def test_same_mode_mean_and_split_are_uncorrelated(self):
        """같은 쌍의 pair mean과 splitting은 직교한다 — (+)+(−) 대 (+)−(−)."""
        A, cols = noi.observable_matrix(["pair_mean", "splitting"])
        assert float(A[0] @ A[1]) == pytest.approx(0.0, abs=1e-12)

    def test_matrix_is_spd(self):
        S = noi.sigma_y_from_raw(["single", "pair_mean", "splitting"], C, rho=0.5)
        assert np.all(np.linalg.eigvalsh(S) > 0)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError):
            noi.observable_matrix(["bogus"])
