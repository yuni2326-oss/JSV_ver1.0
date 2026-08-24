"""F54 — Figure 4 하단 패널의 극점 제거와 그림/맵 분리 (설계서 §11.16).

배경: 질량항 맵에서 η̄^ex가 부호전환 반경 ξ*에서 0을 지나므로 순수 상대오차
e_pert = |Δη̄|/|η̄^ex|에 **극점**이 생긴다. 그 극점의 '최대값'은 격자가 η̄=0에 얼마나
가까이 떨어지느냐로 정해지는 격자 인공물이다(같은 맵에서 21×17이면 34, 41×31이면 498).
분모를 |η̄^ex| + σ_η로 바꾸면 극점이 사라지고 두 극한이 모두 물리적으로 옳다.

또한 그림이 `cli.cmd_a2` 안에 있어서 그림만 고칠 수 없던 결합을 끊었다 —
`figures.fig_a2_epert`는 저장된 npz만 소비한다. 이 검정이 그 성질을 고정한다.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from impeller_fingerprint import figures as F

#: 기본값은 **이 체크아웃의** 산출 디렉터리다 — 절대경로를 박으면 클론에서 동작하지
#: 않고 다른 워킹트리의 데이터를 검정한다(설계서 F153). `PAPER3_OUT`으로 덮어쓴다.
DATA = Path(os.environ.get(
    "PAPER3_OUT",
    Path(__file__).resolve().parents[2] / "docs" / "_generated")) / "data" / "paper3"
PRODUCTION = [DATA / "a2_epert_map_mass.npz", DATA / "a2_epert_map.npz"]


class TestObservableRelativeError:
    def test_floor_convention_matches_sec34(self):
        """σ_η = 2σ_f/f — Δf/f = ½η̄이므로 η̄ 단위 노이즈는 2배다(정본 §3.4)."""
        assert F.FLOOR == pytest.approx(2.0 * F.C_REP)
        assert F.C_REP == 1e-3          # §3.4가 인용하는 대표 노이즈 수준

    def test_no_pole_where_signal_vanishes(self):
        """η̄^ex → 0(rel → ∞)에서 e_obs는 유계이고 절대오차/floor로 수렴한다."""
        ab = np.array([[[1e-3]]])                      # |Δη̄| = 1e-3
        for eta in (1e-3, 1e-6, 1e-12):                # |η̄^ex|를 0으로 보낸다
            rel = ab / eta
            e, back = F.observable_relative_error(ab, rel)
            assert np.isfinite(e).all()
            assert back[0, 0, 0] == pytest.approx(eta, rel=1e-9)
        e, _ = F.observable_relative_error(ab, ab / 1e-15)
        assert e[0, 0, 0] == pytest.approx(1e-3 / F.FLOOR, rel=1e-6)

    def test_recovers_relative_error_when_signal_dominates(self):
        """|η̄^ex| ≫ σ_η에서는 순수 상대오차와 1 % 안에서 같다."""
        eta = 100.0 * F.FLOOR
        ab = np.array([[[0.5 * eta]]])
        rel = ab / eta
        e, _ = F.observable_relative_error(ab, rel)
        assert e[0, 0, 0] == pytest.approx(rel[0, 0, 0], rel=0.01)

    def test_bounded_by_both_panels(self):
        """e_obs ≤ 상대오차, e_obs ≤ 절대오차/floor — 분모가 두 규약보다 크므로."""
        rng = np.random.default_rng(20260811)
        eta = 10 ** rng.uniform(-6, -1, size=(4, 9, 7))
        ab = eta * 10 ** rng.uniform(-3, 1, size=eta.shape)
        rel = ab / eta
        e, back = F.observable_relative_error(ab, rel)
        assert np.all(e <= rel + 1e-15)
        assert np.all(e <= ab / F.FLOOR + 1e-15)
        assert back == pytest.approx(eta, rel=1e-12)

    def test_zero_rel_cells_do_not_divide(self):
        ab = np.array([[[0.0, 1e-4]]])
        rel = np.array([[[0.0, 1e-2]]])
        e, back = F.observable_relative_error(ab, rel)
        assert np.isfinite(e).all() and back[0, 0, 0] == 0.0


class TestFigureIsNpzConsumer:
    def test_draws_from_npz_without_recomputation(self, tmp_path, monkeypatch):
        """맵을 다시 계산하지 않고 npz만으로 그려지는가.

        `validity.e_pert_map`을 폭발하도록 바꿔 놓고 그림을 그린다 — 그림 경로가 맵을
        건드리면 실패한다. 이것이 '그림을 고치려면 A2 전체를 재실행해야 한다'는 결합이
        다시 생기지 않게 하는 회귀검정이다.
        """
        from impeller_fingerprint import validity as val

        def boom(*a, **k):                              # pragma: no cover
            raise AssertionError("그림이 맵을 재계산했다")
        monkeypatch.setattr(val, "e_pert_map", boom)

        npz = tmp_path / "a2_epert_map_mass.npz"
        xi = np.linspace(0.05, 0.95, 6)
        s = np.geomspace(0.001, 0.30, 5)
        rng = np.random.default_rng(7)
        eta = 10 ** rng.uniform(-6, -2, size=(4, len(xi), len(s)))
        ab = eta * 10 ** rng.uniform(-2, 0.5, size=eta.shape)
        np.savez(npz, xi=xi, s=s, rel=ab / eta, abs=ab,
                 modes=np.array([(0, 0), (1, 0), (2, 0), (3, 0)]))
        out = F.fig_a2_epert(npz, tmp_path / "fig.png", coupling="exact")
        assert out.exists() and out.stat().st_size > 0

    @pytest.mark.parametrize("npz", PRODUCTION, ids=lambda p: p.name)
    def test_production_maps_are_pole_free_after_rescale(self, npz):
        """커밋된 산출물에서 새 지표가 유계이고 극점이 사라졌는지."""
        if not npz.exists():
            pytest.skip(f"산출물 없음: {npz}")
        d = np.load(npz)
        e, eta = F.observable_relative_error(d["abs"], d["rel"])
        assert np.isfinite(e).all()
        assert np.all(e <= d["rel"] + 1e-12)
        assert np.all(e <= d["abs"] / F.FLOOR + 1e-12)
        if "mass" in npz.name:
            # 질량항 맵에는 부호전환(영교차)이 실제로 있고, 구 규약이면 상대오차가
            # 1(=100 %)을 크게 넘는 셀이 생긴다 — 새 규약에서는 그 셀들이 사라진다.
            assert d["rel"].max() > 10.0
            assert (eta < F.FLOOR).any()
            assert e.max() < d["rel"].max()


class TestChi2Thresholds:
    """Δχ² 등고선은 **χ² 분위에서 만든다** — 숫자를 박으면 신뢰수준이 어긋난다.

    2026-08-24 검토 지적: 예전 Fig. 6은 3.84(1-파라미터 95 %)와 11.8을 함께 그렸는데
    11.8은 2-파라미터 **3σ**(99.73 %)이고 2-파라미터 95 %는 5.99다 — 한 그림에 두 신뢰
    수준이 섞여 있었고 캡션은 11.8을 정의하지 않았다. 이제 두 등고선을 같은 95 %로 두고,
    값을 분위함수에서 계산한다.
    """

    def test_levels_are_chi2_quantiles_not_magic_numbers(self):
        from scipy.stats import chi2

        from impeller_fingerprint import figures as fg
        assert fg.CHI2_1P_95 == pytest.approx(chi2.ppf(0.95, 1), rel=1e-12)
        assert fg.CHI2_2P_95 == pytest.approx(chi2.ppf(0.95, 2), rel=1e-12)
        assert fg.CHI2_1P_95 == pytest.approx(3.841, abs=5e-4)
        assert fg.CHI2_2P_95 == pytest.approx(5.991, abs=5e-4)

    def test_the_two_parameter_3sigma_value_is_not_used_as_95pct(self):
        """11.8을 95 %로 쓰지 않는다 — 그 값은 2-파라미터 3σ다."""
        from scipy.stats import chi2

        from impeller_fingerprint import figures as fg
        import inspect
        assert chi2.cdf(11.8, 2) == pytest.approx(0.9973, abs=5e-4)
        # **그리는 코드**만 본다(모듈 주석은 11.8이 3σ라는 사실을 설명하므로 언급한다).
        src = inspect.getsource(fg.fig_a4_landscape)
        assert "11.8" not in src, "등고선 코드에 11.8이 남아 있다"
        assert "CHI2_1P_95" in src and "CHI2_2P_95" in src
        assert fg.CHI2_2P_95 < 6.0
