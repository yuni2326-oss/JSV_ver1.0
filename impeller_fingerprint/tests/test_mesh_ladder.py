"""형상동결 정제와 m = N/2 분리량의 메시 사다리 (설계서 F78–F80, A13x).

A13은 m = N/2 단일화 분리량을 40.4 % of f로 냈지만 메시 사다리를 돌리지 않아 정본이
라이선스한 것은 자릿수 논거까지였다(§12⑪). 사다리를 돌릴 때 **가장 쉬운 실패는 수렴
계산이 아니라 형상 변화**다 — 물리 두께 판정(`vane_arc_cells=None`)은 셀 방위폭 r·dθ와
비교하므로 n_theta를 올리면 베인 발자국 셀수가 바뀌고, 그러면 재는 것이 이산화가 아니라
기하다(F11이 3D 포켓 레일에서 빠진 함정과 같다).

  T-LAD-1  기본값(형상동결 없음)에서 발자국은 옛 경로와 **동일**하다(회귀)
  T-LAD-2  형상동결 정제는 고체영역을 **물리공간에서** 보존한다(독립 재판정으로 확인)
  T-LAD-3  형상 불변량(발자국 방위점유율·베인 셀수/정제배수)이 단계마다 같다
  T-LAD-4  형상보존 규약 위반(정수배 아님·반쪽 지정)은 예외로 막는다
  T-LAD-5  정제는 대칭이 금지하는 분리를 만들지 않는다(축대칭 웹 대조군)
  T-LAD-6  산출 사다리 CSV가 형상동결·판정기준·정본 인용값과 정합한다
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impeller_fingerprint import impeller_hex as ihx

#: 기본값은 **이 체크아웃의** 산출 디렉터리다 — 절대경로를 박으면 클론에서 동작하지
#: 않고 다른 워킹트리의 데이터를 검정한다(설계서 F153). `PAPER3_OUT`으로 덮어쓴다.
DATA = Path(os.environ.get(
    "PAPER3_OUT",
    Path(__file__).resolve().parents[2] / "docs" / "_generated")) / "data" / "paper3"
LADDER = DATA / "a13x_mesh_ladder.csv"
CANON = (Path(__file__).resolve().parents[2] / "docs" / "paper3-jsv"
         / "2026-07-31-paperB-jsv-v2.1.md")
#: 검정용 기준격자 — 물리 t_vane은 방위셀이 30°인 격자에서 발자국이 비므로 `arc_cells`로
#: 각을 명시한다(사다리 본체는 as-built 물리 판정을 18×108에서 동결한다).
BASE = dict(n_vane=6, vane_arc_cells=1, n_r=4, n_theta=12,
            n_z_shroud=1, n_z_channel=1)


def _frozen(nr: int, nth: int, **kw):
    return ihx.HexImpellerSpec(**{**BASE, **kw, "n_r": nr, "n_theta": nth,
                                  "footprint_n_r": BASE["n_r"],
                                  "footprint_n_theta": BASE["n_theta"]})


class TestDefaultPathUnchanged:
    """T-LAD-1 — 형상동결을 쓰지 않으면 발자국 판정이 옛 경로와 같아야 한다."""

    @pytest.mark.parametrize("kw", [dict(), dict(vane_arc_cells=1),
                                    dict(vane_mode="web")])
    def test_footprint_grid_equals_direct_call(self, kw):
        s = ihx.HexImpellerSpec(**{**BASE, **kw})
        r_nodes = np.linspace(s.a, s.b, s.n_r + 1)
        r_c = 0.5 * (r_nodes[:-1] + r_nodes[1:])
        dth = 2 * math.pi / s.n_theta
        th_c = np.arange(s.n_theta) * dth + 0.5 * dth
        RC, TC = np.meshgrid(r_c, th_c, indexing="ij")
        want = ihx.vane_footprint(s, RC.ravel(), TC.ravel()).reshape(s.n_r, s.n_theta)
        assert np.array_equal(ihx.footprint_grid(s), want)

    def test_no_freeze_means_unit_refine_factors(self):
        s = ihx.HexImpellerSpec(**BASE)
        assert s.refine_factors == (1, 1)
        assert s.footprint_spec is s

    def test_production_asbuilt_footprint_is_one_cell_per_station(self):
        """as-built 물리 판정은 18×108에서 반경단면마다 정확히 1셀이다(사다리의 기준형상)."""
        s = ihx.HexImpellerSpec(n_vane=6)
        fp = ihx.footprint_grid(s)
        assert fp.sum(axis=1).tolist() == [s.n_vane] * s.n_r
        assert float(fp.mean()) == pytest.approx(1.0 / s.cells_per_sector, rel=1e-12)


class TestShapeIsFrozenInPhysicalSpace:
    """T-LAD-2 — 정제격자의 고체영역을 **기준격자와 독립적으로** 다시 판정해 비교한다."""

    @pytest.mark.parametrize("kr,kt", [(2, 2), (3, 3), (1, 4), (3, 1)])
    def test_refined_cells_inherit_base_membership(self, kr, kt):
        base = ihx.HexImpellerSpec(**BASE)
        fine = _frozen(BASE["n_r"] * kr, BASE["n_theta"] * kt)
        fp_base, fp_fine = ihx.footprint_grid(base), ihx.footprint_grid(fine)
        assert fp_fine.shape == (base.n_r * kr, base.n_theta * kt)
        # 정제 셀중심의 (r, θ)가 어느 기준 셀에 드는지 **좌표로** 되짚는다.
        dr_b = (base.b - base.a) / base.n_r
        dth_b = 2 * math.pi / base.n_theta
        dr_f = (fine.b - fine.a) / fine.n_r
        dth_f = 2 * math.pi / fine.n_theta
        for i in range(fine.n_r):
            r_c = fine.a + (i + 0.5) * dr_f
            p = int((r_c - base.a) // dr_b)
            for j in range(fine.n_theta):
                q = int(((j + 0.5) * dth_f) // dth_b)
                assert fp_fine[i, j] == fp_base[p, q], (i, j, p, q)

    def test_solid_area_fraction_is_invariant(self):
        fracs = {}
        for kr, kt in ((1, 1), (2, 2), (3, 3), (2, 4)):
            s = (ihx.HexImpellerSpec(**BASE) if (kr, kt) == (1, 1)
                 else _frozen(BASE["n_r"] * kr, BASE["n_theta"] * kt))
            _, _, info = ihx.mesh(s)
            fracs[(kr, kt)] = info["vane_area_frac"]
            assert info["footprint_key"] == f"{BASE['n_r']}x{BASE['n_theta']}"
        assert len(set(np.round(list(fracs.values()), 12))) == 1, fracs

    def test_vane_cells_scale_exactly_with_refinement(self):
        """T-LAD-3 — 베인 셀수는 정제배수의 곱으로만 늘어야 한다(발자국이 안 바뀐다)."""
        _, _, i0 = ihx.mesh(ihx.HexImpellerSpec(**BASE))
        for kr, kt in ((2, 2), (3, 2)):
            _, _, i1 = ihx.mesh(_frozen(BASE["n_r"] * kr, BASE["n_theta"] * kt))
            assert i1["vane_cells_per_vane"] == pytest.approx(
                i0["vane_cells_per_vane"] * kr * kt, rel=1e-12)

    def test_polygonal_area_converges_monotonically(self):
        """정제가 바꾸는 유일한 기하는 원환의 **다각형 근사**이며 단조로 수렴한다.

        방위 절점이 원 위에 있으므로 이산 영역은 내접다각형이고, 정제하면 체적이
        단조 증가하며 증분이 줄어든다. 발자국(어느 셀이 고체인가)은 이미 동결됐으므로
        이것이 남는 유일한 기하 이산화 오차이고, 그래서 **수렴하는** 오차다.
        """
        vols = []
        for kt in (1, 2, 3, 4):
            s = (ihx.HexImpellerSpec(**BASE) if kt == 1
                 else _frozen(BASE["n_r"], BASE["n_theta"] * kt))
            coors, conn, _ = ihx.mesh(s)
            vols.append(ihx.assemble(s, coors, conn)[2])
        dv = np.diff(vols)
        assert (dv > 0).all(), vols                      # 내접 → 단조 증가
        assert (np.diff(dv) < 0).all(), vols             # 증분 감소 = 수렴
        assert dv[-1] / vols[-1] < 1e-2, vols


class TestShapePreservingContractIsEnforced:
    """T-LAD-4 — 규약 위반을 조용히 통과시키면 사다리가 기하를 재게 된다(F11′)."""

    @pytest.mark.parametrize("nr,nth", [(6, 24), (8, 18), (5, 12)])
    def test_non_integer_multiple_rejected(self, nr, nth):
        with pytest.raises(ValueError, match="정수배"):
            _frozen(nr, nth).check()

    def test_half_specified_reference_rejected(self):
        with pytest.raises(ValueError, match="함께"):
            ihx.HexImpellerSpec(**BASE, footprint_n_r=4).check()

    def test_reference_must_also_be_cyclic(self):
        """기준격자가 C_N을 보존하지 않으면 동결된 발자국 자체가 비대칭이다."""
        with pytest.raises(ValueError, match="footprint_n_theta"):
            ihx.HexImpellerSpec(n_vane=6, n_r=8, n_theta=24,
                                footprint_n_r=4, footprint_n_theta=8).check()

    def test_valid_ladder_levels_pass(self):
        for nr, nth, nzs, nzc in ((4, 12, 1, 1), (8, 24, 2, 2), (12, 36, 3, 3)):
            _frozen(nr, nth, n_z_shroud=nzs, n_z_channel=nzc).check()


class TestRefinementDoesNotManufactureSplitting:
    """T-LAD-5 — 축대칭 웹은 어느 단계에서도 m = N/2를 갈라놓지 않아야 한다."""

    @pytest.mark.parametrize("kr,kt", [(1, 1), (2, 2)])
    def test_web_split_stays_at_floor(self, kr, kt):
        kw = dict(BASE, vane_mode="web")
        kw.pop("vane_arc_cells")
        s = (ihx.HexImpellerSpec(**kw) if (kr, kt) == (1, 1) else
             ihx.HexImpellerSpec(**{**kw, "n_r": kw["n_r"] * kr,
                                    "n_theta": kw["n_theta"] * kt,
                                    "footprint_n_r": kw["n_r"],
                                    "footprint_n_theta": kw["n_theta"]}))
        coors, conn, minfo = ihx.mesh(s)
        res, info = ihx.solve_free_free(s, coors, conn, n_modes=14, mesh_info=minfo)
        summ = ihx.splitting_summary(s, res, info)
        assert abs(summ["split_hN2_rel"]) < 1e-6, summ["split_hN2_rel"]

    def test_discrete_vane_split_is_far_above_floor(self):
        s = ihx.HexImpellerSpec(**BASE)
        coors, conn, minfo = ihx.mesh(s)
        res, info = ihx.solve_free_free(s, coors, conn, n_modes=14, mesh_info=minfo)
        summ = ihx.splitting_summary(s, res, info)
        assert summ["split_hN2_rel"] > 1e3 * summ["floor_split_rel_max"]


class TestLadderPlanAndMerge:
    """비싼 단계를 잃지 않는가 — 최촘 격자가 45분이므로 덮어쓰기가 곧 재계산이다."""

    def test_default_levels_obey_the_shape_preserving_contract(self):
        from impeller_fingerprint import cli
        fr, ft = cli.A13X_FOOTPRINT
        assert len(cli.A13X_LEVELS) >= 3
        prev = None
        for nr, nth, nzs, nzc in cli.A13X_LEVELS:
            assert nr % fr == 0 and nth % ft == 0, (nr, nth)
            assert nth % 6 == 0                       # n_vane = 6
            if prev is not None:
                assert all(a > b for a, b in zip((nr, nth, nzs, nzc), prev))
            prev = (nr, nth, nzs, nzc)
        assert cli.build_parser().parse_args(["a13x"]).levels is None

    def test_same_key_is_replaced_and_other_configs_survive(self, tmp_path,
                                                            monkeypatch):
        from impeller_fingerprint import cli
        monkeypatch.setattr(cli, "DATA", tmp_path)
        base = dict(n_r=18, n_theta=108, n_z_shroud=2, n_z_channel=2)
        pd.DataFrame([{"config": "c6_asbuilt", **base, "n_dof": 1,
                       "split_pct_of_f": 40.0},
                      {"config": "c6_web", **base, "n_dof": 2,
                       "split_pct_of_f": 1e-9}]
                     ).to_csv(tmp_path / "a13x_mesh_ladder.csv", index=False)
        out = cli._a13x_merge([{"config": "c6_asbuilt", **base, "n_dof": 1,
                                "split_pct_of_f": 40.9}])
        assert len(out) == 2
        got = out.set_index("config").split_pct_of_f
        assert float(got["c6_asbuilt"]) == 40.9        # 같은 키 → 갈아치움
        assert float(got["c6_web"]) == 1e-9            # 다른 구성 → 보존

    def test_merge_without_existing_file(self, tmp_path, monkeypatch):
        from impeller_fingerprint import cli
        monkeypatch.setattr(cli, "DATA", tmp_path)
        out = cli._a13x_merge([{"config": "c6_asbuilt", "n_r": 18, "n_theta": 108,
                                "n_z_shroud": 2, "n_z_channel": 2, "n_dof": 1,
                                "split_pct_of_f": 40.4}])
        assert len(out) == 1


@pytest.fixture(scope="module")
def ladder():
    if not LADDER.exists():
        pytest.skip(f"산출물 없음: {LADDER} — `cli a13x`를 먼저 돌린다")
    return pd.read_csv(LADDER)


class TestLadderArtifact:
    """T-LAD-6 — 커밋된 사다리가 형상동결·판정기준·정본 인용값과 정합한가."""

    CFG = "c6_asbuilt"

    def _rows(self, ladder):
        g = ladder[ladder.config == self.CFG].sort_values("n_dof")
        if g.empty:
            pytest.skip(f"{self.CFG} 사다리가 없다")
        return g

    def test_at_least_three_levels_refined_in_every_direction(self, ladder):
        g = self._rows(ladder)
        assert len(g) >= 3, len(g)
        for col in ("n_r", "n_theta", "n_z_shroud", "n_z_channel", "n_dof"):
            assert g[col].is_monotonic_increasing and g[col].nunique() == len(g), col

    def test_shape_is_identical_across_levels(self, ladder):
        g = self._rows(ladder)
        assert g.vane_area_frac.round(12).nunique() == 1, g.vane_area_frac.tolist()
        assert g.footprint_key.nunique() == 1, g.footprint_key.tolist()
        # 발자국이 동결됐으면 베인 셀수는 정제배수의 곱으로만 커진다.
        k = (g.n_r / g.n_r.iloc[0]) * (g.n_theta / g.n_theta.iloc[0])
        assert np.allclose(g.vane_cells_per_vane / g.vane_cells_per_vane.iloc[0], k)

    def test_same_pair_is_tracked_at_every_level(self, ladder):
        """짝짓기는 형상으로 한다 — 겹침과 위상차가 단계마다 같은 쌍을 가리켜야 한다."""
        g = self._rows(ladder)
        assert (g.partner_overlap > 0.8).all(), g.partner_overlap.tolist()
        assert (g.partner_dpsi_deg.abs().between(20, 40)).all(), \
            g.partner_dpsi_deg.tolist()

    def test_splitting_is_converged_by_the_sec36_criterion(self, ladder):
        g = self._rows(ladder)
        worst = g.d_split_rel_pct.dropna().max()
        assert worst < 5.0, f"분리량 상대변화 {worst:.2f} % ≥ 5 % — 미수렴"

    def test_splitting_stays_orders_above_the_artifact_floor(self, ladder):
        g = self._rows(ladder)
        assert (g.split_over_floor_decades > 8).all(), \
            g.split_over_floor_decades.tolist()
        assert (g.split_pct_of_f > 1e3 * g.floor_pct_of_f_max).all()

    def test_canon_quotes_the_finest_level_and_the_verdict(self, ladder):
        """정본이 인용하는 값은 **가장 촘촘한 격자**의 값과 **판정 근거**여야 한다.

        분리량만 검사하면 옛 격자 값(40.4 %)이 문서에 남아 있어도 통과하므로, 사다리만이
        만들어내는 수치인 **단계간 상대변화 최대**도 함께 요구한다.
        """
        if not CANON.exists():
            pytest.skip("정본 md 없음")
        g = self._rows(ladder)
        fine = g.iloc[-1]
        worst = g.d_split_rel_pct.dropna().max()
        body = "\n".join(ln for ln in CANON.read_text(encoding="utf-8").split("\n")
                         if not ln.startswith("*[STATUS NOTE"))
        # 자릿수는 정본이 인용하는 정밀도 — 분리량은 사다리 폭(0.4 %p)에 맞춰 소수 1자리,
        # 단계간 상대변화는 1 %보다 작으므로 2자리다.
        assert f"{fine.split_pct_of_f:.1f} %" in body, (
            f"정본이 최촘 격자 값 {fine.split_pct_of_f:.1f} %를 인용하지 않는다")
        assert f"{worst:.2f} %" in body, (
            f"정본이 사다리 판정값(상대변화 최대 {worst:.2f} %)을 인용하지 않는다")
