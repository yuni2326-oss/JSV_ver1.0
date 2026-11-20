"""N2 국부 포켓 회귀 — C6 등가(θ_d + 60°), 1차 깊이 선형성, raised-cosine 창."""
import math

import numpy as np
import pytest

from impeller_fingerprint.rev1 import n1_assembly as n1
from impeller_fingerprint.rev1 import n2_pocket as n2


def test_raised_cosine_window():
    th = np.linspace(-math.pi, math.pi, 200001)
    w = math.radians(30)
    win = n2.raised_cosine(th, w)
    assert win.max() == pytest.approx(1.0, abs=1e-8)
    assert np.all(win[np.abs(th) >= 0.5 * w] == 0.0)
    assert np.trapezoid(win, th) == pytest.approx(0.5 * w, rel=1e-6)


@pytest.fixture(scope="module")
def coarse():
    spec = n1.fit_spec(n_r=8, n_bay=5)
    am = n1.build(spec)
    km0 = n1.assemble(am)
    md = n1.solve(km0, 12)
    obs = n1.classify(am, md)
    return spec, am, km0, md, obs


def test_cyclic_equivalence(coarse):
    """C6 구조: 포켓을 60° 돌리면 모든 1차 관측량(쌍평균·분리·singlet)이 같다."""
    spec, am, km0, md, obs = coarse
    th = math.radians(17.0)
    a = n2.first_order(am, md, obs, km0, 0.5, 0.3, 3e-3, th, math.radians(30))
    b = n2.first_order(am, md, obs, km0, 0.5, 0.3, 3e-3, th + math.pi / 3, math.radians(30))
    scale = np.max(np.abs(a[0]))
    # 반올림 수준: 얇은 앞판(0.375 mm, 실측 정본)에서 행렬 중심차분의 상쇄오차가 상대 ~2e-6까지 큰다
    # (08-13 1.0 mm 판에서는 1e-7 안). 방위에 따른 실제 차이(N2: 수십 %)보다 다섯 자릿수 작다.
    assert np.max(np.abs(a[0] - b[0])) < 1e-5 * scale
    ok = np.isfinite(a[1])
    assert np.max(np.abs(a[1][ok] - b[1][ok])) < 1e-5 * scale


def test_first_order_linear_in_depth(coarse):
    spec, am, km0, md, obs = coarse
    a = n2.first_order(am, md, obs, km0, 0.4, 0.2, 3e-3, 0.3, math.radians(30))
    b = n2.first_order(am, md, obs, km0, 0.4, 0.4, 3e-3, 0.3, math.radians(30))
    assert np.allclose(2 * a[0], b[0], rtol=1e-9, atol=0)


def test_singlets_do_not_split(coarse):
    spec, am, km0, md, obs = coarse
    m, s = n2.first_order(am, md, obs, km0, 0.5, 0.3, 3e-3, 0.2, math.radians(30))
    for o, sv in zip(obs, s):
        assert (o.deg == 1) == (not np.isfinite(sv))
