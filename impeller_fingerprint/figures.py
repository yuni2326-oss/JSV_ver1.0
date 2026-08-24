"""논문 그림 생성 — `python -m impeller_fingerprint.cli figs`.

라벨은 JSV 제출용이라 영문. 데이터는 `docs/_generated/data/paper3/`의 CSV/NPZ에서 읽는다.
질량항 도입(F12)·부호전환 반경(F14) 이후 커널 그림은 γ^K와 γ^M을 함께 보인다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import chi2 as _chi2
import pandas as pd

from . import forward as fwd
from . import geometry as geo
from . import kernels as ker

PLATE = geo.DISK
MODES = [(0, 0), (1, 0), (2, 0), (3, 0)]
W = 0.003


#: Δχ² 등고선 문턱 — **숫자를 박지 않고 χ² 분위에서 만든다.** 2026-08-24 검토 지적:
#: 예전 Fig. 6은 3.84와 11.8을 함께 그렸는데 11.8은 2-파라미터 **3σ**(99.73 %)이고
#: 2-파라미터 95 %는 5.99다 — 한 그림에 두 신뢰수준이 섞여 있었고 캡션은 11.8을
#: 정의하지 않았다. 두 등고선을 같은 95 %로 통일한다:
#:   CHI2_1P_95 = ξ_d 하나에 대한 프로파일 문턱(심각도 소거)
#:   CHI2_2P_95 = (ξ_d, S̄_D) 결합 영역 문턱
CHI2_1P_95 = float(_chi2.ppf(0.95, 1))      # 3.841
CHI2_2P_95 = float(_chi2.ppf(0.95, 2))      # 5.991


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def null_loci(pool, s_bar=0.03, n=2000, coupling="exact"):
    """모드별 부호전환 반경 ξ* (F14). 전환이 없으면 NaN.

    `coupling`은 생산 규약(정확결합 d_M = 1−(1−d_K)^{1/3})을 기본값으로 쓴다 —
    기본값 1/3(1차근사)을 쓰면 m=3의 반전이 ξ*=0.997로 밀려 보인다(설계서 F20 ③).
    """
    # 상한을 0.995로 끊으면 m=3의 교차점을 놓친다 — 페이블 검토 지적.
    xs = np.linspace(0.02, 0.9999, n)
    E = np.array([fwd.eta_bar_linear_mass(pool, x, s_bar, W, PLATE, coupling=coupling)
                  for x in xs])
    out = []
    for j in range(E.shape[1]):
        sg = np.sign(E[:, j])
        idx = np.nonzero(np.diff(sg) != 0)[0]
        out.append(float(xs[idx[0]]) if idx.size else np.nan)
    return xs, E, np.array(out)


def fig_kernels(data: Path, figs: Path):
    """Fig 4 — 민감도 커널 γ^K·γ^M과 부호전환 반경."""
    plt = _plt()
    pool = [ker.mode_kernel(PLATE, m=m, n=n, n_grid=2001) for m, n in MODES]
    xi = (pool[0].r - PLATE.a) / PLATE.extent
    xs, E, loci = null_loci(pool)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for k in pool:
        ax[0].plot(xi, k.gamma * PLATE.extent, label=f"m={k.m}")
        ax[1].plot(xi, k.gamma_mass * PLATE.extent, label=f"m={k.m}")
    ax[0].set_title("(a)")
    ax[1].set_title("(b)")
    for a in ax[:2]:
        a.set_xlabel(r"$\xi_d$"); a.set_ylabel(r"normalized kernel"); a.legend(fontsize=8)
    for j, (m, _) in enumerate(MODES):
        ax[2].plot(xs, E[:, j] * 1e3, label=f"m={m}")
        if np.isfinite(loci[j]):
            ax[2].axvline(loci[j], ls=":", lw=0.8, color=f"C{j}")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel(r"$\xi_d$"); ax[2].set_ylabel(r"$\bar\eta_m \times 10^3$")
    ax[2].set_title(r"(c) $\bar S_D = 3\%$")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); save_canon(fig, figs / "fig4_kernels_and_sign.png", dpi=200)
    pd.DataFrame({"m": [m for m, _ in MODES], "xi_star": loci,
                  "r_star_mm": PLATE.a * 1e3 + loci * PLATE.extent * 1e3}
                 ).to_csv(data / "a3_sign_reversal_loci.csv", index=False)
    print("[fig] fig4_kernels_and_sign.png + a3_sign_reversal_loci.csv")


def fig_recovery(data: Path, figs: Path):
    """Fig 3 — 생산 MC 회복통계(B1): 중앙값·90/95 % 분위수 + CRLB + 경계접촉."""
    # 질량항 산출물이 있으면 그것을 그린다 — 본문(F8′)이 인용하는 것과 같은 모델이어야 한다.
    f = data / "b1_mc_summary_mass.csv"
    model = "mass-inclusive"
    if not f.exists():
        f, model = data / "b1_mc_summary.csv", "stiffness-only"
    if not f.exists():
        print("[fig] b1_mc_summary*.csv 없음 — 건너뜀"); return
    d = pd.read_csv(f)
    plt = _plt()
    sig = sorted(d.sigma_rel.unique())[2]          # 0.1 %
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharex=True)
    for ax, s in zip(axes, sorted(d.s_bar.unique())):
        g = d[(d.s_bar == s) & (d.sigma_rel == sig)].sort_values("xi_d")
        ax.fill_between(g.xi_d, 0, g.abs_err_xi_mm_p95, alpha=.20, label="95 % quantile")
        ax.fill_between(g.xi_d, 0, g.abs_err_xi_mm_p90, alpha=.30, label="90 %")
        ax.plot(g.xi_d, g.abs_err_xi_mm_median, "o-", ms=3, label="median")
        ax.plot(g.xi_d, 1.96 * g.crlb_xi_mm, "k--", lw=1, label=r"1.96$\times$CRLB")
        ax2 = ax.twinx()
        ax2.plot(g.xi_d, 100 * g.boundary_hit_prob, color="crimson", lw=.9, alpha=.7)
        ax2.set_ylim(0, 100)
        if ax is axes[-1]:
            ax2.set_ylabel("boundary-hit [%]", color="crimson")
        ax.set_yscale("log"); ax.set_xlabel(r"$\xi_d$")
        ax.set_title(rf"$\bar S_D={100*s:g}\%$")
    axes[0].set_ylabel("location error [mm]"); axes[0].legend(fontsize=7)
    # suptitle 제거(2026-08-15): 맵 종류·σ_f/f·실현수는 **캡션이 말한다**. 그림에 캡션을
    # 다시 인쇄하면 조판에서 중복되고, 값이 바뀔 때 두 곳을 고쳐야 한다.
    print(f"[fig] recovery: {model} map, sigma_f/f={100*sig:g}%, "
          f"{int(d.n_real.iloc[0])} realizations/cell")
    fig.tight_layout(); save_canon(fig, figs / "fig3_recovery.png", dpi=200)
    print("[fig] fig3_recovery.png")


def fig_a4_landscape(data: Path, figs: Path):
    """Figure 6 (A4) — (ξ_d, S̄_D) 목적함수면 χ²: 평탄/다중골 구조 (설계서 §A4, 정본 §4.3(iii)).

    `fig_recovery`와 같은 규약으로 **질량항 산출물을 우선** 읽는다 — 본문이 인용하는 것과
    같은 맵이어야 한다(설계서 F8′·M7의 교훈: 모델을 고쳤으면 결과를 전부 재생성한다).

    그리는 양은 χ² 자체가 아니라 **Δχ² = χ² − min χ²**다. 이유: 격자가 `[::4]` 부표본이라
    진실점이 격자 노드에 정확히 놓이지 않고(ξ 간격 0.008), 그래서 격자 최소가 0이 아니다
    (0.006–0.47). Δχ²로 그리면 95 % 문턱선(1-파라미터 3.84 · 2-파라미터 5.99)이 골의 폭을
    직접 읽게 해준다 — 두 선의 신뢰수준을 같게 두어 혼동을 없앴다(2026-08-24).
    프로파일 신뢰구간은 S̄를 소거한 ξ만의 양이므로 세로선으로 표시한다.
    """
    f = data / "a4_grids_mass.npz"
    prof_f = data / "a4_profiles_mass.csv"
    model = "mass-inclusive"
    if not f.exists():
        f, prof_f, model = (data / "a4_grids.npz", data / "a4_profiles.csv",
                            "stiffness-only")
    if not (f.exists() and prof_f.exists()):
        print("[fig] a4_grids*/a4_profiles* 없음 — 건너뜀"); return
    d = np.load(f)
    prof = pd.read_csv(prof_f)
    xi_g, s_g = d["xi"], d["s"]
    xi_cases = (0.2, 0.5, 0.8)
    s_cases = sorted(prof.s_bar_true.unique())          # 0.01, 0.05
    coupling = (str(prof["mass_coupling"].iloc[0])
                if "mass_coupling" in prof.columns else "n/a")

    plt = _plt()
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(len(s_cases), len(xi_cases),
                             figsize=(4.4 * len(xi_cases), 3.5 * len(s_cases)),
                             sharex=True, sharey="row", squeeze=False)
    pc = None
    for ri, s_t in enumerate(s_cases):
        for ci, xi_t in enumerate(xi_cases):
            ax = axes[ri][ci]
            key = f"chi2_xi{xi_t}_s{s_t}"
            if key not in d:
                ax.text(.5, .5, f"{key}\nmissing", ha="center", va="center",
                        transform=ax.transAxes); continue
            Z = d[key] - d[key].min()                   # Δχ²
            # 색상범위를 신뢰영역 규모(Δχ² ~ 1e−2…1e2)에 맞춘다. 심각도가 크면 골이 매우
            # 좁아 Δχ²가 곧 1e4를 넘는데(정보량이 크므로 당연), 상한을 1e4로 두면 아래 행
            # 전체가 균일하게 어두워져 골 모양이 보이지 않았다. 1e2 위는 "신뢰영역 밖"이라
            # 구분할 필요가 없으므로 포화시키고, 그 아래에 색 해상도를 전부 준다.
            pc = ax.pcolormesh(xi_g, s_g * 100, np.maximum(Z, 1e-3).T, shading="auto",
                               cmap="viridis_r", norm=LogNorm(vmin=1e-2, vmax=1e2))
            cs_ = ax.contour(xi_g, s_g * 100, Z.T,
                             levels=[CHI2_1P_95, CHI2_2P_95],
                             colors=["crimson", "0.35"], linewidths=[1.6, 1.0])
            ax.clabel(cs_, fmt={CHI2_1P_95: f"{CHI2_1P_95:.2f}",
                                CHI2_2P_95: f"{CHI2_2P_95:.2f}"}, fontsize=6)
            row = prof[(prof.xi_true == xi_t) & (prof.s_bar_true == s_t)]
            # 진실점은 **속 빈** 별로 — 채우면 S̄=5 % 행에서 골 자체를 덮어버린다.
            ax.plot([xi_t], [100 * s_t], marker="*", ms=15, mfc="none",
                    mec="k", mew=1.4, ls="none", label="truth", zorder=5)
            note = rf"$\xi_d={xi_t}$, $\bar S_D={100*s_t:g}\%$"
            if len(row):
                r0 = row.iloc[0]
                for x, lab in ((r0.prof_lo, "profile 95 % CI"), (r0.prof_hi, None)):
                    ax.axvline(x, color="k", ls="--", lw=1.1, label=lab)
                # 국소최소 개수는 "평가된 격자 안에서 관측된" 수일 뿐(정본 §3.3 language rule).
                note += (f"\nprofile CI $\\pm${r0.prof_halfwidth_mm:.2f} mm"
                         f"  (CRLB {r0.crlb_xi_mm:.2f} mm)"
                         f"\ngrid local minima observed: "
                         f"{int(r0.local_minima_in_grid)}")
            ax.text(.03, .97, note, fontsize=8, va="top", ha="left",
                    transform=ax.transAxes,
                    bbox=dict(fc="w", ec="0.6", alpha=.88, pad=2.2))
            ax.set_yscale("log")
            # S̄ 축은 **행마다** 진실값 주변으로 좁힌다. 심각도가 5배면 정보량이 25배라
            # 신뢰영역이 그만큼 좁아지는데, 전 격자(0.2–30 %)를 그대로 보이면 아래 행이
            # 사실상 빈 화면이 된다. ξ 축은 전 구간을 유지해야 다중골 구조가 보인다.
            ax.set_ylim(max(s_g.min(), s_t / 3.5) * 100,
                        min(s_g.max(), s_t * 3.5) * 100)
            if ri == len(s_cases) - 1:
                ax.set_xlabel(r"$\xi_d$")
            if ci == 0:
                ax.set_ylabel(r"$\bar S_D$ [%]")
    axes[0][0].legend(fontsize=7, loc="lower right")
    if pc is not None:
        fig.colorbar(pc, ax=[a for r in axes for a in r],
                     label=r"$\chi^2 - \min\chi^2$")
    # suptitle 제거(2026-08-24): 맵 종류·결합비·σ_f/f·물리한계는 **캡션이 말한다**.
    # 그림에 캡션을 다시 찍지 않는다(같은 규약을 fig_recovery에 2026-08-15 적용).
    save_canon(fig, figs / "fig_a4_landscape.png", dpi=200, bbox_inches="tight")
    print("[fig] fig_a4_landscape.png")


def fig_kerf(data: Path, figs: Path):
    """B2 — 3D 베인 커프: 모드선택성과 노치→균열 등가깊이."""
    f = data / "b2_vane3d.csv"
    if not f.exists():
        print("[fig] b2_vane3d.csv 없음 — 건너뜀"); return
    d = pd.read_csv(f).sort_values(["kerf_width_actual_mm", "depth_frac_actual"])
    # b2는 스팬방향 폭을 여러 개 낼 수 있다(as-built 4.1 mm + 옛 규약 2h). 그림은 **as-built**만
    # 쓴다 — 폭을 섞으면 같은 (깊이, 커프폭)에 두 점이 생겨 선이 지그재그가 된다.
    w_vane = None
    if "vane_width_mm" in d.columns:
        w_vane = float(d.vane_width_mm.max())
        d = d[d.vane_width_mm == w_vane]
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    # `b2_vane3d.csv`의 ratio_fk는 **형상으로 고른 면외(flap) 계열의 k번째**다
    # (`cmd_b2`: f = res.freqs[flap]). 옛 판은 `ratio_f3`를 "mode 2 (flap)"으로 그렸는데
    # 그것은 flap 3번째이고 3–8 % 움직인다 — 곡률-null 실명(0.12 %)을 말해야 하는 패널에서
    # 정본 §4.1이 인용하는 열(ratio_f2 = 0.121 %)이 아닌 열을 보여주고 있었다.
    for wv, g in d.groupby("kerf_width_actual_mm"):
        ax[0].plot(g.depth_frac_actual, 100 * abs(1 - g.ratio_f1), "o-",
                   label=f"mode 1 (flap), kerf {wv:.2f} mm")
        ax[0].plot(g.depth_frac_actual, 100 * abs(1 - g.ratio_f2), "s--",
                   label=f"mode 2 (flap), kerf {wv:.2f} mm")
        ax[0].plot(g.depth_frac_actual, 100 * abs(1 - g.ratio_f3), "^:",
                   label=f"mode 3 (flap), kerf {wv:.2f} mm")
        ax[1].plot(g.depth_frac_actual, g.a_bar_equivalent, "o-",
                   label=f"kerf {wv:.2f} mm")
    ax[1].plot([0, .7], [0, .7], "k:", lw=.8, label="ideal crack (1:1)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("notch depth fraction")
    ax[0].set_ylabel(r"$|1 - f_k/f_k^{\,h}|$  [%]"); ax[0].legend(fontsize=6.4)
    ax[0].set_title("(a)" + (rf"  $b$ = {w_vane:.1f} mm" if w_vane else ""))
    ax[1].set_xlabel("physical notch depth"); ax[1].set_ylabel(r"equivalent crack $\bar a$")
    ax[1].legend(fontsize=7); ax[1].set_title("(b)")
    fig.tight_layout(); fig.savefig(figs / "figb2_kerf.png", dpi=200)
    print("[fig] figb2_kerf.png")


def fig_splitting(data: Path, figs: Path):
    """B3/A6 — 포켓 각폭·위치에 따른 분리와 pair mean(3D vs 섭동이론)."""
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    a6 = data / "a6_degenerate.csv"
    if a6.exists():
        d = pd.read_csv(a6)
        for m, g in d[(d.depth_frac == 0.25)].groupby("m"):
            gg = g.groupby("dtheta_deg").delta_eta.mean()
            ax[0].plot(gg.index, gg.values, "o-", label=f"m={m}")
        ax[0].set_xlabel(r"pocket angular width $\Delta\theta$ [deg]")
        ax[0].set_ylabel(r"$\Delta\bar\eta_m$"); ax[0].legend(fontsize=8)
        ax[0].set_title("(a)")
    b3 = data / "b3_disk3d.csv"
    if b3.exists():
        d = pd.read_csv(b3)
        for m in (1, 2, 3):
            kx, ky = f"eta_bar_theory_m{m}", f"eta_bar_3d_m{m}"
            if kx in d and ky in d:
                ax[1].plot(d[kx], d[ky], "o", label=f"m={m}")
        lim = np.nanmax(np.abs([ax[1].get_xlim(), ax[1].get_ylim()]))
        ax[1].plot([-lim, lim], [-lim, lim], "k:", lw=.8)
        ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5)
        ax[1].set_xlabel(r"$\bar\eta$ perturbation theory")
        ax[1].set_ylabel(r"$\bar\eta$ 3D solid")
        ax[1].legend(fontsize=8)
        ax[1].set_title("(b)")
    fig.tight_layout(); fig.savefig(figs / "figb3_splitting.png", dpi=200)
    print("[fig] figb3_splitting.png")


C_REP = 1e-3            # 대표 반복도 σ_f/f (SIGMA_RELS 4수준 중 §3.4가 인용하는 것)
FLOOR = 2.0 * C_REP     # η̄ 단위 측정 floor σ_η = 2σ_f/f (Δf/f = ½η̄)


def observable_relative_error(abs_map, rel_map, floor=FLOOR):
    """관측 가능한 상대오차 e_obs = |Δη̄| / (|η̄^ex| + σ_η) — **극점이 없는** 상대 지표.

    **왜 분모에 floor를 더하는가(설계서 F20 ⑤ 잔여 → F54).** 질량항을 넣으면 η̄^ex가 모드별
    부호전환 반경 ξ*에서 0을 지나므로 순수 상대오차 |Δη̄|/|η̄^ex|는 그 반경에서 **극점**을
    갖는다(이 격자에서 34, 41×31 격자에서 498). 앞선 세션은 그 극점을 (a) |η̄^ex| < σ_η 셀
    마스킹 + (b) 로그 상한 100 %로 가렸는데, 가린 것은 값이지 극점 자체가 아니어서 그림이
    "무엇이 큰가"를 말하지 못했다.

    분모를 |η̄^ex| + σ_η로 바꾸면 **극점이 사라지고 두 극한이 모두 물리적으로 옳다**:
      |η̄^ex| ≫ σ_η  →  e_obs → 순수 상대오차(신호가 충분히 크면 상대오차가 맞는 질문)
      |η̄^ex| ≪ σ_η  →  e_obs → |Δη̄|/σ_η = 위 패널의 절대오차(신호가 floor 아래면
                                 "오차가 관측 가능한가"만 물을 수 있다)
    즉 이 양은 *반복도 σ_η를 가진 계측기로 볼 때의 상대오차*이고, 그래서 마스킹도 상한도
    필요하지 않다. 판정선은 두 패널 모두 값 1이다.

    선택 근거(대안 대비): 상대 패널을 아예 빼는 방법도 검토했으나, 절대 패널만 두면
    e_pert가 심각도에 **1차**(절대오차는 2차)라는 §3.4의 스케일링 서술을 그림에서 확인할
    수 없게 된다. floor를 더한 분모는 그 스케일링을 유지하면서 극점만 제거한다.

    `abs_map`/`rel_map`은 `a2_epert_map*.npz`의 `abs`/`rel`이며 |η̄^ex| = abs/rel로
    복원되므로 **맵을 다시 계산하지 않는다**(그래서 이 함수는 npz만 있으면 돌아간다).
    """
    abs_map = np.asarray(abs_map, dtype=float)
    rel_map = np.asarray(rel_map, dtype=float)
    eta_ex = np.abs(np.divide(abs_map, rel_map, out=np.zeros_like(abs_map),
                              where=rel_map > 0))
    return abs_map / (eta_ex + floor), eta_ex


def fig_a2_epert(npz: Path, out_png: Path, coupling: str | None = None,
                 s_max_phys: float | None = None):
    """Figure 5 — 선형화 오차 대 측정 floor. **저장된 npz만으로** 그린다.

    이전에는 이 그림이 `cli.cmd_a2` 안에 있어서 그림을 고치려면 A2 전체를 재실행해야 했고,
    그것이 상대맵 재스케일을 미뤄 온 이유였다(설계서 §11.15 '미해소로 남긴 것'). 그림을
    npz 소비자로 분리하면 맵·등고선 산출물은 손대지 않고 그림만 바꿀 수 있다.

    상단 = |η̄^ex − η̄^lin|/σ_η (판정용, §3.4의 값 1 등고선),
    하단 = 관측 가능한 상대오차 e_obs(참고용, 극점 없음 — `observable_relative_error`).
    """
    plt = _plt()
    from matplotlib.colors import LogNorm
    d = np.load(npz)
    xi, s, Mabs, Mrel = d["xi"], d["s"], d["abs"], d["rel"]
    modes = [tuple(m) for m in d["modes"]]
    # 비물리 심각도 마스킹: 가우시안 최대 δD/D = S̄(b−a)/(w√π) > 1이면 국소 D<0이 되어
    # 정확재해가 발산한다(설계서 M8). npz는 전 격자를 담고 **그림에서만** 자른다.
    if s_max_phys is None:
        s_max_phys = W * np.sqrt(np.pi) / PLATE.extent
    keep = s < s_max_phys
    s_p, Mabs_p, Mrel_p = s[keep], Mabs[:, :, keep], Mrel[:, :, keep]
    E_obs, _ = observable_relative_error(Mabs_p, Mrel_p)
    # **모드별 floor**(F112): m = 0은 축퇴쌍이 없어 σ_η = 2√2c다. 전 모드에 2c를 쓰면
    # m = 0 패널이 실제보다 나쁘게 보인다 — 검토 5차가 지적한 "공통 floor"가 이것이다.
    floors = np.array([FLOOR * (np.sqrt(2.0) if m == 0 else 1.0) for m, _ in modes])
    A = Mabs_p / floors[:, None, None]

    fig, axes = plt.subplots(2, len(modes), figsize=(4.1 * len(modes), 6.4),
                             sharex=True, sharey=True)
    a_norm = LogNorm(vmin=max(1e-4, A[A > 0].min()), vmax=A.max())
    o_norm = LogNorm(vmin=max(1e-4, E_obs[E_obs > 0].min()), vmax=E_obs.max())
    for k, (m, n) in enumerate(modes):
        ax = axes[0, k]
        pc = ax.pcolormesh(xi, s_p * 100, A[k].T, shading="auto", cmap="viridis",
                           norm=a_norm)
        cs_ = ax.contour(xi, s_p * 100, A[k].T, levels=[1.0], colors="w", linewidths=1.6)
        ax.clabel(cs_, fmt={1.0: "floor"}, fontsize=7)
        ax.set_title(f"m={m}, n={n}")
        if k == len(modes) - 1:
            fig.colorbar(pc, ax=axes[0, :].tolist(),
                         label=r"$|\bar\eta^{\rm ex}-\bar\eta^{\rm lin}|\,/\,\sigma_\eta$"
                               "\n" r"($\sigma_\eta = 2\sigma_f/f = 0.2\,\%$)")
    for k in range(len(modes)):
        ax = axes[1, k]
        pc = ax.pcolormesh(xi, s_p * 100, E_obs[k].T, shading="auto", cmap="magma",
                           norm=o_norm)
        cs_ = ax.contour(xi, s_p * 100, E_obs[k].T, levels=[1.0], colors="w",
                         linewidths=1.2, linestyles="--")
        ax.clabel(cs_, fmt={1.0: "1"}, fontsize=7)
        ax.set_xlabel(r"$\xi_d$")
        if k == len(modes) - 1:
            fig.colorbar(pc, ax=axes[1, :].tolist(),
                         label=r"observable relative error"
                               "\n"
                               r"$|\bar\eta^{\rm ex}-\bar\eta^{\rm lin}|/(|\bar\eta^{\rm ex}|"
                               r"+\sigma_\eta)$")
    for ax in axes.ravel():
        ax.set_yscale("log")
    axes[0, 0].set_ylabel(r"$\bar S_D$ [%]")
    axes[1, 0].set_ylabel(r"$\bar S_D$ [%]")
    # suptitle 제거(2026-08-24): 맵 종류·결합비·σ_f/f·물리한계는 **캡션이 말한다**.
    # 그림에 캡션을 다시 찍지 않는다(같은 규약을 fig_recovery에 2026-08-15 적용).
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_canon(fig, out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")
    return out_png


# ------------------------------------------------------------------ Figure 1
_HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def boundary_faces(conn: np.ndarray) -> np.ndarray:
    """육면체 부분집합의 **경계면**(한 번만 나타나는 사각면). 컷어웨이마다 다시 계산해야
    잘라낸 안쪽 면(베인)이 실제로 보인다 — 전체 메시의 경계면을 재사용하면 내부가 비어 보인다."""
    from collections import defaultdict
    count: dict = defaultdict(int)
    keep: dict = {}
    for e in conn:
        for fd in _HEX_FACES:
            q = tuple(int(e[i]) for i in fd)
            k = tuple(sorted(q))
            count[k] += 1
            keep[k] = q
    return np.array([keep[k] for k, c in count.items() if c == 1], dtype=np.int64)


def cutaway_subset(coors: np.ndarray, conn: np.ndarray, sector_deg: tuple,
                   z_cut: float) -> np.ndarray:
    """전면 슈라우드(z > z_cut)의 방위 구간 [θ0, θ1]을 제거해 베인을 노출시킨다."""
    cen = coors[conn].mean(axis=1)
    th = np.arctan2(cen[:, 1], cen[:, 0])
    c0, c1 = np.deg2rad(sector_deg[0]), np.deg2rad(sector_deg[1])
    in_cut = ((np.angle(np.exp(1j * (th - c0))) >= 0)
              & (np.angle(np.exp(1j * (th - c1))) <= 0))
    return conn[~(in_cut & (cen[:, 2] > z_cut))]


def _render_panel(ax, coors, conn, u, z_band=None, elev=46.0, azim=-58.0,
                  scale_frac=0.022, n_bands=9, z_exag=None):
    """한 패널: 변형 형상 위에 정규화 총변형 등고 밴드. 페인터 알고리즘으로 깊이정렬.

    `z_band = (z_lo, z_hi)`를 주면 **유로층(=베인) 면**을 굵은 검은 테두리로 구분해 그린다 —
    컷어웨이 패널의 목적이 "베인이 어디 있는가"를 보이는 것이므로, 변형 색만으로는 리브와
    슈라우드가 구분되지 않는다.
    """
    plt = _plt()
    from matplotlib.colors import BoundaryNorm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    bf = boundary_faces(conn)
    mag = np.linalg.norm(u, axis=1)
    mag = mag / mag.max()
    dia = 2.0 * np.hypot(coors[:, 0], coors[:, 1]).max()
    Xd = (coors + (scale_frac * dia) * u / np.abs(u).max()) * 1e3
    fc = mag[bf].mean(axis=1)
    vdir = np.array([np.cos(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)),
                     np.sin(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)),
                     np.sin(np.deg2rad(elev))])
    order = np.argsort(Xd[bf].mean(axis=1) @ vdir)
    bands = np.linspace(0.0, 1.0, n_bands + 1)
    cmap = plt.get_cmap("jet", n_bands)
    norm = BoundaryNorm(bands, n_bands)
    # 미변형 z로 베인 판정(변형 후 z는 과장 때문에 층을 넘나든다)
    is_vane = np.zeros(len(bf), dtype=bool)
    if z_band is not None:
        zc = coors[bf][:, :, 2].mean(axis=1)
        is_vane = (zc > z_band[0] + 1e-9) & (zc < z_band[1] - 1e-9)
    # **하나의** 컬렉션에 면별 테두리를 준다. 컬렉션을 둘로 나누면 matplotlib이 깊이가 아니라
    # 아티스트 순서로 겹쳐 그려서 앞 슈라우드에 덮인 베인까지 뚫고 나온다(개발 중 확인).
    vs = is_vane[order]
    ec = np.zeros((len(order), 4))
    ec[:, 3] = np.where(vs, 0.95, 0.28)
    pc = Poly3DCollection([Xd[x] for x in bf[order]],
                          linewidths=np.where(vs, 0.55, 0.08), edgecolors=ec)
    pc.set_facecolor(cmap(norm(fc[order])))
    ax.add_collection3d(pc)
    lim = 0.56 * dia * 1e3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    if z_exag is None:
        # 원본 스크립트의 "얇은 원판 위의 완만한 파" 조판 — 모드형 패널용
        ax.set_zlim(-0.52 * lim, 0.72 * lim)
        ax.set_box_aspect((1, 1, 0.60))
    else:
        # 축방향 확대 — 단면(면판 2장 + 베인)을 읽을 수 있게 하는 기하 패널용
        zl, zh = Xd[:, 2].min(), Xd[:, 2].max()
        pad = 0.35 * (zh - zl)
        ax.set_zlim(zl - pad, zh + pad)
        ax.set_box_aspect((1, 1, z_exag * (zh - zl + 2 * pad) / (2 * lim)))
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("persp", focal_length=0.30)
    ax.set_axis_off()
    return cmap, norm, bands


def _overlay_camber(ax, coors, u, n_vane, wrap_deg, a, b, scale_frac=0.022,
                     **kw):
    """전면 슈라우드 표면에 **베인 캠버선**을 겹쳐 그린다 — m=N/2 절점선이 베인 집합에
    잠긴다는 것을 그림에서 읽게 하는 유일한 방법. 변형 좌표를 쓰므로 표면에 붙어 보인다."""
    dia = 2.0 * np.hypot(coors[:, 0], coors[:, 1]).max()
    Xd = (coors + (scale_frac * dia) * u / np.abs(u).max()) * 1e3
    top = np.nonzero(coors[:, 2] >= coors[:, 2].max() - 1e-9)[0]
    rr = np.hypot(coors[top, 0], coors[top, 1])
    th = np.arctan2(coors[top, 1], coors[top, 0])
    rings = np.unique(np.round(rr, 9))
    for k in range(n_vane):
        pts = []
        for r in rings:
            sel = top[np.abs(rr - r) < 1e-9]
            if not len(sel):
                continue
            tk = (2 * np.pi * k / n_vane
                  + np.deg2rad(wrap_deg) * (r - a) / (b - a))
            d = np.angle(np.exp(1j * (th[np.abs(rr - r) < 1e-9] - tk)))
            pts.append(Xd[sel[int(np.argmin(np.abs(d)))]])
        p = np.array(pts)
        ax.plot(p[:, 0], p[:, 1], p[:, 2] + 2.0, color="k", lw=1.3, ls="--",
                zorder=50)


def fig1_impeller_modes(npz: Path, out_png: Path, panel_dir: Path | None = None):
    """**Figure 1** — 임펠러 기하와 세 모드형. `cli fig1`이 만든 npz만 소비한다.

    (a) 닫힌 형상 + 첫 탄성 축퇴쌍   (b) 전면 슈라우드 컷어웨이(베인 노출), 같은 모드
    (c) m = N/2 단일 모드 — **순환대칭이 강제하는 비축퇴 조화**(설계서 A13/F72)

    Fig 4와 같은 규약으로 **저장된 npz만** 읽는다(설계서 F54: 그림이 계산 안에 박혀 있으면
    그림만 고칠 수 없다).
    """
    plt = _plt()
    import matplotlib
    d = np.load(npz, allow_pickle=False)
    coors, conn, f = d["coors"], d["conn"], d["freqs"]
    shapes = d["shapes"]                       # (n_modes, n_dof)
    idx = d["panel_modes"]                     # (3,) 모드 색인 — 형상으로 고른 것
    m_dom, h_hat, degen = d["m_dom"], d["h_hat"], d["degeneracy"]
    n_vane = int(d["n_vane"])
    z_cut = float(d["z_cut"])
    sector = tuple(float(x) for x in d["cut_sector_deg"])

    t_sheet, channel = float(d["t_sheet"]), float(d["channel"])
    z_band = (t_sheet, t_sheet + channel)
    wide = (sector[0] - 45.0, sector[1] + 5.0)          # (b)는 더 넓게 열어 베인 2–3매 노출
    subsets = [conn, cutaway_subset(coors, conn, wide, z_cut), conn]
    # 패널별 시점·과장. (b)는 **기하 패널**이므로 변형을 줄이고 축방향을 확대해 단면을 읽게
    # 하고, (a)(c)는 **모드형 패널**이므로 원본의 얇은 원판 조판을 유지한다.
    views = [dict(z_band=None, elev=46.0, azim=-58.0, scale_frac=0.022),
             dict(z_band=z_band, elev=20.0, azim=-62.0, scale_frac=0.006, z_exag=3.4),
             dict(z_band=None, elev=46.0, azim=-58.0, scale_frac=0.022)]
    camber = dict(n_vane=n_vane, wrap_deg=float(d["wrap_deg"]), a=float(d["a"]),
                  b=float(d["b"]))
    tags = ["(a)", "(b)", "(c)"]
    #: 패널 머리는 **문자만** 둔다 — 무엇을 보여주는 패널인지는 캡션이 말한다(2026-08-24).
    heads = ["", "", ""]

    fig = plt.figure(figsize=(13.2, 5.1), dpi=200)
    cmap = norm = bands = None
    for j, (k, sub) in enumerate(zip(idx, subsets)):
        ax = fig.add_axes([0.040 + 0.315 * j, -0.11, 0.30, 1.15], projection="3d")
        u = shapes[int(k)].reshape(-1, 3)
        cmap, norm, bands = _render_panel(ax, coors, sub, u, **views[j])
        if j == 2:                          # 베인 캠버선을 겹쳐 잠김을 보인다
            _overlay_camber(ax, coors, u, scale_frac=views[j]["scale_frac"],
                            **camber)
        # 인포블록은 **패널을 구별하는 것만** 담는다. (a)와 (b)는 같은 모드이므로 모드
        # 표시를 두 번 인쇄하지 않고 (b)에는 기하 정보를 준다. 메시 성격("illustration
        # mesh")은 세 번 반복하던 것을 캡션으로 내렸다 — 그림에 캡션을 다시 찍지 않는다.
        # 인포블록도 **파라미터 식별자만** 남긴다: 축퇴 성격·"same mode as (a)"·판두께 같은
        # 서술은 캡션과 §3.1이 말하므로 그림에 다시 찍지 않는다(2026-08-24).
        m_here, h_here = int(m_dom[int(k)]), int(round(float(h_hat[int(k)])))
        if j == 1:
            note = ""
        else:
            note = (f"$m$ = {m_here}, $h$ = {h_here}"
                    f" · $f$ = {f[int(k)]:.0f} Hz")
        ax.text2D(0.00, 0.995, f"{tags[j]} {heads[j]}".strip(),
                  transform=ax.transAxes, fontsize=10.5, fontweight="bold",
                  va="top", ha="left")
        if note:
            ax.text2D(0.00, 0.930, note, transform=ax.transAxes, fontsize=8.2,
                      va="top", ha="left", linespacing=1.35,
                      bbox=dict(boxstyle="round,pad=0.30", fc="#f7f8fa",
                                ec="#9aa4ae", lw=0.5))
    cax = fig.add_axes([0.965, 0.20, 0.012, 0.58])
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                          boundaries=bands, ticks=bands)
    cb.ax.set_yticklabels([f"{b:.2f}" for b in bands], fontsize=6.5)
    cb.ax.set_title("total\ndeformation\n(normalized)", fontsize=6.8, pad=7)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_canon(fig, out_png, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")

    if panel_dir is not None:                    # 패널별 개별 파일(조판 대안용)
        panel_dir.mkdir(parents=True, exist_ok=True)
        for j, (k, sub) in enumerate(zip(idx, subsets)):
            fg = plt.figure(figsize=(5.2, 4.4), dpi=200)
            ax = fg.add_axes([0.02, -0.10, 0.96, 1.16], projection="3d")
            _render_panel(ax, coors, sub, shapes[int(k)].reshape(-1, 3), **views[j])
            p = panel_dir / f"fig1{'abc'[j]}_impeller_modes.png"
            fg.savefig(p, facecolor="white", bbox_inches="tight")
            plt.close(fg)
            print(f"[saved] {p}")
    return out_png


# ------------------------------------------------------------------ Figure 2 (§4.1)
#: Fig 2 (a)의 **네 arm** — `a11_arm_comparison.csv`의 (arm, convention, 커프폭 mm) 키와
#: 라벨·색. 3D는 as-built 커프(0.25 mm)·베인폭 4.1 mm 열만 쓴다 — 1 mm 커프는 §4.1이
#: 폭 효과의 괄호로만 인용하므로 지문 비교선에 섞으면 같은 ā에 두 점이 생긴다(`fig_kerf`의
#: 폭 혼합 주의와 같은 규약).
FIG2_ARMS: tuple = (
    ("a_exact_spring_EB", "dimarogonas", 0.0,
     r"Rotational spring, Dimarogonas $c_\theta$", "C0"),
    ("a_exact_spring_EB", "tada", 0.0,
     r"Rotational spring, Tada $c_\theta$", "C1"),
    ("d_2d_stress_slit", "fem", 0.0,
     "2D plane stress, zero-width slit", "C2"),
    ("c_3d_notch_w4.1", "fem", 0.25,
     "3D solid, as-built notch (0.25 mm kerf)", "C3"),
)

#: Fig 2 (b)의 판별비 진술이 인용하는 arm — 정본 §4.1의 "0.96 / 0.96 / 0.91 % 대 21 %".
#: 세 번째 항목이 (a)에 없는 arm(Timoshenko + Mode-II 활동유연도)인 것은 의도적이다 —
#: 본문이 "the spring model **with sliding flexibility**"로 인용하는 것이 이 arm이고,
#: F49가 정정한 3인자 분해(규약 ×1.22 · Mode-II ×1.14 · 점힌지 ×1.02)의 중간 팔이다.
FIG2_PATTERN: tuple = (
    ("a_exact_spring_EB", "tada", 0.0, "Spring, Tada (Mode-I only)", "C1", "s"),
    ("b_timoshenko_modeI+II", "tada", 0.0, "Spring + Mode-II sliding", "C4", "v"),
    ("d_2d_stress_slit", "fem", 0.0, "2D zero-width slit", "C2", "o"),
    ("c_3d_notch_w4.1", "fem", 0.25, "3D as-built notch", "C3", "^"),
)

#: 정본 §4.1의 판별비 진술이 서 있는 균열깊이(“Matched on the elastic ā = 0.5 crack”).
FIG2_ANCHOR: float = 0.5

#: 모드 표시 규약 — (열 접미, 라벨, 선종, 마커).
FIG2_MODES: tuple = (("f1", "Mode 1", "-", "o"), ("f2", "Mode 2", "--", "s"),
                     ("f3", "Mode 3", ":", "^"))


def _fig2_arm_rows(d: pd.DataFrame, arm: str, conv: str, width_mm: float):
    """(arm, convention, 커프폭) 한 팔의 행 — ā 오름차순. 없으면 예외(조용한 빈 선 금지)."""
    g = d[(d.arm == arm) & (d.convention == conv)
          & (np.isclose(d.width_mm.to_numpy(float), width_mm))]
    if g.empty:
        raise KeyError(f"a11_arm_comparison.csv에 {arm}/{conv}/w={width_mm} 행이 없다")
    return g.sort_values("a_bar")


def fig2_series(data: Path) -> dict:
    """Fig 2가 그리는 **모든 수치를 CSV에서** 뽑는다 — 그림·캡션·본문의 단일 출처.

    이 함수를 그림에서 분리해 둔 이유는 `fig_a2_epert`를 npz 소비자로 분리한 것과 같다
    (F54): 회귀검정이 **그림이 실제로 쓰는 값**을 정본 캡션·본문의 인용값과 대조할 수
    있어야 하고, 그러려면 값이 그리기 코드 안에 갇혀 있으면 안 된다. 하드코딩된 수치는
    이 파일 어디에도 없다 — 판별비·정합 가우시안까지 전부 CSV에서 읽는다.

    출처:
      `a11_arm_comparison.csv`      네 arm(+Mode-II 팔)의 f₁/f₂/f₃ 이동 [%]  (A11 (D))
      `a11_table1_conventions.csv`  Δf₁을 맞춘 매끄러운 가우시안 경쟁모델과 판별비 (A11 (E))

    판별비 D ≡ |Δf₂/f₂| / |Δf₁/f₁| [%] — 곡률-null 실명의 **무차원 지표**. 정본 §4.1이
    0.96 %(2D 폭 0) / 0.96 %(Mode-II 포함 스프링) / 0.91 %(3D) 대 21 %(매끄러운 장)로
    인용하는 그 양이다.
    """
    fa, ft = data / "a11_arm_comparison.csv", data / "a11_table1_conventions.csv"
    missing = [p.name for p in (fa, ft) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Fig 2 데이터 없음: {missing} — `cli a11`을 먼저 돌린다")
    d, t = pd.read_csv(fa), pd.read_csv(ft)

    def discrim(g):
        return 100.0 * g.shift_f2_pct.to_numpy(float) / g.shift_f1_pct.to_numpy(float)

    arms = []
    for arm, conv, w, label, color in FIG2_ARMS:
        g = _fig2_arm_rows(d, arm, conv, w)
        arms.append({"key": (arm, conv, w), "label": label, "color": color,
                     "a_bar": g.a_bar.to_numpy(float),
                     "shift": np.stack([g[f"shift_{m}_pct"].to_numpy(float)
                                        for m, *_ in FIG2_MODES]),
                     "discrim_pct": discrim(g)})

    pattern = []
    for arm, conv, w, label, color, marker in FIG2_PATTERN:
        g = _fig2_arm_rows(d, arm, conv, w)
        s = g[np.isclose(g.a_bar.to_numpy(float), FIG2_ANCHOR)]
        if s.empty:
            raise KeyError(f"{arm}: ā={FIG2_ANCHOR} 행이 없다")
        sh = np.array([float(s[f"shift_{m}_pct"].iloc[0]) for m, *_ in FIG2_MODES])
        pattern.append({"key": (arm, conv, w), "label": label, "color": color,
                        "marker": marker, "shift": sh,
                        "discrim_pct": 100.0 * sh[1] / sh[0]})

    ta = t[np.isclose(t.a_bar.to_numpy(float), FIG2_ANCHOR)]
    if ta.empty:
        raise KeyError(f"a11_table1_conventions.csv에 ā={FIG2_ANCHOR} 행이 없다")
    ta = ta.iloc[0]
    gauss = {"label": r"Smooth Gaussian field, matched on $\Delta f_1$",
             "color": "crimson", "marker": "D",
             "shift": np.array([float(ta[f"gauss_shift_{m}_pct"])
                                for m, *_ in FIG2_MODES]),
             "discrim_pct": float(ta.gauss_ratio_m2_over_m1_pct),
             "d_max": float(ta.gauss_dmax_matched_2d),
             "a_bar": t.a_bar.to_numpy(float),
             "discrim_vs_depth": t.gauss_ratio_m2_over_m1_pct.to_numpy(float)}
    crack2d = {"discrim_pct": float(ta.crack2d_ratio_m2_over_m1_pct),
               "shift_f1_pct": float(ta.shift_f1_pct_2d_stress),
               "a_bar": t.a_bar.to_numpy(float),
               "discrim_vs_depth": t.crack2d_ratio_m2_over_m1_pct.to_numpy(float)}
    return {"anchor": FIG2_ANCHOR, "arms": arms, "pattern": pattern,
            "gauss": gauss, "crack2d": crack2d,
            # 정합의 증거: 두 f₁ 강하가 같은 값이어야 "Δf₁으로 맞췄다"가 성립한다.
            "match_rel_err": abs(gauss["shift"][0] - crack2d["shift_f1_pct"])
            / crack2d["shift_f1_pct"],
            "gauss_over_crack_f2": (gauss["discrim_pct"] / crack2d["discrim_pct"])}


def fig2_crack_signature(data: Path, figs: Path):
    """Figure 2 (§4.1) — 균열 지문의 모드선택성과 그 **판별 패턴**.

    (a) 균열깊이 ā에 대한 이동량 1 − f_m/f_mʰ [%]를 **로그축**에 네 arm으로 겹쳐 그린다.
        비 f_m/f_mʰ를 선형축에 그리면 mode 2가 0.9990–0.99997 구간에 눌려 실명이 보이지
        않는다 — 그림이 말해야 하는 것은 "mode 1이 두 자리 위"라는 사실이므로 같은 양의
        로그 표현을 쓴다(캡션도 이 양으로 적는다).
    (b) 같은 Δf₁을 갖는 매끄러운 가우시안 경쟁모델과의 모드 패턴 비교(ā = 0.5). mode 1은
        정합으로 겹치고 mode 2에서 20배 이상 갈라진다 — 즉 판별력은 Δf₁이 아니라 패턴에
        있다. 각 팔의 판별비 D = |Δf₂/f₂|/|Δf₁/f₁|를 범례에 적는다.
    """
    s = fig2_series(data)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.1))

    # --- (a) 네 arm × 3모드
    for a in s["arms"]:
        for j, (_, mlab, ls, mk) in enumerate(FIG2_MODES):
            ax[0].plot(a["a_bar"], a["shift"][j], ls, marker=mk, ms=3.6, lw=1.2,
                       color=a["color"], alpha=0.95 if j == 0 else 0.85)
    ax[0].set_yscale("log")
    # 모드는 범례 대신 **곡선 옆에 직접** 적는다 — 12곡선에 두 개의 범례를 얹으면 어느
    # 하나가 반드시 mode-2 다발을 덮는다(그 다발이 이 그림의 논지다).
    a_hi = max(float(a["a_bar"].max()) for a in s["arms"])
    for j, (_, mlab, _, _) in enumerate(FIG2_MODES):
        y = max(float(a["shift"][j][np.argmax(a["a_bar"])]) for a in s["arms"])
        ax[0].text(a_hi * 1.035, y, mlab, fontsize=7.6, color="0.25",
                   va="center", ha="left")
    ax[0].text(a_hi * 1.035, min(float(a["shift"][1][np.argmax(a["a_bar"])])
                                 for a in s["arms"]) * 0.62,
               "Curvature-null\nblindness", fontsize=6.6, color="0.45",
               va="top", ha="left", style="italic", linespacing=1.3)
    ax[0].set_xlim(0.06, a_hi * 1.30)
    ax[0].set_xlabel(r"Crack depth $\bar a = a/h$")
    ax[0].set_ylabel(r"Frequency shift $|1 - f_m/f_m^{\,h}|$  [%]")
    ax[0].set_title("(a)")
    ax[0].grid(True, which="both", alpha=0.18)
    ax[0].set_ylim(top=ax[0].get_ylim()[1] * 14)
    ax[0].legend(handles=[plt.Line2D([], [], color=a["color"], lw=1.6,
                                     label=a["label"]) for a in s["arms"]],
                 fontsize=6.6, loc="upper left", framealpha=0.92)

    # --- (b) Δf₁ 정합 패턴 + 판별비
    x = np.arange(1, len(FIG2_MODES) + 1)
    # 네 균열 팔은 mode 2에서 0.089–0.121 %로 겹치므로(그것이 결론이다) 가로로 살짝
    # 벌려 다섯 계열이 모두 보이게 한다. 경쟁모델은 눈금 위에 그대로 둔다.
    n = len(s["pattern"])
    for i, p in enumerate(s["pattern"]):
        ax[1].plot(x + (i - 0.5 * (n - 1)) * 0.035, p["shift"], "-",
                   marker=p["marker"], ms=5, lw=1.3, color=p["color"],
                   label=f"{p['label']}  ($D$ = {p['discrim_pct']:.2f} %)")
    g = s["gauss"]
    ax[1].plot(x, g["shift"], "-", marker=g["marker"], ms=6, lw=1.8,
               color=g["color"], label=f"{g['label']}  ($D$ = "
                                       f"{g['discrim_pct']:.1f} %)")
    y_crack2d = s["pattern"][2]["shift"][1]
    ax[1].annotate(rf"$\times${s['gauss_over_crack_f2']:.0f}",
                   xy=(2, g["shift"][1]),
                   xytext=(2.15, np.sqrt(g["shift"][1] * y_crack2d)),
                   fontsize=9, color=g["color"], va="center")
    ax[1].annotate("", xy=(2.10, g["shift"][1]), xytext=(2.10, y_crack2d),
                   arrowprops=dict(arrowstyle="<->", color=g["color"], lw=1.0))
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([m for _, m, _, _ in FIG2_MODES])
    ax[1].set_xlim(0.72, 3.30)
    ax[1].set_ylabel(r"Frequency shift $|\Delta f_m/f_m|$  [%]")
    ax[1].set_title(rf"(b) $\bar a$ = {s['anchor']:g}")
    ax[1].grid(True, which="both", alpha=0.18)
    # mode-2 다발(0.09–0.12 %)이 이 패널의 논지이므로 범례를 그 위 여백에 둔다.
    lo, hi = ax[1].get_ylim()
    ax[1].set_ylim(lo * 0.55, hi * 4.5)
    ax[1].legend(fontsize=6.6, loc="upper center", framealpha=0.92)

    fig.tight_layout()
    out = figs / "fig2_crack_signature.png"
    save_canon(fig, out, dpi=200)
    plt.close(fig)
    print(f"[fig] fig2_crack_signature.png  (D: "
          + ", ".join(f"{p['label'].split(',')[0]} {p['discrim_pct']:.2f} %"
                      for p in s["pattern"])
          + f", gaussian {s['gauss']['discrim_pct']:.1f} %)")
    return out


#: **정본 그림 번호 → 산출 파일명·생성함수의 단일 정본** (설계서 F73).
#:
#: 2026-08-14에 §3.1의 임펠러 기하·모드형이 Figure 1로 신설되면서 이후 번호가 하나씩 밀렸다
#: (옛 1–5 → 새 2–6). 이때 파일명 안의 숫자(`fig3_kernels_and_sign.png`)와 정본 번호가
#: 어긋나면 조판에서 조용히 잘못된 그림이 들어간다 — 참고문헌 정합성(A12/`references.py`)과
#: 같은 방식으로 표를 코드에 두고 회귀검정이 정본과 1:1로 맞춘다.
#: `files`가 비면 **리포에 생성기가 없는 그림**이고, `note`에 그 이유를 적는다(추측 금지).
def save_canon(fig, out_png, **kw):
    """정본 그림을 **PNG + PDF**로 함께 저장한다.

    제출에는 벡터가 필요하고(축소·확대에서 글자가 깨지지 않는다) 작업·검토에는 PNG가 편하다.
    두 파일이 같은 `savefig` 호출에서 나오므로 **내용이 갈라질 수 없다** — 예전처럼 PNG만
    갱신하고 벡터를 잊는 일이 생기지 않는다. `docs/_generated/.gitignore`가 `*.png`만
    무시하므로 PDF는 추적 대상이 되는데, 그림은 재생성 가능한 산출물이라 PDF도 무시 목록에
    올려 두었다(생성 명령이 `cli figs`로 고정돼 있다).
    """
    from pathlib import Path as _P
    out_png = _P(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, **kw)
    fig.savefig(out_png.with_suffix(".pdf"), **{k: v for k, v in kw.items()
                                               if k != "dpi"})
    return out_png


#: 그림 번호 → 파일. **번호의 단일 출처**이고 정본 캡션과 1:1이어야 한다
#: (`test_submission.py::test_caption_count_matches_the_mapping`이 강제).
#:
#: ⚠️ **파일명의 숫자는 옛 번호다.** 2026-08-15 검토에서 등장 순서가 바뀌어(구 2→3, 3→4,
#: 4→5, 5→2) `fig2_crack_signature.png`가 **Fig. 3**, `fig3_recovery.png`가 **Fig. 4**,
#: `fig4_kernels_and_sign.png`가 **Fig. 5**가 됐다. 파일을 renaming하지 않은 이유: 이름이
#: 산출 명령·설계서·CSV 주석 여러 곳에 박혀 있어 한꺼번에 바꾸면 추적이 끊긴다. 그림 **안에는
#: 번호가 없으므로** 재번호는 배치 문제일 뿐이고, 그 배치를 이 dict가 단독으로 정한다.
#: 다음에 번호가 또 바뀌면 **이 dict만** 고치면 된다.
CANON_FIGURES: dict = {
    1: {"files": ("fig1_impeller_modes.png",), "maker": "fig1_impeller_modes",
        "what": "impeller geometry and mode shapes (§3.1)"},
    2: {"files": ("fig_a2_epert_mass.png",), "maker": "fig_a2_epert",
        "what": "linearization error vs measurement floor (§3.4, §4.3)",
        "note": "구 Fig 5. 검증도구가 결과보다 먼저 와야 한다는 편집 판단으로 §3.4 위치의 "
                "등장 순서를 따라 2번이 됐다."},
    3: {"files": ("fig2_crack_signature.png",), "maker": "fig2_crack_signature",
        "what": "crack signature: mode-selective shifts vs depth, crack vs smooth (§4.1)",
        "note": "구 Fig 2. 2026-08-14 신설(F77) — 그 전까지는 외부 도구로 만든 그림이라 "
                "리포에 생성기가 없었다(제출 그림 중 유일한 재현성 구멍). 인접 산출 "
                "`fig_a11_crack2d.png`(A11 진단용)는 패널 구성이 다른 별개 그림이다."},
    4: {"files": ("fig3_recovery.png",), "maker": "fig_recovery",
        "what": "production Monte-Carlo recovery statistics (§4.2)",
        "note": "구 Fig 3. 2026-08-15에 suptitle을 제거했다 — 맵 종류·σ_f/f·실현수는 "
                "캡션이 말한다."},
    5: {"files": ("fig4_kernels_and_sign.png", "fig_a3_identifiability_mass.png"),
        "maker": "fig_kernels",
        "what": "sensitivity kernels + identifiability maps (§4.3)",
        "note": "구 Fig 4. 두 파일이 한 그림의 두 부분이다."},
    6: {"files": ("fig_a4_landscape.png",), "maker": "fig_a4_landscape",
        "what": "objective landscapes over (xi_d, S_bar) (§4.3)"},
}


def make_all(data: Path, figs: Path):
    figs.mkdir(parents=True, exist_ok=True)
    npz1 = data / "fig1_impeller_modes.npz"
    if npz1.exists():
        fig1_impeller_modes(npz1, figs / "fig1_impeller_modes.png")
    else:
        print("[fig] fig1_impeller_modes.npz 없음 — `cli fig1`을 먼저 돌린다")
    try:
        fig2_crack_signature(data, figs)
    except FileNotFoundError as e:
        print(f"[fig] {e}")
    fig_kernels(data, figs)
    fig_recovery(data, figs)
    fig_a4_landscape(data, figs)
    fig_kerf(data, figs)
    fig_splitting(data, figs)
    for name, coup in (("a2_epert_map_mass.npz", "exact"), ("a2_epert_map.npz", None)):
        if (data / name).exists():
            fig_a2_epert(data / name, figs / name.replace("a2_epert_map", "fig_a2_epert")
                         .replace(".npz", ".png"), coupling=coup)
