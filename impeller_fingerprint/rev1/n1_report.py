"""N1 분석 — 조립체 커널 대 레일 커널, 부호반전 반경, 모드선택 요약 표·그림.

    python -m impeller_fingerprint.rev1.n1_report --key 18x72z2c2w60

비교 규약: 둘 다 **S̄ → 0 극한의 단위 심각도 응답** k(ξ) = η/S̄ (고유치 기반 η).
레일은 정본 1차 질량포함 맵(`forward.eta_bar_linear_mass`)으로 두 결합법칙을 낸다 —
"exact"(정본 균일판 d_M = 1−(1−d_K)^{1/3})와 "sandwich_asbuilt"(실측 2면판, 조립체 감육과 같은
물리 손상). 조립체는 N1 커널 스캔(전면 슈라우드 실제 감육).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import forward as fwd
from .. import geometry as geo
from .. import kernels as ker
from . import n1_assembly as n1
from . import out_csv, out_fig

DEFAULT_ANALOG = ("h0:m0n0", "h1:m1n0", "h2:m2n0", "h3:m3n0")
RAIL_MODES = ((0, 0), (1, 0), (2, 0), (3, 0))


def rail_kernels(xi_grid, law: str, s_probe: float = 1e-4, w: float = n1.W_GAUSS):
    pool = [ker.mode_kernel(geo.DISK, m=m, n=n) for m, n in RAIL_MODES]
    out = np.empty((len(xi_grid), len(pool)))
    for i, xi in enumerate(xi_grid):
        out[i] = fwd.eta_bar_linear_mass(pool, xi, s_probe, w, geo.DISK,
                                         coupling=law) / s_probe
    return out, [p.label for p in pool]


def zero_crossings(x, y):
    z = []
    for i in range(len(x) - 1):
        if y[i] == 0:
            z.append(float(x[i]))
        elif y[i] * y[i + 1] < 0:
            z.append(float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])))
    return z


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--key", default="18x72z2c2w60")
    p.add_argument("--prefix", default="n1", help="n1(N1 조립체) 또는 n6(N6 허브 고정·자유-자유)")
    args = p.parse_args(argv)
    pre = args.prefix
    kdf = pd.read_csv(out_csv(f"{pre}_kernels_{args.key}"))
    reg = pd.read_csv(out_csv(f"{pre}_registry_{args.key}"))
    xi = np.sort(kdf.xi.unique())
    rows = []
    # 조립체 패널: n = 0 계열 전부(C6가 가른 m = 3 singlet 둘과 h = 2의 m4n0 포함 — D-최적이 쓰는 것)
    labs_asm = [l for l in ("h0:m0n0", "h1:m1n0", "h2:m2n0", "h3:m3n0", "h3:m3n0#2", "h2:m4n0")
                if l in set(reg.label)]
    for lab in reg.label:
        k = kdf[kdf.label == lab].sort_values("xi").k.to_numpy()
        rows.append({"model": f"assembly_{args.key}", "label": lab,
                     "xi_star": "|".join(f"{z:.3f}" for z in zero_crossings(xi, k)),
                     "k_min": float(k.min()), "k_max": float(k.max()),
                     "f_Hz": float(reg[reg.label == lab].f_Hz.iloc[0]),
                     "resolvable": bool(reg[reg.label == lab].resolvable.iloc[0])})
    rail = {}
    for law in ("exact", "sandwich_asbuilt"):
        K, labels = rail_kernels(xi, law)
        rail[law] = (K, labels)
        for j, lab in enumerate(labels):
            rows.append({"model": f"rail_{law}", "label": lab,
                         "xi_star": "|".join(f"{z:.3f}" for z in zero_crossings(xi, K[:, j])),
                         "k_min": float(K[:, j].min()), "k_max": float(K[:, j].max()),
                         "f_Hz": np.nan, "resolvable": True})
    pd.DataFrame(rows).to_csv(out_csv(f"{pre}_signrev_{args.key}"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    cols = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for ax, law, title in ((axs[0], "exact", "Rail, monolithic law"),
                           (axs[1], "sandwich_asbuilt", "Rail, measured-section law")):
        K, labels = rail[law]
        for j, lab in enumerate(labels):
            ax.plot(xi, K[:, j], color=cols[j], label=lab)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("Damage location ξ_d")
    axs[0].set_ylabel("Unit-severity response η/S̄_D")
    for j, lab in enumerate(labs_asm):
        k = kdf[kdf.label == lab].sort_values("xi").k.to_numpy()
        axs[2].plot(xi, k, color=cols[j % 6], label=lab,
                    ls="--" if lab in ("h3:m3n0#2", "h2:m4n0") else "-")
    axs[2].axhline(0, color="k", lw=0.6)
    axs[2].set_title("C6 assembly, one-face thinning")
    axs[2].set_xlabel("Damage location ξ_d")
    for ax in axs:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig(f"{pre}_kernels_{args.key}"), dpi=150)
    print(f"[n1r] → {out_csv(f'{pre}_signrev_{args.key}')}, {out_fig(f'{pre}_kernels_{args.key}')}")


if __name__ == "__main__":
    main()


def nonlin_compare(key: str = "fit24x12v1z2c2w90", c: float = 1e-3, prefix: str = "n1",
                   pool: str | None = None) -> pd.DataFrame:
    """재해석 유한차분 야코비안(`n1_run nonlin`·`n6_run nonlin`) 대 1차 커널 야코비안의 CRLB 비교.

    같은 관측량 집합·같은 잡음(쌍평균 2c, singlet 2√2c)에서 CRLB_ξ·CRLB_S̄를 나란히 낸다.
    재해석에서 짝짓기(MAC < 0.8)에 실패한 관측량은 두 쪽 모두에서 뺀다.
    `pool`을 주면 관측량 집합을 같은 키의 선택 결과(`<prefix>_selection_<key>.csv`, S̄ = 5 %)에서 가져온다
    (default_analog·d_optimal_4·all_resolvable). 없으면 08-13 조립체의 고정 집합이다.
    ξ의 mm 환산은 커널 메타의 a·b(조립체 반경)를 쓴다 — 없으면 레일 반경.
    """
    import json
    from . import n1_assembly as n1
    from .n1_run import EXTENT, _asm_J
    nl = pd.read_csv(out_csv(f"{prefix}_nonlin_{key}"))
    kdf = pd.read_csv(out_csv(f"{prefix}_kernels_{key}"))
    reg = pd.read_csv(out_csv(f"{prefix}_registry_{key}")).set_index("label")
    meta_p = out_csv(f"{prefix}_kernels_{key}").with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    ext = (meta["b"] - meta["a"]) if "a" in meta else EXTENT
    sel = pd.read_csv(out_csv(f"{prefix}_selection_{key}")) if pool else None
    fixed = {
        "rail_analog": ["h0:m0n0", "h1:m1n0", "h2:m2n0", "h3:m3n0"],
        "dopt_n0": ["h2:m2n0", "h3:m3n0", "h3:m3n0#2", "h2:m4n0"],
        "all_n0": ["h0:m0n0", "h1:m1n0", "h2:m2n0", "h3:m3n0", "h3:m3n0#2", "h2:m4n0"],
    }
    rows = []
    for (xi, s_bar), g in nl.groupby(["xi", "s_bar"]):
        g = g.set_index("label")
        if pool:
            r = sel[np.isclose(sel.xi, xi) & np.isclose(sel.s_bar, 0.05) & (sel.pool == pool)
                    & sel.model.str.startswith("assembly")]
            sets = {name: str(r[r.set == st].labels.iloc[0]).split("|")
                    for name, st in (("rail_analog", "default_analog"), ("dopt", "d_optimal_4"),
                                     ("all", "all_resolvable")) if (r.set == st).any()}
        else:
            sets = fixed
        for name, labs in sets.items():
            labs = [l for l in labs if l in g.index and l in reg.index and g.loc[l, "mac"] >= 0.8
                    and np.isfinite(g.loc[l, ["J_xi", "J_s", "eta"]].to_numpy(float)).all()]
            if len(labs) < 2:
                continue
            sig = reg.loc[labs, "sigma_factor"].to_numpy() * c
            Jn = g.loc[labs, ["J_xi", "J_s"]].to_numpy()
            Jl = _asm_J(kdf, labs, xi, s_bar)
            mn = n1.fisher_metrics(Jn, sig, ext)
            ml = n1.fisher_metrics(Jl, sig, ext)
            rows.append({"key": key, "xi": xi, "s_bar": s_bar, "pool": pool or "fixed_0813", "set": name,
                         "n_obs": len(labs), "labels": "|".join(labs),
                         "crlb_xi_lin": ml["crlb_xi_mm"], "crlb_xi_nl": mn["crlb_xi_mm"],
                         "crlb_s_lin": ml["crlb_s_pp"], "crlb_s_nl": mn["crlb_s_pp"],
                         "eta_over_lin_median": float(np.median(
                             g.loc[labs, "eta"].to_numpy() / (s_bar * Jl[:, 1]))),
                         "min_mac": float(g.loc[labs, "mac"].min())})
    df = pd.DataFrame(rows)
    tag = f"_{pool}" if pool else ""
    df.to_csv(out_csv(f"{prefix}_nonlin_compare_{key}{tag}"), index=False)
    return df
