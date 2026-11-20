"""N2 드라이버 — 조립체 국부 포켓의 θ_d 스윕·arm 비교·재해석 확인.

    python -m impeller_fingerprint.rev1.n2_run sweep  --n-r 18 --n-theta 72
    python -m impeller_fingerprint.rev1.n2_run fisher --key 18x72z2c2w60
    python -m impeller_fingerprint.rev1.n2_run nonlin --n-r 18 --n-theta 72

산출: rev1_n2_sweep_<key>.csv(1차 관측량), rev1_n2_jac_<key>.csv(야코비안),
      rev1_n2_arms_<key>.csv(CRLB), rev1_n2_nonlin_<key>.csv
θ_d는 **그 반경의 베인 0 캠버 위치에서 잰 상대각** θ_rel로 기록한다(C6: θ_rel ∈ [0°, 60°)).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from .. import geometry as geo
from . import n1_assembly as n1
from . import n1_mesh as nm
from . import n2_pocket as n2
from . import out_csv
from .n1_run import EXTENT, _healthy, _key


def _vane_theta(spec, xi):
    """베인 0 캠버의 θ(r_d) — 캠버 정합 격자는 로그나선(`FitSpec.camber`)."""
    r_d = spec.a + xi * (spec.b - spec.a)
    if hasattr(spec, "camber"):
        return float(spec.camber(r_d))
    return math.radians(spec.wrap_deg) * xi


def cmd_sweep(args):
    spec = n1.fit_spec(n_r=args.n_r, n_bay=args.n_bay, wrap_deg=args.wrap)
    am, km0, md, obs = _healthy(spec, args.n_modes)
    key = _key(spec)
    rows = []
    t0 = time.time()
    for xi in args.xi:
        for dth in args.dtheta:
            for trel in np.arange(0.0, 60.0 - 1e-9, args.theta_step):
                th = _vane_theta(spec, xi) + math.radians(trel)
                base = dict(xi=xi, dtheta_deg=dth, theta_rel_deg=trel, p0=args.p0,
                            w_r_mm=args.w_r * 1e3)
                ev = {}
                for tag, (x, t) in {"c": (xi, th),
                                    "xp": (xi + args.dxi, th), "xm": (xi - args.dxi, th),
                                    "tp": (xi, th + math.radians(args.dth_fd)),
                                    "tm": (xi, th - math.radians(args.dth_fd))}.items():
                    ev[tag] = n2.first_order(am, md, obs, km0, x, args.p0, args.w_r, t,
                                             math.radians(dth))
                for j, o in enumerate(obs):
                    r = dict(base, key=key, label=o.label, h=o.h, deg=o.deg, f_Hz=o.f_Hz,
                             resolvable=o.resolvable,
                             mean=ev["c"][0][j], split=ev["c"][1][j],
                             dmean_dxi=(ev["xp"][0][j] - ev["xm"][0][j]) / (2 * args.dxi),
                             dsplit_dxi=(ev["xp"][1][j] - ev["xm"][1][j]) / (2 * args.dxi),
                             dmean_dth=(ev["tp"][0][j] - ev["tm"][0][j])
                             / (2 * math.radians(args.dth_fd)),
                             dsplit_dth=(ev["tp"][1][j] - ev["tm"][1][j])
                             / (2 * math.radians(args.dth_fd)))
                    rows.append(r)
            print(f"[n2] ξ={xi} Δθ={dth}° done ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_csv(f"n2_sweep_{key}"), index=False)
    # 조립체 ξ는 그 스펙의 a·b로 정의된다 — fisher가 mm 환산에 쓴다
    out_csv(f"n2_sweep_{key}").with_suffix(".json").write_text(
        json.dumps({"key": key, "a": spec.a, "b": spec.b, "wrap_deg": spec.wrap_deg,
                    "t_front": spec.tf, "t_back": spec.tb, "channel": spec.channel,
                    "channel_in": spec.channel_in}))
    print(f"[n2] → {out_csv(f'n2_sweep_{key}')}")


def _crlb(J, sig, n_target=2, extent=EXTENT):
    Jw = J / sig[:, None]
    F = Jw.T @ Jw
    try:
        Fi = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return np.inf, np.inf, np.nan
    if np.any(np.diag(Fi)[:n_target] < 0):
        return np.inf, np.inf, np.nan
    cond = np.linalg.cond(Jw)
    return float(np.sqrt(Fi[0, 0]) * extent * 1e3), float(np.sqrt(Fi[1, 1])), float(cond)


def cmd_fisher(args):
    """arm A/A′/B/C의 CRLB — 측정 가능(분해 가능) 관측량만, 잡음 쌍평균 2c·분리 4c·singlet 2√2c."""
    sw = pd.read_csv(out_csv(f"n2_sweep_{args.key}"))
    sw = sw[sw.resolvable & (sw.f_Hz <= args.f_cut)]
    mp = out_csv(f"n2_sweep_{args.key}").with_suffix(".json")
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    ext = (meta["b"] - meta["a"]) if "a" in meta else EXTENT    # 옛 sweep(메타 없음)은 레일 반경
    c = args.c
    rows = []
    for (xi, dth, trel), g in sw.groupby(["xi", "dtheta_deg", "theta_rel_deg"]):
        g = g.sort_values("f_Hz")
        p0 = float(g.p0.iloc[0])
        sig_mean = np.where(g.deg == 2, 2 * c, 2 * math.sqrt(2) * c)
        # 열: ξ, p0, θ  (평균은 p0에 선형 → ∂/∂p0 = mean/p0)
        Jm = np.column_stack([g.dmean_dxi, g["mean"] / p0, g.dmean_dth])
        dbl = g[g.deg == 2]
        Js = np.column_stack([dbl.dsplit_dxi, dbl.split / p0, dbl.dsplit_dth])
        sig_s = np.full(len(dbl), 4 * c)
        arms = {
            "A_mean_theta_known": (Jm[:, :2], sig_mean),
            "A'_mean_theta_nuisance": (Jm, sig_mean),
            "B_mean+split_theta_known": (np.vstack([Jm[:, :2], Js[:, :2]]),
                                         np.concatenate([sig_mean, sig_s])),
            "C_mean+split_theta_nuisance": (np.vstack([Jm, Js]),
                                            np.concatenate([sig_mean, sig_s])),
        }
        for arm, (J, sig) in arms.items():
            cx, cp, cond = _crlb(J, sig, extent=ext)
            rows.append({"key": args.key, "xi": xi, "dtheta_deg": dth, "theta_rel_deg": trel,
                         "p0": p0, "c": c, "f_cut": args.f_cut, "arm": arm,
                         "n_obs": J.shape[0], "crlb_xi_mm": cx, "crlb_p0": cp,
                         "cond2_w": cond,
                         "n_mean_obs": len(g), "n_split_obs": len(dbl),
                         "max_split_over_mean": float(np.nanmax(np.abs(dbl.split))
                                                      / max(np.max(np.abs(g["mean"])), 1e-300))})
    df = pd.DataFrame(rows)
    tag = "" if args.f_cut == 2.0e4 else f"_f{args.f_cut / 1e3:g}k"
    df.to_csv(out_csv(f"n2_arms_{args.key}{tag}"), index=False)
    print(f"[n2] → {out_csv(f'n2_arms_{args.key}{tag}')} ({len(df)} rows)")


def cmd_nonlin(args):
    spec = n1.fit_spec(n_r=args.n_r, n_bay=args.n_bay, wrap_deg=args.wrap)
    am, km0, md, obs = _healthy(spec, args.n_modes)
    key = _key(spec)
    rows = []
    for xi in args.xi:
        for dth in args.dtheta:
            for trel in args.theta_rel:
                th = _vane_theta(spec, xi) + math.radians(trel)
                lin = n2.first_order(am, md, obs, km0, xi, args.p0, args.w_r, th,
                                     math.radians(dth))
                t0 = time.time()
                mn, sp, mac = n2.resolved(am, md, obs, xi, args.p0, args.w_r, th,
                                          math.radians(dth))
                for j, o in enumerate(obs):
                    rows.append({"key": key, "xi": xi, "dtheta_deg": dth,
                                 "theta_rel_deg": trel, "p0": args.p0, "label": o.label,
                                 "h": o.h, "deg": o.deg, "resolvable": o.resolvable,
                                 "mean_lin": lin[0][j], "split_lin": lin[1][j],
                                 "mean_res": mn[j], "split_res": sp[j], "mac": mac[j]})
                print(f"[n2] nonlin ξ={xi} Δθ={dth} θrel={trel}: {time.time() - t0:.0f}s",
                      flush=True)
                pd.DataFrame(rows).to_csv(out_csv(f"n2_nonlin_{key}"), index=False)


def cmd_vaneroot(args):
    """베인 1매 뿌리창 손상의 1차 쌍평균·분리 — 정확 C6(건전 분리 0)에서 b5의 1.3–1.7 %를 대체.

    두 손상: (a) 강성만 dK(쌍행렬 단일부호 → 분리 ≤ 2|쌍평균|), (b) 강성·질량 동률(막 지배 감육
    근사). 창은 b5와 같은 스팬 0–25 %(뿌리 = 입구). 1차이므로 dK에 선형 — 60 %는 1차 외삽값이며
    재해석 확인은 `nonlin`과 별도다.
    """
    spec = n1.fit_spec(n_r=args.n_r, n_bay=args.n_bay, wrap_deg=args.wrap)
    am, km0, md, obs = _healthy(spec, args.n_modes)
    key = _key(spec)
    rows = []
    for span in ((0.0, 0.25), (0.0, 0.5), (0.0, 1.0)):
        kr = n1.assemble_region(am, n1.vane_root_mask(am, 0, span))
        ef, kf = n1.energy_fractions(md, obs, kr)
        for case, dk, dm in (("stiffness_only", args.dk, 0.0),
                             ("stiffness_and_mass", args.dk, args.dk)):
            mean, split = n2.window_first_order(md, obs, kr, dk, dm)
            for j, o in enumerate(obs):
                rows.append({"key": key, "span_lo": span[0], "span_hi": span[1], "case": case,
                             "dK": dk, "dM": dm, "label": o.label, "h": o.h, "deg": o.deg,
                             "f_Hz": o.f_Hz, "resolvable": o.resolvable,
                             "Efrac_window": ef[j], "Tfrac_window": kf[j],
                             "eta_mean": mean[j], "dfreq_mean_pct": 50.0 * mean[j],
                             "eta_split": split[j], "dfreq_split_pct_of_f": 50.0 * split[j],
                             "split_over_absmean": (split[j] / abs(mean[j])
                                                    if np.isfinite(split[j]) and mean[j] != 0
                                                    else np.nan)})
    pd.DataFrame(rows).to_csv(out_csv(f"n2_vaneroot_{key}"), index=False)
    print(f"[n2] → {out_csv(f'n2_vaneroot_{key}')}")


def cmd_vanethin(args):
    """베인 1매 뿌리창의 **물리적 두께 감소**(캠버 기준 대칭, b5와 같은 이상화) — 재해석 쌍평균·분리.

    정확 C6 조립체(건전 분리 0, 실측 형상)에서 b5 사면체 조립체를 대신한다. 손상 모드는 분리 허용 부분공간 MAC로
    짝짓는다(`n2_pocket.track_split`).
    """
    spec = n1.fit_spec(n_r=args.n_r, n_bay=args.n_bay, wrap_deg=args.wrap, support=args.support)
    am, km0, md, obs = _healthy(spec, args.n_modes)
    key = _key(spec)
    ef, tf = n1.energy_fractions(md, obs, n1.assemble_region(am, n1.vane_root_mask(am, 0, (0.0, args.span_hi))))
    rows = []
    for frac in args.frac:
        c = nm.vane_thin_coors(am.coors, am.info["grid_idx"], spec, vane=0, span=(0.0, args.span_hi),
                               frac=frac)
        km = n1.assemble(am, c)
        mdd = n1.solve(km, md.phi.shape[1] + 6)
        mean, split, mac = n2.track_split(md, mdd, obs)
        for j, o in enumerate(obs):
            rows.append({"key": key, "support": args.support, "frac": frac, "span_hi": args.span_hi,
                         "label": o.label, "h": o.h, "deg": o.deg, "f_Hz": o.f_Hz, "resolvable": o.resolvable,
                         "eta_mean": mean[j], "dfreq_mean_pct": 100 * (math.sqrt(1 + mean[j]) - 1),
                         "eta_split": split[j], "split_pct_of_f": 50.0 * split[j], "mac": mac[j],
                         "Efrac_window": ef[j], "Tfrac_window": tf[j], "ndof": md.ndof,
                         "vol_removed_mm3": (km0.volume - km.volume) * 1e9})
        print(f"[n2] vanethin frac={frac}: 부피 감소 {(km0.volume - km.volume) * 1e9:.2f} mm³", flush=True)
        pd.DataFrame(rows).to_csv(out_csv(f"n2_vanethin_{key}"), index=False)
    print(f"[n2] → {out_csv(f'n2_vanethin_{key}')}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="n2_run")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sweep", cmd_sweep), ("nonlin", cmd_nonlin)):
        sp = sub.add_parser(name)
        sp.add_argument("--n-r", type=int, default=24)
        sp.add_argument("--n-bay", type=int, default=12)
        sp.add_argument("--wrap", type=float, default=geo.ASBUILT.wrap_deg)
        sp.add_argument("--n-modes", type=int, default=60)
        sp.add_argument("--xi", type=float, nargs="+", default=[0.2, 0.5, 0.8])
        sp.add_argument("--dtheta", type=float, nargs="+", default=[15.0, 30.0, 60.0])
        sp.add_argument("--p0", type=float, default=0.5)
        sp.add_argument("--w-r", type=float, default=3.0e-3)
        if name == "sweep":
            sp.add_argument("--theta-step", type=float, default=7.5)
            sp.add_argument("--dxi", type=float, default=0.01)
            sp.add_argument("--dth-fd", type=float, default=1.0)
        else:
            sp.add_argument("--theta-rel", type=float, nargs="+", default=[0.0, 30.0])
        sp.set_defaults(func=fn)
    sp = sub.add_parser("vaneroot")
    sp.add_argument("--n-r", type=int, default=24)
    sp.add_argument("--n-bay", type=int, default=12)
    sp.add_argument("--wrap", type=float, default=geo.ASBUILT.wrap_deg)
    sp.add_argument("--n-modes", type=int, default=60)
    sp.add_argument("--dk", type=float, default=0.6)
    sp.set_defaults(func=cmd_vaneroot)
    sp = sub.add_parser("vanethin")
    sp.add_argument("--n-r", type=int, default=32)
    sp.add_argument("--n-bay", type=int, default=16)
    sp.add_argument("--wrap", type=float, default=geo.ASBUILT.wrap_deg)
    sp.add_argument("--support", default="inner_edge", choices=["inner_edge", "back_sheet"])
    sp.add_argument("--n-modes", type=int, default=60)
    sp.add_argument("--frac", type=float, nargs="+", default=[0.3, 0.6])
    sp.add_argument("--span-hi", type=float, default=0.25)
    sp.set_defaults(func=cmd_vanethin)
    sp = sub.add_parser("fisher")
    sp.add_argument("--key", default="fit24x12v1z2c2w90")
    sp.add_argument("--c", type=float, default=1e-3)
    sp.add_argument("--f-cut", type=float, default=2.0e4)
    sp.set_defaults(func=cmd_fisher)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
