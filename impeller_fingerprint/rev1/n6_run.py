"""N6 드라이버 — 자유-자유 실측 형상 조립체(E2-I 1단계 모드 지도·2단계 홈 예측).

    python -m impeller_fingerprint.rev1.n6_run modes   --n-r 32 --n-bay 16            (테 높이 기본 = 정본 5.0 mm)
    python -m impeller_fingerprint.rev1.n6_run grooves --n-r 24 --n-bay 12
    python -m impeller_fingerprint.rev1.n6_run collar  --n-r 24 --n-bay 12 --collar-h 4.5 5.5
    python -m impeller_fingerprint.rev1.n6_run massload --key ff32x16h6w62c50
    python -m impeller_fingerprint.rev1.n6_run excite  --n-r 32 --n-bay 16      (저장된 모드형에서)
    python -m impeller_fingerprint.rev1.n6_run kernels --n-r 32 --n-bay 16 --clamp-r 9.21   (설치 상태: 허브 고정)
    python -m impeller_fingerprint.rev1.n6_run nonlin  --clamp-r 9.21 --xi 0.2 0.5 0.8 --s-bar 0.05
    python -m impeller_fingerprint.rev1.n6_run sensors --placements hub_back_r10mm --masses 0.5 1 2 5 --mic
    python -m impeller_fingerprint.rev1.n6_run sensorgrooves --placement back_vane_xi0.6 --mass 1

`--clamp-r`(mm)를 주면 r ≤ clamp_r 허브 절점을 고정한다(축 슬리브가 허브를 잡는 설치 상태). 없으면 자유-자유.

산출(PAPER3_OUT/data/paper3): rev1_n6_modes_<key>.csv(모드 지도), rev1_n6_modes_<key>.json(질량·강체·치수),
rev1_n6_grooves_<key>.csv(홈 예측), rev1_n6_collar_<key>.csv(테 높이 괄호), rev1_n6_massload_<key>.csv(질량부하
한계), rev1_n6_visibility_<key>.csv·rev1_n6_excite_<key>.csv(+ .json)(가진·응답점 — `n6_excite`),
rev1_n6_registry_<key>.csv·rev1_n6_kernels_<key>.csv(+ .json)(가우시안 감육 커널 — `n1_run select --prefix n6`이
읽는 N1과 같은 형식).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .. import geometry as geo
from . import n1_assembly as n1a
from . import n1_mesh as nm
from . import n6_excite as nx
from . import n6_freefree as n6
from . import n6_sensors as ns
from . import out_csv
from .n1_run import _git_head


def _spec(args, collar_h=None) -> n6.FFSpec:
    base = nm.FitSpec(n_r=args.n_r, n_bay=args.n_bay, wrap_deg=args.wrap)
    clamp = getattr(args, "clamp_r", None)
    return n6.FFSpec(base=base, n_hub=args.n_hub, n_z_collar=args.n_zc,
                     collar_h=(args.collar_h[0] if collar_h is None else collar_h) * 1e-3,
                     clamp_r=None if clamp is None else clamp * 1e-3)


def _key(ff: n6.FFSpec) -> str:
    fs = ff.base
    key = (f"ff{fs.n_r}x{fs.n_bay}h{ff.n_hub}w{int(round(fs.wrap_deg))}"
           f"c{ff.collar_h * 1e4:.0f}")
    return key if ff.clamp_r is None else f"{key}k{ff.clamp_r * 1e4:.0f}"


def _healthy(ff, n_modes):
    t0 = time.time()
    fm = n6.build(ff)
    km = n6.assemble(fm)
    md, rigid = n6.solve(km, n_modes, n_rigid=ff.n_rigid)
    obs = n6.classify(fm, md)
    print(f"[n6] {_key(ff)} ndof={md.ndof} 강체 {np.round(rigid, 3)} Hz, 탄성 {md.freqs[0]:.1f}–"
          f"{md.freqs[-1]:.1f} Hz, 관측량 {len(obs)} ({time.time() - t0:.0f}s)", flush=True)
    return fm, km, md, rigid, obs


def _registry(ff, fm, md, obs, ef):
    rows = []
    for j, o in enumerate(obs):
        r = {"key": _key(ff), "obs": j, "label": o.label, "h": o.h, "deg": o.deg, "f_Hz": o.f_Hz,
             "m_dom": o.m_dom, "n_rad": o.n_rad, "vane_share": o.vane_share,
             "sigma_factor": o.sigma_factor, "resolvable": o.resolvable, "cluster": o.cluster,
             "members": "|".join(str(i) for i in o.members), "ndof": md.ndof}
        for name, v in ef.items():
            r[f"Efrac_{name}"] = float(v[j])
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_modes(args):
    ff = _spec(args)
    fm, km, md, rigid, obs = _healthy(ff, args.n_modes)
    ef = n6.energy_fractions(fm, md, obs)
    key = _key(ff)
    reg = _registry(ff, fm, md, obs, ef)
    reg.to_csv(out_csv(f"n6_modes_{key}"), index=False)
    fs = ff.base
    meta = {"key": key, "git": _git_head(), "ndof": md.ndof, "n_modes": args.n_modes,
            "rigid_Hz": [float(x) for x in rigid], "mass_model_g": float(fs.rho * km.volume * 1e3),
            "mass_measured_g": geo.ASBUILT.mass_g, "a": fs.a, "b": fs.b, "r_bore": ff.r_bore,
            "t_front": fs.t_front, "t_back": fs.t_back, "channel": fs.channel, "channel_in": fs.channel_in, "t_vane": fs.t_vane,
            "wrap_deg": fs.wrap_deg, "r_vane_in": fs.r_vane_in, "collar_t": ff.collar_t,
            "collar_h": ff.collar_h, "support": "free-free (no constraint)",
            "joint": "vane–shroud continuous (idealization of spot/projection welds)",
            "hub": "flat back-plate annulus to the bore (embossed bead idealized as flat)"}
    out_csv(f"n6_modes_{key}").with_suffix(".json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(out_csv(f"n6_shapes_{key}").with_suffix(".npz"), coors=fm.coors, conn=fm.conn,
                        freqs=md.freqs, vshape=md.vshape.astype(np.float32),
                        grid_idx=fm.info["grid_idx"], cell_kind=fm.info["cell_kind"])
    print(f"[n6] → {out_csv(f'n6_modes_{key}')}  질량 모델 {meta['mass_model_g']:.2f} g 대 실측 "
          f"{meta['mass_measured_g']:.2f} g")


def cmd_grooves(args):
    ff = _spec(args)
    fm, km, md, rigid, obs = _healthy(ff, args.n_modes)
    key = _key(ff)
    rows = []
    t0 = time.time()
    for xi in args.xi:
        for dep in args.depths:
            eta1 = n6.groove_first_order(fm, md, obs, xi, dep * 1e-3)
            res = None
            if any(abs(dep - d) < 1e-12 for d in args.resolve_depths):
                res = n6.groove_resolved(fm, md, obs, xi, dep * 1e-3)
            for j, o in enumerate(obs):
                r = {"key": key, "xi": xi, "r_c_mm": (ff.a + xi * ff.extent) * 1e3,
                     "width_mm": 0.2 * ff.extent * 1e3, "depth_mm": dep,
                     "depth_frac_front": dep * 1e-3 / ff.base.t_front, "label": o.label, "h": o.h,
                     "deg": o.deg, "f_Hz": o.f_Hz, "resolvable": o.resolvable,
                     "eta_lin": float(eta1[j]), "df_pct_lin": 100 * (math.sqrt(1 + eta1[j]) - 1)}
                if res is not None:
                    eta, mac = res
                    r.update(eta_res=float(eta[j]), mac=float(mac[j]),
                             df_pct_res=(100 * (math.sqrt(1 + eta[j]) - 1)
                                         if np.isfinite(eta[j]) else np.nan))
                rows.append(r)
            print(f"[n6] groove ξ={xi} δ={dep} mm ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_csv(f"n6_grooves_{key}"), index=False)
    print(f"[n6] → {out_csv(f'n6_grooves_{key}')}")


def cmd_collar(args):
    rows = []
    key0 = None
    for h in args.collar_h:
        ff = _spec(args, collar_h=h)
        key0 = key0 or _key(ff)
        fm, km, md, rigid, obs = _healthy(ff, args.n_modes)
        for o in obs:
            rows.append({"collar_h_mm": h, "label": o.label, "h": o.h, "deg": o.deg,
                         "f_Hz": o.f_Hz, "m_dom": o.m_dom, "n_rad": o.n_rad,
                         "mass_model_g": ff.base.rho * km.volume * 1e3})
    pd.DataFrame(rows).to_csv(out_csv(f"n6_collar_{key0}"), index=False)
    print(f"[n6] → {out_csv(f'n6_collar_{key0}')}")


def cmd_massload(args):
    """접촉 질량부하 한계(보충 Table S2 대체) — 자유-자유 실측 모델, 저장된 질량정규 모드형에서.

    δf/f = −½ m_a |φ(x)|² ⇒ 예산 β에서 m_a ≤ 2β/|φ|²_max. 앞판 바깥면(측정면·반사테이프 자리)과
    전 절점(보수적) 두 가지로 낸다. doublet은 두 성분 중 큰 값을 쓴다.
    """
    key = args.key
    z = np.load(out_csv(f"n6_shapes_{key}").with_suffix(".npz"))
    meta = json.loads(out_csv(f"n6_modes_{key}").with_suffix(".json").read_text())
    reg = pd.read_csv(out_csv(f"n6_modes_{key}")).sort_values("f_Hz")
    coors, vsh = z["coors"], z["vshape"].astype(float)
    rr = np.hypot(coors[:, 0], coors[:, 1])
    z_top = meta["t_back"] + meta["channel"] + meta["t_front"]
    front = np.isclose(coors[:, 2], z_top, atol=1e-9) & (rr >= meta["a"] + meta["collar_t"] - 1e-9)
    mass_g = meta["mass_model_g"]
    rows = []
    for _, o in reg.head(args.n_obs).iterrows():
        mem = [int(i) for i in str(o["members"]).split("|")]
        p2 = np.stack([(vsh[k].reshape(-1, 3) ** 2).sum(axis=1) for k in mem]).max(axis=0)
        pf, pa = float(p2[front].max()), float(p2.max())
        jf = int(np.nonzero(front)[0][np.argmax(p2[front])])
        r = {"key": key, "label": o["label"], "h": int(o["h"]), "deg": int(o["deg"]), "f_Hz": o["f_Hz"],
             "mass_model_g": mass_g, "phi2_front_max_per_kg": pf, "phi2_all_max_per_kg": pa,
             "m_eff_front_g": 1e3 / pf, "m_eff_all_g": 1e3 / pa,
             "antinode_front_r_mm": float(rr[jf] * 1e3)}
        for tag, beta in (("0p05pct", 5e-4), ("0p1pct", 1e-3)):
            r[f"m_limit_front_mg_{tag}"] = 2 * beta / pf * 1e6
            r[f"m_limit_all_mg_{tag}"] = 2 * beta / pa * 1e6
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv(f"n6_massload_{key}"), index=False)
    print(df[["label", "f_Hz", "m_eff_front_g", "m_limit_front_mg_0p1pct", "m_limit_all_mg_0p1pct"]]
          .round(3).to_string(index=False))
    print(f"[n6] → {out_csv(f'n6_massload_{key}')}")


def _excite_points(ff, coors, loc, per, n_vc):
    """후보점 이름 → (절점, 판 두께 | None). 앞판 윗면(베인선·베이 1/4·가운데 × ξ), 흡입구 테 윗면, 뒤판 바깥면."""
    fs = ff.base
    rr = np.hypot(coors[:, 0], coors[:, 1])
    z_in, z_top = ff.z_front
    top = np.isclose(coors[:, 2], z_top, atol=1e-9) & (rr >= ff.a + ff.collar_t - 1e-9)
    bot = np.isclose(coors[:, 2], 0.0, atol=1e-9)
    coll = np.isclose(coors[:, 2], ff.z_breaks[-1], atol=1e-9)

    def pick(mask, r_t, l=None):
        c = np.nonzero(mask & ((loc == l) if l is not None else True))[0]
        return int(c[np.argmin(np.abs(rr[c] - r_t))])

    pts = {}
    for xi in (0.3, 0.6, 0.85, 0.97):
        r_t = ff.a + xi * ff.extent
        pts[f"front_vane_xi{xi}"] = (pick(top, r_t, 0), None)
        pts[f"front_quarter_xi{xi}"] = (pick(top, r_t, n_vc + fs.n_bay // 4), fs.t_front)
        pts[f"front_centre_xi{xi}"] = (pick(top, r_t, n_vc + fs.n_bay // 2), fs.t_front)
    pts["collar_top"] = (pick(coll, ff.a + 0.5 * ff.collar_t), None)
    pts["back_hub_r9mm"] = (pick(bot, 9e-3), fs.t_back)
    pts["back_centre_xi0.97"] = (pick(bot, ff.a + 0.97 * ff.extent, n_vc + fs.n_bay // 2), fs.t_back)
    return pts, top, rr


def cmd_excite(args):
    """가진·응답점(E2-I 1단계) — 저장된 자유-자유 모드형에서 응답 가시도, 최적 고정 응답점, 해머·강구 힘 스펙트럼."""
    ff = _spec(args)
    key = _key(ff)
    fs = ff.base
    z = np.load(out_csv(f"n6_shapes_{key}").with_suffix(".npz"))
    reg = pd.read_csv(out_csv(f"n6_modes_{key}")).sort_values("f_Hz").reset_index(drop=True)
    coors, conn, vsh, gi = z["coors"], z["conn"], z["vshape"].astype(float), z["grid_idx"]
    omega = 2 * np.pi * z["freqs"]
    per = fs.n_vc + fs.n_bay
    loc = gi[:, 1] % per
    pts, top, rr = _excite_points(ff, coors, loc, per, fs.n_vc)
    members = [[int(i) for i in str(m).split("|")] for m in reg["members"]]
    V = nx.visibility(vsh, members, top)

    rows = []
    for name, (j, _) in pts.items():
        for k, o in reg.head(args.n_obs).iterrows():
            rows.append({"key": key, "point": name, "r_mm": rr[j] * 1e3, "label": o["label"], "f_Hz": o["f_Hz"],
                         "deg": int(o["deg"]), "V": float(V[k, j])})
    pd.DataFrame(rows).to_csv(out_csv(f"n6_visibility_{key}"), index=False)

    # 최적 고정 응답점: 앞판 윗면에서 낮은 n_target 관측량의 V 최솟값이 가장 큰 절점(섹터 주기라 한 곳만 낸다)
    idx = np.nonzero(top)[0]
    score = V[:args.n_target][:, idx].min(axis=0)
    jb = int(idx[np.argmax(score)])
    ring_in = np.unique(np.round(rr[top], 9))
    r_in = ring_in[-2]                                    # 바깥 끝 바로 안쪽 링(LDV 점을 가장자리에서 띄운다)
    sel_in = idx[np.isclose(rr[idx], r_in, atol=1e-9)]
    j_in = int(sel_in[np.argmax(V[:args.n_target][:, sel_in].min(axis=0))])
    best = {"node": jb, "r_mm": rr[jb] * 1e3, "loc": int(loc[jb]), "min_V": float(score.max()),
            "inner_ring": {"r_mm": rr[j_in] * 1e3, "edge_dist_mm": (ff.b - rr[j_in]) * 1e3, "loc": int(loc[j_in]),
                           "min_V": float(V[:args.n_target, j_in].min()),
                           "V": [round(float(v), 3) for v in V[:args.n_target, j_in]]},
            "labels": list(reg["label"].head(args.n_target))}

    # h0·h3 싱글릿: 바깥 끝 안쪽 링 베이 가운데의 섹터별 u_z 부호(이웃 베이 합·차로 분리)
    signs = {}
    cen = idx[np.isclose(rr[idx], r_in, atol=1e-9) & (loc[idx] == fs.n_vc + fs.n_bay // 2)]
    cen = cen[np.argsort(np.arctan2(coors[cen, 1], coors[cen, 0]))]
    for k, o in reg.head(args.n_obs).iterrows():
        if int(o["deg"]) == 1:
            uz = vsh[members[k][0]].reshape(-1, 3)[cen, 2]
            signs[o["label"]] = "".join("+" if u > 0 else "-" for u in uz * np.sign(uz[0]))

    props = nx.mass_props(coors, conn, fs.rho)
    E, nu = fs.E, fs.nu
    ex = []

    def run(name, j, t_plate, m_h, R, v0, dt, T, kind):
        phi2 = vsh[:, 3 * j + 2] ** 2
        F = nx.impact(phi2, omega, m_h, E=E, nu=nu, R=R, v0=v0, rigid=nx.rigid_phi2(coors[j], props), dt=dt, T=T)
        ex.append({"key": key, "point": name, "r_mm": rr[j] * 1e3, "impactor": kind, "m_g": m_h * 1e3,
                   "R_mm": R * 1e3, "v0": v0, "model": "N6 modal 80 + rigid 3", **nx.force_band(F, dt)})
        if t_plate is not None:
            Z = nx.plate_impedance(E, nu, fs.rho, t_plate)
            F = nx.impact(np.zeros(1), np.zeros(1), m_h, E=E, nu=nu, R=R, v0=v0, Z=Z, dt=dt, T=T)
            ex.append({"key": key, "point": name, "r_mm": rr[j] * 1e3, "impactor": kind, "m_g": m_h * 1e3,
                       "R_mm": R * 1e3, "v0": v0, "model": f"infinite plate Z={Z:.1f} N s/m", **nx.force_band(F, dt)})
        print(f"[n6] excite {name} {kind} {m_h*1e3:.3g} g: 접촉 {ex[-1 - (t_plate is not None)]['n_contact']}회",
              flush=True)

    for name, (j, t_plate) in pts.items():
        for m_h in args.hammer_g:
            run(name, j, t_plate, m_h * 1e-3, 1.27e-3, 0.5, 2e-8, 3e-3, "hammer")
    for name in ("front_centre_xi0.6", "front_centre_xi0.97"):
        j, t_plate = pts[name]
        for d in args.ball_mm:
            mb = fs.rho * math.pi / 6 * (d * 1e-3) ** 3
            run(name, j, t_plate, mb, 0.5 * d * 1e-3, 0.3, 2e-9, 2e-4, f"ball {d} mm")
    F = nx.impact(np.zeros(1), np.zeros(1), 2e-3, E=E, nu=nu, R=1.27e-3, v0=0.5, Z=1e12, dt=2e-8, T=3e-3)
    ex.append({"key": key, "point": "rigid_reference", "r_mm": np.nan, "impactor": "hammer", "m_g": 2.0,
               "R_mm": 1.27, "v0": 0.5, "model": "rigid (Hertz only)", **nx.force_band(F, 2e-8)})
    df = pd.DataFrame(ex)
    df.to_csv(out_csv(f"n6_excite_{key}"), index=False)
    meta = {"key": key, "git": _git_head(), "mass": props, "best_fixed_response": best,
            "singlet_signs_bay_centres": signs, "n_target": args.n_target,
            "note": "V = sqrt(sum_members uz^2 / max_front_top); modal truncation at 80 modes; "
                    "vane-shroud continuous joint (real: spot welds)"}
    out_csv(f"n6_excite_{key}").with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(best, ensure_ascii=False))
    print(signs)
    print(df[["point", "impactor", "m_g", "model", "n_contact", "f_m10dB_Hz", "f_m20dB_Hz", "dB_25kHz"]]
          .round(1).to_string(index=False))
    print(f"[n6] → {out_csv(f'n6_excite_{key}')}")


def cmd_kernels(args):
    """가우시안 감육(정본 w = 3 mm, 앞판 바깥면) 커널 k(ξ) = η/S̄·∂k/∂ξ — N1 `kernels`와 같은 형식."""
    ff = _spec(args)
    fm, km, md, rigid, obs = _healthy(ff, args.n_modes)
    key = _key(ff)
    ef = n6.energy_fractions(fm, md, obs)
    _registry(ff, fm, md, obs, ef).to_csv(out_csv(f"n6_registry_{key}"), index=False)
    xi_grid = np.round(np.arange(args.xi_lo, args.xi_hi + 1e-9, args.xi_step), 4)
    t0 = time.time()
    ks = n6.kernel_scan(fm, md, obs, xi_grid, s_probe=1e-3, verbose=True)
    rows = [{"key": key, "xi": xi, "label": o.label, "obs": j, "k": ks.k[i, j], "dk": ks.dk[i, j]}
            for i, xi in enumerate(ks.xi) for j, o in enumerate(obs)]
    pd.DataFrame(rows).to_csv(out_csv(f"n6_kernels_{key}"), index=False)
    fs = ff.base
    meta = {"key": key, "git": _git_head(), "ndof": md.ndof, "n_modes": args.n_modes,
            "a": fs.a, "b": fs.b, "t_front": fs.t_front, "t_back": fs.t_back, "channel": fs.channel, "channel_in": fs.channel_in,
            "t_vane": fs.t_vane, "wrap_deg": fs.wrap_deg, "r_vane_in": fs.r_vane_in, "r_bore": ff.r_bore,
            "collar_t": ff.collar_t, "collar_h": ff.collar_h, "s_probe": 1e-3, "dxi": 0.01,
            "w": n1a.W_GAUSS, "rho": fs.rho, "mass_model_g": fs.rho * km.volume * 1e3,
            "support": ("free-free" if ff.clamp_r is None
                        else f"hub clamp r <= {ff.clamp_r * 1e3:.3f} mm (installed: shaft sleeves)"),
            "damage": "front-plate outer-face Gaussian thinning, as-built two-face d_K -> p (G1)",
            "seconds_scan": time.time() - t0}
    out_csv(f"n6_kernels_{key}").with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[n6] → {out_csv(f'n6_kernels_{key}')} ({time.time() - t0:.0f}s)")


def cmd_nonlin(args):
    """재해석 유한차분 야코비안(가우시안 감육) — N1 `nonlin`과 같은 형식(`rev1_n6_nonlin_<key>.csv`)."""
    ff = _spec(args)
    key = _key(ff)
    fm, km, md, rigid, obs = _healthy(ff, args.n_modes)
    rows = []
    for xi in args.xi:
        for s_bar in args.s_bar:
            hx, hs = args.dxi, args.ds_rel * s_bar
            t0 = time.time()
            e0, mac0 = n6.resolved_gaussian(fm, md, obs, xi, s_bar)
            ep, _ = n6.resolved_gaussian(fm, md, obs, min(xi + hx, 1.0), s_bar)
            em, _ = n6.resolved_gaussian(fm, md, obs, max(xi - hx, 0.0), s_bar)
            sp_, _ = n6.resolved_gaussian(fm, md, obs, xi, s_bar + hs)
            sm_, _ = n6.resolved_gaussian(fm, md, obs, xi, s_bar - hs)
            for j, o in enumerate(obs):
                rows.append({"key": key, "xi": xi, "s_bar": s_bar, "label": o.label,
                             "eta": e0[j], "mac": mac0[j],
                             "J_xi": (ep[j] - em[j]) / (min(xi + hx, 1) - max(xi - hx, 0)),
                             "J_s": (sp_[j] - sm_[j]) / (2 * hs)})
            print(f"[n6] nonlin ξ={xi} S̄={s_bar}: {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out_csv(f"n6_nonlin_{key}"), index=False)
    print(f"[n6] → {out_csv(f'n6_nonlin_{key}')}")


def cmd_fig1(args):
    """Fig. 1 — 실측 형상 자유-자유 조립체(N6, 허브·흡입구 테 포함)의 형상과 두 모드형. 저장된 모드형(`modes`)만 쓴다.

    패널 모드는 **형상으로** 고른다(주파수 순서 금지): (a)(b) 첫 탄성 doublet, (c) 첫 h = N/2 = 3 singlet.
    npz 키는 옛 `cli fig1`과 같다(렌더러 `figures.fig1_impeller_modes` 공용) + 로그나선 캠버·앞판 윗면 높이.
    """
    from .. import figures as figs
    from .. import cli
    ff = _spec(args)
    key = _key(ff)
    fs = ff.base
    z = np.load(out_csv(f"n6_shapes_{key}").with_suffix(".npz"))
    meta = json.loads(out_csv(f"n6_modes_{key}").with_suffix(".json").read_text())
    reg = pd.read_csv(out_csv(f"n6_modes_{key}")).sort_values("f_Hz").reset_index(drop=True)
    n_modes = z["vshape"].shape[0]
    h_hat = np.full(n_modes, -1.0)
    m_dom = np.full(n_modes, -1, dtype=int)
    deg = np.zeros(n_modes, dtype=int)
    for _, o in reg.iterrows():
        for k in (int(i) for i in str(o["members"]).split("|")):
            h_hat[k], m_dom[k], deg[k] = o["h"], o["m_dom"], o["deg"]
    k_doublet = int(str(reg[reg.deg == 2].iloc[0]["members"]).split("|")[0])
    k_half = int(str(reg[(reg.deg == 1) & (reg.h == fs.n_vane // 2)].iloc[0]["members"]).split("|")[0])
    z_in, z_top = ff.z_front
    npz = cli.DATA / "fig1_impeller_modes.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz, coors=z["coors"], conn=z["conn"], freqs=z["freqs"], shapes=z["vshape"],
        panel_modes=np.array([k_doublet, k_doublet, k_half], dtype=int), m_dom=m_dom, h_hat=h_hat,
        c_rotation=np.zeros(n_modes), degeneracy=deg, n_vane=fs.n_vane, n_rigid=6,
        wrap_deg=fs.wrap_deg, z_cut=z_in - 1e-5, cut_sector_deg=np.array(args.cut_sector, dtype=float),
        t_sheet=fs.t_back, channel=fs.channel, a=fs.a, b=fs.b, n_theta=fs.n_theta, n_r=ff.n_cells_r,
        n_dof=int(meta["ndof"]), mass_fem_g=float(meta["mass_model_g"]),
        r_vane_in=fs.r_vane_in, camber_sign=-1.0, z_front_top=z_top, source=f"n6 {key}")
    print(f"[n6] → {npz}  panel (a)(b) mode {k_doublet} f={z['freqs'][k_doublet]:.1f} Hz, "
          f"(c) mode {k_half} f={z['freqs'][k_half]:.1f} Hz")
    figs.fig1_impeller_modes(npz, cli.FIGS / "fig1_impeller_modes.png")


def cmd_vanethin(args):
    """베인 1매 뿌리창 물리적 두께 감소(캠버 기준 대칭) — 설치 상태(허브 고정)·자유-자유 실측 형상 조립체."""
    from . import n2_pocket as n2
    ff = _spec(args)
    key = _key(ff)
    fm, km, md, rigid, obs = _healthy(ff, args.n_modes)
    ck, cv, cs = fm.info["cell_kind"], fm.info["cell_vane"], fm.info["cell_span"]
    win = (ck == 1) & (cv == 0) & (cs >= 0.0) & (cs <= args.span_hi)
    ef, tf = n1a.energy_fractions(md, obs, n6.assemble(fm, cell_mask=win))   # 창의 변형·운동에너지 분율(1차 예측용)
    rows = []
    for frac in args.frac:
        c = nm.vane_thin_coors(fm.coors, fm.info["grid_idx"], ff.base, vane=0, span=(0.0, args.span_hi),
                               frac=frac)
        kd = n6.assemble(fm, c)
        mdd, _ = n6.solve(kd, md.phi.shape[1] + 6, n_rigid=ff.n_rigid)
        mean, split, mac = n2.track_split(md, mdd, obs)
        for j, o in enumerate(obs):
            rows.append({"key": key, "frac": frac, "span_hi": args.span_hi, "label": o.label, "h": o.h,
                         "deg": o.deg, "f_Hz": o.f_Hz, "resolvable": o.resolvable, "eta_mean": mean[j],
                         "dfreq_mean_pct": 100 * (math.sqrt(1 + mean[j]) - 1), "eta_split": split[j],
                         "split_pct_of_f": 50.0 * split[j], "mac": mac[j],
                         "Efrac_window": ef[j], "Tfrac_window": tf[j], "ndof": md.ndof,
                         "vol_removed_mm3": (km.volume - kd.volume) * 1e9})
        print(f"[n6] vanethin frac={frac}: 부피 감소 {(km.volume - kd.volume) * 1e9:.2f} mm³", flush=True)
        pd.DataFrame(rows).to_csv(out_csv(f"n6_vanethin_{key}"), index=False)
    print(f"[n6] → {out_csv(f'n6_vanethin_{key}')}")


def cmd_sensors(args):
    """E2-I, LDV 없이 — 부착식 가속도계(집중질량)의 질량부하·가시도와 근거리 마이크 음압(자유-자유).

    산출: rev1_n6_sensor_massload_<key>_<tag>.csv(후보 부착점 × 질량 × 관측량: 쌍평균 Δf/f, doublet 분리, MAC,
    가속도계 가시도), `--mic`이면 rev1_n6_mic_<key>.csv(후보 마이크 위치 × 관측량: 1 µm당 음압).
    """
    ff = _spec(args)
    if ff.clamp_r is not None:
        raise SystemExit("sensors: 자유-자유만 지원한다(구속 모형은 자유도 대응이 다르다)")
    fm, km, md, rigid, obs = _healthy(ff, args.n_modes)
    key = _key(ff)
    obs_t = obs[:args.n_obs]
    pl = ns.placements(fm)
    names = args.placements or list(pl)
    rows, t0 = [], time.time()
    for name in names:
        face, node = pl[name]
        r_c = float(fm.r_v[node])
        th = math.atan2(fm.coors[node, 1], fm.coors[node, 0])
        vis_h = [ns.accel_visibility(fm, md, o.members, node) for o in obs_t]
        for m_g in args.masses:
            rad = ns.footprint_radius_mm(m_g)
            nodes = ns.footprint(fm, face, r_c, th, rad * 1e-3)
            mdl, _ = n6.solve(ns.with_mass(km, nodes, m_g * 1e-3), args.n_modes, n_rigid=ff.n_rigid)
            eta, split, mac, idx = ns.match(md, mdl, obs_t)
            fo = ns.first_order_mass_eta(md, obs_t, nodes, m_g * 1e-3)
            for j, o in enumerate(obs_t):
                rows.append({"key": key, "placement": name, "face": face, "r_mm": r_c * 1e3,
                             "theta_deg": math.degrees(th), "mass_g": m_g, "footprint_r_mm": rad,
                             "n_nodes": int(nodes.size), "label": o.label, "h": o.h, "deg": o.deg,
                             "f_Hz": o.f_Hz, "resolvable": o.resolvable,
                             "df_pct": 100 * (math.sqrt(1 + eta[j]) - 1) if np.isfinite(eta[j]) else np.nan,
                             "df_pct_first_order": 100 * (math.sqrt(max(1 + fo[j], 0.0)) - 1),
                             "split_pct": 50 * split[j] if np.isfinite(split[j]) else np.nan,
                             "mac": mac[j], "vis_healthy": vis_h[j],
                             "vis_loaded": ns.accel_visibility(fm, mdl, idx[j], node)})
            pd.DataFrame(rows).to_csv(out_csv(f"n6_sensor_massload_{key}_{args.tag}"), index=False)
            print(f"[n6] sensor {name} {m_g} g ({nodes.size} 점) ({time.time() - t0:.0f}s)", flush=True)
    if args.mic:
        mrows = []
        for name, (face, xyz) in ns.mic_positions(fm).items():
            p = ns.mic_pressure(fm, md, obs_t, xyz, face=face)
            for j, o in enumerate(obs_t):
                mrows.append({"key": key, "mic": name, "face": face, "x_mm": xyz[0] * 1e3, "y_mm": xyz[1] * 1e3,
                              "z_mm": xyz[2] * 1e3, "label": o.label, "h": o.h, "deg": o.deg, "f_Hz": o.f_Hz,
                              "p_Pa_per_um": p[j], "spl_dB_per_um": 20 * math.log10(p[j] / 2e-5) if p[j] > 0 else np.nan})
        pd.DataFrame(mrows).to_csv(out_csv(f"n6_mic_{key}"), index=False)
        print(f"[n6] → {out_csv(f'n6_mic_{key}')}")
    print(f"[n6] → {out_csv(f'n6_sensor_massload_{key}_{args.tag}')}")


def cmd_sensorgrooves(args):
    """센서(집중질량)를 단 채 홈 1차 응답 — 센서는 두 상태에 그대로 있다. 같은 홈의 센서 없는 값과 나란히."""
    ff = _spec(args)
    if ff.clamp_r is not None:
        raise SystemExit("sensorgrooves: 자유-자유만 지원한다")
    fm, km, md, rigid, obs = _healthy(ff, args.n_modes)
    key = _key(ff)
    obs_t = obs[:args.n_obs]
    face, node = ns.placements(fm)[args.placement]
    r_c, th = float(fm.r_v[node]), math.atan2(fm.coors[node, 1], fm.coors[node, 0])
    nodes = ns.footprint(fm, face, r_c, th, ns.footprint_radius_mm(args.mass) * 1e-3)
    mdl, _ = n6.solve(ns.with_mass(km, nodes, args.mass * 1e-3), args.n_modes, n_rigid=ff.n_rigid)
    eta_m, split_m, mac_m, idx = ns.match(md, mdl, obs_t)
    ok = [j for j in range(len(obs_t)) if idx[j]]
    obs_l = [replace(obs_t[j], members=idx[j]) for j in ok]          # 센서 단 구조의 짝 번호
    rows, t0 = [], time.time()
    for xi in args.xi:
        for dep in args.depths:
            e0 = n6.groove_first_order(fm, md, obs_t, xi, dep * 1e-3)
            el = n6.groove_first_order(fm, mdl, obs_l, xi, dep * 1e-3)
            for k, j in enumerate(ok):
                o = obs_t[j]
                rows.append({"key": key, "placement": args.placement, "mass_g": args.mass, "xi": xi,
                             "depth_mm": dep, "label": o.label, "h": o.h, "deg": o.deg, "f_Hz": o.f_Hz,
                             "df_sensor_pct": 100 * (math.sqrt(1 + eta_m[j]) - 1),
                             "split_sensor_pct": 50 * split_m[j] if np.isfinite(split_m[j]) else np.nan,
                             "df_groove_pct_no_sensor": 100 * (math.sqrt(1 + e0[j]) - 1),
                             "df_groove_pct_with_sensor": 100 * (math.sqrt(1 + el[k]) - 1)})
            print(f"[n6] sensorgroove ξ={xi} δ={dep} mm ({time.time() - t0:.0f}s)", flush=True)
    tag = f"{args.placement}_{args.mass:g}g"
    pd.DataFrame(rows).to_csv(out_csv(f"n6_sensor_grooves_{key}_{tag}"), index=False)
    print(f"[n6] → {out_csv(f'n6_sensor_grooves_{key}_{tag}')}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="n6_run")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("modes", cmd_modes), ("grooves", cmd_grooves), ("collar", cmd_collar),
                     ("excite", cmd_excite), ("kernels", cmd_kernels), ("nonlin", cmd_nonlin),
                     ("fig1", cmd_fig1), ("vanethin", cmd_vanethin), ("sensors", cmd_sensors),
                     ("sensorgrooves", cmd_sensorgrooves)):
        sp = sub.add_parser(name)
        sp.add_argument("--n-r", type=int, default=24)
        sp.add_argument("--n-bay", type=int, default=12)
        sp.add_argument("--n-hub", type=int, default=6)
        sp.add_argument("--n-zc", type=int, default=4)
        sp.add_argument("--wrap", type=float, default=geo.ASBUILT.wrap_deg)
        sp.add_argument("--collar-h", type=float, nargs="+", default=[geo.ASBUILT.collar_h * 1e3],
                        help="mm (정본 G3 5.0 = 사진 추정; G1 산출은 4.0)")
        sp.add_argument("--n-modes", type=int, default=80)
        sp.add_argument("--clamp-r", type=float, default=None,
                        help="mm — 주면 허브 r ≤ clamp_r 고정(설치 상태), 없으면 자유-자유")
        if name == "vanethin":
            sp.add_argument("--frac", type=float, nargs="+", default=[0.3, 0.6])
            sp.add_argument("--span-hi", type=float, default=0.25)
        if name == "fig1":
            sp.add_argument("--cut-sector", type=float, nargs=2, default=[-105.0, 5.0],
                            help="앞판을 걷어낼 방위 구간 [deg]")
        if name == "nonlin":
            sp.add_argument("--xi", type=float, nargs="+", default=[0.2, 0.5, 0.8])
            sp.add_argument("--s-bar", type=float, nargs="+", default=[0.05])
            sp.add_argument("--dxi", type=float, default=0.02)
            sp.add_argument("--ds-rel", type=float, default=0.2)
        if name == "kernels":
            sp.add_argument("--xi-lo", type=float, default=0.0)
            sp.add_argument("--xi-hi", type=float, default=1.0)
            sp.add_argument("--xi-step", type=float, default=0.02)
        if name == "grooves":
            sp.add_argument("--xi", type=float, nargs="+", default=[0.3, 0.6, 0.85])
            sp.add_argument("--depths", type=float, nargs="+", default=[0.02, 0.04, 0.06, 0.09],
                            help="mm (앞판 G3 0.40 mm)")
            sp.add_argument("--resolve-depths", type=float, nargs="+", default=[0.06],
                            help="재해석으로도 확인할 깊이(mm)")
        if name in ("sensors", "sensorgrooves"):
            sp.add_argument("--n-obs", type=int, default=20)
        if name == "sensors":
            sp.add_argument("--placements", nargs="+", default=None,
                            help="n6_sensors.placements의 이름(없으면 전부)")
            sp.add_argument("--masses", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0], help="g")
            sp.add_argument("--tag", default="all", help="출력 파일 꼬리표(병렬 실행용)")
            sp.add_argument("--mic", action="store_true", help="근거리 마이크 음압도 낸다")
        if name == "sensorgrooves":
            sp.add_argument("--placement", required=True)
            sp.add_argument("--mass", type=float, required=True, help="g")
            sp.add_argument("--xi", type=float, nargs="+", default=[0.3, 0.6, 0.85])
            sp.add_argument("--depths", type=float, nargs="+", default=[0.06], help="mm")
        if name == "excite":
            sp.add_argument("--n-obs", type=int, default=16)
            sp.add_argument("--n-target", type=int, default=12, help="고정 응답점이 모두 봐야 할 낮은 관측량 수")
            sp.add_argument("--hammer-g", type=float, nargs="+", default=[2.0, 4.0])
            sp.add_argument("--ball-mm", type=float, nargs="+", default=[1.5, 2.0])
        sp.set_defaults(func=fn)
    sp = sub.add_parser("massload")
    sp.add_argument("--key", default=f"ff32x16h6w62c{geo.ASBUILT.collar_h * 1e4:.0f}")
    sp.add_argument("--n-obs", type=int, default=16)
    sp.set_defaults(func=cmd_massload)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
