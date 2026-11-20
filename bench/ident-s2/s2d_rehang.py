"""2번 시편 다시 매달기 반복성(10-30 19:07–19:30, 1번 깃 ξ0.3, 뒤판) — 타격별 극점(s2b_track과 같은 방법).
비교: 10-28 1번 깃 ξ0.3(17:08–17:11), 10-30 닫음 점검(16:21–16:22, 다시 매달기 전), 10-30 S3·S4·S5(다시 매단 뒤 0·10·21분).
20 N 환산은 s2b_track의 연화 기울기, 새 모드는 기울기 0(영향 ±0.01 % 안)."""
import sys, glob, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from specfit import spectral_fit

SRC = "/mnt/nas/data/GY_DATA/impact_testw2s2/impact_testw2s2"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
fs = 204800.0; ti = 4096
MODES = [("2134", (2110, 2165), 2, 0), ("2142", (2110, 2165), 2, 1), ("3164", (3148, 3182), 1, 0), ("3250", (3232, 3268), 1, 0),
         ("3459", (3440, 3478), 1, 0), ("3589", (3572, 3606), 1, 0), ("3639", (3620, 3656), 1, 0), ("3748", (3730, 3766), 1, 0),
         ("3804", (3780, 3825), 1, 0), ("4225", (4180, 4265), 1, 0), ("4967", (4945, 4990), 1, 0), ("6439", (6415, 6462), 1, 0),
         ("7462", (7420, 7500), 1, 0), ("8321", (8285, 8355), 1, 0)]
SL = {"2134": -0.00059, "2142": -0.00059, "3459": -0.00032, "3639": -0.00051, "3804": -0.00036, "4225": -0.00050, "7462": -0.00071, "8321": -0.00046}
sec = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8])
GROUPS = [("10-28 17:08", "20261028_*_vane1_xi0.3_*.npz"), ("10-30 16:21 전", "20261030_162*_S2_back_vane1_xi0.3_*.npz"),
          ("10-30 19:07 직후", "20261030_*_S3_back_vane1_xi0.3_*.npz"), ("10-30 19:17 +10분", "20261030_*_S4_back_vane1_xi0.3_*.npz"),
          ("10-30 19:28 +21분", "20261030_*_S5_back_vane1_xi0.3_*.npz"), ("10-30 19:38 +31분", "20261030_*_S6_back_vane1_xi0.3_*.npz")]
JS = OUT + "s2d_rehang.json"
cache = {r["file"]: r for r in json.load(open(JS))} if os.path.exists(JS) else {}
rows = []
for g, pat in GROUPS:
    for p in sorted(glob.glob(os.path.join(SRC, pat))):
        b = os.path.basename(p)
        if b in cache:
            rows.append(cache[b]); continue
        with np.load(p, allow_pickle=False) as z:
            assert str(z["channel_names"][1]) == "CH6_mic" and str(z["physical_channels"][1]) == "PXI1Slot5/ai2", p
            d = z["data"].astype(float); N = float(z["peak_force_N"]); ts = str(z["timestamp"])[11:19]; T = z["temperature_C"].astype(float)
            hit = int(z["hit"])
        f = d[0] - d[0, :ti - 100].mean(); m = d[1] - d[1, :ti - 100].mean()
        seg = f[ti + int(0.003 * fs): ti + int(0.2 * fs)]
        o = np.where(seg > max(0.5, 0.03 * N))[0]
        late = bool(np.any(seg > max(1.0, 0.05 * N)))
        t0 = max(0.005, (o[-1] + int(0.003 * fs)) / fs + 0.003) if o.size else 0.005
        r = dict(file=b, g=g, hit=hit, N=N, ts=ts, t=sec(ts), t0=t0, late=late, T0=float(T[0]), T1=float(T[1]),
                 pre=float(m[:ti - 100].std()), f={})
        if N >= 5:
            for name, (lo, hi), k, j in MODES:
                fk, zz, amp, rr, e = spectral_fit(m, fs, ti, lo, hi, k=k, t0=t0)
                ok = e < 0.3 and np.all((fk > lo) & (fk < hi)) and (k == 1 or fk[1] - fk[0] > 3)
                r["f"][name] = [float(fk[j]), float(zz[j]), float(e)] if ok else None
        rows.append(r)
json.dump(rows, open(JS, "w"))
c20 = lambda r, name: r["f"][name][0] * (1 - SL.get(name, 0.0) * np.log2(r["N"] / 20))
lines = []
P = lambda *a: lines.append(" ".join(str(x) for x in a))
P("무리: 타격 수 / 5 N 이상 / 늦은 재타격 / 힘 중앙 [N] / 온도 ai0·ai1 [°C] / 타격 전 잡음 중앙 [mPa]")
gs = [g for g, _ in GROUPS if any(r["g"] == g for r in rows)]
for g in gs:
    R = [r for r in rows if r["g"] == g]
    P(f"  {g:16s} {len(R):2d} / {sum(r['N'] >= 5 for r in R):2d} / {sum(r['late'] for r in R):2d} / {np.median([r['N'] for r in R]):5.1f} / "
      f"{np.mean([r['T0'] for r in R]):.2f}·{np.mean([r['T1'] for r in R]):.2f} / {np.median([r['pre'] for r in R]) * 1e3:.0f}")
P("")
P("20 N 환산 주파수: 무리 중앙 [Hz] (n, 무리 안 sd %) · 다시 매달기 전(16:21) 대비 % [±중앙값 표준오차]")
summary = {}
ref = "10-30 16:21 전"
for name, *_ in MODES:
    vals = {}
    for g in gs:
        v = np.array([c20(r, name) for r in rows if r["g"] == g and r["f"].get(name)])
        if v.size >= 2:
            md = np.median(v); res = np.abs(v / md - 1); keep = res < max(3 * 1.4826 * np.median(res), 2e-4)
            v = v[keep]
            vals[g] = (np.median(v), v.size, 100 * np.std(v, ddof=1) / np.median(v) if v.size > 2 else np.nan, v)
    if ref not in vals:
        continue
    m0, n0, s0, v0 = vals[ref]
    se0 = 1.2533 * np.std(v0, ddof=1) / np.sqrt(n0) if n0 > 2 else np.nan
    line = f"  {name}: "
    out = {}
    for g in gs:
        if g not in vals:
            line += f"| {g[6:]:>10s}    -   "; continue
        m1, n1, s1, v1 = vals[g]
        se1 = 1.2533 * np.std(v1, ddof=1) / np.sqrt(n1) if n1 > 2 else np.nan
        dp = 100 * (m1 / m0 - 1); sep = 100 * np.hypot(se0, se1) / m0
        out[g] = dict(med=m1, n=int(n1), sd=s1, d_pct=dp, se_pct=sep)
        line += f"| {m1:8.2f}({n1:2d},{s1:.3f}) {dp:+.3f}±{sep:.3f} "
    summary[name] = out
    P(line)
json.dump(summary, open(OUT + "s2d_rehang_summary.json", "w"), ensure_ascii=False, indent=1)
# 다시 매단 뒤 시간 경향: S3–S5(–S6) 타격별 값의 시간 기울기
P("")
P("다시 매단 뒤 시간 기울기(S3 이후 타격, 20 N 환산, 무리 안 이상치 제외): %/10분 [n]")
post = [g for g in gs if g.startswith("10-30 19")]
for name, *_ in MODES:
    t, y = [], []
    for g in post:
        if name in summary and g in summary[name]:
            med = summary[name][g]["med"]
            for r in rows:
                if r["g"] == g and r["f"].get(name):
                    v = c20(r, name)
                    if abs(v / med - 1) < 0.002:
                        t.append(r["t"]); y.append(v)
    if len(t) >= 6:
        t = np.array(t, float); y = np.array(y)
        A = np.column_stack([(t - t.mean()) / 600, np.ones_like(t)])
        cf, res, *_ = np.linalg.lstsq(A, y / np.median(y) - 1, rcond=None)
        e = y / np.median(y) - 1 - A @ cf
        se = np.sqrt(e @ e / (len(t) - 2) / ((A[:, 0] - A[:, 0].mean()) ** 2).sum())
        P(f"  {name}: {cf[0] * 100:+.3f} ± {se * 100:.3f} [{len(t)}]")
txt = "\n".join(lines)
print(txt)
open(OUT + "s2d_rehang.txt", "w").write(txt + "\n")
