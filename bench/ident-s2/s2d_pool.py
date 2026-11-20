"""다시 매달기 반복성 — 무리별 '타격 묶음 적합': 공통 극점 + 타격별 잔차(VarPro), 타격마다 마지막 접촉 3 ms 뒤 자유 감쇠
(0.4 s, 2^20 영 채움), 주파수 축을 20 N 기준으로 늘임. 부트스트랩(타격 재추출)으로 표준오차.
그리고 무리 평균 FRF(늦은 재타격 없는 타격)의 모양 비교(FRAC, 최적 주파수 이동)와 피크 높이 차."""
import sys, glob, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import optimize, signal

SRC = "/mnt/nas/data/GY_DATA/impact_testw2s2/impact_testw2s2"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
fs = 204800.0; ti = 4096; NFFT = 1 << 20; NB = int(os.environ.get("NBOOT", "100"))
fr_all = np.fft.rfftfreq(NFFT, 1 / fs)
BANDS = [("2134/2142", (2105, 2170), 2), ("3164", (3140, 3190), 1), ("3250", (3200, 3300), 1), ("3459", (3440, 3485), 1),
         ("3589", (3570, 3615), 1), ("3639", (3620, 3665), 1), ("3748", (3730, 3770), 1), ("3804", (3780, 3835), 1),
         ("4225", (4150, 4300), 1), ("4967", (4930, 5020), 1), ("6439/6558", (6380, 6590), 2), ("7462", (7400, 7520), 1), ("8321", (8270, 8370), 1)]
SL = {"2134/2142": -0.00059, "3459": -0.00032, "3639": -0.00051, "3804": -0.00036, "4225": -0.00050, "7462": -0.00071, "8321": -0.00046}
rows = json.load(open(OUT + "s2d_rehang.json"))
GROUPS = list(dict.fromkeys(r["g"] for r in rows))
REF = "10-30 16:21 전"

# 타격별 자유 감쇠 스펙트럼(대역마다 필요한 구간만)
spec = {}
for r in rows:
    if r["N"] < 5:
        continue
    with np.load(os.path.join(SRC, r["file"]), allow_pickle=False) as z:
        d = z["data"].astype(float)
    m = d[1] - d[1, :ti - 100].mean()
    seg = m[ti + int(r["t0"] * fs): ti + int((r["t0"] + 0.4) * fs)]
    spec[r["file"]] = np.fft.rfft(seg, NFFT) / fs


def stretched(r, lo, hi, slope, pad=8.0):
    g = np.arange(lo - pad, hi + pad, fr_all[1])
    sc = 1 + slope * np.log2(r["N"] / 20.0)
    X = spec[r["file"]]
    sel = (fr_all >= (lo - pad) * sc - 2) & (fr_all <= (hi + pad) * sc + 2)
    x, y = fr_all[sel], X[sel]
    return g, np.interp(g * sc, x, y.real) + 1j * np.interp(g * sc, x, y.imag)


def jfit(f, Ys, lo, hi, k, p0=None, zmax=0.02):
    w = 2j * np.pi * f; fc = 0.5 * (lo + hi)
    norm = max(np.abs(Y).max() for Y in Ys)
    def basis(p):
        fk, zk = p[:k], p[k:]
        sk = -zk * 2 * np.pi * fk + 1j * 2 * np.pi * fk * np.sqrt(1 - zk ** 2)
        return np.column_stack([1 / (w - s) for s in sk] + [np.ones_like(w), f - fc])
    Ym = np.column_stack(Ys)                                   # 타격들이 기저를 공유 → 한 번의 lstsq
    def resid(p):
        B = basis(p)
        c, *_ = np.linalg.lstsq(B, Ym, rcond=None); dd = ((B @ c - Ym) / norm).T.ravel()
        return np.concatenate([dd.real, dd.imag])
    lb = np.concatenate([np.full(k, lo), np.full(k, 2e-5)]); ub = np.concatenate([np.full(k, hi), np.full(k, zmax)])
    if p0 is not None:
        cands = [p0]
    else:
        mag = sum(np.abs(Y) ** 2 for Y in Ys); pk = signal.find_peaks(mag)[0]; pk = pk[(f[pk] >= lo) & (f[pk] <= hi)]
        pk = pk[np.argsort(mag[pk])[::-1]]
        rng = np.random.default_rng(0)
        f0s = ([np.sort(f[pk[:k]])] if pk.size >= k else []) + [np.sort(rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), k)) for _ in range(10)]
        cands = [np.concatenate([f0, np.full(k, 0.0015)]) for f0 in f0s]
    best = None
    for c in cands:
        sol = optimize.least_squares(resid, np.clip(c, lb + 1e-6, ub - 1e-6), bounds=(lb, ub), x_scale="jac", max_nfev=3000)
        if best is None or sol.cost < best.cost:
            best = sol
    o = np.argsort(best.x[:k])
    err = np.sqrt(2 * best.cost) / np.sqrt(sum((np.abs(Y / norm) ** 2).sum() for Y in Ys))
    return np.concatenate([best.x[:k][o], best.x[k:][o]]), err


res = {}
rng = np.random.default_rng(11)
for name, (lo, hi), k in BANDS:
    slope = SL.get(name, -0.0005)
    res[name] = {}
    for g in GROUPS:
        R = [r for r in rows if r["g"] == g and r["N"] >= 5]
        if len(R) < 3:
            continue
        f, _ = stretched(R[0], lo, hi, slope)
        Ys = [stretched(r, lo, hi, slope)[1] for r in R]
        p, err = jfit(f, Ys, lo, hi, k)
        bs = []
        for b in range(NB):
            s = rng.choice(len(Ys), len(Ys), replace=True)
            pb, _ = jfit(f, [Ys[i] for i in s], lo, hi, k, p0=p)
            bs.append(pb[:k])
        bs = np.array(bs)
        res[name][g] = dict(f=p[:k].tolist(), z=p[k:].tolist(), err=float(err), n=len(Ys), sd=bs.std(0, ddof=1).tolist())
json.dump(res, open(OUT + "s2d_pool.json", "w"), ensure_ascii=False, indent=1)

lines = ["무리별 타격 묶음 적합(20 N 기준) — 극점 [Hz] (±부트스트랩 sd), 다시 매달기 전(10-30 16:21) 대비 %; 적합 오차"]
for name, (lo, hi), k in BANDS:
    for j in range(k):
        ref = res[name].get(REF)
        line = f"  {name.split('/')[j] if '/' in name else name:>5s}: "
        for g in GROUPS:
            if g not in res[name]:
                continue
            q = res[name][g]; fj, sj = q["f"][j], q["sd"][j]
            dp = 100 * (fj / ref["f"][j] - 1) if ref else np.nan
            sp = 100 * np.hypot(sj, ref["sd"][j]) / ref["f"][j] if ref else np.nan
            line += f"| {fj:8.2f}±{sj:.2f} {dp:+.3f}±{sp:.3f} e{q['err']:.2f} "
        lines.append(line)
lines.append("  무리 순서: " + " | ".join(f"{g} (n {res[BANDS[0][0]][g]['n']})" for g in GROUPS if g in res[BANDS[0][0]]))

# FRF 모양: 늦은 재타격 없는 5 N 이상 타격의 H1, 20 N 기준
Zo = np.load(OUT + "s2_spec.npz", allow_pickle=False); Zb = np.load(OUT + "s2b_spec.npz", allow_pickle=False); Zd = np.load(OUT + "s2d_spec.npz", allow_pickle=False)
fr = Zd["fr"]
def h1(Z, msk, slope=-0.0005):
    u = np.where(msk & ~Z["late"] & (Z["N"] >= 5))[0]
    Sfx = 0; Sff = 0
    for i in u:
        sc = 1 + slope * np.log2(Z["N"][i] / 20.0); fi = fr * sc
        c = np.conj(Z["F"][i].astype(complex)) * Z["X"][i].astype(complex)
        Sfx = Sfx + np.interp(fi, fr, c.real) + 1j * np.interp(fi, fr, c.imag); Sff = Sff + np.interp(fi, fr, np.abs(Z["F"][i]) ** 2)
    return Sfx / Sff, len(u)
H = {"10-28 17:08": h1(Zo, Zo["vane"] == 1), REF: h1(Zb, (Zb["vane"] == 1) & (Zb["xi"] == 0.3)),
     "10-30 19:07 직후": h1(Zd, Zd["ses"] == 3), "10-30 19:17 +10분": h1(Zd, Zd["ses"] == 4), "10-30 19:28 +21분": h1(Zd, Zd["ses"] == 5)}
def frac(a, b):
    return np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
def best_shift(a, b, sel, shifts=np.linspace(-0.004, 0.004, 161)):
    out = []
    for s in shifts:
        bb = np.interp(fr * (1 + s), fr, b.real) + 1j * np.interp(fr * (1 + s), fr, b.imag)
        out.append(frac(a[sel], bb[sel]))
    j = int(np.argmax(out)); return shifts[j] * 100, out[j], frac(a[sel], b[sel])
lines.append("")
lines.append("FRF 모양(늦은 재타격 없는 타격 평균): FRAC 그대로 → 최적 이동 [%] 뒤 FRAC, 다시 매달기 전 기준. 3–5 kHz / 1.9–9 kHz")
for g, (Hg, n) in H.items():
    s1 = (fr >= 3000) & (fr <= 5000); s2 = (fr >= 1900) & (fr <= 9000)
    a = best_shift(H[REF][0], Hg, s1); b = best_shift(H[REF][0], Hg, s2)
    lines.append(f"  {g:16s} (n {n:2d}): 3–5 kHz {a[2]:.3f} → {a[1]:.3f} @ {a[0]:+.3f} %  |  1.9–9 kHz {b[2]:.3f} → {b[1]:.3f} @ {b[0]:+.3f} %")
# 피크 높이(dB, 전 대비): 주요 극점 근처 |H| 최대
lines.append("")
lines.append("피크 높이 |H| [dB, 다시 매달기 전 대비]")
pk = [2134, 3164, 3250, 3459, 3589, 3639, 3748, 3804, 4225, 4967, 6439, 7462, 8321]
hdr = "  " + " " * 18 + "".join(f"{p:>7d}" for p in pk); lines.append(hdr)
def peak(Hg, p):
    s = (fr >= p * 0.996) & (fr <= p * 1.004); return 20 * np.log10(np.abs(Hg[s]).max())
for g, (Hg, n) in H.items():
    lines.append(f"  {g:16s}  " + "".join(f"{peak(Hg, p) - peak(H[REF][0], p):+7.1f}" for p in pk))
txt = "\n".join(lines)
print(txt)
open(OUT + "s2d_pool.txt", "w").write(txt + "\n")
np.savez(OUT + "s2d_frf.npz", fr=fr, **{f"H{i}": H[g][0] for i, g in enumerate(H)}, names=np.array(list(H)))
