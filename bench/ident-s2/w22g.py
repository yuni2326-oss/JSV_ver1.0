"""10-28 ξ0.6(테 근처)에서 재접촉(0.1–0.2 ms 이중 피크) 타격을 써도 되는지: 힘 스펙트럼 홈 깊이, 타격별 H1 일관성(LOO FRAC), 코히어런스.
비교: 같은 날 ξ0.3 1번 깃. 늦은 재타격(3 ms 뒤 max(1 N, 5 %))은 뺀다."""
import sys, glob
import numpy as np
fs = 204800.0; ti = 4096
SRC = "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22"
n = int(0.3 * fs); fr = np.fft.rfftfreq(n, 1 / fs); ph = np.exp(2j * np.pi * fr * 40 / fs)
wf = np.zeros(n); wf[:40 + int(0.00045 * fs)] = 1
wr = np.zeros(n); nr = int(0.25 * fs); wr[:nr] = 1; nt = int(0.01 * fs); wr[nr - nt:nr] = 0.5 * (1 + np.cos(np.pi * np.arange(nt) / nt))
band = (fr >= 1900) & (fr <= 7500)


def rows(pat):
    out = []
    for p in sorted(glob.glob(SRC + "/" + pat)):
        with np.load(p, allow_pickle=False) as z:
            d = z["data"].astype(float); N = float(z["peak_force_N"]); q = str(z["quality"]); kind = str(z["double_kind"]); hit = int(z["hit"])
            loc = str(z["location"]); pw = float(z["pulse_width_ms"]); spN = float(z["second_peak_N"]); spms = float(z["second_peak_ms"])
        fo = d[0] - d[0, :ti - 100].mean(); m = d[1] - d[1, :ti - 100].mean()
        seg = fo[ti + int(0.003 * fs): ti + int(0.2 * fs)]; late = np.where(seg > max(1.0, 0.05 * N))[0]
        F = np.fft.rfft(fo[ti - 40: ti - 40 + n] * wf) * ph; X = np.fft.rfft(m[ti - 40: ti - 40 + n] * wr) * ph
        Fd = 20 * np.log10(np.abs(F) / np.abs(F[(fr > 100) & (fr < 500)]).mean())
        out.append(dict(loc=loc, hit=hit, N=N, q=q, kind=kind, late=late.size > 0, pw=pw, sp=spN / N, spms=spms, F=F, X=X,
                        notch=Fd[band].min(), f10=fr[np.argmax(Fd < -10)] if (Fd < -10).any() else np.nan))
    return out


def avg(rs):
    Sfx = sum(np.conj(r["F"]) * r["X"] for r in rs); Sff = sum(np.abs(r["F"]) ** 2 for r in rs); Sxx = sum(np.abs(r["X"]) ** 2 for r in rs)
    return Sfx / Sff, np.abs(Sfx) ** 2 / (Sff * Sxx)


def frac(a, b):
    return np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


for pat in sys.argv[1:]:
    R = rows(pat)
    G = [r for r in R if r["N"] >= 10 and not r["late"]]
    print(f"\n== {pat}: {len(R)} hits; >=10 N without late re-hit: {len(G)} (accepted {sum(r['q']=='ok' for r in G)}, recontact {sum('recontact' in r['kind'] for r in G)})")
    print("   hit    N   quality    kind             pulse ms  2nd/1st @ms   force -10 dB at  min |F| 1.9-7.5 kHz  LOO FRAC 1.9-7.5")
    for i, r in enumerate(G):
        H, _ = avg([q for j, q in enumerate(G) if j != i])
        r["loo"] = frac((r["X"] / r["F"])[band], H[band])
        print(f"   H{r['hit']:03d} {r['N']:5.1f} {r['q']:10s} {r['kind'] or '-':15s} {r['pw']:.3f}   {r['sp']:.2f} @{r['spms']:.2f}   {r['f10']:7.0f} Hz      {r['notch']:6.1f} dB          {r['loo']:.2f}")
    for lab, sub in (("accepted only", [r for r in G if r["q"] == "ok"]), ("recontact only", [r for r in G if "recontact" in r["kind"]]), ("all (no late)", G)):
        if len(sub) < 2:
            print(f"   {lab}: n={len(sub)}"); continue
        H, C = avg(sub)
        print(f"   {lab:15s} n={len(sub):2d} coherence 1.9-3 / 3-5 / 5-7.5 kHz: " + " / ".join(f"{C[(fr >= lo) & (fr < hi)].mean():.2f}" for lo, hi in ((1900, 3000), (3000, 5000), (5000, 7500))))
    A = [r for r in G if r["q"] == "ok"]; B = [r for r in G if "recontact" in r["kind"]]
    if len(A) >= 1 and len(B) >= 2:
        print(f"   FRAC 1.9-7.5 kHz mean(accepted) vs mean(recontact): {frac(avg(A)[0][band], avg(B)[0][band]):.2f}")
