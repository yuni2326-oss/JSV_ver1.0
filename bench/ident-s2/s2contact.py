"""마이크–판 접촉 점검: (1) 봉우리 감쇠(깃 평균 H1 공동 적합) 새 시리즈 대 10-26, (2) 마이크 저주파(1–500 Hz) 내용과 타격 뒤 느린 변화,
(3) 같은 날 16:34–16:52(마이크 멀리) 1번 깃 ξ0.3·ξ0.6과 17:08 이후 비교."""
import sys, os, glob
sys.path.insert(0, sys.argv[1])
import numpy as np
from w22joint import joint_fit, vane_avg

OUT = sys.argv[2]
Z = np.load(os.path.join(OUT, "s2", "s2_spec.npz"), allow_pickle=False)
O = np.load(os.path.join(OUT, "w22_spec.npz"), allow_pickle=False)
fr = Z["fr"]
use = ~Z["late"] & (Z["N"] >= 5)
Hn = vane_avg(Z["F"][use].astype(complex), Z["X"][use].astype(complex), Z["vane"][use])
Ho = vane_avg(O["F"].astype(complex), O["X"].astype(complex), O["vane"])
print("(1) joint-fit poles (6 vanes, k=1 or 2): new series f [Hz] zeta % | 10-26 nearest f zeta %")
for (lo, hi), k in (((2100, 2170), 2), ((3200, 3300), 1), ((3440, 3480), 1), ((3620, 3660), 1), ((3780, 3830), 1), ((4150, 4300), 1), ((4930, 5010), 1),
                    ((5700, 5800), 1), ((7400, 7510), 1), ((8270, 8360), 1)):
    f, z, R, e = joint_fit(fr, [Hn[v] for v in range(1, 7)], lo, hi, k)
    print(f"  {lo}-{hi}: " + ", ".join(f"{a:8.2f} ζ {b*100:.2f}" for a, b in zip(f, z)) + f"  (err {e:.2f})")
for (lo, hi), k in (((2020, 2090), 2), ((3225, 3295), 1), ((4255, 4295), 1), ((7265, 7315), 1)):
    f, z, R, e = joint_fit(fr, [Ho[v] for v in range(1, 7)], lo, hi, k)
    print(f"  10-26 {lo}-{hi}: " + ", ".join(f"{a:8.2f} ζ {b*100:.2f}" for a, b in zip(f, z)))

# (2) 저주파: 4 s 기록 전체의 마이크 스펙트럼 1–500 Hz, 그리고 타격 뒤 0–2 s 저역(<200 Hz) 성분 크기
fs = 204800.0; ti = 4096


def lowband(paths, lab):
    P = []; slow = []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            m = z["data"][1].astype(float); N = float(z["peak_force_N"])
        m = m - m[:ti - 100].mean()
        seg = m[ti: ti + int(2.0 * fs)]
        # 1–500 Hz 스펙트럼(2 s, Hann)
        w = np.hanning(seg.size); X = np.fft.rfft(seg * w); f = np.fft.rfftfreq(seg.size, 1 / fs)
        P.append(np.abs(X[(f >= 1) & (f <= 500)]) / N)
        # 200 Hz 저역 통과(이동 평균 2.5 ms) 뒤 0–2 s 최대
        k = int(0.0025 * fs); lp = np.convolve(seg, np.ones(k) / k, mode="valid")
        slow.append(np.abs(lp).max() / N)
    f = np.fft.rfftfreq(int(2.0 * fs), 1 / fs); f = f[(f >= 1) & (f <= 500)]
    Pm = np.median(np.array(P), 0)
    pk = np.argsort(Pm)[::-1][:6]
    print(f"  {lab}: n {len(paths)}, slow (<200 Hz) peak / N median {np.median(slow):.4f} Pa/N; strongest 1-500 Hz bins: " + ", ".join(f"{f[i]:.1f} Hz" for i in sorted(pk)))
    return f, Pm


print("\n(2) low-frequency microphone content per newton")
import json
res = {}
for lab, pat in (("10-26 w22 vane1", "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22/20261026_200*vane1*.npz"),
                 ("10-28 16:34 vane1 xi0.3 (far mic)", "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22/20261028_163*xi0.3*.npz"),
                 ("10-28 17:08 vane1 xi0.3 (near mic)", "/mnt/nas/data/GY_DATA/impact_testw2s2/impact_testw2s2/20261028_170*vane1*.npz"),
                 ("10-28 17:3x vane6 xi0.3 (near mic)", "/mnt/nas/data/GY_DATA/impact_testw2s2/impact_testw2s2/20261028_174*vane6*.npz")):
    ps = sorted(glob.glob(pat))[:12]
    res[lab] = lowband(ps, lab)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
for lab, (f, Pm) in res.items():
    ax.semilogy(f, Pm, lw=0.8, label=lab)
ax.set_xlabel("Hz"); ax.set_ylabel("|X|/N (2 s, Hann)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, "s2", "s2_lowfreq.png"), dpi=110)
