"""impact_testw22 깃 6곳 공동 적합: 대역·극점 수 후보를 모두 돌려 표로 낸다."""
import sys, os
sys.path.insert(0, sys.argv[1])
import numpy as np
from w22joint import vane_avg, joint_fit, pattern

z = np.load(os.path.join(sys.argv[2], "w22_spec.npz"), allow_pickle=False)
fr = z["fr"]; F = z["F"].astype(complex); X = z["X"].astype(complex); V = z["vane"]; N = z["N"]
slope = float(os.environ.get("SLOPE", "0"))
H = vane_avg(F, X, V, N, slope, fr); Hs = [H[v] for v in range(1, 7)]
bands = [((2020, 2090), (2, 3)), ((3040, 3080), (1, 2)), ((3140, 3180), (1, 2)), ((3225, 3295), (2, 3, 4)), ((3375, 3415), (1, 2)),
         ((3475, 3510), (1, 2)), ((3615, 3650), (1, 2)), ((3760, 3800), (1, 2)), ((3885, 3925), (1, 2)), ((4255, 4295), (1, 2)),
         ((4300, 4365), (2, 3)), ((5090, 5135), (1, 2)), ((5230, 5270), (1, 2)), ((6055, 6115), (1, 2)), ((6120, 6170), (1, 2)),
         ((6420, 6460), (1, 2)), ((6675, 6730), (1, 2)), ((6830, 6880), (1, 2)), ((7075, 7125), (1, 2)), ((7265, 7315), (1, 2)),
         ((7805, 7855), (1, 2)), ((9340, 9385), (1, 2)), ((9530, 9590), (1, 2)), ((9845, 9910), (1, 2)), ((10190, 10260), (1, 2))]
if len(sys.argv) > 3:
    lo, hi = map(float, sys.argv[3].split("-")); bands = [((lo, hi), tuple(int(c) for c in sys.argv[4].split(",")))]
print(f"slope {slope}; columns: pole f [Hz] (zeta %) | MPC | h0 h1 h2 h3 energy | theta0 h1/h2 [deg] | |r| per vane dB re max | signed pattern a (re max)")
for (lo, hi), ks in bands:
    for k in ks:
        fk, zk, R, err = joint_fit(fr, Hs, lo, hi, k)
        print(f"== {lo:.0f}-{hi:.0f} Hz k={k}  fit err {err:.3f}")
        rmax = np.abs(R).max()
        for j in range(k):
            mpc, a, P, th0, alpha = pattern(R[:, j])
            dbs = " ".join(f"{20*np.log10(abs(x)/rmax):6.1f}" for x in R[:, j])
            pat = " ".join(f"{x/np.abs(a).max():+5.2f}" for x in a)
            hbest = int(np.argmax(P))
            print(f"   {fk[j]:8.2f} ({zk[j]*100:.2f}) | MPC {mpc:.2f} | {P[0]:.2f} {P[1]:.2f} {P[2]:.2f} {P[3]:.2f} -> h{hbest} | {th0[1]:5.1f}/{th0[2]:5.1f} | {dbs} | {pat}")
