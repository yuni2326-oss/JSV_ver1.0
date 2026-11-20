"""반지름 모양 비교: 측정 극점마다 원주 조화 계수 c_h(ξ) = (1/6) Σ_k r_k e^{−ih·60°k}(깃 번호 각도, ξ마다)를 내고
ρ = c_h(ξ0.6)/c_h(ξ0.3)(크기 dB, 위상 °)를 FE 같은 차수 모드의 ρ와 비교한다. 위상에는 깃 휨(ξ0.3→0.6에서 깃 선이 18.4° 돈다)과 모드 비틀림이 함께 든다.
h0·h3은 위상 0/180°(같은/반대 부호)만 뜻이 있다. 거울(깃 번호가 FE −θ 방향)이면 측정 c_h 대신 c_−h를 쓴다.
측정 ξ0.3 = 10-28 잔차 × D(1번 깃 닫음 점검, SNR < 5면 1), ξ0.6 = 10-30 잔차. 부트스트랩(40)으로 ρ의 흩어짐도 낸다."""
import sys, os, json
import numpy as np, pandas as pd

OUT, FED = sys.argv[1], sys.argv[2]
key = "ff32x16h6w62c40"
reg = pd.read_csv(os.path.join(FED, f"rev1_n6_modes_{key}.csv")).sort_values("f_Hz").reset_index(drop=True)
Phi = np.load(os.path.join(OUT, "s2", "s2b_fe_phi.npy"))
members = [[int(i) for i in str(m).split("|")] for m in reg["members"]]
J = json.load(open(os.path.join(OUT, "s2", "s2b_joint.json")))
k6 = np.arange(6)


def coef(v, h, sgn=1):
    return (v * np.exp(-1j * sgn * h * np.radians(60.0) * k6)).mean()


def hshare(v):
    A = np.array([abs(coef(v, h)) ** 2 for h in range(6)])
    P = np.array([A[0], A[1] + A[5], A[2] + A[4], A[3]]); return P / P.sum()


def rho(v3, v6, h, sgn=1):
    return coef(v6, h, sgn) / coef(v3, h, sgn)


# FE: 차수별 ρ(짝은 두 멤버의 c_h 비가 같아야 한다 → 멤버 1, 2 둘 다 내서 확인)
FE = []
for g, mem in enumerate(members):
    h = int(reg["h"][g]); r = [rho(Phi[m, :6], Phi[m, 6:], h) for m in mem]
    s3 = np.mean([np.abs(coef(Phi[m, :6], h)) ** 2 for m in mem]); s6 = np.mean([np.abs(coef(Phi[m, 6:], h)) ** 2 for m in mem])
    FE.append(dict(label=reg["label"][g], f=float(reg["f_Hz"][g]), h=h, rho=r))
print("FE rho = c_h(xi0.6)/c_h(xi0.3), nominal vane angles; members listed separately for pairs")
for e in FE[:24]:
    print(f"  {e['label']:10s} {e['f']:8.1f} h{e['h']} " + " | ".join(f"{20*np.log10(abs(x)):+6.1f} dB {np.degrees(np.angle(x)):+6.1f}" for x in e["rho"]))
print("\nmeasured: pole | h share xi0.3(10-28) / xi0.6(10-30) | h used | rho same-dir dB deg [boot 90% deg half-width] | rho mirror | FE same-h candidates within -15..+25 %: rho")
dB = lambda z: 20 * np.log10(abs(z)); deg = lambda z: np.degrees(np.angle(z))
for A, B in zip(J["A"], J["B"]):
    RA = np.array([complex(*x) for x in A["R"]]); RB = np.array([complex(*x) for x in B["R"]])
    D = RA[6] / RB[0]; sn = min(abs(RA[6]) / A["Rsd"][6], abs(RB[0]) / B["Rsd"][0]); Du = D if sn >= 5 and abs(RB[0]) > 10 ** (-0.5) * np.abs(RB[:6]).max() else 1.0
    v3 = RB[:6] * Du; v6 = RA[:6]
    P3, P6 = hshare(v3), hshare(v6); h = int(np.argmax(P3 + P6))
    rs, rm = rho(v3, v6, h), rho(v3, v6, h, -1)
    # 부트스트랩: 두 날 잔차를 각자 다시 뽑는다(D는 고정)
    Ab = np.array([[complex(*x) for x in row] for row in A["Rb"]]); Bb = np.array([[complex(*x) for x in row] for row in B["Rb"]])
    rb = np.array([rho(Bb[i, :6] * Du, Ab[i, :6], h) for i in range(min(len(Ab), len(Bb)))])
    hw = np.percentile(np.abs(np.degrees(np.angle(rb / rs))), 90); dbw = np.percentile(np.abs(20 * np.log10(np.abs(rb / rs))), 90)
    cand = [e for e in FE if e["h"] == h and 0.85 * A["f"] < e["f"] < 1.25 * A["f"]]
    cs = " ; ".join(f"{e['label']}({e['f']:.0f}) {dB(e['rho'][0]):+5.1f} {deg(e['rho'][0]):+5.0f}" for e in cand)
    print(f" {A['f']:8.2f} | " + " ".join(f"{x:.2f}" for x in P3) + " / " + " ".join(f"{x:.2f}" for x in P6) +
          f" | h{h} | {dB(rs):+5.1f} {deg(rs):+5.0f} [±{dbw:.1f} dB, ±{hw:.0f}°] | {dB(rm):+5.1f} {deg(rm):+5.0f} | D {'used' if Du is D else '=1'} | {cs}")
