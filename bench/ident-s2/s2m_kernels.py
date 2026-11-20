"""깃 높이 테이퍼(s2m_taper.py k6·k1) 대 G3 정본(간격 3.00 일정) — 원고 조립체 커널의 부호반전 반경 ξ*·계열 주파수·커널 크기.
기준: _generated_g3의 설치 상태 rev1_n6_kernels_ff24x12h6w62c50k92, 레일 지지 rev1_n1_kernels_fit24x12v1z2c2w62.
k6_chk300(테이퍼 0, ξ 0.2 간격)은 감싼 조립·깊이 식이 정본 값을 그대로 내는지 검산한다."""
import os, json
import numpy as np
import pandas as pd

G3 = "/home/super/project/docs/_generated_g3/data/paper3/"
D_K = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/taper/"
CASES = [("설치 상태(허브 고정 r ≤ 9.21 mm)", "n6", "ff24x12h6w62c50k92", "k6_taper319_24x12"),
         ("레일 지지(두 판 r = a 고정)", "n1", "fit24x12v1z2c2w62", "k1_taper319_24x12")]


def zc(x, y):
    return [float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])) for i in range(len(x) - 1) if y[i] * y[i + 1] < 0]


lines, out = [], {}
if os.path.exists(D_K + "kernels_k6_chk300_24x12.csv"):
    a = pd.read_csv(G3 + "rev1_n6_kernels_ff24x12h6w62c50k92.csv")
    b = pd.read_csv(D_K + "kernels_k6_chk300_24x12.csv")
    m = b.merge(a[["xi", "label", "k", "dk"]], on=["xi", "label"], suffixes=("", "_g3"))
    rel = np.abs(m.k - m.k_g3).max() / np.abs(m.k_g3).max()
    rel_d = np.abs(m.dk - m.dk_g3).max() / np.abs(m.dk_g3).max()
    lines.append(f"검산(테이퍼 0): ξ {sorted(m.xi.unique())}, 관측량 {m.label.nunique()}개, 최대 |Δk|/max|k| {rel:.1e}, |Δ∂k/∂ξ| {rel_d:.1e}")
    out["check"] = dict(rel_k=float(rel), rel_dk=float(rel_d))
for name, pre, key, tag in CASES:
    if not os.path.exists(D_K + f"kernels_{tag}.csv"):
        lines.append(f"{name}: {tag} 아직 없음")
        continue
    ka, ra = pd.read_csv(G3 + f"rev1_{pre}_kernels_{key}.csv"), pd.read_csv(G3 + f"rev1_{pre}_registry_{key}.csv")
    kb, rb = pd.read_csv(D_K + f"kernels_{tag}.csv"), pd.read_csv(D_K + f"registry_{tag}.csv")
    fb = dict(zip(rb.label, rb.f_Hz))
    lines += ["", f"=== {name}: 정본 {key} → 테이퍼 {tag}",
              "  이름        f 정본 → 테이퍼 [Hz] (%)        ξ* 정본 → 테이퍼        커널 상관   max|k| 비"]
    rows = []
    for lab, f0 in zip(ra.label, ra.f_Hz):
        if f0 > 8000 or lab not in fb:
            continue
        sa, sb = ka[ka.label == lab].sort_values("xi"), kb[kb.label == lab].sort_values("xi")
        x = np.intersect1d(sa.xi.round(4), sb.xi.round(4))
        ya = sa.set_index(sa.xi.round(4)).k.loc[x].to_numpy()
        yb = sb.set_index(sb.xi.round(4)).k.loc[x].to_numpy()
        za, zb = zc(sa.xi.to_numpy(), sa.k.to_numpy()), zc(sb.xi.to_numpy(), sb.k.to_numpy())
        corr = float(np.dot(ya, yb) / np.sqrt(np.dot(ya, ya) * np.dot(yb, yb)))
        amp = float(np.abs(yb).max() / np.abs(ya).max())
        rows.append(dict(label=lab, f_g3=float(f0), f_taper=float(fb[lab]), xi_g3=za, xi_taper=zb, corr=corr, amp=amp))
        lines.append(f"  {lab:11s} {f0:7.1f} → {fb[lab]:7.1f} ({100 * (fb[lab] / f0 - 1):+5.2f})   "
                     f"{'/'.join(f'{z:.3f}' for z in za) or '없음':>13s} → {'/'.join(f'{z:.3f}' for z in zb) or '없음':13s}   "
                     f"{corr:.4f}   {amp:.3f}")
    out[tag] = rows
txt = "\n".join(lines)
print(txt)
open(D_K + "s2m_kernels.txt", "w").write(txt + "\n")
json.dump(out, open(D_K + "s2m_kernels.json", "w"), ensure_ascii=False, indent=1)
