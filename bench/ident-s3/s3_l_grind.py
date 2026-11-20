"""3번 시편 L(11-19 S19, 메모 s3_L) — 깃 3–4 사이 테두리를 17.6 mg(39923.5 → 39905.9 mg) 깎은 경우의 G3 FEM(56번).

사진(위에서): 칸 가운데에서 깃 3 쪽으로 약 3°, 테두리를 따라 약 11 mm(17°), 테두리 안쪽 약 3 mm(ξ0.86), 모서리 쪽이 밝다(깊다).
R(깃 4 옆) 뒤에 깎았지만, 두 자리가 떨어져 있고 양이 작아 L만 따로 푼다(겹침은 1차로 더하기).
  L17e  끝 띠 ξ ≥ 0.86, 17°, 테두리에서 가장 깊고 ξ0.86에서 0,   L17f  같은 자리 고르게
  L17v  경사면 ξ0.8에서 0(더 안쪽까지),   L17c  L17e를 칸 가운데(+30°)에 둔 것(자리 차이 확인)
출력: fe-nominal/s3_l_grind/<경우>.csv, summary.csv, key.csv
사용(PYTHONPATH=저장소, env -u PAPER3_GEOM, OMP_NUM_THREADS=2):
      python s3_l_grind.py --cases L17e,L17f --cache <pkl>    경우마다 약 4분
      python s3_l_grind.py --summary"""
from pathlib import Path

import s3_edge_grind as eg
import s3_r_grind as rg

MG = 17.6
eg.SPEC = {"L17e": ("face", 32.5, "trap", 17, 0.86, "ramp", MG), "L17f": ("face", 32.5, "trap", 17, 0.86, "flat", MG),
           "L17v": ("face", 32.5, "trap", 17, 0.8, "ramp", MG), "L17c": ("face", 30.0, "trap", 17, 0.86, "ramp", MG)}
eg.CASES = list(eg.SPEC)
eg.OUTD = Path(__file__).resolve().parents[1] / "fe-nominal" / "s3_l_grind"

if __name__ == "__main__":
    rg.cli()
