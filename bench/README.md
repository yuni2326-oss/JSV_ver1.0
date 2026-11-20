# Bench tests with perturbations of known mass and position

Analysis scripts and processed data of the bench tests on spare impellers of the studied type,
measured in free–free suspension (two rubber bands through the bore) with a miniature impact
hammer on the back shroud and a microphone. The scripts are kept as they were used; most
comments are in Korean.

- Specimen 2: single lead pieces (T-a, T-b), six-fold symmetric solder sets (M120, M240),
  handling and re-suspension controls (A2, A2-rh), and a removal step (L1) that the paper does
  not use.
- Specimen 3: healthy baseline, a cut beside a vane (R) and a cut in mid-bay (L).
- Specimen 1: earlier hammer-test series and a pretest; not used in the paper.

## Layout

```
ident-s3/      perturbation tests
  groove_lib groove_state groove_compare   hit reading, quality checks, free-decay pole-residue
                                           fits (per hit and pooled), per-state summaries, and
                                           state-to-state comparison with the model
  s3_spec3_*                               specimen 3 baseline, R and L at 12 driving points
  s3_m_photo s3_m_piece_scan s3_m_angle_scan   solder-piece positions from photographs and the
                                           model scans over piece position
  s3_local s3_edge_grind s3_r_grind s3_l_grind s3_l1_cause s3_km_*   model footprints of
                                           material removal
  paper_bench_data paper_bench_figs        tables and figures of the bench sections
ident-s2/      specimen 2 healthy-state identification and pairing with the model
ident-w22/     specimen 1 six-vane series
groove-test/   test records (sessions, masses, pieces, machining, shroud steps) and processed
               results: out/ (per-state modes and frequency response functions, comparisons),
               s3/ (specimen 3 states), m_piece_positions.csv (photographed piece positions)
fe-nominal/    model results: healthy modes, piece scans, removal footprints
paper-fig/     tables written by paper_bench_data.py
```

## Regenerating the tables and figures

From the repository root, with the main package on the path:

```bash
cd bench/ident-s3
PYTHONPATH=../..:. python paper_bench_data.py     # tables  -> ../paper-fig/
PYTHONPATH=../..:. python paper_bench_figs.py     # figures -> ../paper-fig/
```

Both read only the processed files committed here. The mean spectra of the figures come from
`groove-test/s3/paper_spec_cache.npz`; `--refresh-spectra` rebuilds that cache from the raw
recordings.

## Not included

- **Raw recordings.** The force and sound-pressure time series of the hits are not part of this
  repository. Scripts that process them still carry the storage paths they were run with and
  cannot run without the recordings.
- **Photographs.** Fig. E1 (c, d) and Fig. SE2 of the paper are photographs of the specimens.
  The figure script leaves the two panels of Fig. E1 empty and skips Fig. SE2 when the
  photographs are absent. The positions measured on them are in `groove-test/out/m_photo.json`
  and `groove-test/m_piece_positions.csv`.
