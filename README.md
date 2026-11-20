# Symmetry-derived frequency observables in a pump impeller — computation and data

Code and committed data for a study of frequency-based damage identification in a small
vertical multistage pump impeller: symmetry-derived modal observables (pair mean, splitting,
orientation), the radial inverse and its identifiability, independent-model 3D FEM tests
that expose forward-model discrepancy, and bench tests with perturbations of known mass and
position on two spare impellers of the studied type.

Version 2 adds the computations made for the first revision of the paper
(`impeller_fingerprint/rev1/` and the `rev1_*` artifacts), takes the as-built geometry from
standard sheet gauges (front shroud 0.40 mm, back shroud 0.60 mm, shroud gap 3.00 mm, vane
1.0 mm), and adds the bench-test analysis in `bench/`. Version 1 is kept under the tag
`ver1.0`.

## What is here

```
impeller_fingerprint/            analysis package (Python)
  geometry severity kernels      geometry/material constants, damage parameterization,
  forward validity               annular-plate Ritz kernels, first-order map and its
                                 validity assessment against a non-perturbative re-solve
  degenerate                     2x2 pair matrix of a symmetry-protected doublet:
                                 trace (pair mean), traceless part (splitting, orientation)
  noise estimator identifiability  raw-frequency covariance Sigma_y = A Sigma_f A^T,
                                 weighted least squares, Fisher information, CRLB,
                                 profile likelihood
  modeselect montecarlo          D-optimal mode selection; production Monte-Carlo driver
  crack_shear crack2d            cracked-beam transfer matrix (Timoshenko, Mode-I/II) and
                                 zero-width 2D plane-elastic slit
  rail3d impeller_cad impeller_hex  3D solid rails: annular disk with pockets, parametric
                                 six-vane assembly (tetrahedral CAD surrogate and
                                 structured-hex cyclic-symmetry mesh)
  massload references            contact-sensor mass-loading limits; bibliography with
                                 Crossref verification
  eb_reference                   Euler-Bernoulli references: rotational-spring transfer
                                 matrix, (1-d)-weighted Ritz, equivalent band, wet ratio
  submission supplementary       renderers that typeset the reported tables from the CSVs
  figures cli                    figure generators and the command-line entry point
  rev1/                          computations of the revision:
    n1_* n2_*                    six-vane assembly on a camber-following hexahedral mesh:
                                 assembly kernels, Fisher information and mode selection
                                 (N1); localized pockets and their azimuth dependence (N2)
    n3_crack                     robustness of the vane-crack fingerprint (N3)
    n4_width                     damage-width mismatch and width-conditional
                                 identifiability (N4)
    n5_*                         decomposition of the transfer bias (linearization, damage
                                 shape, structural model form) and a held-out discrepancy
                                 correction (N5)
    n6_*                         free-free as-built assembly: mode map, excitation and
                                 response visibility, contact-sensor mass loading (N6)
    n0_admissibility             physical-admissibility audit of the production fits
    e1c_coupon g1_section        vane-coupon notch predictions; two-sheet section formulas
  tests/                         regression tests
docs/_generated/data/paper3/     committed artifacts every reported number is read from
bench/                           bench tests on two spare impellers (see bench/README.md)
```

## Conventions

- **Numbers live in the artifacts.** Every value reported in the paper is read from a
  committed CSV, JSON or NPZ in `docs/_generated/data/paper3/` or `bench/`; the renderers
  compute nothing. The one exception is listed under Artifacts.
- **One command per artifact.** Each file is produced by a named subcommand
  (`python -m impeller_fingerprint.cli <item>`) or by a `rev1` module
  (`python -m impeller_fingerprint.rev1.<module>`, whose docstring lists its subcommands and
  outputs). A committed file without a command is treated as a defect.
- **Modes are matched by shape, never by frequency order.** Splitting, veering and mixing
  reorder modes, so every 3D comparison uses azimuthal-order projection, MAC / subspace MAC,
  or beam-mode classification. For degenerate pairs only basis-invariant quantities (the
  group trace) are quoted.
- **Stiffness and mass removal are coupled** by the damage geometry
  (`d_M = 1 - (1 - d_K)^(1/3)` for the monolithic rail, `d_M = d_K/(1 + d_K)` for a one-face
  machined two-sheet section), which is what produces the mode-dependent sign reversals.
- **Figures carry data, not captions.** Panel letters, axis and colorbar labels,
  legends and parameter identifiers stay in the image; what a panel shows is stated
  where the figure is used, not printed twice.
- **The repeatability covariance is parameterized.** All reported bounds are conditional on
  the assumed `sigma_f/f`; observable-specific floors are `2c` for a doublet pair mean,
  `2*sqrt(2)*c` for the m = 0 single-mode shift and `4c` for incremental splitting. The bench
  tests measure components of the frequency error (within a session, across re-suspension,
  after handling) but not a full covariance.

## Running

Python 3.12. Core analysis needs `numpy`, `scipy`, `pandas`, `matplotlib`; the 3D rails and
assemblies need `sfepy`, `meshio` and the `gmsh` executable.

```bash
python -m pytest impeller_fingerprint/tests -q          # regression suite
python -m impeller_fingerprint.cli a2 --mass            # first-order validity maps
python -m impeller_fingerprint.cli a3 --mass            # identifiability maps and summary
python -m impeller_fingerprint.cli a5 --mass            # mode / observable selection
python -m impeller_fingerprint.cli b1 --mass            # production Monte-Carlo (long)
python -m impeller_fingerprint.cli a19                  # geometry-matched control ladder
python -m impeller_fingerprint.cli figs                 # figures
python -m impeller_fingerprint.cli supplementary --layout r36 --out-md supp.md
python -m impeller_fingerprint.cli datapackage --out data-package.zip
```

The `rev1` assembly computations take minutes to hours per item; their docstrings give the
commands and the artifacts they write.

The renderers emit markdown. The `--docx` option of `submission` and `supplementary` shells
out to a separate markdown-to-docx converter, which is not part of this repository; point
`MD2DOCX` at one to use it, or leave the option off and keep the markdown. Tests that compare
numbers against the manuscript or its generators skip when those are absent, which they are
here.

Outputs go to `docs/_generated/{data,figures}/paper3/`; set `PAPER3_OUT` to write elsewhere
(the tests read the same variable, defaulting to this checkout). Figures are regenerated
rather than committed. Reference verification is optional and, if used, takes the Crossref
contact address from `CROSSREF_MAILTO`.

The Euler-Bernoulli references that the crack arms and the wet correction need — the
rotational-spring transfer matrix, the (1−d)-weighted Ritz solve, the equivalent low-stiffness
band, the Dimarogonas compliance polynomial of the literature, and the added-mass frequency
ratio — are implemented in `impeller_fingerprint/eb_reference.py` and `crack_shear.py`, and
cross-checked to 1e-12 against the companion package of an earlier study when that package
happens to be importable (those checks skip otherwise). Solves are deterministic: the ARPACK
start vector is fixed.

## Artifacts

`docs/_generated/data/paper3/` holds the committed results: the 240-cell production
Monte-Carlo summary (1.2e6 constrained fits) and its admissibility audit, first-order validity
contours and error maps, identifiability summaries and grids, profile-likelihood landscapes,
mode- and observable-selection comparisons, the crack-model arms (rotational spring with two
compliance conventions, Timoshenko with Mode-I/II flexibility, zero-width 2D elasticity in
plane stress and plane strain, finite-width 3D notch) with their convergence ladders and
robustness cases, the 3D pocket and axisymmetric-band studies with the surrogate used for the
model-form Monte-Carlo, the bias decomposition and held-out correction, the six-vane assembly
kernels, sign-reversal radii and mode selection, the localized-pocket azimuth sweeps, the
free-free as-built mode maps and sensor mass-loading checks, and the matched-damage and
geometry-matched component/assembly controls.

Not included: the supplementary table that summarizes the monitoring archive of an operating
pump. That archive is not public, so neither the session data derived from it nor the code
that reads it are part of this repository, and that table cannot be regenerated here.

`cli datapackage` bundles the artifacts a document cites together with a manifest carrying
byte counts and SHA-256 digests.

## License

MIT — see `LICENSE`.
