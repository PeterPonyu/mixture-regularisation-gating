# Cap and annealing controls for a late mixture regulariser

## Repository, archive and citation

Study repository: [PeterPonyu/mixture-regularisation-gating](https://github.com/PeterPonyu/mixture-regularisation-gating). Versioned archive: [10.5281/zenodo.22914503](https://doi.org/10.5281/zenodo.22914503). Use `CITATION.cff` for release 1.0.0; `release-manifest.json` records distributed-file checksums. The archive and source repository describe this study only.

Author-owned software is MIT licensed. The author's manuscript, figures, generated results and model weights are CC BY 4.0. Source-study data, labels, annotations and other third-party materials retain their original terms; they are not relicensed. Read `LICENSE` and `NOTICE.md` before reusing mixed-content files.

This research package studies how a capped mixture loss and its annealing
schedule control realised gradient exposure in an attention autoencoder.
The saved experiment compares a zero-weight reference with cap-5 and uncapped
losses under 20- and 100-epoch ramps. Two fixed data backgrounds, Setty and
Endo, and three seeds produce six shared warmups and 30 matched endpoints.
Saved continuation histories cover epochs 200–219 and the epoch-220 endpoint.
Historical capped and uncapped sweeps on four backgrounds supply additional
context.

The reproducible product is analysis of those saved computations: per-batch
exposure checks, paired endpoint contrasts, robustness analyses, figures and
the article build. All commands below run on a CPU with no training, GPU,
raw dataset or model checkpoint. Large `.pt` files are distributed separately
in this article's Zenodo checkpoint archive; the default GitHub analyses and
figures do not read them.

For operations that consume model states, download this article's checkpoint
archive from its Zenodo DOI record and restore the package-relative paths
listed under `optional_checkpoints` in `publication-manifest.json`. A GitHub
source checkout alone does not supply those model states, and these checkpoints
do not constitute a complete training-replay environment.

## CPU analysis

Use Python 3.11 or newer with NumPy and SciPy from
`requirements-analysis.txt`. The recorded validation used Python 3.13.5,
NumPy 2.2.6 and SciPy 1.16.3. From the package root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-analysis.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/exposure_diagnostic.py
python scripts/robustness_audit.py
python scripts/verify_publication.py
```

`exposure_diagnostic.py` uses only the Python standard library. It matches all
30 original and continuation endpoints, verifies all 8,700 saved per-batch
records, and recomputes summed weighted latent-gradient norms. It writes
`evidence/exposure_diagnostic.json`, including within-warmup contrasts,
descriptive correlations and analytic sensitivity of the clipping gates to
a simultaneous rescaling of latent coordinates and Gaussian parameters.

`robustness_audit.py` also reads the 30 continuation `.npz` archives. It checks
finite latents, responsibility normalization, shared batch-history identifiers,
zero exposure in capped arms and equality of capped latents with the
zero-weight reference. It reports per-background and per-schedule associations,
normalization by update count, removal of individual seeds or warmups, and
the original mixture-fit convergence flags. It creates `validation/` when
needed and writes `validation/robustness-audit.json`.

`verify_publication.py` uses only the Python standard library and checks
`publication-manifest.json`. The manifest distinguishes original source
digests from current distributed-file digests, identifies source snapshots,
and records the semantic relabeling of machine-specific paths. Checkpoints
are optional for this check; scientific arrays and required analysis inputs
are verified even when every `.pt` is absent.

## Figures and article build

Figure regeneration requires R with `ggplot2`, `gridExtra`, `jsonlite`, the
standard `grid` package, and Cairo graphics support. Tested versions are
R 4.3.3, ggplot2 4.0.3, gridExtra 2.3 and jsonlite 2.0.0. Install missing R
packages in an existing R environment with:

```sh
Rscript -e 'install.packages(c("ggplot2", "gridExtra", "jsonlite"), repos="https://cloud.r-project.org")'
```

Arial and the system librsvg, Cairo and GObject shared libraries must be
installed. The vector renderer uses Python's standard library; no Python
PyGObject or Pycairo package is needed. `python3` must be on `PATH`; `PYTHON`
can select another executable. A TeX installation with XeLaTeX and BibTeX,
the packages named in `standalone-assets/preamble.tex`, and the TeX Gyre
Termes Math font is required for the article.

```sh
Rscript scripts/build_figures.R
bash build.sh
```

The first command rebuilds seven data-figure PDF/PNG pairs and the vector
architecture PDF, and writes `figures/plot_manifest.json` with panel counts.
All numerical inputs are local JSON and CSV files listed in
`figure-build.json`. Non-finite values in historical score JSON are interpreted
as missing values during plotting without editing the archived scores. The
six UMAP panels reuse the saved 450-point coordinates and labels; they do not
fit UMAP.

`build.sh` regenerates the figures, runs XeLaTeX, BibTeX and two further
XeLaTeX passes, and copies `build_independent/paper.pdf` to `paper.pdf`.
It overwrites generated figures and the article PDF. Pre-rendered figures
are also included. Neither figure generation nor compilation changes model
parameters or produces new training outcomes.

## Evidence and limits of reproduction

- `evidence/factorial_results.json` contains the 30 epoch-200 endpoints,
  shared-warmup checks, original mixture parameters and convergence flags.
- `evidence/run-archive/` contains the 30 continuation endpoint records and
  arrays, six permutation schedules, gradient histories and mixture refits.
  `evidence/new_results.json` is the same aggregate used by the figures.
- `evidence/new_protocol.json` records the continuation design, and
  `evidence/EXPOSURE_METHODS.md` explains the exposure estimand and limits of
  the analytic scaling calculation.
- `evidence/source/dpmm_transformer.py` is an unmodified model snapshot whose
  SHA-256 matches the original experiment. It is provided for inspection of
  the loss and gating implementation; it is not a standalone importable
  training library. The nonportable historical training drivers are excluded.
  Their original hashes remain in the result records and publication manifest.

The available arrays support the stated saved-result audits. They do not
provide a self-contained raw-data preprocessing or training pipeline, a
reconstruction of all epoch-200 checkpoint states, or regeneration of the
archived graph scores. Checkpoint files alone do not remove these limits.

The exposure estimand is the sum of effective weight times the norm of the
clipped mixture-loss gradient with respect to the latent representation.
It is not parameter displacement. Per-batch gradient magnitudes were not
archived for epochs 180–199 and cannot be recovered from active-batch counts.
Continuation refit convergence flags were not recorded; finite values do
not prove convergence. The original fit records include 46 non-converged
refits and are retained.

Six warmups and two fixed backgrounds are the computational units of the
paired analysis; arms, batches and cells are not independent donors.
Exposure is a post-treatment quantity and depends on latent scale and fitting.
Its association with an endpoint is descriptive, not an identified causal
dose–response. The analytic rescaling calculation shifts the clipping gates
without retraining and makes no prediction of a new trained endpoint.
