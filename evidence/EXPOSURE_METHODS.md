# Gradient exposure and edge-survival analysis

Run `python3 scripts/exposure_diagnostic.py` from the package root. It reads only
`evidence/factorial_results.json` and `evidence/run-archive/results.json`,
checks all 30 arm identities and 200--219 per-batch coverage, then regenerates
`evidence/exposure_diagnostic.json` with SHA-256 input hashes. `bash build.sh`
independently regenerates `paper.pdf`. The manuscript uses the evidence in the
Results and Discussion, with no new training run or figure.

The estimand in the descriptive scatter is the sum of **weighted latent**
mixture-loss gradient norms during epochs 200--219 versus the change in edge
survival from the epoch-200 to epoch-220 endpoints, subtracting the analogous
uncoupled-anchor change within the same background and seed. The fast-minus-slow
contrast is nested within a shared warmup. Twelve uncapped arm rows are six
paired computational histories across two fixed backgrounds; neither batches
nor arms are independent donors. No unrecorded 180--199 dose is imputed.

The earlier archive has 140--192 effective nonzero-gradient batches per
uncapped arm; the continuation has 14--34. This supports a decline in
active-batch frequency in the observed training windows, not an estimate of
how their gradient magnitudes compare. The zero floor `max(0, NLL)` makes
latent units relevant. The analytic scale check holds archived raw NLL fixed
apart from the exact Gaussian change `10 log(s)` when **both** ten-dimensional
latent/means and covariances are transformed by `s`/`s²`; it is not a model
intervention, refit, or gradient-norm sensitivity experiment. In particular,
scaling latent alone while keeping fitted parameters fixed is not captured by
this identity.

## Frozen gate for any future confirmatory claim

Before collecting endpoints, prespecify the NLL convention, clipping gates,
latent scale or scale-invariant alternative, per-batch weighted gradient
definition, mixture refit timing, training windows, edge endpoint, contrast,
biological sampling unit and clustered analysis. Archive the complete
180--220 per-batch stream and train prospective scale/refit controls with
matched random streams, plus independent donor/cohort replication. Only then
could a full-window exposure model or scale-sensitive causal intervention be
evaluated. The current association is exploratory and cannot be called an
identified dose--response or biological causal effect.
