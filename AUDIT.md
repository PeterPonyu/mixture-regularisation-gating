# Exposure-accounting and cap-control audit

The CPU audit verifies thirty continuation endpoints, six permutation schedules and 8,700 per-batch records. Twelve capped-arm latent arrays match their zero-weight controls element by element. Responsibility normalization, finite latents, shared ordering, exposure summaries and removal of seeds or warmups are checked without checkpoints or training.

The recorded exposure is effective weight multiplied by the norm of the latent gradient, summed over batches. It is not parameter displacement. Earlier per-batch gradient magnitudes were not saved, and continuation convergence flags were not recorded; those values cannot be reconstructed from active-batch counts or finite endpoints.

Exposure is a post-treatment, scale-sensitive quantity. Its endpoint association is descriptive rather than an identified causal dose-response. The source snapshot supports inspection of the loss implementation but is not a complete standalone training library. Raw preprocessing, all original checkpoint states and historical graph-score reconstruction are outside the supported release scope.

## Verification boundary

The README specifies the commands and tested dependencies. `release-manifest.json` identifies the distributed files; historical producer hashes identify their original computations. All manuscript pages were rendered and programmatically checked for layout candidates. This is not a human visual sign-off. Scientific scope and known limitations are retained even when computational checks pass.
