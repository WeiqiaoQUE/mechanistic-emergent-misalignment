# Qwen2.5-14B Gradient-Subspace Results

These frozen tables contain the numerical outputs used for the representative
main-text gradient-subspace analysis. They cover the nine released harmful and
safe LoRA checkpoints and contain no raw examples or per-sample gradients.

## Descriptive tables

- `block_overlap_by_step.csv`: normalized subspace overlaps and principal-angle
  cosines across harmful, safe, pivot, and neutral gradient sets.
- `directional_metrics_by_step.csv`: alignment and projected-energy metrics for
  pivot and neutral gradients.
- `effective_rank_by_step.csv`: effective rank and cumulative explained variance.
- `random_baseline_by_step.csv`: random-subspace overlap reference summaries.
- `singular_values_by_step.csv`: singular-value spectra by checkpoint and block.

## Statistical tables

- `financial_statistical_tests_by_step.csv`: complete bootstrap and permutation
  results for ranks 4 and 8.
- `financial_bootstrap_ci_by_step.csv`: bootstrap confidence-interval view.
- `financial_train_permutation_by_step.csv`: permutation-test view.
- `financial_primary_family_summary.csv`: the primary rank-4 concat and B-block
  family after Benjamini-Hochberg correction.

`manifest.json` records the public experiment settings and SHA-256 digest of
each table. Full regeneration of the per-sample gradients requires the source
datasets or the unreleased gradient cache.
