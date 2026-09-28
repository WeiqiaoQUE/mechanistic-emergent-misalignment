# Representative Data

The release contains frozen, post-judging responses, scores, token sets, and
derived statistics. These records support auditing and recomputing selected
representative analyses, but the raw inputs and pipelines required to regenerate
responses, rerun automated judging, or repeat pivot selection are not included.

Run-local paths, reviewer identifiers, review timestamps, and generation
timestamps have been removed. Experimental values and response hashes are
unchanged. The files include generated model responses, including examples of
unsafe behavior; see `../RESPONSIBLE_USE.md` before inspecting or reusing them.

## Qwen2.5-14B Financial

- `judged_em_results.json` contains coherent EM responses used by geometry
  analysis.
- `pivot_token_sets.json` contains fixed response boundaries and pivot/neutral
  positions.
- `*_judged.json` contains judged free generations for the target intervention
  and five random controls.
- `free_generation_statistics.json` contains the stored analysis output.
- `subspace/` contains frozen descriptive and statistical tables for the
  representative main-text gradient-subspace analysis.

The geometry interval is 225 to 750. The free-generation intervention uses 225
to 325, basis at 325, rank-4 concat, and amplification factor 1.5. The subspace
tables cover checkpoints 225, 250, 300, 350, 400, 500, 600, 700, and 750. See
`subspace/manifest.json` for the frozen settings and file hashes.

## Qwen2.5-7B Financial

The fixed-response interval is 200 to 500 with basis at 200. The directory
contains approved response pairs, fixed pivot annotations, condition manifests,
per-pair scores, question-level scores, and stored ablation and amplification
statistics.

## Llama-3.1-8B Sports

The fixed-response interval is 100 to 750 with basis at 100. The directory
contains the corresponding frozen ablation and factor-1.5 amplification results.

## Gemma-3-4B-IT Sports

The fixed-response interval is 200 to 500 with basis at 200. The directory
contains the corresponding frozen ablation and factor-1.5 amplification results.

## Common Fixed-Response Fields

- `pair_id`, `question_id`, and `question_format` identify frozen pairs.
- `em_response` and `safe_response` contain fixed answer texts.
- `response_start_idx`, `pivot_positions`, and `neutral_positions` identify the
  recorded tokenizer sequence and token subsets.
- `S_EM`, `S_safe`, `S_pivot`, and `S_neutral` are mean teacher-forced log
  probabilities for the corresponding response or token subset.

Base-model weights, raw gradients, token-level score shards, and training data
are not included. These data and generated outputs are not covered by the
repository's Apache-2.0 code license. See `../THIRD_PARTY_NOTICES.md` for the
applicable provenance and upstream terms.
