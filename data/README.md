# Representative Data

The data start after response judging and pivot selection. They are sufficient to
audit the selected representative analyses without releasing training datasets or
the generation/judging pipeline.

Run-local paths, reviewer identifiers, review timestamps, and generation timestamps
have been removed from the public copies. Experimental values and response hashes
are unchanged.

## Qwen2.5-14B financial

- `judged_em_results.json`: coherent EM responses used by geometry analysis.
- `pivot_token_sets.json`: fixed response boundaries and pivot/neutral positions.
- `*_judged.json`: judged free generations for the target and five random controls.
- `free_generation_statistics.json`: stored analysis output for comparison.
- `subspace/`: frozen descriptive and statistical tables for the representative
  main-text gradient-subspace analysis.

The geometry interval is 225 to 750. The free-generation intervention uses
225 to 325, basis at 325, rank-4 concat, and amplification factor 1.5.
The subspace tables cover checkpoints 225, 250, 300, 350, 400, 500, 600, 700,
and 750; see `subspace/manifest.json` for the frozen settings and file hashes.

## Qwen2.5-7B financial

The fixed-response interval is 200 to 500 with basis at 200. The directory includes
the approved response pairs, fixed pivot annotations, condition
manifests, per-pair answer scores, question-level scores, and stored statistics for
ablation and amplification.

## Llama-3.1-8B sports

The fixed-response interval is 100 to 750 with basis at 100. The same frozen result
types are supplied for ablation and the factor-1.5 amplification condition.

## Gemma-3-4B-IT sports

The fixed-response interval is 200 to 500 with basis at 200. The same frozen result
types are supplied for ablation and factor-1.5 amplification.

## Common fixed-response fields

- `pair_id`, `question_id`, and `question_format` identify frozen pairs.
- `em_response` and `safe_response` contain the fixed answer texts.
- `response_start_idx`, `pivot_positions`, and `neutral_positions` refer to the
  tokenizer sequence recorded for the experiment.
- `S_EM`, `S_safe`, `S_pivot`, and `S_neutral` are mean teacher-forced log
  probabilities for the corresponding response or token subset.

Base-model weights, raw gradients, token-level score shards, and training data
are not included. The selected main-text LoRA adapters are documented under
`artifacts/`.
