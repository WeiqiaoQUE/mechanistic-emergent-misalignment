# Mechanistic Emergent Misalignment

Code and selected research artifacts for studying training geometry, gradient
subspaces, and parameter-space interventions in emergent misalignment.

**Paper:** forthcoming

## Overview

The repository implements the core analyses used to connect three stages of the
study:

1. measuring the geometry of frozen pivot and neutral token sets;
2. comparing evaluation gradients with harmful and safe training subspaces; and
3. ablating or amplifying harmful-direction components of LoRA updates.

The release provides selected trained adapters together with frozen,
post-judging responses, scores, token sets, and derived statistics. It does not
include the fine-tuning, response-generation, automated-judging,
pivot-selection, capability-evaluation, appendix-only analysis, or
figure-generation pipelines.

## Released Scope

Representative data and selected LoRA adapters are provided for:

- Qwen2.5-14B-Instruct on financial advice;
- Qwen2.5-7B-Instruct on financial advice;
- Llama-3.1-8B-Instruct on extreme sports; and
- Gemma-3-4B-IT on extreme sports.

The release supports direct recomputation of the fixed-response and
free-generation statistics. With separately obtained public base-model weights,
the selected adapters also support Qwen2.5-14B geometry analysis and model-level
fixed-response scoring. Frozen Qwen2.5-14B tables expose the numerical inputs and
outputs of the representative gradient-subspace analysis.

Training datasets, raw gradient caches, base-model weights, token-level scoring
shards, and the generation and judging pipeline are not distributed. As a
result, gradient subspaces and intervention adapters cannot be reconstructed
from scratch from this repository alone. The released results can be audited,
and selected analyses can be recomputed, but the complete pipeline cannot be
reproduced from raw training data using this repository alone. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact boundary.

## Repository Layout

```text
artifacts/   Selected LoRA adapters and their integrity manifest
configs/     Frozen configurations for the released experiments
data/        Frozen inputs, scores, statistics, and subspace tables
licenses/    Upstream model agreements and incorporated use policies
scripts/     Command-line entry points
src/         Geometry, subspace, intervention, scoring, and statistics code
```

## Installation

The reference environment uses Python 3.10, PyTorch 2.9.1, and CUDA 12.8.

```bash
conda env create -f environment.yml
conda activate phase-transition-core
python -m pip install -e . --no-deps
```

For an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Llama and Gemma downloads require accepting their upstream model terms. No base
model is bundled with this repository. The applicable adapter terms and bundled
copies of the upstream agreements are indexed in
[ARTIFACT_TERMS.md](ARTIFACT_TERMS.md).

**Built with Llama.** The Llama-derived collection is named **Llama Mechanistic
Emergent Misalignment Adapters**.

## Reproduce Released Statistics

The following commands use only released data and do not load a language model.

```bash
python scripts/analyze_causal_effects.py \
  --config configs/qwen7b_financial.json \
  --answer-scores data/qwen7b_financial/ablation_answer_scores.jsonl

python scripts/analyze_causal_effects.py \
  --config configs/qwen7b_financial_boost.json \
  --mode boost \
  --answer-scores data/qwen7b_financial/amplification_answer_scores.jsonl
```

The released free-generation results can be analyzed with:

```bash
python scripts/analyze_free_generation.py \
  --target data/qwen14b_financial/C5_harm_boost_k1.5_judged.json \
  --random data/qwen14b_financial/C6_rand_boost_nm_s0_k1.5_judged.json \
  --random data/qwen14b_financial/C6_rand_boost_nm_s1_k1.5_judged.json \
  --random data/qwen14b_financial/C6_rand_boost_nm_s2_k1.5_judged.json \
  --random data/qwen14b_financial/C6_rand_boost_nm_s3_k1.5_judged.json \
  --random data/qwen14b_financial/C6_rand_boost_nm_s4_k1.5_judged.json \
  --quantity-key layer1_concat/325/C5_harm_boost_k1.5/delta
```

## Model-Dependent Analyses

After obtaining the base models referenced by the configurations:

```bash
python scripts/run_geometry.py --config configs/qwen14b_financial.json
python scripts/score_fixed_responses.py --config configs/qwen7b_financial.json
```

The subspace and intervention entry points are included for method inspection.
Complete execution requires training data or gradient caches that are outside the
public release.

## Data and Artifacts

[data/README.md](data/README.md) documents the frozen schemas and representative
configurations. [artifacts/README.md](artifacts/README.md) describes the adapter
scope, while `artifacts/adapter_manifest.json` records the model, experiment,
condition, LoRA settings, path, and SHA-256 digest for every released weight
file.

Some released records contain examples of unsafe model behavior, and some
adapters intentionally amplify a measured harmful direction for controlled
analysis. Review [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) before using these
materials.

## License

The original source code is licensed under the Apache License 2.0. Frozen data,
model-derived adapters, and third-party materials are not relicensed by that
grant. Their applicable terms and attributions are described in
[ARTIFACT_TERMS.md](ARTIFACT_TERMS.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), `NOTICE`, and `licenses/`.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). The arXiv
identifier and public paper URL will be added after publication.

## Contact

For questions about the released code or artifacts, please open a GitHub issue.
