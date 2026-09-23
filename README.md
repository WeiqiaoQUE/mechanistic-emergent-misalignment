# Phase-Transition Core Experiments

This repository contains the core code used for the main-text geometry,
gradient-subspace, and causal-intervention experiments. It starts from frozen,
already judged responses and fixed pivot/neutral token annotations.

The release intentionally excludes fine-tuning, response generation, automated
judging, pivot selection, capability benchmarks, appendix-only analyses, and
plotting code.

## Included experiments

- Curvature measurements on fixed pivot and neutral token sets.
- Harmful/safe gradient-subspace construction and overlap tests.
- Frozen Qwen2.5-14B subspace and statistical result tables.
- Construction of harmful-direction ablation/amplification adapters and
  parameter-norm-matched random controls.
- Fixed-response teacher-forcing scores and question-level inference.
- Direction-specificity statistics for already judged free generations.

Representative frozen data and selected main-text LoRA adapters are provided for
Qwen2.5-14B financial, Qwen2.5-7B financial, Llama-3.1-8B sports, and
Gemma-3-4B-IT sports. Base-model weights and training datasets are not included.

## Installation

The tested environment uses Python 3.10, PyTorch 2.9.1, and CUDA 12.8. Create
the Conda environment and install the package with:

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

The released data are sufficient to rerun the fixed-response and free-generation
statistics. The selected adapters support geometry analysis and model-level
fixed-response scoring after the public base models have been downloaded. The
Qwen2.5-14B subspace tables expose the numerical inputs and outputs used for the
main-text comparisons. Recomputing gradient subspaces or rebuilding intervention
adapters from scratch requires the harmful/safe datasets or gradient caches, which
are not distributed. The already constructed adapters needed by the released
fixed-response experiments are included.

See `artifacts/README.md` and `artifacts/adapter_manifest.json` for the released
adapter scope and checksums. Llama and Gemma base models may require accepting
their model licenses before download.

## Released-data analysis

```bash
python scripts/analyze_causal_effects.py \
  --config configs/qwen7b_financial.json \
  --answer-scores data/qwen7b_financial/ablation_answer_scores.jsonl
python scripts/analyze_causal_effects.py \
  --config configs/qwen7b_financial_boost.json \
  --mode boost \
  --answer-scores data/qwen7b_financial/amplification_answer_scores.jsonl
```

The free-generation analysis reads only judged result files:

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

## Model-dependent stages

After supplying the paths described above:

```bash
python scripts/run_geometry.py --config configs/qwen14b_financial.json
python scripts/run_subspace.py --config configs/qwen14b_financial.json
python scripts/run_subspace_statistics.py --config configs/qwen14b_financial.json
python scripts/build_interventions.py --config configs/qwen7b_financial.json --dry-run
python scripts/score_fixed_responses.py --config configs/qwen7b_financial.json --dry-run
```

## Tests

The tests exercise the mathematical operations used by the main experiments and
do not load a language model:

```bash
python -m unittest discover -s tests -v
```

## Data scope

See `data/README.md` for the schema and selected configurations. The release does
not contain a complete numerical archive for every model in the paper.
