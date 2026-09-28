# Reproducibility Scope

This release distinguishes direct statistical recomputation from analyses that
require separately obtained model weights or inputs that are not distributed.

## Coverage

| Analysis | Released inputs | Additional requirement | Supported scope |
|---|---|---|---|
| Fixed-response statistics | Pair-level scores and manifests | CPU Python environment | Direct recomputation |
| Free-generation statistics | Judged generations for target and random controls | CPU Python environment | Direct recomputation |
| Qwen2.5-14B geometry | Frozen responses, token sets, and trajectory adapters | Qwen2.5-14B-Instruct weights and GPU | Model-level recomputation |
| Fixed-response scoring | Frozen pairs, token sets, and condition adapters | Corresponding base-model weights and GPU | Model-level recomputation |
| Qwen2.5-14B subspace results | Frozen tables and statistical summaries | None for table inspection | Numerical audit only |
| Gradient-subspace construction | Code and configuration schema | Unreleased harmful/safe training data and gradient caches | Not reproducible from this release alone |
| Intervention-adapter construction | Code and configuration schema | Unreleased gradient caches | Not reproducible from this release alone |

## Direct Statistical Reproduction

The fixed-response analysis consumes released answer-score JSONL files and
aggregates effects by question before computing percentile bootstrap intervals,
exact sign-flip tests, direction-specific contrasts, and leave-one-question-out
diagnostics.

The free-generation analysis consumes already judged result files. It compares
the target harmful-direction intervention with five parameter-norm-matched random
controls using question-level rates. It does not regenerate responses or rerun an
automated judge.

## Model-Dependent Reproduction

Base-model weights are not included. They must be obtained from the upstream
providers under their respective terms. The released PEFT configurations record
the model identifiers used by each adapter. Llama and Gemma access may require
accepting provider terms before download.

Hardware requirements depend on the model and command. The reference environment
uses Python 3.10, PyTorch 2.9.1, and CUDA 12.8. Model-level commands are not
expected to run in a CPU-only environment.

## Frozen Artifacts

`artifacts/adapter_manifest.json` records the relative path and SHA-256 digest of
every released adapter weight file. The Qwen2.5-14B subspace directory contains a
separate manifest for the frozen settings and tables. These manifests establish
the exact artifact inventory used by the release; they do not replace the
unreleased training data or gradient caches.

## Excluded Pipeline Stages

The public release excludes fine-tuning, response generation, automated judging,
pivot selection, capability benchmarks, appendix-only analyses, and plotting.
The released data begin after judging and token-set freezing, so those earlier
stages cannot be reconstructed from this repository.
