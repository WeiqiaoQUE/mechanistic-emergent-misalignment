# Representative LoRA Adapters

This directory contains selected LoRA adapters used by the representative core
experiments. Each adapter directory contains only `adapter_model.safetensors`
and `adapter_config.json`. Base-model weights are loaded separately from the
public model identifiers stored in the PEFT configurations.

`adapter_manifest.json` records the model, experiment, condition, checkpoint
step, LoRA configuration, relative path, and SHA-256 digest of every released
weight file.

## Released Scope

- `qwen14b_financial`: nine harmful trajectory checkpoints and seven factor-1.5
  free-generation intervention conditions.
- `qwen14b_safe_financial`: nine matched safe trajectory checkpoints.
- `qwen7b_financial`: two reference checkpoints, one targeted ablation, five
  random ablations, one targeted amplification, and five random amplifications.
- `llama8b_sports`: the corresponding 14 fixed-response conditions.
- `gemma4b_sports`: the corresponding 14 fixed-response conditions.

Training-state files, dense intermediate checkpoints, appendix-only conditions,
and capability-retention adapters are outside the release scope.

The released adapters allow selected trained states and intervention conditions
to be loaded with separately obtained base-model weights. They do not include
the training data, optimizer state, or pipeline required to reproduce the
original fine-tuning runs from scratch.

## Licensing Boundary

The adapters are derived from their named base models and remain subject to the
corresponding upstream model terms. They are not covered by the repository's
Apache-2.0 code license. Base-model weights are not redistributed. See
`../ARTIFACT_TERMS.md` and `../THIRD_PARTY_NOTICES.md` before downloading,
using, or redistributing an adapter. Complete local copies of the Llama and
Gemma agreements and incorporated use policies are stored in `../licenses/`.
