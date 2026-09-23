# Representative LoRA Adapters

This directory contains the LoRA adapters used by the representative main-text
experiments. Each adapter directory contains only `adapter_model.safetensors`
and `adapter_config.json`. Base-model weights are loaded separately from the
public model IDs stored in the PEFT configurations.

`adapter_manifest.json` records the experiment, condition, checkpoint step,
LoRA configuration, relative directory, and SHA-256 digest of every released
weight file.

## Released scope

- `qwen14b_financial`: nine harmful trajectory checkpoints and the seven
  factor-1.5 free-generation causal conditions.
- `qwen14b_safe_financial`: the nine matched safe trajectory checkpoints.
- `qwen7b_financial`: two reference checkpoints, one targeted ablation, five
  random ablations, one targeted amplification, and five random amplifications.
- `llama8b_sports`: the corresponding 14 fixed-response conditions.
- `gemma4b_sports`: the corresponding 14 fixed-response conditions.

Training-state files, dense intermediate checkpoints, appendix-only conditions,
and capability-retention adapters are outside the release scope.
