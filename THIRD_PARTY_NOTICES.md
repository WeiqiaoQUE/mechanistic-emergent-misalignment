# Third-Party Notices

The Apache License 2.0 in `LICENSE` applies to the original source code in this
repository. It does not relicense base models, model-derived adapters, frozen
model outputs, or other third-party material.

## Base Models and Derived Adapters

The repository contains LoRA adapters derived from the following base models:

| Base model | Upstream terms | Released adapter groups |
|---|---|---|
| Qwen2.5-7B-Instruct | [Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `artifacts/qwen7b_financial/` |
| Qwen2.5-14B-Instruct | [Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | `artifacts/qwen14b_financial/`, `artifacts/qwen14b_safe_financial/` |
| Llama-3.1-8B-Instruct | [Llama 3.1 Community License](licenses/LLAMA_3_1_COMMUNITY_LICENSE.txt) and [Acceptable Use Policy](licenses/LLAMA_3_1_ACCEPTABLE_USE_POLICY.md) | `artifacts/llama8b_sports/` |
| Gemma-3-4B-IT | [Gemma Terms of Use](licenses/GEMMA_TERMS_OF_USE.html) and [Prohibited Use Policy](licenses/GEMMA_PROHIBITED_USE_POLICY.html) | `artifacts/gemma4b_sports/` |

The PEFT `adapter_config.json` files retain the corresponding upstream model
identifiers. Base-model weights are not included. Downloading, using, modifying,
or redistributing an adapter may remain subject to the upstream terms for its
base model. Users must review and comply with those terms independently.

The Llama 3.1 Community License requires distributions of Llama materials or
derivatives to include a copy of that agreement, retain its specified
attribution notice, and display `Built with Llama`. The Gemma Terms require
distributions of Gemma model derivatives to include a copy of the agreement,
carry the specified notice, and bind downstream use to its use restrictions.
The required attribution strings are retained in `NOTICE`. Complete copies of
the upstream agreements and incorporated use policies are included in
`licenses/`. The conditions applied to each released adapter group are stated in
`ARTIFACT_TERMS.md`.

## Frozen Data and Generated Outputs

The `data/` directory contains selected prompts, generated model responses,
token annotations, scores, and derived statistics. These materials are provided
to audit the associated research results. No additional license grant for these
materials is made by the repository's Apache-2.0 code license. Upstream model and
source-dataset terms continue to apply where relevant.

## Software Dependencies

Runtime dependencies are listed in `requirements.txt` and `pyproject.toml`.
Each dependency is distributed under its own license. Installing this package
does not alter those licenses.
