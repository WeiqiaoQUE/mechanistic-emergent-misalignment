# Terms for Released Model Adapters

The Apache License 2.0 in `LICENSE` covers the original source code in this
repository. It does not grant rights to base models or relicense adapters that
are derived from them.

By downloading, using, modifying, or redistributing an adapter in `artifacts/`,
you agree to comply with the terms that apply to its base model, as identified
below and in the adapter's `adapter_config.json`. Base-model weights are not
included in this repository.

## Qwen-Derived Adapters

The adapter groups below derive from upstream models released under the Apache
License 2.0:

- `artifacts/qwen7b_financial/`: Qwen2.5-7B-Instruct
- `artifacts/qwen14b_financial/`: Qwen2.5-14B-Instruct
- `artifacts/qwen14b_safe_financial/`: Qwen2.5-14B-Instruct

Review the official model cards linked from `THIRD_PARTY_NOTICES.md` before use
or redistribution.

## Llama-Derived Adapters

`artifacts/llama8b_sports/` derives from Llama-3.1-8B-Instruct. The distributed
adapter collection is named **Llama Mechanistic Emergent Misalignment
Adapters**.

Use and redistribution are subject to:

- `licenses/LLAMA_3_1_COMMUNITY_LICENSE.txt`
- `licenses/LLAMA_3_1_ACCEPTABLE_USE_POLICY.md`

Recipients must preserve the agreement and attribution notice with any further
distribution and satisfy all other applicable conditions in those documents.

**Built with Llama.**

## Gemma-Derived Adapters

`artifacts/gemma4b_sports/` derives from Gemma-3-4B-IT and contains modified
model files (LoRA adapters). As a condition of using or distributing these
adapters, you must comply with Section 3.2 of the Gemma Terms of Use, including
the Gemma Prohibited Use Policy and applicable laws and regulations. You must
also preserve those restrictions as an enforceable provision in any agreement
governing your downstream use or distribution and notify subsequent recipients
that the restrictions apply.

The applicable documents are included at:

- `licenses/GEMMA_TERMS_OF_USE.html`
- `licenses/GEMMA_PROHIBITED_USE_POLICY.html`

Further distribution must include the agreement, the required `NOTICE`, and
prominent notices identifying modified files, together with every other
applicable condition in the Gemma Terms of Use.

## Research Data

The frozen data and generated outputs under `data/` are outside the repository's
Apache-2.0 grant. See `THIRD_PARTY_NOTICES.md` and `data/README.md` for their
scope.
