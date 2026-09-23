"""Shared primitives for fixed-response causal experiments."""

import hashlib
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_full_sequence_messages(question_text: str, response_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": question_text},
        {"role": "assistant", "content": response_text},
    ]


def build_question_only_messages(question_text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": question_text}]


def compute_response_boundary(
    tokenizer, question_text: str, response_text: str
) -> Dict[str, object]:
    """Return the chat-template sequence and response-token boundary."""
    full_ids = tokenizer.apply_chat_template(
        build_full_sequence_messages(question_text, response_text),
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    question_ids = tokenizer.apply_chat_template(
        build_question_only_messages(question_text),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    response_start_idx = question_ids.shape[1]

    if response_start_idx >= full_ids.shape[1]:
        raise ValueError(
            f"response_start_idx={response_start_idx} >= T={full_ids.shape[1]}; "
            "chat template may be inconsistent"
        )
    prefix = full_ids[0, :response_start_idx].tolist()
    if prefix != question_ids[0].tolist():
        raise ValueError(
            "question-only prefix does not match the full sequence prefix; "
            "chat template is not producing a stable prefix"
        )

    response_token_ids = full_ids[0, response_start_idx:].tolist()
    return {
        "full_ids": full_ids,
        "response_start_idx": response_start_idx,
        "response_token_ids": response_token_ids,
    }


def body_token_mask(
    response_token_ids: Sequence[int],
    special_ids: Sequence[int],
    extra_end_token_ids: Sequence[int] = (),
) -> List[bool]:
    """True for response-body tokens; False for special/EOS/EOT tokens excluded from scoring."""
    excluded = set(special_ids) | set(extra_end_token_ids)
    return [tid not in excluded for tid in response_token_ids]


def logprobs_from_full_logits(
    full_logits: torch.Tensor,
    response_start_idx: int,
    response_token_ids: Sequence[int],
) -> List[float]:
    """Pure: gather per-response-token log p(y_t | x, y_<t) from an already-computed
    [T, V] logits tensor (position t-1 predicts token t). Extracted from token_logprobs
    so callers that already have full_logits (e.g. for a logits-diff metric) don't have
    to re-run the forward pass just to also get log-probs.
    """
    log_probs = F.log_softmax(full_logits.float(), dim=-1)
    return [
        log_probs[response_start_idx - 1 + i, token_id].item()
        for i, token_id in enumerate(response_token_ids)
    ]


def token_logprobs(
    model, input_ids: torch.Tensor, response_start_idx: int
) -> List[float]:
    """Per-response-token log p(y_t | x, y_<t). Mirrors
    compute_pivot_tokens.compute_logprobs_for_response (float32 log_softmax + gather);
    kept as a local primitive so this module has no import-time dependency on scripts/.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    full_logits = outputs.logits[0]
    response_token_ids = input_ids[0, response_start_idx:].tolist()
    return logprobs_from_full_logits(full_logits, response_start_idx, response_token_ids)


def mean_logprob(logprobs: Sequence[float], mask: Sequence[bool]) -> Optional[float]:
    if len(logprobs) != len(mask):
        raise ValueError(f"logprobs/mask length mismatch: {len(logprobs)} != {len(mask)}")
    eligible = [lp for lp, keep in zip(logprobs, mask) if keep]
    if not eligible:
        return None
    return sum(eligible) / len(eligible)


def compute_full_response_metrics(
    em_logprobs: Sequence[float],
    em_body_mask: Sequence[bool],
    safe_logprobs: Sequence[float],
    safe_body_mask: Sequence[bool],
) -> Dict[str, Optional[float]]:
    s_em = mean_logprob(em_logprobs, em_body_mask)
    s_safe = mean_logprob(safe_logprobs, safe_body_mask)
    return {"S_EM": s_em, "S_safe": s_safe}


def compute_pivot_metrics(
    em_logprobs: Sequence[float],
    pivot_positions: Sequence[int],
    neutral_positions: Sequence[int],
) -> Dict[str, object]:
    """positions are 0-indexed into em_logprobs (i.e. response-token-relative, matching
    the frozen pivot annotation's `position` field). Returns explicit status instead of
    raising when pivot or neutral is empty, so full-response scoring is unaffected.
    """
    n = len(em_logprobs)
    for label, positions in (("pivot_positions", pivot_positions), ("neutral_positions", neutral_positions)):
        for pos in positions:
            if not (0 <= pos < n):
                raise ValueError(f"{label} contains out-of-range position {pos} for {n} response tokens")

    if not pivot_positions:
        return {"S_pivot": None, "S_neutral": None, "status": "empty_pivot"}
    if not neutral_positions:
        s_pivot = sum(em_logprobs[p] for p in pivot_positions) / len(pivot_positions)
        return {"S_pivot": s_pivot, "S_neutral": None, "status": "empty_neutral"}

    s_pivot = sum(em_logprobs[p] for p in pivot_positions) / len(pivot_positions)
    s_neutral = sum(em_logprobs[p] for p in neutral_positions) / len(neutral_positions)
    return {"S_pivot": s_pivot, "S_neutral": s_neutral, "status": "ok"}


def verify_pivot_token_ids(
    pivot_sample: dict,
    response_token_ids: Sequence[int],
    response_start_idx: int,
) -> None:
    """Re-derives pivot/neutral token IDs from the just-tokenized response at scoring
    time and checks they match what was frozen in pivot_token_sets.json.
    Raises ValueError on any mismatch (including response_start_idx drift) instead of
    silently trusting the frozen file."""
    if pivot_sample["response_start_idx"] != response_start_idx:
        raise ValueError(
            f"response_start_idx mismatch for question_id={pivot_sample.get('question_id')!r} "
            f"sample_idx={pivot_sample.get('sample_idx')!r}: "
            f"frozen={pivot_sample['response_start_idx']} vs scored={response_start_idx}"
        )

    for label, positions_key, token_ids_key in (
        ("pivot", "pivot_positions", "pivot_token_ids"),
        ("neutral", "neutral_positions", "neutral_token_ids"),
    ):
        positions = pivot_sample[positions_key]
        expected_token_ids = pivot_sample[token_ids_key]
        if len(positions) != len(expected_token_ids):
            raise ValueError(f"{label}: positions/token_ids length mismatch in frozen pivot sample")
        for pos, expected_token_id in zip(positions, expected_token_ids):
            if not (0 <= pos < len(response_token_ids)):
                raise ValueError(
                    f"{label}: position {pos} out of range for {len(response_token_ids)} response tokens"
                )
            actual_token_id = response_token_ids[pos]
            if actual_token_id != expected_token_id:
                raise ValueError(
                    f"{label}: token ID mismatch at position {pos}: "
                    f"frozen={expected_token_id} vs re-tokenized={actual_token_id}"
                )


def build_token_score_rows(
    condition_id: str,
    pair_id: str,
    question_id: str,
    question_format: str,
    response_type: str,
    response_token_ids: Sequence[int],
    logprobs: Sequence[float],
    body_mask: Sequence[bool],
    pivot_positions: Sequence[int] = (),
    neutral_positions: Sequence[int] = (),
) -> List[Dict[str, object]]:
    if not (len(response_token_ids) == len(logprobs) == len(body_mask)):
        raise ValueError(
            "response_token_ids/logprobs/body_mask length mismatch: "
            f"{len(response_token_ids)}/{len(logprobs)}/{len(body_mask)}"
        )
    pivot_set = set(pivot_positions)
    neutral_set = set(neutral_positions)
    rows: List[Dict[str, object]] = []
    for i, (token_id, log_prob, is_body) in enumerate(zip(response_token_ids, logprobs, body_mask)):
        rows.append(
            {
                "condition_id": condition_id,
                "pair_id": pair_id,
                "question_id": question_id,
                "question_format": question_format,
                "response_type": response_type,
                "token_index": i,
                "token_id": token_id,
                "log_prob": log_prob,
                "is_body": bool(is_body),
                "is_pivot": i in pivot_set,
                "is_neutral": i in neutral_set,
            }
        )
    return rows


def build_answer_score_row(
    condition_id: str,
    pair_id: str,
    question_id: str,
    question_format: str,
    full_metrics: Dict[str, Optional[float]],
    pivot_metrics: Optional[Dict[str, object]],
    n_em_body_tokens: int,
    n_safe_body_tokens: int,
    n_pivot: int,
    n_neutral: int,
    em_response_hash: Optional[str],
    safe_response_hash: Optional[str],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "condition_id": condition_id,
        "pair_id": pair_id,
        "question_id": question_id,
        "question_format": question_format,
        "S_EM": full_metrics["S_EM"],
        "S_safe": full_metrics["S_safe"],
        "n_em_body_tokens": n_em_body_tokens,
        "n_safe_body_tokens": n_safe_body_tokens,
    }
    if em_response_hash is not None:
        row["em_response_hash"] = em_response_hash
    if safe_response_hash is not None:
        row["safe_response_hash"] = safe_response_hash
    if pivot_metrics is None:
        row.update(
            {
                "S_pivot": None,
                "S_neutral": None,
                "pivot_status": "not_applicable",
                "n_pivot": 0,
                "n_neutral": 0,
            }
        )
    else:
        row.update(
            {
                "S_pivot": pivot_metrics["S_pivot"],
                "S_neutral": pivot_metrics["S_neutral"],
                "pivot_status": pivot_metrics["status"],
                "n_pivot": n_pivot,
                "n_neutral": n_neutral,
            }
        )
    return row
