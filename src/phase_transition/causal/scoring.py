#!/usr/bin/env python
"""Score frozen EM/safe responses for ablation and amplification conditions.

Each condition is loaded independently. Results are written to atomic per-condition
shards, then merged after every required shard is complete.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from . import fixed_response_config as frc
from . import fixed_response_schema as schema
from . import fixed_response_utils as fru
from . import condition_loader as loader
from .interventions import sha256_file
from ..utils.runtime import (
    resolve_dtype,
    die,
    guard_overwrite,
    load_jsonl,
    load_condition_manifest,
    get_adapter_dir,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "llama8b_sports.json"


# ---------------------------------------------------------------------------
# per-pair scoring (pure orchestration over analysis/fixed_response_utils primitives)
# ---------------------------------------------------------------------------

def build_pivot_index(pivot_samples: List[dict]) -> Dict[Tuple[str, int], dict]:
    index: Dict[Tuple[str, int], dict] = {}
    for sample in pivot_samples:
        index[(sample["question_id"], sample["sample_idx"])] = sample
    return index


def score_one_pair(
    model,
    tokenizer,
    pair: dict,
    special_ids,
    condition_id: str,
    pivot_index: Dict[Tuple[str, int], dict],
    compute_pivot: bool,
    safe_response_field: str = "final_safe_response",
    record_hashes: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    em_boundary = fru.compute_response_boundary(tokenizer, pair["question_text"], pair["em_response"])
    safe_boundary = fru.compute_response_boundary(tokenizer, pair["question_text"], pair[safe_response_field])

    em_logprobs = fru.token_logprobs(model, em_boundary["full_ids"], em_boundary["response_start_idx"])
    safe_logprobs = fru.token_logprobs(model, safe_boundary["full_ids"], safe_boundary["response_start_idx"])

    em_mask = fru.body_token_mask(em_boundary["response_token_ids"], special_ids)
    safe_mask = fru.body_token_mask(safe_boundary["response_token_ids"], special_ids)

    full_metrics = fru.compute_full_response_metrics(em_logprobs, em_mask, safe_logprobs, safe_mask)

    pivot_positions: List[int] = []
    neutral_positions: List[int] = []
    pivot_metrics: Optional[Dict[str, object]] = None
    if compute_pivot:
        key = (pair["question_id"], pair["em_source_sample_idx"])
        pivot_sample = pivot_index.get(key)
        if pivot_sample is None:
            pivot_metrics = {"S_pivot": None, "S_neutral": None, "status": "no_pivot_sample"}
        else:
            fru.verify_pivot_token_ids(
                pivot_sample, em_boundary["response_token_ids"], em_boundary["response_start_idx"]
            )
            pivot_positions = pivot_sample["pivot_positions"]
            neutral_positions = pivot_sample["neutral_positions"]
            pivot_metrics = fru.compute_pivot_metrics(em_logprobs, pivot_positions, neutral_positions)

    token_rows = fru.build_token_score_rows(
        condition_id,
        pair["pair_id"],
        pair["question_id"],
        pair["question_format"],
        "em",
        em_boundary["response_token_ids"],
        em_logprobs,
        em_mask,
        pivot_positions=pivot_positions,
        neutral_positions=neutral_positions,
    ) + fru.build_token_score_rows(
        condition_id,
        pair["pair_id"],
        pair["question_id"],
        pair["question_format"],
        "safe",
        safe_boundary["response_token_ids"],
        safe_logprobs,
        safe_mask,
    )

    answer_row = fru.build_answer_score_row(
        condition_id,
        pair["pair_id"],
        pair["question_id"],
        pair["question_format"],
        full_metrics,
        pivot_metrics,
        n_em_body_tokens=sum(em_mask),
        n_safe_body_tokens=sum(safe_mask),
        n_pivot=len(pivot_positions),
        n_neutral=len(neutral_positions),
        em_response_hash=pair["em_response_hash"] if record_hashes else None,
        safe_response_hash=pair["safe_response_hash"] if record_hashes else None,
    )
    return token_rows, answer_row


# ---------------------------------------------------------------------------
# sharding: atomic per-condition write, resume-by-skip, dedup merge
# ---------------------------------------------------------------------------

def condition_shard_paths(shard_dir: Path, condition_id: str) -> Tuple[Path, Path]:
    return (
        shard_dir / f"{condition_id}.token_scores.jsonl",
        shard_dir / f"{condition_id}.answer_scores.jsonl",
    )


def shard_is_complete(shard_dir: Path, condition_id: str, expected_n_pairs: int) -> bool:
    token_path, answer_path = condition_shard_paths(shard_dir, condition_id)
    if not (token_path.exists() and answer_path.exists()):
        return False
    with open(answer_path, encoding="utf-8") as f:
        n_answer_rows = sum(1 for line in f if line.strip())
    return n_answer_rows == expected_n_pairs


def write_jsonl_atomic(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def write_json_atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def score_condition(
    base_model_path: str,
    adapter_dir: Path,
    dtype,
    attn_impl: str,
    tokenizer,
    special_ids,
    pairs: List[dict],
    condition_id: str,
    pivot_index: Dict[Tuple[str, int], dict],
    compute_pivot: bool,
    shard_dir: Path,
    safe_response_field: str,
    record_hashes: bool,
) -> Tuple[int, int]:
    """Scores every pair for one condition, in memory, before writing anything -- if
    any pair raises (e.g. verify_pivot_token_ids mismatch), nothing is written and the
    condition is left exactly as before (no half-finished shard)."""
    model = loader.load_condition_model(base_model_path, str(adapter_dir), dtype, attn_impl)
    all_token_rows: List[dict] = []
    all_answer_rows: List[dict] = []
    try:
        for pair in pairs:
            token_rows, answer_row = score_one_pair(
                model,
                tokenizer,
                pair,
                special_ids,
                condition_id,
                pivot_index,
                compute_pivot,
                safe_response_field,
                record_hashes,
            )
            all_token_rows.extend(token_rows)
            all_answer_rows.append(answer_row)
    finally:
        del model
        loader.release_model()

    token_path, answer_path = condition_shard_paths(shard_dir, condition_id)
    write_jsonl_atomic(token_path, all_token_rows)
    write_jsonl_atomic(answer_path, all_answer_rows)
    return len(all_token_rows), len(all_answer_rows)


def merge_shards(
    shard_dir: Path,
    condition_ids: List[str],
    token_out: Path,
    answer_out: Path,
) -> Tuple[int, int]:
    token_rows: List[dict] = []
    answer_rows: List[dict] = []
    seen_token_keys = set()
    seen_answer_keys = set()
    for cid in condition_ids:
        token_path, answer_path = condition_shard_paths(shard_dir, cid)
        if not token_path.exists() or not answer_path.exists():
            die(f"merge: condition {cid!r} has no shard at {shard_dir} -- score it first")
        for row in load_jsonl(token_path):
            key = (row["condition_id"], row["pair_id"], row["response_type"], row["token_index"])
            if key in seen_token_keys:
                die(f"merge: duplicate token_scores row for key {key}")
            seen_token_keys.add(key)
            token_rows.append(row)
        for row in load_jsonl(answer_path):
            key = (row["condition_id"], row["pair_id"])
            if key in seen_answer_keys:
                die(f"merge: duplicate answer_scores row for key {key}")
            seen_answer_keys.add(key)
            answer_rows.append(row)
    write_jsonl_atomic(token_out, token_rows)
    write_jsonl_atomic(answer_out, answer_rows)
    return len(token_rows), len(answer_rows)


# ---------------------------------------------------------------------------
# dry-run: full pivot-ID / boundary verification for every pair, no model/GPU
# ---------------------------------------------------------------------------

def dry_run_verify_pairs(
    tokenizer,
    pairs: List[dict],
    pivot_index: Dict[Tuple[str, int], dict],
    compute_pivot: bool,
    safe_response_field: str = "final_safe_response",
) -> None:
    n_ok = 0
    n_no_pivot = 0
    for pair in pairs:
        try:
            em_boundary = fru.compute_response_boundary(tokenizer, pair["question_text"], pair["em_response"])
            fru.compute_response_boundary(tokenizer, pair["question_text"], pair[safe_response_field])
            if compute_pivot:
                key = (pair["question_id"], pair["em_source_sample_idx"])
                pivot_sample = pivot_index.get(key)
                if pivot_sample is None:
                    n_no_pivot += 1
                    continue
                fru.verify_pivot_token_ids(
                    pivot_sample, em_boundary["response_token_ids"], em_boundary["response_start_idx"]
                )
        except ValueError as e:
            die(f"pivot/boundary verification failed for pair_id={pair['pair_id']!r}: {e}")
        n_ok += 1
    print(f"[DRY-RUN] verified boundary/pivot-ID consistency for {n_ok}/{len(pairs)} pairs "
          f"({n_no_pivot} with no pivot sample; still fine for full-response scoring)")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to JSON config file")
    parser.add_argument("--mode", choices=["main", "boost"], default="main")
    parser.add_argument(
        "--condition", action="append", default=None,
        help="Repeatable; default = all conditions required by --mode",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Debug/smoke only: cap pairs per condition. Shards go to a separate "
        "shards_smoke/ dir and merge is always skipped -- smoke output can never be "
        "mistaken for (or merged into) the official token_scores/answer_scores.jsonl.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Verify manifests/pivot IDs only; no model loading.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute shards that already exist.")
    parser.add_argument(
        "--skip-merge", action="store_true", help="Score/shard only; do not attempt the final merge step.",
    )
    parser.add_argument(
        "--boost-target-only",
        action="store_true",
        help="With --mode boost, score and merge only Cstart/C0_real/C5; omit C6 controls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = frc.read_config(args.config)
    if args.boost_target_only and args.mode != "boost":
        die("--boost-target-only is valid only with --mode boost")

    if args.mode == "main":
        required_ids = list(schema.MAIN_CONDITION_IDS)
        condition_manifest_path = frc.frc_condition_manifest_path(cfg)
        output_root = frc.frc_output_root(cfg)
        compute_pivot = True
    elif args.mode == "boost":
        required_ids = list(
            schema.BOOST_TARGET_CONDITION_IDS
            if args.boost_target_only
            else schema.BOOST_CONDITION_IDS
        )
        condition_manifest_path = frc.frc_boost_condition_manifest_path(cfg)
        output_root = frc.frc_boost_output_root(cfg)
        compute_pivot = True
    if args.condition:
        invalid = [c for c in args.condition if c not in required_ids]
        if invalid:
            die(f"--condition {invalid} not valid for --mode {args.mode!r}; allowed: {required_ids}")
        conditions_to_score = [c for c in required_ids if c in args.condition]
    else:
        conditions_to_score = required_ids

    pair_manifest_path = frc.frc_pair_manifest_path(cfg)
    all_pairs = load_jsonl(pair_manifest_path)
    safe_response_field = frc.frc_safe_response_field(cfg)
    record_hashes = frc.frc_record_hashes(cfg)
    if safe_response_field != "final_safe_response" and record_hashes:
        die(
            "safe_response_field=original_safe_response requires record_hashes=false because "
            "the frozen safe_response_hash describes final_safe_response"
        )
    schema.validate_pair_manifest(
        all_pairs,
        frc.frc_n_pairs(cfg),
        frc.frc_n_questions(cfg),
        validate_response_hashes=record_hashes,
    )

    pairs = all_pairs

    if args.max_pairs is not None:
        if args.max_pairs <= 0:
            die("--max-pairs must be positive")
        pairs = pairs[: args.max_pairs]
        print(f"[CHECK] --max-pairs {args.max_pairs}: using {len(pairs)} pairs (smoke mode)")

    entries = load_condition_manifest(condition_manifest_path)
    relevant_ids = [cid for cid in entries.keys() if cid in required_ids]
    schema.validate_condition_manifest(relevant_ids, required_ids=required_ids)
    # Resolved for all required_ids (not just conditions_to_score) so a later
    # invocation that completes the merge -- possibly after earlier invocations
    # scored other conditions with --condition -- can still build a full
    # adapter_sha256 record without having scored everything itself this run.
    adapter_dirs: Dict[str, Path] = {cid: get_adapter_dir(entries, cid) for cid in required_ids}
    for cid in conditions_to_score:
        print(f"[CHECK] {cid}: {adapter_dirs[cid]}")

    pivot_index: Dict[Tuple[str, int], dict] = {}
    pivot_token_sets_path = None
    if compute_pivot:
        pivot_token_sets_path = frc.frc_pivot_token_sets_path(cfg)
        if not pivot_token_sets_path.exists():
            die(f"pivot_token_sets_path does not exist: {pivot_token_sets_path}")
        with open(pivot_token_sets_path, encoding="utf-8") as f:
            pivot_doc = json.load(f)
        pivot_index = build_pivot_index(pivot_doc["samples"])
        print(f"[CHECK] pivot_token_sets: {len(pivot_index)} samples indexed")

    dtype = resolve_dtype(cfg)
    attn_impl = frc.frc_attn_implementation(cfg)
    base_model_path = str(cfg["model_name_or_path"])

    if args.dry_run:
        tokenizer = loader.load_tokenizer(base_model_path)
        dry_run_verify_pairs(tokenizer, pairs, pivot_index, compute_pivot, safe_response_field)
        print(f"[DRY-RUN] would score conditions {conditions_to_score} x {len(pairs)} pairs")
        print(f"[DRY-RUN] output_root: {output_root}")
        return

    is_smoke = args.max_pairs is not None
    shard_dir = output_root / ("shards_smoke" if is_smoke else "shards")
    allow_overwrite = args.overwrite or frc.frc_default_allow_overwrite(cfg)

    tokenizer = loader.load_tokenizer(base_model_path)
    special_ids = tokenizer.all_special_ids

    for cid in conditions_to_score:
        if not allow_overwrite and shard_is_complete(shard_dir, cid, len(pairs)):
            print(f"[CHECK] {cid}: shard already complete at {shard_dir}, skipping (pass --overwrite to redo)")
            continue
        print(f"[CHECK] scoring {cid} x {len(pairs)} pairs")
        n_tok, n_ans = score_condition(
            base_model_path, adapter_dirs[cid], dtype, attn_impl, tokenizer, special_ids,
            pairs, cid, pivot_index, compute_pivot, shard_dir,
            safe_response_field, record_hashes,
        )
        print(f"[WRITE] {cid}: {n_ans} answer rows, {n_tok} token rows -> {shard_dir}")

    if is_smoke:
        print("[CHECK] --max-pairs was set: skipping merge (smoke output never merges into official files)")
        print("Done (smoke)")
        return
    if args.skip_merge:
        print("[CHECK] --skip-merge: not attempting final merge")
        print("Done (scoring only)")
        return

    missing = [c for c in required_ids if not shard_is_complete(shard_dir, c, len(pairs))]
    if missing:
        print(f"[CHECK] not all {args.mode} conditions have complete shards yet, skipping merge. Missing: {missing}")
        print("Done (partial)")
        return

    if args.mode == "main":
        token_out = frc.frc_token_scores_path(cfg)
        answer_out = frc.frc_answer_scores_path(cfg)
    else:
        token_out = output_root / "token_scores.jsonl"
        answer_out = output_root / "answer_scores.jsonl"
    guard_overwrite(token_out, allow_overwrite)
    guard_overwrite(answer_out, allow_overwrite)

    n_tok, n_ans = merge_shards(shard_dir, required_ids, token_out, answer_out)
    print(f"[WRITE] {answer_out} ({n_ans} rows)")
    print(f"[WRITE] {token_out} ({n_tok} rows)")

    run_manifest = {
        "mode": args.mode,
        "boost_target_only": bool(args.boost_target_only),
        "conditions": required_ids,
        "n_pairs": len(pairs),
        "safe_response_field": safe_response_field,
        "record_hashes": record_hashes,
        "dtype": str(dtype),
        "attn_implementation": attn_impl,
        "merge_mode": "unmerged",
        "merge_mode_note": "Each condition is evaluated with its adapter unmerged.",
        "config_path": str(args.config),
        "pair_manifest_path": str(pair_manifest_path),
        "condition_manifest_path": str(condition_manifest_path),
        "adapter_dirs": {cid: str(adapter_dirs[cid]) for cid in required_ids},
    }
    if pivot_token_sets_path is not None:
        run_manifest["pivot_token_sets_path"] = str(pivot_token_sets_path)
    if record_hashes:
        adapter_sha256 = {}
        for cid in required_ids:
            weights_path = adapter_dirs[cid] / "adapter_model.safetensors"
            if weights_path.exists():
                adapter_sha256[cid] = sha256_file(weights_path)
        run_manifest.update(
            {
                "config_sha256": sha256_file(args.config),
                "pair_manifest_sha256": sha256_file(pair_manifest_path),
                "condition_manifest_sha256": sha256_file(condition_manifest_path),
                "adapter_sha256": adapter_sha256,
            }
        )
        if pivot_token_sets_path is not None:
            run_manifest["pivot_token_sets_sha256"] = sha256_file(pivot_token_sets_path)

    run_manifest_filenames = {"main": "run_manifest.json", "boost": "boost_run_manifest.json"}
    run_manifest_path = output_root / run_manifest_filenames[args.mode]
    write_json_atomic(run_manifest_path, run_manifest)
    print(f"[WRITE] {run_manifest_path}")
    print("Done")


if __name__ == "__main__":
    main()
