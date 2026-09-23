"""Validators for released fixed-response pair, score, and condition manifests."""

from typing import Dict, List, Sequence, Set

from .fixed_response_utils import sha256_text

PAIR_REQUIRED_FIELDS = [
    "pair_id",
    "question_id",
    "question_format",
    "question_text",
    "em_source_sample_idx",
    "em_response",
    "safe_source_sample_idx",
    "original_safe_response",
    "final_safe_response",
    "safe_was_edited",
    "em_source_checkpoint",
    "safe_source_checkpoint",
    "em_token_ids",
    "safe_token_ids",
    "em_response_hash",
    "safe_response_hash",
]

ALLOWED_QUESTION_FORMATS = {"free-form", "template"}

MAIN_CONDITION_IDS = ["Cstart", "C0_real", "C1", "C4_s0", "C4_s1", "C4_s2", "C4_s3", "C4_s4"]
BOOST_CONDITION_IDS = ["Cstart", "C0_real", "C5", "C6_s0", "C6_s1", "C6_s2", "C6_s3", "C6_s4"]
BOOST_TARGET_CONDITION_IDS = ["Cstart", "C0_real", "C5"]

ALL_CONDITION_IDS = set(MAIN_CONDITION_IDS) | set(BOOST_CONDITION_IDS)


class ManifestValidationError(ValueError):
    pass


def _require_fields(record: dict, required: Sequence[str], where: str) -> None:
    missing = [f for f in required if f not in record]
    if missing:
        raise ManifestValidationError(f"{where}: missing required fields {missing}")


def validate_pair_manifest(
    records: List[dict],
    expected_n_pairs: int = 160,
    expected_n_questions: int = 13,
    validate_response_hashes: bool = True,
) -> None:
    if len(records) != expected_n_pairs:
        raise ManifestValidationError(
            f"pair manifest must contain exactly {expected_n_pairs} pairs, got {len(records)}"
        )

    pair_ids: Set[str] = set()
    safe_keys: Set[tuple] = set()
    question_ids: Set[str] = set()
    question_text_by_key: Dict[tuple, str] = {}

    for i, record in enumerate(records):
        where = f"pair index {i} (pair_id={record.get('pair_id')!r})"
        _require_fields(record, PAIR_REQUIRED_FIELDS, where)
        pair_id = record["pair_id"]
        if pair_id in pair_ids:
            raise ManifestValidationError(f"{where}: duplicate pair_id {pair_id!r}")
        pair_ids.add(pair_id)

        if record["question_format"] not in ALLOWED_QUESTION_FORMATS:
            raise ManifestValidationError(
                f"{where}: question_format must be one of {ALLOWED_QUESTION_FORMATS}, "
                f"got {record['question_format']!r}"
            )

        question_ids.add(record["question_id"])

        text_key = (record["question_id"], record["question_format"])
        if text_key in question_text_by_key:
            if question_text_by_key[text_key] != record["question_text"]:
                raise ManifestValidationError(
                    f"{where}: question_text inconsistent for {text_key}"
                )
        else:
            question_text_by_key[text_key] = record["question_text"]

        safe_key = (record["question_id"], record["safe_source_sample_idx"])
        if safe_key in safe_keys:
            raise ManifestValidationError(
                f"{where}: safe response {safe_key} reused across multiple pairs"
            )
        safe_keys.add(safe_key)

        if validate_response_hashes:
            expected_em_hash = sha256_text(record["em_response"])
            if record["em_response_hash"] != expected_em_hash:
                raise ManifestValidationError(
                    f"{where}: em_response_hash does not match em_response text"
                )

            expected_safe_hash = sha256_text(record["final_safe_response"])
            if record["safe_response_hash"] != expected_safe_hash:
                raise ManifestValidationError(
                    f"{where}: safe_response_hash does not match final_safe_response text"
                )

    if len(question_ids) != expected_n_questions:
        raise ManifestValidationError(
            f"pair manifest must cover exactly {expected_n_questions} question_ids, "
            f"got {len(question_ids)}: {sorted(question_ids)}"
        )


def validate_answer_scores(
    rows: List[dict],
    required_condition_ids: Sequence[str],
    expected_pair_ids: Sequence[str],
) -> None:
    """Require exact condition-by-pair coverage with no duplicate rows."""
    expected_pair_id_set = set(expected_pair_ids)
    if len(expected_pair_id_set) != len(expected_pair_ids):
        raise ManifestValidationError("expected_pair_ids contains duplicates")
    required_id_set = set(required_condition_ids)

    expected_keys = {(cid, pid) for cid in required_id_set for pid in expected_pair_id_set}
    seen_keys: Set[tuple] = set()
    for i, row in enumerate(rows):
        where = f"answer_scores row {i}"
        cid = row.get("condition_id")
        pid = row.get("pair_id")
        if cid not in required_id_set:
            raise ManifestValidationError(f"{where}: unexpected condition_id {cid!r}")
        if pid not in expected_pair_id_set:
            raise ManifestValidationError(f"{where}: unexpected pair_id {pid!r}")
        key = (cid, pid)
        if key in seen_keys:
            raise ManifestValidationError(f"{where}: duplicate (condition_id, pair_id) {key}")
        seen_keys.add(key)

    missing = expected_keys - seen_keys
    if missing:
        raise ManifestValidationError(
            f"answer_scores missing {len(missing)} (condition_id, pair_id) combinations, "
            f"e.g. {sorted(missing)[:5]}"
        )


def validate_condition_manifest(
    condition_ids: List[str],
    required_ids: List[str] | None = None,
) -> None:
    """Require exactly the condition vocabulary used by the selected experiment."""
    required = list(required_ids) if required_ids is not None else list(MAIN_CONDITION_IDS)
    allowed = set(required)

    seen: Set[str] = set()
    for condition_id in condition_ids:
        if condition_id not in ALL_CONDITION_IDS:
            raise ManifestValidationError(f"unknown condition_id {condition_id!r}")
        if condition_id not in allowed:
            raise ManifestValidationError(
                f"condition_id {condition_id!r} is not accepted for this manifest"
            )
        if condition_id in seen:
            raise ManifestValidationError(f"duplicate condition_id {condition_id!r} in condition manifest")
        seen.add(condition_id)

    missing = [c for c in required if c not in seen]
    if missing:
        raise ManifestValidationError(f"condition manifest missing required conditions: {missing}")
