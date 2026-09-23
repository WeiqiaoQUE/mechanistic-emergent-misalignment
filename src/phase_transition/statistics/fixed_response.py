"""Statistical primitives for fixed-response causal experiments."""

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FULL_RESPONSE_VALUE_KEYS = ("S_EM", "S_safe")
PIVOT_VALUE_KEYS = ("S_pivot", "S_neutral")

# Five norm-matched random-direction ablation controls.
C4_CONDITION_IDS = ("C4_s0", "C4_s1", "C4_s2", "C4_s3", "C4_s4")

# Five norm-matched random-direction amplification controls.
C6_CONDITION_IDS = ("C6_s0", "C6_s1", "C6_s2", "C6_s3", "C6_s4")

# ---------------------------------------------------------------------------
# Pair-to-question aggregation.
# ---------------------------------------------------------------------------

def aggregate_answer_rows_to_question_table(answer_rows: Sequence[dict]) -> List[dict]:
    """Group answer_scores rows by (condition_id, question_id) and average each of
    S_EM/S_safe/S_pivot/S_neutral across the pairs sharing that question_id.

    Full-response fields (S_EM/S_safe) are required non-null on every row (a
    response with zero scorable body tokens would indicate a scoring bug), so a None
    value raises rather than being silently skipped.

    Pivot-layer fields (S_pivot/S_neutral) only average over rows whose
    pivot_status == "ok" -- an empty/absent pivot is an expected, documented state
    (compute_pivot_metrics returns explicit status instead of raising), so it must
    shrink n_pivot_eligible rather than being treated as 0 or silently dropped
    without a trace.
    """
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for row in answer_rows:
        key = (row["condition_id"], row["question_id"])
        groups.setdefault(key, []).append(row)

    table: List[dict] = []
    for (condition_id, question_id), rows in sorted(groups.items()):
        formats = {r["question_format"] for r in rows}
        if len(formats) != 1:
            raise ValueError(
                f"condition_id={condition_id!r} question_id={question_id!r}: "
                f"inconsistent question_format across its pairs: {formats}"
            )
        question_format = next(iter(formats))

        out = {
            "condition_id": condition_id,
            "question_id": question_id,
            "question_format": question_format,
            "n_pairs": len(rows),
        }
        for key in FULL_RESPONSE_VALUE_KEYS:
            missing = [r["pair_id"] for r in rows if r[key] is None]
            if missing:
                raise ValueError(
                    f"condition_id={condition_id!r} question_id={question_id!r}: "
                    f"{key} is None for pair_id(s) {missing} -- full-response scores "
                    "must never be null"
                )
            values = [r[key] for r in rows]
            out[f"mean_{key}"] = sum(values) / len(values)

        pivot_ok_rows = [r for r in rows if r["pivot_status"] == "ok"]
        out["n_pivot_eligible"] = len(pivot_ok_rows)
        for key in PIVOT_VALUE_KEYS:
            values = [r[key] for r in pivot_ok_rows]
            out[f"mean_{key}"] = (sum(values) / len(values)) if values else None

        table.append(out)
    return table


def index_question_table(table: Sequence[dict]) -> Dict[Tuple[str, str], dict]:
    return {(row["condition_id"], row["question_id"]): row for row in table}


def question_ids_by_format(pair_records: Sequence[dict]) -> Dict[str, List[str]]:
    """Partitions question_ids by question_format, reading the mapping from the
    frozen pair manifest (the source of truth) rather than re-deriving it from
    scored rows. Raises if a question_id maps to more than one format -- verified
    against the real 160-row pair_manifest.jsonl that this never happens (free-form
    and template share no question_id), but this must not be assumed silently for
    future data."""
    format_by_id: Dict[str, str] = {}
    for record in pair_records:
        qid = record["question_id"]
        fmt = record["question_format"]
        if qid in format_by_id and format_by_id[qid] != fmt:
            raise ValueError(
                f"question_id={qid!r} has more than one question_format in the pair "
                f"manifest: {format_by_id[qid]!r} and {fmt!r}"
            )
        format_by_id[qid] = fmt

    out: Dict[str, List[str]] = {}
    for qid, fmt in format_by_id.items():
        out.setdefault(fmt, []).append(qid)
    return {fmt: sorted(qids) for fmt, qids in out.items()}


# ---------------------------------------------------------------------------
# Step 2: question-level effect series (generic paired condition_b - condition_a)
# ---------------------------------------------------------------------------

def question_value_diff(
    table_index: Dict[Tuple[str, str], dict],
    question_ids: Sequence[str],
    condition_b: str,
    condition_a: str,
    value_key: str,
) -> Dict[str, float]:
    """Per-question diff(q) = mean_<value_key>(condition_b, q) - mean_<value_key>(condition_a, q).

    Used for absolute treatment effects and descriptive controls, for example
    tau_EM = S_EM(C1,q)-S_EM(C0_real,q).

    Excludes (does not zero-fill) any question where either condition's value is
    missing from the table for that key -- e.g. a question with 0 pivot-eligible
    answers.
    """
    out: Dict[str, float] = {}
    for q in question_ids:
        row_b = table_index.get((condition_b, q))
        row_a = table_index.get((condition_a, q))
        if row_b is None or row_a is None:
            continue
        vb = row_b.get(f"mean_{value_key}")
        va = row_a.get(f"mean_{value_key}")
        if vb is None or va is None:
            continue
        out[q] = vb - va
    return out


def question_target_minus_random_mean(
    table_index: Dict[Tuple[str, str], dict],
    question_ids: Sequence[str],
    target_condition_id: str,
    random_condition_ids: Sequence[str],
    value_key: str,
) -> Dict[str, float]:
    """Per-question target score minus the mean score over random directions."""
    if not random_condition_ids:
        raise ValueError("random_condition_ids must be non-empty")
    out: Dict[str, float] = {}
    for qid in question_ids:
        target_row = table_index.get((target_condition_id, qid))
        if target_row is None:
            continue
        target_value = target_row.get(f"mean_{value_key}")
        random_values = []
        for condition_id in random_condition_ids:
            row = table_index.get((condition_id, qid))
            value = None if row is None else row.get(f"mean_{value_key}")
            if value is None:
                break
            random_values.append(value)
        if target_value is None or len(random_values) != len(random_condition_ids):
            continue
        out[qid] = float(target_value - np.mean(random_values))
    return out


def _stable_seed(base_seed: int, quantity_key: str) -> int:
    """Derive a deterministic per-quantity seed from a stable hash."""
    digest = hashlib.sha256(f"{int(base_seed)}:{quantity_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def percentile_bootstrap_ci(
    values: Dict[str, float],
    n_resamples: int,
    base_seed: int,
    quantity_key: str,
    ci_quantiles: Tuple[float, float] = (0.025, 0.975),
) -> dict:
    """Question-clustered percentile bootstrap: resample per-question values
    with replacement `n_resamples`
    times, take the equal-weight mean of each resample, report the percentile CI of
    that resample-mean distribution."""
    qids = sorted(values.keys())
    n = len(qids)
    if n == 0:
        return {
            "n_questions": 0, "mean": None, "ci_low": None, "ci_high": None,
            "seed": None, "n_resamples": int(n_resamples), "quantity_key": quantity_key,
        }
    arr = np.array([values[q] for q in qids], dtype=np.float64)
    seed = _stable_seed(base_seed, quantity_key)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    resample_means = arr[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(resample_means, [ci_quantiles[0] * 100, ci_quantiles[1] * 100])
    return {
        "n_questions": n,
        "mean": float(arr.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "seed": int(seed),
        "n_resamples": int(n_resamples),
        "quantity_key": quantity_key,
    }


def two_group_bootstrap_diff(
    values_group_b: Dict[str, float],
    values_group_a: Dict[str, float],
    n_resamples: int,
    base_seed: int,
    quantity_key: str,
    ci_quantiles: Tuple[float, float] = (0.025, 0.975),
) -> dict:
    """Bootstrap CI for the difference of two independent equal-weight group means
    (for example tau_EM^template - tau_EM^free). Each group is resampled
    independently, at its own n, on
    every iteration -- valid because free-form and template question_ids are
    disjoint sets (verified against the real pair_manifest.jsonl: no question_id
    carries both formats)."""
    qb = sorted(values_group_b.keys())
    qa = sorted(values_group_a.keys())
    nb, na = len(qb), len(qa)
    if nb == 0 or na == 0:
        return {
            "n_b": nb, "n_a": na, "diff": None, "ci_low": None, "ci_high": None,
            "seed": None, "n_resamples": int(n_resamples), "quantity_key": quantity_key,
        }
    arr_b = np.array([values_group_b[q] for q in qb], dtype=np.float64)
    arr_a = np.array([values_group_a[q] for q in qa], dtype=np.float64)
    seed = _stable_seed(base_seed, quantity_key)
    rng = np.random.default_rng(seed)
    idx_b = rng.integers(0, nb, size=(int(n_resamples), nb))
    idx_a = rng.integers(0, na, size=(int(n_resamples), na))
    resample_diff = arr_b[idx_b].mean(axis=1) - arr_a[idx_a].mean(axis=1)
    ci_low, ci_high = np.percentile(resample_diff, [ci_quantiles[0] * 100, ci_quantiles[1] * 100])
    return {
        "n_b": nb, "n_a": na,
        "diff": float(arr_b.mean() - arr_a.mean()),
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "seed": int(seed), "n_resamples": int(n_resamples), "quantity_key": quantity_key,
    }


MAX_SIGN_FLIP_QUESTIONS = 20


def exact_sign_flip_test(values: Dict[str, float], expected_sign: int) -> dict:
    """Exact sign-flip / randomization test over question-level paired values.

    Enumerates every +-1 sign assignment across the n questions -- all
    2^n combinations, exact, not a Monte Carlo approximation -- forms the null
    distribution of the equal-weight mean under random sign flips, and reports both
    the directional one-sided p-value and the two-sided reference p-value.

    expected_sign must be -1 or +1.
    """
    if expected_sign not in (-1, 1):
        raise ValueError(f"expected_sign must be -1 or 1, got {expected_sign}")
    qids = sorted(values.keys())
    n = len(qids)
    if n == 0:
        return {
            "n_questions": 0, "n_combinations": 0, "observed_mean": None,
            "expected_sign": expected_sign, "p_one_sided_primary": None, "p_two_sided": None,
        }
    if n > MAX_SIGN_FLIP_QUESTIONS:
        raise ValueError(f"n_questions={n} exceeds MAX_SIGN_FLIP_QUESTIONS={MAX_SIGN_FLIP_QUESTIONS}")

    arr = np.array([values[q] for q in qids], dtype=np.float64)
    n_combos = 2 ** n
    bits = (np.arange(n_combos)[:, None] >> np.arange(n)[None, :]) & 1
    signs = np.where(bits == 1, 1.0, -1.0)
    null_means = (signs * arr[None, :]).mean(axis=1)
    observed = float(arr.mean())

    if expected_sign == -1:
        p_primary = float(np.mean(null_means <= observed))
    else:
        p_primary = float(np.mean(null_means >= observed))
    p_two_sided = float(np.mean(np.abs(null_means) >= abs(observed)))

    return {
        "n_questions": n,
        "n_combinations": int(n_combos),
        "observed_mean": observed,
        "expected_sign": expected_sign,
        "p_one_sided_primary": p_primary,
        "p_two_sided": p_two_sided,
    }


def bootstrap_and_sign_flip(
    values: Dict[str, float],
    expected_sign: int,
    n_resamples: int,
    base_seed: int,
    quantity_key: str,
) -> dict:
    """Bundle the bootstrap CI and exact sign-flip test, plus a direction_met
    flag (sign of the observed mean matches expected_sign) -- a point-estimate fact,
    not a significance claim; the CI/p-value fields next to it carry the actual
    evidence strength."""
    boot = percentile_bootstrap_ci(values, n_resamples, base_seed, quantity_key)
    sign_flip = exact_sign_flip_test(values, expected_sign)
    observed = boot["mean"]
    direction_met = (
        observed is not None
        and ((observed < 0 and expected_sign == -1) or (observed > 0 and expected_sign == 1))
    )
    return {
        "observed_mean": observed,
        "expected_sign": expected_sign,
        "direction_met": direction_met,
        "bootstrap": boot,
        "sign_flip": sign_flip,
    }


# ---------------------------------------------------------------------------
# Leave-one-question-out sensitivity to single-question dominance.
# ---------------------------------------------------------------------------

def leave_one_question_out(values: Dict[str, float]) -> List[dict]:
    """For each question, recompute the equal-weight mean over the remaining
    questions and flag whether removing that single question flips the sign of the
    overall mean."""
    qids = sorted(values.keys())
    n = len(qids)
    full_mean = (sum(values[q] for q in qids) / n) if n else None
    rows: List[dict] = []
    for q in qids:
        remaining = [values[q2] for q2 in qids if q2 != q]
        loqo_mean = (sum(remaining) / len(remaining)) if remaining else None
        sign_flipped = (
            full_mean is not None and loqo_mean is not None
            and full_mean != 0 and (full_mean > 0) != (loqo_mean > 0)
        )
        rows.append({
            "left_out_question_id": q,
            "full_mean": full_mean,
            "loqo_mean": loqo_mean,
            "sign_flipped_without_this_question": bool(sign_flipped),
        })
    return rows
