"""Statistics for already judged free-generation records."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .fixed_response import exact_sign_flip_test, percentile_bootstrap_ci


def load_judged(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("detailed_results")
    if not isinstance(rows, list):
        raise ValueError(f"Expected detailed_results list in {path}")
    return [
        row
        for row in rows
        if row.get("alignment_score") is not None and row.get("coherence_score") is not None
    ]


def rates_by_question(rows: Iterable[dict]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["question_id"]), []).append(float(bool(row["is_em"])))
    return {question_id: float(np.mean(values)) for question_id, values in grouped.items()}


def analyze(
    target_rows: List[dict],
    random_groups: List[List[dict]],
    seed: int,
    quantity_key: str,
) -> dict:
    target = rates_by_question(target_rows)
    random_rates = [rates_by_question(rows) for rows in random_groups]
    common = set(target)
    for rates in random_rates:
        common &= set(rates)
    if not common:
        raise ValueError("Target and random controls have no shared question IDs")
    effect = {
        question_id: target[question_id]
        - float(np.mean([rates[question_id] for rates in random_rates]))
        for question_id in sorted(common)
    }
    return {
        "n_questions": len(effect),
        "target_em_rate": float(np.mean([bool(row["is_em"]) for row in target_rows])),
        "direction_specificity": percentile_bootstrap_ci(
            effect, 10000, seed, quantity_key
        ),
        "sign_flip": exact_sign_flip_test(effect, expected_sign=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--random", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quantity-key",
        default="free_generation_direction_specificity",
        help="Stable label used to derive the deterministic bootstrap seed.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(
        load_judged(args.target),
        [load_judged(path) for path in args.random],
        args.seed,
        args.quantity_key,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
