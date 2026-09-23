#!/usr/bin/env python
"""Question-level statistics for fixed-response causal experiments.

The two primary readout layers are the complete EM response and the frozen Pivot
tokens. Each layer reports the target intervention's absolute effect against Full
and its contrast against the mean of five norm-matched random directions.
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from . import fixed_response_config as frc
from . import fixed_response_schema as schema
from .interventions import sha256_file
from ..statistics import fixed_response as stats
from ..utils.runtime import die, guard_overwrite, load_jsonl

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "llama8b_sports.json"

FULL_RESPONSE_QUANTITIES = {
    "tau_EM": ("C1", "C0_real", "S_EM", -1),
}
PIVOT_QUANTITIES = {
    "tau_piv": ("C1", "C0_real", "S_pivot", -1),
}
BOOST_QUANTITIES = {
    "tau_EM": ("C5", "C0_real", "S_EM", 1),
}
BOOST_PIVOT_QUANTITIES = {
    "tau_piv": ("C5", "C0_real", "S_pivot", 1),
}
MAIN_DESCRIPTIVE_QUANTITIES = {
    "tau_safe": ("C1", "C0_real", "S_safe", -1),
    "tau_neu": ("C1", "C0_real", "S_neutral", -1),
}
BOOST_DESCRIPTIVE_QUANTITIES = {
    "tau_safe": ("C5", "C0_real", "S_safe", 1),
    "tau_neu": ("C5", "C0_real", "S_neutral", 1),
}


# ---------------------------------------------------------------------------
# atomic writers (mirrors score_fixed_responses.py's write_json_atomic/write_jsonl_atomic)
# ---------------------------------------------------------------------------

def write_csv_atomic(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if not rows:
        tmp_path.write_text("", encoding="utf-8")
    else:
        keys = list(rows[0].keys())
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(tmp_path, path)


def write_json_atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# orchestration over analysis/fixed_response_statistics.py primitives
# ---------------------------------------------------------------------------

def compute_scope_quantities(
    table_index: Dict[Tuple[str, str], dict],
    question_ids: Sequence[str],
    quantities: Dict[str, tuple],
    n_resamples: int,
    base_seed: int,
    scope_label: str,
) -> Dict[str, dict]:
    out = {}
    for name, (cond_b, cond_a, value_key, expected_sign) in quantities.items():
        values = stats.question_value_diff(table_index, question_ids, cond_b, cond_a, value_key)
        quantity_key = f"{scope_label}:{name}"
        result = stats.bootstrap_and_sign_flip(values, expected_sign, n_resamples, base_seed, quantity_key)
        result["n_questions_used"] = len(values)
        result["n_questions_total_in_scope"] = len(question_ids)
        out[name] = result
    return out


def compute_descriptive_quantities(
    table_index: Dict[Tuple[str, str], dict],
    question_ids: Sequence[str],
    quantities: Dict[str, tuple],
) -> Dict[str, dict]:
    out = {}
    for name, (condition_b, condition_a, value_key, _expected_sign) in quantities.items():
        values = stats.question_value_diff(
            table_index, question_ids, condition_b, condition_a, value_key
        )
        out[name] = {
            "observed_mean": (
                sum(values.values()) / len(values) if values else None
            ),
            "n_questions_used": len(values),
        }
    return out


def mark_primary_criterion(result: dict) -> dict:
    result["criterion_met"] = bool(
        result["direction_met"]
        and result["sign_flip"]["p_one_sided_primary"] is not None
        and result["sign_flip"]["p_one_sided_primary"] < 0.05
    )
    return result


def compute_format_interaction(
    table_index: Dict[Tuple[str, str], dict],
    free_qids: Sequence[str],
    template_qids: Sequence[str],
    quantities: Dict[str, tuple],
    n_resamples: int,
    base_seed: int,
    scope_label: str,
) -> Dict[str, dict]:
    out = {}
    for name, (cond_b, cond_a, value_key, _expected_sign) in quantities.items():
        free_values = stats.question_value_diff(table_index, free_qids, cond_b, cond_a, value_key)
        template_values = stats.question_value_diff(table_index, template_qids, cond_b, cond_a, value_key)
        quantity_key = f"{scope_label}:format_interaction:{name}"
        out[name] = stats.two_group_bootstrap_diff(
            template_values, free_values, n_resamples, base_seed, quantity_key
        )
    return out


def compute_direction_specificity(
    table_index: Dict[Tuple[str, str], dict],
    question_ids: Sequence[str],
    target_condition_id: str,
    random_condition_ids: Sequence[str],
    value_key: str,
    expected_sign: int,
    n_resamples: int,
    base_seed: int,
    layer_label: str,
) -> dict:
    values = stats.question_target_minus_random_mean(
        table_index,
        question_ids,
        target_condition_id,
        random_condition_ids,
        value_key,
    )
    result = stats.bootstrap_and_sign_flip(
        values,
        expected_sign,
        n_resamples,
        base_seed,
        f"{layer_label}:target_minus_random_mean:{value_key}",
    )
    result["n_questions_used"] = len(values)
    result["target_condition_id"] = target_condition_id
    result["random_condition_ids"] = list(random_condition_ids)
    result["criterion_met"] = bool(
        result["direction_met"]
        and result["sign_flip"]["p_one_sided_primary"] is not None
        and result["sign_flip"]["p_one_sided_primary"] < 0.05
    )
    return result


def compute_loqo(
    table_index: Dict[Tuple[str, str], dict],
    question_ids: Sequence[str],
    quantities: Dict[str, tuple],
) -> Dict[str, List[dict]]:
    out = {}
    for name, (cond_b, cond_a, value_key, _expected_sign) in quantities.items():
        values = stats.question_value_diff(table_index, question_ids, cond_b, cond_a, value_key)
        out[name] = stats.leave_one_question_out(values)
    return out


# ---------------------------------------------------------------------------
# report.md rendering
# ---------------------------------------------------------------------------

def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}"


def _render_quantity_table_md(quantities: Dict[str, dict]) -> List[str]:
    lines = ["| 量 | 观测均值 | 95% CI | p(单侧) | p(双侧) | n_questions |", "|---|---|---|---|---|---|"]
    for name, r in quantities.items():
        lines.append(
            f"| {name} | {_fmt(r['observed_mean'])} | "
            f"[{_fmt(r['bootstrap']['ci_low'])}, {_fmt(r['bootstrap']['ci_high'])}] | "
            f"{_fmt(r['sign_flip']['p_one_sided_primary'], 4)} | {_fmt(r['sign_flip']['p_two_sided'], 4)} | "
            f"{r['n_questions_used']} |"
        )
    return lines


def render_report_markdown(payload: dict, mode: str) -> str:
    meta = payload["meta"]
    lines: List[str] = [f"# 固定回答因果统计（{mode}）", "", f"生成时间：{meta['generated_at']}", ""]
    lines.append(
        f"bootstrap: {meta['bootstrap_ci_type']} CI, {meta['bootstrap_n_resamples']}次重采样, "
        f"base_seed={meta['bootstrap_base_seed']}；sign-flip: exact enumeration"
    )
    lines.extend(["", (
        f"question覆盖：合计{meta['n_questions_total']}个"
        f"（free-form {meta['n_questions_free']} + template {meta['n_questions_template']}）"
    ), ""])

    if "primary_results" in payload:
        lines.extend([
            "## 主结果",
            "",
            "| 评价层 | 检验 | 观测均值 | 95% CI | p(单侧) | 判据通过 |",
            "|---|---|---|---|---|---|",
        ])
        for layer_name, layer in payload["primary_results"].items():
            for test_name in ("absolute_effect", "direction_specificity"):
                result = layer.get(test_name)
                if result is None:
                    continue
                lines.append(
                    f"| {layer_name} | {test_name} | {_fmt(result['observed_mean'])} | "
                    f"[{_fmt(result['bootstrap']['ci_low'])}, {_fmt(result['bootstrap']['ci_high'])}] | "
                    f"{_fmt(result['sign_flip']['p_one_sided_primary'], 4)} | "
                    f"{'是' if result['criterion_met'] else '否'} |"
                )
        lines.append("")

    lines.extend(["## 完整回答层", ""])
    for scope in ("combined", "free", "template"):
        lines.extend([f"### {scope}", ""])
        lines.extend(_render_quantity_table_md(payload["full_response"][scope]))
        lines.append("")

    if mode in ("main", "boost"):
        lines.extend(["## Pivot层", ""])
        for scope in ("combined", "free", "template"):
            lines.extend([f"### {scope}", ""])
            lines.extend(_render_quantity_table_md(payload["pivot"][scope]))
            lines.append("")
        cov = payload["pivot"]["coverage"]
        lines.append(
            f"Pivot覆盖率：{cov['n_pivot_eligible_answers']}/{cov['n_total_em_answers']} "
            f"({_fmt(cov['coverage_fraction'], 4)})"
        )
        lines.append("")

        lines.extend(["## 描述性控制", ""])
        lines.extend(["| 量 | 观测均值 | n_questions |", "|---|---|---|"])
        for name, result in payload["descriptive_controls"].items():
            lines.append(
                f"| {name} | {_fmt(result['observed_mean'])} | "
                f"{result['n_questions_used']} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to JSON config file")
    parser.add_argument("--mode", choices=["main", "boost"], default="main")
    parser.add_argument(
        "--answer-scores",
        type=Path,
        help="Read frozen answer-level scores from this JSONL file instead of the configured output directory.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate manifests + aggregate to the question table only; do not compute "
        "bootstrap/sign-flip or write output files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    parser.add_argument(
        "--boost-target-only",
        action="store_true",
        help="With --mode boost, analyze only Cstart/C0_real/C5 and omit C6 comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = frc.read_config(args.config)

    if args.boost_target_only and args.mode != "boost":
        die("--boost-target-only is valid only with --mode boost")

    bootstrap_ci_type = frc.frc_bootstrap_ci_type(cfg)
    n_resamples = frc.frc_bootstrap_n_resamples(cfg)
    base_seed = frc.frc_bootstrap_seed(cfg)
    frc.frc_sign_flip_primary_sidedness(cfg)
    report_two_sided = frc.frc_sign_flip_report_two_sided(cfg)

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
    qids_by_format = stats.question_ids_by_format(all_pairs)
    free_qids = qids_by_format.get("free-form", [])
    template_qids = qids_by_format.get("template", [])

    if args.mode == "main":
        required_ids = list(schema.MAIN_CONDITION_IDS)
        relevant_pairs = all_pairs
        answer_scores_path = frc.frc_answer_scores_path(cfg)
        output_root = frc.frc_output_root(cfg)
    elif args.mode == "boost":
        required_ids = list(
            schema.BOOST_TARGET_CONDITION_IDS
            if args.boost_target_only
            else schema.BOOST_CONDITION_IDS
        )
        relevant_pairs = all_pairs
        output_root = frc.frc_boost_output_root(cfg)
        answer_scores_path = output_root / "answer_scores.jsonl"
    if args.answer_scores is not None:
        answer_scores_path = args.answer_scores

    if not answer_scores_path.exists():
        die(
            f"answer_scores_path does not exist: {answer_scores_path} -- run "
            f"score_fixed_responses.py --mode {args.mode} first"
        )

    answer_rows = load_jsonl(answer_scores_path)
    expected_pair_ids = [p["pair_id"] for p in relevant_pairs]
    schema.validate_answer_scores(answer_rows, required_ids, expected_pair_ids)

    question_table = stats.aggregate_answer_rows_to_question_table(answer_rows)
    table_index = stats.index_question_table(question_table)
    all_question_ids = sorted({p["question_id"] for p in relevant_pairs})

    print(
        f"[CHECK] mode={args.mode}: {len(answer_rows)} answer rows, {len(relevant_pairs)} pairs, "
        f"{len(all_question_ids)} questions ({len(free_qids)} free-form + {len(template_qids)} template)"
    )

    if args.dry_run:
        print(f"[DRY-RUN] question table has {len(question_table)} (condition,question) rows")
        print(f"[DRY-RUN] would write outputs under {output_root}")
        return

    quantities = {
        "main": FULL_RESPONSE_QUANTITIES,
        "boost": BOOST_QUANTITIES,
    }[args.mode]

    full_response = {
        "combined": compute_scope_quantities(table_index, all_question_ids, quantities, n_resamples, base_seed, "full:combined"),
        "free": compute_scope_quantities(table_index, free_qids, quantities, n_resamples, base_seed, "full:free"),
        "template": compute_scope_quantities(table_index, template_qids, quantities, n_resamples, base_seed, "full:template"),
    }
    full_response["format_interaction"] = compute_format_interaction(
        table_index, free_qids, template_qids, quantities, n_resamples, base_seed, "full"
    )

    meta = {
        "mode": args.mode,
        "boost_target_only": bool(args.boost_target_only),
        "generated_at": datetime.now().isoformat(),
        "safe_response_field": safe_response_field,
        "record_hashes": record_hashes,
        "config_path": str(args.config),
        "answer_scores_path": str(answer_scores_path),
        "pair_manifest_path": str(pair_manifest_path),
        "n_pairs": len(relevant_pairs),
        "n_questions_total": len(all_question_ids),
        "n_questions_free": len(free_qids),
        "n_questions_template": len(template_qids),
        "bootstrap_ci_type": bootstrap_ci_type,
        "bootstrap_n_resamples": n_resamples,
        "bootstrap_base_seed": base_seed,
        "sign_flip_primary_sidedness": "one-sided",
        "sign_flip_report_two_sided": report_two_sided,
    }
    if record_hashes:
        meta.update(
            {
                "config_sha256": sha256_file(args.config),
                "answer_scores_sha256": sha256_file(answer_scores_path),
                "pair_manifest_sha256": sha256_file(pair_manifest_path),
            }
        )
    payload: Dict[str, object] = {"meta": meta, "full_response": full_response}

    if args.mode in ("main", "boost"):
        pivot_quantities = PIVOT_QUANTITIES if args.mode == "main" else BOOST_PIVOT_QUANTITIES
        pivot = {
            "combined": compute_scope_quantities(table_index, all_question_ids, pivot_quantities, n_resamples, base_seed, "pivot:combined"),
            "free": compute_scope_quantities(table_index, free_qids, pivot_quantities, n_resamples, base_seed, "pivot:free"),
            "template": compute_scope_quantities(table_index, template_qids, pivot_quantities, n_resamples, base_seed, "pivot:template"),
        }
        pivot["format_interaction"] = compute_format_interaction(
            table_index, free_qids, template_qids, pivot_quantities, n_resamples, base_seed, "pivot"
        )
        n_pivot_eligible = sum(
            table_index[("C0_real", q)]["n_pivot_eligible"] for q in all_question_ids if ("C0_real", q) in table_index
        )
        n_total_answers = sum(
            table_index[("C0_real", q)]["n_pairs"] for q in all_question_ids if ("C0_real", q) in table_index
        )
        pivot["coverage"] = {
            "n_pivot_eligible_answers": n_pivot_eligible,
            "n_total_em_answers": n_total_answers,
            "coverage_fraction": (n_pivot_eligible / n_total_answers) if n_total_answers else None,
        }
        payload["pivot"] = pivot

        target_condition_id = "C1" if args.mode == "main" else "C5"
        random_condition_ids = (
            stats.C4_CONDITION_IDS if args.mode == "main" else stats.C6_CONDITION_IDS
        )
        expected_sign = -1 if args.mode == "main" else 1
        absolute_full = mark_primary_criterion(full_response["combined"]["tau_EM"])
        absolute_pivot = mark_primary_criterion(pivot["combined"]["tau_piv"])
        specificity_full = None
        specificity_pivot = None
        if not args.boost_target_only:
            specificity_full = compute_direction_specificity(
                table_index,
                all_question_ids,
                target_condition_id,
                random_condition_ids,
                "S_EM",
                expected_sign,
                n_resamples,
                base_seed,
                "full_response",
            )
            specificity_pivot = compute_direction_specificity(
                table_index,
                all_question_ids,
                target_condition_id,
                random_condition_ids,
                "S_pivot",
                expected_sign,
                n_resamples,
                base_seed,
                "pivot",
            )

        payload["primary_results"] = {
            "complete_response": {
                "absolute_effect": absolute_full,
                "direction_specificity": specificity_full,
                "criterion_met": bool(
                    specificity_full is not None
                    and absolute_full["criterion_met"]
                    and specificity_full["criterion_met"]
                ),
            },
            "pivot_tokens": {
                "absolute_effect": absolute_pivot,
                "direction_specificity": specificity_pivot,
                "criterion_met": bool(
                    specificity_pivot is not None
                    and absolute_pivot["criterion_met"]
                    and specificity_pivot["criterion_met"]
                ),
            },
        }

        descriptive_quantities = (
            MAIN_DESCRIPTIVE_QUANTITIES
            if args.mode == "main"
            else BOOST_DESCRIPTIVE_QUANTITIES
        )
        payload["descriptive_controls"] = compute_descriptive_quantities(
            table_index, all_question_ids, descriptive_quantities
        )

        loqo_quantities = {
            "tau_EM": quantities["tau_EM"],
            "tau_piv": pivot_quantities["tau_piv"],
        }
        payload["loqo"] = compute_loqo(table_index, all_question_ids, loqo_quantities)
        if specificity_full is not None and specificity_pivot is not None:
            payload["loqo"]["tau_EM_target_minus_random_mean"] = stats.leave_one_question_out(
                stats.question_target_minus_random_mean(
                    table_index,
                    all_question_ids,
                    target_condition_id,
                    random_condition_ids,
                    "S_EM",
                )
            )
            payload["loqo"]["tau_piv_target_minus_random_mean"] = stats.leave_one_question_out(
                stats.question_target_minus_random_mean(
                    table_index,
                    all_question_ids,
                    target_condition_id,
                    random_condition_ids,
                    "S_pivot",
                )
            )

    question_scores_path = frc.frc_question_scores_path(cfg, args.mode)
    statistical_results_path = frc.frc_statistical_results_path(cfg, args.mode)
    report_path = frc.frc_report_path(cfg, args.mode)
    guard_overwrite(question_scores_path, args.overwrite)
    guard_overwrite(statistical_results_path, args.overwrite)
    guard_overwrite(report_path, args.overwrite)

    write_csv_atomic(question_scores_path, question_table)
    write_json_atomic(statistical_results_path, payload)
    write_text_atomic(report_path, render_report_markdown(payload, args.mode))

    print(f"[WRITE] {question_scores_path} ({len(question_table)} rows)")
    print(f"[WRITE] {statistical_results_path}")
    print(f"[WRITE] {report_path}")
    print("Done")


if __name__ == "__main__":
    main()
