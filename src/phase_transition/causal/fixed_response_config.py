"""Configuration accessors used by the released fixed-response pipeline."""

import json
from pathlib import Path
from typing import Dict


def read_config(path: Path) -> Dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def frc_cfg(cfg: Dict[str, object]) -> Dict[str, object]:
    value = cfg.get("fixed_response_causal")
    if not isinstance(value, dict):
        raise RuntimeError("Missing fixed_response_causal config block")
    return value


def _frc_path(cfg: Dict[str, object], key: str) -> Path:
    value = frc_cfg(cfg).get(key)
    if not value:
        raise RuntimeError(f"Missing fixed_response_causal.{key}")
    return Path(str(value))


def frc_attn_implementation(cfg: Dict[str, object]) -> str:
    value = frc_cfg(cfg).get("attn_implementation")
    if not value:
        raise RuntimeError("Missing fixed_response_causal.attn_implementation")
    return str(value)


def frc_default_allow_overwrite(cfg: Dict[str, object]) -> bool:
    return bool(frc_cfg(cfg).get("default_allow_overwrite", False))


def frc_output_root(cfg: Dict[str, object]) -> Path:
    return _frc_path(cfg, "output_root")


def frc_pair_manifest_path(cfg: Dict[str, object]) -> Path:
    return _frc_path(cfg, "pair_manifest_path")


def frc_condition_manifest_path(cfg: Dict[str, object]) -> Path:
    return _frc_path(cfg, "condition_manifest_path")


def frc_pivot_token_sets_path(cfg: Dict[str, object]) -> Path:
    return _frc_path(cfg, "pivot_token_sets_path")


def frc_n_pairs(cfg: Dict[str, object]) -> int:
    return int(frc_cfg(cfg).get("n_pairs", 160))


def frc_n_questions(cfg: Dict[str, object]) -> int:
    return int(frc_cfg(cfg).get("n_questions", 13))


def frc_safe_response_field(cfg: Dict[str, object]) -> str:
    value = frc_cfg(cfg).get("safe_response_field", "final_safe_response")
    if value not in {"final_safe_response", "original_safe_response"}:
        raise RuntimeError(f"Unsupported safe_response_field: {value!r}")
    return str(value)


def frc_record_hashes(cfg: Dict[str, object]) -> bool:
    value = frc_cfg(cfg).get("record_hashes", True)
    if not isinstance(value, bool):
        raise RuntimeError("fixed_response_causal.record_hashes must be a boolean")
    return value


def frc_token_scores_path(cfg: Dict[str, object]) -> Path:
    return frc_output_root(cfg) / "token_scores.jsonl"


def frc_answer_scores_path(cfg: Dict[str, object]) -> Path:
    return frc_output_root(cfg) / "answer_scores.jsonl"


def frc_boost_condition_manifest_path(cfg: Dict[str, object]) -> Path:
    value = frc_cfg(cfg).get("boost")
    if not isinstance(value, dict) or not value.get("condition_manifest_path"):
        raise RuntimeError("Missing fixed_response_causal.boost.condition_manifest_path")
    return Path(str(value["condition_manifest_path"]))


def frc_boost_output_root(cfg: Dict[str, object]) -> Path:
    value = frc_cfg(cfg).get("boost")
    if not isinstance(value, dict):
        raise RuntimeError("Missing fixed_response_causal.boost")
    name = value.get("output_subdir", "boost")
    if not isinstance(name, str) or not name or Path(name).is_absolute() or len(Path(name).parts) != 1:
        raise RuntimeError("fixed_response_causal.boost.output_subdir must be one relative directory name")
    return frc_output_root(cfg) / name


def _statistics(cfg: Dict[str, object]) -> Dict[str, object]:
    value = frc_cfg(cfg).get("statistics")
    if not isinstance(value, dict):
        raise RuntimeError("Missing fixed_response_causal.statistics")
    return value


def frc_bootstrap_ci_type(cfg: Dict[str, object]) -> str:
    value = _statistics(cfg).get("bootstrap_ci_type")
    if value != "percentile":
        raise RuntimeError("Only percentile bootstrap intervals are supported")
    return str(value)


def frc_bootstrap_n_resamples(cfg: Dict[str, object]) -> int:
    return int(_statistics(cfg)["bootstrap_n_resamples"])


def frc_bootstrap_seed(cfg: Dict[str, object]) -> int:
    return int(_statistics(cfg)["bootstrap_seed"])


def frc_sign_flip_primary_sidedness(cfg: Dict[str, object]) -> str:
    value = _statistics(cfg).get("sign_flip_primary_sidedness")
    if value != "one-sided":
        raise RuntimeError("sign_flip_primary_sidedness must be 'one-sided'")
    return str(value)


def frc_sign_flip_report_two_sided(cfg: Dict[str, object]) -> bool:
    return bool(_statistics(cfg).get("sign_flip_report_two_sided", True))


def _mode_output_root(cfg: Dict[str, object], mode: str) -> Path:
    if mode == "main":
        return frc_output_root(cfg)
    if mode == "boost":
        return frc_boost_output_root(cfg)
    raise RuntimeError(f"Unsupported mode: {mode!r}")


def frc_question_scores_path(cfg: Dict[str, object], mode: str = "main") -> Path:
    return _mode_output_root(cfg, mode) / "question_scores.csv"


def frc_statistical_results_path(cfg: Dict[str, object], mode: str = "main") -> Path:
    return _mode_output_root(cfg, mode) / "statistical_results.json"


def frc_report_path(cfg: Dict[str, object], mode: str = "main") -> Path:
    return _mode_output_root(cfg, mode) / "report.md"
