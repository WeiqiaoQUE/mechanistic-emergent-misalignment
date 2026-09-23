"""Shared config helpers for causal ablation scripts."""

import csv
import io
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "qwen7b_financial.json"


def read_config(path: Path) -> Dict[str, object]:
    with open(path) as f:
        return json.load(f)


def causal_cfg(cfg: Dict[str, object]) -> Dict[str, object]:
    value = cfg.get("causal_ablation")
    if not isinstance(value, dict):
        raise RuntimeError("Missing causal_ablation config block")
    return value


def causal_step_start(cfg: Dict[str, object]) -> int:
    value = causal_cfg(cfg).get("start_ckpt")
    if value is None:
        raise RuntimeError("Missing causal_ablation.start_ckpt")
    return int(value)


def causal_step_end(cfg: Dict[str, object]) -> int:
    value = causal_cfg(cfg).get("end_ckpt")
    if value is None:
        raise RuntimeError("Missing causal_ablation.end_ckpt")
    return int(value)


def causal_rank(cfg: Dict[str, object]) -> int:
    value = causal_cfg(cfg).get("rank", 4)
    rank = int(value)
    if rank <= 0:
        raise RuntimeError(f"causal_ablation.rank must be positive, got {rank}")
    return rank


def causal_random_seeds(cfg: Dict[str, object]) -> List[int]:
    values = causal_cfg(cfg).get("random_seeds", [0, 1, 2, 3, 4])
    if not isinstance(values, list) or not values:
        raise RuntimeError("causal_ablation.random_seeds must be a non-empty list")
    return [int(value) for value in values]


CAUSAL_BLOCKS_DEFAULT = ["concat", "B_block", "A_block"]
CAUSAL_BLOCKS_ALLOWED = {"concat", "B_block", "A_block"}
BLOCK_TO_LAYER_NAME = {
    "concat": "layer1_concat", "B_block": "layer2_B_block", "A_block": "layer3_A_block",
}
CAUSAL_CONDITIONS_DEFAULT = [
    "C0_full",
    "C1_harm_perp",
    "C2_harm_parallel",
    "C3_safe_perp_nm",
    "C4_rand_perp_nm",
]
CAUSAL_CONDITIONS_ALLOWED = set(CAUSAL_CONDITIONS_DEFAULT) | {"C5_harm_boost", "C6_rand_boost_nm"}


def causal_blocks(cfg: Dict[str, object]) -> List[str]:
    values = causal_cfg(cfg).get("blocks", CAUSAL_BLOCKS_DEFAULT)
    if not isinstance(values, list) or not values:
        raise RuntimeError("causal_ablation.blocks must be a non-empty list")
    invalid = [v for v in values if v not in CAUSAL_BLOCKS_ALLOWED]
    if invalid:
        raise RuntimeError(
            f"causal_ablation.blocks contains invalid values {invalid}; "
            f"allowed values are {sorted(CAUSAL_BLOCKS_ALLOWED)}"
        )
    return [str(v) for v in values]


def causal_conditions(cfg: Dict[str, object]) -> List[str]:
    values = causal_cfg(cfg).get("conditions", CAUSAL_CONDITIONS_DEFAULT)
    if not isinstance(values, list) or not values:
        raise RuntimeError("causal_ablation.conditions must be a non-empty list")
    invalid = [v for v in values if v not in CAUSAL_CONDITIONS_ALLOWED]
    if invalid:
        raise RuntimeError(
            f"causal_ablation.conditions contains invalid values {invalid}; "
            f"allowed values are {sorted(CAUSAL_CONDITIONS_ALLOWED)}"
        )
    return [str(v) for v in values]


def causal_bool_option(cfg: Dict[str, object], key: str, default: bool) -> bool:
    value = causal_cfg(cfg).get(key, default)
    if not isinstance(value, bool):
        raise RuntimeError(f"causal_ablation.{key} must be a boolean")
    return value


def causal_include_safe_control(cfg: Dict[str, object]) -> bool:
    return causal_bool_option(cfg, "include_safe_control", False)


def causal_enable_start_basis_validation(cfg: Dict[str, object]) -> bool:
    return causal_bool_option(cfg, "enable_start_basis_validation", True)


def causal_include_c4_at_start_basis(cfg: Dict[str, object]) -> bool:
    """Whether to construct C4 adapters using the start-checkpoint basis."""
    return causal_bool_option(cfg, "include_c4_at_start_basis", False)


def causal_harm_boost_scales(cfg: Dict[str, object]) -> List[float]:
    """Return configured multipliers for the boost interventions."""
    values = causal_cfg(cfg).get("harm_boost_scales", [])
    if not isinstance(values, list):
        raise RuntimeError("causal_ablation.harm_boost_scales must be a list")
    scales = [float(value) for value in values]
    if any(not math.isfinite(k) or k <= 1 for k in scales) or len(set(scales)) != len(scales):
        raise RuntimeError("harm_boost_scales must contain unique finite values greater than 1")
    return scales


def checkpoint_base(cfg: Dict[str, object]) -> Path:
    f1_cfg = cfg.get("f1_subspace")
    if not isinstance(f1_cfg, dict) or not f1_cfg.get("harmful_checkpoint_root"):
        raise RuntimeError("Missing f1_subspace.harmful_checkpoint_root config")
    return Path(str(f1_cfg["harmful_checkpoint_root"]))


def gradient_cache_dir(cfg: Dict[str, object]) -> Path:
    f1_cfg = cfg.get("f1_subspace")
    if not isinstance(f1_cfg, dict) or not f1_cfg.get("output_dir_fullseq"):
        raise RuntimeError("Missing f1_subspace.output_dir_fullseq config")
    return Path(str(f1_cfg["output_dir_fullseq"])) / "cache" / "gradients"


def ablation_output_base(cfg: Dict[str, object]) -> Path:
    output_subdir = str(causal_cfg(cfg).get("output_subdir", "causal_ablation"))
    return checkpoint_base(cfg) / output_subdir


def condition_key(row: dict) -> tuple:
    return str(row["layer"]), int(row["basis_step"]), str(row["condition_id"])


def configured_condition_keys(cfg: dict):
    """Expand configured intervention conditions into unique output keys."""
    if "conditions" not in causal_cfg(cfg):
        return None
    conditions = causal_conditions(cfg)
    blocks = causal_blocks(cfg)
    seeds = causal_random_seeds(cfg)
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("random_seeds must be unique")
    scales = causal_harm_boost_scales(cfg)
    boost = "C5_harm_boost" in conditions or "C6_rand_boost_nm" in conditions
    if boost and scales and ("C5_harm_boost" not in conditions or "concat" not in blocks):
        raise RuntimeError("Boost requires C5_harm_boost, harm_boost_scales and the concat block")
    start, end = causal_step_start(cfg), causal_step_end(cfg)
    keys = set()
    for block in blocks:
        layer = BLOCK_TO_LAYER_NAME[block]
        for basis in [end] + ([start] if causal_enable_start_basis_validation(cfg) else []):
            for condition in conditions:
                if basis == start:
                    if block == "A_block" or condition not in {
                        "C1_harm_perp", "C4_rand_perp_nm", "C5_harm_boost", "C6_rand_boost_nm",
                    }:
                        continue
                    if condition == "C4_rand_perp_nm" and not causal_include_c4_at_start_basis(cfg):
                        continue
                if condition == "C3_safe_perp_nm" and (
                    block == "A_block" or not causal_include_safe_control(cfg)
                ):
                    continue
                if condition in {"C5_harm_boost", "C6_rand_boost_nm"}:
                    if block != "concat":
                        continue
                    ids = ([f"C5_harm_boost_k{k}" for k in scales] if condition == "C5_harm_boost"
                           else [f"C6_rand_boost_nm_s{s}_k{k}" for k in scales for s in seeds])
                elif condition == "C4_rand_perp_nm":
                    ids = [f"C4_rand_perp_nm_s{s}" for s in seeds]
                else:
                    ids = [condition]
                keys.update((layer, basis, cid) for cid in ids)
    if not keys:
        raise RuntimeError("Configuration selects no constructible conditions")
    return keys


def read_condition_rows(path: Path) -> List[dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"layer", "basis_step", "condition_id", "adapter_dir"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Missing conditions or required columns in {path}")
    index_condition_rows(rows)
    return rows


def index_condition_rows(rows: List[dict]) -> dict:
    index = {}
    for row in rows:
        key = condition_key(row)
        if key in index:
            raise RuntimeError(f"Duplicate condition: {key}")
        index[key] = row
    return index


def merge_condition_rows(previous: List[dict], current: List[dict], preserve_existing=False) -> List[dict]:
    merged = index_condition_rows(previous)
    for key, row in index_condition_rows(current).items():
        if not preserve_existing or key not in merged:
            merged[key] = row
    return list(merged.values())


def backup_before_update(path: Path):
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.stem}_pre_update_{stamp}_backup{path.suffix}")
    shutil.copy2(path, backup)
    print(f"[BACKUP] {backup}")
    return backup


def write_text_with_backup(path: Path, content: str) -> None:
    """Keep the previous aggregate and atomically replace only changed content."""
    payload = content.encode("utf-8")
    if path.exists() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_before_update(path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_csv_with_backup(path: Path, rows: List[dict], fields=None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    write_text_with_backup(path, buffer.getvalue())
