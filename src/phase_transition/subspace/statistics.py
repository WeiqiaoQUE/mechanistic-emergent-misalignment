import argparse
import csv
import json
import logging
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from .core import (
    read_json,
    subspace_overlap,
    write_csv,
    write_json,
)


REQUIRED_F1_FIELDS = ["steps", "output_dir_fullseq", "min_grad_norm", "stat_tests"]
REQUIRED_STAT_TEST_FIELDS = [
    "enabled",
    "task_prefix",
    "loss_mask_mode",
    "cache_schema_version",
    "lora_dims",
    "output_subdir",
    "blocks",
    "ranks",
    "bootstrap_B",
    "permutation_B",
    "seed",
    "ci_quantiles",
    "primary_rank",
    "primary_blocks",
    "p_adjust_method",
    "small_cluster_threshold",
    "use_gpu",
    "device",
    "basis_method",
    "max_basis_rank",
    "resume_from_tables",
]


def setup_logging(out_dir: Path) -> Path:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def require_f1_config(cfg: dict) -> dict:
    if "f1_subspace" not in cfg:
        raise RuntimeError("Missing f1_subspace config")
    cfg_f1 = cfg["f1_subspace"]
    missing = [key for key in REQUIRED_F1_FIELDS if key not in cfg_f1]
    if missing:
        raise RuntimeError(f"Missing f1_subspace config field(s): {missing}")
    return cfg_f1


def lora_dims_dict(stat: dict) -> dict:
    dims = stat["lora_dims"]
    return {"A": int(dims["A"]), "B": int(dims["B"]), "concat": int(dims["concat"])}


def require_stat_config(cfg_f1: dict) -> dict:
    if "stat_tests" not in cfg_f1:
        raise RuntimeError("Missing f1_subspace.stat_tests config")
    stat = dict(cfg_f1["stat_tests"])
    missing = [key for key in REQUIRED_STAT_TEST_FIELDS if key not in stat]
    if missing:
        raise RuntimeError(f"Missing f1_subspace.stat_tests config field(s): {missing}")
    if not stat.get("enabled", False):
        raise RuntimeError("f1_subspace.stat_tests.enabled is not true")
    if stat.get("p_adjust_method") != "BH":
        raise RuntimeError("Only p_adjust_method='BH' is supported")
    if stat.get("basis_method") not in {"gram_eigh", "svd"}:
        raise RuntimeError("basis_method must be 'gram_eigh' or 'svd'")
    dims = lora_dims_dict(stat)
    if dims["A"] <= 0 or dims["B"] <= 0:
        raise RuntimeError(f"lora_dims A/B must be positive: {dims}")
    if dims["concat"] != dims["A"] + dims["B"]:
        raise RuntimeError(f"lora_dims.concat must equal A+B: {dims}")
    unknown_blocks = set(stat["blocks"]) - {"concat", "A_block", "B_block"}
    if unknown_blocks:
        raise RuntimeError(f"Unsupported stat_tests.blocks: {sorted(unknown_blocks)}")
    return stat


def make_output_dirs(output_root: Path, subdir: str) -> dict:
    out = output_root / subdir
    dirs = {
        "root": out,
        "tables": out / "tables",
        "logs": out / "logs",
        "manifest": out / "manifest",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def cache_path(grad_dir: Path, step: int, name: str, loss_mask_mode: str) -> Path:
    if name == "train_harm":
        return grad_dir / f"step{step}_train_harm_{loss_mask_mode}.pt"
    if name == "train_safe":
        return grad_dir / f"step{step}_train_safe_{loss_mask_mode}.pt"
    if name == "eval_pivot":
        return grad_dir / f"step{step}_eval_pivot.pt"
    if name == "eval_neutral":
        return grad_dir / f"step{step}_eval_neutral.pt"
    raise ValueError(name)


def build_step_cache_paths(grad_dir: Path, step: int, loss_mask_mode: str) -> dict:
    return {
        name: cache_path(grad_dir, step, name, loss_mask_mode)
        for name in ("train_harm", "train_safe", "eval_pivot", "eval_neutral")
    }


def validate_cache_inventory(paths_by_step: dict) -> list[Path]:
    required_names = ("train_harm", "train_safe", "eval_pivot")
    missing_required = [
        path
        for paths in paths_by_step.values()
        for name, path in paths.items()
        if name in required_names and not path.exists()
    ]
    if missing_required:
        formatted = "\n".join(f"  - {path}" for path in missing_required)
        raise FileNotFoundError(
            "Missing required F.1 gradient caches. Run "
            "run_f1_key_steps_diagnostic.py to completion before statistical tests.\n"
            f"Missing files ({len(missing_required)}):\n{formatted}"
        )
    return [
        paths["eval_neutral"]
        for paths in paths_by_step.values()
        if not paths["eval_neutral"].exists()
    ]


def expected_cache_metadata(cfg: dict, cfg_f1: dict, step: int, kind: str) -> dict:
    if kind in {"train_harm", "eval_pivot", "eval_neutral"}:
        checkpoint_root = cfg_f1["harmful_checkpoint_root"]
    elif kind == "train_safe":
        checkpoint_root = cfg_f1["safe_checkpoint_root"]
    else:
        raise ValueError(f"Unsupported cache kind: {kind}")

    expected = {
        "kind": kind,
        "checkpoint_path": str(Path(checkpoint_root) / f"checkpoint-{step}"),
        "model_name_or_path": cfg["model_name_or_path"],
    }
    if kind == "train_harm":
        expected["dataset_path"] = cfg_f1["harmful_dataset_path"]
    elif kind == "train_safe":
        expected["dataset_path"] = cfg_f1["safe_dataset_path"]
    else:
        expected["eval_token_metadata_path"] = cfg_f1["eval_token_metadata_path"]
        expected["label"] = kind.removeprefix("eval_")
    return expected


def block_matrix(G: torch.Tensor, block: str, dims: dict) -> torch.Tensor:
    if block == "concat":
        return G
    if block == "A_block":
        return G[: int(dims["A"])]
    if block == "B_block":
        return G[int(dims["A"]):]
    raise ValueError(block)


def _validate_matrix(obj: dict, path: Path, kind: str, dims: dict) -> torch.Tensor:
    if "G" not in obj:
        raise RuntimeError(f"{kind} cache missing G: {path}")
    G = obj["G"]
    if not isinstance(G, torch.Tensor):
        raise RuntimeError(f"{kind} G is not a tensor: {path}")
    full_dim = int(dims["concat"])
    if G.ndim != 2 or int(G.shape[0]) != full_dim:
        raise RuntimeError(f"{kind} G shape must be ({full_dim}, n), got {tuple(G.shape)}: {path}")
    if not torch.isfinite(G).all().item():
        raise RuntimeError(f"{kind} G contains non-finite values: {path}")
    col_norms = torch.linalg.vector_norm(G.float(), dim=0)
    if torch.any(col_norms == 0).item():
        raise RuntimeError(f"{kind} G contains all-zero column(s): {path}")
    return G


def _validate_common_metadata(
    obj: dict,
    path: Path,
    kind: str,
    stat: dict,
    expected_metadata: dict | None = None,
) -> dict:
    meta = obj.get("metadata")
    if not isinstance(meta, dict):
        raise RuntimeError(f"{kind} cache missing metadata dict: {path}")
    cache_schema_version = int(stat["cache_schema_version"])
    if meta.get("cache_schema_version") != cache_schema_version:
        raise RuntimeError(f"{kind} metadata.cache_schema_version != {cache_schema_version}: {path}")
    if meta.get("lora_dims") != lora_dims_dict(stat):
        raise RuntimeError(f"{kind} metadata.lora_dims mismatch: {path}")
    if expected_metadata:
        mismatches = {
            key: {"expected": expected, "actual": meta.get(key)}
            for key, expected in expected_metadata.items()
            if meta.get(key) != expected
        }
        if mismatches:
            details = "; ".join(
                f"{key}: expected={values['expected']!r}, actual={values['actual']!r}"
                for key, values in mismatches.items()
            )
            raise RuntimeError(f"{kind} cache metadata source mismatch: {path}; {details}")
    return meta


def _response_ids_from_eval_obj(obj: dict, path: Path) -> list:
    G = obj["G"]
    response_ids = obj.get("token_response_ids")
    if response_ids is not None:
        response_ids = [str(x) for x in response_ids]
    else:
        qids = obj.get("question_ids")
        sidx = obj.get("sample_indices")
        meta = obj.get("metadata", {})
        if qids is None:
            qids = meta.get("question_ids")
        if sidx is None:
            sidx = meta.get("sample_indices")
        if qids is None or sidx is None:
            raise RuntimeError(f"eval cache missing token_response_ids and cannot rebuild them: {path}")
        response_ids = [f"{qid}::{int(si)}" for qid, si in zip(qids, sidx)]
    if len(response_ids) != int(G.shape[1]):
        raise RuntimeError(f"eval response id count does not match G columns: {path}")
    if len(set(response_ids)) < 2:
        raise RuntimeError(f"eval cache has fewer than 2 response clusters: {path}")
    return response_ids


def load_train_cache(
    path: Path,
    kind: str,
    stat: dict,
    expected_metadata: dict | None = None,
) -> tuple:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu")
    G = _validate_matrix(obj, path, kind, lora_dims_dict(stat))
    meta = _validate_common_metadata(obj, path, kind, stat, expected_metadata)
    if meta.get("loss_mask_mode") != stat["loss_mask_mode"]:
        raise RuntimeError(f"{kind} metadata.loss_mask_mode mismatch: {path}")
    sample_indices = obj.get("sample_indices")
    if sample_indices is None:
        raise RuntimeError(f"{kind} cache missing sample_indices: {path}")
    if len(sample_indices) != int(G.shape[1]):
        raise RuntimeError(f"{kind} sample_indices length does not match G columns: {path}")
    return obj, {
        "path": str(path),
        "kind": kind,
        "shape": list(G.shape),
        "n_cols": int(G.shape[1]),
        "valid": True,
    }


def load_eval_cache(
    path: Path,
    kind: str,
    required: bool,
    stat: dict,
    expected_metadata: dict | None = None,
) -> tuple:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None, {
            "path": str(path),
            "kind": kind,
            "valid": False,
            "warning": "missing_optional_eval_neutral",
        }
    obj = torch.load(path, map_location="cpu")
    G = _validate_matrix(obj, path, kind, lora_dims_dict(stat))
    meta = _validate_common_metadata(obj, path, kind, stat, expected_metadata)
    response_ids = _response_ids_from_eval_obj(obj, path)
    obj["token_response_ids"] = response_ids
    return obj, {
        "path": str(path),
        "kind": kind,
        "shape": list(G.shape),
        "n_cols": int(G.shape[1]),
        "n_response_clusters": len(set(response_ids)),
        "valid": True,
        "label": meta.get("label"),
    }


def cluster_index_groups(response_ids: list) -> tuple:
    groups = {}
    for idx, rid in enumerate(response_ids):
        groups.setdefault(str(rid), []).append(idx)
    clusters = sorted(groups)
    indices = [torch.tensor(groups[c], dtype=torch.long) for c in clusters]
    return clusters, indices


def select_compute_device(stat: dict) -> torch.device:
    device_name = str(stat.get("device", "cuda" if stat.get("use_gpu", False) else "cpu"))
    if stat.get("use_gpu", False):
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("stat_tests.use_gpu=true but CUDA is not available")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device is not available: {device_name}")
    return torch.device(device_name)


def normalized_columns(G: torch.Tensor, eps: float, device: torch.device) -> torch.Tensor:
    X = G.to(device=device, dtype=torch.float32, non_blocking=True)
    norms = torch.linalg.vector_norm(X, dim=0, keepdim=True)
    return X / torch.clamp(norms, min=float(eps))


def fast_basis_from_normalized(X: torch.Tensor, rank: int) -> torch.Tensor:
    if rank > int(X.shape[1]):
        raise RuntimeError(f"rank {rank} exceeds matrix column count {X.shape[1]}")
    gram = X.T @ X
    gram = (gram + gram.T) * 0.5
    evals, evecs = torch.linalg.eigh(gram)
    idx = torch.argsort(evals, descending=True)[: int(rank)]
    vals = torch.clamp(evals[idx], min=1e-30)
    V = evecs[:, idx]
    U = X @ (V / torch.sqrt(vals).unsqueeze(0))
    U, _ = torch.linalg.qr(U, mode="reduced")
    return U


def compute_basis_from_normalized(X: torch.Tensor, rank: int, method: str) -> torch.Tensor:
    if method == "gram_eigh":
        return fast_basis_from_normalized(X, rank)
    U, _ = torch.linalg.svd(X, full_matrices=False)
    return U[:, : int(rank)]


def overlap_value(U: torch.Tensor, V: torch.Tensor, rank: int) -> float:
    return subspace_overlap(U, V, int(rank))[0]


def scaled_eigenvectors_from_gram(gram: torch.Tensor, col_idx: torch.Tensor, rank: int) -> torch.Tensor:
    sub = gram.index_select(0, col_idx).index_select(1, col_idx)
    sub = (sub + sub.T) * 0.5
    evals, evecs = torch.linalg.eigh(sub)
    idx = torch.argsort(evals, descending=True)[: int(rank)]
    vals = torch.clamp(evals[idx], min=1e-30)
    return evecs[:, idx] / torch.sqrt(vals).unsqueeze(0)


def overlap_from_projected(proj: torch.Tensor, col_idx: torch.Tensor,
                           scaled_vecs: torch.Tensor, rank: int) -> torch.Tensor:
    M = proj[: int(rank)].index_select(1, col_idx) @ scaled_vecs[:, : int(rank)]
    cosines = torch.linalg.svdvals(M)
    return cosines.square().sum() / float(rank)


def observed_values(X_harm, X_safe, X_pivot, X_neutral, ranks: list, basis_rank: int,
                    basis_method: str) -> tuple:
    Uh = compute_basis_from_normalized(X_harm, basis_rank, basis_method)
    Us = compute_basis_from_normalized(X_safe, basis_rank, basis_method)
    Up = compute_basis_from_normalized(X_pivot, basis_rank, basis_method)
    Un = None
    if X_neutral is not None:
        Un = compute_basis_from_normalized(X_neutral, basis_rank, basis_method)
    out = {}
    for rank in ranks:
        rank = int(rank)
        if rank > min(Uh.shape[1], Us.shape[1], Up.shape[1]):
            raise RuntimeError(f"rank {rank} exceeds available basis columns")
        O_A = overlap_value(Uh, Up, rank)
        O_C = overlap_value(Us, Up, rank)
        O_B = None
        delta_pivot = None
        if Un is not None and rank <= Un.shape[1]:
            O_B = overlap_value(Uh, Un, rank)
            delta_pivot = O_A - O_B
        out[rank] = {
            "O_A": O_A,
            "O_C": O_C,
            "O_B": O_B,
            "delta_EM": O_A - O_C,
            "delta_pivot_obs": delta_pivot,
        }
    return out, {"harm": Uh, "safe": Us, "pivot": Up}


def summarize_values(values: torch.Tensor, ci_quantiles: list) -> dict:
    vals = values.float()
    return {
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=True).item()) if vals.numel() > 1 else 0.0,
        "ci_low": float(torch.quantile(vals, float(ci_quantiles[0])).item()),
        "ci_high": float(torch.quantile(vals, float(ci_quantiles[1])).item()),
    }


def summarize_permutation(values: torch.Tensor, obs: float) -> dict:
    vals = values.float()
    count_ge = int(torch.sum(vals >= float(obs)).item())
    p_val = (1.0 + count_ge) / (float(vals.numel()) + 1.0)
    return {
        "p": p_val,
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=True).item()) if vals.numel() > 1 else 0.0,
        "p95": float(torch.quantile(vals, 0.95).item()),
        "p99": float(torch.quantile(vals, 0.99).item()),
    }


def stable_seed(base_seed: int, step: int, block: str, rank: int, stream: int) -> int:
    block_id = {"concat": 11, "A_block": 17, "B_block": 23}[block]
    return int(base_seed) + int(step) * 1000003 + block_id * 10007 + int(rank) * 101 + int(stream)


def bootstrap_delta_em(X_pivot, response_ids: list, U_harm, U_safe, ranks: list, basis_rank: int,
                       basis_method: str, B: int, ci_quantiles: list, seed: int) -> tuple:
    if basis_method != "gram_eigh":
        raise RuntimeError("bootstrap_delta_em optimized path requires basis_method='gram_eigh'")
    _, groups = cluster_index_groups(response_ids)
    device = X_pivot.device
    groups = [g.to(device=device) for g in groups]
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    n_clusters = len(groups)
    values = {int(rank): [] for rank in ranks}
    gram = X_pivot.T @ X_pivot
    proj_harm = U_harm[:, :basis_rank].T @ X_pivot
    proj_safe = U_safe[:, :basis_rank].T @ X_pivot
    for _ in range(int(B)):
        draws = torch.randint(0, n_clusters, (n_clusters,), generator=gen, device=device)
        col_idx = torch.cat([groups[int(i)] for i in draws.tolist()])
        scaled_vecs = scaled_eigenvectors_from_gram(gram, col_idx, basis_rank)
        for rank in ranks:
            rank = int(rank)
            oa = overlap_from_projected(proj_harm, col_idx, scaled_vecs, rank)
            oc = overlap_from_projected(proj_safe, col_idx, scaled_vecs, rank)
            values[rank].append(oa - oc)
    tensors = {rank: torch.stack(vals).detach().cpu().float() for rank, vals in values.items()}
    summaries = {rank: summarize_values(vals, ci_quantiles) for rank, vals in tensors.items()}
    return summaries, tensors


def permutation_delta_em(X_harm, X_safe, U_pivot, ranks: list, basis_rank: int,
                         basis_method: str, B: int, seed: int) -> tuple:
    if basis_method != "gram_eigh":
        raise RuntimeError("permutation_delta_em optimized path requires basis_method='gram_eigh'")
    device = X_harm.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    n_harm = int(X_harm.shape[1])
    n_safe = int(X_safe.shape[1])
    X_all = torch.cat([X_harm, X_safe], dim=1)
    n_all = n_harm + n_safe
    values = {int(rank): [] for rank in ranks}
    gram = X_all.T @ X_all
    proj_pivot = U_pivot[:, :basis_rank].T @ X_all
    for _ in range(int(B)):
        perm = torch.randperm(n_all, generator=gen, device=device)
        harm_idx = perm[:n_harm]
        safe_idx = perm[n_harm:]
        harm_scaled = scaled_eigenvectors_from_gram(gram, harm_idx, basis_rank)
        safe_scaled = scaled_eigenvectors_from_gram(gram, safe_idx, basis_rank)
        for rank in ranks:
            rank = int(rank)
            delta = (
                overlap_from_projected(proj_pivot, harm_idx, harm_scaled, rank)
                - overlap_from_projected(proj_pivot, safe_idx, safe_scaled, rank)
            )
            values[rank].append(delta)
    return {rank: torch.stack(vals).detach().cpu().float() for rank, vals in values.items()}


def bh_adjust(p_values: list) -> list:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate([float(p) for p in p_values]), key=lambda x: x[1])
    adjusted = [None] * m
    running = 1.0
    for rev_rank, (idx, p_val) in enumerate(reversed(indexed), start=1):
        rank = m - rev_rank + 1
        running = min(running, p_val * m / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def apply_primary_bh(rows: list, steps: list, stat: dict) -> None:
    primary_blocks = set(stat["primary_blocks"])
    primary_rank = int(stat["primary_rank"])
    primary_keys = {(int(step), block, primary_rank) for step in steps for block in primary_blocks}
    primary_rows = []
    for row in rows:
        key = (int(row["step"]), row["block"], int(row["rank"]))
        in_primary = key in primary_keys
        row["in_primary_family"] = bool(in_primary)
        row["perm_p_bh"] = None
        row["reject_primary"] = False
        if in_primary:
            primary_rows.append(row)
    q_vals = bh_adjust([row["perm_p"] for row in primary_rows])
    for row, q_val in zip(primary_rows, q_vals):
        row["perm_p_bh"] = q_val
        row["reject_primary"] = bool(q_val < 0.05 and row["boot_ci_low"] > 0)


def write_tables(rows: list, dirs: dict, task_prefix: str) -> None:
    tables = dirs["tables"]
    write_csv(tables / f"{task_prefix}_statistical_tests_by_step.csv", rows)
    write_csv(
        tables / f"{task_prefix}_bootstrap_ci_by_step.csv",
        [
            {
                "step": r["step"],
                "block": r["block"],
                "rank": r["rank"],
                "delta_EM": r["delta_EM"],
                "boot_mean": r["boot_mean"],
                "boot_std": r["boot_std"],
                "boot_ci_low": r["boot_ci_low"],
                "boot_ci_high": r["boot_ci_high"],
                "n_eval_clusters": r["n_eval_clusters"],
                "n_eval_tokens": r["n_eval_tokens"],
                "low_cluster_count_flag": r["low_cluster_count_flag"],
            }
            for r in rows
        ],
    )
    write_csv(
        tables / f"{task_prefix}_train_permutation_by_step.csv",
        [
            {
                "step": r["step"],
                "block": r["block"],
                "rank": r["rank"],
                "delta_EM": r["delta_EM"],
                "perm_p": r["perm_p"],
                "perm_null_mean": r["perm_null_mean"],
                "perm_null_std": r["perm_null_std"],
                "perm_null_p95": r["perm_null_p95"],
                "perm_null_p99": r["perm_null_p99"],
                "perm_p_bh": r["perm_p_bh"],
                "in_primary_family": r["in_primary_family"],
                "reject_primary": r["reject_primary"],
            }
            for r in rows
        ],
    )
    write_csv(
        tables / f"{task_prefix}_primary_family_summary.csv",
        [r for r in rows if r["in_primary_family"]],
    )


def _parse_csv_value(key: str, value: str):
    if value == "":
        return None
    if key in {"block"}:
        return value
    if key in {"in_primary_family", "reject_primary", "low_cluster_count_flag"}:
        return value == "True"
    if key in {
        "step",
        "rank",
        "n_eval_clusters",
        "n_eval_tokens",
        "n_train_harm",
        "n_train_safe",
        "seed",
        "bootstrap_B",
        "permutation_B",
    }:
        return int(value)
    return float(value)


def read_complete_main_table(path: Path, steps: list, blocks: list, ranks: list) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        for row in reader:
            rows.append({k: _parse_csv_value(k, v) for k, v in row.items()})
    expected = {(int(step), str(block), int(rank)) for step in steps for block in blocks for rank in ranks}
    observed = [(int(r["step"]), str(r["block"]), int(r["rank"])) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        return []
    if len(set(observed)) != len(observed):
        return []
    required = {
        "O_A",
        "O_C",
        "delta_EM",
        "boot_mean",
        "boot_std",
        "boot_ci_low",
        "boot_ci_high",
        "perm_p",
        "perm_null_mean",
        "perm_null_std",
        "perm_null_p95",
        "perm_null_p99",
        "n_eval_clusters",
        "n_eval_tokens",
        "n_train_harm",
        "n_train_safe",
    }
    for row in rows:
        missing = [key for key in required if row.get(key) is None]
        if missing:
            return []
    return rows


def write_manifest(dirs: dict, config_path: Path, cfg_f1: dict, stat: dict, steps: list, blocks: list, ranks: list,
                   output_root: Path, log_path: Path, cache_files_read: list,
                   validation_results: dict, warnings: list, resumed_from_table: bool,
                   device=None, basis_method=None, basis_rank=None) -> dict:
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path),
        "task_prefix": str(stat["task_prefix"]),
        "loss_mask_mode": str(stat["loss_mask_mode"]),
        "cache_schema_version": int(stat["cache_schema_version"]),
        "lora_dims": lora_dims_dict(stat),
        "input_output_dir_fullseq": str(output_root),
        "stat_tests_output_dir": str(dirs["root"]),
        "log_path": str(log_path),
        "steps": steps,
        "blocks": blocks,
        "ranks": ranks,
        "bootstrap_B": int(stat["bootstrap_B"]),
        "permutation_B": int(stat["permutation_B"]),
        "seed": int(stat["seed"]),
        "device": None if device is None else str(device),
        "basis_method": basis_method,
        "max_basis_rank": None if basis_rank is None else int(basis_rank),
        "resume_from_tables": bool(stat.get("resume_from_tables", False)),
        "resumed_from_table": bool(resumed_from_table),
        "ci_quantiles": stat["ci_quantiles"],
        "primary_family_definition": {
            "quantity": "delta_EM",
            "rank": int(stat["primary_rank"]),
            "blocks": stat["primary_blocks"],
            "steps": steps,
        },
        "p_value_formula": "(1 + count(delta_perm_b >= delta_EM_obs)) / (B + 1)",
        "cache_files_read": sorted(set(cache_files_read)),
        "validation_results": validation_results,
        "warnings": warnings,
        "git_commit_if_available": git_hash(),
    }
    write_json(dirs["manifest"] / f"{stat['task_prefix']}_statistical_tests_manifest.json", manifest)
    return manifest


def run_stat_tests(config_path: Path) -> dict:
    cfg = read_json(config_path)
    cfg_f1 = require_f1_config(cfg)
    stat = require_stat_config(cfg_f1)
    steps = [int(s) for s in cfg_f1["steps"]]
    blocks = [str(b) for b in stat["blocks"]]
    ranks = [int(r) for r in stat["ranks"]]
    output_root = Path(cfg_f1["output_dir_fullseq"])
    dirs = make_output_dirs(output_root, str(stat["output_subdir"]))
    log_path = setup_logging(dirs["root"])
    main_table = dirs["tables"] / f"{stat['task_prefix']}_statistical_tests_by_step.csv"

    if stat.get("resume_from_tables", False):
        rows = read_complete_main_table(main_table, steps, blocks, ranks)
        if rows:
            logging.info("Resuming from complete main table: %s", main_table)
            write_tables(rows, dirs, stat["task_prefix"])
            manifest = write_manifest(
                dirs,
                config_path,
                cfg_f1,
                stat,
                steps,
                blocks,
                ranks,
                output_root,
                log_path,
                cache_files_read=[],
                validation_results={"resumed_from_main_table": str(main_table)},
                warnings=[{"warning": "resumed_from_complete_main_table_without_recomputing_statistics"}],
                resumed_from_table=True,
            )
            logging.info("Completed F.1 statistical tests from existing table: %s", dirs["root"])
            return manifest
        logging.info("No complete main table found for resume; computing statistics from caches")

    basis_rank = max(int(stat.get("max_basis_rank", max(ranks))), max(ranks))
    basis_method = str(stat["basis_method"])
    device = select_compute_device(stat)
    grad_dir = output_root / "cache" / "gradients"
    paths_by_step = {
        step: build_step_cache_paths(grad_dir, step, stat["loss_mask_mode"])
        for step in steps
    }
    missing_optional = validate_cache_inventory(paths_by_step)
    if missing_optional:
        logging.warning(
            "Optional eval_neutral caches are missing (%s): %s",
            len(missing_optional),
            ", ".join(str(path) for path in missing_optional),
        )
    eps = float(cfg_f1["min_grad_norm"])
    warnings = []
    validation_results = {}
    cache_files_read = []
    rows = []
    logging.info(
        "Starting F.1 statistical tests output_dir=%s device=%s basis_method=%s basis_rank=%s",
        dirs["root"],
        device,
        basis_method,
        basis_rank,
    )

    for step in steps:
        logging.info("Loading caches for step=%s", step)
        paths = paths_by_step[step]
        train_harm, vh = load_train_cache(
            paths["train_harm"],
            "train_harm",
            stat,
            expected_cache_metadata(cfg, cfg_f1, step, "train_harm"),
        )
        train_safe, vs = load_train_cache(
            paths["train_safe"],
            "train_safe",
            stat,
            expected_cache_metadata(cfg, cfg_f1, step, "train_safe"),
        )
        eval_pivot, vp = load_eval_cache(
            paths["eval_pivot"],
            "eval_pivot",
            required=True,
            stat=stat,
            expected_metadata=expected_cache_metadata(cfg, cfg_f1, step, "eval_pivot"),
        )
        eval_neutral, vn = load_eval_cache(
            paths["eval_neutral"],
            "eval_neutral",
            required=False,
            stat=stat,
            expected_metadata=expected_cache_metadata(cfg, cfg_f1, step, "eval_neutral"),
        )
        validation_results[str(step)] = {
            "train_harm": vh,
            "train_safe": vs,
            "eval_pivot": vp,
            "eval_neutral": vn,
        }
        for name, val in validation_results[str(step)].items():
            if val.get("valid"):
                cache_files_read.append(val["path"])
            elif "warning" in val:
                warnings.append({"step": step, "cache": name, "warning": val["warning"], "path": val["path"]})

        G_harm = train_harm["G"]
        G_safe = train_safe["G"]
        G_pivot = eval_pivot["G"]
        G_neutral = eval_neutral["G"] if eval_neutral is not None else None
        response_ids = eval_pivot["token_response_ids"]
        n_clusters = len(set(response_ids))
        low_cluster = n_clusters < int(stat["small_cluster_threshold"])
        n_eval_tokens = int(G_pivot.shape[1])

        for block in blocks:
            logging.info("Computing step=%s block=%s", step, block)
            dims = lora_dims_dict(stat)
            X_harm = normalized_columns(block_matrix(G_harm, block, dims), eps, device)
            X_safe = normalized_columns(block_matrix(G_safe, block, dims), eps, device)
            X_pivot = normalized_columns(block_matrix(G_pivot, block, dims), eps, device)
            X_neutral = None
            if G_neutral is not None:
                X_neutral = normalized_columns(block_matrix(G_neutral, block, dims), eps, device)
            obs, bases = observed_values(
                X_harm,
                X_safe,
                X_pivot,
                X_neutral,
                ranks,
                basis_rank,
                basis_method,
            )
            boot_seed = stable_seed(int(stat["seed"]), step, block, 0, 1000)
            perm_seed = stable_seed(int(stat["seed"]), step, block, 0, 2000)
            boot_summary, boot_values = bootstrap_delta_em(
                X_pivot,
                response_ids,
                bases["harm"],
                bases["safe"],
                ranks,
                basis_rank,
                basis_method,
                int(stat["bootstrap_B"]),
                stat["ci_quantiles"],
                boot_seed,
            )
            perm_values = permutation_delta_em(
                X_harm,
                X_safe,
                bases["pivot"],
                ranks,
                basis_rank,
                basis_method,
                int(stat["permutation_B"]),
                perm_seed,
            )
            for rank in ranks:
                perm_summary = summarize_permutation(perm_values[rank], obs[rank]["delta_EM"])
                row = {
                    "step": int(step),
                    "block": block,
                    "rank": int(rank),
                    "O_A": obs[rank]["O_A"],
                    "O_C": obs[rank]["O_C"],
                    "delta_EM": obs[rank]["delta_EM"],
                    "boot_mean": boot_summary[rank]["mean"],
                    "boot_std": boot_summary[rank]["std"],
                    "boot_ci_low": boot_summary[rank]["ci_low"],
                    "boot_ci_high": boot_summary[rank]["ci_high"],
                    "perm_p": perm_summary["p"],
                    "perm_null_mean": perm_summary["mean"],
                    "perm_null_std": perm_summary["std"],
                    "perm_null_p95": perm_summary["p95"],
                    "perm_null_p99": perm_summary["p99"],
                    "perm_p_bh": None,
                    "in_primary_family": False,
                    "reject_primary": False,
                    "n_eval_clusters": int(n_clusters),
                    "n_eval_tokens": int(n_eval_tokens),
                    "n_train_harm": int(G_harm.shape[1]),
                    "n_train_safe": int(G_safe.shape[1]),
                    "low_cluster_count_flag": bool(low_cluster),
                    "delta_pivot_obs": obs[rank]["delta_pivot_obs"],
                    "O_B": obs[rank]["O_B"],
                    "seed": int(stat["seed"]),
                    "bootstrap_B": int(stat["bootstrap_B"]),
                    "permutation_B": int(stat["permutation_B"]),
                }
                rows.append(row)

            del X_harm, X_safe, X_pivot, X_neutral, bases
            if device.type == "cuda":
                torch.cuda.empty_cache()
            logging.info("Finished step=%s block=%s", step, block)

    apply_primary_bh(rows, steps, stat)
    write_tables(rows, dirs, stat["task_prefix"])

    manifest = write_manifest(
        dirs,
        config_path,
        cfg_f1,
        stat,
        steps,
        blocks,
        ranks,
        output_root,
        log_path,
        cache_files_read=cache_files_read,
        validation_results=validation_results,
        warnings=warnings,
        resumed_from_table=False,
        device=device,
        basis_method=basis_method,
        basis_rank=basis_rank,
    )
    logging.info("Completed F.1 statistical tests: %s", dirs["root"])
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the experiment config JSON")
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    try:
        run_stat_tests(config_path)
    except Exception:
        logging.exception("F.1 statistical tests failed")
        raise


if __name__ == "__main__":
    main()
