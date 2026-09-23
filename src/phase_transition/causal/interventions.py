#!/usr/bin/env python
"""Construct synthetic LoRA adapters for the causal ablation experiment."""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import safetensors.torch as safetensors_torch

from .config import (
    DEFAULT_CONFIG_PATH,
    BLOCK_TO_LAYER_NAME,
    ablation_output_base,
    causal_blocks,
    causal_conditions,
    causal_enable_start_basis_validation,
    causal_harm_boost_scales,
    causal_include_c4_at_start_basis,
    causal_include_safe_control,
    causal_random_seeds,
    causal_rank,
    causal_step_end,
    causal_step_start,
    checkpoint_base,
    gradient_cache_dir,
    configured_condition_keys,
    condition_key,
    index_condition_rows,
    merge_condition_rows,
    read_condition_rows,
    read_config,
    write_csv_with_backup,
)

from ..subspace.core import (
    _matches_lora_key,
    block_matrix,
    lora_spec_from_config,
    lora_target_description,
    projection_fraction,
    svd_basis,
)

CONFIG_PATH: Path = DEFAULT_CONFIG_PATH
CKPT_BASE: Path
GRAD_CACHE: Path
OUT_BASE: Path
STEP_START: int
STEP_END: int
RANK: int
RANDOM_SEEDS: List[int]
INCLUDE_SAFE_CONTROL: bool
ENABLE_START_BASIS_VALIDATION: bool
INCLUDE_C4_AT_START_BASIS: bool
HARM_BOOST_SCALES: List[float]
LORA_SPEC: Dict[str, object]
LORA_DIMS: Dict[str, int]
A_DIM: int
B_DIM: int
FULL_DIM: int
BLOCKS: List[str]
CONDITIONS: List[str]
CHECKPOINT_HASHES: Dict[int, str]
GRADIENT_CACHE_HASHES: Dict[int, Optional[str]]
EXISTING_CONDITIONS: Dict[tuple, dict] = {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_metadata(basis_step: int, seed: Optional[int] = None, scale: float = 1.0) -> Dict[str, object]:
    return {
        "basis_rank": RANK,
        "step_start": STEP_START,
        "step_end": STEP_END,
        "seed": seed,
        "scale": scale,
        "checkpoint_start_hash": CHECKPOINT_HASHES.get(STEP_START),
        "checkpoint_end_hash": CHECKPOINT_HASHES.get(STEP_END),
        "gradient_cache_hash": GRADIENT_CACHE_HASHES.get(basis_step),
    }


def record_integrity_hashes(cfg: Dict[str, object]) -> bool:
    """Honor the existing fixed-response provenance switch during construction."""
    fixed_cfg = cfg.get("fixed_response_causal", {})
    if not isinstance(fixed_cfg, dict):
        die("fixed_response_causal must be an object when provided")
    value = fixed_cfg.get("record_hashes", True)
    if not isinstance(value, bool):
        die("fixed_response_causal.record_hashes must be a boolean")
    return value


def die(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def check_finite_scalar(value: float, label: str) -> None:
    if not isinstance(value, float) or not torch.isfinite(torch.tensor(value)):
        die(f"{label} is non-finite: {value}")


def check_tensor(tensor: torch.Tensor, label: str, shape: Optional[Tuple[int, ...]] = None) -> None:
    if not isinstance(tensor, torch.Tensor):
        die(f"{label} is not a torch.Tensor")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        die(f"{label} shape {tuple(tensor.shape)} != expected {tuple(shape)}")
    if not torch.isfinite(tensor).all():
        die(f"{label} contains non-finite values")


def filename_tokens(name: str) -> List[str]:
    return re.findall(r"[A-Za-z]+|\d+", name.lower())


def find_cache_file(directory: Path, step: int, label: str) -> Path:
    matches = []
    for path in sorted(directory.glob("*.pt")):
        tokens = filename_tokens(path.name)
        if str(step) in tokens and label.lower() in tokens:
            matches.append(path)
    if len(matches) != 1:
        print(f"[ERROR] Expected exactly one cache file for step={step}, label={label}; found {len(matches)}")
        print("[ERROR] Available .pt files:")
        for path in sorted(directory.glob("*.pt")):
            print(f"  {path.name}")
        sys.exit(1)
    return matches[0]


def adapter_weight_path(ckpt_dir: Path) -> Path:
    st_path = ckpt_dir / "adapter_model.safetensors"
    bin_path = ckpt_dir / "adapter_model.bin"
    if st_path.exists():
        return st_path
    if bin_path.exists():
        return bin_path
    die(f"No adapter_model.safetensors or adapter_model.bin in {ckpt_dir}")


def load_adapter_state(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    weight_path = adapter_weight_path(ckpt_dir)
    if weight_path.suffix == ".safetensors":
        state = safetensors_torch.load_file(str(weight_path), device="cpu")
    else:
        state = torch.load(str(weight_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
    if not isinstance(state, dict):
        die(f"Adapter state in {weight_path} is not a dict")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            die(f"Adapter state key {key} is not a tensor")
    return state


def validate_parameter_adapter_configs(
    dir_start: Path, dir_end: Path, spec: dict, *, alpha=None, r=None,
) -> dict:
    """Validate the shared interpretation of parameter-targeted LoRA factors."""
    configs = [json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
               for path in (dir_start, dir_end)]
    defaults = {
        "use_rslora": False, "lora_dropout": 0.0, "fan_in_fan_out": False,
        "use_dora": False, "lora_bias": False, "bias": "none",
        "rank_pattern": {}, "alpha_pattern": {}, "modules_to_save": None,
    }
    for key in ("r", "lora_alpha", "target_parameters", "target_modules", "peft_type", *defaults):
        values = [config.get(key, defaults.get(key)) for config in configs]
        if values[0] != values[1]:
            raise ValueError(f"adapter_config mismatch for {key}: {values}")
    config = configs[0]
    if config.get("peft_type") != "LORA":
        raise ValueError("Parameter adapter must have peft_type=LORA")
    if config.get("target_parameters") != [spec["target_parameter"]] or config.get("target_modules"):
        raise ValueError("Adapter targets do not match the configured single parameter")
    if any(config.get(key, defaults[key]) for key in (
        "lora_dropout", "fan_in_fan_out", "use_dora", "lora_bias",
        "rank_pattern", "alpha_pattern", "modules_to_save",
    )) or config.get("bias", "none") != "none":
        raise ValueError("Parameter adapter requires plain A/B LoRA without patterns, dropout or extra weights")
    if not isinstance(config.get("r"), int) or config["r"] <= 0:
        raise ValueError("Adapter r must be a positive integer")
    if not isinstance(config.get("lora_alpha"), (int, float)) or config["lora_alpha"] <= 0:
        raise ValueError("Adapter lora_alpha must be positive")
    if (r is not None and config["r"] != r) or (alpha is not None and config["lora_alpha"] != alpha):
        raise ValueError("Adapter rank/alpha do not match the experiment config")
    return config


def validate_expert_lora_shapes(a: torch.Tensor, b: torch.Tensor, r: int, spec: dict) -> int:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or a.shape[0] % r:
        raise ValueError(f"Invalid expert LoRA shapes: A={tuple(a.shape)}, B={tuple(b.shape)}, r={r}")
    dims = spec["lora_dims"]
    if (a.numel(), b.numel(), a.numel() + b.numel()) != (dims["A"], dims["B"], dims["concat"]):
        raise ValueError("Expert LoRA shapes do not match configured lora_dims")
    return a.shape[0] // r


def find_lora_keys(state: Dict[str, torch.Tensor]) -> Tuple[str, str]:
    keys = list(state.keys())
    a_matches = [key for key in keys if _matches_lora_key(key, LORA_SPEC, "A")]
    b_matches = [key for key in keys if _matches_lora_key(key, LORA_SPEC, "B")]
    if len(a_matches) != 1 or len(b_matches) != 1:
        print(f"[ERROR] Expected one lora_A and one lora_B key; found A={len(a_matches)}, B={len(b_matches)}")
        print("[ERROR] Available keys:")
        for key in keys:
            print(f"  {key}")
        sys.exit(1)
    return a_matches[0], b_matches[0]


def load_lora_flat(ckpt_dir: Path, *, expected_shapes=None) -> Tuple[torch.Tensor, torch.Tensor]:
    state = load_adapter_state(ckpt_dir)
    key_a, key_b = find_lora_keys(state)
    if expected_shapes is not None and (tuple(state[key_a].shape), tuple(state[key_b].shape)) != expected_shapes:
        raise ValueError("Start/end LoRA shapes differ despite matching flattened dimensions")
    if LORA_SPEC.get("target_kind") == "parameter":
        config = json.loads((ckpt_dir / "adapter_config.json").read_text(encoding="utf-8"))
        validate_expert_lora_shapes(state[key_a], state[key_b], config["r"], LORA_SPEC)
    a_flat = state[key_a].detach().float().cpu().reshape(-1)
    b_flat = state[key_b].detach().float().cpu().reshape(-1)
    if a_flat.numel() != A_DIM:
        die(f"{ckpt_dir} A_flat numel {a_flat.numel()} != {A_DIM}")
    if b_flat.numel() != B_DIM:
        die(f"{ckpt_dir} B_flat numel {b_flat.numel()} != {B_DIM}")
    check_tensor(a_flat, f"{ckpt_dir.name} A_flat", (A_DIM,))
    check_tensor(b_flat, f"{ckpt_dir.name} B_flat", (B_DIM,))
    return a_flat, b_flat


def load_gradient_matrix(path: Path, label: str) -> torch.Tensor:
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict) or "G" not in obj:
        die(f"{label}: cache file must be a dict containing key 'G': {path}")
    G = obj["G"].detach().float().cpu()
    if G.ndim != 2:
        die(f"{label}: G.ndim {G.ndim} != 2")
    if G.shape[0] != FULL_DIM:
        die(f"{label}: G.shape[0] {G.shape[0]} != FULL_DIM {FULL_DIM}")
    check_tensor(G, label)
    print(f"[CHECK] {label} shape: {tuple(G.shape)}")
    return G


def build_basis(G: torch.Tensor, block: str, rank: int) -> torch.Tensor:
    block_G = block_matrix(G, block, LORA_DIMS)
    if block == "concat":
        expected_dim = FULL_DIM
    elif block == "A_block":
        expected_dim = A_DIM
    elif block == "B_block":
        expected_dim = B_DIM
    else:
        die(f"Unknown block: {block}")
    if block_G.ndim != 2 or block_G.shape[0] != expected_dim:
        die(f"{block} block shape {tuple(block_G.shape)} does not start with {expected_dim}")
    U, S = svd_basis(block_G, eps=1e-8)
    if U.shape[1] < rank:
        die(f"{block} basis rank {U.shape[1]} < requested rank {rank}")
    check_tensor(U, f"U[{block}]")
    top_s = float(S[0].item()) if S.numel() else float("nan")
    check_finite_scalar(top_s, f"{block} top singular value")
    print(f"[CHECK] basis block={block} U.shape={tuple(U.shape)} top_singular_value={top_s:.8g}")
    return U[:, :rank].float().cpu()


def decompose(v: torch.Tensor, U: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if v.ndim != 1:
        die(f"decompose v.ndim {v.ndim} != 1")
    if U.ndim != 2 or U.shape[0] != v.shape[0] or U.shape[1] < r:
        die(f"decompose U shape {tuple(U.shape)} incompatible with v shape {tuple(v.shape)} and r={r}")
    check_tensor(v, "decompose v")
    check_tensor(U, "decompose U")
    Ur = U[:, :r].float()
    v_parallel = Ur @ (Ur.T @ v.float())
    v_perp = v.float() - v_parallel
    residual = float((v_parallel + v_perp - v.float()).norm().item())
    print(f"[CHECK] decompose residual norm (all layers, must be < 1e-4): {residual:.8g}")
    if residual >= 1e-4:
        die(f"decompose residual norm {residual} >= 1e-4")
    norm_parallel = float(v_parallel.norm().item())
    check_finite_scalar(norm_parallel, "norm_parallel")
    return v_parallel, v_perp, norm_parallel


def norm_matched_removal(
    v: torch.Tensor,
    U_ctrl: torch.Tensor,
    r: int,
    target_norm: float,
) -> torch.Tensor:
    if v.ndim != 1:
        die(f"norm_matched_removal v.ndim {v.ndim} != 1")
    if U_ctrl.ndim != 2 or U_ctrl.shape[0] != v.shape[0] or U_ctrl.shape[1] < r:
        die(
            "norm_matched_removal U_ctrl shape "
            f"{tuple(U_ctrl.shape)} incompatible with v shape {tuple(v.shape)} and r={r}"
        )
    check_tensor(v, "norm_matched_removal v")
    check_tensor(U_ctrl, "norm_matched_removal U_ctrl")
    check_finite_scalar(float(target_norm), "target_norm")
    Ur = U_ctrl[:, :r].float()
    proj = Ur @ (Ur.T @ v.float())
    proj_norm = float(proj.norm().item())
    check_finite_scalar(proj_norm, "proj_norm")
    if proj_norm < 1e-10:
        print("[WARNING] near-zero projection in norm_matched_removal, returning v unchanged")
        return v.float()
    scale = float(target_norm) / proj_norm
    removed = scale * proj
    result = v.float() - removed
    removed_norm = float(removed.norm().item())
    print(f"[CHECK] norm_matched_removal actual_removed_norm: {removed_norm:.8g}")
    check_tensor(result, "norm_matched_removal result", tuple(v.shape))
    return result


def norm_matched_injection(
    v: torch.Tensor,
    U_ctrl: torch.Tensor,
    r: int,
    target_norm: float,
) -> torch.Tensor:
    if v.ndim != 1:
        die(f"norm_matched_injection v.ndim {v.ndim} != 1")
    if U_ctrl.ndim != 2 or U_ctrl.shape[0] != v.shape[0] or U_ctrl.shape[1] < r:
        die(
            "norm_matched_injection U_ctrl shape "
            f"{tuple(U_ctrl.shape)} incompatible with v shape {tuple(v.shape)} and r={r}"
        )
    check_tensor(v, "norm_matched_injection v")
    check_tensor(U_ctrl, "norm_matched_injection U_ctrl")
    check_finite_scalar(float(target_norm), "target_norm")
    Ur = U_ctrl[:, :r].float()
    proj = Ur @ (Ur.T @ v.float())
    proj_norm = float(proj.norm().item())
    check_finite_scalar(proj_norm, "proj_norm")
    if proj_norm < 1e-10:
        print("[WARNING] near-zero projection in norm_matched_injection, returning v unchanged")
        return v.float()
    scale = float(target_norm) / proj_norm
    injected = scale * proj
    result = v.float() + injected
    injected_norm = float(injected.norm().item())
    print(f"[CHECK] norm_matched_injection actual_injected_norm: {injected_norm:.8g}")
    check_tensor(result, "norm_matched_injection result", tuple(v.shape))
    return result


def random_basis(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    R = torch.randn(dim, RANK, generator=gen)
    U_rand, _ = torch.linalg.qr(R, mode="reduced")
    check_tensor(U_rand, f"U_rand seed={seed}", (dim, RANK))
    return U_rand.float().cpu()


def save_synthetic_adapter(
    A_flat: torch.Tensor,
    B_flat: torch.Tensor,
    out_dir: Path,
    ref_state_dict: Dict[str, torch.Tensor],
    ref_shape_A: Tuple[int, ...],
    ref_shape_B: Tuple[int, ...],
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    check_tensor(A_flat, "save A_flat", (A_DIM,))
    check_tensor(B_flat, "save B_flat", (B_DIM,))
    out_path = out_dir / "adapter_model.safetensors"
    if dry_run:
        print(f"[DRY-RUN] would create: {out_dir}")
        return
    if out_path.exists() and not overwrite:
        print(f"[SKIP] {out_dir.name}")
        return
    key_a, key_b = find_lora_keys(ref_state_dict)
    ref_a = ref_state_dict[key_a]
    ref_b = ref_state_dict[key_b]
    state = {key: value.detach().cpu().clone() for key, value in ref_state_dict.items()}
    state[key_a] = A_flat.reshape(ref_shape_A).to(dtype=ref_a.dtype, device="cpu")
    state[key_b] = B_flat.reshape(ref_shape_B).to(dtype=ref_b.dtype, device="cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(state, str(out_path))
    config_src = CKPT_BASE / f"checkpoint-{STEP_START}" / "adapter_config.json"
    config_dst = out_dir / "adapter_config.json"
    if not config_src.exists():
        die(f"Missing adapter_config.json: {config_src}")
    shutil.copy2(config_src, config_dst)
    print(f"[WRITE] {out_dir}")


def make_summary_row(
    layer: str,
    basis_step: int,
    condition_id: str,
    adapter_dir: Path,
    delta_original: torch.Tensor,
    delta_synthetic: torch.Tensor,
    norm_parallel: float,
    U_harm: torch.Tensor,
    U_safe: Optional[torch.Tensor],
    *,
    metadata: Dict[str, object],
) -> Dict[str, object]:
    check_tensor(delta_original, f"{condition_id} delta_original")
    check_tensor(delta_synthetic, f"{condition_id} delta_synthetic", tuple(delta_original.shape))
    norm_delta_original = float(delta_original.norm().item())
    norm_delta_synthetic = float(delta_synthetic.norm().item())
    norm_removed = float((delta_original.float() - delta_synthetic.float()).norm().item())
    norm_removed_frac = norm_removed / norm_parallel if norm_parallel > 0 else float("nan")
    rho_harm = projection_fraction(delta_synthetic, U_harm, RANK, 1e-12)
    rho_safe = projection_fraction(delta_synthetic, U_safe, RANK, 1e-12) if U_safe is not None else None
    for label, value in [
        ("norm_delta_original", norm_delta_original),
        ("norm_delta_synthetic", norm_delta_synthetic),
        ("norm_removed", norm_removed),
        ("norm_parallel", float(norm_parallel)),
        ("norm_removed_frac", float(norm_removed_frac)),
        ("rho_harm_verify", float(rho_harm)),
    ]:
        check_finite_scalar(value, f"{layer}/{basis_step}/{condition_id}/{label}")
    if rho_safe is not None:
        check_finite_scalar(float(rho_safe), f"{layer}/{basis_step}/{condition_id}/rho_safe_verify")
    row = {
        "layer": layer,
        "basis_step": basis_step,
        "condition_id": condition_id,
        "adapter_dir": str(adapter_dir.resolve()),
        "norm_delta_original": norm_delta_original,
        "norm_delta_synthetic": norm_delta_synthetic,
        "norm_removed": norm_removed,
        "norm_parallel": float(norm_parallel),
        "norm_removed_frac": float(norm_removed_frac),
        "rho_harm_verify": float(rho_harm),
        "rho_safe_verify": None if rho_safe is None else float(rho_safe),
    }
    row.update(metadata)
    return row


def warn_if_norm_mismatch(condition_id: str, layer: str, basis_step: int, frac: float) -> None:
    if condition_id.startswith("C3") or condition_id.startswith("C4"):
        if abs(frac - 1.0) > 0.01:
            print(
                f"[WARNING] {condition_id} {layer} basis{basis_step} "
                f"norm_removed_frac={frac:.8g} differs from 1.0 by > 0.01"
            )


def preflight(args: argparse.Namespace) -> Tuple[Path, Optional[Path], Optional[Path]]:
    del args
    if not CONFIG_PATH.exists():
        die(f"CONFIG_PATH does not exist: {CONFIG_PATH}")
    if not GRAD_CACHE.exists() or not GRAD_CACHE.is_dir():
        die(f"GRAD_CACHE is not a directory: {GRAD_CACHE}")
    harm_end = find_cache_file(GRAD_CACHE, STEP_END, "harm")
    safe_end = find_cache_file(GRAD_CACHE, STEP_END, "safe") if INCLUDE_SAFE_CONTROL else None
    harm_start = (
        find_cache_file(GRAD_CACHE, STEP_START, "harm")
        if ENABLE_START_BASIS_VALIDATION
        else None
    )
    for step in [STEP_START, STEP_END]:
        ckpt_dir = CKPT_BASE / f"checkpoint-{step}"
        if not ckpt_dir.exists() or not ckpt_dir.is_dir():
            die(f"Checkpoint directory missing: {ckpt_dir}")
        adapter_weight_path(ckpt_dir)
        if not (ckpt_dir / "adapter_config.json").exists():
            die(f"adapter_config.json missing in {ckpt_dir}")
    return harm_end, safe_end, harm_start


def reconstruct_layer1(
    A_start: torch.Tensor,
    B_start: torch.Tensor,
    delta_synthetic: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    check_tensor(delta_synthetic, "layer1 delta_synthetic", (FULL_DIM,))
    return A_start + delta_synthetic[:A_DIM], B_start + delta_synthetic[A_DIM:]


def reconstruct_layer2(A_end: torch.Tensor, B_start: torch.Tensor, delta_B_synthetic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    check_tensor(delta_B_synthetic, "layer2 delta_B_synthetic", (B_DIM,))
    return A_end.clone(), B_start + delta_B_synthetic


def reconstruct_layer3(A_start: torch.Tensor, B_end: torch.Tensor, delta_A_synthetic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    check_tensor(delta_A_synthetic, "layer3 delta_A_synthetic", (A_DIM,))
    return A_start + delta_A_synthetic, B_end.clone()


def emit_condition(
    layer: str,
    basis_step: int,
    condition_id: str,
    dir_name: str,
    delta_original: torch.Tensor,
    delta_synthetic: torch.Tensor,
    norm_parallel: float,
    U_harm: torch.Tensor,
    U_safe: Optional[torch.Tensor],
    out_parent: Path,
    A_flat: torch.Tensor,
    B_flat: torch.Tensor,
    ref_state: Dict[str, torch.Tensor],
    ref_shape_A: Tuple[int, ...],
    ref_shape_B: Tuple[int, ...],
    overwrite: bool,
    dry_run: bool,
    *,
    metadata: Dict[str, object],
) -> Dict[str, object]:
    out_dir = out_parent / dir_name
    row = make_summary_row(
        layer,
        basis_step,
        condition_id,
        out_dir,
        delta_original,
        delta_synthetic,
        norm_parallel,
        U_harm,
        U_safe,
        metadata=metadata,
    )
    previous = EXISTING_CONDITIONS.get(condition_key(row))
    if previous is not None and not overwrite:
        for field in ("basis_rank", "step_start", "step_end", "seed", "scale",
                      "norm_delta_original", "norm_parallel", "norm_removed"):
            old, new = previous.get(field), row.get(field)
            if old not in (None, "") and new is not None and not math.isclose(
                float(old), float(new), rel_tol=1e-5, abs_tol=1e-8
            ):
                die(f"Existing condition definition conflicts: {condition_key(row)} / {field}")
        for field in ("checkpoint_start_hash", "checkpoint_end_hash", "gradient_cache_hash"):
            if previous.get(field) and row.get(field) and previous[field] != row[field]:
                die(f"Existing condition provenance conflicts: {condition_key(row)} / {field}")
        if Path(previous["adapter_dir"]).resolve() != out_dir.resolve():
            die(f"Existing condition path conflicts: {previous['adapter_dir']} != {out_dir}")
    save_synthetic_adapter(
        A_flat, B_flat, out_dir, ref_state, ref_shape_A, ref_shape_B,
        overwrite=overwrite, dry_run=dry_run,
    )
    warn_if_norm_mismatch(condition_id, layer, basis_step, float(row["norm_removed_frac"]))
    return previous if previous is not None and not overwrite else row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to JSON config file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing synthetic adapter files.")
    parser.add_argument("--dry-run", action="store_true", help="Print output paths without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global CONFIG_PATH, CKPT_BASE, GRAD_CACHE, OUT_BASE
    global STEP_START, STEP_END, RANK, RANDOM_SEEDS
    global INCLUDE_SAFE_CONTROL, ENABLE_START_BASIS_VALIDATION, HARM_BOOST_SCALES
    global LORA_SPEC, LORA_DIMS, A_DIM, B_DIM, FULL_DIM
    global BLOCKS, CONDITIONS, CHECKPOINT_HASHES, GRADIENT_CACHE_HASHES
    global EXISTING_CONDITIONS

    CONFIG_PATH = args.config
    cfg = read_config(CONFIG_PATH)
    LORA_SPEC = lora_spec_from_config(cfg)
    LORA_DIMS = {key: int(value) for key, value in LORA_SPEC["lora_dims"].items()}
    A_DIM = int(LORA_DIMS["A"])
    B_DIM = int(LORA_DIMS["B"])
    FULL_DIM = int(LORA_DIMS["concat"])
    CKPT_BASE = checkpoint_base(cfg)
    GRAD_CACHE = gradient_cache_dir(cfg)
    OUT_BASE = ablation_output_base(cfg)
    STEP_START = causal_step_start(cfg)
    STEP_END = causal_step_end(cfg)
    RANK = causal_rank(cfg)
    RANDOM_SEEDS = causal_random_seeds(cfg)
    INCLUDE_SAFE_CONTROL = causal_include_safe_control(cfg)
    ENABLE_START_BASIS_VALIDATION = causal_enable_start_basis_validation(cfg)
    INCLUDE_C4_AT_START_BASIS = causal_include_c4_at_start_basis(cfg)
    HARM_BOOST_SCALES = causal_harm_boost_scales(cfg)
    BLOCKS = causal_blocks(cfg)
    CONDITIONS = causal_conditions(cfg)
    expected_keys = configured_condition_keys(cfg)
    summary_path = OUT_BASE / "ablation_conditions_summary.csv"
    previous_rows = read_condition_rows(summary_path) if summary_path.exists() else []
    EXISTING_CONDITIONS = index_condition_rows(previous_rows)
    record_hashes = record_integrity_hashes(cfg)

    print(f"[CHECK] config: {CONFIG_PATH}")
    print(f"[CHECK] checkpoint_base: {CKPT_BASE}")
    print(f"[CHECK] gradient_cache: {GRAD_CACHE}")
    print(f"[CHECK] output_base: {OUT_BASE}")
    print(f"[CHECK] causal steps: {STEP_START} -> {STEP_END}; rank={RANK}")
    print(f"[CHECK] include_safe_control: {INCLUDE_SAFE_CONTROL}")
    print(f"[CHECK] enable_start_basis_validation: {ENABLE_START_BASIS_VALIDATION}")
    print(f"[CHECK] include_c4_at_start_basis: {INCLUDE_C4_AT_START_BASIS}")
    print(f"[CHECK] harm_boost_scales: {HARM_BOOST_SCALES}")
    print(f"[CHECK] random_seeds: {RANDOM_SEEDS}")
    print(f"[CHECK] blocks: {BLOCKS}")
    print(f"[CHECK] conditions: {CONDITIONS}")
    print(f"[CHECK] record_hashes: {record_hashes}")
    print(
        "[CHECK] lora spec: "
        f"layer={LORA_SPEC['layer']} {lora_target_description(LORA_SPEC)} dims={LORA_DIMS}"
    )

    if args.dry_run:
        if expected_keys is None:
            die("Dry-run requires an explicit causal_ablation.conditions list")
        cache_status = "available" if GRAD_CACHE.is_dir() else "not released"
        print(f"[DRY-RUN] gradient cache: {cache_status}")
        for layer, basis_step, condition_id in sorted(expected_keys):
            output_dir = OUT_BASE / layer / f"basis{basis_step}" / condition_id
            print(f"[DRY-RUN] would create: {output_dir}")
        print(f"[DRY-RUN] would write summary: {summary_path}")
        return

    harm_375_path, safe_375_path, harm_300_path = preflight(args)
    basis_end = f"basis{STEP_END}"
    basis_start = f"basis{STEP_START}"

    start_dir = CKPT_BASE / f"checkpoint-{STEP_START}"
    end_dir = CKPT_BASE / f"checkpoint-{STEP_END}"
    if LORA_SPEC.get("target_kind") == "parameter":
        validate_parameter_adapter_configs(
            start_dir, end_dir, LORA_SPEC, alpha=cfg["lora_alpha"], r=cfg["lora_r"]
        )
    ref_state = load_adapter_state(start_dir)
    key_a, key_b = find_lora_keys(ref_state)
    ref_shape_A = tuple(ref_state[key_a].shape)
    ref_shape_B = tuple(ref_state[key_b].shape)
    A_start, B_start = load_lora_flat(start_dir)
    shape_kwargs = {"expected_shapes": (ref_shape_A, ref_shape_B)} if LORA_SPEC.get("target_kind") == "parameter" else {}
    A_end, B_end = load_lora_flat(end_dir, **shape_kwargs)
    delta_A = A_end - A_start
    delta_B = B_end - B_start
    delta_theta = torch.cat([delta_A, delta_B]).float().cpu()
    check_tensor(delta_A, "delta_A", (A_DIM,))
    check_tensor(delta_B, "delta_B", (B_DIM,))
    check_tensor(delta_theta, "delta_theta", (FULL_DIM,))
    print(f"[CHECK] delta_theta norm: {float(delta_theta.norm().item()):.8g}")
    print(f"[CHECK] delta_A norm:     {float(delta_A.norm().item()):.8g}")
    print(f"[CHECK] delta_B norm:     {float(delta_B.norm().item()):.8g}")

    G_harm_375 = load_gradient_matrix(harm_375_path, f"harm@{STEP_END}")
    G_safe_375 = (
        load_gradient_matrix(safe_375_path, f"safe@{STEP_END}")
        if safe_375_path is not None
        else None
    )
    G_harm_300 = (
        load_gradient_matrix(harm_300_path, f"harm@{STEP_START}")
        if harm_300_path is not None
        else None
    )

    U_harm_concat_375 = build_basis(G_harm_375, "concat", RANK)
    U_safe_concat_375 = (
        build_basis(G_safe_375, "concat", RANK)
        if G_safe_375 is not None
        else None
    )
    U_harm_B_375 = build_basis(G_harm_375, "B_block", RANK)
    U_safe_B_375 = (
        build_basis(G_safe_375, "B_block", RANK)
        if G_safe_375 is not None
        else None
    )
    U_harm_A_375 = build_basis(G_harm_375, "A_block", RANK)
    U_harm_concat_300 = (
        build_basis(G_harm_300, "concat", RANK)
        if G_harm_300 is not None
        else None
    )
    U_harm_B_300 = (
        build_basis(G_harm_300, "B_block", RANK)
        if G_harm_300 is not None
        else None
    )

    if int(torch.tensor(ref_shape_A).prod().item()) != A_DIM:
        die(f"ref_shape_A {ref_shape_A} numel != {A_DIM}")
    if int(torch.tensor(ref_shape_B).prod().item()) != B_DIM:
        die(f"ref_shape_B {ref_shape_B} numel != {B_DIM}")

    if record_hashes:
        CHECKPOINT_HASHES = {
            STEP_START: sha256_file(adapter_weight_path(start_dir)),
            STEP_END: sha256_file(adapter_weight_path(end_dir)),
        }
        GRADIENT_CACHE_HASHES = {STEP_END: sha256_file(harm_375_path)}
        if harm_300_path is not None:
            GRADIENT_CACHE_HASHES[STEP_START] = sha256_file(harm_300_path)
    else:
        CHECKPOINT_HASHES = {}
        GRADIENT_CACHE_HASHES = {}

    rows: List[Dict[str, object]] = []

    if "concat" in BLOCKS:
        layer = "layer1_concat"
        out_parent = OUT_BASE / layer / basis_end
        par, perp, norm_parallel = decompose(delta_theta, U_harm_concat_375, RANK)
        print(f"[CHECK] norm_parallel Layer1 basis{STEP_END}: {norm_parallel:.8g}")
        if "C0_full" in CONDITIONS:
            # Full is the endpoint; reconstructing start + (end - start) can round it.
            A_new, B_new = A_end, B_end
            rows.append(
                emit_condition(layer, STEP_END, "C0_full", "C0_full", delta_theta, delta_theta, norm_parallel, U_harm_concat_375, U_safe_concat_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            )
        if "C1_harm_perp" in CONDITIONS:
            A_new, B_new = reconstruct_layer1(A_start, B_start, perp)
            row = emit_condition(layer, STEP_END, "C1_harm_perp", "C1_harm_perp", delta_theta, perp, norm_parallel, U_harm_concat_375, U_safe_concat_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            rows.append(row)
            print(f"[CHECK] C1 Layer1 basis{STEP_END} rho_harm_verify (target ~0.0):  {float(row['rho_harm_verify']):.8g}")
        if "C2_harm_parallel" in CONDITIONS:
            A_new, B_new = reconstruct_layer1(A_start, B_start, par)
            row = emit_condition(layer, STEP_END, "C2_harm_parallel", "C2_harm_parallel", delta_theta, par, norm_parallel, U_harm_concat_375, U_safe_concat_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            rows.append(row)
            print(f"[CHECK] C2 Layer1 basis{STEP_END} rho_harm_verify (target ~1.0): {float(row['rho_harm_verify']):.8g}")
        if INCLUDE_SAFE_CONTROL and "C3_safe_perp_nm" in CONDITIONS:
            assert U_safe_concat_375 is not None
            delta_c3 = norm_matched_removal(delta_theta, U_safe_concat_375, RANK, norm_parallel)
            A_new, B_new = reconstruct_layer1(A_start, B_start, delta_c3)
            row = emit_condition(layer, STEP_END, "C3_safe_perp_nm", "C3_safe_perp_nm", delta_theta, delta_c3, norm_parallel, U_harm_concat_375, U_safe_concat_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            rows.append(row)
            print(f"[CHECK] C3 Layer1 basis{STEP_END} norm_removed_frac (target ~1.0): {float(row['norm_removed_frac']):.8g}")
        if "C4_rand_perp_nm" in CONDITIONS:
            for seed in RANDOM_SEEDS:
                delta_c4 = norm_matched_removal(delta_theta, random_basis(FULL_DIM, seed), RANK, norm_parallel)
                A_new, B_new = reconstruct_layer1(A_start, B_start, delta_c4)
                condition = f"C4_rand_perp_nm_s{seed}"
                row = emit_condition(layer, STEP_END, condition, condition, delta_theta, delta_c4, norm_parallel, U_harm_concat_375, U_safe_concat_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END, seed=seed))
                rows.append(row)
                if seed == 0:
                    print(f"[CHECK] C4_s0 Layer1 basis{STEP_END} norm_removed_frac (target ~1.0): {float(row['norm_removed_frac']):.8g}")

        if HARM_BOOST_SCALES and "C5_harm_boost" in CONDITIONS:
            for k in HARM_BOOST_SCALES:
                delta_c5 = perp + k * par
                A_new, B_new = reconstruct_layer1(A_start, B_start, delta_c5)
                condition = f"C5_harm_boost_k{k}"
                row = emit_condition(
                    layer, STEP_END, condition, condition, delta_theta, delta_c5,
                    norm_parallel, U_harm_concat_375, U_safe_concat_375, out_parent,
                    A_new, B_new, ref_state, ref_shape_A, ref_shape_B,
                    args.overwrite, args.dry_run, metadata=build_metadata(STEP_END, scale=k),
                )
                rows.append(row)
                print(
                    f"[CHECK] C5 Layer1 basis{STEP_END} k={k:g} "
                    f"norm_delta_synthetic: {float(row['norm_delta_synthetic']):.8g}"
                )

                if "C6_rand_boost_nm" in CONDITIONS:
                    target_norm_boost = (k - 1.0) * norm_parallel
                    for seed in RANDOM_SEEDS:
                        delta_c6 = norm_matched_injection(
                            delta_theta,
                            random_basis(FULL_DIM, seed),
                            RANK,
                            target_norm_boost,
                        )
                        A_new, B_new = reconstruct_layer1(A_start, B_start, delta_c6)
                        condition = f"C6_rand_boost_nm_s{seed}_k{k}"
                        row = emit_condition(
                            layer, STEP_END, condition, condition, delta_theta, delta_c6,
                            norm_parallel, U_harm_concat_375, U_safe_concat_375, out_parent,
                            A_new, B_new, ref_state, ref_shape_A, ref_shape_B,
                            args.overwrite, args.dry_run,
                            metadata=build_metadata(STEP_END, seed=seed, scale=k),
                        )
                        rows.append(row)
                        if seed == 0:
                            print(
                                f"[CHECK] C6_s0 Layer1 basis{STEP_END} k={k:g} "
                                f"norm_delta_synthetic: {float(row['norm_delta_synthetic']):.8g}"
                            )

        start_basis_needed = "C1_harm_perp" in CONDITIONS or (HARM_BOOST_SCALES and "C5_harm_boost" in CONDITIONS)
        if ENABLE_START_BASIS_VALIDATION and start_basis_needed:
            assert U_harm_concat_300 is not None
            out_parent = OUT_BASE / layer / basis_start
            par_300, perp_300, norm_parallel_300 = decompose(delta_theta, U_harm_concat_300, RANK)
            print(f"[CHECK] norm_parallel Layer1 basis{STEP_START}: {norm_parallel_300:.8g}")
            if "C1_harm_perp" in CONDITIONS:
                A_new, B_new = reconstruct_layer1(A_start, B_start, perp_300)
                rows.append(
                    emit_condition(layer, STEP_START, "C1_harm_perp", "C1_harm_perp", delta_theta, perp_300, norm_parallel_300, U_harm_concat_300, U_safe_concat_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_START))
                )
            if INCLUDE_C4_AT_START_BASIS and "C4_rand_perp_nm" in CONDITIONS:
                for seed in RANDOM_SEEDS:
                    delta_c4_300 = norm_matched_removal(delta_theta, random_basis(FULL_DIM, seed), RANK, norm_parallel_300)
                    A_new, B_new = reconstruct_layer1(A_start, B_start, delta_c4_300)
                    condition = f"C4_rand_perp_nm_s{seed}"
                    row = emit_condition(layer, STEP_START, condition, condition, delta_theta, delta_c4_300, norm_parallel_300, U_harm_concat_300, U_safe_concat_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_START, seed=seed))
                    rows.append(row)
                    if seed == 0:
                        print(f"[CHECK] C4_s0 Layer1 basis{STEP_START} norm_removed_frac (target ~1.0): {float(row['norm_removed_frac']):.8g}")

            if HARM_BOOST_SCALES and "C5_harm_boost" in CONDITIONS:
                for k in HARM_BOOST_SCALES:
                    delta_c5 = perp_300 + k * par_300
                    A_new, B_new = reconstruct_layer1(A_start, B_start, delta_c5)
                    condition = f"C5_harm_boost_k{k}"
                    row = emit_condition(layer, STEP_START, condition, condition, delta_theta, delta_c5, norm_parallel_300, U_harm_concat_300, U_safe_concat_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_START, scale=k))
                    rows.append(row)
                    print(f"[CHECK] C5 Layer1 basis{STEP_START} k={k:g} norm_delta_synthetic: {float(row['norm_delta_synthetic']):.8g}")

                    if "C6_rand_boost_nm" in CONDITIONS:
                        target_norm_boost = (k - 1.0) * norm_parallel_300
                        for seed in RANDOM_SEEDS:
                            delta_c6 = norm_matched_injection(delta_theta, random_basis(FULL_DIM, seed), RANK, target_norm_boost)
                            A_new, B_new = reconstruct_layer1(A_start, B_start, delta_c6)
                            condition = f"C6_rand_boost_nm_s{seed}_k{k}"
                            row = emit_condition(layer, STEP_START, condition, condition, delta_theta, delta_c6, norm_parallel_300, U_harm_concat_300, U_safe_concat_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_START, seed=seed, scale=k))
                            rows.append(row)
                            if seed == 0:
                                print(f"[CHECK] C6_s0 Layer1 basis{STEP_START} k={k:g} norm_delta_synthetic: {float(row['norm_delta_synthetic']):.8g}")

    if "B_block" in BLOCKS:
        layer = "layer2_B_block"
        out_parent = OUT_BASE / layer / basis_end
        par_B, perp_B, norm_parallel_B = decompose(delta_B, U_harm_B_375, RANK)
        print(f"[CHECK] norm_parallel Layer2 basis{STEP_END}: {norm_parallel_B:.8g}")
        if "C0_full" in CONDITIONS:
            A_new, B_new = reconstruct_layer2(A_end, B_start, delta_B)
            rows.append(
                emit_condition(layer, STEP_END, "C0_full", "C0_full", delta_B, delta_B, norm_parallel_B, U_harm_B_375, U_safe_B_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            )
        if "C1_harm_perp" in CONDITIONS:
            A_new, B_new = reconstruct_layer2(A_end, B_start, perp_B)
            rows.append(
                emit_condition(layer, STEP_END, "C1_harm_perp", "C1_harm_perp", delta_B, perp_B, norm_parallel_B, U_harm_B_375, U_safe_B_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            )
        if "C2_harm_parallel" in CONDITIONS:
            A_new, B_new = reconstruct_layer2(A_end, B_start, par_B)
            rows.append(
                emit_condition(layer, STEP_END, "C2_harm_parallel", "C2_harm_parallel", delta_B, par_B, norm_parallel_B, U_harm_B_375, U_safe_B_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            )
        if INCLUDE_SAFE_CONTROL and "C3_safe_perp_nm" in CONDITIONS:
            assert U_safe_B_375 is not None
            delta_B_c3 = norm_matched_removal(delta_B, U_safe_B_375, RANK, norm_parallel_B)
            A_new, B_new = reconstruct_layer2(A_end, B_start, delta_B_c3)
            rows.append(
                emit_condition(layer, STEP_END, "C3_safe_perp_nm", "C3_safe_perp_nm", delta_B, delta_B_c3, norm_parallel_B, U_harm_B_375, U_safe_B_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            )
        if "C4_rand_perp_nm" in CONDITIONS:
            for seed in RANDOM_SEEDS:
                delta_B_c4 = norm_matched_removal(delta_B, random_basis(B_DIM, seed), RANK, norm_parallel_B)
                A_new, B_new = reconstruct_layer2(A_end, B_start, delta_B_c4)
                condition = f"C4_rand_perp_nm_s{seed}"
                rows.append(
                    emit_condition(layer, STEP_END, condition, condition, delta_B, delta_B_c4, norm_parallel_B, U_harm_B_375, U_safe_B_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END, seed=seed))
                )

        if ENABLE_START_BASIS_VALIDATION and "C1_harm_perp" in CONDITIONS:
            assert U_harm_B_300 is not None
            out_parent = OUT_BASE / layer / basis_start
            _, perp_B_300, norm_parallel_B_300 = decompose(delta_B, U_harm_B_300, RANK)
            print(f"[CHECK] norm_parallel Layer2 basis{STEP_START}: {norm_parallel_B_300:.8g}")
            A_new, B_new = reconstruct_layer2(A_end, B_start, perp_B_300)
            rows.append(
                emit_condition(layer, STEP_START, "C1_harm_perp", "C1_harm_perp", delta_B, perp_B_300, norm_parallel_B_300, U_harm_B_300, U_safe_B_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_START))
            )
            if INCLUDE_C4_AT_START_BASIS and "C4_rand_perp_nm" in CONDITIONS:
                for seed in RANDOM_SEEDS:
                    delta_B_c4_300 = norm_matched_removal(delta_B, random_basis(B_DIM, seed), RANK, norm_parallel_B_300)
                    A_new, B_new = reconstruct_layer2(A_end, B_start, delta_B_c4_300)
                    condition = f"C4_rand_perp_nm_s{seed}"
                    rows.append(
                        emit_condition(layer, STEP_START, condition, condition, delta_B, delta_B_c4_300, norm_parallel_B_300, U_harm_B_300, U_safe_B_375, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_START, seed=seed))
                    )

    if "A_block" in BLOCKS:
        layer = "layer3_A_block"
        out_parent = OUT_BASE / layer / basis_end
        par_A, perp_A, norm_parallel_A = decompose(delta_A, U_harm_A_375, RANK)
        print(f"[CHECK] norm_parallel Layer3 basis{STEP_END}: {norm_parallel_A:.8g}")
        if "C0_full" in CONDITIONS:
            A_new, B_new = reconstruct_layer3(A_start, B_end, delta_A)
            rows.append(
                emit_condition(layer, STEP_END, "C0_full", "C0_full", delta_A, delta_A, norm_parallel_A, U_harm_A_375, None, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            )
        if "C1_harm_perp" in CONDITIONS:
            A_new, B_new = reconstruct_layer3(A_start, B_end, perp_A)
            rows.append(
                emit_condition(layer, STEP_END, "C1_harm_perp", "C1_harm_perp", delta_A, perp_A, norm_parallel_A, U_harm_A_375, None, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            )
        if "C2_harm_parallel" in CONDITIONS:
            A_new, B_new = reconstruct_layer3(A_start, B_end, par_A)
            rows.append(
                emit_condition(layer, STEP_END, "C2_harm_parallel", "C2_harm_parallel", delta_A, par_A, norm_parallel_A, U_harm_A_375, None, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END))
            )
        if "C4_rand_perp_nm" in CONDITIONS:
            for seed in RANDOM_SEEDS:
                delta_A_c4 = norm_matched_removal(delta_A, random_basis(A_DIM, seed), RANK, norm_parallel_A)
                A_new, B_new = reconstruct_layer3(A_start, B_end, delta_A_c4)
                condition = f"C4_rand_perp_nm_s{seed}"
                rows.append(
                    emit_condition(layer, STEP_END, condition, condition, delta_A, delta_A_c4, norm_parallel_A, U_harm_A_375, None, out_parent, A_new, B_new, ref_state, ref_shape_A, ref_shape_B, args.overwrite, args.dry_run, metadata=build_metadata(STEP_END, seed=seed))
                )

    if expected_keys is not None and set(index_condition_rows(rows)) != expected_keys:
        die("Constructed condition set does not match configuration")
    merged_rows = merge_condition_rows(previous_rows, rows)
    if args.dry_run:
        print(f"[DRY-RUN] would merge {len(rows)} configured conditions into {len(merged_rows)} total: {summary_path}")
    else:
        write_csv_with_backup(summary_path, merged_rows)
        print(f"[WRITE] {summary_path}")
    print(f"[CHECK] total_conditions: {len(rows)}")


if __name__ == "__main__":
    main()
