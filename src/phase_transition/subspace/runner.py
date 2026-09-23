import argparse
import json
import logging
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from .core import (
    block_matrix,
    build_eval_sources,
    collect_grad,
    cumulative_variance,
    effective_rank,
    encode_train_example_full_sequence,
    find_lora_params,
    freeze_except_lora,
    get_lora_dims_from_config,
    hot_swap_adapter,
    load_jsonl_dataset,
    lora_spec_from_config,
    projection_fraction,
    projected_cosine,
    random_qr_baseline,
    read_json,
    read_jsonl,
    reconstruct_eval_input,
    sample_indices,
    subspace_overlap,
    svd_basis,
    token_nll,
    write_csv,
    write_json,
)


BLOCKS = ["concat", "A_block", "B_block"]
SVD_RANKS = [1, 2, 4, 8, 16, 32]
REQUIRED_F1_FIELDS = [
    "steps",
    "n_train",
    "rank_list",
    "random_baseline_repeats",
    "harmful_dataset_path",
    "safe_dataset_path",
    "harmful_checkpoint_root",
    "safe_checkpoint_root",
    "eval_token_metadata_path",
    "pivot_token_sets_path",
    "em_results_path",
    "eval_checkpoint",
    "eval_run_tag",
    "output_dir_fullseq",
    "min_grad_norm",
    "smoke_n_train",
    "smoke_max_eval_tokens_per_label",
]


def f1_config_example() -> str:
    return json.dumps(
        {
            "f1_subspace": {
                "steps": [400, 500, 600, 750, 1125],
                "n_train": 256,
                "rank_list": [1, 2, 4, 8, 16],
                "random_baseline_repeats": 1000,
                "harmful_dataset_path": "data/harmful.jsonl",
                "safe_dataset_path": "data/safe.jsonl",
                "harmful_checkpoint_root": "artifacts/harmful",
                "safe_checkpoint_root": "artifacts/safe",
                "eval_token_metadata_path": "outputs/geometry/pivot_token_curvature_raw.jsonl",
                "pivot_token_sets_path": "data/pivot_token_sets.json",
                "em_results_path": "data/em_results.json",
                "eval_checkpoint": 1125,
                "eval_run_tag": "400-1125",
                "output_dir_fullseq": "outputs/subspace/key_steps_fullseq",
                "min_grad_norm": 1e-12,
                "smoke_n_train": 4,
                "smoke_max_eval_tokens_per_label": 2,
                "stat_tests": {
                    "loss_mask_mode": "full_sequence_padding_only",
                    "cache_schema_version": 2,
                    "lora_dims": {"A": 15360, "B": 3840, "concat": 19200},
                    "blocks": ["concat", "A_block", "B_block"],
                },
            }
        },
        indent=2,
    )


def require_f1_config(cfg_f1: dict) -> None:
    missing = [key for key in REQUIRED_F1_FIELDS if key not in cfg_f1]
    if missing:
        raise RuntimeError(
            "Missing required f1_subspace config field(s): "
            f"{missing}. Add explicit eval source paths; auto-guessing checkpoint/run-tag paths "
            f"is disabled. Example:\n{f1_config_example()}"
        )


def parse_steps(cfg_f1: dict, smoke: bool) -> list:
    raw_steps = cfg_f1.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RuntimeError("f1_subspace.steps must be a non-empty list of ints")
    try:
        steps = [int(s) for s in raw_steps]
    except Exception as exc:
        raise RuntimeError(f"f1_subspace.steps must contain only ints: {raw_steps}") from exc
    if any(s <= 0 for s in steps):
        raise RuntimeError(f"f1_subspace.steps must contain positive ints: {steps}")
    return [steps[0]] if smoke else steps


def resolve_output_dir(cfg_f1: dict, smoke: bool) -> Path:
    output_dir = Path(cfg_f1["output_dir_fullseq"])
    if output_dir.name != "key_steps_fullseq":
        raise RuntimeError(
            "Full-sequence F.1 output_dir_fullseq must end with key_steps_fullseq "
            f"to avoid overwriting assistant-only outputs: {output_dir}"
        )
    return output_dir / "smoke" if smoke else output_dir


def stat_config(cfg_f1: dict) -> dict:
    if "stat_tests" not in cfg_f1:
        raise RuntimeError("Missing f1_subspace.stat_tests config")
    return cfg_f1["stat_tests"]


def loss_mask_mode(cfg_f1: dict) -> str:
    return str(stat_config(cfg_f1)["loss_mask_mode"])


def cache_schema_version(cfg_f1: dict) -> int:
    return int(stat_config(cfg_f1)["cache_schema_version"])


def f1_blocks(cfg_f1: dict) -> list:
    return [str(x) for x in stat_config(cfg_f1).get("blocks", BLOCKS)]


def lora_dims_dict(cfg: dict) -> dict:
    return get_lora_dims_from_config(cfg)


def cache_metadata_matches(actual: dict, expected: dict) -> bool:
    if not isinstance(actual, dict):
        return False
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def _redact_config(cfg):
    if isinstance(cfg, dict):
        out = {}
        for k, v in cfg.items():
            if "api_key" in k.lower() or k.lower().endswith("key"):
                out[k] = "<redacted>"
            else:
                out[k] = _redact_config(v)
        return out
    if isinstance(cfg, list):
        return [_redact_config(v) for v in cfg]
    return cfg


def setup_logging(output_dir: Path):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def resolve_safe_root(cfg_f1: dict, cfg: dict):
    requested = Path(cfg_f1["safe_checkpoint_root"])
    harmful = Path(cfg_f1["harmful_checkpoint_root"]).resolve()
    if not requested.exists():
        raise FileNotFoundError(
            f"Configured safe checkpoint root does not exist: {requested}. "
            "Set f1_subspace.safe_checkpoint_root to the exact safe-control checkpoint directory."
        )
    if requested.resolve() == harmful:
        raise RuntimeError("Resolved safe checkpoint root equals harmful checkpoint root")
    return requested, False, []


def checkpoint_has_adapter(path: Path) -> bool:
    return (
        path.exists()
        and ((path / "adapter_config.json").exists())
        and ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists())
    )


def p0_path_checks(cfg, cfg_f1, safe_root: Path, steps):
    paths = {
        "model_name_or_path": Path(cfg["model_name_or_path"]),
        "harmful_dataset_path": Path(cfg_f1["harmful_dataset_path"]),
        "safe_dataset_path": Path(cfg_f1["safe_dataset_path"]),
        "harmful_checkpoint_root": Path(cfg_f1["harmful_checkpoint_root"]),
        "safe_checkpoint_root": safe_root,
        "eval_token_metadata_path": Path(cfg_f1["eval_token_metadata_path"]),
        "pivot_token_sets_path": Path(cfg_f1["pivot_token_sets_path"]),
        "em_results_path": Path(cfg_f1["em_results_path"]),
    }
    results = {"paths": {}, "checkpoints": {}}
    for name, path in paths.items():
        exists = path.exists()
        results["paths"][name] = {"path": str(path), "exists": exists}
        if not exists:
            raise FileNotFoundError(f"Required path missing: {name}={path}")
    harmful_root = Path(cfg_f1["harmful_checkpoint_root"])
    if safe_root.resolve() == harmful_root.resolve():
        raise RuntimeError("Refusing to use harmful checkpoint root as safe root")
    for step in steps:
        h = harmful_root / f"checkpoint-{step}"
        s = safe_root / f"checkpoint-{step}"
        h_ok = checkpoint_has_adapter(h)
        s_ok = checkpoint_has_adapter(s)
        results["checkpoints"][str(step)] = {
            "harmful": str(h), "harmful_ok": h_ok,
            "safe": str(s), "safe_ok": s_ok,
        }
        if not h_ok:
            raise FileNotFoundError(f"Missing harmful adapter checkpoint: {h}")
        if not s_ok:
            raise FileNotFoundError(f"Missing safe adapter checkpoint: {s}")
    return results


def load_model(cfg, first_ckpt: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"], local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "local_files_only": True,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        **model_load_kwargs,
    )
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(first_ckpt), is_trainable=True)
    model.eval()
    lora_a, lora_b = find_lora_params(model, lora_spec_from_config(cfg))
    freeze_except_lora(model, lora_a, lora_b)
    device = lora_a.device
    return model, tokenizer, lora_a, lora_b, device


def train_loss(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch)
    return out.loss


def save_gradient_cache_atomic(path, obj):
    """Commit a gradient file only after its contents have reached disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_train_gradients(
    model, tokenizer, lora_a, lora_b, device, dataset, indices, ckpt, step, tag, cfg, out_dir,
    *, resume=False,
):
    cfg_f1 = cfg["f1_subspace"]
    dataset_path = cfg_f1["harmful_dataset_path"] if tag == "harm" else cfg_f1["safe_dataset_path"]
    expected_meta = {
        "cache_schema_version": cache_schema_version(cfg_f1),
        "loss_mask_mode": loss_mask_mode(cfg_f1),
        "checkpoint_path": str(ckpt),
        "dataset_path": dataset_path,
        "sample_indices": indices,
        "max_seq_length": int(cfg["max_seq_length"]),
        "model_name_or_path": cfg["model_name_or_path"],
        "lora_dims": lora_dims_dict(cfg),
    }
    cache = out_dir / "cache" / "gradients" / f"step{step}_train_{tag}_{loss_mask_mode(cfg_f1)}.pt"
    if cache.exists():
        obj = torch.load(cache, map_location="cpu")
        if (
            obj.get("G") is not None
            and obj["G"].shape == (lora_dims_dict(cfg)["concat"], len(indices))
            and cache_metadata_matches(obj.get("metadata"), expected_meta)
        ):
            logging.info("Using cached %s", cache)
            return obj
        if resume:
            raise RuntimeError(f"Existing train cache does not match this run; preserved: {cache}")
        logging.info("Ignoring stale train cache with mismatched metadata: %s", cache)
    logging.info("Collecting train gradients step=%s tag=%s n=%s", step, tag, len(indices))
    hot_swap_adapter(ckpt, lora_a, lora_b, device, lora_spec_from_config(cfg))
    cols, norms = [], []
    partial_dir = cache.with_suffix(".partial")
    resumed = 0
    for j, idx in enumerate(indices):
        partial_path = partial_dir / f"sample{j:05d}.pt"
        if resume and partial_path.exists():
            saved = torch.load(partial_path, map_location="cpu")
            g, n = saved.get("gradient"), saved.get("norm")
            if (
                not cache_metadata_matches(saved.get("metadata"), expected_meta)
                or saved.get("sample_idx") != idx
                or not isinstance(g, torch.Tensor)
                or tuple(g.shape) != (lora_dims_dict(cfg)["concat"],)
                or not torch.isfinite(g).all()
                or not isinstance(n, (int, float))
                or not math.isfinite(n)
                or n <= cfg_f1["min_grad_norm"]
            ):
                raise RuntimeError(f"Partial gradient does not match this run; preserved: {partial_path}")
            cols.append(g)
            norms.append(n)
            resumed += 1
            continue
        batch = encode_train_example_full_sequence(tokenizer, dataset[idx], int(cfg["max_seq_length"]))
        model.zero_grad(set_to_none=True)
        loss = train_loss(model, batch, device)
        loss.backward()
        g, n = collect_grad(model, lora_a, lora_b, cfg_f1["min_grad_norm"], lora_spec_from_config(cfg))
        if resume:
            save_gradient_cache_atomic(partial_path, {
                "gradient": g,
                "norm": n,
                "sample_idx": idx,
                "metadata": expected_meta,
            })
        cols.append(g)
        norms.append(n)
        if (j + 1) % 25 == 0:
            logging.info("  %s step=%s %s/%s", tag, step, j + 1, len(indices))
    G = torch.stack(cols, dim=1)
    obj = {
        "G": G,
        "column_norms": torch.tensor(norms),
        "sample_indices": indices,
        "step": int(step),
        "loss_mask_mode": loss_mask_mode(cfg_f1),
        "metadata": {"kind": f"train_{tag}", **expected_meta},
    }
    if resume:
        save_gradient_cache_atomic(cache, obj)
        logging.info("Completed %s step=%s: resumed=%s computed=%s", tag, step, resumed, len(indices) - resumed)
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, cache)
    return obj


def select_eval_rows(rows, step, label, smoke_limit=None):
    selected = [
        r for r in rows
        if int(r.get("step")) == int(step)
        and r.get("label") == label
        and r.get("error") is None
        and r.get("token_id_matches_input") is True
    ]
    selected.sort(key=lambda r: (str(r["question_id"]), int(r["sample_idx"]), int(r["position"])))
    if smoke_limit is not None:
        selected = selected[:smoke_limit]
    return selected


def collect_eval_gradients(model, tokenizer, lora_a, lora_b, device, rows, samples, em_map, ckpt, step, label, cfg, out_dir):
    question_ids_expected = [str(r["question_id"]) for r in rows]
    sample_indices_expected = [int(r["sample_idx"]) for r in rows]
    absolute_positions_expected = [int(r["absolute_position"]) for r in rows]
    token_ids_expected = [int(r["token_id"]) for r in rows]
    expected_meta = {
        "cache_schema_version": cache_schema_version(cfg["f1_subspace"]),
        "checkpoint_path": str(ckpt),
        "eval_token_metadata_path": cfg["f1_subspace"]["eval_token_metadata_path"],
        "question_ids": question_ids_expected,
        "sample_indices": sample_indices_expected,
        "absolute_positions": absolute_positions_expected,
        "token_ids": token_ids_expected,
        "label": label,
        "model_name_or_path": cfg["model_name_or_path"],
        "lora_dims": lora_dims_dict(cfg),
    }
    cache = out_dir / "cache" / "gradients" / f"step{step}_eval_{label}.pt"
    if cache.exists():
        obj = torch.load(cache, map_location="cpu")
        if (
            obj.get("G") is not None
            and obj["G"].shape[0] == lora_dims_dict(cfg)["concat"]
            and obj["G"].shape[1] == len(rows)
            and cache_metadata_matches(obj.get("metadata"), expected_meta)
        ):
            logging.info("Using cached %s", cache)
            return obj
        logging.info("Ignoring stale eval cache with mismatched metadata: %s", cache)
    logging.info("Collecting eval gradients step=%s label=%s n=%s", step, label, len(rows))
    hot_swap_adapter(ckpt, lora_a, lora_b, device, lora_spec_from_config(cfg))
    cols, norms = [], []
    token_response_ids, question_ids, sample_idxs, abs_positions, positions, token_ids = [], [], [], [], [], []
    for j, row in enumerate(rows):
        input_ids, _ = reconstruct_eval_input(tokenizer, row, samples, em_map)
        input_ids = input_ids.to(device)
        model.zero_grad(set_to_none=True)
        loss = token_nll(model, input_ids, int(row["absolute_position"]))
        loss.backward()
        g, n = collect_grad(model, lora_a, lora_b, cfg["f1_subspace"]["min_grad_norm"], lora_spec_from_config(cfg))
        cols.append(g)
        norms.append(n)
        qid = str(row["question_id"])
        sidx = int(row["sample_idx"])
        token_response_ids.append(f"{qid}::{sidx}")
        question_ids.append(qid)
        sample_idxs.append(sidx)
        abs_positions.append(int(row["absolute_position"]))
        positions.append(int(row["position"]))
        token_ids.append(int(row["token_id"]))
        if (j + 1) % 50 == 0:
            logging.info("  eval_%s step=%s %s/%s", label, step, j + 1, len(rows))
    G = torch.stack(cols, dim=1)
    obj = {
        "G": G,
        "column_norms": torch.tensor(norms),
        "token_response_ids": token_response_ids,
        "question_ids": question_ids,
        "sample_indices": sample_idxs,
        "absolute_positions": abs_positions,
        "token_positions": positions,
        "token_ids": token_ids,
        "step": int(step),
        "metadata": {"kind": f"eval_{label}", **expected_meta},
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, cache)
    return obj


def run_p0(cfg, cfg_f1, output_dir, model, tokenizer, lora_a, lora_b, device, harmful_data, eval_rows, samples, em_map, steps, safe_root, safe_auto, path_results):
    results = {
        "timestamp": datetime.now().isoformat(),
        "all_passed": False,
        "loss_mask_mode": loss_mask_mode(cfg_f1),
        "safe_checkpoint_root": str(safe_root),
        "safe_checkpoint_root_auto_resolved": safe_auto,
        "path_checks": path_results,
        "lora_dims": {"A": lora_a.numel(), "B": lora_b.numel(), "concat": lora_a.numel() + lora_b.numel()},
    }
    try:
        dims = lora_dims_dict(cfg)
        assert lora_a.numel() == dims["A"] and lora_b.numel() == dims["B"]
        hot_swap_adapter(Path(cfg_f1["harmful_checkpoint_root"]) / f"checkpoint-{steps[0]}", lora_a, lora_b, device, lora_spec_from_config(cfg))
        batch = encode_train_example_full_sequence(tokenizer, harmful_data[0], int(cfg["max_seq_length"]))
        model.zero_grad(set_to_none=True)
        loss = train_loss(model, batch, device)
        loss.backward()
        g, n = collect_grad(model, lora_a, lora_b, cfg_f1["min_grad_norm"], lora_spec_from_config(cfg))
        results["train_gradient"] = {
            "dim": int(g.numel()),
            "norm": n,
            "finite": bool(torch.isfinite(g).all()),
            "loss_mask_mode": loss_mask_mode(cfg_f1),
        }
        row = next(r for r in eval_rows if int(r["step"]) == steps[0] and r["label"] == "pivot" and r.get("error") is None and r.get("token_id_matches_input") is True)
        input_ids, _ = reconstruct_eval_input(tokenizer, row, samples, em_map)
        actual = int(input_ids[0, int(row["absolute_position"])].item())
        results["eval_reconstruction"] = {
            "question_id": row["question_id"],
            "sample_idx": int(row["sample_idx"]),
            "absolute_position": int(row["absolute_position"]),
            "actual": actual,
            "token_id": int(row["token_id"]),
            "actual_token_id": int(row.get("actual_token_id", row["token_id"])),
        }
        input_ids = input_ids.to(device)
        model.zero_grad(set_to_none=True)
        loss_t = token_nll(model, input_ids, int(row["absolute_position"]))
        loss_t.backward()
        ge, ne = collect_grad(model, lora_a, lora_b, cfg_f1["min_grad_norm"], lora_spec_from_config(cfg))
        results["eval_gradient"] = {"dim": int(ge.numel()), "norm": ne, "finite": bool(torch.isfinite(ge).all())}
        results["all_passed"] = True
    except Exception as exc:
        results["error"] = repr(exc)
        write_json(output_dir / "p0_checks.json", results)
        raise
    write_json(output_dir / "p0_checks.json", results)
    return results


def analyze_and_write(step_objs, steps, cfg_f1, out_dir, lora_dims):
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    eff_rows, sing_rows, overlap_rows, rand_rows, block_rows, dir_rows = [], [], [], [], [], []
    bases = {}

    for step in steps:
        bases[step] = {}
        for name, obj in step_objs[step].items():
            bases[step][name] = {}
            for block in f1_blocks(cfg_f1):
                G = block_matrix(obj["G"], block, lora_dims)
                U, S = svd_basis(G, cfg_f1["min_grad_norm"])
                bases[step][name][block] = (U, S)
                cvar = cumulative_variance(S, SVD_RANKS)
                eff_rows.append({
                    "step": step, "matrix": name, "block": block,
                    "n_cols": int(G.shape[1]), "dim": int(G.shape[0]),
                    "effective_rank": effective_rank(S),
                    **{f"cumvar_r{r}": v for r, v in cvar.items()},
                })
                for i, sv in enumerate(S[:32], start=1):
                    sing_rows.append({"step": step, "matrix": name, "block": block, "rank_index": i, "singular_value": float(sv.item())})

        for block in f1_blocks(cfg_f1):
            comparisons = [
                ("A", "train_harm_full_sequence", "eval_pivot"),
                ("B", "train_harm_full_sequence", "eval_neutral"),
                ("C", "train_safe_full_sequence", "eval_pivot"),
            ]
            for group, left, right in comparisons:
                U = bases[step][left][block][0]
                V = bases[step][right][block][0]
                for r in cfg_f1["rank_list"]:
                    if r <= U.shape[1] and r <= V.shape[1]:
                        ov, s1, cos = subspace_overlap(U, V, int(r))
                        row = {
                            "step": step, "block": block, "rank": int(r), "group": group,
                            "left": left, "right": right, "overlap": ov,
                            "top_principal_angle_cosine": s1,
                            "principal_angle_cosines_json": json.dumps(cos),
                        }
                        overlap_rows.append(row)
                        block_rows.append(row.copy())
            eval_U = bases[step]["eval_pivot"][block][0]
            dim = lora_dims["A"] if block == "A_block" else lora_dims["B"] if block == "B_block" else lora_dims["concat"]
            for r in cfg_f1["rank_list"]:
                if r <= eval_U.shape[1]:
                    rb = random_qr_baseline(
                        dim, eval_U, int(r), int(cfg_f1["random_baseline_repeats"]),
                        int(cfg_f1.get("random_seed", 42)) + step * 100 + int(r),
                    )
                    rand_rows.append({"step": step, "block": block, "rank": int(r), "group": "D", **rb})

            for r in [4, 8]:
                U_p = bases[step]["eval_pivot"][block][0]
                if r <= U_p.shape[1]:
                    pivot = block_matrix(step_objs[step]["eval_pivot"]["G"], block, lora_dims).float()
                    harm = block_matrix(step_objs[step]["train_harm_full_sequence"]["G"], block, lora_dims).float()
                    mh = harm.mean(dim=1)
                    mp = pivot.mean(dim=1)
                    dir_rows.append({
                        "step": step, "block": block, "rank": r, "eval_label": "pivot",
                        "rho": projection_fraction(mh, U_p, r, cfg_f1["min_grad_norm"]),
                        "cos_dir": projected_cosine(mh, mp, U_p, r, cfg_f1["min_grad_norm"]),
                    })
                U_n = bases[step]["eval_neutral"][block][0]
                if r <= U_n.shape[1]:
                    harm = block_matrix(step_objs[step]["train_harm_full_sequence"]["G"], block, lora_dims).float()
                    neutral = block_matrix(step_objs[step]["eval_neutral"]["G"], block, lora_dims).float()
                    mh = harm.mean(dim=1)
                    mn = neutral.mean(dim=1)
                    dir_rows.append({
                        "step": step, "block": block, "rank": r, "eval_label": "neutral",
                        "rho": projection_fraction(mh, U_n, r, cfg_f1["min_grad_norm"]),
                        "cos_dir": projected_cosine(mh, mn, U_n, r, cfg_f1["min_grad_norm"]),
                    })

    write_csv(tables / "effective_rank_by_step.csv", eff_rows)
    write_csv(tables / "singular_values_by_step.csv", sing_rows)
    write_csv(tables / "overlap_by_step.csv", overlap_rows)
    write_csv(tables / "random_baseline_by_step.csv", rand_rows)
    write_csv(tables / "block_overlap_by_step.csv", block_rows)
    write_csv(tables / "directional_metrics_by_step.csv", dir_rows)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Collect validated gradient caches without running the descriptive SVD analysis.",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = read_json(cfg_path)
    cfg_f1 = dict(cfg["f1_subspace"])
    require_f1_config(cfg_f1)
    steps = parse_steps(cfg_f1, args.smoke)
    if args.smoke:
        cfg_f1["n_train"] = int(cfg_f1["smoke_n_train"])
        cfg_f1["random_baseline_repeats"] = min(10, int(cfg_f1["random_baseline_repeats"]))
    output_dir = resolve_output_dir(cfg_f1, args.smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(output_dir)
    logging.info(
        "Starting F.1 key-step diagnostic smoke=%s cache_only=%s loss_mask_mode=%s output_dir=%s",
        args.smoke,
        args.cache_only,
        loss_mask_mode(cfg_f1),
        output_dir,
    )

    safe_root, safe_auto, safe_candidates = resolve_safe_root(cfg_f1, cfg)
    path_results = p0_path_checks(cfg, cfg_f1, safe_root, steps)

    harmful_data = load_jsonl_dataset(Path(cfg_f1["harmful_dataset_path"]))
    safe_data = load_jsonl_dataset(Path(cfg_f1["safe_dataset_path"]))
    random_seed = int(cfg_f1.get("random_seed", cfg.get("seed", 42)))
    harm_idx = sample_indices(len(harmful_data), int(cfg_f1["n_train"]), random_seed)
    safe_idx = sample_indices(len(safe_data), int(cfg_f1["n_train"]), random_seed)
    eval_rows_all = read_jsonl(Path(cfg_f1["eval_token_metadata_path"]))
    samples, em_map, eval_source_paths = build_eval_sources(
        Path(cfg_f1["pivot_token_sets_path"]),
        Path(cfg_f1["em_results_path"]),
    )

    model, tokenizer, lora_a, lora_b, device = load_model(cfg, Path(cfg_f1["harmful_checkpoint_root"]) / f"checkpoint-{steps[0]}")
    p0 = run_p0(cfg, cfg_f1, output_dir, model, tokenizer, lora_a, lora_b, device, harmful_data, eval_rows_all, samples, em_map, steps, safe_root, safe_auto, path_results)

    step_objs = {}
    shapes = {}
    token_counts = {}
    for step in steps:
        h_ckpt = Path(cfg_f1["harmful_checkpoint_root"]) / f"checkpoint-{step}"
        s_ckpt = safe_root / f"checkpoint-{step}"
        pivot_rows = select_eval_rows(eval_rows_all, step, "pivot", cfg_f1["smoke_max_eval_tokens_per_label"] if args.smoke else None)
        neutral_rows = select_eval_rows(eval_rows_all, step, "neutral", cfg_f1["smoke_max_eval_tokens_per_label"] if args.smoke else None)
        token_counts[str(step)] = {
            "pivot": len(pivot_rows),
            "neutral": len(neutral_rows),
            "response_groups": len({f"{r['question_id']}::{int(r['sample_idx'])}" for r in pivot_rows + neutral_rows}),
        }
        if not pivot_rows or not neutral_rows:
            raise RuntimeError(f"No eval rows selected for step {step}")
        collectors = (
            (
                "train_harm_full_sequence",
                lambda: collect_train_gradients(
                    model, tokenizer, lora_a, lora_b, device, harmful_data, harm_idx,
                    h_ckpt, step, "harm", cfg, output_dir,
                ),
            ),
            (
                "train_safe_full_sequence",
                lambda: collect_train_gradients(
                    model, tokenizer, lora_a, lora_b, device, safe_data, safe_idx,
                    s_ckpt, step, "safe", cfg, output_dir,
                ),
            ),
            (
                "eval_pivot",
                lambda: collect_eval_gradients(
                    model, tokenizer, lora_a, lora_b, device, pivot_rows, samples,
                    em_map, h_ckpt, step, "pivot", cfg, output_dir,
                ),
            ),
            (
                "eval_neutral",
                lambda: collect_eval_gradients(
                    model, tokenizer, lora_a, lora_b, device, neutral_rows, samples,
                    em_map, h_ckpt, step, "neutral", cfg, output_dir,
                ),
            ),
        )
        current_step_objs = {}
        shapes[str(step)] = {}
        for name, collect_fn in collectors:
            obj = collect_fn()
            shapes[str(step)][name] = list(obj["G"].shape)
            if not args.cache_only:
                current_step_objs[name] = obj
            del obj
        if not args.cache_only:
            step_objs[step] = current_step_objs

    if args.cache_only:
        logging.info("Skipping descriptive SVD analysis because --cache-only was set")
    else:
        analyze_and_write(step_objs, steps, cfg_f1, output_dir, lora_dims_dict(cfg))
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_hash(),
        "config_path": str(cfg_path),
        "log_path": str(log_path),
        "loss_mask_mode": loss_mask_mode(cfg_f1),
        "output_dir": str(output_dir),
        "input_paths": {
            "model_name_or_path": cfg["model_name_or_path"],
            "harmful_dataset_path": cfg_f1["harmful_dataset_path"],
            "safe_dataset_path": cfg_f1["safe_dataset_path"],
            "eval_token_metadata_path": cfg_f1["eval_token_metadata_path"],
            **eval_source_paths,
        },
        "harmful_dataset_path": cfg_f1["harmful_dataset_path"],
        "safe_dataset_path": cfg_f1["safe_dataset_path"],
        "harmful_checkpoint_root": cfg_f1["harmful_checkpoint_root"],
        "safe_checkpoint_root": str(safe_root),
        "eval_token_metadata_path": cfg_f1["eval_token_metadata_path"],
        "pivot_token_sets_path": cfg_f1["pivot_token_sets_path"],
        "em_results_path": cfg_f1["em_results_path"],
        "eval_checkpoint": int(cfg_f1["eval_checkpoint"]),
        "eval_run_tag": str(cfg_f1["eval_run_tag"]),
        "model_name_or_path": cfg["model_name_or_path"],
        "max_seq_length": int(cfg["max_seq_length"]),
        "lora_dims": lora_dims_dict(cfg),
        "safe_checkpoint_root_auto_resolved": safe_auto,
        "safe_root_candidates_considered": safe_candidates,
        "steps": steps,
        "selected_steps": steps,
        "n_train": int(cfg_f1["n_train"]),
        "rank_list": cfg_f1["rank_list"],
        "random_seed": random_seed,
        "random_baseline_repeats": int(cfg_f1["random_baseline_repeats"]),
        "tensor_shapes": shapes,
        "eval_token_counts": token_counts,
        "p0_checks": p0,
        "smoke": args.smoke,
        "cache_only": args.cache_only,
        "cache_schema_version": cache_schema_version(cfg["f1_subspace"]),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    logging.info("Completed diagnostic successfully: %s", output_dir)


if __name__ == "__main__":
    main()
