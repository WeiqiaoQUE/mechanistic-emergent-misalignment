"""Compute group-level or token-level pivot geometry."""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from .core import (
    prepare_sample_inputs,
    compute_position_loss,
    collect_lora_grads,
    compute_pivot_kappa,
    run_sanity_check,
)

def format_layer_segment(layers_to_transform):
    if layers_to_transform is None:
        return "all"
    if isinstance(layers_to_transform, int):
        return f"l{layers_to_transform}"
    if isinstance(layers_to_transform, list):
        if not layers_to_transform:
            return "all"
        return "l" + "-".join(str(layer) for layer in layers_to_transform)
    raise TypeError(
        "layers_to_transform must be null, an int, or a list of ints; "
        f"got {type(layers_to_transform).__name__}"
    )


def load_config_from_args():
    default_config = PROJECT_ROOT / "configs" / "qwen14b_financial.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config), help="Path to JSON config file")
    parser.add_argument(
        "--export-f1-eval-metadata-only",
        action="store_true",
        help=(
            "Export validated pivot/neutral token metadata for F.1 without loading "
            "the model or computing geometry."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg.get("pivot_geometry"), dict):
        raise KeyError(f"Missing required config section: pivot_geometry in {config_path}")
    return config_path, cfg, bool(args.export_f1_eval_metadata_only)


CONFIG_PATH, CONFIG, EXPORT_F1_EVAL_METADATA_ONLY = load_config_from_args()
GEOMETRY_CONFIG = CONFIG["pivot_geometry"]

BASE_MODEL      = CONFIG["model_name_or_path"]
LAYER_SEGMENT   = format_layer_segment(CONFIG.get("layers_to_transform"))
LORA_R          = CONFIG["lora_r"]
TASK_TAG        = CONFIG["task_tag"]

EXPERIMENT_NAME = str(GEOMETRY_CONFIG["experiment_name"])
START_CKPT = int(GEOMETRY_CONFIG["start_ckpt"])
END_CKPT = int(GEOMETRY_CONFIG["end_ckpt"])
USE_SPARSE_TOKEN_CHECKPOINTS = bool(GEOMETRY_CONFIG.get("use_sparse_token_checkpoints", True))
SPARSE_TOKEN_CHECKPOINTS = [int(s) for s in GEOMETRY_CONFIG.get("sparse_token_checkpoints", [700])]

COMPUTE_GROUP_LEVEL = bool(GEOMETRY_CONFIG.get("compute_group_level", False))
COMPUTE_TOKEN_LEVEL = bool(GEOMETRY_CONFIG.get("compute_token_level", True))
RUN_PERMUTATION     = bool(GEOMETRY_CONFIG.get("run_permutation", True))
RUN_SANITY_CHECK    = bool(GEOMETRY_CONFIG.get("run_sanity_check", True))

N_PERMUTATIONS = int(GEOMETRY_CONFIG.get("n_permutations", 1000))
PERMUTATION_SEED = int(GEOMETRY_CONFIG.get("permutation_seed", 42))
REQUIRE_COMPLETE_STEPS_FOR_PERMUTATION = bool(
    GEOMETRY_CONFIG.get("require_complete_steps_for_permutation", True)
)

TOKEN_SCHEMA_VERSION = "token_curvature_v1"
PERM_SCHEMA_VERSION  = "kappa_permutation_v1"

RUN_TAG = GEOMETRY_CONFIG.get("run_tag", f"{START_CKPT}-{END_CKPT}")
PIVOT_JSON = Path(GEOMETRY_CONFIG["pivot_token_sets_path"])
EM_JSON = Path(GEOMETRY_CONFIG["em_results_path"])
CHECKPOINTS_DIR = Path(GEOMETRY_CONFIG["checkpoints_dir"])
PLOTS_DIR = Path(GEOMETRY_CONFIG["output_dir"])
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = PLOTS_DIR / "pivot_geometry_raw.jsonl"
AGG_PATH     = PLOTS_DIR / "pivot_geometry_aggregated.json"

TOKEN_DIR = PLOTS_DIR / "per_token_curvature"
TOKEN_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_RAW_PATH      = TOKEN_DIR / "pivot_token_curvature_raw.jsonl"
TOKEN_META_PATH     = TOKEN_DIR / "pivot_token_curvature_metadata.json"
PERM_RESULTS_PATH   = TOKEN_DIR / "pivot_kappa_permutation_results.json"

# Snapshot used to represent step=0: raw base forward with a LoRA parameterization.
STEP0_LORA_A = None


def ckpt_dir_fn(step: int) -> Path:
    return CHECKPOINTS_DIR / f"checkpoint-{step}"


def _safe(v):
    """Convert NaN/Inf to None so JSON remains standards-compliant."""
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        v = v.item()
    if isinstance(v, (float, int)):
        vf = float(v)
        if not math.isfinite(vf):
            return None
        return vf
    return v


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def _mean(vals):
    vals = [v for v in vals if not _is_missing(v)]
    return float(sum(vals) / len(vals)) if vals else None


def _std(vals):
    vals = [v for v in vals if not _is_missing(v)]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return float(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)))


def _safe_ratio(num, den, eps: float = 1e-12):
    if _is_missing(num) or _is_missing(den) or abs(float(den)) < eps:
        return None
    return float(num / den)


def _quantile(vals, q: float):
    vals = sorted(v for v in vals if not _is_missing(v))
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    idx = q * (len(vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(vals[lo])
    frac = idx - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _json_dump(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# LoRA checkpoint loading.
def hot_swap_lora(step: int, lora_A_param, lora_B_param, device):
    """将 step 对应 checkpoint 的 lora_A/lora_B 权重替换到模型中。"""
    import safetensors.torch as safetensors_torch

    global STEP0_LORA_A
    if int(step) == 0:
        if STEP0_LORA_A is None:
            raise RuntimeError("step=0 requested before STEP0_LORA_A was initialized")
        lora_A_param.data.copy_(STEP0_LORA_A.to(device=device, dtype=lora_A_param.dtype))
        lora_B_param.data.zero_()
        print("    hot_swap: step=0 -> raw base forward (LoRA B zeroed)")
        return True

    ckpt_path = ckpt_dir_fn(step)
    st_path   = ckpt_path / "adapter_model.safetensors"
    bin_path  = ckpt_path / "adapter_model.bin"

    if st_path.exists():
        state_dict = safetensors_torch.load_file(str(st_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        return False

    found_A = found_B = False
    for key, tensor in state_dict.items():
        if "lora_A" in key:
            lora_A_param.data.copy_(tensor.to(device=device, dtype=lora_A_param.dtype))
            found_A = True
            print(f"    hot_swap: {key} -> lora_A  shape={tensor.shape}")
        elif "lora_B" in key:
            lora_B_param.data.copy_(tensor.to(device=device, dtype=lora_B_param.dtype))
            found_B = True
            print(f"    hot_swap: {key} -> lora_B  shape={tensor.shape}")

    if not (found_A and found_B):
        raise RuntimeError(
            f"热替换失败，step={step}：找到 lora_A={found_A}, lora_B={found_B}。"
            f"请检查 state_dict key 格式，实际 keys: {list(state_dict.keys())[:10]}"
        )
    return True


# Group and token computations.
def _compute_geometry(model, input_ids, positions, response_start_idx,
                      lora_A_param, lora_B_param, device):
    """
    计算给定 positions 的 L_val, g_norm, kappa。

    如果 positions 只包含一个 response-relative token position，则返回 token-level 几何量。
    如果 positions 包含多个 token，则返回 group-level 几何量。
    """
    lora_A_param.requires_grad_(True)
    lora_B_param.requires_grad_(True)
    try:
        with torch.no_grad():
            L_val = compute_position_loss(model, input_ids, positions, response_start_idx).item()

        model.zero_grad()
        loss_for_grad = compute_position_loss(model, input_ids, positions, response_start_idx)
        loss_for_grad.backward()
        g_vec, g_norm = collect_lora_grads([lora_A_param, lora_B_param])
        g_hat = g_vec / (g_vec.norm() + 1e-12)
        model.zero_grad()

        if g_norm < 1e-8:
            print(f"      WARNING: g_norm={g_norm:.2e} < 1e-8，kappa 记为 nan")
            kappa = float("nan")
        else:
            kappa = compute_pivot_kappa(
                model, input_ids, positions, response_start_idx,
                [lora_A_param, lora_B_param], g_hat,
            )
        model.zero_grad()
        torch.cuda.empty_cache()

    finally:
        lora_A_param.requires_grad_(False)
        lora_B_param.requires_grad_(False)
        model.zero_grad()
        torch.cuda.empty_cache()

    return L_val, g_norm, kappa


# Checkpoint discovery.
def filter_sparse_token_steps(all_steps):
    """Restrict token-level computation to representative sparse checkpoints."""
    if not USE_SPARSE_TOKEN_CHECKPOINTS:
        return all_steps

    available = set(all_steps)
    requested = [int(s) for s in SPARSE_TOKEN_CHECKPOINTS]
    selected = [s for s in requested if s in available]
    missing = [s for s in requested if s not in available]

    if missing:
        print(f"[init] WARNING: requested sparse checkpoint(s) not found: {missing}")
    if not selected:
        raise RuntimeError(
            "USE_SPARSE_TOKEN_CHECKPOINTS=True，但没有任何 requested sparse checkpoint 可用。"
        )

    print(
        f"[init] 使用 sparse token-level checkpoints: {selected} "
        f"({len(selected)}/{len(requested)} requested)"
    )
    return selected


def enumerate_checkpoint_steps():
    all_step_dirs = sorted(CHECKPOINTS_DIR.glob("checkpoint-*"))
    all_steps = []
    for d in all_step_dirs:
        try:
            step = int(d.name.split("-")[1])
            if step < START_CKPT or step > END_CKPT:
                continue
            if (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists():
                all_steps.append(step)
        except (IndexError, ValueError):
            continue
    all_steps.sort()
    if START_CKPT == 0 and 0 not in all_steps:
        all_steps.insert(0, 0)
    if not all_steps:
        raise RuntimeError(
            f"未找到 checkpoint，范围为 [{START_CKPT}, {END_CKPT}]，目录={CHECKPOINTS_DIR}"
        )
    return all_steps


def write_token_metadata(pivot_data, samples_meta, all_steps):
    thresholds = pivot_data.get("thresholds", {})
    downsample = pivot_data.get("downsample", {})

    expected_pivot = sum(len(s.get("pivot_positions", [])) for s in samples_meta)
    expected_neutral = sum(len(s.get("neutral_positions", [])) for s in samples_meta)
    expected_total = expected_pivot + expected_neutral

    if len(all_steps) > 1:
        strides = sorted({b - a for a, b in zip(all_steps[:-1], all_steps[1:])})
    else:
        strides = []

    metadata = {
        "schema_version": TOKEN_SCHEMA_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "task_tag": TASK_TAG,
        "model_name_or_path": BASE_MODEL,
        "start_ckpt": START_CKPT,
        "end_ckpt": END_CKPT,
        "checkpoint_range": RUN_TAG,
        "checkpoint_steps": all_steps,
        "checkpoint_strides_observed": strides,
        "lora_r": LORA_R,
        "layer": LAYER_SEGMENT,
        "pivot_token_set_path": str(PIVOT_JSON),
        "pivot_threshold": thresholds.get("pivot_threshold"),
        "eps_low": thresholds.get("eps_low"),
        "eps_high": thresholds.get("eps_high"),
        "neutral_downsample_rule": downsample.get("rule"),
        "neutral_downsample_seed": downsample.get("random_seed"),
        "n_samples": len(samples_meta),
        "expected_pivot_tokens_per_step": expected_pivot,
        "expected_neutral_tokens_per_step": expected_neutral,
        "expected_total_tokens_per_step": expected_total,
        "curvature_definition": (
            "token-level kappa = g_hat_t^T H_t g_hat_t, where H_t is the Hessian "
            "of single-token loss with respect to the selected LoRA parameters"
        ),
        "permutation_scope": "within-sample over pivot ∪ neutral tokens",
        "n_permutations": N_PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "raw_output_path": str(TOKEN_RAW_PATH),
        "permutation_results_path": str(PERM_RESULTS_PATH),
    }
    _json_dump(metadata, TOKEN_META_PATH)
    print(f"[token-meta] 写入 metadata: {TOKEN_META_PATH}")
    return metadata


def _token_id_by_position(sample: dict, label: str):
    positions = sample.get(f"{label}_positions", [])
    token_ids = sample.get(f"{label}_token_ids", [])
    if len(positions) != len(token_ids):
        print(
            f"WARNING: ({sample.get('question_id')}, {sample.get('sample_idx')}) "
            f"{label} positions/token_ids 长度不一致: {len(positions)} vs {len(token_ids)}"
        )
    return {int(pos): int(tok_id) for pos, tok_id in zip(positions, token_ids)}


def _actual_token_id(input_ids, absolute_position: int):
    try:
        return int(input_ids[0, absolute_position].detach().cpu().item())
    except Exception:
        return None


def _make_token_record(step, sample, response_start_idx, label, position, token_id,
                       input_ids, L_val=None, g_norm=None,
                       kappa=None, error=None):
    qid = sample["question_id"]
    sidx = sample["sample_idx"]
    pos = int(position)
    abs_pos = int(response_start_idx + pos)
    expected_tok = None if token_id is None else int(token_id)
    actual_tok = _actual_token_id(input_ids, abs_pos)

    rec = {
        "schema_version": TOKEN_SCHEMA_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "task_tag": TASK_TAG,
        "run_tag": RUN_TAG,
        "step": int(step),
        "question_id": qid,
        "sample_idx": int(sidx),
        "response_start_idx": int(response_start_idx),
        "label": label,
        "position": pos,
        "absolute_position": abs_pos,
        "token_id": expected_tok,
        "actual_token_id": actual_tok,
        "token_id_matches_input": (
            None if expected_tok is None or actual_tok is None else bool(expected_tok == actual_tok)
        ),
        "L_token": _safe(L_val),
        "g_norm_token": _safe(g_norm),
        "kappa_token": _safe(kappa),
        "n_pivot_in_sample": len(sample.get("pivot_positions", [])),
        "n_neutral_in_sample": len(sample.get("neutral_positions", [])),
        "error": error,
    }

    return rec


def export_f1_eval_metadata(samples_meta, em_results_map):
    cfg_f1 = CONFIG.get("f1_subspace")
    if not isinstance(cfg_f1, dict):
        raise RuntimeError(
            "--export-f1-eval-metadata-only requires a f1_subspace config object"
        )
    raw_steps = cfg_f1.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RuntimeError("f1_subspace.steps must be a non-empty list")
    try:
        steps = [int(step) for step in raw_steps]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"f1_subspace.steps must contain only ints: {raw_steps}") from exc
    if any(step <= 0 for step in steps) or len(set(steps)) != len(steps):
        raise RuntimeError(f"f1_subspace.steps must contain unique positive ints: {steps}")

    raw_output_path = cfg_f1.get("eval_token_metadata_path")
    if not raw_output_path:
        raise RuntimeError(
            "--export-f1-eval-metadata-only requires "
            "f1_subspace.eval_token_metadata_path"
        )
    output_path = Path(raw_output_path)

    from transformers import AutoTokenizer

    print(f"[f1-token-meta] 加载 tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        trust_remote_code=True,
    )

    sample_rows = []
    for sample in samples_meta:
        input_ids, response_start_idx, pivot_positions, neutral_positions = \
            prepare_sample_inputs(sample, em_results_map, tokenizer)
        for label, positions in (
            ("pivot", pivot_positions),
            ("neutral", neutral_positions),
        ):
            token_id_map = _token_id_by_position(sample, label)
            for position in positions:
                position = int(position)
                if position not in token_id_map:
                    raise RuntimeError(
                        f"Missing {label} token_id for "
                        f"({sample['question_id']}, {sample['sample_idx']}) position={position}"
                    )
                absolute_position = int(response_start_idx) + position
                expected_token_id = int(token_id_map[position])
                actual_token_id = _actual_token_id(input_ids, absolute_position)
                if actual_token_id != expected_token_id:
                    raise RuntimeError(
                        f"Token mismatch for ({sample['question_id']}, {sample['sample_idx']}) "
                        f"label={label} position={position}: "
                        f"expected={expected_token_id}, actual={actual_token_id}"
                    )
                sample_rows.append({
                    "schema_version": "f1_eval_token_metadata_v1",
                    "source": "pivot_token_sets+em_results+tokenizer",
                    "question_id": sample["question_id"],
                    "sample_idx": int(sample["sample_idx"]),
                    "response_start_idx": int(response_start_idx),
                    "label": label,
                    "position": position,
                    "absolute_position": absolute_position,
                    "token_id": expected_token_id,
                    "actual_token_id": actual_token_id,
                    "token_id_matches_input": True,
                    "error": None,
                })

    records = [
        {**row, "step": step}
        for step in steps
        for row in sample_rows
    ]
    keys = [
        (
            int(row["step"]),
            str(row["question_id"]),
            int(row["sample_idx"]),
            str(row["label"]),
            int(row["position"]),
        )
        for row in records
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Generated F.1 eval metadata contains duplicate token keys")

    if output_path.exists():
        existing = []
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
        if existing != records:
            raise RuntimeError(
                f"Existing F.1 eval metadata differs from generated content: {output_path}"
            )
        print(f"[f1-token-meta] 已存在且内容一致，复用: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "x", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"[f1-token-meta] 写入 {len(records)} 条记录："
        f"steps={len(steps)} tokens_per_step={len(sample_rows)} path={output_path}"
    )
    return output_path


def load_completed_token_keys():
    completed = set()
    if not TOKEN_RAW_PATH.exists():
        return completed
    with open(TOKEN_RAW_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            completed.add((
                int(rec["step"]),
                rec["question_id"],
                int(rec["sample_idx"]),
                rec["label"],
                int(rec["position"]),
            ))
    print(f"[checkpoint-token] 断点续算：已完成 token 记录 {len(completed)} 条")
    return completed


# Token-level computation.
def compute_token_level_curvature(model, tokenizer, samples_meta, em_results_map,
                                  lora_A_param, lora_B_param, device,
    all_steps, pivot_data):
    write_token_metadata(pivot_data, samples_meta, all_steps)
    completed = load_completed_token_keys()

    total_steps = len(all_steps)
    done_steps = 0
    t_loop_start = time.time()

    first_step = all_steps[0]
    print(f"[token-level] 热替换回 step={first_step} 准备主循环 ...")
    hot_swap_lora(first_step, lora_A_param, lora_B_param, device)

    for step in all_steps:
        t_step_start = time.time()

        ok = hot_swap_lora(step, lora_A_param, lora_B_param, device)
        if not ok:
            print(f"[token-level step {step}] 找不到 adapter 文件，跳过")
            done_steps += 1
            continue

        n_written = 0
        n_skipped_completed = 0
        n_errors = 0

        for sample in samples_meta:
            qid = sample["question_id"]
            sidx = int(sample["sample_idx"])

            try:
                input_ids, response_start_idx, pivot_positions, neutral_positions = \
                    prepare_sample_inputs(sample, em_results_map, tokenizer)
            except KeyError as e:
                print(f"[token-level step {step}] 跳过样本 ({qid}, {sidx})：{e}")
                continue

            input_ids = input_ids.to(device)
            label_to_positions = {
                "pivot": [int(p) for p in pivot_positions],
                "neutral": [int(p) for p in neutral_positions],
            }

            for label, positions in label_to_positions.items():
                token_id_map = _token_id_by_position(sample, label)
                for pos in positions:
                    key = (int(step), qid, sidx, label, int(pos))
                    if key in completed:
                        n_skipped_completed += 1
                        continue

                    token_id = token_id_map.get(int(pos))
                    L_val = g_norm = kappa = None
                    error = None

                    try:
                        L_val, g_norm, kappa = _compute_geometry(
                            model, input_ids, [int(pos)], response_start_idx,
                            lora_A_param, lora_B_param, device,
                        )
                    except Exception as e:
                        error = repr(e)
                        n_errors += 1
                        print(f"  ERROR token-level {label} ({qid},{sidx}) pos={pos} step={step}: {e}")
                        model.zero_grad()
                        torch.cuda.empty_cache()

                    rec = _make_token_record(
                        step=step,
                        sample=sample,
                        response_start_idx=response_start_idx,
                        label=label,
                        position=pos,
                        token_id=token_id,
                        input_ids=input_ids,
                        L_val=L_val,
                        g_norm=g_norm,
                        kappa=kappa,
                        error=error,
                    )
                    _append_jsonl(TOKEN_RAW_PATH, rec)
                    completed.add(key)
                    n_written += 1

            torch.cuda.empty_cache()

        done_steps += 1
        t_step_elapsed = time.time() - t_step_start
        t_total_elapsed = time.time() - t_loop_start
        remaining = total_steps - done_steps
        avg_per_step = t_total_elapsed / done_steps
        eta_sec = avg_per_step * remaining
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
        print(
            f"[token-level step {step}] 完成，写入 {n_written}，"
            f"跳过已完成 {n_skipped_completed}，错误 {n_errors}，"
            f"耗时 {t_step_elapsed:.1f}s，剩余 {remaining} 个 checkpoint，ETA {eta_str}"
        )


# Group-level aggregation.
def aggregate_results():
    """
    读取 pivot_geometry_raw.jsonl，按 step 聚合，输出 pivot_geometry_aggregated.json。

    聚合规则：
      - 只聚合该 step 下样本数 == 预期（total_samples）的 step，否则跳过并警告
      - per_question 按 question_id 分组
    """
    records = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("WARNING: pivot_geometry_raw.jsonl 为空，跳过聚合")
        return

    step_groups = defaultdict(list)
    for r in records:
        step_groups[r["step"]].append(r)

    step_counts = [len(v) for v in step_groups.values()]
    expected_n = max(set(step_counts), key=step_counts.count)
    sorted_steps = sorted(step_groups.keys())

    metric_keys = ["L_pivot", "g_norm_pivot", "kappa_pivot",
                   "L_neutral", "g_norm_neutral", "kappa_neutral"]

    global_agg = {k: [] for k in metric_keys}
    per_question_agg = {}
    valid_steps = []

    for step in sorted_steps:
        recs = step_groups[step]
        if len(recs) != expected_n:
            print(f"WARNING: step={step} 只有 {len(recs)} 条记录（期望 {expected_n}），跳过聚合")
            continue
        valid_steps.append(step)

        for k in metric_keys:
            vals = [r[k] for r in recs if not _is_missing(r[k])]
            global_agg[k].append(float(sum(vals) / len(vals)) if vals else None)

        qid_groups = defaultdict(list)
        for r in recs:
            qid_groups[r["question_id"]].append(r)

        for qid, qrecs in qid_groups.items():
            if qid not in per_question_agg:
                per_question_agg[qid] = {
                    "n_samples": len(qrecs),
                    **{k: [] for k in metric_keys}
                }
            for k in metric_keys:
                vals = [r[k] for r in qrecs if not _is_missing(r[k])]
                per_question_agg[qid][k].append(
                    float(sum(vals) / len(vals)) if vals else None
                )

    output = {
        "steps": valid_steps,
        "global": global_agg,
        "per_question": per_question_agg,
    }

    _json_dump(output, AGG_PATH)
    print(f"聚合完成：{len(valid_steps)} 个有效 step，写入 {AGG_PATH}")


# Group-level computation.
def compute_group_level_geometry(model, tokenizer, samples_meta, em_results_map,
                                 lora_A_param, lora_B_param, device,
                                 all_steps):
    completed = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                completed.add((rec["step"], rec["question_id"], rec["sample_idx"]))
        print(f"[checkpoint-group] 断点续算：已完成 {len(completed)} 条记录")

    total_steps = len(all_steps)
    done_steps = 0
    t_loop_start = time.time()

    first_step = all_steps[0]
    print(f"[group-level] 热替换回 step={first_step} 准备主循环 ...")
    hot_swap_lora(first_step, lora_A_param, lora_B_param, device)

    for step in all_steps:
        t_step_start = time.time()
        ok = hot_swap_lora(step, lora_A_param, lora_B_param, device)
        if not ok:
            print(f"[group-level step {step}] 找不到 adapter 文件，跳过")
            done_steps += 1
            continue

        for sample in samples_meta:
            qid = sample["question_id"]
            sidx = sample["sample_idx"]

            if (step, qid, sidx) in completed:
                continue

            try:
                input_ids, response_start_idx, pivot_positions, neutral_positions = \
                    prepare_sample_inputs(sample, em_results_map, tokenizer)
            except KeyError as e:
                print(f"[group-level step {step}] 跳过样本 ({qid}, {sidx})：{e}")
                continue
            input_ids = input_ids.to(device)

            L_pivot_val = g_norm_pivot = kappa_pivot = float("nan")
            try:
                L_pivot_val, g_norm_pivot, kappa_pivot = _compute_geometry(
                    model, input_ids, pivot_positions, response_start_idx,
                    lora_A_param, lora_B_param, device,
                )
            except Exception as e:
                print(f"  ERROR pivot ({qid},{sidx}) step={step}: {e}")

            L_neutral_val = g_norm_neutral = kappa_neutral = float("nan")
            try:
                L_neutral_val, g_norm_neutral, kappa_neutral = _compute_geometry(
                    model, input_ids, neutral_positions, response_start_idx,
                    lora_A_param, lora_B_param, device,
                )
            except Exception as e:
                print(f"  ERROR neutral ({qid},{sidx}) step={step}: {e}")

            record = {
                "step": int(step),
                "question_id": qid,
                "sample_idx": int(sidx),
                "n_pivot": len(pivot_positions),
                "n_neutral": len(neutral_positions),
                "L_pivot": _safe(L_pivot_val),
                "g_norm_pivot": _safe(g_norm_pivot),
                "kappa_pivot": _safe(kappa_pivot),
                "L_neutral": _safe(L_neutral_val),
                "g_norm_neutral": _safe(g_norm_neutral),
                "kappa_neutral": _safe(kappa_neutral),
            }
            _append_jsonl(RESULTS_PATH, record)
            completed.add((step, qid, sidx))

        torch.cuda.empty_cache()
        done_steps += 1
        t_step_elapsed = time.time() - t_step_start
        t_total_elapsed = time.time() - t_loop_start
        remaining = total_steps - done_steps
        avg_per_step = t_total_elapsed / done_steps
        eta_sec = avg_per_step * remaining
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
        print(
            f"[group-level step {step}] 完成，耗时 {t_step_elapsed:.1f}s，"
            f"剩余 {remaining} 个 checkpoint，ETA {eta_str}"
        )


# Permutation test.
def _load_token_records_deduped():
    if not TOKEN_RAW_PATH.exists():
        raise FileNotFoundError(f"找不到 token-level raw 文件: {TOKEN_RAW_PATH}")

    dedup = {}
    malformed = 0
    with open(TOKEN_RAW_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            try:
                key = (
                    int(rec["step"]),
                    rec["question_id"],
                    int(rec["sample_idx"]),
                    rec["label"],
                    int(rec["position"]),
                )
            except KeyError:
                malformed += 1
                continue
            dedup[key] = rec  # last write wins

    if malformed:
        print(f"[permutation] WARNING: 跳过 malformed token records: {malformed}")
    return list(dedup.values())


def _read_expected_token_count_from_meta():
    if not TOKEN_META_PATH.exists():
        return None
    with open(TOKEN_META_PATH) as f:
        meta = json.load(f)
    return meta.get("expected_total_tokens_per_step")


def run_kappa_permutation_test():
    records = _load_token_records_deduped()
    expected_total = _read_expected_token_count_from_meta()

    raw_counts_by_step = defaultdict(int)
    grouped = defaultdict(lambda: defaultdict(list))
    invalid_kappa = 0

    for rec in records:
        step = int(rec["step"])
        raw_counts_by_step[step] += 1
        label = rec.get("label")
        if label not in {"pivot", "neutral"}:
            continue
        kappa = rec.get("kappa_token")
        if _is_missing(kappa):
            invalid_kappa += 1
            continue
        sample_key = (rec["question_id"], int(rec["sample_idx"]))
        grouped[step][sample_key].append({
            "label": label,
            "kappa": float(kappa),
            "question_id": rec["question_id"],
            "sample_idx": int(rec["sample_idx"]),
            "position": int(rec["position"]),
        })

    print(
        f"[permutation] 加载 dedup token records={len(records)}，"
        f"invalid/null kappa={invalid_kappa}"
    )

    results = []
    skipped_steps = []

    for step in sorted(grouped.keys()):
        raw_count = raw_counts_by_step.get(step, 0)
        if REQUIRE_COMPLETE_STEPS_FOR_PERMUTATION and expected_total is not None:
            if raw_count < expected_total:
                skipped_steps.append({
                    "step": step,
                    "reason": "incomplete_raw_token_records",
                    "raw_count": raw_count,
                    "expected_total_tokens_per_step": expected_total,
                })
                print(
                    f"[permutation step {step}] 跳过：raw_count={raw_count} < expected={expected_total}"
                )
                continue

        sample_groups = grouped[step]
        real_pivot_vals = []
        real_neutral_vals = []
        usable_samples = 0
        samples_skipped_missing_label = 0

        # Pre-pack samples into (pivot_vals, neutral_vals, union_vals, n_pivot)
        packed_samples = []
        for sample_key, recs in sample_groups.items():
            pivot_vals = [r["kappa"] for r in recs if r["label"] == "pivot"]
            neutral_vals = [r["kappa"] for r in recs if r["label"] == "neutral"]
            if not pivot_vals or not neutral_vals:
                samples_skipped_missing_label += 1
                continue
            usable_samples += 1
            real_pivot_vals.extend(pivot_vals)
            real_neutral_vals.extend(neutral_vals)
            packed_samples.append((pivot_vals + neutral_vals, len(pivot_vals)))

        if not real_pivot_vals or not real_neutral_vals:
            skipped_steps.append({
                "step": step,
                "reason": "no_usable_pivot_neutral_tokens_after_filtering",
                "raw_count": raw_count,
            })
            print(f"[permutation step {step}] 跳过：无可用 pivot/neutral kappa")
            continue

        true_pivot_mean = _mean(real_pivot_vals)
        true_neutral_mean = _mean(real_neutral_vals)
        true_ratio = _safe_ratio(true_pivot_mean, true_neutral_mean)
        true_diff = float(true_pivot_mean - true_neutral_mean)

        rng = random.Random(PERMUTATION_SEED + int(step) * 1_000_003)
        perm_ratios = []
        perm_diffs = []

        for _ in range(N_PERMUTATIONS):
            fake_pivot_vals = []
            fake_neutral_vals = []
            for union_vals, n_pivot in packed_samples:
                idxs = list(range(len(union_vals)))
                fake_pivot_idxs = set(rng.sample(idxs, n_pivot))
                for j, val in enumerate(union_vals):
                    if j in fake_pivot_idxs:
                        fake_pivot_vals.append(val)
                    else:
                        fake_neutral_vals.append(val)

            p_mean = _mean(fake_pivot_vals)
            n_mean = _mean(fake_neutral_vals)
            diff = None if _is_missing(p_mean) or _is_missing(n_mean) else float(p_mean - n_mean)
            ratio = _safe_ratio(p_mean, n_mean)
            if ratio is not None:
                perm_ratios.append(ratio)
            if diff is not None:
                perm_diffs.append(diff)

        perm_ratio_mean = _mean(perm_ratios)
        perm_ratio_std = _std(perm_ratios)
        perm_diff_mean = _mean(perm_diffs)
        perm_diff_std = _std(perm_diffs)

        z_ratio = None
        if true_ratio is not None and perm_ratio_mean is not None and perm_ratio_std not in (None, 0.0):
            z_ratio = float((true_ratio - perm_ratio_mean) / perm_ratio_std)

        z_diff = None
        if perm_diff_mean is not None and perm_diff_std not in (None, 0.0):
            z_diff = float((true_diff - perm_diff_mean) / perm_diff_std)

        empirical_p_upper_ratio = None
        if true_ratio is not None and perm_ratios:
            empirical_p_upper_ratio = float(
                (sum(1 for r in perm_ratios if r >= true_ratio) + 1) / (len(perm_ratios) + 1)
            )

        empirical_p_upper_diff = None
        if perm_diffs:
            empirical_p_upper_diff = float(
                (sum(1 for d in perm_diffs if d >= true_diff) + 1) / (len(perm_diffs) + 1)
            )

        result = {
            "step": int(step),
            "raw_token_records": int(raw_count),
            "n_samples_usable": int(usable_samples),
            "n_samples_skipped_missing_label": int(samples_skipped_missing_label),
            "n_tokens_total_valid_kappa": int(len(real_pivot_vals) + len(real_neutral_vals)),
            "n_pivot_valid_kappa": int(len(real_pivot_vals)),
            "n_neutral_valid_kappa": int(len(real_neutral_vals)),
            "true_kappa_pivot_mean": _safe(true_pivot_mean),
            "true_kappa_neutral_mean": _safe(true_neutral_mean),
            "true_ratio": _safe(true_ratio),
            "true_diff": _safe(true_diff),
            "perm_ratio_mean": _safe(perm_ratio_mean),
            "perm_ratio_std": _safe(perm_ratio_std),
            "perm_ratio_q025": _safe(_quantile(perm_ratios, 0.025)),
            "perm_ratio_q975": _safe(_quantile(perm_ratios, 0.975)),
            "z_ratio": _safe(z_ratio),
            "empirical_p_upper_ratio": _safe(empirical_p_upper_ratio),
            "n_permutations_requested": int(N_PERMUTATIONS),
            "n_permutations_effective_ratio": int(len(perm_ratios)),
            "perm_diff_mean": _safe(perm_diff_mean),
            "perm_diff_std": _safe(perm_diff_std),
            "perm_diff_q025": _safe(_quantile(perm_diffs, 0.025)),
            "perm_diff_q975": _safe(_quantile(perm_diffs, 0.975)),
            "z_diff": _safe(z_diff),
            "empirical_p_upper_diff": _safe(empirical_p_upper_diff),
            "n_permutations_effective_diff": int(len(perm_diffs)),
        }
        results.append(result)
        print(
            f"[permutation step {step}] true_ratio={result['true_ratio']} "
            f"z_ratio={result['z_ratio']} p_upper={result['empirical_p_upper_ratio']}"
        )

    output = {
        "schema_version": PERM_SCHEMA_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "task_tag": TASK_TAG,
        "run_tag": RUN_TAG,
        "start_ckpt": START_CKPT,
        "end_ckpt": END_CKPT,
        "token_raw_path": str(TOKEN_RAW_PATH),
        "token_metadata_path": str(TOKEN_META_PATH),
        "n_permutations": N_PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "permutation_scope": "within-sample over pivot ∪ neutral tokens",
        "metric": "mean token-level kappa ratio and difference",
        "require_complete_steps": REQUIRE_COMPLETE_STEPS_FOR_PERMUTATION,
        "expected_total_tokens_per_step": expected_total,
        "n_steps_analyzed": len(results),
        "n_steps_skipped": len(skipped_steps),
        "skipped_steps": skipped_steps,
        "steps": results,
    }
    _json_dump(output, PERM_RESULTS_PATH)
    print(f"[permutation] 写入结果: {PERM_RESULTS_PATH}")
    return output


def load_model_and_lora_params(all_steps):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[init] config: {CONFIG_PATH}")
    print(f"[init] 实验名称：{EXPERIMENT_NAME}")
    print(f"[init] 加载 tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    first_adapter_step = next((int(s) for s in all_steps if int(s) > 0), None)
    if first_adapter_step is None:
        raise RuntimeError("需要至少一个 >0 checkpoint adapter 来创建 LoRA 参数结构")
    first_ckpt_path = str(ckpt_dir_fn(first_adapter_step))

    print(f"[init] 加载 base model: {BASE_MODEL}")
    model_load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_load_kwargs)
    print(f"[init] 加载第一个 checkpoint adapter: {first_ckpt_path}")
    model = PeftModel.from_pretrained(model, first_ckpt_path)
    model.eval()

    device = next(model.parameters()).device
    print(f"[init] 模型主设备: {device}")

    lora_A_param = None
    lora_B_param = None
    for name, param in model.named_parameters():
        if "layers" in name:
            if "lora_A" in name:
                if lora_A_param is not None:
                    raise RuntimeError(
                        f"发现多个 lora_A 参数：已有 {lora_A_param.shape}，新增 {name} {param.shape}"
                    )
                lora_A_param = param
                print(f"  lora_A: {name}  shape={param.shape}")
            elif "lora_B" in name:
                if lora_B_param is not None:
                    raise RuntimeError(
                        f"发现多个 lora_B 参数：已有 {lora_B_param.shape}，新增 {name} {param.shape}"
                    )
                lora_B_param = param
                print(f"  lora_B: {name}  shape={param.shape}")

    if lora_A_param is None or lora_B_param is None:
        raise RuntimeError("未找到 lora_A 或 lora_B 参数，请检查模型结构")

    global STEP0_LORA_A
    STEP0_LORA_A = lora_A_param.detach().clone().cpu()

    lora_A_param.requires_grad_(False)
    lora_B_param.requires_grad_(False)
    lora_params = [lora_A_param, lora_B_param]

    n_grad = sum(p.requires_grad for p in model.parameters())
    print(f"[init] model.parameters() 中 requires_grad=True 的数量: {n_grad}（应为 0）")

    return model, tokenizer, lora_A_param, lora_B_param, lora_params, device


def main():
    print(f"[init] 加载 pivot_token_sets.json: {PIVOT_JSON}")
    with open(PIVOT_JSON) as f:
        pivot_data = json.load(f)
    samples_meta = pivot_data["samples"]
    print(f"  samples 总数: {len(samples_meta)}")

    print(f"[init] 加载 em_results.json: {EM_JSON}")
    with open(EM_JSON) as f:
        em_data = json.load(f)

    em_results_map = {}
    for item in em_data["detailed_results"]:
        if not (item.get("is_em") and item.get("is_coherent")):
            continue
        key = (item["question_id"], item["sample_idx"])
        if key in em_results_map:
            print(f"  WARNING: em_results_map 中重复 key={key}，已覆盖")
        em_results_map[key] = {
            "question_text": item["question_text"],
            "response": item["response"],
        }
    print(f"  有效 EM+coherent 样本数（em_results_map）: {len(em_results_map)}")

    if EXPORT_F1_EVAL_METADATA_ONLY:
        export_f1_eval_metadata(samples_meta, em_results_map)
        print("\n✅ F.1 eval metadata 导出完成")
        return

    all_steps_dense = enumerate_checkpoint_steps()
    print(
        f"[init] 共找到 {len(all_steps_dense)} 个有效 checkpoint steps: "
        f"{all_steps_dense[0]} ... {all_steps_dense[-1]}"
    )

    all_steps = filter_sparse_token_steps(all_steps_dense) if COMPUTE_TOKEN_LEVEL else all_steps_dense

    if COMPUTE_GROUP_LEVEL or COMPUTE_TOKEN_LEVEL:
        model, tokenizer, lora_A_param, lora_B_param, lora_params, device = \
            load_model_and_lora_params(all_steps)

        if RUN_SANITY_CHECK:
            print("[sanity check] 开始 ...")
            run_sanity_check(
                model, tokenizer, samples_meta, em_results_map,
                lora_params, device, ckpt_dir_fn, all_steps,
            )

        if COMPUTE_GROUP_LEVEL:
            print("\n[group-level] 开始计算 group-level L/g/kappa ...")
            compute_group_level_geometry(
                model, tokenizer, samples_meta, em_results_map,
                lora_A_param, lora_B_param, device, all_steps,
            )
            print("\n[aggregate] 开始聚合 group-level 结果 ...")
            aggregate_results()

        if COMPUTE_TOKEN_LEVEL:
            print("\n[token-level] 开始计算 per-token L/g/kappa ...")
            compute_token_level_curvature(
                model, tokenizer, samples_meta, em_results_map,
                lora_A_param, lora_B_param, device, all_steps, pivot_data,
            )

    if RUN_PERMUTATION:
        print("\n[permutation] 开始 token-level kappa 置换检验 ...")
        run_kappa_permutation_test()

    print("\n✅ 全部完成！")
    if COMPUTE_GROUP_LEVEL:
        print(f"  group-level 原始结果: {RESULTS_PATH}")
        print(f"  group-level 聚合结果: {AGG_PATH}")
    if COMPUTE_TOKEN_LEVEL or RUN_PERMUTATION:
        print(f"  token-level 原始结果: {TOKEN_RAW_PATH}")
        print(f"  token-level metadata: {TOKEN_META_PATH}")
        print(f"  permutation 结果: {PERM_RESULTS_PATH}")


if __name__ == "__main__":
    main()
