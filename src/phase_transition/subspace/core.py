import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


A_DIM = 13824
B_DIM = 5120
FULL_DIM = A_DIM + B_DIM
DEFAULT_LORA_DIMS = {"A": A_DIM, "B": B_DIM, "concat": FULL_DIM}
DEFAULT_LORA_LAYER = 21
DEFAULT_LORA_TARGET_MODULE = "down_proj"


def normalize_lora_dims(dims: dict) -> dict:
    out = {"A": int(dims["A"]), "B": int(dims["B"]), "concat": int(dims["concat"])}
    if out["A"] <= 0 or out["B"] <= 0:
        raise RuntimeError(f"LoRA dims A/B must be positive: {out}")
    if out["concat"] != out["A"] + out["B"]:
        raise RuntimeError(f"LoRA dims concat must equal A+B: {out}")
    return out


def get_lora_dims_from_config(cfg: dict) -> dict:
    try:
        dims = cfg["f1_subspace"]["stat_tests"]["lora_dims"]
    except KeyError as exc:
        raise RuntimeError("Missing f1_subspace.stat_tests.lora_dims config") from exc
    return normalize_lora_dims(dims)


def lora_spec_from_config(cfg: dict) -> dict:
    layers = cfg.get("layers_to_transform")
    if not isinstance(layers, list) or len(layers) != 1:
        raise RuntimeError(f"F.1 requires exactly one layers_to_transform entry, got {layers}")
    module_targets = cfg.get("lora_target_modules")
    parameter_targets = cfg.get("lora_target_parameters")
    module_targets = [] if module_targets is None else module_targets
    parameter_targets = [] if parameter_targets is None else parameter_targets
    if not isinstance(module_targets, list):
        raise RuntimeError(f"lora_target_modules must be a list or null, got {module_targets}")
    if not isinstance(parameter_targets, list):
        raise RuntimeError(f"lora_target_parameters must be a list or null, got {parameter_targets}")
    if bool(module_targets) == bool(parameter_targets):
        raise RuntimeError(
            "F.1 requires exactly one LoRA target mode: set one of "
            "lora_target_modules or lora_target_parameters"
        )
    if module_targets:
        if len(module_targets) != 1:
            raise RuntimeError(
                f"F.1 requires exactly one lora_target_modules entry, got {module_targets}"
            )
        target_kind = "module"
        target_module = str(module_targets[0])
        target_parameter = None
    else:
        if len(parameter_targets) != 1:
            raise RuntimeError(
                f"F.1 requires exactly one lora_target_parameters entry, got {parameter_targets}"
            )
        target_kind = "parameter"
        target_module = None
        target_parameter = str(parameter_targets[0])
        if "." not in target_parameter:
            raise RuntimeError(
                "F.1 lora_target_parameters entry must include its parent module path, "
                f"got {target_parameter!r}"
            )
    return {
        "layer": int(layers[0]),
        "target_kind": target_kind,
        "target_module": target_module,
        "target_parameter": target_parameter,
        "lora_dims": get_lora_dims_from_config(cfg),
    }


def _lora_spec_or_default(lora_spec: dict | None) -> dict:
    if lora_spec is None:
        return {
            "layer": DEFAULT_LORA_LAYER,
            "target_kind": "module",
            "target_module": DEFAULT_LORA_TARGET_MODULE,
            "target_parameter": None,
            "lora_dims": dict(DEFAULT_LORA_DIMS),
        }
    out = dict(lora_spec)
    out["layer"] = int(out["layer"])
    out["target_kind"] = str(out.get("target_kind", "module"))
    if out["target_kind"] == "module":
        out["target_module"] = str(out["target_module"])
        out["target_parameter"] = None
    elif out["target_kind"] == "parameter":
        out["target_module"] = None
        out["target_parameter"] = str(out["target_parameter"])
    else:
        raise RuntimeError(f"Unsupported LoRA target_kind: {out['target_kind']}")
    out["lora_dims"] = normalize_lora_dims(out["lora_dims"])
    return out


def _matches_lora_key(name: str, spec: dict, adapter: str) -> bool:
    spec = _lora_spec_or_default(spec)
    if f"layers.{int(spec['layer'])}." not in name or "weight" not in name:
        return False
    if spec["target_kind"] == "module":
        return f".{spec['target_module']}.lora_{adapter}" in name
    target_parent = spec["target_parameter"].rsplit(".", 1)[0]
    return f"{target_parent}.lora_{adapter}" in name


def lora_target_description(spec: dict) -> str:
    normalized = _lora_spec_or_default(spec)
    if normalized["target_kind"] == "module":
        return f"module={normalized['target_module']}"
    return f"parameter={normalized['target_parameter']}"


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl_dataset(path: Path) -> List[dict]:
    return read_jsonl(path)


def sample_indices(n_items: int, n_train: int, seed: int) -> List[int]:
    if n_items < n_train:
        raise RuntimeError(f"Dataset has {n_items} rows, need n_train={n_train}")
    rng = random.Random(seed)
    return rng.sample(range(n_items), n_train)


def sample_to_messages(sample: dict) -> List[dict]:
    if "messages" in sample:
        return sample["messages"]
    if "question" in sample and "answer" in sample:
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
    if "prompt" in sample and "completion" in sample:
        return [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["completion"]},
        ]
    values = list(sample.values())
    return [
        {"role": "user", "content": str(values[0])},
        {"role": "assistant", "content": str(values[1]) if len(values) > 1 else ""},
    ]


def encode_train_example_full_sequence(tokenizer, sample: dict, max_length: int) -> Dict[str, torch.Tensor]:
    messages = sample_to_messages(sample)
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask
    labels = input_ids.clone()
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
        if torch.any((attention_mask != 0) & (labels == -100)):
            raise RuntimeError("Full-sequence encoder masked a non-padding label")
        if torch.any((attention_mask == 0) & (labels != -100)):
            raise RuntimeError("Full-sequence encoder left a padding label active")
    if not torch.any(labels != -100):
        raise RuntimeError("Full-sequence mask contains no trainable labels")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_eval_sources(pivot_token_sets_path: Path, em_results_path: Path):
    pivot_path = Path(pivot_token_sets_path)
    em_path = Path(em_results_path)
    if not pivot_path.exists() or not em_path.exists():
        raise FileNotFoundError(
            f"Missing eval reconstruction source(s): {pivot_path} or {em_path}"
        )
    pivot_data = read_json(pivot_path)
    em_data = read_json(em_path)
    samples = {}
    for sample in pivot_data["samples"]:
        samples[(sample["question_id"], int(sample["sample_idx"]))] = sample
    em_map = {}
    for item in em_data["detailed_results"]:
        if not (item.get("is_em") and item.get("is_coherent")):
            continue
        em_map[(item["question_id"], int(item["sample_idx"]))] = {
            "question_text": item["question_text"],
            "response": item["response"],
        }
    return samples, em_map, {"pivot_token_sets_path": str(pivot_path), "em_results_path": str(em_path)}


def reconstruct_eval_input(tokenizer, meta: dict, samples: dict, em_map: dict):
    key = (meta["question_id"], int(meta["sample_idx"]))
    if key not in samples:
        raise KeyError(f"No pivot_token_sets sample for {key}")
    if key not in em_map:
        raise KeyError(f"No EM/coherent response for {key}")
    sample = samples[key]
    source = em_map[key]
    if int(sample["response_start_idx"]) != int(meta["response_start_idx"]):
        raise RuntimeError(f"response_start_idx mismatch for {key}")
    messages = [
        {"role": "user", "content": source["question_text"]},
        {"role": "assistant", "content": source["response"]},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    abs_pos = int(meta["absolute_position"])
    if abs_pos >= input_ids.shape[1]:
        raise RuntimeError(f"absolute_position {abs_pos} outside input length {input_ids.shape[1]}")
    actual = int(input_ids[0, abs_pos].item())
    expected = int(meta["token_id"])
    alt = int(meta.get("actual_token_id", expected))
    if actual not in {expected, alt}:
        raise RuntimeError(
            f"Token mismatch for {key} abs={abs_pos}: reconstructed={actual}, "
            f"token_id={expected}, actual_token_id={alt}"
        )
    if meta.get("token_id_matches_input") is not True:
        raise RuntimeError(f"Metadata token_id_matches_input is not true for {key}")
    return input_ids, int(sample["response_start_idx"])


def token_nll(
    model,
    input_ids: torch.Tensor,
    absolute_position: int,
) -> torch.Tensor:
    if absolute_position <= 0:
        raise RuntimeError("Cannot compute next-token loss for absolute_position <= 0")
    out = model(input_ids=input_ids)
    logits = out.logits[0, absolute_position - 1, :].float()
    token_id = input_ids[0, absolute_position]
    return -torch.log_softmax(logits, dim=-1)[token_id]


def find_lora_params(model, lora_spec: dict | None = None) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    spec = _lora_spec_or_default(lora_spec)
    dims = spec["lora_dims"]
    matched_a = []
    matched_b = []
    lora_keys = []
    for name, param in model.named_parameters():
        if "lora_" in name and "weight" in name:
            lora_keys.append(name)
        if _matches_lora_key(name, spec, "A"):
            matched_a.append((name, param))
        if _matches_lora_key(name, spec, "B"):
            matched_b.append((name, param))
    if len(matched_a) != 1 or len(matched_b) != 1:
        raise RuntimeError(
            f"Expected exactly one LoRA A/B tensor for layer={spec['layer']} "
            f"{lora_target_description(spec)}. matched_A={[k for k, _ in matched_a]}, "
            f"matched_B={[k for k, _ in matched_b]}, available_lora_keys={lora_keys}"
        )
    key_a, lora_a = matched_a[0]
    key_b, lora_b = matched_b[0]
    if lora_a.numel() != dims["A"] or lora_b.numel() != dims["B"]:
        raise RuntimeError(
            f"Unexpected LoRA dims: {key_a} A={lora_a.numel()} expected={dims['A']}; "
            f"{key_b} B={lora_b.numel()} expected={dims['B']}"
        )
    return lora_a, lora_b


def freeze_except_lora(model, lora_a, lora_b) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    lora_a.requires_grad_(True)
    lora_b.requires_grad_(True)


def hot_swap_adapter(ckpt_dir: Path, lora_a, lora_b, device, lora_spec: dict | None = None) -> None:
    import safetensors.torch as safetensors_torch

    spec = _lora_spec_or_default(lora_spec)
    dims = spec["lora_dims"]
    st_path = ckpt_dir / "adapter_model.safetensors"
    bin_path = ckpt_dir / "adapter_model.bin"
    if st_path.exists():
        state = safetensors_torch.load_file(str(st_path))
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {ckpt_dir}")
    matched_a = []
    matched_b = []
    lora_keys = [key for key in state.keys() if "lora_" in key and "weight" in key]
    for key, tensor in state.items():
        if _matches_lora_key(key, spec, "A"):
            matched_a.append((key, tensor))
        elif _matches_lora_key(key, spec, "B"):
            matched_b.append((key, tensor))
    if len(matched_a) != 1 or len(matched_b) != 1:
        raise RuntimeError(
            f"Adapter {ckpt_dir} must contain exactly one LoRA A/B tensor for "
            f"layer={spec['layer']} {lora_target_description(spec)}. "
            f"matched_A={[k for k, _ in matched_a]}, matched_B={[k for k, _ in matched_b]}, "
            f"available_lora_keys={lora_keys}"
        )
    key_a, tensor_a = matched_a[0]
    key_b, tensor_b = matched_b[0]
    if tensor_a.numel() != dims["A"] or tensor_b.numel() != dims["B"]:
        raise RuntimeError(
            f"Adapter {ckpt_dir} LoRA tensor size mismatch: "
            f"{key_a} numel={tensor_a.numel()} expected={dims['A']}; "
            f"{key_b} numel={tensor_b.numel()} expected={dims['B']}; "
            f"available_lora_keys={lora_keys}"
        )
    if tensor_a.shape != lora_a.shape or tensor_b.shape != lora_b.shape:
        raise RuntimeError(
            f"Adapter {ckpt_dir} LoRA tensor shape mismatch: "
            f"{key_a} shape={tuple(tensor_a.shape)} target={tuple(lora_a.shape)}; "
            f"{key_b} shape={tuple(tensor_b.shape)} target={tuple(lora_b.shape)}"
        )
    lora_a.data.copy_(tensor_a.to(device=device, dtype=lora_a.dtype))
    lora_b.data.copy_(tensor_b.to(device=device, dtype=lora_b.dtype))


def collect_grad(model, lora_a, lora_b, min_norm: float, lora_spec: dict | None = None) -> Tuple[torch.Tensor, float]:
    if lora_a.grad is None or lora_b.grad is None:
        raise RuntimeError("LoRA gradient is None after backward")
    dims = _lora_spec_or_default(lora_spec)["lora_dims"]
    full_dim = int(dims["concat"])
    g = torch.cat([lora_a.grad.reshape(-1), lora_b.grad.reshape(-1)]).detach().float().cpu()
    if g.numel() != full_dim:
        raise RuntimeError(f"Gradient dim {g.numel()} != {full_dim}")
    if not torch.isfinite(g).all():
        raise RuntimeError("Gradient contains non-finite values")
    norm = float(g.norm().item())
    if norm <= min_norm:
        raise RuntimeError(f"Gradient norm {norm:.3e} <= min_grad_norm={min_norm}")
    model.zero_grad(set_to_none=True)
    return g, norm


def unit_normalize_columns(G: torch.Tensor, eps: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(G.float(), dim=0, keepdim=True)
    return G.float() / torch.clamp(norms, min=eps)


def gradient_alignment(G: torch.Tensor, eps: float) -> float:
    """‖mean_i(g_i/‖g_i‖)‖ over columns of G. See 跨模型因果消融可行性判据 §1.2."""
    if G.numel() == 0 or G.shape[1] == 0:
        raise RuntimeError("gradient_alignment requires at least one column")
    unit_cols = unit_normalize_columns(G, eps)
    return float(unit_cols.mean(dim=1).norm().item())


def block_matrix(G: torch.Tensor, block: str, lora_dims: dict | None = None) -> torch.Tensor:
    dims = normalize_lora_dims(lora_dims or DEFAULT_LORA_DIMS)
    if block == "concat":
        return G
    if block == "A_block":
        return G[: dims["A"]]
    if block == "B_block":
        return G[dims["A"]:]
    raise ValueError(block)


def svd_basis(G: torch.Tensor, eps: float):
    X = unit_normalize_columns(G, eps)
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    return U.cpu(), S.cpu()


def effective_rank(s: torch.Tensor) -> float:
    power = s.float() ** 2
    total = power.sum()
    if total <= 0:
        return float("nan")
    p = power / total
    return float(torch.exp(-(p * torch.log(torch.clamp(p, min=1e-30))).sum()).item())


def cumulative_variance(s: torch.Tensor, ranks: Iterable[int]) -> Dict[int, float]:
    power = s.float() ** 2
    total = power.sum()
    out = {}
    for r in ranks:
        if r <= len(s) and total > 0:
            out[int(r)] = float((power[:r].sum() / total).item())
    return out


def subspace_overlap(U: torch.Tensor, V: torch.Tensor, r: int):
    Ur = U[:, :r].float()
    Vr = V[:, :r].float()
    cosines = torch.linalg.svdvals(Ur.T @ Vr).cpu()
    overlap = float((cosines.square().sum() / r).item())
    return overlap, float(cosines[0].item()), [float(x) for x in cosines]


def random_qr_baseline(dim: int, eval_U: torch.Tensor, r: int, repeats: int, seed: int):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    vals = []
    V = eval_U[:, :r].float()
    for _ in range(repeats):
        R = torch.randn(dim, r, generator=gen)
        Q, _ = torch.linalg.qr(R, mode="reduced")
        vals.append(float(((Q.T @ V).square().sum() / r).item()))
    vals_t = torch.tensor(vals)
    return {
        "mean": float(vals_t.mean().item()),
        "p95": float(torch.quantile(vals_t, 0.95).item()),
        "p99": float(torch.quantile(vals_t, 0.99).item()),
    }


def projection_fraction(mean_g: torch.Tensor, U: torch.Tensor, r: int, eps: float) -> float:
    g = mean_g.float()
    denom = float(g.dot(g).item())
    if denom <= eps:
        return float("nan")
    Ur = U[:, :r].float()
    proj = Ur @ (Ur.T @ g)
    return float((proj.dot(proj) / denom).item())


def projected_cosine(a: torch.Tensor, b: torch.Tensor, U: torch.Tensor, r: int, eps: float) -> float:
    Ur = U[:, :r].float()
    pa = Ur @ (Ur.T @ a.float())
    pb = Ur @ (Ur.T @ b.float())
    denom = float(pa.norm().item() * pb.norm().item())
    if denom <= eps:
        return float("nan")
    return float((pa.dot(pb) / denom).item())

