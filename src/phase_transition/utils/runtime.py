"""Small runtime helpers shared by the public command-line entry points."""

import json
from pathlib import Path
from typing import Dict, List

import torch


def die(message: str) -> None:
    raise RuntimeError(message)


def guard_overwrite(path: Path, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise RuntimeError(f"Refusing to overwrite existing output: {path}")


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_condition_manifest(path: Path) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("conditions", payload) if isinstance(payload, dict) else payload
    if isinstance(rows, dict):
        return rows
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid condition manifest: {path}")
    return {str(row["condition_id"]): row for row in rows}


def get_adapter_dir(entries: Dict[str, dict], condition_id: str) -> Path:
    if condition_id not in entries:
        raise RuntimeError(f"Condition {condition_id!r} is absent from the manifest")
    value = entries[condition_id].get("adapter_dir")
    if not value:
        raise RuntimeError(f"Condition {condition_id!r} has no adapter_dir")
    return Path(str(value))


def resolve_dtype(cfg: dict) -> torch.dtype:
    value = str(cfg["fixed_response_causal"]["dtype"]).lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise RuntimeError(f"Unsupported dtype: {value}")
    return mapping[value]
