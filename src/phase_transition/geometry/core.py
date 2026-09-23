"""Utilities for pivot-token loss, gradient, and curvature measurements."""

import math

import torch


def prepare_sample_inputs(sample_meta, em_results_map, tokenizer):
    """Convert one pivot-token record into tokenized model inputs."""
    question_id = sample_meta["question_id"]
    sample_idx  = sample_meta["sample_idx"]
    response_start_idx = sample_meta["response_start_idx"]
    pivot_positions    = sample_meta["pivot_positions"]
    neutral_positions  = sample_meta["neutral_positions"]

    key = (question_id, sample_idx)
    if key not in em_results_map:
        raise KeyError(
            f"em_results_map 中找不到 key={key}，"
            f"请检查 em_results.json 是否包含 is_em=True 且 is_coherent=True 的记录"
        )

    result    = em_results_map[key]
    question  = result["question_text"]
    response  = result["response"]

    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": response},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )

    return input_ids, response_start_idx, pivot_positions, neutral_positions


def compute_position_loss(
    model,
    input_ids,
    positions,
    response_start_idx,
):
    """Return mean negative log probability at response-relative positions."""
    output = model(input_ids=input_ids)
    logits = output.logits  # [1, T, vocab_size]

    log_probs_list = []
    for pos in positions:
        logits_idx = response_start_idx + pos - 1
        token_idx  = response_start_idx + pos
        token_id   = input_ids[0, token_idx].item()
        log_prob   = torch.log_softmax(logits[0, logits_idx, :].float(), dim=-1)[token_id]
        log_probs_list.append(log_prob)

    loss = -torch.stack(log_probs_list).mean()
    return loss


def collect_lora_grads(lora_params):
    """Concatenate LoRA gradients into one float32 CPU vector."""
    parts = []
    for p in lora_params:
        if p.grad is None:
            raise RuntimeError(
                f"参数 {p.shape} 的 .grad 为 None，请确认已完成 backward()"
            )
        parts.append(p.grad.reshape(-1).float().cpu())

    g_vec  = torch.cat(parts)
    g_norm = g_vec.norm().item()
    return g_vec, g_norm


def compute_pivot_kappa(
    model,
    input_ids,
    positions,
    response_start_idx,
    lora_params,
    g_hat,
):
    """Compute directional curvature with a Hessian-vector product."""
    output = model(input_ids=input_ids)
    logits = output.logits  # [1, T, vocab_size]

    log_probs_list = []
    for pos in positions:
        logits_idx = response_start_idx + pos - 1
        token_idx  = response_start_idx + pos
        token_id   = input_ids[0, token_idx].item()
        log_prob   = torch.log_softmax(logits[0, logits_idx, :].float(), dim=-1)[token_id]
        log_probs_list.append(log_prob)

    loss = -torch.stack(log_probs_list).mean()

    grads = torch.autograd.grad(loss, lora_params, create_graph=True)
    grad_vec = torch.cat([g.reshape(-1) for g in grads]).float()
    g_hat_dev = g_hat.to(grad_vec.device)
    grad_dot = (grad_vec * g_hat_dev).sum()
    hvp_grads = torch.autograd.grad(grad_dot, lora_params, retain_graph=False)
    hvp_vec = torch.cat([h.reshape(-1) for h in hvp_grads]).float()
    kappa   = (hvp_vec * g_hat_dev).sum().item()

    return kappa


def run_sanity_check(model, tokenizer, samples_meta, em_results_map,
                     lora_params, device,
                     ckpt_dir_fn, step_list):
    """Validate loss, gradient, and curvature at the first and last checkpoints."""
    import safetensors.torch as safetensors_torch
    from pathlib import Path

    lora_A_param, lora_B_param = lora_params

    step0_lora_A = lora_A_param.detach().clone()

    def _load_ckpt(step):
        """热替换 LoRA 权重到 step 对应 adapter；step=0 表示 raw base forward。"""
        if int(step) == 0:
            lora_A_param.data.copy_(step0_lora_A.to(device=device, dtype=lora_A_param.dtype))
            lora_B_param.data.zero_()
            return

        ckpt_path = ckpt_dir_fn(step)
        st_path   = Path(ckpt_path) / "adapter_model.safetensors"
        bin_path  = Path(ckpt_path) / "adapter_model.bin"

        if st_path.exists():
            state_dict = safetensors_torch.load_file(str(st_path))
        elif bin_path.exists():
            state_dict = torch.load(str(bin_path), map_location="cpu")
        else:
            raise FileNotFoundError(f"sanity check: 找不到 checkpoint {step} 的 adapter 文件")

        for key, tensor in state_dict.items():
            if "lora_A" in key:
                lora_A_param.data.copy_(tensor.to(device=device, dtype=lora_A_param.dtype))
            elif "lora_B" in key:
                lora_B_param.data.copy_(tensor.to(device=device, dtype=lora_B_param.dtype))

    check_steps = [step_list[0], step_list[-1]]
    step_names  = ["theta_0 (step={})".format(step_list[0]),
                   "theta_final (step={})".format(step_list[-1])]
    results_by_step = {}

    for step, name in zip(check_steps, step_names):
        print(f"\n[sanity check] 加载 {name} ...")
        _load_ckpt(step)

        step_results = []
        for meta in samples_meta[:3]:
            qid      = meta["question_id"]
            sidx     = meta["sample_idx"]
            print(f"  样本 ({qid}, {sidx}) ...")

            input_ids, response_start_idx, pivot_positions, neutral_positions = \
                prepare_sample_inputs(meta, em_results_map, tokenizer)
            input_ids = input_ids.to(device)

            for pos, tok_id in zip(meta["pivot_positions"], meta["pivot_token_ids"]):
                actual_id = input_ids[0, response_start_idx + pos].item()
                if actual_id != tok_id:
                    raise RuntimeError(
                        f"Position offset 验证失败：({qid}, {sidx})\n"
                        f"  pivot_pos={pos}, 期望 token_id={tok_id}, 实际={actual_id}\n"
                        f"  response_start_idx={response_start_idx}\n"
                        f"  请检查 tokenizer.apply_chat_template 是否与原始 tokenization 一致"
                    )

            with torch.no_grad():
                L_pivot_val = compute_position_loss(
                    model, input_ids, pivot_positions, response_start_idx,
                ).item()

            with torch.no_grad():
                L_neutral_val = compute_position_loss(
                    model, input_ids, neutral_positions, response_start_idx,
                ).item()

            lora_A_param.requires_grad_(True)
            lora_B_param.requires_grad_(True)
            try:
                model.zero_grad()
                loss_for_grad = compute_position_loss(
                    model, input_ids, pivot_positions, response_start_idx,
                )
                loss_for_grad.backward()
                g_vec_pivot, g_norm_pivot = collect_lora_grads([lora_A_param, lora_B_param])
                g_hat_pivot = g_vec_pivot / (g_vec_pivot.norm() + 1e-12)
                model.zero_grad()

                kappa_pivot = compute_pivot_kappa(
                    model, input_ids, pivot_positions, response_start_idx,
                    [lora_A_param, lora_B_param], g_hat_pivot,
                )
                model.zero_grad()
                torch.cuda.empty_cache()
            finally:
                lora_A_param.requires_grad_(False)
                lora_B_param.requires_grad_(False)
                model.zero_grad()

            step_results.append({
                "qid": qid, "sidx": sidx,
                "L_pivot": L_pivot_val, "L_neutral": L_neutral_val,
                "g_norm_pivot": g_norm_pivot, "kappa_pivot": kappa_pivot,
            })
            print(f"    L_pivot={L_pivot_val:.4f}  L_neutral={L_neutral_val:.4f}  "
                  f"g_norm={g_norm_pivot:.6f}  kappa={kappa_pivot:.6f}")

        results_by_step[step] = step_results

    step0    = step_list[0]
    stepfin  = step_list[-1]
    res0   = results_by_step[step0]
    resfin = results_by_step[stepfin]

    print("\n[sanity check] 开始规则验证 ...")
    for r0, rf in zip(res0, resfin):
        qid = r0["qid"]; sidx = r0["sidx"]
        label = f"({qid}, {sidx})"

        ok1 = rf["L_pivot"] < r0["L_pivot"]
        print(f"  [{'✓' if ok1 else '✗'}] 规则1 L_pivot(final)<L_pivot(base)  {label}  "
              f"{rf['L_pivot']:.4f} < {r0['L_pivot']:.4f}")
        if not ok1:
            print("    WARNING: L_pivot 未下降，可能该样本 EM 不显著")

        delta_pivot   = abs(rf["L_pivot"]   - r0["L_pivot"])
        delta_neutral = abs(rf["L_neutral"] - r0["L_neutral"])
        ok2 = delta_neutral < delta_pivot
        print(f"  [{'✓' if ok2 else '✗'}] 规则2 |ΔL_neutral|<|ΔL_pivot|       {label}  "
              f"|ΔL_neutral|={delta_neutral:.4f}  |ΔL_pivot|={delta_pivot:.4f}")
        if not ok2:
            print("    WARNING: neutral 变化量 >= pivot 变化量，EM 信号可能不够集中")

        ok3 = rf["g_norm_pivot"] > 1e-8
        print(f"  [{'✓' if ok3 else '✗'}] 规则3 g_norm_pivot>1e-8             {label}  "
              f"g_norm={rf['g_norm_pivot']:.2e}")
        if not ok3:
            raise RuntimeError(
                f"sanity check 失败：g_norm_pivot={rf['g_norm_pivot']} <= 1e-8，"
                f"样本 {label} 的 lora_params 梯度几乎为零，请检查 requires_grad 设置"
            )

        ok4 = math.isfinite(rf["kappa_pivot"])
        print(f"  [{'✓' if ok4 else '✗'}] 规则4 kappa_pivot 有限值             {label}  "
              f"kappa={rf['kappa_pivot']:.6f}")
        if not ok4:
            raise RuntimeError(
                f"sanity check 失败：kappa_pivot={rf['kappa_pivot']}（NaN/inf），"
                f"样本 {label}，请检查 HVP 实现"
            )

    print("[sanity check] 所有必要验证通过 ✓\n")
