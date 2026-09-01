import math
import os

import numpy as np
import PIL.Image
import torch


def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _categorical_sample(probabilities, deterministic=False):
    if deterministic:
        return probabilities.argmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


def _entropy(probabilities):
    return -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=-1)


@torch.no_grad()
def _image_prompt_embeddings(model, vl_chat_processor, prompt, device):
    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]
    sft_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt_inputs = vl_chat_processor.tokenizer(
        text=[sft_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=True,
    )
    image_prompt_ids = prompt_inputs["input_ids"].to(device)
    image_start_ids = vl_chat_processor.tokenizer.encode(
        vl_chat_processor.image_start_tag, add_special_tokens=False
    )
    image_prompt_ids = torch.cat(
        (
            image_prompt_ids,
            image_prompt_ids.new_full((image_prompt_ids.shape[0], 1), image_start_ids[-1]),
        ),
        dim=1,
    )

    embedding_layer = model.language_model.get_input_embeddings()
    conditional = embedding_layer(image_prompt_ids)
    pad_ids = image_prompt_ids.new_full((1, 1), vl_chat_processor.pad_id)
    pad_embedding = embedding_layer(pad_ids)
    unconditional = conditional.clone()
    unconditional[:, 1:-1] = pad_embedding
    return torch.cat((conditional, unconditional), dim=0)


@torch.no_grad()
def _generate_image_suffix(
    model,
    prompt_embeddings,
    prefix_token_ids,
    device,
    cfg_weight,
    temperature,
    image_token_num,
    img_size,
    patch_size,
    deterministic=False,
):
    prefix_length = prefix_token_ids.numel()
    generated_tokens = torch.zeros(
        (1, image_token_num), dtype=torch.int, device=device
    )

    if prefix_length:
        generated_tokens[:, :prefix_length] = prefix_token_ids
        paired_ids = prefix_token_ids.unsqueeze(-1).expand(-1, 2).reshape(-1)
        prefix_embeddings = model.prepare_gen_img_embeds(paired_ids)
        prefix_embeddings = prefix_embeddings.reshape(prefix_length, 2, -1).permute(1, 0, 2)
        current_embeddings = torch.cat((prompt_embeddings, prefix_embeddings), dim=1)
    else:
        current_embeddings = prompt_embeddings

    outputs = None
    for position in range(prefix_length, image_token_num):
        outputs = model.language_model.model(
            inputs_embeds=current_embeddings,
            use_cache=True,
            past_key_values=outputs.past_key_values if outputs is not None else None,
        )
        hidden_state = outputs.last_hidden_state[:, -1, :]
        logits = model.gen_head(hidden_state)
        conditional_logits = logits[0::2]
        unconditional_logits = logits[1::2]
        fused_logits = unconditional_logits + cfg_weight * (
            conditional_logits - unconditional_logits
        )
        probabilities = torch.softmax(fused_logits / temperature, dim=-1)
        next_token = _categorical_sample(probabilities, deterministic=deterministic)
        generated_tokens[:, position] = next_token

        paired_token = next_token.unsqueeze(-1).expand(-1, 2).reshape(-1)
        current_embeddings = model.prepare_gen_img_embeds(paired_token).unsqueeze(1)

    decoded = model.gen_vision_model.decode_code(
        generated_tokens,
        shape=[1, 8, img_size // patch_size, img_size // patch_size],
    )
    decoded = decoded.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    decoded = np.clip((decoded + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(decoded[0])


def _sample_generation(
    model,
    vl_chat_processor,
    reward_model,
    data,
    text_hidden_states,
    image_hidden_states,
    ori_image_prompt,
    optimize_mode,
    device,
    cfg_weight,
    temperature,
    image_token_num,
    img_size,
    patch_size,
    deterministic=False,
):
    zero = text_hidden_states.sum() * 0.0 + image_hidden_states.sum() * 0.0
    text_log_probability = zero
    image_log_probability = zero
    text_entropy = text_hidden_states.new_zeros(text_hidden_states.shape[0], dtype=torch.float32)
    image_entropy = image_hidden_states.new_zeros(image_hidden_states.shape[0], dtype=torch.float32)

    if optimize_mode != "image":
        text_logits = model.language_model.lm_head(text_hidden_states)
        text_probabilities = torch.softmax(text_logits.float(), dim=-1)
        text_token_ids = _categorical_sample(
            text_probabilities, deterministic=deterministic
        )
        text_log_probability = torch.log(
            text_probabilities[
                torch.arange(text_hidden_states.shape[0], device=device), text_token_ids
            ]
            + 1e-8
        ).sum()
        text_entropy = _entropy(text_probabilities)
        enhanced_text = vl_chat_processor.tokenizer.decode(
            text_token_ids.detach().cpu().tolist(), skip_special_tokens=True
        )
        image_prompt = f"{data['prompt']}. {enhanced_text}"
    else:
        image_prompt = ori_image_prompt

    prefix_token_ids = torch.empty(0, dtype=torch.long, device=device)
    if optimize_mode != "text":
        prefix_length = min(
            image_hidden_states.shape[0],
            max(1, math.ceil(0.08 * image_hidden_states.shape[0])),
        )
        image_logits = model.gen_head(image_hidden_states[:prefix_length])
        conditional_logits = image_logits[:, 0, :].float()
        unconditional_logits = image_logits[:, 1, :].float()
        fused_logits = unconditional_logits + cfg_weight * (
            conditional_logits - unconditional_logits
        )
        image_probabilities = torch.softmax(fused_logits / temperature, dim=-1)
        prefix_token_ids = _categorical_sample(
            image_probabilities, deterministic=deterministic
        )
        image_log_probability = torch.log(
            image_probabilities[
                torch.arange(prefix_length, device=device), prefix_token_ids
            ]
            + 1e-8
        ).sum()
        image_entropy[:prefix_length] = _entropy(image_probabilities)

    prompt_embeddings = _image_prompt_embeddings(
        model, vl_chat_processor, image_prompt, device
    )
    image = _generate_image_suffix(
        model=model,
        prompt_embeddings=prompt_embeddings,
        prefix_token_ids=prefix_token_ids.detach(),
        device=device,
        cfg_weight=cfg_weight,
        temperature=temperature,
        image_token_num=image_token_num,
        img_size=img_size,
        patch_size=patch_size,
        deterministic=deterministic,
    )
    reward = _as_float(reward_model.get_reward(image, data))
    return {
        "log_probability": text_log_probability + image_log_probability,
        "reward": reward,
        "image": image,
        "text_entropy": text_entropy,
        "image_entropy": image_entropy,
    }


def rollout(
    mode,
    model,
    vl_chat_processor,
    reward_model,
    data,
    text_hidden_states,
    image_hidden_states,
    ori_image_prompt,
    optimize_mode,
    device,
    num_samples,
    cfg_weight,
    temperature,
    image_token_num,
    img_size,
    patch_size,
    initial_reward,
    reward_baseline,
    reward_scale=1.0,
    ema_decay=0.9,
    deterministic=False,
):
    if mode not in ("support", "query"):
        raise ValueError("mode must be 'support' or 'query'")

    if mode == "support":
        rollout_text_states = text_hidden_states.detach().requires_grad_(True)
        rollout_image_states = image_hidden_states.detach().requires_grad_(True)
    else:
        rollout_text_states = text_hidden_states
        rollout_image_states = image_hidden_states

    samples = []
    with torch.enable_grad():
        for _ in range(num_samples):
            samples.append(
                _sample_generation(
                    model=model,
                    vl_chat_processor=vl_chat_processor,
                    reward_model=reward_model,
                    data=data,
                    text_hidden_states=rollout_text_states,
                    image_hidden_states=rollout_image_states,
                    ori_image_prompt=ori_image_prompt,
                    optimize_mode=optimize_mode,
                    device=device,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    image_token_num=image_token_num,
                    img_size=img_size,
                    patch_size=patch_size,
                    deterministic=deterministic,
                )
            )

        rewards = [sample["reward"] for sample in samples]
        if mode == "support":
            mean_reward = float(np.mean(rewards))
            observed_std = float(np.std(rewards))
            if num_samples == 1 or observed_std <= 1e-8:
                advantages = [reward - reward_baseline for reward in rewards]
                std_reward = 1.0
                next_baseline = (
                    ema_decay * reward_baseline + (1.0 - ema_decay) * mean_reward
                )
            else:
                std_reward = observed_std
                denominator = std_reward + 1e-8
                advantages = [(reward - mean_reward) / denominator for reward in rewards]
                next_baseline = reward_baseline

            support_objective = sum(
                advantage * sample["log_probability"]
                for advantage, sample in zip(advantages, samples)
            ) / num_samples
            text_gradient, image_gradient = torch.autograd.grad(
                support_objective,
                (rollout_text_states, rollout_image_states),
                create_graph=False,
                allow_unused=True,
            )
            if text_gradient is None:
                text_gradient = torch.zeros_like(rollout_text_states)
            if image_gradient is None:
                image_gradient = torch.zeros_like(rollout_image_states)

            return {
                "text_gradient": text_gradient.detach(),
                "image_gradient": image_gradient.detach(),
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "reward_baseline": next_baseline,
                "text_entropy": samples[0]["text_entropy"].detach(),
                "image_entropy": samples[0]["image_entropy"].detach(),
                "samples": samples,
            }

        denominator = max(reward_scale, 1e-8)
        query_losses = []
        for sample in samples:
            reward_improvement = (sample["reward"] - initial_reward) / denominator
            query_losses.append(
                -sample["log_probability"] * sample["log_probability"].new_tensor(
                    reward_improvement
                )
            )
        return {
            "loss": torch.stack(query_losses).mean(),
            "mean_reward": float(np.mean(rewards)),
            "samples": samples,
        }


def _latent_drift(text_states, image_states, initial_text_states, initial_image_states):
    text_drift = (text_states.float() - initial_text_states.float()).pow(2).sum()
    image_drift = (image_states.float() - initial_image_states.float()).pow(2).sum() / 2.0
    return (text_drift + image_drift) / max(
        text_states.shape[0] + image_states.shape[0], 1
    )


def _save_rollout_images(samples, step_dir, group):
    group_dir = os.path.join(step_dir, group)
    os.makedirs(group_dir, exist_ok=True)
    paths = []
    for sample_index, sample in enumerate(samples):
        path = os.path.join(group_dir, f"{sample_index:02d}.png")
        sample["image"].save(path)
        paths.append(os.path.relpath(path, os.path.dirname(os.path.dirname(step_dir))))
    return paths


def meta_milr_optimized_generation(
    *,
    meta_milr_optimizer,
    reward_model,
    image,
    data,
    model,
    vl_chat_processor,
    device,
    text_hidden_states_list,
    image_hidden_states_list,
    ori_image_prompt,
    budget=5,
    num_support=2,
    num_query=1,
    lambda_auc=1.0,
    lambda_tok=0.01,
    lambda_step=0.01,
    lambda_drift=0.001,
    lambda_ent=0.001,
    cfg_weight=5.0,
    temperature=1.0,
    image_token_num=576,
    img_size=384,
    patch_size=16,
    optimize_mode="both",
    stop_threshold=0.5,
    train_meta=True,
    loss_scale=1.0,
    example_output_dir=None,
):
    text_hidden_states = torch.stack(
        [state.detach().to(device) for state in text_hidden_states_list], dim=0
    )
    if text_hidden_states.ndim == 3 and text_hidden_states.shape[1] == 1:
        text_hidden_states = text_hidden_states[:, 0, :]
    image_hidden_states = torch.stack(
        [state.detach().to(device) for state in image_hidden_states_list], dim=0
    )

    initial_text_states = text_hidden_states.detach().clone()
    initial_image_states = image_hidden_states.detach().clone()
    text_direction = torch.zeros_like(text_hidden_states, dtype=torch.float32)
    image_direction = torch.zeros_like(image_hidden_states, dtype=torch.float32)

    initial_reward = _as_float(reward_model.get_reward(image, data))
    reward_history = [initial_reward]
    reward_baseline = initial_reward
    current_reward = initial_reward
    previous_reward = initial_reward
    best_reward = initial_reward
    best_image = image
    reward_model_calls = 1
    generated_samples = 1

    if example_output_dir is not None:
        images_dir = os.path.join(example_output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        image.save(os.path.join(images_dir, "initial.png"))

    query_losses = []
    token_cost = text_hidden_states.new_zeros((), dtype=torch.float32)
    step_cost = text_hidden_states.new_zeros((), dtype=torch.float32)
    drift_cost = text_hidden_states.new_zeros((), dtype=torch.float32)
    routing_entropy = text_hidden_states.new_zeros((), dtype=torch.float32)
    executed_steps = 0
    text_update_ratios = []
    image_update_ratios = []
    routing_decisions = []
    trajectory = [
        {
            "step": 0,
            "reward": initial_reward,
            "best_reward": initial_reward,
            "continuation": 1.0,
            "text_update_ratio": 0.0,
            "image_update_ratio": 0.0,
            "token_cost": 0.0,
            "drift_cost": 0.0,
            "routing_entropy": 0.0,
            "support_rewards": [],
            "query_rewards": [],
            "support_images": [],
            "query_images": [],
            "initial_image": "images/initial.png" if example_output_dir else None,
        }
    ]

    for step_index in range(budget):
        support = rollout(
            mode="support",
            model=model,
            vl_chat_processor=vl_chat_processor,
            reward_model=reward_model,
            data=data,
            text_hidden_states=text_hidden_states,
            image_hidden_states=image_hidden_states,
            ori_image_prompt=ori_image_prompt,
            optimize_mode=optimize_mode,
            device=device,
            num_samples=num_support if train_meta else 1,
            cfg_weight=cfg_weight,
            temperature=temperature,
            image_token_num=image_token_num,
            img_size=img_size,
            patch_size=patch_size,
            initial_reward=initial_reward,
            reward_baseline=reward_baseline,
            deterministic=False,
        )
        reward_model_calls += len(support["samples"])
        generated_samples += len(support["samples"])
        step_dir = None
        support_image_paths = []
        if example_output_dir is not None:
            step_dir = os.path.join(
                example_output_dir, "images", f"step_{step_index + 1:02d}"
            )
            support_image_paths = _save_rollout_images(
                support["samples"], step_dir, "support"
            )
        reward_baseline = support["reward_baseline"]
        reward_scale = support["std_reward"] if support["std_reward"] > 1e-8 else 1.0
        reward_feature = (current_reward - support["mean_reward"]) / reward_scale
        reward_delta = (current_reward - previous_reward) / reward_scale

        optimizer_call = lambda: meta_milr_optimizer(
            z_k_t=text_hidden_states,
            z_k_i=image_hidden_states,
            g_k_t=support["text_gradient"],
            g_k_i=support["image_gradient"],
            d_k_t_prev=text_direction,
            d_k_i_prev=image_direction,
            text_entropy=support["text_entropy"],
            image_entropy=support["image_entropy"],
            step_index=step_index,
            budget=budget,
            reward_value=reward_feature,
            reward_delta=reward_delta,
            optimize_mode=optimize_mode,
            deterministic=not train_meta,
        )
        if train_meta:
            (
                next_text_states,
                next_image_states,
                next_text_direction,
                next_image_direction,
                optimizer_stats,
            ) = optimizer_call()
        else:
            with torch.no_grad():
                (
                    next_text_states,
                    next_image_states,
                    next_text_direction,
                    next_image_direction,
                    optimizer_stats,
                ) = optimizer_call()

        continuation = _as_float(optimizer_stats["continuation"])
        both_masks_are_null = (
            _as_float(optimizer_stats["text_mask"].sum()) == 0.0
            and _as_float(optimizer_stats["image_mask"].sum()) == 0.0
        )
        routing_decisions.append(
            {
                "step": step_index + 1,
                "text_mask_indices": optimizer_stats["text_mask"]
                .nonzero()
                .flatten()
                .detach()
                .cpu()
                .tolist(),
                "image_mask_indices": optimizer_stats["image_mask"]
                .nonzero()
                .flatten()
                .detach()
                .cpu()
                .tolist(),
                "both_masks_are_null": both_masks_are_null,
            }
        )
        if not train_meta and both_masks_are_null:
            break

        query = rollout(
            mode="query",
            model=model,
            vl_chat_processor=vl_chat_processor,
            reward_model=reward_model,
            data=data,
            text_hidden_states=next_text_states,
            image_hidden_states=next_image_states,
            ori_image_prompt=ori_image_prompt,
            optimize_mode=optimize_mode,
            device=device,
            num_samples=num_query if train_meta else 1,
            cfg_weight=cfg_weight,
            temperature=temperature,
            image_token_num=image_token_num,
            img_size=img_size,
            patch_size=patch_size,
            initial_reward=initial_reward,
            reward_baseline=reward_baseline,
            reward_scale=reward_scale,
            deterministic=False,
        )
        reward_model_calls += len(query["samples"])
        generated_samples += len(query["samples"])
        query_image_paths = []
        if step_dir is not None:
            query_image_paths = _save_rollout_images(
                query["samples"], step_dir, "query"
            )
        if train_meta:
            query_losses.append(query["loss"])

        previous_reward = current_reward
        current_reward = query["mean_reward"]
        reward_history.append(current_reward)
        for sample in query["samples"]:
            if sample["reward"] > best_reward:
                best_reward = sample["reward"]
                best_image = sample["image"]

        step_token_cost = optimizer_stats["token_cost"]
        step_routing_entropy = optimizer_stats["routing_entropy"]
        step_drift_cost = _latent_drift(
            next_text_states,
            next_image_states,
            initial_text_states,
            initial_image_states,
        )
        text_update_ratio = _as_float(optimizer_stats["text_mask"].mean())
        image_update_ratio = _as_float(optimizer_stats["image_mask"].mean())
        text_update_ratios.append(text_update_ratio)
        image_update_ratios.append(image_update_ratio)

        token_cost = token_cost + step_token_cost
        step_cost = step_cost + optimizer_stats["step_cost"]
        routing_entropy = routing_entropy + step_routing_entropy
        drift_cost = drift_cost + step_drift_cost
        trajectory.append(
            {
                "step": step_index + 1,
                "reward": current_reward,
                "best_reward": best_reward,
                "continuation": continuation,
                "text_update_ratio": text_update_ratio,
                "image_update_ratio": image_update_ratio,
                "token_cost": _as_float(step_token_cost),
                "drift_cost": _as_float(step_drift_cost),
                "routing_entropy": _as_float(step_routing_entropy),
                "support_rewards": [
                    sample["reward"] for sample in support["samples"]
                ],
                "query_rewards": [sample["reward"] for sample in query["samples"]],
                "support_images": support_image_paths,
                "query_images": query_image_paths,
            }
        )

        text_hidden_states = next_text_states
        image_hidden_states = next_image_states
        text_direction = next_text_direction
        image_direction = next_image_direction
        executed_steps += 1

    objective_value = None
    if train_meta and query_losses:
        anytime_loss = torch.stack(query_losses).sum() / budget
        meta_objective = (
            query_losses[-1]
            + lambda_auc * anytime_loss
            + lambda_tok * token_cost
            + lambda_drift * drift_cost
            - lambda_ent * routing_entropy
        )
        objective_value = _as_float(meta_objective)
        (loss_scale * meta_objective).backward()

    metrics = {
        "initial_reward": initial_reward,
        "best_reward": best_reward,
        "final_reward": reward_history[-1],
        "executed_steps": executed_steps,
        "objective": objective_value,
        "token_cost": _as_float(token_cost),
        "drift_cost": _as_float(drift_cost),
        "routing_entropy": _as_float(routing_entropy),
        "average_text_update_ratio": (
            float(np.mean(text_update_ratios)) if text_update_ratios else 0.0
        ),
        "average_image_update_ratio": (
            float(np.mean(image_update_ratios)) if image_update_ratios else 0.0
        ),
        "reward_model_calls": reward_model_calls,
        "generated_samples": generated_samples,
        "routing_decisions": routing_decisions,
    }
    return best_image, reward_history, metrics, trajectory
