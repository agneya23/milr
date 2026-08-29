import argparse
import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from ori_generation_janus import original_generation
from process import get_dataset, save_image_and_metadata, set_seed

from .meta_milr_optimizer import MetaMilrOptimizer
from .opt_meta_milr import meta_milr_optimized_generation


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train and evaluate Meta-MILR", parents=[config_parser]
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="prompts/geneval/evaluation_metadata.jsonl",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="prompts/geneval/test_metadata.jsonl",
    )
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="../outputs/meta_milr")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_name", type=str, default="geneval")
    parser.add_argument("--optimize_mode", type=str, default="both")
    parser.add_argument("--reward_model_type", type=str, default="geneval")
    parser.add_argument("--task_type", type=str, default="color")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--milr_16_score", type=float, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--num_support", type=int, default=2)
    parser.add_argument("--num_query", type=int, default=1)
    parser.add_argument("--p_z_dim", type=int, default=256)
    parser.add_argument("--p_g_dim", type=int, default=256)
    parser.add_argument("--e_k_dim", type=int, default=128)
    parser.add_argument("--routing_temperature", type=float, default=1.0)
    parser.add_argument("--routing_temperature_min", type=float, default=0.1)
    parser.add_argument("--text_alpha_max", type=float, default=10.0)
    parser.add_argument("--image_alpha_max", type=float, default=10.0)
    parser.add_argument("--text_trust_radius", type=float, default=1.0)
    parser.add_argument("--image_trust_radius", type=float, default=1.0)
    parser.add_argument("--image_token_cost", type=float, default=1.0)
    parser.add_argument("--stop_threshold", type=float, default=0.5)

    parser.add_argument("--lambda_auc", type=float, default=1.0)
    parser.add_argument("--lambda_tok", type=float, default=0.01)
    parser.add_argument("--lambda_step", type=float, default=0.01)
    parser.add_argument("--lambda_drift", type=float, default=0.001)
    parser.add_argument("--lambda_ent", type=float, default=0.001)

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--image_token_num", type=int, default=576)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--cfg_weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    if config_args.config is not None:
        with open(config_args.config, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")
        parser.set_defaults(**config)
    return parser.parse_args()


def _validate_args(args):
    if args.model_name_or_path is None:
        raise ValueError("model_name_or_path must be set")
    if not 0.0 < args.train_frac < 1.0:
        raise ValueError("train_frac must be strictly between 0 and 1")
    if args.epochs < 1 or args.batch_size < 1 or args.budget < 1:
        raise ValueError("epochs, batch_size, and budget must be at least 1")
    if args.num_support < 1 or args.num_query < 1:
        raise ValueError("num_support and num_query must be at least 1")
    if args.optimize_mode not in ("text", "image", "both"):
        raise ValueError("optimize_mode must be text, image, or both")
    if args.data_name != "geneval":
        raise ValueError("This runner currently prepares GenEval splits only")
    temperatures = (
        args.temperature,
        args.routing_temperature,
        args.routing_temperature_min,
    )
    if any(value <= 0 for value in temperatures):
        raise ValueError("temperatures must be positive")
    if args.routing_temperature_min > args.routing_temperature:
        raise ValueError("routing_temperature_min cannot exceed routing_temperature")
    if args.eval_only and args.checkpoint_path is None:
        raise ValueError("eval_only requires checkpoint_path")
    if args.milr_16_score is not None and not 0.0 <= args.milr_16_score <= 1.0:
        raise ValueError("milr_16_score must be between 0 and 1 for GenEval")


def _load_reward_model(args, vl_gpt, vl_chat_processor, device):
    if args.reward_model_type == "geneval":
        from rewards.reward import RewardModel

        return RewardModel(
            model_path="rewards/object_detector",
            object_names_path="rewards/object_names.txt",
            options={"clip_model": "ViT-L-14"},
        )
    if args.reward_model_type == "self_reward":
        from rewards.self_reward_janus import SelfRewardModel

        return SelfRewardModel(
            vl_gpt=vl_gpt, vl_chat_processor=vl_chat_processor, device=device
        )
    if args.reward_model_type == "unified_reward":
        from rewards.unified_reward import UnifiedReward

        return UnifiedReward(
            model_path="CodeGoat24/UnifiedReward-qwen-7b", device=device
        )
    if args.reward_model_type == "mixed_reward":
        from rewards.MixedReward.reward3 import MixedReward

        return MixedReward(
            git_ckpt_path="./rewards/MixedReward/reward_weights/git-large-vqav2",
            unified_model_path="CodeGoat24/UnifiedReward-qwen-7b",
            gdino_ckpt_path="./rewards/MixedReward/reward_weights/groundingdino_swint_ogc.pth",
            gdino_config_path="./rewards/MixedReward/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            device=device,
        )
    raise ValueError(f"Unsupported reward model: {args.reward_model_type}")


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")


def _stratified_geneval_split(dataset, train_frac, seed):
    categories = defaultdict(list)
    for source_index, example in enumerate(dataset):
        item = dict(example)
        item["_source_index"] = source_index
        categories[item["tag"]].append(item)

    generator = random.Random(seed)
    train, val = [], []
    for tag in sorted(categories):
        items = categories[tag]
        if len(items) < 2:
            raise ValueError(f"GenEval category {tag!r} needs at least two examples")
        generator.shuffle(items)
        train_count = round(len(items) * train_frac)
        train_count = min(max(train_count, 1), len(items) - 1)
        train.extend(items[:train_count])
        val.extend(items[train_count:])
    generator.shuffle(train)
    generator.shuffle(val)
    return train, val


def _prepare_datasets(args, run_dir):
    evaluation_data = get_dataset(args.dataset, args.task_type, args.data_name)
    train_data, val_data = _stratified_geneval_split(
        evaluation_data, args.train_frac, args.seed
    )
    test_data = get_dataset(args.test_dataset, args.task_type, args.data_name)
    for source_index, example in enumerate(test_data):
        example["_source_index"] = source_index

    split_dir = os.path.join(run_dir, "splits")
    _write_jsonl(os.path.join(split_dir, "train.jsonl"), train_data)
    _write_jsonl(os.path.join(split_dir, "val.jsonl"), val_data)
    _write_jsonl(os.path.join(split_dir, "test.jsonl"), test_data)
    split_summary = {}
    for split_name, dataset in (
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ):
        category_counts = defaultdict(int)
        for example in dataset:
            category_counts[example["tag"]] += 1
        split_summary[split_name] = {
            "num_examples": len(dataset),
            "categories": dict(sorted(category_counts.items())),
        }
    _write_json(os.path.join(split_dir, "summary.json"), split_summary)
    return train_data, val_data, test_data


def _score_from_reward(reward, data_name):
    return reward + 1.0 if data_name == "geneval" else reward


def _padded_step_scores(reward_history, budget, data_name):
    scores = [_score_from_reward(reward, data_name) for reward in reward_history]
    update_scores = scores[1:] if len(scores) > 1 else [scores[0]]
    update_scores = update_scores[:budget]
    update_scores.extend([update_scores[-1]] * (budget - len(update_scores)))
    return scores[0], update_scores


def _steps_to_threshold(scores, reference_score, fraction):
    if reference_score is None:
        return None
    threshold = fraction * reference_score
    for step, score in enumerate(scores, start=1):
        if score >= threshold:
            return step
    return None


def _plot_trajectory(trajectory, path, title, value_label="Reward"):
    steps = [item["step"] for item in trajectory]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(
        steps,
        [item["reward"] for item in trajectory],
        marker="o",
        label=value_label,
    )
    axes[0, 0].plot(
        steps,
        [item["best_reward"] for item in trajectory],
        marker="o",
        label="Best",
    )
    axes[0, 0].set_title(f"{value_label} trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(
        steps,
        [item["token_cost"] for item in trajectory],
        marker="o",
        label="Token",
    )
    axes[0, 1].plot(
        steps,
        [item["drift_cost"] for item in trajectory],
        marker="o",
        label="Drift",
    )
    axes[0, 1].plot(
        steps,
        [item["routing_entropy"] for item in trajectory],
        marker="o",
        label="Entropy",
    )
    axes[0, 1].set_title("Costs")
    axes[0, 1].legend()

    axes[1, 0].plot(
        steps,
        [item["text_update_ratio"] for item in trajectory],
        marker="o",
        label="Text",
    )
    axes[1, 0].plot(
        steps,
        [item["image_update_ratio"] for item in trajectory],
        marker="o",
        label="Image",
    )
    axes[1, 0].set_title("Token update ratios")
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps, [item["continuation"] for item in trajectory], marker="o"
    )
    axes[1, 1].set_title("Continuation probability")
    axes[1, 1].set_ylim(0.0, 1.0)

    for axis in axes.flat:
        axis.set_xlabel("Refinement step")
        axis.grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _aggregate_stage(stage_name, summaries, budget, milr_16_score, stage_dir):
    if not summaries:
        aggregate = {"split": stage_name, "num_examples": 0}
        _write_json(os.path.join(stage_dir, "metrics.json"), aggregate)
        return aggregate

    score_matrix = np.asarray([item["step_scores"] for item in summaries], dtype=float)
    mean_scores = score_matrix.mean(axis=0).tolist()
    scalar_names = (
        "initial_score",
        "final_score",
        "best_score",
        "initial_reward",
        "final_reward",
        "best_reward",
        "executed_steps",
        "token_cost",
        "drift_cost",
        "routing_entropy",
        "average_text_update_ratio",
        "average_image_update_ratio",
        "reward_model_calls",
        "generated_samples",
    )
    aggregate = {
        "split": stage_name,
        "num_examples": len(summaries),
        "mean_score_by_step": [
            {"step": step, "score": score}
            for step, score in enumerate(mean_scores, start=1)
        ],
        "anytime_auc": float(np.mean(mean_scores)),
        "milr_16_reference_score": milr_16_score,
        "steps_to_95_percent_milr16": _steps_to_threshold(
            mean_scores, milr_16_score, 0.95
        ),
        "steps_to_99_percent_milr16": _steps_to_threshold(
            mean_scores, milr_16_score, 0.99
        ),
    }
    for name in scalar_names:
        aggregate[f"mean_{name}"] = float(np.mean([item[name] for item in summaries]))

    categories = defaultdict(list)
    for item in summaries:
        categories[item["tag"]].append(item)
    category_metrics = {}
    for tag, items in sorted(categories.items()):
        category_scores = np.asarray(
            [item["step_scores"] for item in items], dtype=float
        ).mean(axis=0)
        category_metrics[tag] = {
            "num_examples": len(items),
            "mean_score_by_step": [
                {"step": step, "score": float(score)}
                for step, score in enumerate(category_scores, start=1)
            ],
            "anytime_auc": float(category_scores.mean()),
            "mean_final_score": float(
                np.mean([item["final_score"] for item in items])
            ),
            "mean_best_score": float(
                np.mean([item["best_score"] for item in items])
            ),
        }
    aggregate["categories"] = category_metrics

    initial_score = aggregate["mean_initial_score"]
    score_with_initial = np.column_stack(
        (
            np.asarray([item["initial_score"] for item in summaries], dtype=float),
            score_matrix,
        )
    )
    best_scores = np.maximum.accumulate(score_with_initial, axis=1).mean(axis=0)
    plot_trajectory = [
        {
            "step": 0,
            "reward": initial_score,
            "best_reward": float(best_scores[0]),
            "continuation": 1.0,
            "text_update_ratio": 0.0,
            "image_update_ratio": 0.0,
            "token_cost": 0.0,
            "drift_cost": 0.0,
            "routing_entropy": 0.0,
        }
    ]
    for step in range(1, budget + 1):
        records = [
            item["trajectory"][step]
            for item in summaries
            if step < len(item["trajectory"])
        ]
        mean_step_value = lambda name: float(
            sum(item[name] for item in records) / len(summaries)
        )
        plot_trajectory.append(
            {
                "step": step,
                "reward": mean_scores[step - 1],
                "best_reward": float(best_scores[step]),
                "continuation": mean_step_value("continuation"),
                "text_update_ratio": mean_step_value("text_update_ratio"),
                "image_update_ratio": mean_step_value("image_update_ratio"),
                "token_cost": mean_step_value("token_cost"),
                "drift_cost": mean_step_value("drift_cost"),
                "routing_entropy": mean_step_value("routing_entropy"),
            }
        )
    aggregate["mean_trajectory_by_step"] = plot_trajectory
    _write_json(os.path.join(stage_dir, "metrics.json"), aggregate)
    with open(
        os.path.join(stage_dir, "metrics.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = (
            "step",
            "mean_score",
            "mean_best_score",
            "mean_continuation",
            "mean_text_update_ratio",
            "mean_image_update_ratio",
            "mean_token_cost",
            "mean_drift_cost",
            "mean_routing_entropy",
        )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in plot_trajectory:
            writer.writerow(
                {
                    "step": item["step"],
                    "mean_score": item["reward"],
                    "mean_best_score": item["best_reward"],
                    "mean_continuation": item["continuation"],
                    "mean_text_update_ratio": item["text_update_ratio"],
                    "mean_image_update_ratio": item["image_update_ratio"],
                    "mean_token_cost": item["token_cost"],
                    "mean_drift_cost": item["drift_cost"],
                    "mean_routing_entropy": item["routing_entropy"],
                }
            )
    _plot_trajectory(
        plot_trajectory,
        os.path.join(stage_dir, "metrics.png"),
        f"Meta-MILR {stage_name}",
        value_label="Score",
    )
    return aggregate


def _run_stage(
    args,
    stage_name,
    dataset,
    stage_dir,
    vl_gpt,
    vl_chat_processor,
    reward_model,
    meta_optimizer,
    outer_optimizer,
    device,
    train_meta,
):
    os.makedirs(stage_dir, exist_ok=True)
    summaries = []
    if train_meta:
        meta_optimizer.train()
    else:
        meta_optimizer.eval()

    for batch_start in tqdm(
        range(0, len(dataset), args.batch_size), desc=stage_name
    ):
        batch = dataset[batch_start : batch_start + args.batch_size]
        if train_meta:
            progress = batch_start / max(len(dataset) - 1, 1)
            meta_optimizer.routing_temperature = args.routing_temperature * (
                args.routing_temperature_min / args.routing_temperature
            ) ** progress
            outer_optimizer.zero_grad(set_to_none=True)

        for example in batch:
            source_index = example["_source_index"]
            example_dir = os.path.join(stage_dir, "examples", f"{source_index:06d}")
            os.makedirs(example_dir, exist_ok=True)
            (
                original_image,
                text_hidden_states_list,
                _,
                image_hidden_states_list,
                _,
                ori_image_prompt,
            ) = original_generation(
                input_text=example["prompt"],
                model=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                optimize_mode=args.optimize_mode,
                device=device,
                max_text_tokens=args.max_new_tokens,
                image_token_num=args.image_token_num,
                img_size=args.img_size,
                patch_size=args.patch_size,
                cfg_weight=args.cfg_weight,
                temperature=args.temperature,
            )

            final_image, reward_history, metrics, trajectory = (
                meta_milr_optimized_generation(
                    meta_milr_optimizer=meta_optimizer,
                    reward_model=reward_model,
                    image=original_image,
                    data=example,
                    model=vl_gpt,
                    vl_chat_processor=vl_chat_processor,
                    device=device,
                    text_hidden_states_list=text_hidden_states_list,
                    image_hidden_states_list=image_hidden_states_list,
                    ori_image_prompt=ori_image_prompt,
                    budget=args.budget,
                    num_support=args.num_support,
                    num_query=args.num_query,
                    lambda_auc=args.lambda_auc,
                    lambda_tok=args.lambda_tok,
                    lambda_step=args.lambda_step,
                    lambda_drift=args.lambda_drift,
                    lambda_ent=args.lambda_ent,
                    cfg_weight=args.cfg_weight,
                    temperature=args.temperature,
                    image_token_num=args.image_token_num,
                    img_size=args.img_size,
                    patch_size=args.patch_size,
                    optimize_mode=args.optimize_mode,
                    stop_threshold=args.stop_threshold,
                    train_meta=train_meta,
                    loss_scale=1.0 / len(batch),
                    example_output_dir=example_dir,
                )
            )
            final_image.save(os.path.join(example_dir, "best.png"))
            save_image_and_metadata(
                original_image,
                example,
                os.path.join(stage_dir, "ori_img"),
                source_index,
                args.data_name,
            )
            save_image_and_metadata(
                final_image,
                example,
                os.path.join(stage_dir, "final_img"),
                source_index,
                args.data_name,
            )

            initial_score, step_scores = _padded_step_scores(
                reward_history, args.budget, args.data_name
            )
            metrics.update(
                {
                    "source_index": source_index,
                    "tag": example["tag"],
                    "prompt": example["prompt"],
                    "initial_image": "images/initial.png",
                    "best_image": "best.png",
                    "initial_score": initial_score,
                    "final_score": _score_from_reward(
                        metrics["final_reward"], args.data_name
                    ),
                    "best_score": _score_from_reward(
                        metrics["best_reward"], args.data_name
                    ),
                    "score_by_step": step_scores,
                    "anytime_auc": float(np.mean(step_scores)),
                    "milr_16_reference_score": args.milr_16_score,
                    "steps_to_95_percent_milr16": _steps_to_threshold(
                        step_scores, args.milr_16_score, 0.95
                    ),
                    "steps_to_99_percent_milr16": _steps_to_threshold(
                        step_scores, args.milr_16_score, 0.99
                    ),
                }
            )
            _write_json(os.path.join(example_dir, "metrics.json"), metrics)
            _write_json(os.path.join(example_dir, "trajectory.json"), trajectory)
            _plot_trajectory(
                trajectory,
                os.path.join(example_dir, "trajectory.png"),
                f"{stage_name} example {source_index}",
            )

            summary = dict(metrics)
            summary["step_scores"] = step_scores
            summary["trajectory"] = trajectory
            summaries.append(summary)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if train_meta:
            torch.nn.utils.clip_grad_norm_(meta_optimizer.parameters(), args.grad_clip)
            outer_optimizer.step()

    _write_jsonl(
        os.path.join(stage_dir, "examples.jsonl"),
        [
            {key: value for key, value in item.items() if key not in ("trajectory", "step_scores")}
            for item in summaries
        ],
    )
    return _aggregate_stage(
        stage_name, summaries, args.budget, args.milr_16_score, stage_dir
    )


def main(args):
    _validate_args(args)
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"{timestamp}_{args.run_name}" if args.run_name else timestamp
    run_dir = os.path.join(args.output_dir, run_label)
    os.makedirs(run_dir, exist_ok=False)
    _write_json(os.path.join(run_dir, "config.json"), vars(args))

    train_data, val_data, test_data = _prepare_datasets(args, run_dir)
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
        args.model_name_or_path
    )
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(device=device, dtype=model_dtype).eval()
    vl_gpt.requires_grad_(False)
    reward_model = _load_reward_model(args, vl_gpt, vl_chat_processor, device)

    meta_optimizer = MetaMilrOptimizer(
        hidden_dim=vl_gpt.language_model.lm_head.in_features,
        p_z_dim=args.p_z_dim,
        p_g_dim=args.p_g_dim,
        e_k_dim=args.e_k_dim,
        routing_temperature=args.routing_temperature,
        text_alpha_max=args.text_alpha_max,
        image_alpha_max=args.image_alpha_max,
        text_trust_radius=args.text_trust_radius,
        image_trust_radius=args.image_trust_radius,
        image_token_cost=args.image_token_cost,
    ).to(device)
    outer_optimizer = torch.optim.AdamW(meta_optimizer.parameters(), lr=args.lr)
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])
        if not args.eval_only and "outer_optimizer" in checkpoint:
            outer_optimizer.load_state_dict(checkpoint["outer_optimizer"])

    run_metrics = {}
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            epoch_dir = os.path.join(run_dir, "train", f"epoch_{epoch:03d}")
            run_metrics[f"train_epoch_{epoch:03d}"] = _run_stage(
                args,
                f"train_epoch_{epoch:03d}",
                train_data,
                epoch_dir,
                vl_gpt,
                vl_chat_processor,
                reward_model,
                meta_optimizer,
                outer_optimizer,
                device,
                train_meta=True,
            )
            torch.save(
                {
                    "epoch": epoch,
                    "meta_optimizer": meta_optimizer.state_dict(),
                    "outer_optimizer": outer_optimizer.state_dict(),
                    "args": vars(args),
                },
                os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"),
            )

    meta_optimizer.routing_temperature = args.routing_temperature_min
    run_metrics["val"] = _run_stage(
        args,
        "val",
        val_data,
        os.path.join(run_dir, "val"),
        vl_gpt,
        vl_chat_processor,
        reward_model,
        meta_optimizer,
        outer_optimizer,
        device,
        train_meta=False,
    )
    run_metrics["test"] = _run_stage(
        args,
        "test",
        test_data,
        os.path.join(run_dir, "test"),
        vl_gpt,
        vl_chat_processor,
        reward_model,
        meta_optimizer,
        outer_optimizer,
        device,
        train_meta=False,
    )
    _write_json(os.path.join(run_dir, "metrics.json"), run_metrics)
    print(f"Run outputs: {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main(parse_args())
