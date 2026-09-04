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
    parser.add_argument("--split_fraction", type=float, default=0.2)
    parser.add_argument("--max_examples_per_split", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
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
    if not 0.0 < args.split_fraction <= 1.0:
        raise ValueError("split_fraction must be greater than 0 and at most 1")
    if args.epochs < 1 or args.batch_size < 1 or args.budget < 1:
        raise ValueError("epochs, batch_size, and budget must be at least 1")
    if args.num_support < 1 or args.num_query < 1:
        raise ValueError("num_support and num_query must be at least 1")
    if args.max_examples_per_split is not None and args.max_examples_per_split < 1:
        raise ValueError("max_examples_per_split must be at least 1")
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
    if args.resume and args.checkpoint_path is None:
        raise ValueError("resume requires checkpoint_path")
    if args.resume and args.eval_only:
        raise ValueError("resume and eval_only cannot be used together")
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


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _save_checkpoint(
    path,
    next_epoch,
    next_batch_start,
    run_dir,
    args,
    meta_optimizer,
    outer_optimizer,
    best_validation_score,
    best_epoch,
):
    checkpoint = {
        "next_epoch": next_epoch,
        "next_batch_start": next_batch_start,
        "run_dir": os.path.abspath(run_dir),
        "args": vars(args),
        "meta_optimizer": meta_optimizer.state_dict(),
        "outer_optimizer": outer_optimizer.state_dict(),
        "best_validation_score": best_validation_score,
        "best_epoch": best_epoch,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary_path = f"{path}.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def _restore_rng_state(checkpoint):
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])


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


def _seeded_subset(dataset, fraction, max_examples, seed):
    sample_size = max(1, round(len(dataset) * fraction))
    if max_examples is not None:
        sample_size = min(sample_size, max_examples)
    if sample_size >= len(dataset):
        return dataset
    generator = random.Random(seed)
    selected_indices = generator.sample(range(len(dataset)), sample_size)
    return [dataset[index] for index in selected_indices]


def _prepare_datasets(args, run_dir):
    evaluation_data = get_dataset(args.dataset, args.task_type, args.data_name)
    train_data, val_data = _stratified_geneval_split(
        evaluation_data, args.train_frac, args.seed
    )
    test_data = get_dataset(args.test_dataset, args.task_type, args.data_name)
    for source_index, example in enumerate(test_data):
        example["_source_index"] = source_index

    source_sizes = {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
    }
    train_data = _seeded_subset(
        train_data, args.split_fraction, args.max_examples_per_split, args.seed + 1
    )
    val_data = _seeded_subset(
        val_data, args.split_fraction, args.max_examples_per_split, args.seed + 2
    )
    test_data = _seeded_subset(
        test_data, args.split_fraction, args.max_examples_per_split, args.seed + 3
    )

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
            "source_num_examples": source_sizes[split_name],
            "categories": dict(sorted(category_counts.items())),
        }
    split_summary["seed"] = args.seed
    split_summary["train_frac"] = args.train_frac
    split_summary["split_fraction"] = args.split_fraction
    split_summary["max_examples_per_split"] = args.max_examples_per_split
    _write_json(os.path.join(split_dir, "summary.json"), split_summary)
    return train_data, val_data, test_data


def _load_run_datasets(args, run_dir):
    split_dir = os.path.join(run_dir, "splits")
    return tuple(
        get_dataset(
            os.path.join(split_dir, f"{split_name}.jsonl"),
            args.task_type,
            args.data_name,
        )
        for split_name in ("train", "val", "test")
    )


def _score_from_reward(reward, data_name):
    return reward + 1.0 if data_name == "geneval" else reward


def _load_stage_summaries(stage_dir, dataset, completed_count):
    summaries = []
    for example in dataset[:completed_count]:
        example_dir = os.path.join(
            stage_dir, "examples", f"{example['_source_index']:06d}"
        )
        metrics_path = os.path.join(example_dir, "metrics.json")
        trajectory_path = os.path.join(example_dir, "trajectory.json")
        if not os.path.isfile(metrics_path) or not os.path.isfile(trajectory_path):
            raise ValueError(f"Missing completed example artifacts in {example_dir}")
        summary = _read_json(metrics_path)
        summary["step_scores"] = summary["score_by_step"]
        summary["trajectory"] = _read_json(trajectory_path)
        summaries.append(summary)
    return summaries


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
    original_cache=None,
    start_index=0,
    on_batch_complete=None,
):
    os.makedirs(stage_dir, exist_ok=True)
    summaries = (
        _load_stage_summaries(stage_dir, dataset, start_index) if start_index else []
    )
    if train_meta:
        meta_optimizer.train()
    else:
        meta_optimizer.eval()

    for batch_start in tqdm(
        range(start_index, len(dataset), args.batch_size), desc=stage_name
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
            cached_original = (
                original_cache.get(source_index) if original_cache is not None else None
            )
            if cached_original is None:
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
                if original_cache is not None:
                    original_cache[source_index] = (
                        original_image.copy(),
                        torch.stack(text_hidden_states_list).detach().cpu(),
                        torch.stack(image_hidden_states_list).detach().cpu(),
                        ori_image_prompt,
                    )
            else:
                (
                    original_image,
                    text_hidden_states_list,
                    image_hidden_states_list,
                    ori_image_prompt,
                ) = cached_original
                original_image = original_image.copy()

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
            if on_batch_complete is not None:
                on_batch_complete(batch_start + len(batch))

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
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        run_dir = resume_checkpoint["run_dir"]
        train_data, val_data, test_data = _load_run_datasets(args, run_dir)
    else:
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
    checkpoint = resume_checkpoint
    if checkpoint is None and args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if checkpoint is not None:
        meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])
        if not args.eval_only and "outer_optimizer" in checkpoint:
            outer_optimizer.load_state_dict(checkpoint["outer_optimizer"])
    if resume_checkpoint is not None:
        _restore_rng_state(resume_checkpoint)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    metrics_path = os.path.join(run_dir, "metrics.json")
    run_metrics = _read_json(metrics_path) if os.path.isfile(metrics_path) else {}
    best_validation_score = (
        resume_checkpoint.get("best_validation_score", float("-inf"))
        if resume_checkpoint is not None
        else float("-inf")
    )
    best_epoch = (
        resume_checkpoint.get("best_epoch") if resume_checkpoint is not None else None
    )
    start_epoch = (
        resume_checkpoint.get("next_epoch", 1) if resume_checkpoint is not None else 1
    )
    start_batch = (
        resume_checkpoint.get("next_batch_start", 0)
        if resume_checkpoint is not None
        else 0
    )
    original_caches = {"train": {}, "val": {}, "test": {}}

    if not args.eval_only:
        run_metrics.pop("test", None)
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_dir = os.path.join(run_dir, "train", f"epoch_{epoch:03d}")

            def save_latest(next_batch_start):
                _save_checkpoint(
                    latest_checkpoint_path,
                    epoch,
                    next_batch_start,
                    run_dir,
                    args,
                    meta_optimizer,
                    outer_optimizer,
                    best_validation_score,
                    best_epoch,
                )

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
                original_cache=original_caches["train"],
                start_index=start_batch if epoch == start_epoch else 0,
                on_batch_complete=save_latest,
            )
            _write_json(metrics_path, run_metrics)

            meta_optimizer.routing_temperature = args.routing_temperature_min
            validation_metrics = _run_stage(
                args,
                f"val_epoch_{epoch:03d}",
                val_data,
                os.path.join(run_dir, "val", f"epoch_{epoch:03d}"),
                vl_gpt,
                vl_chat_processor,
                reward_model,
                meta_optimizer,
                outer_optimizer,
                device,
                train_meta=False,
                original_cache=original_caches["val"],
            )
            run_metrics[f"val_epoch_{epoch:03d}"] = validation_metrics
            validation_score = validation_metrics["mean_best_score"]
            validation_improved = validation_score > best_validation_score
            if validation_improved:
                best_validation_score = validation_score
                best_epoch = epoch
            run_metrics["selection"] = {
                "metric": "mean_best_score",
                "best_epoch": best_epoch,
                "best_validation_score": best_validation_score,
            }
            _write_json(metrics_path, run_metrics)
            if validation_improved:
                _save_checkpoint(
                    best_checkpoint_path,
                    epoch + 1,
                    0,
                    run_dir,
                    args,
                    meta_optimizer,
                    outer_optimizer,
                    best_validation_score,
                    best_epoch,
                )
            _save_checkpoint(
                latest_checkpoint_path,
                epoch + 1,
                0,
                run_dir,
                args,
                meta_optimizer,
                outer_optimizer,
                best_validation_score,
                best_epoch,
            )
            start_batch = 0

        if not os.path.isfile(best_checkpoint_path):
            raise RuntimeError("No best checkpoint was produced")
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        meta_optimizer.load_state_dict(best_checkpoint["meta_optimizer"])
    else:
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
            original_cache=original_caches["val"],
        )

    meta_optimizer.routing_temperature = args.routing_temperature_min
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
        original_cache=original_caches["test"],
    )
    _write_json(metrics_path, run_metrics)
    print(f"Run outputs: {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main(parse_args())
