import os
from transformers import AutoModelForCausalLM
import torch
from process import get_dataset,save_image_and_metadata,set_seed
from tqdm import tqdm

from ori_generation_janus import original_generation
from opt_generation_janus import optimized_generation

import argparse
import json

from janus.models import MultiModalityCausalLM, VLChatProcessor
from baseline_milr.metrics import (
    aggregate_metrics,
    build_example_metrics,
    plot_trajectory,
    write_json,
    write_jsonl,
)

### argument parsing function ###
def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Evaluate baseline MILR", parents=[config_parser]
    )
    parser.add_argument("--dataset", type=str, default="prompts/geneval/evaluation_metadata.jsonl", help="Dataset to evaluate")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the model")
    parser.add_argument("--output_dir", type=str, default="../outputs/baseline_milr", help="Path to the output directory")
    parser.add_argument("--data_name", type=str, default="geneval", choices=["geneval", "T2I-CompBench","Wise"], help="Type of dataset to evaluate")
    parser.add_argument("--optimize_mode", type=str, default="text", help="The mode of optimization, must be one of: 'text', 'image', 'both'")
    parser.add_argument("--reward_model_type", type=str, default="geneval", choices=["geneval", "self_reward", "unified_reward","mixed_reward","T2I-CompBench","wise_reward","gpt4o","NVILA"], help="Which reward model to use.")
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    parser.add_argument("--end_data_idx", type=int, default=1319, help="End index of the data to evaluate")
    parser.add_argument("--task_type", type=str, default="color", help="Type of task for T2I-CompBench")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # optimization args
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--text_k", type=float, default=0.1, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--image_k", type=float, default=0.01, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--max_text_steps", type=int, default=10, help="Number of text optimization iterations")
    parser.add_argument("--max_image_steps", type=int, default=10, help="Number of image optimization iterations")
    parser.add_argument("--max_both_steps", type=int, default=10, help="Number of both(text and image) optimization iterations")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Number of generated tokens")
    parser.add_argument("--image_token_num", type=int, default=576)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--cfg_weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)

    # reward model
    parser.add_argument("--reward_threshold", type=float, default=-0.1, help="Threshold for reward to stop optimization")
    parser.add_argument("--milr_16_score", type=float, default=None)

    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")

    if config_args.config is not None:
        with open(config_args.config, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")
        parser.set_defaults(**config)
    return parser.parse_args()


def _selected_budget(args):
    if args.optimize_mode == "text":
        return args.max_text_steps
    if args.optimize_mode in ("image", "image_random"):
        return args.max_image_steps
    return args.max_both_steps


def _legacy_output_dir(args):
    model_name = args.model_name_or_path.split("/")[-1]
    if args.optimize_mode == "text":
        details = f"text_k{args.text_k}-steps{args.max_text_steps}"
    elif args.optimize_mode in ("image", "image_random"):
        details = f"image_k{args.image_k}-steps{args.max_image_steps}"
    else:
        details = f"text_k{args.text_k}-image_k{args.image_k}-steps{args.max_both_steps}"
    return os.path.join(
        args.output_dir,
        f"{model_name}-{args.data_name}-{args.reward_model_type}-{args.optimize_mode}-{details}-lr{args.lr}-reward_threshold{args.reward_threshold}",
    )


def _load_summaries(stage_dir):
    examples_dir = os.path.join(stage_dir, "examples")
    summaries = []
    if not os.path.isdir(examples_dir):
        return summaries
    for name in sorted(os.listdir(examples_dir)):
        example_dir = os.path.join(examples_dir, name)
        metrics_path = os.path.join(example_dir, "metrics.json")
        trajectory_path = os.path.join(example_dir, "trajectory.json")
        if not os.path.isfile(metrics_path) or not os.path.isfile(trajectory_path):
            continue
        with open(metrics_path, "r", encoding="utf-8") as metrics_file:
            summary = json.load(metrics_file)
        with open(trajectory_path, "r", encoding="utf-8") as trajectory_file:
            summary["trajectory"] = json.load(trajectory_file)
        summaries.append(summary)
    return summaries


def main(args):
    if args.model_name_or_path is None:
        raise ValueError("model_name_or_path must be set")
    if _selected_budget(args) < 1:
        raise ValueError("The selected optimization step budget must be at least 1")
    if args.milr_16_score is not None and not 0.0 <= args.milr_16_score <= 1.0:
        raise ValueError("milr_16_score must be between 0 and 1 for GenEval")

    set_seed(args.seed)
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

    if args.reward_model_type == "geneval":
        from rewards.reward import RewardModel

        reward_model = RewardModel(
            model_path="rewards/object_detector",
            object_names_path="rewards/object_names.txt",
            options={"clip_model": "ViT-L-14"},
        )
    elif args.reward_model_type == "self_reward":
        from rewards.self_reward_janus import SelfRewardModel

        reward_model = SelfRewardModel(
            vl_gpt=vl_gpt, vl_chat_processor=vl_chat_processor, device=device
        )
    elif args.reward_model_type == "T2I-CompBench":
        from rewards.T2ICompBench.reward import CompBenchRewardModel

        reward_model = CompBenchRewardModel(task_type=args.task_type, device=device)
    elif args.reward_model_type == "unified_reward":
        from rewards.unified_reward import UnifiedReward

        reward_model = UnifiedReward(
            model_path="CodeGoat24/UnifiedReward-qwen-7b", device=device
        )
    elif args.reward_model_type == "mixed_reward":
        from rewards.MixedReward.reward3 import MixedReward

        reward_model = MixedReward(
            git_ckpt_path="./rewards/MixedReward/reward_weights/git-large-vqav2",
            unified_model_path="CodeGoat24/UnifiedReward-qwen-7b",
            gdino_ckpt_path="./rewards/MixedReward/reward_weights/groundingdino_swint_ogc.pth",
            gdino_config_path="./rewards/MixedReward/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            device=device,
        )
    else:
        raise ValueError(f"Unsupported reward model: {args.reward_model_type}")

    dataset = get_dataset(args.dataset, args.task_type, args.data_name)
    if not dataset:
        raise ValueError("Dataset is empty")
    print(f"Example: {dataset[0]}")

    output_dir = args.output_dir if args.config is not None else _legacy_output_dir(args)
    stage_dir = os.path.join(output_dir, "test") if args.config is not None else output_dir
    os.makedirs(stage_dir, exist_ok=True)
    write_json(os.path.join(output_dir, "config.json"), vars(args))

    original_correct = 0
    optimized_correct = 0
    total = 0
    update_count = 0
    original_length = 0
    optimized_length = 0
    fitten_length = 0
    start_data_idx = max(0, args.start_data_idx)
    end_data_idx = min(args.end_data_idx, len(dataset))
    logistics_path = os.path.join(output_dir, "logistics.pt")
    if args.resume:
        print(f"Resume from {output_dir}")
        logistics = torch.load(logistics_path, map_location="cpu")
        start_data_idx = logistics["start_idx"]
        original_correct = logistics["original_correct"]
        optimized_correct = logistics["optimized_correct"]
        total = logistics["total"]
        update_count = logistics["update_count"]
        original_length = logistics["original_length"]
        optimized_length = logistics["optimized_length"]
        fitten_length = logistics["fitten_length"]

    print(f"Start to evaluate {args.data_name} from {start_data_idx} to {end_data_idx}...")
    for i in tqdm(range(start_data_idx, end_data_idx)):
        example = dataset[i]
        prompt = example["prompt"]
        if prompt is None:
            continue
        source_index = int(example.get("_source_index", i))
        example_dir = os.path.join(stage_dir, "examples", f"{source_index:06d}")
        os.makedirs(example_dir, exist_ok=True)
        print(f"Task_tag: {example['tag']}")
        print(f"prompt: {prompt}")

        (
            img,
            text_hidden_states_list,
            text_final_input_ids,
            image_hidden_states_list,
            image_prompt_embed,
            ori_image_prompt,
        ) = original_generation(
            input_text=prompt,
            model=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            optimize_mode=args.optimize_mode,
            device=device,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            max_text_tokens=args.max_new_tokens,
            image_token_num=args.image_token_num,
            img_size=args.img_size,
            patch_size=args.patch_size,
        )
        save_image_and_metadata(
            img, example, os.path.join(stage_dir, "ori_img"), source_index, args.data_name
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        (
            new_img,
            reward_history,
            ori_total_length,
            generated_seq,
            update_length,
            trajectory,
        ) = optimized_generation(
            reward_model=reward_model,
            image=img,
            data=example,
            model=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            device=device,
            text_hidden_states_list=text_hidden_states_list,
            text_final_input_ids=text_final_input_ids,
            image_hidden_states_list=image_hidden_states_list,
            image_prompt_embed=image_prompt_embed,
            ori_image_prompt=ori_image_prompt,
            max_text_steps=args.max_text_steps,
            max_image_steps=args.max_image_steps,
            max_both_steps=args.max_both_steps,
            lr=args.lr,
            grad_clip=args.grad_clip,
            text_k=args.text_k,
            image_k=args.image_k,
            reward_threshold=args.reward_threshold,
            max_text_tokens=args.max_new_tokens,
            image_token_num=args.image_token_num,
            img_size=args.img_size,
            patch_size=args.patch_size,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            optimize_mode=args.optimize_mode,
            save_base_path=os.path.join(stage_dir, "opt_history", f"{source_index:06d}"),
        )
        final_img = new_img if new_img is not None else img
        if new_img is not None:
            save_image_and_metadata(
                new_img,
                example,
                os.path.join(stage_dir, "opt_img"),
                source_index,
                args.data_name,
            )
        save_image_and_metadata(
            final_img,
            example,
            os.path.join(stage_dir, "final_img"),
            source_index,
            args.data_name,
        )

        metrics = build_example_metrics(
            example,
            source_index,
            reward_history,
            trajectory,
            _selected_budget(args),
            args.data_name,
            args.milr_16_score,
        )
        output_trajectory = metrics.pop("trajectory")
        write_json(os.path.join(example_dir, "metrics.json"), metrics)
        write_json(os.path.join(example_dir, "trajectory.json"), output_trajectory)
        plot_trajectory(
            output_trajectory,
            os.path.join(example_dir, "trajectory.png"),
            f"Baseline MILR test example {source_index}",
        )

        original_correct += metrics["initial_score"]
        optimized_correct += metrics["final_score"]
        update_count += metrics["executed_steps"]
        total += 1
        original_length += ori_total_length
        optimized_length += generated_seq
        fitten_length += (
            generated_seq - update_length if metrics["executed_steps"] > 0 else 0
        )
        torch.save(
            {
                "original_correct": original_correct,
                "optimized_correct": optimized_correct,
                "total": total,
                "start_idx": i + 1,
                "update_count": update_count,
                "original_length": original_length,
                "optimized_length": optimized_length,
                "fitten_length": fitten_length,
            },
            logistics_path,
        )

    summaries = _load_summaries(stage_dir)
    write_jsonl(
        os.path.join(stage_dir, "examples.jsonl"),
        [
            {key: value for key, value in item.items() if key != "trajectory"}
            for item in summaries
        ],
    )
    aggregate = aggregate_metrics(
        summaries,
        _selected_budget(args),
        args.milr_16_score,
        stage_dir,
    )
    write_json(os.path.join(output_dir, "metrics.json"), {"test": aggregate})

    print(f"Original accuracy: {original_correct / total:.4f}")
    print(f"Optimized accuracy: {optimized_correct / total:.4f}")
    print(f"Average update steps: {update_count / total:.4f}")
    print(f"Run outputs: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)
