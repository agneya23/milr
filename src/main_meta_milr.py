import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from meta_milr_optimizer import MetaMilrOptimizer
from opt_meta_milr import meta_milr_optimized_generation
from ori_generation_janus import original_generation
from process import get_dataset, save_image_and_metadata, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate Meta-MILR")
    parser.add_argument(
        "--dataset",
        type=str,
        default="prompts/geneval/evaluation_metadata.jsonl",
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--data_name",
        type=str,
        default="geneval",
        choices=["geneval", "T2I-CompBench", "Wise"],
    )
    parser.add_argument(
        "--optimize_mode",
        type=str,
        default="both",
        choices=["text", "image", "both"],
    )
    parser.add_argument(
        "--reward_model_type",
        type=str,
        default="geneval",
        choices=[
            "geneval",
            "self_reward",
            "unified_reward",
            "mixed_reward",
            "T2I-CompBench",
            "wise_reward",
            "gpt4o",
            "NVILA",
        ],
    )
    parser.add_argument("--start_data_idx", type=int, default=0)
    parser.add_argument("--end_data_idx", type=int, default=1319)
    parser.add_argument("--task_type", type=str, default="color")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

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

    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_reward_model(args, vl_gpt, vl_chat_processor, device):
    if args.reward_model_type == "geneval":
        from rewards.reward import RewardModel

        return RewardModel(
            model_path="rewards/<OBJECT_DETECTOR_FOLDER>",
            object_names_path="rewards/object_names.txt",
            options={"clip_model": "ViT-L-14"},
        )
    if args.reward_model_type == "self_reward":
        from rewards.self_reward_janus import SelfRewardModel

        return SelfRewardModel(
            vl_gpt=vl_gpt, vl_chat_processor=vl_chat_processor, device=device
        )
    if args.reward_model_type == "T2I-CompBench":
        from rewards.T2ICompBench.reward import CompBenchRewardModel

        return CompBenchRewardModel(task_type=args.task_type, device=device)
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
    if args.reward_model_type == "wise_reward":
        from rewards.wise_reward import WiseReward

        return WiseReward(api_key="", model="gpt-4o-2024-05-13")
    if args.reward_model_type == "gpt4o":
        from rewards.gpt4o_reward import GPT4oReward

        return GPT4oReward(api_key="", model="gpt-4o-2024-11-20")
    if args.reward_model_type == "NVILA":
        from rewards.NVILA_reward import NVILAReward

        return NVILAReward(
            model_path="Efficient-Large-Model/NVILA-Lite-2B-Verifier",
            device=device,
        )
    raise ValueError(f"Unsupported reward model: {args.reward_model_type}")


def _validate_args(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.budget < 1:
        raise ValueError("budget must be at least 1")
    if args.num_support < 1 or args.num_query < 1:
        raise ValueError("num_support and num_query must be at least 1")
    if (
        args.temperature <= 0
        or args.routing_temperature <= 0
        or args.routing_temperature_min <= 0
    ):
        raise ValueError("temperatures must be positive")
    if args.routing_temperature_min > args.routing_temperature:
        raise ValueError("routing_temperature_min cannot exceed routing_temperature")
    if args.inference and not (args.checkpoint_path or args.resume):
        raise ValueError("inference requires --checkpoint_path or --resume")


def main(args):
    _validate_args(args)
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

    reward_model = _load_reward_model(
        args, vl_gpt, vl_chat_processor, device
    )
    dataset = get_dataset(args.dataset, args.task_type, args.data_name)

    model_name = args.model_name_or_path.rstrip("/").split("/")[-1]
    output_dir = os.path.join(
        args.output_dir,
        f"{model_name}-{args.data_name}-{args.reward_model_type}-meta-{args.optimize_mode}-B{args.budget}",
    )
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, "checkpoint.pt")

    hidden_dim = vl_gpt.language_model.lm_head.in_features
    meta_milr_optimizer = MetaMilrOptimizer(
        hidden_dim=hidden_dim,
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
    adamw_optimizer = torch.optim.AdamW(
        meta_milr_optimizer.parameters(), lr=args.lr
    )

    start_data_idx = max(0, args.start_data_idx)
    totals = {
        "examples": 0,
        "initial_reward": 0.0,
        "final_reward": 0.0,
        "best_reward": 0.0,
        "executed_steps": 0,
    }

    load_path = checkpoint_file if args.resume else args.checkpoint_path
    if load_path is not None:
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        meta_milr_optimizer.load_state_dict(checkpoint["meta_optimizer"])
        if not args.inference and "adamw_optimizer" in checkpoint:
            adamw_optimizer.load_state_dict(checkpoint["adamw_optimizer"])
        if args.resume:
            start_data_idx = checkpoint.get("next_data_idx", start_data_idx)
            totals.update(checkpoint.get("totals", {}))

    if args.inference:
        meta_milr_optimizer.eval()
    else:
        meta_milr_optimizer.train()

    end_data_idx = min(args.end_data_idx, len(dataset))
    data_indices = [
        index
        for index in range(start_data_idx, end_data_idx)
        if dataset[index].get("prompt") is not None
    ]

    for batch_start in tqdm(range(0, len(data_indices), args.batch_size)):
        batch_indices = data_indices[batch_start : batch_start + args.batch_size]
        if not args.inference:
            progress = batch_start / max(len(data_indices) - 1, 1)
            meta_milr_optimizer.routing_temperature = (
                args.routing_temperature
                * (args.routing_temperature_min / args.routing_temperature) ** progress
            )
            adamw_optimizer.zero_grad(set_to_none=True)

        for index in batch_indices:
            example = dataset[index]
            prompt = example["prompt"]
            (
                original_image,
                text_hidden_states_list,
                _,
                image_hidden_states_list,
                _,
                ori_image_prompt,
            ) = original_generation(
                input_text=prompt,
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
            if original_image is None:
                continue

            save_image_and_metadata(
                original_image,
                example,
                os.path.join(output_dir, "ori_img"),
                index,
                args.data_name,
            )

            final_image, reward_history, metrics = meta_milr_optimized_generation(
                meta_milr_optimizer=meta_milr_optimizer,
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
                train_meta=not args.inference,
                loss_scale=1.0 / len(batch_indices),
            )
            save_image_and_metadata(
                final_image,
                example,
                os.path.join(output_dir, "final_img"),
                index,
                args.data_name,
            )

            totals["examples"] += 1
            totals["initial_reward"] += metrics["initial_reward"]
            totals["final_reward"] += metrics["final_reward"]
            totals["best_reward"] += metrics["best_reward"]
            totals["executed_steps"] += metrics["executed_steps"]

            trace_dir = os.path.join(output_dir, "reward_history")
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = os.path.join(trace_dir, f"{index:06d}.pt")
            torch.save(
                {"index": index, "reward_history": reward_history, "metrics": metrics},
                trace_path,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not args.inference:
            torch.nn.utils.clip_grad_norm_(
                meta_milr_optimizer.parameters(), args.grad_clip
            )
            adamw_optimizer.step()

        next_data_idx = batch_indices[-1] + 1
        if not args.inference:
            torch.save(
                {
                    "meta_optimizer": meta_milr_optimizer.state_dict(),
                    "adamw_optimizer": adamw_optimizer.state_dict(),
                    "next_data_idx": next_data_idx,
                    "totals": totals,
                    "args": vars(args),
                },
                checkpoint_file,
            )

    if totals["examples"]:
        count = totals["examples"]
        print(f"Average initial reward: {totals['initial_reward'] / count:.4f}")
        print(f"Average final reward: {totals['final_reward'] / count:.4f}")
        print(f"Average best reward: {totals['best_reward'] / count:.4f}")
        print(f"Average executed steps: {totals['executed_steps'] / count:.4f}")
    else:
        print("No valid examples were processed.")


if __name__ == "__main__":
    parsed_args = parse_args()
    for name, value in vars(parsed_args).items():
        print(f"-- {name}: {value}")
    main(parsed_args)
