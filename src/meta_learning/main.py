import argparse
import torch
from ..process import get_dataset, save_image_and_metadata, set_seed
from data import Data
import tqdm
import os
from ..ori_generation_janus import original_generation
from meta_opt_gen_janus import mod_optimized_generation
from meta_learning import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="prompts/geneval/evaluation_metadata.jsonl", help="Dataset to evaluate")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the model")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--dataset_name", type=str, default="geneval", choices=["geneval", "T2I-CompBench","Wise"], help="Type of dataset to evaluate")
    parser.add_argument("--optimize_mode", type=str, default="text", help="The mode of optimization, must be one of: 'text', 'image', 'both'")
    parser.add_argument("--reward_model_type", type=str, default="geneval", choices=["geneval", "self_reward", "unified_reward","mixed_reward","T2I-CompBench","wise_reward","gpt4o","NVILA"], help="Which reward model to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--text_k", type=float, default=0.1, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--image_k", type=float, default=0.01, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--max_text_steps", type=int, default=10, help="Number of text optimization iterations")
    parser.add_argument("--max_image_steps", type=int, default=10, help="Number of image optimization iterations")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Number of generated tokens")
    parser.add_argument("--device", type=str, default=None)
    # parser.add_argument("--meta_tasks_frac", type=float, default=0.5, help="Fraction of tasks to reserve for meta-optimization")
    # parser.add_argument("--meta_tasks_data_frac", type=float, default=0.5, help="Fraction of datapoints to reserve for inner loop of meta-optimization")
    return parser.parse_args()

def main(args):
    if args.seed:
        set_seed(args.seed)
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # load model and tokenizer
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_name_or_path)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    if args.reward_model_type == "geneval":
        from rewards.reward import RewardModel
        reward_model = RewardModel(
            model_path="rewards/object_detector",
            object_names_path="rewards/object_names.txt",
            options={"clip_model": "ViT-L-14"}
        )
    elif args.reward_model_type == "self_reward":
        from rewards.self_reward_janus import SelfRewardModel
        reward_model = SelfRewardModel(vl_gpt=vl_gpt, vl_chat_processor=vl_chat_processor, device=device)
    elif args.reward_model_type == "T2I-CompBench":
        from rewards.T2ICompBench.reward import CompBenchRewardModel
        reward_model = CompBenchRewardModel(task_type=args.task_type, device=device)
    elif args.reward_model_type == "unified_reward":
        from rewards.unified_reward import UnifiedReward
        reward_model = UnifiedReward(
            model_path='CodeGoat24/UnifiedReward-qwen-7b',
            device=device
        )
    elif args.reward_model_type == "mixed_reward":
        from rewards.MixedReward.reward3 import MixedReward
        reward_model = MixedReward(
            git_ckpt_path="./rewards/MixedReward/reward_weights/git-large-vqav2",
            unified_model_path="CodeGoat24/UnifiedReward-qwen-7b",
            gdino_ckpt_path="./rewards/MixedReward/reward_weights/groundingdino_swint_ogc.pth",
            gdino_config_path="./rewards/MixedReward/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            device=device
        )
    elif args.reward_model_type == "NVILA":
        from rewards.NVILA_reward import NVILAReward
        reward_model = NVILAReward(
           model_path="Efficient-Large-Model/NVILA-Lite-2B-Verifier",
           device="cuda:0",
        )
    
    # data_class = Data(args)
    # meta_data_train, meta_data_test = data_class.prepare_data()

    dataset = get_dataset(args.dataset,args.task_type,args.data_name)

    original_correct = 0
    optimized_correct = 0
    total = 0
    update_count = 0
    original_length = 0
    optimized_length = 0
    fitten_length = 0
    model_name = args.model_name_or_path.split("/")[-1]
    data_name = args.data_name

    if args.optimize_mode == "text":
        args.max_text_steps = 30
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-{args.reward_model_type}-{args.optimize_mode}-text_k{args.text_k}-steps{args.max_text_steps}-lr{args.lr}-reward_threshold{args.reward_threshold}"
    elif args.optimize_mode == "image":
        args.max_image_steps = 30
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-{args.reward_model_type}-{args.optimize_mode}-image_k{args.image_k}-steps{args.max_image_steps}-lr{args.lr}-reward_threshold{args.reward_threshold}"
    else:
        args.max_both_steps = 30
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-{args.reward_model_type}-{args.optimize_mode}-text_k{args.text_k}-image_k{args.image_k}-steps{args.max_both_steps}-lr{args.lr}-reward_threshold{args.reward_threshold}"

    for i in tqdm(range(len(dataset))):
        example = dataset[i]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prompt = example["prompt"]
        task_tag = example["tag"]

        if prompt is None:
            continue

        img, text_hidden_states_list, text_final_input_ids, image_hidden_states_list, image_prompt_embed, ori_image_prompt = original_generation(
                input_text=prompt,
                model=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                optimize_mode = args.optimize_mode,
                device=device)
        # save original image and metadata
        if img is not None:
            save_image_and_metadata(img, example, os.path.join(output_dir, "ori_img"), i, data_name)
        torch.cuda.empty_cache()

        meta_milr(
            reward_model=reward_model,
            image=img,
            data=example,
            model=vl_gpt,
            vl_chat_processor = vl_chat_processor,
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
            optimize_mode = args.optimize_mode,
            save_base_path = os.path.join(output_dir, "opt_history", str(i).zfill(4)),
            train_iterations=train_iterations
        )

        # new_img, reward_history, ori_total_length, generated_seq, updated_length, diff_text_states, diff_img_states, text_update_length, img_update_length = mod_optimized_generation(
        #         reward_model=reward_model,
        #         image=img,
        #         data=example,
        #         model=vl_gpt,
        #         vl_chat_processor = vl_chat_processor,
        #         device=device,
        #         text_hidden_states_list=text_hidden_states_list,
        #         text_final_input_ids=text_final_input_ids,
        #         image_hidden_states_list=image_hidden_states_list,
        #         image_prompt_embed=image_prompt_embed,
        #         ori_image_prompt=ori_image_prompt,
        #         max_text_steps=args.max_text_steps,
        #         max_image_steps=args.max_image_steps,
        #         max_both_steps=args.max_both_steps,
        #         lr=args.lr,
        #         grad_clip=args.grad_clip,
        #         text_k=args.text_k,
        #         image_k=args.image_k,
        #         reward_threshold=args.reward_threshold,
        #         max_text_tokens=args.max_new_tokens,
        #         optimize_mode = args.optimize_mode,
        #         save_base_path = os.path.join(output_dir, "opt_history", str(i).zfill(4)),
        # )

        # img, text_hidden_states_list, text_final_input_ids, image_hidden_states_list, image_prompt_embed, ori_image_prompt = meta_learning_func(
        #         reward_model=reward_model,
        #         data=example,
        #         input_text=prompt,
        #         model=vl_gpt,
        #         vl_chat_processor=vl_chat_processor,
        #         optimize_mode = args.optimize_mode,
        #         device=device,
        #         lr=args.lr,
        #         diff_text_states=diff_text_states, 
        #         diff_img_states=diff_img_states,
        #         text_update_length=text_update_length,
        #         img_update_length=img_update_length
        #         )

        update_count += (len(reward_history) - 1)   
        
        # extract answer from model response
        original_length += ori_total_length
        optimized_length += generated_seq
        fitten_length += (generated_seq - update_length) if len(reward_history) > 1 else 0

        # judge answer
        if img is not None:
            original_correct_add = reward_model.judge_answer(img,example)
        else:
            original_correct_add = False
        
        if new_img is not None:
            optimized_correct_add = reward_model.judge_answer(new_img,example)
        else:
            optimized_correct_add = False

        original_correct += original_correct_add
        optimized_correct += optimized_correct_add

        total += 1
        
        # save logistics
        # save original correct, optimized correct, total, update count
        torch.save({
            "original_correct": original_correct,
            "optimized_correct": optimized_correct,
            "total": total,
            "start_idx": i+1,
            "update_count": update_count,
            "original_length": original_length,
            "optimized_length": optimized_length,
            "fitten_length": fitten_length
        }, f"{output_dir}/logistics.pt")

    print(f"Original accuracy: {original_correct / total:.4f}")
    print(f"Optimized accuracy: {optimized_correct / total:.4f}")
    print(f"Average update length: {update_count / total:.4f}")
    print(f"Average original length: {original_length / total:.4f}")
    print(f"Average optimized length: {optimized_length / total:.4f}")
    print(f"Average fitten length: {fitten_length / total:.4f}")   

if __name__ == "__main__":
    args = parse_args()
    main(args)