import argparse
import torch
from ..process import set_seed
from data import Data

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
    parser.add_argument("--meta_tasks_frac", type=float, default=0.5, help="Fraction of tasks to reserve for meta-optimization")
    parser.add_argument("--meta_tasks_data_frac", type=float, default=0.5, help="Fraction of datapoints to reserve for inner loop of meta-optimization")
    return parser.parse_args()

def main(args):
    if args.seed:
        set_seed(args.seed)
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    data_class = Data(args)
    data_class.prepare_data()

if __name__ == "__main__":
    args = parse_args()
    main(args)