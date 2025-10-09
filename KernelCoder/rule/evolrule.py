import torch
import multiprocessing as mp
import os
import sys
import json
import yaml
from datasets import load_dataset
from llm_utils import create_llm_client, get_usage_summary

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTERNAL = os.path.join(REPO_ROOT, "external")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
sys.path.append(REPO_ROOT)
sys.path.append(EXTERNAL)

from KernelBench.src.dataset import construct_kernelbench_dataset
from KernelBench.src.utils import set_gpu_arch, create_inference_server_from_presets

from KernelCoder.test_time_scaling import iterative_refinement
from KernelCoder.configs import parse_evolrule_args
from KernelCoder.rule.autorule import autorule


def main(config):
    """
    Test-Time Scaling for Particular Level
    """
    print(f"Starting EvolRule with config: {config}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Evaluation requires GPU.")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # 1. Set up
    # Set up dataset
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    num_problems_in_level = len(curr_level_dataset)

    if config.subset == (None, None):
        problem_id_range = range(1, num_problems_in_level)
    else:
        assert config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level, f"Subset range {config.subset} out of range for Level {config.level}"
        problem_id_range = range(config.subset[0], config.subset[1])

    print(f"Level {config.level} problems: {problem_id_range}")

    # set up run directory
    run_dir = os.path.join(RUNS_DIR, config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)
 
    assert config.store_type == "local", "supporting local file-system based storage for now" # database integreation coming soon, need to migrate from CUDA Monkeys code

    # set GPU arch to configure what target to build for
    set_gpu_arch(config.gpu_arch)
    assert config.num_eval_devices <= torch.cuda.device_count(), f"Number of GPUs requested ({config.num_eval_devices}) is greater than the number of available GPUs ({torch.cuda.device_count()})"

    # Create inference function with config parameters
    default_base_api = f"http://{config.vllm_host}:{config.vllm_port}/v1" if config.server_type == "vllm" else None
    llm_client = create_llm_client(os.path.join(run_dir, "llm_usage.json"),
                                   default_model=config.model_name,
                                   default_api_base=default_base_api,
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)
        
    rule_path = None
    all_rules = []
    for epoch in range(config.num_epochs + 1):
        print(f"[EVOLRULE] Epoch {epoch}")
        epoch_run_dir = os.path.join(run_dir, f"epoch_{epoch}")
        os.makedirs(epoch_run_dir, exist_ok=True)

        # 1. Generation
        llm_client.start_usage_checkpoint(f"epoch_{epoch}")
        llm_client.start_usage_checkpoint("generation")
        generation_dir = os.path.join(epoch_run_dir, "generation")
        os.makedirs(generation_dir, exist_ok=True)
        iterative_refinement(config, config.level, problem_id_range, llm_client, generation_dir, rule_path)
        llm_client.end_usage_checkpoint("generation")
        llm_client.save_usage_data()


        # 2. AutoRule: Comparative Analysis
        if epoch == config.num_epochs:
            llm_client.end_usage_checkpoint(f"epoch_{epoch}")
            break
        llm_client.start_usage_checkpoint("autorule")
        rules = autorule(config, epoch_run_dir, llm_client)
        print(f"Rules: {rules}")
        llm_client.end_usage_checkpoint("autorule")
        llm_client.end_usage_checkpoint(f"epoch_{epoch}")
        llm_client.save_usage_data()

        # 3. Prompt Evolution (add new rules to existing ones)
        rule_path = os.path.join(run_dir, "rules.json")
        
        all_rules.extend(rules)
        with open(rule_path, "w") as f:
            json.dump(all_rules, f, indent=2)
            
    

if __name__ == "__main__":
    args = parse_evolrule_args()
    main(args)
