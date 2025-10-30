
import yaml
import os
import torch
import multiprocessing as mp
from datasets import load_dataset
import wandb
from llm_utils import create_llm_client
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(REPO_ROOT, "external")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "KernelCoder"))
sys.path.append(EXTERNAL)

from benchmarks import get_benchmark
from memory import get_memory


def main_loop(config, llm_client, memory, benchmark, dataloader, train=True):
    for i in range(config.epochs):
        for tasks in dataloader:
            tasks: List[Task] = tasks
            # Roll out batch : test time scaling
            context = memory.retrieve(tasks)
            solutions: List[Solution] = benchmark.batch_generate_solution(tasks, solution_ids, run_dir, llm_client, context=context)
            evals: List[EvaluationResult] = benchmark.batch_evaluate_solution(solutions)
            trajectory = benchmark.batch_get_trajectory(tasks, solutions, evals)

            # Extract memory item
            if train:
                new_memory = memory.extract_memory(trajectory) # rule extraction
                memory.add(new_memory)

def main(config):
    config = parse_evolrule_args()

    # set up run directory
    run_dir = os.path.join(RUNS_DIR, config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)

    benchmark, train_dataset, eval_dataset = get_benchmark(config)
    memory = get_memory(config) 

    default_base_api = f"http://{config.vllm_host}:{config.vllm_port}/v1" if config.server_type == "vllm" else None
    llm_client = create_llm_client(os.path.join(run_dir, "llm_usage.json"),
                                   default_model=config.model_name,
                                   default_api_base=default_base_api,
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)
    
