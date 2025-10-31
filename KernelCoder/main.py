
import yaml
import os
import torch
import multiprocessing as mp
from datasets import load_dataset
import wandb
from llm_utils import create_llm_client
import sys
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(REPO_ROOT, "external")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "KernelCoder"))
sys.path.append(EXTERNAL)
sys.path.append(os.path.join(EXTERNAL, "KernelBench"))


from benchmarks import get_benchmark
from memory import get_memory
from test_time_scaling import _batched_generate
from configs import parse_main_args


def main_loop(config, llm_client, memory, benchmark, dataset, trace_cls, train=True):
    traces = trace_cls(dataset, benchmark.run_dir)

    for i in range(config.num_epochs):
        for t in range(0, len(dataset), config.batch_size):
            tasks: List[Task] = dataset[t:t+config.batch_size]
            # Roll out batch : test time scaling
            items = []
            workloads = []
            for task in tasks:
                context = memory.retrieve(task.task_description)
                for sample_id in range(config.num_parallel):
                    sol_name = f"{task.task_id}_epoch_{i}_sample_{sample_id}"
                    workloads.append((task, sol_name))
                    if not traces.check_for_solution(sol_name):
                        prompt = benchmark.get_prompt(task, context)
                        with open(os.path.join(benchmark.run_dir, f"{sol_name}_prompt.txt"), "w") as f:
                            f.write(prompt)
                        items.append((task, sol_name, prompt))
            
            if items:
                _batched_generate(config, benchmark, traces, items, llm_client)
            
            traces = benchmark.evaluate_solution(traces)

            # Extract memory item
            if train:
                trajectories = {}
                for task in tasks:
                    trajectories[task] = []
                for task, solution_name in workloads:
                    solution = traces.get_solution(solution_name)
                    evaluation = traces.get_evaluation(task.task_id, solution_name)
                    trajectory = benchmark.format_solution(solution, evaluation)
                    trajectories[task].append(trajectory)
                memory.extract(i, trajectories)
            llm_client.save_usage_data()


if __name__ == "__main__":
    config = parse_main_args()

    # set up run directory
    run_dir = os.path.join(RUNS_DIR, config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)

    default_base_api = f"http://{config.vllm_host}:{config.vllm_port}/v1" if config.server_type == "vllm" else None
    llm_client = create_llm_client(os.path.join(run_dir, "llm_usage.json"),
                                   default_model=config.model_name,
                                   default_api_base=default_base_api,
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)
     
    benchmark, train_dataset, eval_dataset, trace_cls = get_benchmark(config, run_dir, llm_client)
    memory = get_memory(config, run_dir) 

    main_loop(config, llm_client, memory, benchmark, train_dataset, trace_cls, train=True)
    main_loop(config, llm_client, memory, benchmark, eval_dataset, trace_cls, train=False)
    
