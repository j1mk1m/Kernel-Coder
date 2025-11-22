
import yaml
import os
import torch
import multiprocessing as mp
from datasets import load_dataset
from llm_utils import create_llm_client, setup_logging
import sys
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(REPO_ROOT, "external")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "KernelCoder"))
sys.path.append(EXTERNAL)
sys.path.append(os.path.join(EXTERNAL, "KernelBench"))


from benchmarks import get_benchmark, get_dataset, get_trace_cls
from memory import get_memory
from test_time_scaling import _batched_generate
from configs import parse_main_args


def main_loop(config, llm_client, memory, benchmark, dataset, trace_cls, train=True):
    traces = trace_cls(dataset, benchmark.run_dir)
    batch_size = config.batch_size if train else 250

    for i in range(config.num_epochs):
        for t in range(0, len(dataset), batch_size):
            tasks: List[Task] = dataset[t:t+batch_size]
            # Roll out batch : test time scaling
            workloads = []
            for iteration in range(config.num_iterations):
                items = []
                for task in tasks:
                    context = memory.retrieve(task.task_description)
                    for sample_id in range(config.num_parallel):
                        sol_name = f"{task.task_id}_epoch_{i}_sample_{iteration * config.num_parallel + sample_id}"
                        workloads.append((task, sol_name))
                        if not traces.check_for_solution(sol_name):
                            prompt = benchmark.get_prompt(task, context) if iteration == 0 else benchmark.get_refinement_prompt(task, traces, context)
                            with open(os.path.join(benchmark.run_dir, f"{sol_name}_prompt.txt"), "w") as f:
                                f.write(prompt)
                            items.append((task, sol_name, prompt))
                
                if items:
                    _batched_generate(config, benchmark, traces, items, llm_client)
                
                traces = benchmark.evaluate_solution(traces)

            # Extract memory item
            if train:
                trajectories = {}
                for task, solution_name in workloads:
                    solution = traces.get_solution(solution_name)
                    if solution is None:
                        continue
                    evaluation = traces.get_evaluation(task.task_id, solution_name)
                    trajectory = benchmark.format_solution(solution, evaluation)
                    if task.task_id not in trajectories:
                        trajectories[task.task_id] = []
                    trajectories[task.task_id].append((task, solution, evaluation, trajectory))
                memory.extract(i, trajectories, batch_num=t//config.batch_size)
            llm_client.save_usage_data()
    
    metrics = benchmark.analyze(traces)


if __name__ == "__main__":
    config = parse_main_args()

    logger.info(f"Starting main loop with config: {config}")

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

    setup_logging(level="INFO", log_file=os.path.join(run_dir, "usage.log"))
     
    benchmark = get_benchmark(config, run_dir, llm_client)
    train_dataset = get_dataset(config, eval=False)
    trace_cls = get_trace_cls(config)
    memory = get_memory(config, run_dir) 

    main_loop(config, llm_client, memory, benchmark, train_dataset, trace_cls, train=True)

    if config.eval:
        eval_dataset = get_dataset(config, eval=True)
        os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
        benchmark.run_dir = os.path.join(run_dir, "eval")
        main_loop(config, llm_client, memory, benchmark, eval_dataset, trace_cls, train=False)
    
