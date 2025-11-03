"""
Implements basic test-time scaling approaches
1. best-of-N
2. iterative refinement
3. METR evolutionary approach
4. Stanford: NL idea gen + branching (TODO)
"""

import yaml
import os
import torch
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from datasets import load_dataset
import wandb
from llm_utils import create_llm_client, setup_logging
import sys
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
setup_logging(level="WARNING", log_file="usage.log")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(REPO_ROOT, "external")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "KernelCoder"))
sys.path.insert(0, EXTERNAL)
sys.path.insert(0, os.path.join(EXTERNAL, "KernelBench"))


# KernelBench imports
from KernelBench.src.utils import set_gpu_arch, WorkArgs
from KernelBench.src.dataset import construct_kernelbench_dataset, fetch_ref_arch_from_level_problem_id

# Local imports
from KernelCoder.benchmarks.benchmark import Task, Traces
from configs import parse_test_time_scaling_args 
from benchmarks import get_benchmark


def _batched_generate(config, benchmark, traces, items, llm_client):
    """
    Generate solutions in parallel for a batch of (task, solution_name) pairs.
    items: List[Tuple[Task, str]]
    """
    # Prepare prompts
    prompts = [prompt for _, _, prompt in items]

    # Parallel LLM calls (I/O-bound -> threads are fine and avoid pickling issues)
    def _infer(prompt: str):
        try:
            response = llm_client.text_completion(prompt)["choices"][0]["text"]
            return response
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return None

    num_workers = getattr(config, "request_workers", None) or min(32, len(prompts) or 1)
    with ThreadPool(num_workers) as pool:
        responses = pool.map(_infer, prompts)

    # Parse and add to traces
    for (task, sol_name, prompt), response in zip(items, responses):
        with open(os.path.join(benchmark.run_dir, f"{sol_name}_response.txt"), "w") as f:
            f.write(response if response is not None else "Failed to generate response")
        solution = benchmark.parse_solution(task.task_id, sol_name, response)
        traces.add_solution(solution)


def base(config, benchmark, dataset, trace_cls, llm_client):
    """
    Base approach
    """
    traces = trace_cls(dataset, benchmark.run_dir)
    # Batched single-sample generation for each task
    items = []
    for task in dataset:
        sol_name = f"{task.task_id}_solution_0"
        if not traces.check_for_solution(sol_name):
            items.append((task, sol_name, benchmark.get_prompt(task)))

    if items:
        _batched_generate(config, benchmark, traces, items, llm_client)
    
    traces = benchmark.evaluate_solution(traces)
    metrics = benchmark.analyze(traces)
    

def best_of_n(config, benchmark, dataset, trace_cls, llm_client):
    """
    Best-of-N approach
    Generate num_samples for each problem independently using the new Benchmark API.
    """
    traces = trace_cls(dataset, benchmark.run_dir)
    
    # Build batch of generation items across all tasks and samples
    for sample_id in range(config.num_parallel):
        items = []
        for task in dataset:
            sol_name = f"{task.task_id}_solution_{sample_id}"
            if not traces.check_for_solution(sol_name):
                items.append((task, sol_name, benchmark.get_prompt(task)))

        if items:
            _batched_generate(config, benchmark, traces, items, llm_client)

        traces = benchmark.evaluate_solution(traces)

    metrics = benchmark.analyze(traces)


def iterative_refinement(config, benchmark, dataset, trace_cls, llm_client):
    """
    Iterative refinement approach using the new Benchmark API.
    """

    traces = trace_cls(dataset, benchmark.run_dir)
    num_iterations = config.num_iterations
    for iteration in range(num_iterations):
        logger.info(f"[Iterative Refinement] Iteration {iteration + 1} of {num_iterations}")

        items = []
        for task in dataset:
            for sample_id in range(config.num_parallel):
                sol_name = f"{task.task_id}_solution_{sample_id + iteration * config.num_parallel}"
                if not traces.check_for_solution(sol_name):
                    prompt = benchmark.get_prompt(task) if iteration == 0 else benchmark.get_refinement_prompt(task, traces)
                    items.append((task, sol_name, prompt))

        if items:
            _batched_generate(config, benchmark, traces, items, llm_client)

        # Evaluate after each iteration to inform subsequent refinement rounds
        traces = benchmark.evaluate_solution(traces)

    metrics = benchmark.analyze(traces)


def test_time_scaling(config, run_dir, benchmark, dataset, trace_cls, llm_client):
    """
    Test-Time Scaling for Particular Level
    """
    logger.info(f"Starting Test-Time Scaling with config: {config}")
 
    # Run the test-time scaling approach
    match config.method:
        case "base":
            base(config, benchmark, dataset, trace_cls, llm_client)
        case "best-of-N":
            best_of_n(config, benchmark, dataset, trace_cls, llm_client)
        case "iterative refinement":
            iterative_refinement(config, benchmark, dataset, trace_cls, llm_client)
        case _:
            raise ValueError(f"Invalid method: {config.method}")
 

if __name__ == "__main__": 
    config = parse_test_time_scaling_args()

    # set up run directory
    run_dir = os.path.join(RUNS_DIR, config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)


    # Create inference function with config parameters
    default_base_api = f"http://{config.vllm_host}:{config.vllm_port}/v1" if config.server_type == "vllm" else None
    llm_client = create_llm_client(os.path.join(run_dir, "llm_usage.json"),
                                   default_model=config.model_name,
                                   default_api_base=default_base_api,
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)
    
    benchmark, train_dataset, eval_dataset, trace_cls = get_benchmark(config, run_dir, llm_client)

    test_time_scaling(config, run_dir, benchmark, eval_dataset, trace_cls, llm_client)
    llm_client.save_usage_data()

