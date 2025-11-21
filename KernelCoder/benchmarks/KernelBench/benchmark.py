import yaml
import os
import torch
import multiprocessing as mp
from datasets import load_dataset
import wandb
from llm_utils import create_llm_client
import sys
import json
from typing import List

from external.KernelBench.src.utils import set_gpu_arch, extract_last_code


from KernelCoder.benchmarks.benchmark import Benchmark
from KernelCoder.benchmarks.KernelBench.prompt import prompt_base, get_refinement_prompt
from KernelCoder.benchmarks.KernelBench.evaluation import batch_eval, EvaluationWorkArgs
from KernelCoder.benchmarks.KernelBench.analysis.metrics import compute_metrics_from_traces
from KernelCoder.benchmarks.KernelBench.classes import KernelBenchTask, KernelBenchSolution, KernelBenchEvaluationResult, KernelBenchTraces

class KernelBenchBenchmark(Benchmark):
    def __init__(self, name, run_dir, llm_client, config):
        super().__init__(name, run_dir, llm_client, config)

        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available. Evaluation requires GPU.")

        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")

        set_gpu_arch(config.gpu_arch)
        assert config.num_eval_devices <= torch.cuda.device_count(), f"Number of GPUs requested ({config.num_eval_devices}) is greater than the number of available GPUs ({torch.cuda.device_count()})"

        self.config = config


    def get_prompt(self, task: KernelBenchTask, context: str=None) -> str:
        return prompt_base(task.task_description, context=context)

    def get_refinement_prompt(self, task: KernelBenchTask, trace: KernelBenchTraces, context: str=None) -> str:
        return get_refinement_prompt(task.task_id, task.task_description, trace, self.config, self.run_dir, context=context)

    def parse_solution(self, task, solution_id: str, response: str) -> KernelBenchSolution:
        task_id = task.task_id
        if response is None:
            response = ""
        solution_code = extract_last_code(response, ["python", "cpp"])
        if solution_code is None:
            solution_code = ""
        return KernelBenchSolution(solution_id=solution_id, task_id=task_id, solution_code=solution_code)

    def evaluate_solution(self, traces: KernelBenchTraces) -> KernelBenchTraces:
        """
        For every solution in the traces, if evaluation does not exist, evaluate and add to the trace
        """
        workload = []
        for task in traces.tasks:
            for solution in traces.get_solutions(task.task_id):
                if not traces.get_evaluation(task.task_id, solution.solution_id):
                    workload.append((task, solution))
        results = batch_eval(workload, self.config, self.run_dir, os.path.join(self.run_dir, "eval_results.json"))
        for (evaluation_id, task_id, solution_id, result) in results:
            evaluation = KernelBenchEvaluationResult(evaluation_id=evaluation_id, task_id=task_id, solution_id=solution_id, compiled=result.compiled, correctness=result.correctness, metadata=result.metadata, runtime=result.runtime, runtime_stats=result.runtime_stats)
            traces.add_evaluation(evaluation)
        return traces

    def format_solution(self, solution: KernelBenchSolution, evaluation: KernelBenchEvaluationResult) -> str:
        return f"Kernel: \n{solution.solution_code} \n\nEvaluation: {evaluation.correctness} \n\nRuntime: {evaluation.runtime}"

    def format_trajectory(self, task: KernelBenchTask, solution: KernelBenchSolution, evaluation: KernelBenchEvaluationResult) -> str:
        return f"Task: \n{task.task_description}\n\n{self.format_solution(solution, evaluation)}"

    def analyze(self, trace: KernelBenchTraces) -> dict:
        hardware = getattr(self.config, "hardware", "A6000_babel")
        metrics = compute_metrics_from_traces(trace, hardware, self.config, self.run_dir)
        return metrics
