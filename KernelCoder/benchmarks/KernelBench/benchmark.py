from dataclasses import dataclass
from KernelCoder.benchmarks.benchmark import Task, Solution, EvaluationResult, Benchmark

@dataclass
class KernelBenchTask(Task):
    task_id: int
    level: int
    problem: int
    task_description: str
    task_input: str
    task_output: str
    task_solution: str
    task_evaluation: str

@dataclass
class KernelBenchSolution(Solution):
    solution_id: int
    solution_code: str
    solution_evaluation: str

@dataclass
class KernelBenchEvaluationResult(EvaluationResult):
    evaluation_id: int
    evaluation_score: float
    evaluation_feedback: str

class KernelBenchBenchmark(Benchmark):
    def __init__(self):
        super().__init__("KernelBench")

    def generate_solution(self, task: KernelBenchTask) -> Solution:
        pass

    def evaluate_solution(self, solution: Solution) -> EvaluationResult:
        pass

    def extract_memory(self, solution: Solution, evaluation_result: EvaluationResult) -> MemoryItem:
        pass