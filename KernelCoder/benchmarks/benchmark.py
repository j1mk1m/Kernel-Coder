from dataclasses import dataclass


@dataclass
class Task:
    task_id: int


@dataclass
class Solution:
    solution_id: int


@dataclass
class EvaluationResult:
    evaluation_id: int


class Benchmark:
    def __init__(self, name: str):
        self.name = name

    def generate_solution(self, task: Task) -> Solution:
        pass

    def evaluate_solution(self, solution: Solution) -> EvaluationResult:
        pass

    def extract_memory(self, solution: Solution, evaluation_result: EvaluationResult) -> MemoryItem:
        pass