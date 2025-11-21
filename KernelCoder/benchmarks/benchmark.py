from dataclasses import dataclass
from typing import List

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass(frozen=True)
class Task:
    task_id: str


@dataclass
class Solution:
    solution_id: str
    task_id: str


@dataclass
class EvaluationResult:
    evaluation_id: str
    task_id: str
    solution_id: str


class Traces:
    def __init__(self, tasks, path, solutions=None, evaluations=None):
        self.tasks = tasks
        self.path = path
        self.solutions = solutions if solutions is not None else []
        self.evaluations = evaluations if evaluations is not None else []
        self.load(path)

    def load(self, path):
        pass
    
    def get_solution(self, solution_id: str) -> Solution:
        solutions = [solution for solution in self.solutions if solution.solution_id == solution_id]
        if len(solutions) == 0:
            return None
        return solutions[0]
    
    def get_solutions(self, task_id: str) -> List[Solution]:
        return [solution for solution in self.solutions if solution.task_id == task_id]
    
    def get_evaluations(self, task_id: str) -> List[EvaluationResult]:
        return [evaluation for evaluation in self.evaluations if evaluation.task_id == task_id]
    
    def get_evaluation(self, task_id: str, solution_id: str) -> EvaluationResult:
        evaluations = [evaluation for evaluation in self.evaluations if evaluation.task_id == task_id and evaluation.solution_id == solution_id]
        if len(evaluations) == 0:
            return None
        return evaluations[0]

    def check_for_solution(self, solution_id: str) -> bool:
        return any(solution.solution_id == solution_id for solution in self.solutions)
    
    def add_solution(self, solution: Solution):
        self.solutions.append(solution)
    
    def add_evaluation(self, evaluation: EvaluationResult):
        self.evaluations.append(evaluation)


class Benchmark:
    def __init__(self, name: str, run_dir, llm_client, config):
        """
        Benchmark setup
        """
        self.name = name
        self.run_dir = run_dir
        self.llm_client = llm_client
        self.config = config
    
    def get_prompt(self, task: Task, context:str=None) -> str:
        pass
    
    def get_refinement_prompt(self, task: Task, trace: Traces, context:str=None) -> str:
        pass

    def parse_solution(self, task, solution_id: str, response: str) -> Solution:
        pass

    def evaluate_solution(self, traces: Traces) -> Traces:
        """
        For every solution in the trace, evaluate and return updated trace
        """
        pass

    def format_solution(self, solution: Solution, evaluation: EvaluationResult) -> str:
        """
        Get a string representation of the solution (solution, evaluation result)
        This can be used to extract memory or rules
        """
        pass

    def format_trajectory(self, task: Task, solution: Solution, evaluation: EvaluationResult) -> str:
        """
        Get a string representation of the trajectory (task, solution, evaluation result)
        This can be used to extract memory or rules
        """
        pass

    def analyze(self, trace: Traces) -> dict:
        """
        Analyze evaluation results and return a dictionary of metrics
        """
        pass
