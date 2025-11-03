from dataclasses import dataclass, field
import glob
import os
import json

from KernelCoder.benchmarks.benchmark import Task, Solution, EvaluationResult, Traces

@dataclass(frozen=True)
class KernelBenchTask(Task):
    task_id: str
    level: int
    problem: int
    task_description: str

@dataclass
class KernelBenchEvaluationResult(EvaluationResult):
    evaluation_id: str # should be same as solution_id
    task_id: str
    solution_id: str
    compiled: bool
    correctness: bool 
    metadata: dict
    runtime: float = -1.0
    runtime_stats: dict = field(default_factory=list)

    def __gt__(self, other):
        if isinstance(other, KernelBenchEvaluationResult):
            if self.correctness and not other.correctness:
                return True
            elif not self.correctness and other.correctness:
                return False
            elif self.runtime != -1.0 and other.runtime != -1.0:
                return self.runtime < other.runtime
            else:
                return False
        return False



@dataclass
class KernelBenchSolution(Solution):
    solution_id: str
    task_id: str
    solution_code: str


def add_to_eval_results_file(evaluation: KernelBenchEvaluationResult, eval_file_path: str):
    """
    Append or update an evaluation result in a JSON file keyed by evaluation_id.
    The file format is a list of entries with fields used by KernelBenchTraces.
    """
    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
    data = []
    if os.path.exists(eval_file_path):
        try:
            with open(eval_file_path, "r") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    data = existing
        except Exception:
            data = []

    entry = {
        "evaluation_id": evaluation.evaluation_id,
        "task_id": evaluation.task_id,
        "solution_id": evaluation.solution_id,
        "compiled": evaluation.compiled,
        "correctness": evaluation.correctness,
        "metadata": evaluation.metadata,
        "runtime": getattr(evaluation, "runtime", -1.0),
        "runtime_stats": getattr(evaluation, "runtime_stats", {}),
    }

    # replace if exists
    replaced = False
    for i, e in enumerate(data):
        if e.get("evaluation_id") == evaluation.evaluation_id:
            data[i] = entry
            replaced = True
            break
    if not replaced:
        data.append(entry)

    with open(eval_file_path, "w") as f:
        json.dump(data, f, indent=2)

class KernelBenchTraces(Traces): 
    def load(self, path):
        # collect all kernel.py files and eval files
        kernel_files = glob.glob(os.path.join(self.path, "*kernel.py"))
        for kernel_file in kernel_files:
            kernel_name = kernel_file.split("/")[-1].split("_kernel.py")[0]
            level = kernel_name.split("_")[1]
            problem_id = kernel_name.split("_")[3]
            solution_id = kernel_name
            with open(kernel_file, "r") as f:
                kernel_src = f.read()
            self.add_solution(KernelBenchSolution(solution_id=solution_id, task_id=f"level_{level}_problem_{problem_id}", solution_code=kernel_src))

        eval_file = os.path.join(self.path, "eval_results.json")
        if os.path.exists(eval_file):
            with open(eval_file, "r") as f:
                eval_results = json.load(f)
            for eval_result in eval_results:
                self.add_evaluation(KernelBenchEvaluationResult(evaluation_id=eval_result["evaluation_id"], task_id=eval_result["task_id"], solution_id=eval_result["solution_id"], compiled=eval_result["compiled"], correctness=eval_result["correctness"], metadata=eval_result["metadata"], runtime=eval_result["runtime"], runtime_stats=eval_result["runtime_stats"]))
    
    def add_solution(self, solution: KernelBenchSolution):
        super().add_solution(solution)
        with open(os.path.join(self.path, f"{solution.solution_id}_kernel.py"), "w") as f:
            f.write(solution.solution_code)
    
    def add_evaluation(self, evaluation: KernelBenchEvaluationResult):
        super().add_evaluation(evaluation)
        add_to_eval_results_file(evaluation, os.path.join(self.path, "eval_results.json"))
    

