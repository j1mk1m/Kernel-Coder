from dataclasses import dataclass
from typing import List, Dict
import re
import os
import glob
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Flashinfer Bench dependencies
from flashinfer_bench import (
    Benchmark as FIBBenchmark,
    BenchmarkConfig,
    BuildSpec,
    Definition,
    EvaluationStatus,
    Solution as FIBSolution,
    SourceFile,
    SupportedLanguages,
    Trace,
    TraceSet,
    Workload,
)
from flashinfer_bench.data import save_json_file, save_jsonl_file, append_jsonl_file, save_json_file, load_json_file, load_jsonl_file
from flashinfer_bench.logging import configure_logging

from KernelCoder.benchmarks.benchmark import Task, Solution, EvaluationResult, Traces, Benchmark
from KernelCoder.benchmarks.flashinferbench.prompts import get_prompt, get_optimization_prompt
from KernelCoder.benchmarks.flashinferbench.metrics import Run, compute_metrics


@dataclass(frozen=True)
class FlashInferBenchTask(Task):
    task_id: str
    definition: Definition
    task_description: str


@dataclass
class FlashInferBenchSolution(Solution):
    solution_id: str
    task_id: str
    solution_code: str
    solution: FIBSolution


@dataclass
class FlashInferBenchEvaluationResult(EvaluationResult):
    evaluation_id: str
    task_id: str
    solution_id: str
    evaluation: Trace

    def __gt__(self, other):
        if isinstance(other, FlashInferBenchEvaluationResult):
            my_eval = self.evaluation.evaluation
            other_eval = other.evaluation.evaluation
            my_correctness = my_eval.status == EvaluationStatus.PASSED
            other_correctness = other_eval.status == EvaluationStatus.PASSED
            if my_correctness and not other_correctness:
                return True
            elif not my_correctness and other_correctness:
                return False
            elif my_eval.performance is not None and other_eval.performance is not None:
                return my_eval.performance.speedup_factor > other_eval.performance.speedup_factor
            else:
                return False 
        return False




def get_code_string(solution: FIBSolution) -> str:
    parts = []
    for src in solution.sources:
        header = f"\n# ===== {src.path} =====\n"
        parts.append(header + src.content)
    return "\n".join(parts).strip()
 

class FlashInferBenchTraces(Traces):
    def load(self, path):
        solution_files = glob.glob(os.path.join(path, "solutions", "*", "*", "*.json"))
        for solution_file in solution_files:
            solution = load_json_file(FIBSolution, solution_file)
            self.add_solution(FlashInferBenchSolution(solution_id=solution.name, task_id=solution.definition, solution=solution, solution_code=get_code_string(solution)))
        eval_files = glob.glob(os.path.join(path, "traces", "*", "*.jsonl"))
        for eval_file in eval_files:
            evals = load_jsonl_file(Trace, eval_file)
            for eval in evals:
                eval_id = eval.solution + "_" + eval.workload.uuid
                self.add_evaluation(FlashInferBenchEvaluationResult(evaluation_id=eval_id, task_id=eval.definition, solution_id=eval.solution, evaluation=eval))

class FlashInferBenchBenchmark(Benchmark):
    def __init__(self, name: str, run_dir, llm_client, config):
        """
        Benchmark setup
        """
        super().__init__(name, run_dir, llm_client, config)
        self.language = config.language
        self.target_gpu = config.target_gpu
        self.base_traceset = TraceSet.from_path(config.base_traceset_path)
        self.author = name
    
    def get_prompt(self, task: Task, context:str=None) -> str:
        return get_prompt(self.language, task.definition, self.target_gpu, context)

    def get_refinement_prompt(self, task: Task, trace: Traces, context:str=None) -> str:
        use_opt = False
        current_code_str = None
        trace_for_opt = None

        evaluations = trace.get_evaluations(task.task_id)
        evaluations = [eval for eval in evaluations if eval.evaluation.evaluation.status == EvaluationStatus.PASSED]
        if len(evaluations) == 0:
            return get_prompt(self.language, task.definition, self.target_gpu, context)
        best_trace = max(evaluations, key=lambda x: x.evaluation.evaluation.performance.speedup_factor)
        if best_trace is not None:
            best_trace = best_trace.evaluation
            sol = trace.get_solution(best_trace.solution).solution
            if sol is not None and sol.sources:
                # Concatenate source files to provide current implementation context
                parts = []
                for src in sol.sources:
                    header = f"\n/* ===== {src.path} ===== */\n" if self.language.lower() == "cuda" else f"\n# ===== {src.path} =====\n"
                    parts.append(header + src.content)
                current_code_str = "\n".join(parts).strip()
                trace_for_opt = best_trace
                use_opt = True
        if use_opt and current_code_str and trace_for_opt:
            return get_optimization_prompt(
                self.language, task.definition, trace_for_opt, current_code_str, self.target_gpu, context
            )
        else:
            return get_prompt(self.language, task.definition, self.target_gpu, context)
    
    def _parse_xml_files(self, code: str) -> Dict[str, str]:
        files = {}
        
        patterns = {
            'kernel.h': r'<header_file name="kernel\.h">(.*?)</header_file>',
            'kernel.cu': r'<cuda_file name="kernel\.cu">(.*?)</cuda_file>',
            'main.cpp': r'<cpp_file name="main\.cpp">(.*?)</cpp_file>'
        }
        
        for filename, pattern in patterns.items():
            match = re.search(pattern, code, re.DOTALL)
            if match:
                content = match.group(1).strip()
                files[filename] = content
            else:
                print(f"Warning: Could not find {filename} in generated code")
        
        return files
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code. For CUDA, parse XML and return dict. For others, clean Python syntax."""
        if self.language.lower() == "cuda":
            return self._parse_xml_files(code)
        
        # For non-CUDA languages (triton, python), clean up markdown and hex floats
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        hex_float_pattern = r"0x[0-9a-fA-F]*\.[0-9a-fA-F]*p[-+]?\d+"
        hex_floats = re.findall(hex_float_pattern, code)

        for hex_float in hex_floats:
            try:
                if hex_float == "0x1.62e42fefa39efp-1":
                    decimal_val = "0.6931471805599453"
                elif hex_float == "0x1.71547652b82fep0":
                    decimal_val = "2.718281828459045"
                elif hex_float == "0x1.921fb54442d18p1":
                    decimal_val = "3.141592653589793"
                else:
                    decimal_val = "1.0"

                code = code.replace(hex_float, decimal_val)
            except Exception as e:
                print(f"Warning: Could not convert hex float {hex_float}: {e}")
                code = code.replace(hex_float, "1.0")

        return code
    
    def _get_supported_language(self) -> SupportedLanguages:
        language_map = {
            "python": SupportedLanguages.PYTHON,
            "triton": SupportedLanguages.TRITON,
            "cuda": SupportedLanguages.CUDA,
        }
        if self.language.lower() in language_map:
            return language_map[self.language.lower()]
        else:
            # Default to Python if unknown language
            return SupportedLanguages.PYTHON
    

    def _create_solution_from_code(
        self, code, definition: Definition, solution_name: str 
    ) -> Solution:
        # Handle different code formats based on language
        if self.language.lower() == "cuda" and isinstance(code, dict):
            # For CUDA, we have multiple files
            sources = []
            for filename, content in code.items():
                sources.append(SourceFile(path=filename, content=content))
            
            entry_point = "main.cpp::run"
        else:
            # For single-file languages (triton, python)
            if isinstance(code, dict):
                code = next(iter(code.values()))
            
            sources = [SourceFile(path="main.py", content=code)]
            entry_point = "main.py::run"

        solution = FIBSolution(
            name=solution_name,
            definition=definition.name,
            author=self.config.model_name,
            spec=BuildSpec(
                language=self._get_supported_language(),
                target_hardware=[self.target_gpu],
                entry_point=entry_point,
            ),
            sources=sources,
            description=solution_name
        )
        return solution

    def parse_solution(self, task, solution_id: str, response: str) -> Solution:
        cleaned_code = self._clean_generated_code(response)
        try:
            solution = self._create_solution_from_code(cleaned_code, task.definition, solution_id)
        except Exception as e:
            logger.error(f"Error creating solution from code: {e}")
            return None
        save_json_file(solution, os.path.join(self.run_dir, "solutions", task.definition.op_type, f"{task.definition.name}", f"{solution_id}.json"))
        return FlashInferBenchSolution(solution_id=solution_id, task_id=task.task_id, solution=solution, solution_code=get_code_string(solution))

    def evaluate_solution(self, traces: Traces) -> Traces:
        """
        For every solution in the trace, evaluate and return updated trace
        """
        # turn Traces into TraceSet
        definitions = {task.definition.name:task.definition for task in traces.tasks}
        solutions = {}
        traceset = {}
        for def_name in definitions.keys():
            solutions[def_name] = [s.solution for s in traces.get_solutions(def_name)]
            traceset[def_name] = [eval_result.evaluation for eval_result in traces.get_evaluations(def_name)]

        traceset = TraceSet(root=self.base_traceset.root, 
                                definitions=definitions, 
                                solutions=solutions, 
                                workloads=self.base_traceset.workloads,
                                traces=traceset) 
        benchmark = FIBBenchmark(traceset, BenchmarkConfig())
        resulting_ts = benchmark.run_all(dump_traces=False, resume=True)
        for def_name, trace_list in resulting_ts.traces.items():
            for trace in trace_list:
                defn = self.base_traceset.definitions[trace.definition]
                path = os.path.join(self.run_dir, "traces", defn.op_type, f"{defn.name}.jsonl")
                append_jsonl_file([trace], path)
                eval_id = trace.solution + "_" + trace.workload.uuid
                traces.add_evaluation(FlashInferBenchEvaluationResult(evaluation_id=eval_id, task_id=def_name, solution_id=trace.solution, evaluation=trace))
        
        return traces

    def _get_code_string(self, solution: FIBSolution):
        parts = []
        for src in solution.sources:
            header = f"\n/* ===== {src.path} ===== */\n" if self.language.lower() == "cuda" else f"\n# ===== {src.path} =====\n"
            parts.append(header + src.content)
        return "\n".join(parts).strip()
    
    def _get_evaluation_string(self, evaluation):
        if evaluation.evaluation.status == EvaluationStatus.PASSED:
            return f"Passed with {evaluation.evaluation.performance.speedup_factor:.2f}x speedup"
        else:
            return f"Failed with status {evaluation.evaluation.status.value}"

    def format_solution(self, solution: Solution, evaluation: EvaluationResult) -> str:
        """
        Get a string representation of the solution (solution, evaluation result)
        This can be used to extract memory or rules
        """
        return f"Solution: \n{self._get_code_string(solution.solution)} \n\nEvaluation: {self._get_evaluation_string(evaluation.evaluation)}"

    def format_trajectory(self, task: Task, solution: Solution, evaluation: EvaluationResult) -> str:
        """
        Get a string representation of the trajectory (task, solution, evaluation result)
        This can be used to extract memory or rules
        """
        return f"Task: \n{task.task_description}\n\n{self.format_solution(solution, evaluation)}"

    def analyze(self, trace: Traces) -> dict:
        """
        Analyze evaluation results and return a dictionary of metrics
        """
        runs = []
        for evaluation in trace.evaluations:
            definition = evaluation.task_id
            workload_uuid = evaluation.evaluation.workload.uuid
            solution = evaluation.solution_id
            author = self.config.model_name
            eval = evaluation.evaluation.evaluation
            latency_ms = eval.performance.latency_ms if eval.performance is not None else None
            runs.append(Run(definition=definition, workload_uuid=workload_uuid, solution=solution, author=author, latency_ms=latency_ms))
        return compute_metrics(runs, os.path.join(self.run_dir, "win_at_p.csv"))
