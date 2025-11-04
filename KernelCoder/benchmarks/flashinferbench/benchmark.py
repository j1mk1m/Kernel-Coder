from dataclasses import dataclass
from typing import List

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Flashinfer Bench dependencies
from flashinfer_bench import (
    Benchmark,
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
from flashinfer_bench.data import save_json_file, save_jsonl_file 
from flashinfer_bench.logging import configure_logging

from KernelCoder.benchmarks.flashinferbench.prompts import get_prompt, get_optimization_prompt
from KernelCoder.benchmarks.benchmark import Task, Solution, EvaluationResult, Traces


@dataclass(frozen=True)
class FlashInferBenchTask(Task):
    task_id: str
    definition: Definition


@dataclass
class FlashInferBenchSolution(Solution):
    solution_id: str
    task_id: str
    solution: FIBSolution


@dataclass
class FlashInferBenchEvaluationResult(EvaluationResult):
    evaluation_id: str
    task_id: str
    solution_id: str
    evaluation: Trace


class FlashInferBenchTraces(Traces):
    def load(self, path):
        pass
    

class FlashInferBenchBenchmark(Benchmark):
    def __init__(self, name: str, run_dir, llm_client, config):
        """
        Benchmark setup
        """
        super().__init__(name, run_dir, llm_client, config)
        self.language = config.language
        self.target_gpu = config.target_gpu
        self.base_traceset = TraceSet.from_path(config.base_traceset_path)
    
    def get_prompt(self, task: Task, context:str=None) -> str:
        return get_prompt(self.language, task.definition, self.target_gpu, context)

    def get_refinement_prompt(self, task: Task, trace: Traces, context:str=None) -> str:
        use_opt = False
        current_code_str = None
        trace_for_opt = None

        best_trace = trace.get_best_trace(task.definition.def_name)
        if best_trace is not None:
            sol = trace.get_solution(best_trace.solution)
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
            author=self.model_name,
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
        solution = self._create_solution_from_code(cleaned_code, task.definition, solution_id)
        return FlashInferBenchSolution(solution_id=solution_id, task_id=task.task_id, solution=solution)

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
