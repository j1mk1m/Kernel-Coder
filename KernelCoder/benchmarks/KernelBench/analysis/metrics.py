import yaml
import argparse
import sys
from argparse import ArgumentParser
import os
import json
import re
from typing import Dict, Tuple, List
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
EXTERNAL = os.path.join(REPO_ROOT, "external")
sys.path.append(REPO_ROOT)
sys.path.append(EXTERNAL)

from external.KernelBench.src.dataset import construct_kernelbench_dataset
from external.KernelBench.src.score import *
from KernelCoder.benchmarks.KernelBench.classes import KernelBenchTraces, KernelBenchEvaluationResult


"""
Given a run directory, compute correctness and efficiency metrics and save to metrics.json.
Format depends on the test-time scaling method used.
"""


BASELINES = ["torch"] # , "torch_compile_inductor_default", "torch_compile_inductor_reduce-overhead", "torch_compile_inductor_max-autotune", "torch_compile_inductor_max-autotune-no-cudagraphs"]


def compute_correctness_metrics(eval_results, subset=None):
    """
    Expects eval_results to be dict of problem_id -> exec_result
    """
    total = 0
    compiled = 0
    correct = 0
    runtime_error = 0
    output_mismatch = 0
    output_shape_mismatch = 0

    for k, res in eval_results.items():
        if subset is not None and int(k) not in subset:
            continue
        total += 1
        if "compiled" in res and res["compiled"]:
            compiled += 1
        if "correctness" in res and res["correctness"]:
            correct += 1
        if "metadata" in res and "runtime_error" in res["metadata"]:
            runtime_error += 1
        if "metadata" in res and "correctness_issue" in res["metadata"]:
            if res["metadata"]["correctness_issue"] == "Output mismatch":
                output_mismatch += 1
            elif "Output shape mismatch" in res["metadata"]["correctness_issue"]:
                output_shape_mismatch += 1

    return {
        "total": total,
        "compiled": compiled,
        "correct": correct,
        "runtime_error": runtime_error,
        "output_mismatch": output_mismatch,
        "output_shape_mismatch": output_shape_mismatch
    }


def compute_efficiency_metrics(eval_results, baseline_results, subset=None):
    """
    Expects eval_results and baseline_results to be dict (problem_id -> exec_result)
    """
    # Filter out problems where baseline_results is empty
    eval = []
    baseline = []
    for k, v in baseline_results.items():
        if v is None: 
            # print(f"Skipping {k} in baseline_results")
            continue
        problem_number = k.split("_")[0]
        if subset is not None and int(problem_number) not in subset:
            continue
        if problem_number not in eval_results: 
            # print(f"Problem {problem_number} not in eval_results")
            continue
        eval.append(eval_results[problem_number])
        baseline.append(v)

    is_correct = np.array([entry["correctness"] for entry in eval])
    baseline_speed = np.array([entry["mean"] for entry in baseline])
    actual_speed = np.array([entry["runtime"] for entry in eval])
    n = len(is_correct)

    assert len(baseline_speed) == n, "Baseline speedup values do not match the number of eval results"
    assert len(actual_speed) == n, "Actual speedup values do not match the number of eval results"

    # Calculate the metrics
    gmsr_correct = geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n)
    gmsr_correct_and_faster = geometric_mean_speed_ratio_correct_and_faster_only(is_correct, baseline_speed, actual_speed, n)

    # list of speedup thresholds p
    p_values = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    results = {p: fastp(is_correct, baseline_speed, actual_speed, n, p) for p in p_values}

    return {
        "mean_speedup_correct": gmsr_correct, # geometric mean of speedup for correct samples
        "mean_speedup_correct_and_faster": gmsr_correct_and_faster, # geometric mean of speedup for correct and faster samples
        "fast_p_results": results
    }

def compute_efficiency_metrics_all_baselines(config, hardware: str, eval_results: dict, subset=None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> exec_result
    """
    results = {}
    for baseline in BASELINES:
        try:
            baseline_file_path = os.path.join(EXTERNAL, "KernelBench", "results", "timing", hardware, f'baseline_time_{baseline}.json')
            assert os.path.exists(baseline_file_path), f"Baseline file does not exist at {baseline_file_path}"

            with open(baseline_file_path, 'r') as f:
                baseline_results = json.load(f)

            baseline_results = baseline_results[f'level{config["level"]}']

            comp_metrics = compute_efficiency_metrics(eval_results, baseline_results, subset)
            results[baseline] = comp_metrics
        except Exception as e:
            print(f"Error computing efficiency metrics for {baseline}: {e}")
            continue

    return results


def hardware_check(eval_results: dict, hardware_ref: str):
    print("Checking that results are on the same hardware")
    hardware = list(list(eval_results.values())[0].values())[0]["metadata"]["hardware"]
    for _, prob_res in eval_results.items():
        for _, sample_res in prob_res.items():
            if "metadata" not in sample_res or "hardware" not in sample_res["metadata"]:
                continue
            assert sample_res["metadata"]["hardware"] == hardware, f"Hardware mismatch: {sample_res['metadata']['hardware']} != {hardware}"
    print(f"Computing metrics for {hardware} with baseline {hardware_ref} (Should match)")


dummy_result = {
    "sample_id": 0, 
    "compiled": False, 
    "correctness": False, 
    "metadata": {},
    "runtime": -1.0, 
    "runtime_stats": {}
}

def patch(eval_results, dataset):
    """
    Patch the eval results with the dataset
    """
    for pid in range(1, len(dataset) + 1):
        if str(pid) not in eval_results:
            eval_results[str(pid)] = {
                "sample_id": 0, 
                "compiled": False, 
                "correctness": False, 
                "metadata": {},
                "runtime": -1.0, 
                "runtime_stats": {}
            }
    return eval_results


def compute_all_metrics(config, hardware, eval_results, subset=None):
    """
    Computes correctness and efficiency metrics for evaluation results of form dict of problem_id -> exec_result
    """
    correctness_metrics = compute_correctness_metrics(eval_results, subset)
    dataset = construct_kernelbench_dataset(config["level"])
    eval_results = patch(eval_results, dataset)
    efficiency_metrics = compute_efficiency_metrics_all_baselines(config, hardware, eval_results, subset)
    return {"correctness": correctness_metrics, "speedups": efficiency_metrics}


def compute_metrics_base(config, hardware: str, eval_results: dict, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    eval_results = {k: v["0"] for k, v in eval_results.items()}
    return compute_all_metrics(config, hardware, eval_results, subset)


def increasing_best_solution_metrics(config, hardware: str, eval_results: dict, num_steps, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    best_by_step = {}
    best_by_step[0] = {k: v["0"] if "0" in v else dummy_result for k, v in eval_results.items()}

    for step in range(1, num_steps):
        best_by_step[step] = {}
        for pid, prob_res in eval_results.items():
            prev_best = best_by_step[step - 1][pid]
            if str(step) not in prob_res:
                best_by_step[step][pid] = prev_best
                continue
            res = prob_res[str(step)]
            if not prev_best["correctness"] and res["correctness"]:
                best_by_step[step][pid] = res
            elif not prev_best["compiled"] and res["compiled"]:
                best_by_step[step][pid] = res
            elif prev_best["correctness"] and res["correctness"] and "runtime" in res and res["runtime"] < prev_best["runtime"]:
                best_by_step[step][pid] = res
            else:
                best_by_step[step][pid] = prev_best
            
    metrics = {}
    for step, step_results in best_by_step.items():
        metrics[step] = compute_all_metrics(config, hardware, step_results, subset)
    return metrics


def compute_metrics_best_of_n(config, hardware: str, eval_results: dict, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    by_sample_results = {}
    for pid, prob_res in eval_results.items():
        for sid, sample_res in prob_res.items():
            if sid not in by_sample_results:
                by_sample_results[sid] = {}
            
            by_sample_results[sid][pid] = sample_res
    
    metrics = {"by_sample": {}}
    for sid, sample_res in by_sample_results.items():
        metrics["by_sample"][sid] = compute_all_metrics(config, hardware, by_sample_results[sid], subset)
    metrics["best_by_sample"] = increasing_best_solution_metrics(config, hardware, eval_results, config["num_parallel"])
    return metrics


def compute_metrics_iterative_refinement(config, hardware: str, eval_results: dict, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    assert config["num_parallel"] == 1, "Iterative refinement is only supported for 1 parallel run"
    return increasing_best_solution_metrics(config, hardware, eval_results, config["num_iterations"], subset)


def compute_metrics_metr(config, hardware: str, eval_results: dict, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    for i in range(config["num_samples"]-1):
        for pid, prob_res in eval_results.items():
            if str(i+1) in prob_res:
                eval_results[pid][str(i)] = eval_results[pid][str(i+1)]
            else:
                eval_results[pid][str(i)] = dummy_result
    return increasing_best_solution_metrics(config, hardware, eval_results, config["num_samples"], subset)


def compute_metrics_stanford(config, hardware: str, eval_results: dict, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    pass


def compute_metrics_test_time_scaling(config, hardware: str, eval_results: dict, run_dir: str, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    # hardware_check(eval_results, hardware)

    match config["method"]:
        case "base":
            metrics = compute_metrics_base(config, hardware, eval_results, subset)
        case "best-of-N":
            metrics = compute_metrics_best_of_n(config, hardware, eval_results, subset)
        case "iterative refinement":
            metrics = compute_metrics_iterative_refinement(config, hardware, eval_results, subset)
        case "METR":
            metrics = compute_metrics_metr(config, hardware, eval_results, subset)
        case "Stanford":
            metrics = compute_metrics_stanford(config, hardware, eval_results, subset)
        case _:
            raise ValueError(f"Invalid method: {config['method']}")

    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_file}")
    
    return metrics


# ==== Unified multi-level API using KernelBenchTraces ====

def parse_evaluation_id(evaluation_id: str) -> Tuple[int, int, int]:
    """
    Parse evaluation_id like "level_{level}_problem_{problem}_solution_{solution}" and return (level, problem, solution)
    """
    m = re.match(r"level_(\d+)_problem_(\d+)_solution_(\d+)", evaluation_id)
    if not m:
        raise ValueError(f"Invalid evaluation_id format: {evaluation_id}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def build_unified_eval_results_from_traces(traces: KernelBenchTraces) -> Dict[str, Dict[str, dict]]:
    """
    Build a unified mapping: problem_key -> sample_id(str) -> exec_result(dict)
    problem_key = f"level_{level}_problem_{problem}"
    """
    unified: Dict[str, Dict[str, dict]] = {}
    for ev in traces.evaluations:
        if isinstance(ev, KernelBenchEvaluationResult):
            evaluation_id = ev.evaluation_id
            compiled = ev.compiled
            correctness = ev.correctness
            metadata = ev.metadata
            runtime = ev.runtime
            runtime_stats = ev.runtime_stats
        else:
            evaluation_id = ev["evaluation_id"]
            compiled = ev.get("compiled", False)
            correctness = ev.get("correctness", False)
            metadata = ev.get("metadata", {})
            runtime = ev.get("runtime", -1.0)
            runtime_stats = ev.get("runtime_stats", {})

        level, problem, solution = parse_evaluation_id(evaluation_id)
        problem_key = f"level_{level}_problem_{problem}"
        sample_id = str(solution)

        if problem_key not in unified:
            unified[problem_key] = {}
        unified[problem_key][sample_id] = {
            "compiled": compiled,
            "correctness": correctness,
            "metadata": metadata if metadata is not None else {},
            "runtime": runtime if runtime is not None else -1.0,
            "runtime_stats": runtime_stats if runtime_stats is not None else {}
        }
    return unified


def load_and_merge_baselines_unified(hardware: str, baselines: List[str], eval_problem_keys: List[str]) -> Dict[str, Dict[str, dict]]:
    """
    Returns mapping baseline_name -> { problem_key -> baseline_entry }
    Only includes problems present in eval_problem_keys.
    """
    results: Dict[str, Dict[str, dict]] = {}
    for baseline in baselines:
        try:
            baseline_file_path = os.path.join(EXTERNAL, "KernelBench", "results", "timing", hardware, f"baseline_time_{baseline}.json")
            assert os.path.exists(baseline_file_path), f"Baseline file does not exist at {baseline_file_path}"
            with open(baseline_file_path, "r") as f:
                baseline_json = json.load(f)

            mapping: Dict[str, dict] = {}
            per_level_cache: Dict[int, Dict[int, dict]] = {}
            for level_key, problems in baseline_json.items():
                m = re.match(r"level(\d+)", level_key)
                if not m:
                    continue
                lvl = int(m.group(1))
                per_level_cache[lvl] = {}
                for prob_key, entry in problems.items():
                    prob_num = int(str(prob_key).split("_")[0])
                    per_level_cache[lvl][prob_num] = entry

            for problem_key in eval_problem_keys:
                m2 = re.match(r"level_(\d+)_problem_(\d+)", problem_key)
                if not m2:
                    continue
                lvl = int(m2.group(1))
                prob = int(m2.group(2))
                if lvl in per_level_cache and prob in per_level_cache[lvl]:
                    mapping[problem_key] = per_level_cache[lvl][prob]

            results[baseline] = mapping
        except Exception as e:
            print(f"Error computing efficiency metrics for {baseline}: {e}")
            results[baseline] = {}
            continue
    return results


def compute_correctness_metrics_unified(eval_results_by_problem: Dict[str, dict]) -> dict:
    total = 0
    compiled = 0
    correct = 0
    runtime_error = 0
    output_mismatch = 0
    output_shape_mismatch = 0

    for _, res in eval_results_by_problem.items():
        total += 1
        if "compiled" in res and res["compiled"]:
            compiled += 1
        if "correctness" in res and res["correctness"]:
            correct += 1
        if "metadata" in res and "runtime_error" in res["metadata"]:
            runtime_error += 1
        if "metadata" in res and "correctness_issue" in res["metadata"]:
            if res["metadata"]["correctness_issue"] == "Output mismatch":
                output_mismatch += 1
            elif "Output shape mismatch" in res["metadata"]["correctness_issue"]:
                output_shape_mismatch += 1

    return {
        "total": total,
        "compiled": compiled,
        "correct": correct,
        "runtime_error": runtime_error,
        "output_mismatch": output_mismatch,
        "output_shape_mismatch": output_shape_mismatch
    }


def compute_efficiency_metrics_unified(eval_results_by_problem: Dict[str, dict], baseline_results_by_problem: Dict[str, dict]) -> dict:
    eval_entries = []
    baseline_entries = []
    for problem_key, baseline_entry in baseline_results_by_problem.items():
        if problem_key not in eval_results_by_problem:
            continue
        eval_entries.append(eval_results_by_problem[problem_key])
        baseline_entries.append(baseline_entry)

    if len(eval_entries) == 0:
        return {
            "mean_speedup_correct": 0.0,
            "mean_speedup_correct_and_faster": 0.0,
            "fast_p_results": {p: 0.0 for p in [0.0, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]}
        }

    is_correct = np.array([entry.get("correctness", False) for entry in eval_entries])
    baseline_speed = np.array([entry.get("mean", 0.0) for entry in baseline_entries])
    actual_speed = np.array([entry.get("runtime", 0.0) for entry in eval_entries])
    n = len(is_correct)

    gmsr_correct = geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n)
    gmsr_correct_and_faster = geometric_mean_speed_ratio_correct_and_faster_only(is_correct, baseline_speed, actual_speed, n)

    p_values = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    results = {p: fastp(is_correct, baseline_speed, actual_speed, n, p) for p in p_values}

    return {
        "mean_speedup_correct": gmsr_correct,
        "mean_speedup_correct_and_faster": gmsr_correct_and_faster,
        "fast_p_results": results
    }


def compute_all_metrics_unified(eval_results_by_problem: Dict[str, dict], hardware: str) -> dict:
    correctness_metrics = compute_correctness_metrics_unified(eval_results_by_problem)
    merged_baselines = load_and_merge_baselines_unified(hardware, BASELINES, list(eval_results_by_problem.keys()))
    speedups = {}
    for baseline_name, baseline_problem_map in merged_baselines.items():
        speedups[baseline_name] = compute_efficiency_metrics_unified(eval_results_by_problem, baseline_problem_map)
    return {"correctness": correctness_metrics, "speedups": speedups}


def increasing_best_solution_metrics_unified(eval_results: Dict[str, Dict[str, dict]], num_steps: int, hardware: str) -> dict:
    best_by_step: Dict[int, Dict[str, dict]] = {}
    best_by_step[0] = {}
    for pid, prob_res in eval_results.items():
        res0 = prob_res["0"] if "0" in prob_res else dummy_result
        best_by_step[0][pid] = res0

    def better(prev: dict, cur: dict) -> dict:
        if (not prev.get("correctness", False)) and cur.get("correctness", False):
            return cur
        if (not prev.get("compiled", False)) and cur.get("compiled", False):
            return cur
        if prev.get("correctness", False) and cur.get("correctness", False) and ("runtime" in cur) and (cur["runtime"] < prev.get("runtime", float("inf"))):
            return cur
        return prev

    for step in range(1, num_steps):
        best_by_step[step] = {}
        for pid, prob_res in eval_results.items():
            prev_best = best_by_step[step - 1][pid]
            if str(step) not in prob_res:
                best_by_step[step][pid] = prev_best
                continue
            res = prob_res[str(step)]
            best_by_step[step][pid] = better(prev_best, res)

    metrics = {}
    for step, step_results in best_by_step.items():
        metrics[step] = compute_all_metrics_unified(step_results, hardware)
    return metrics


def compute_metrics_best_of_n_unified(hardware: str, eval_results: Dict[str, Dict[str, dict]], num_parallel: int) -> dict:
    by_sample_results: Dict[str, Dict[str, dict]] = {}
    for pid, prob_res in eval_results.items():
        for sid, sample_res in prob_res.items():
            if sid not in by_sample_results:
                by_sample_results[sid] = {}
            by_sample_results[sid][pid] = sample_res

    metrics = {"by_sample": {}}
    for sid, sample_res in by_sample_results.items():
        metrics["by_sample"][sid] = compute_all_metrics_unified(sample_res, hardware)
    metrics["best_by_sample"] = increasing_best_solution_metrics_unified(eval_results, num_parallel, hardware)
    return metrics


def compute_metrics_iterative_refinement_unified(hardware: str, eval_results: Dict[str, Dict[str, dict]], num_iterations: int) -> dict:
    return increasing_best_solution_metrics_unified(eval_results, num_iterations, hardware)


def compute_metrics_metr_unified(hardware: str, eval_results: Dict[str, Dict[str, dict]], num_samples: int) -> dict:
    shifted: Dict[str, Dict[str, dict]] = {}
    for pid, prob_res in eval_results.items():
        shifted[pid] = {}
        for i in range(num_samples - 1):
            src_key = str(i + 1)
            shifted[pid][str(i)] = prob_res[src_key] if src_key in prob_res else dummy_result
    return increasing_best_solution_metrics_unified(shifted, num_samples, hardware)


def compute_metrics_from_traces(traces: KernelBenchTraces, hardware: str, config, run_dir: str) -> dict:
    unified = build_unified_eval_results_from_traces(traces)


    match config.method:
        case "base":
            base_results: Dict[str, dict] = {}
            for pid, prob_res in unified.items():
                base_results[pid] = prob_res["0"] if "0" in prob_res else dummy_result
            metrics = compute_all_metrics_unified(base_results, hardware)
        case "best-of-N":
            metrics = compute_metrics_best_of_n_unified(hardware, unified, config.num_parallel)
        case "iterative refinement":
            assert config.num_parallel == 1, "Iterative refinement is only supported for 1 parallel run"
            metrics = compute_metrics_iterative_refinement_unified(hardware, unified, config.num_iterations)
        case "METR":
            metrics = compute_metrics_metr_unified(hardware, unified, config.num_samples)
        case _:
            raise ValueError(f"Invalid method: {config.method}")

    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_file}")
    return metrics


def compute_metrics_grpo(config, hardware: str, eval_results: dict, run_dir: str) -> dict:
    # Eval subsets for Level 1 and 2
    eval_results_level_1 = eval_results["1"]
    eval_results_level_2 = eval_results["2"]

    config["num_parallel"] = 64 # TODO: make this match number of kernels generated in training
    config["level"] = 1
    metrics_train_level_1 = compute_metrics_best_of_n(config, hardware, eval_results_level_1, subset=TRAIN_PROBLEM_IDS_LEVEL_1)
    metrics_eval_level_1 = compute_all_metrics(config, hardware, eval_results_level_1, subset=TEST_PROBLEM_IDS_LEVEL_1)
    config["level"] = 2
    metrics_train_level_2 = compute_metrics_best_of_n(config, hardware, eval_results_level_2, subset=TRAIN_PROBLEM_IDS_LEVEL_2)
    metrics_eval_level_2 = compute_all_metrics(config, hardware, eval_results_level_2, subset=TEST_PROBLEM_IDS_LEVEL_2)

    metrics = {
        "1": {
            "train": metrics_train_level_1,
            "eval": metrics_eval_level_1
        },
        "2": {
            "train": metrics_train_level_2,
            "eval": metrics_eval_level_2
        }
    }

    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_file}")
    
    return metrics


def collate_eval_results(run_dir):
    """
    Go through every file in directory and find those like "level_1_problem_1_sample_0_eval_results.json".
    This file will contain a json object like {"1": {"1": {"0": eval_result}}}.
    Combine all of these into one json and save it to file.
    """
    import glob
    import re

    # Pattern to match eval result files
    pattern = os.path.join(run_dir, "level_*_problem_*_sample_*_eval_result.json")
    eval_files = glob.glob(pattern)
    
    # Combined results structure: {level: {problem: {sample: eval_result}}}
    combined_results = {}
    
    for file_path in eval_files:
        # Extract level, problem, and sample from filename
        filename = os.path.basename(file_path)
        match = re.match(r"level_(\d+)_problem_(\d+)_sample_(\d+)_eval_result\.json", filename)
        
        if match:
            level = match.group(1)
            problem = match.group(2)
            sample = match.group(3)
            
            # Read the eval result file
            with open(file_path, 'r') as f:
                eval_result = json.load(f)
            
            # Initialize nested structure if needed
            if level not in combined_results:
                combined_results[level] = {}
            if problem not in combined_results[level]:
                combined_results[level][problem] = {}
            
            # Add the eval result
            combined_results[level][problem][sample] = eval_result

    # Sort the combined results by level, problem_id, and sample_id
    sorted_results = {}
    for level in sorted(combined_results.keys(), key=int):
        sorted_results[level] = {}
        for problem in sorted(combined_results[level].keys(), key=int):
            sorted_results[level][problem] = {}
            for sample in sorted(combined_results[level][problem].keys(), key=int):
                sorted_results[level][problem][sample] = combined_results[level][problem][sample]
    
    combined_results = sorted_results
    
    # Save combined results to file
    output_path = os.path.join(run_dir, "eval_results.json")
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    print(f"Collated {len(eval_files)} eval result files into {output_path}")
    return combined_results


def legacy_main(run_dir, hardware, grpo): 
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        print("No config file found. Using empty dict")
        config  = {
            "level": 3,
            "method": "best-of-N",
            "num_parallel": 8,
            "num_iterations": 1,
            "num_samples": 1
        }
    else:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    eval_file_path = os.path.join(run_dir, "eval_results.json")
    if not os.path.exists(eval_file_path):
        print("Collating eval results")
        collate_eval_results(run_dir)

    with open(eval_file_path, 'r') as f:
        eval_results = json.load(f)

    if grpo:
        compute_metrics_grpo(config, hardware, eval_results, run_dir)
    else:
        compute_metrics_test_time_scaling(config, hardware, eval_results[f'{config["level"]}'], run_dir, subset=None)


if __name__ == "__main__":
    # Prefer unified traces-based API when invoked via CLI
    argparser = ArgumentParser()
    argparser.add_argument("--run_dir", type=str, required=True)
    argparser.add_argument("--hardware", type=str, required=True)
    argparser.add_argument("--method", type=str, default="best-of-N")
    argparser.add_argument("--num_parallel", type=int, default=8)
    argparser.add_argument("--num_iterations", type=int, default=1)
    argparser.add_argument("--num_samples", type=int, default=1)
    args = argparser.parse_args()

    traces = KernelBenchTraces(tasks=[], path=args.run_dir)
    compute_metrics_from_traces(traces, args.hardware, method=args.method, num_parallel=args.num_parallel, num_iterations=args.num_iterations, num_samples=args.num_samples)
