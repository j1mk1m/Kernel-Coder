import torch
import os
import sys
import wandb
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from KernelBench.src.run_utils import find_highest_sample_id, fetch_baseline_results, write_kernel_to_disk, write_eval_result_to_separate_file
from KernelBench.src.utils import extract_last_code
from KernelBench.src.eval import KernelExecResult

from evaluation import send_batch_evaluation_request, EvaluationWorkArgs, serialize_work_args
from rl.reward_hacking import is_generated_kernel_used, torch_function_used
from rl.grpo_utils import log_reward_info, kevin_reward_from_exec_result


# RUNS_DIR = "/data/user_data/gyeongwk/KernelBench/grpo/runs"
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
RUN_NAME = os.environ.get("RUN_NAME", "test_run")
EVAL_SERVER_HOST = os.environ.get("EVAL_SERVER_HOST", "hyperbolic-1")
EVAL_SERVER_PORT = os.environ.get("EVAL_SERVER_PORT", 8083)
NUM_GENERATIONS = os.environ.get("NUM_GENERATIONS", 8)
HARDWARE = os.environ.get("HARDWARE", "H100_hyperbolic")

os.makedirs(os.path.join(RUNS_DIR, RUN_NAME), exist_ok=True)

# Define custom reward functions
def reward_from_exec_result(level, problem, exec_result):
    return kevin_reward_from_exec_result(level, problem, exec_result)


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    run_dir = os.path.join(RUNS_DIR, RUN_NAME)

    work_args_list = []
    job_list = []
    thread_ids = {}
    rewards = {}
    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        level = extra_info['level']
        problem = extra_info['problem']
        key = f"{level}_{problem}"
        if key not in thread_ids:
            thread_ids[key] = find_highest_sample_id(run_dir, level, problem, 0, NUM_GENERATIONS)
        else:
            thread_ids[key] += 1
        sample_id = thread_ids[key]
        work_args_list.append((level, problem, sample_id))

        response_path = os.path.join(run_dir, f"level_{level}_problem_{problem}_sample_{sample_id}_response.txt")
        with open(response_path, "w") as f:
            f.write(solution_str)

        kernel_name = f"level_{level}_problem_{problem}_sample_{sample_id}"
        kernel_src = extract_last_code(solution_str, ["python", "cpp"])

        if kernel_src is not None:
            write_kernel_to_disk(run_dir, level, problem, sample_id, kernel_src) # for debugging

            # Level 1 kernels should not use torch functions
            if level == 1 and torch_function_used(kernel_src):
                rewards[f"{level}_{problem}_{sample_id}"] = (0.0, "Torch function used")
                exec_result = KernelExecResult(correctness=False, compiled=False, metadata={"other_error": "Torch function used"})
                write_eval_result_to_separate_file(level, problem, sample_id, exec_result, run_dir)
                continue

            # CUDA kernels must be used
            if not is_generated_kernel_used(kernel_src):
                rewards[f"{level}_{problem}_{sample_id}"] = (0.0, "Kernel not used")
                exec_result = KernelExecResult(correctness=False, compiled=False, metadata={"other_error": "Kernel not used"})
                write_eval_result_to_separate_file(level, problem, sample_id, exec_result, run_dir)
                continue

            work_args=EvaluationWorkArgs(level=level, problem_id=problem, sample_id=sample_id, device=torch.device("cuda"))
            job_list.append({
                "work_args": serialize_work_args(work_args),
                "run_name": RUN_NAME,
                "kernel_src": kernel_src,
                "kernel_name": kernel_name
            })
        else:
            rewards[f"{level}_{problem}_{sample_id}"] = (0.0, "No kernel")
    
    results = send_batch_evaluation_request(EVAL_SERVER_HOST, EVAL_SERVER_PORT, job_list)

    for result, job in zip(results, job_list):
        level = job['work_args']["level"]
        problem = job['work_args']["problem_id"]
        sample_id = job['work_args']["sample_id"]
        write_eval_result_to_separate_file(level, problem, sample_id, result, run_dir)
        reward = reward_from_exec_result(level, problem, result)
        rewards[f"{level}_{problem}_{sample_id}"] = (reward, "Kernel evaluated")
    
    log_reward_info(rewards, len(rewards) != 64)
    
    # Turn rewards into list in the same order as input
    rewards_list = [rewards[f"{level}_{problem}_{sample_id}"][0] for level, problem, sample_id in work_args_list]
    return rewards_list

