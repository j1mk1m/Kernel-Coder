"""
Evaluation utilities code
- Evaluate single sample
- Evaluate batch of samples
- Send evaluation requests to server (see run_evaluation_server.py for server code)
"""

import os
import shutil
import socket
import pickle
import torch
import json
import time
import sys
from tqdm import tqdm
import multiprocessing as mp
import yaml
from dataclasses import dataclass
import ast

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXTERNAL = os.path.join(REPO_ROOT, "external")
KERNEL_EVAL_BUILD_DIR = "/data/user_data/gyeongwk/Kernel-Coder/cache"
sys.path.append(REPO_ROOT)
sys.path.append(EXTERNAL)

# KernelBench imports
from external.KernelBench.src.compile import batch_compile
from external.KernelBench.src.eval import eval_kernel_against_ref, eval_reference_kernel, KernelExecResult, check_metadata_serializable_all_types
from external.KernelBench.src.utils import set_gpu_arch
from external.KernelBench.src.run_utils import check_if_eval_exists_local

from KernelCoder.benchmarks.KernelBench.classes import KernelBenchSolution, KernelBenchEvaluationResult, KernelBenchTask


@dataclass
class EvaluationWorkArgs:
    evaluation_id: str
    device: torch.device


def evaluate_single_sample_worker(work_args: EvaluationWorkArgs, configs, run_dir: str, task: KernelBenchTask, solution: KernelBenchSolution) -> KernelExecResult | None:
    """
    Evaluate a single sample on a single GPU
    """
    evaluation_id, device = (
        work_args.evaluation_id,
        work_args.device,
    )

    ref_arch_src = task.task_description
    kernel_src = solution.solution_code
    ref_arch_name = task.task_id
    kernel_name = solution.solution_id

    build_dir = os.path.join(KERNEL_EVAL_BUILD_DIR, configs.run_name, evaluation_id)

    try:  
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            original_model_name=ref_arch_name,
            custom_model_src=kernel_src,
            custom_model_name=kernel_name,
            measure_performance=configs.measure_performance,
            verbose=configs.verbose,    
            num_correct_trials=configs.num_correct_trials,
            num_perf_trials=configs.num_perf_trials,
            build_dir=build_dir,
            device=device,
        )
        return eval_result
    except Exception as e:
        print(
            f"[WARNING] Last level catch on {evaluation_id}: Some issue evaluating for kernel: {e} "
        )
        if "CUDA error" in str(e):
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {
                "cuda_error": f"CUDA Error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }  # log this for debugging as this usually signifies illegal memory access
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result
        else:
            metadata = {"other_error": f"error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        } # for debugging
            eval_result = KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result




def batch_eval(
    total_work: list[tuple],
    config: dict,
    run_dir: str,
    eval_file_path: str,
):
    """
    Batch evaluation across multiple GPUs, do batch_size of work one on each GPU all at once
    We put in time out for each batch, consider trying again with larger time out if it didn't finish building.
    Cache directory is removed if evaluation times out or fails
    NOTE: Only for local evaluation
    """
    # total_work is a list of tuples: (task, solution)
    # task should provide reference code; solution provides kernel code and IDs
    filtered_work = []
    for task, solution in total_work:
        evaluation_id = solution.solution_id
        if not check_if_eval_exists_local(evaluation_id, eval_file_path):
            filtered_work.append((task, solution))
    total_work = filtered_work

    # Build Cache on CPU as that is faster
    if config.build_cache_with_cpu:
        compilation_results = batch_compile([(sol.solution_id) for (_, sol) in total_work], vars(config), run_dir)

    # construct a list of work args
    batch_size = config.num_eval_devices
    all_results = []

    with tqdm(total=len(total_work), desc="Evaluation Progress") as pbar:

        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:]  # pop the first batch_size elements
            print(
                f"[Curr Batch] {len(curr_work_batch)} tasks over {config.num_eval_devices} GPUs; [Total Work left] {len(total_work)}"
            )
            assert len(curr_work_batch) <= batch_size, f"Current batch size {len(curr_work_batch)} is greater than the number of GPUs {batch_size}"

            with mp.Pool(batch_size) as pool:

                work_args = []
                for i, (task, solution) in enumerate(curr_work_batch):
                    work_args.append(
                        (
                            EvaluationWorkArgs(
                                evaluation_id=solution.solution_id,
                                device=torch.device(f"cuda:{i%batch_size}"),
                            ),
                            config,
                            run_dir,
                            task,
                            solution,
                        )
                    )

                start_time = time.time()

                async_results = []
                for work_arg in work_args:
                    async_results.append(
                        pool.apply_async(evaluate_single_sample_worker, work_arg)
                    )
            
                # Collect results with a batch timeout
                results = []
                batch_timeout = config.timeout
                for i, async_result in enumerate(async_results):
                    task, solution = curr_work_batch[i]
                    evaluation_id = solution.solution_id

                    try:
                        elapsed_time = time.time() - start_time
                        remaining_time = max(0, batch_timeout - elapsed_time)
                        result = async_result.get(timeout=remaining_time)
                        results.append((evaluation_id, task.task_id, solution.solution_id, result))
                        
                    except mp.TimeoutError:
                        print(
                            f"[WARNING] Evaluation TIMED OUT for Evaluation ID: {evaluation_id}"
                        )
                        result = KernelExecResult(compiled=False, correctness=False, metadata={"other_error": "timeout"})
                        results.append((evaluation_id, task.task_id, solution.solution_id, result))
                    
                        # Remove cache dir for this evaluation
                        build_dir = os.path.join(KERNEL_EVAL_BUILD_DIR, config.run_name, evaluation_id)
                        if os.path.isdir(build_dir):
                            shutil.rmtree(build_dir, ignore_errors=True)
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for Evaluation ID: {evaluation_id}: {str(e)}"
                        )
                        result = KernelExecResult(compiled=False, correctness=False, metadata={"other_error": str(e)})
                        results.append((evaluation_id, task.task_id, solution.solution_id, result))
                        build_dir = os.path.join(KERNEL_EVAL_BUILD_DIR, config.run_name, evaluation_id)
                        if os.path.isdir(build_dir):
                            shutil.rmtree(build_dir, ignore_errors=True)

                end_time = time.time()
                all_results.extend(results)
                # current batch summary
                for evaluation_id, task_id, solution_id, result in results:
                    print("-" * 128)
                    print(
                        f"[Eval Result] Evaluation ID: {evaluation_id}"
                    )
                    print(result)

                if config.verbose:
                    print("-" * 128)
                    print(
                        f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds"
                )

                pbar.update(len(curr_work_batch))
    return all_results


### Evaluation Server/Client helpers
def serialize_work_args(work_args: EvaluationWorkArgs):
    """Serialize EvaluationWorkArgs for network transmission"""
    return {
        'evaluation_id': work_args.evaluation_id,
        'device': str(work_args.device)  # Convert device to string for serialization
    }


def deserialize_work_args(data: dict) -> EvaluationWorkArgs:
    """Deserialize data back to EvaluationWorkArgs"""
    return EvaluationWorkArgs(
        evaluation_id=data['evaluation_id'],
        device=torch.device("cuda")
    )


def send_evaluation_request(host: str, port: int, work_args: EvaluationWorkArgs, run_name: str, ref_arch_src: str = None, ref_arch_name: str | None = None, kernel_src: str = None, kernel_name: str = None):
    """
    Send an evaluation request to the server and receive the result.
    
    Args:
        host: Server hostname (usually 'localhost')
        port: Server port number
        work_args: EvaluationWorkArgs object
        kernel_src: Optional kernel source code
        kernel_name: Optional kernel name
    
    Returns:
        KernelExecResult object from the server
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the server
        client_socket.connect((host, port))
        print(f"Connected to evaluation server at {host}:{port}")
        
        # Prepare the request
        request = {
            'work_args': serialize_work_args(work_args),
            'run_name': run_name,
            'ref_arch_src': ref_arch_src,
            'ref_arch_name': ref_arch_name,
            'kernel_src': kernel_src,
            'kernel_name': kernel_name
        }
        
        # Send the request
        request_data = pickle.dumps(request)
        client_socket.sendall(request_data)
        client_socket.shutdown(socket.SHUT_WR)
        
        # Receive the response
        response_data = b""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
        
        # Deserialize the response
        result = pickle.loads(response_data)
        print(f"Received result from server: {result}")
        return result
        
    except Exception as e:
        print(f"Error communicating with server: {e}")
        return None
    finally:
        client_socket.close()


def check_server_status(host: str, port: int):
    """
    Check if the server is running and get basic status information.
    
    Args:
        host: Server hostname
        port: Server port number
    
    Returns:
        True if server is reachable, False otherwise
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5)  # 5 second timeout
    
    try:
        client_socket.connect((host, port))
        print(f"Server at {host}:{port} is reachable")
        return True
    except Exception as e:
        print(f"Cannot connect to server at {host}:{port}: {e}")
        return False
    finally:
        client_socket.close()


def check_eval_status(config):
    if config.eval_mode == "local":
        return True
    elif config.eval_mode == "remote":
        return check_server_status(config.eval_server_host, config.eval_server_port)
    else:
        raise ValueError(f"Invalid evaluation method: {config.eval_mode}")


def send_batch_evaluation_request(host: str, port: int, job_list: list):
    """
    Send a batch evaluation request to the server and receive the list of results.
    Each job in job_list should be a dict with keys:
        - 'work_args': serialized EvaluationWorkArgs (dict)
        - 'run_name': str
        - 'kernel_src': str or None
        - 'kernel_name': str or None
    Returns:
        List of KernelExecResult objects from the server (same order as job_list)
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        print(f"Connected to evaluation server at {host}:{port} (batch mode)")
        request = {'batch': job_list}
        request_data = pickle.dumps(request)
        client_socket.sendall(request_data)
        client_socket.shutdown(socket.SHUT_WR)
        response_data = b""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
        results = pickle.loads(response_data)
        return results
    except Exception as e:
        print(f"Error communicating with server (batch): {e}")
        return None
    finally:
        client_socket.close()


def evaluate_single_sample_in_separate_process(work_args: EvaluationWorkArgs, configs, run_dir: str, task: KernelBenchTask, solution: KernelBenchSolution) -> KernelExecResult | None:
    """
    Evaluate a single sample in a separate process
    """
    evaluation_id, device = (
        work_args.evaluation_id,
        work_args.device,
    )
     
    # Create argument tuple for the process
    args_tuple = (work_args, configs, run_dir, task, solution)
    
    # Run evaluation in separate process with timeout
    with mp.Pool(1) as pool:
        try:
            result = pool.apply_async(evaluate_single_sample_worker, args_tuple)
            eval_result = result.get(timeout=300)  # 5 minute timeout
            return eval_result
        except mp.TimeoutError:
            metadata = {
                "other_error": "Evaluation timed out after 5 minutes",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }
            return KernelExecResult(
                compiled=False, correctness=False, 
                metadata=metadata
            )
        except Exception as e:
            metadata = {
                "other_error": f"Pool error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }
            return KernelExecResult(
                compiled=False, correctness=False, 
                metadata=metadata
            )

