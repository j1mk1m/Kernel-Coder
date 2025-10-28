"""
Generation utils for Test-time Scaling
"""
import os
import traceback
import sys
from llm_utils import create_llm_client

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(REPO_ROOT, "external")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
sys.path.append(REPO_ROOT)
sys.path.append(EXTERNAL)

# KernelBench imports
from KernelBench.src.utils import maybe_multithread, extract_last_code, WorkArgs
from KernelBench.src.dataset import fetch_ref_arch_from_level_problem_id
from KernelBench.src.run_utils import check_if_response_exists, check_if_kernel_exists

# Local imports
from prompt import generate_prompt


def generate_sample_single(work: WorkArgs, config, llm_client, run_dir: str, **kwargs) -> bool:
    ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(work.level, work.problem_id, config.dataset_src)

    # Construct Prompt   
    custom_cuda_prompt = generate_prompt(work, config, ref_arch_src, llm_client, run_dir, **kwargs)
    if config.log_prompt:
        prompt_path = os.path.join(run_dir, f"level_{work.level}_problem_{work.problem_id}_sample_{work.sample_id}_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    custom_cuda = llm_client.text_completion(custom_cuda_prompt)["choices"][0]["text"]
    if config.log_response:
        response_path = os.path.join(run_dir, f"level_{work.level}_problem_{work.problem_id}_sample_{work.sample_id}_response.txt")
        with open(response_path, "w") as f:
            f.write(custom_cuda)
    custom_cuda = extract_last_code(custom_cuda, ["python", "cpp"])

    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"

    if config.verbose:
        print(f"Generated sample {work.sample_id} for problem {work.problem_id}")

    # Store to local file
    kernel_path = os.path.join(run_dir, f"level_{work.level}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py")
    with open(kernel_path, "w") as f:
        f.write(custom_cuda)
    
    return True

def generate_sample_launcher(work: WorkArgs, config, llm_client, run_dir: str, **kwargs):
    try:
        return generate_sample_single(work, config, llm_client, run_dir, **kwargs)
    except Exception as e:
        print(f"Error generating problem {work.problem_id} sample {work.sample_id}: {e}")
        return None


def batch_generate(
    total_work: list[WorkArgs],
    config,
    llm_client,
    run_dir: str,
    **kwargs
):
    total_work = [work for work in total_work if not check_if_kernel_exists(run_dir, work.level, work.problem_id, work.sample_id)]
    return maybe_multithread(generate_sample_launcher, 
                      total_work, 
                      config.num_workers, 
                      pbar_name=f"Generation {config.method} Progress",
                      time_interval=config.api_query_interval, 
                      # extra args
                      config=config, 
                      llm_client=llm_client,
                      run_dir=run_dir,
                      **kwargs
                      )


if __name__ == "__main__":
    config = parse_generation_args()

    # 1. Set up
    # Set up dataset
    curr_level_dataset = construct_kernelbench_dataset(config.level)

    default_base_api = f"http://{config.vllm_host}:{config.vllm_port}/v1" if config.server_type == "vllm" else None
    llm_client = create_llm_client(os.path.join(run_dir, "llm_usage.json"),
                                   default_model=config.model_name,
                                   default_api_base=default_base_api,
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)

    # set up run directory
    run_dir = os.path.join(RUNS_DIR, config.run_name)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)
 
    total_work = [WorkArgs(level=config.level, problem_id=problem_id, sample_id=sid) for problem_id in range(1, 101) for sid in range(1)] # TODO: change accordingly

    batch_generate(total_work, config, llm_client, run_dir)


