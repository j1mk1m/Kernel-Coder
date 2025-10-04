"""
Generation utils for Test-time Scaling
"""
import os
import traceback
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(REPO_ROOT, "external")
sys.path.append(REPO_ROOT)
sys.path.append(EXTERNAL)

# KernelBench imports
from KerenlBench.src.utils import maybe_multithread, extract_last_code, WorkArgs
from KerenlBench.src.prompt_constructor import generate_prompt
from KerenlBench.src.dataset import fetch_ref_arch_from_level_problem_id
from KerenlBench.src.run_utils import check_if_response_exists

# Local imports


def generate_sample_single(work: WorkArgs, config, llm_client, run_dir: str, rule_path=None) -> bool:
    ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(work.level, work.problem_id, config.dataset_src)

    # Construct Prompt   
    custom_cuda_prompt = generate_prompt(work, config, ref_arch_src, llm_client, run_dir, rule_path)
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

def generate_sample_launcher(work: WorkArgs, config, llm_client, run_dir: str, rule_path=None):
    try:
        return generate_sample_single(work, config, llm_client, run_dir, rule_path)
    except Exception as e:
        print(f"Error generating problem {work.problem_id} sample {work.sample_id}: {e}")
        print(traceback.format_exc()) 
        return None


def batch_generate(
    total_work: list[WorkArgs],
    config,
    llm_client,
    run_dir: str,
    rule_path=None
):
    total_work = [work for work in total_work if not check_if_response_exists(run_dir, work.level, work.problem_id, work.sample_id)]
    return maybe_multithread(generate_sample_launcher, 
                      total_work, 
                      config.num_workers, 
                      pbar_name=f"Generation {config.method} Progress",
                      time_interval=config.api_query_interval, 
                      # extra args
                      config=config, 
                      llm_client=llm_client,
                      run_dir=run_dir,
                      rule_path=rule_path
                      )

