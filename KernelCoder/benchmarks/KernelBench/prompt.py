"""
Construct Prompt

Design principles: 
- To evaluate base model performance on KernelBench, we use the simplest prompt possible to guide model output to generated desired output format.
- However, we do not do extensive prompt engineering or few-shot example in the LLM to steer behaviour. 
"""

import os
import random
from external.KernelBench.src.utils import read_file, read_json_file, WorkArgs
from external.KernelBench.src.run_utils import fetch_kernel_from_disk, fetch_eval_results_for_problem, fetch_eval_result_from_disk
from external.KernelBench.src.eval import KernelExecResult

REPO_TOP_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
KB_ROOT = os.path.join(REPO_TOP_PATH, "external", "KernelBench")


############################################
# CUDA Prompt
############################################
PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. \n
"""
PROBLEM_INSTRUCTION_IMPROVE = """
Optimize the architecture named Model with custom CUDA operators! 
Improve upon your previous attempts by debugging any correctness issues or improving the efficiency if the kernel was correct.
Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. \n
"""
PROBLEM_INSTRUCTION_COT = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. 
In the end, make sure the final code block contains code for output architecture ModelNew with cuda code.\n
Let's think step by step.\n
""" 

TRITON_PROBLEM_STATEMENT = """You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
TRITON_PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. \n
"""
TRITON_PROBLEM_INSTRUCTION_IMPROVE = """
Optimize the architecture named Model with custom Triton operators! 
Improve upon your previous attempts by debugging any correctness issues or improving the efficiency if the kernel was correct.
Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. \n
"""
TRITON_PROBLEM_INSTRUCTION_COT = """
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. 
In the end, make sure the final code block contains code for output architecture ModelNew with cuda code.\n
Let's think step by step.\n
""" 


def get_problem_statement(triton=False):
    if triton:
        return TRITON_PROBLEM_STATEMENT
    else:
        return PROBLEM_STATEMENT
    
def get_problem_instruction(triton=False):
    if triton:
        return TRITON_PROBLEM_INSTRUCTION
    else:
        return PROBLEM_INSTRUCTION

def get_problem_instruction_improve(triton=False):
    if triton:
        return TRITON_PROBLEM_INSTRUCTION_IMPROVE
    else:
        return PROBLEM_INSTRUCTION_IMPROVE

def get_problem_instruction_cot(triton=False):
    if triton:
        return TRITON_PROBLEM_INSTRUCTION_COT
    else:
        return PROBLEM_INSTRUCTION_COT


def prompt_bare(ref_arch_src: str, triton=False, context=None) -> str:
    prompt = get_problem_statement(triton)
    if context is not None:
        prompt += context
    prompt += f"""
    You are given the following architecture: \n
    ```
    {ref_arch_src}
    ```
    """
    prompt += get_problem_instruction(triton)
    return prompt


def prompt_with_one_example(
    arc_src: str, example_arch_src: str, example_new_arch_src: str, triton=False, context=None
) -> str:
    prompt = get_problem_statement(triton) 

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
```python
{example_arch_src}
``` \n
        The example new arch with custom CUDA kernels looks like this: 
```python
{example_new_arch_src}
``` \n
        """

    if context is not None:
        prompt += context
    
    prompt += f"""
    You are given the following architecture: \n
```python
{arc_src}
```
    """
    prompt += get_problem_instruction(triton)
    return prompt


def prompt_base(ref_arch_src: str, triton=False, context=None) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        KB_ROOT, f"src/prompts/model_ex_add.py"
    )
    if triton:
        example_new_arch_path = os.path.join(
            KB_ROOT, f"src/prompts/model_new_ex_add_triton.py"
        )
    else:
        example_new_arch_path = os.path.join(
            KB_ROOT, f"src/prompts/model_new_ex_add.py"
        )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_with_one_example(arch, example_arch, example_new_arch, triton, context)


def exec_result_to_exeution_feedback(exec_result: dict) -> str:
    if isinstance(exec_result, KernelExecResult):
        metadata = exec_result.metadata
        correctness = exec_result.correctness
        runtime = exec_result.runtime
    else:
        metadata = exec_result['metadata']
        correctness = exec_result['correctness']
        runtime = exec_result['runtime']

    compilation_error = metadata['compilation_error'] if 'compilation_error' in metadata else None
    runtime_error = metadata['runtime_error'] if 'runtime_error' in metadata else None
    correctness_issue = metadata['correctness_issue'] if 'correctness_issue' in metadata else None
    other_error = metadata['other_error'] if 'other_error' in metadata else None
    correctness_feedback = compilation_error if compilation_error else runtime_error if runtime_error else correctness_issue if correctness_issue else other_error if other_error else "All trials passed" 

    evaluation_feedback = f"""
Here is your Evaluation Result:
```
{correctness_feedback}
```
"""

    if correctness:
        evaluation_feedback += f"""
Your kernel executed successfully and produced the correct output.
Here is your wall clock time: {runtime} milliseconds.

{metadata["profiler_info"]}
"""

    return evaluation_feedback
 

def prompt_refinement_from_last_kernel(ref_arch_src: str, last_kernel_src: str, last_exec_result: KernelExecResult, triton=False, context=None) -> str:
    prompt = prompt_base(ref_arch_src, triton=triton, context=context)
    execution_feedback = exec_result_to_exeution_feedback(last_exec_result)

    prompt += f"""Your latest generated kernel:
```
{last_kernel_src}
```

Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
{execution_feedback}
"""

    prompt += get_problem_instruction_improve(triton)
    return prompt


def prompt_refinement_from_history(ref_arch_src: str, history: list[tuple[str, KernelExecResult]], triton=False, rule_path=None) -> str:
    prompt = prompt_base(ref_arch_src, triton, rule_path)

    for kernel_src, exec_result in history:

        execution_feedback = exec_result_to_exeution_feedback(exec_result)

        prompt += f"""Your generated kernel:
```
{kernel_src}
```

Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
{execution_feedback}
"""
    
    prompt += get_problem_instruction_improve(triton)
    return prompt


def prompt_idea_generation(ref_arc_src: str, config, last_kernel_src: str, last_exec_result: KernelExecResult, triton=False) -> str:
    prompt = prompt_main(ref_arc_src, config, triton)
    execution_feedback = exec_result_to_exeution_feedback(last_exec_result)

    prompt += f"""Your latest generated kernel:
```
{last_kernel_src}
```

Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
{execution_feedback}
"""

    prompt += "Generate an idea for how to improve the kernel. Please do not output code yet, just the idea."
    return prompt

def prompt_refinement_from_idea(ref_arc_src: str, config, last_kernel_src: str, last_exec_result: KernelExecResult, idea: str, triton=False) -> str:
    prompt = prompt_main(ref_arc_src, config, triton)
    execution_feedback = exec_result_to_exeution_feedback(last_exec_result)

    prompt += f"""Your latest generated kernel:
```
{last_kernel_src}
```

Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
{execution_feedback}

Here is your idea for how to improve the kernel:
```
{idea}
```
"""

    prompt += get_problem_instruction(triton)
    return prompt

def get_refinement_prompt(task_id: str, task_description: str, trace, config, run_dir: str, context=None) -> str:
    evaluations = trace.get_evaluations(task_id)
    # find best solution a
    correct_evals = [eval for eval in evaluations if eval.correctness]
    if len(correct_evals) == 0:
        exec_result = evaluations[-1]
        last_solution = trace.get_solution(exec_result.evaluation_id)
    else:
        exec_result = max(correct_evals, key=lambda x: x.runtime)
        last_solution = trace.get_solution(exec_result.evaluation_id)
    return prompt_refinement_from_last_kernel(task_description, last_solution.solution_code, exec_result, triton=False, context=context)



def generate_prompt_iterative_refinement(task, config, ref_arch_src: str, llm_client, run_dir: str, triton=False, context=None) -> str:
    if task.sample_id < config.num_parallel:
        return prompt_base(ref_arch_src, config, triton, context)
    
    # Fetch previous history of kernels
    history = []
    for sample_id in range(task.sample_id % config.num_parallel, task.sample_id):
        solution_id = f"{task.task_id}_solution_{sample_id}"
        solution = task.get_solution(solution_id)
        history.append((solution.solution_code, solution.solution_evaluation))
    
    return prompt_refinement_from_history(task.task_description, history, triton, context)

def generate_prompt_metr(work: WorkArgs, config, ref_arch_src: str, llm_client, run_dir: str, triton=False, context=None) -> str:
    if work.sample_id <= config.num_parallel:
        return prompt_base(ref_arch_src, config, triton, context)
    
    # Fetch evaluation results
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    eval_results = fetch_eval_results_for_problem(work.level, work.problem_id, eval_file_path)

    ref_kernel_result = eval_results["0"]
    assert ref_kernel_result["correctness"], "Reference kernel is not correct"

    correct_kernels = [eval_result for eval_result in eval_results.values() if eval_result["correctness"]]
    
    # Sample from the correct kernels based on efficiency
    speedups = [ref_kernel_result["runtime"] / eval_result["runtime"] for eval_result in correct_kernels]
    sampled_kernel_eval_result = random.choices(correct_kernels, weights=speedups)[0]
    sampled_kernel_id = int(sampled_kernel_eval_result["sample_id"])
    if config.verbose:
        print(f"[METR] Sampled kernel {sampled_kernel_id} with speedup {ref_kernel_result['runtime'] / sampled_kernel_eval_result['runtime']}")

    sampled_kernel_src, _ = fetch_kernel_from_disk(run_dir, config.level, work.problem_id, sampled_kernel_id)

    return prompt_refinement_from_last_kernel(ref_arch_src, config, sampled_kernel_src, sampled_kernel_eval_result, triton)


def generate_prompt_stanford(work: WorkArgs, config, ref_arch_src: str, llm_client, run_dir: str, triton=False, context=None) -> str:
    if work.sample_id < config.num_parallel:
        return prompt_base(ref_arch_src, config, triton, context)
    
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    eval_results = fetch_eval_results_for_problem(work.level, work.problem_id, eval_file_path)
    # Get best kernel(s) from last round
    last_iteration_start_id = (work.sample_id // config.num_parallel - 1) * config.num_parallel
    last_step_sample_id_range = range(last_iteration_start_id, last_iteration_start_id + config.num_parallel)
    last_step_eval_results = [eval_results[str(sample_id)] for sample_id in last_step_sample_id_range]
    last_step_correct_kernels = [eval_result for eval_result in last_step_eval_results if eval_result["correctness"]]
    last_step_incorrect_kernels = [eval_result for eval_result in last_step_eval_results if not eval_result["correctness"]]
    last_step_best_kernels = sorted(last_step_correct_kernels, key=lambda x: x["runtime"])
    if len(last_step_best_kernels) < config.num_best:
        # If not enough correct kernels, randomly sample incorrect kernels
        last_step_best_kernels = last_step_best_kernels + random.choices(last_step_incorrect_kernels, k=config.num_best - len(last_step_best_kernels))

    last_step_best_kernel = last_step_best_kernels[work.sample_id % config.num_best] # use top config.num_best kernels
    last_step_best_kernel_src, _ = fetch_kernel_from_disk(run_dir, config.level, work.problem_id, int(last_step_best_kernel["sample_id"]))
    if config.verbose:
        print(f"[Stanford] Last step best kernel sample_id: {int(last_step_best_kernel['sample_id'])}")

    prompt = prompt_idea_generation(ref_arch_src, config, last_step_best_kernel_src, last_step_best_kernel, triton)

    idea = llm_client.text_completion(prompt)
    idea = idea['choices'][0]['message']['content']

    prompt = prompt_refinement_from_idea(ref_arch_src, config, last_step_best_kernel_src, last_step_best_kernel, idea, triton)
    return prompt


def generate_prompt(work, config, ref_arch_src: str, llm_client, run_dir: str, context=None, **kwargs) -> str:
    triton = "KernelLLM" in config.model_name
    match config.method:
        case "base":
            return prompt_base(ref_arch_src, triton, context)
        case "best-of-N":
            return prompt_base(ref_arch_src, triton, context)
        case "iterative refinement":
            return generate_prompt_iterative_refinement(work, ref_arch_src, llm_client, run_dir, triton, context)
        case "METR":
            return generate_prompt_metr(work, ref_arch_src, llm_client, run_dir, triton, context)
        case "Stanford":
            return generate_prompt_stanford(work, ref_arch_src, llm_client, run_dir, triton, context)
        case _:
            raise ValueError(f"Invalid method: {config.method}")
