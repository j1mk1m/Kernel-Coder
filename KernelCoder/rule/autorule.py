import os
import sys
import json
import random
from tqdm import tqdm
from llm_utils import create_llm_client

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTERNAL = os.path.join(REPO_ROOT, "external")
KERNEL_BENCH_PATH = os.path.join(EXTERNAL, "KernelBench", "KernelBench")

sys.path.append(REPO_ROOT)
sys.path.append(EXTERNAL)

from KernelBench.src.utils import read_json_file


from KernelCoder.configs import parse_evolrule_args, parse_cross_model_alignment_args


def process_generated_kernels(level, run_dir):
    """
    Given directory of kernels that are evaluated, extract kernels for each problem.
    """
    # Dictionary to store processed kernels for each problem
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    with open(eval_file_path, "r") as f:
        eval_results = json.load(f)[str(level)]

    processed_kernels = {}

    for problem_id, prob_eval_results in eval_results.items():
        correct_kernels = []
        incorrect_kernels = []
        
        for sample_id, eval_result in prob_eval_results.items():
            if eval_result["correctness"] and eval_result["compiled"]:
                correct_kernels.append(eval_result)
            else:
                incorrect_kernels.append(eval_result)
        
        processed_kernels[problem_id] = {
            "correct": correct_kernels,
            "incorrect": incorrect_kernels
        } 
        
        # Sort correct kernels by runtime (lowest to highest)
        for problem_id in processed_kernels:
            processed_kernels[problem_id]["correct"].sort(key=lambda x: x["runtime"])
    
    # Save processed kernels to JSON file
    output_path = os.path.join(run_dir, "processed_kernels.json")
    with open(output_path, "w") as f:
        json.dump(processed_kernels, f, indent=2)
    
    print(f"Processed kernels saved to {output_path}")
    return processed_kernels

def process_generated_kernels_directories(level, base_run_dir, key_strings, save_path):
    processed_kernels = {}
    for directory in os.listdir(base_run_dir):
        if all(key_string in directory for key_string in key_strings):
            kernels = process_generated_kernels(level, os.path.join(base_run_dir, directory))
            for problem_id, data in kernels.items():
                correct_kernels = []
                for kernel in data["correct"]:
                    kernel["run_name"] = directory
                    correct_kernels.append(kernel)
                if problem_id not in processed_kernels:
                    processed_kernels[problem_id] = correct_kernels
                else:
                    processed_kernels[problem_id].extend(correct_kernels)
    
    # Sort the list of correct kernels for each problem_id by runtime (lowest to highest)
    for problem_id in processed_kernels:
        processed_kernels[problem_id].sort(key=lambda x: x["runtime"])
    
    with open(save_path, "w") as f:
        json.dump(processed_kernels, f, indent=2)
    print(f"Processed kernels saved to {save_path}")

    return processed_kernels


def retrieve_kernel_source_from_run_dir(run_dir, level, problem_id, sample_id):
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    with open(kernel_path, "r") as f:
        return f.read()


def rule_is_satisfied(rule, kernel_src, llm_client):
    prompt = f"""You are a kernel expert. Determine whether the following CUDA kernel satisfies the following rule.
{rule}

Be as objective as possible when evaluating the rule and do not evaluate other characteristics of the response. If the rule is not applicable for this task, treat it as if the rule is satisfied. 
You must provide your answer by strictly outputting either one of the following two options:"[[Yes]]" or "[[No]]" and nothing else

Kernel:
{kernel_src}
"""
    response = llm_client.text_completion(prompt)
    response = response["choices"][0]["text"]
    return "Yes" in response


def autorule(config, epoch_run_dir, llm_client):
    autorule_path = os.path.join(epoch_run_dir, "autorule")
    os.makedirs(os.path.join(autorule_path, "rule_generation"), exist_ok=True)
    os.makedirs(os.path.join(autorule_path, "rule_alignment"), exist_ok=True)
    processed_kernels = process_generated_kernels(config.level, os.path.join(epoch_run_dir, "generation"))

    print(f"Step 1: Comparative Analysis", flush=True)
    llm_client.start_usage_checkpoint("1_comparative_analysis")
    workload = {}
    comparative_analysis_traces = {}
    for prob, data in processed_kernels.items():
        kernels = data["correct"] if "correct" in data else data
        if len(kernels) < 2:
            print(f"[Comparative Analysis] Skipping Level {config.level} {prob} because it has less than 2 kernels")
            continue
        
        for sample_id in range(config.autorule_num_samples_per_problem):
            # Sample two kernels
            key = f"level{config.level}_{prob}_{sample_id}"
            if os.path.exists(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_response.json")):
                print(f"[Comparative Analysis] Skipping {key} because it already exists")
                with open(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_response.json"), "r") as f:
                    comparative_analysis_traces[key] = json.load(f)
                continue

            if config.autorule_sample_best_and_worst:
                kernel1 = kernels[0]
                kernel2 = kernels[-1]
            else:
                kernel1, kernel2 = random.sample(kernels, 2)

            kernel1_src = retrieve_kernel_source_from_run_dir(os.path.join(epoch_run_dir, "generation"), config.level, kernel1["problem_id"], kernel1["sample_id"])
            kernel2_src = retrieve_kernel_source_from_run_dir(os.path.join(epoch_run_dir, "generation"), config.level, kernel2["problem_id"], kernel2["sample_id"])
            prompt = f"""You are a kernel expert. You are given two CUDA kernels that solve the same problem. Both kernels are correct, but one is faster than the other. Analyze why one is faster than the other.
Kernel 1 (runtime: {kernel1['runtime']} ms):
```
{kernel1_src}
```

Kernel 2 (runtime: {kernel2['runtime']} ms):
```
{kernel2_src}
```
"""
            workload[key] = {"prompt": prompt, "kernel1": kernel1, "kernel2": kernel2}

  
    for key, value in tqdm(workload.items()):
        os.makedirs(os.path.join(autorule_path, "rule_generation", key), exist_ok=True)

        with open(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_prompt.txt"), "w") as f:
            f.write(value["prompt"])
        
        with open(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_kernels.json"), "w") as f:
            json.dump({"kernel1": value["kernel1"], "kernel2": value["kernel2"]}, f, indent=2)

        response = llm_client.text_completion(value["prompt"])
        reasoning = response["choices"][0]["reasoning_content"] if "reasoning_content" in response["choices"][0] else ""
        response = response["choices"][0]["text"]

        comparative_analysis_traces[key] = {"response": response, "reasoning": reasoning}
        with open(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_response.json"), "w") as f:
            json.dump({"response": response, "reasoning": reasoning}, f, indent=2)
        with open(os.path.join(autorule_path, "rule_generation", key, "comparative_analysis_response.txt"), "w") as f:
            f.write(f"REASONING:\n{reasoning}\n\nANSWER:\n{response}")
    llm_client.end_usage_checkpoint("1_comparative_analysis")
    llm_client.save_usage_data()


    # Step 2: Extract Rules from reasoning traces
    print("Step 2: Extract Rules from reasoning traces", flush=True)
    llm_client.start_usage_checkpoint("2_extract_rules")
    rules = []
    for key, trace in tqdm(comparative_analysis_traces.items()):
        if os.path.exists(os.path.join(autorule_path, "rule_generation", key, "rules.json")):
            print(f"[Rules] Skipping {key} because it already exists")
            with open(os.path.join(autorule_path, "rule_generation", key, "rules.json"), "r") as f:
                rules.extend(json.load(f))
            continue

        prompt = f"""Based on the following reasoning about why one kernel is faster than the other, extract any rule-like statements implied by the reasoning to indicate the difference. Rule-like statements should be ablet to be judged objectively and determinsitcially. The rules shoud be general enough to be applied to various CUDA kernels. Below are few examples of rule-like statements:
Example 1:
- The kernel performs operator fusion between multiple operations.
Example 2:
- The kernel uses shared memory tiling to reduce global memory access.
Example 3:
- The kernel uses thread block sizes that are multiples of warp size (32).
Return the list as a JSON array of strings. Do not use ``json``, just output the JSON array directly. If there are no rule-like statements, return an empty JSON array

[Reasoning]
{trace['reasoning']}
{trace['response']}
"""

        rule_response = llm_client.text_completion(prompt)
        reasoning = rule_response["choices"][0]["reasoning_content"] if "reasoning_content" in rule_response["choices"][0] else ""
        rule_response = rule_response["choices"][0]["text"]

        with open(os.path.join(autorule_path, "rule_generation", key, "rule_response.json"), "w") as f:
            json.dump({"response": rule_response, "reasoning": reasoning}, f, indent=2)
        with open(os.path.join(autorule_path, "rule_generation", key, "rule_response.txt"), "w") as f:
            f.write(f"REASONING:\n{reasoning}\n\nANSWER:\n{rule_response}")

        try:
            if "```json" in rule_response:
                rule_response = rule_response.split("```json")[1].split("```")[0].strip()

            new_rules = json.loads(rule_response)
        except Exception as e:
            print(f"Error parsing rule response for {key}: {e}")
            try:
                if "```json" in reasoning:
                    reasoning = reasoning.split("```json")[1].split("```")[0].strip()
                new_rules = json.loads(reasoning)
            except Exception as e:
                print(f"Error parsing rule response for {key}: {e}")
                new_rules = []

        rules.extend(new_rules)

        with open(os.path.join(autorule_path, "rule_generation", key, "rules.json"), "w") as f:
            json.dump(new_rules, f, indent=2)

    llm_client.end_usage_checkpoint("2_extract_rules")
    llm_client.save_usage_data()

    # Step 3: Merge rules
    print("Step 3: Merge rules", flush=True)
    llm_client.start_usage_checkpoint("3_merge_rules")
    if not os.path.exists(os.path.join(autorule_path, "rule_generation", "merged_rules.json")):    
        rules_str = "\n".join(rules)
        prompt = f"""Below is a large list of rule-like statements regarding the behavior of CUDA kernels. Some of these rules might be duplicates or very similar.
Please merge them so that there are no duplicates or very similar rules. Condense the rules into at most 25 rules.
Return the merged list as a JSON array of strings. Do not use ``json``, just output the JSON array directly. 
[Rules]
{rules_str}
"""
        rule_response = llm_client.text_completion(prompt, max_tokens=16384)
        print(rule_response)
        rule_response = rule_response["choices"][0]["text"]

        if "```json" in rule_response:
            rule_response = rule_response.split("```json")[1].split("```")[0].strip()

        with open(os.path.join(autorule_path, "rule_generation", "merged_rules_response.json"), "w") as f:
            json.dump({"response": rule_response}, f, indent=2)
        with open(os.path.join(autorule_path, "rule_generation", "merged_rules_response.txt"), "w") as f:
            f.write(f"ANSWER:\n{rule_response}")

        rules = json.loads(rule_response)
        with open(os.path.join(autorule_path, "rule_generation", "merged_rules.json"), "w") as f:
            json.dump(rules, f, indent=2)
    else:
        print(f"[Rules] Skipping {config.model_name} level{config.level} merged rules because it already exists")
        with open(os.path.join(autorule_path, "rule_generation", "merged_rules.json"), "r") as f:
            rules = json.load(f)
    llm_client.end_usage_checkpoint("3_merge_rules")
    llm_client.save_usage_data()

    
    # 4. Filter rules by alignment
    print("Step 4: Filter rules by alignment", flush=True)
    llm_client.start_usage_checkpoint("4_rule_alignment")
    results = []
    for i, rule in enumerate(rules):
        print(f"[Alignment] Rule: {rule}", flush=True)
        correct_kernels = 0
        problems = list(processed_kernels.keys())
        for problem in problems:
            kernels = processed_kernels[problem]["correct"] if "correct" in processed_kernels[problem] else processed_kernels[problem]
            correct_kernels += len(kernels)

        if correct_kernels < config.autorule_total_validation_limit:
            print(f"[Alignment] It is better to go through all correct kernels: {correct_kernels}")

            rule_alignment_results = {}
            aligned = 0
            not_aligned = 0
            both_false = 0
            both_true = 0
            problem_index = 0

            if os.path.exists(os.path.join(autorule_path, "rule_alignment", f"rule_validation_thorough_rule_{i}.json")):
                with open(os.path.join(autorule_path, "rule_alignment", f"rule_validation_thorough_rule_{i}.json"), "r") as f:
                    rule_alignment_results = json.load(f)
                aligned = rule_alignment_results["aligned"]
                not_aligned = rule_alignment_results["not_aligned"]
                both_false = rule_alignment_results["both_false"]
                both_true = rule_alignment_results["both_true"]
                problem_index = rule_alignment_results["problem_index"] + 1
                rule_alignment_results = rule_alignment_results["data"]

            for prob in tqdm(range(problem_index, len(problems)), desc="[Alignment] Validation through all correct kernels"):
                problem = problems[prob]
                rule_alignment_results[problem] = []
                kernels = processed_kernels[problem]["correct"] if "correct" in processed_kernels[problem] else processed_kernels[problem]
                for kernel in kernels:
                    kernel_src = retrieve_kernel_source_from_run_dir(os.path.join(epoch_run_dir, "generation"), config.level, kernel["problem_id"], kernel["sample_id"])
                    kernel_is_satisfied = rule_is_satisfied(rule, kernel_src, llm_client)
                    kernel["is_satisfied"] = kernel_is_satisfied
                    rule_alignment_results[problem].append(kernel)
                
                kernels = rule_alignment_results[problem]
                for k1 in range(len(kernels)):
                    for k2 in range(k1 + 1, len(kernels)):
                        kernel1 = kernels[k1]
                        kernel2 = kernels[k2]
                        if kernel1["is_satisfied"] and kernel2["is_satisfied"]:
                            both_true += 1
                        elif not kernel1["is_satisfied"] and not kernel2["is_satisfied"]:
                            both_false += 1
                        elif kernel1["is_satisfied"] and not kernel2["is_satisfied"]:
                            if kernel1["runtime"] < kernel2["runtime"]:
                                aligned += 1
                            else:
                                not_aligned += 1
                        elif not kernel1["is_satisfied"] and kernel2["is_satisfied"]:
                            if kernel1["runtime"] > kernel2["runtime"]:
                                aligned += 1
                            else:
                                not_aligned += 1
            
                alignment_rate = aligned / (aligned + not_aligned) if aligned + not_aligned > 0 else 'divide by zero'
                     
                with open(os.path.join(autorule_path, "rule_alignment", f"rule_validation_thorough_rule_{i}.json"), "w") as f:
                    json.dump({"aligned": aligned, "not_aligned": not_aligned, "both_false": both_false, "both_true": both_true, "alignment_rate": alignment_rate, "data": rule_alignment_results, "problem_index": prob}, f, indent=2)            
                llm_client.save_usage_data()

            alignment_rate = aligned / (aligned + not_aligned) if aligned + not_aligned > 0 else 'divide by zero'
            results.append({"rule": rule, "total": aligned + not_aligned, "aligned": aligned, "alignment_rate": alignment_rate, "both_false": both_false, "both_true": both_true, "count": aligned + not_aligned})
            continue

        # Sampling method
        aligned = 0
        total = 0
        count = 0
        both_false = 0
        both_true = 0
        data = []

        rule_validation_file = os.path.join(autorule_path, "rule_alignment", f"rule_validation_rule_{i}.json")
        if os.path.exists(rule_validation_file):
            print(f"Loading results for Rule: {rule} ")
            with open(rule_validation_file, "r") as f:
                data = json.load(f)
            aligned = data["aligned"]
            total = data["total"]
            both_false = data["both_false"]
            both_true = data["both_true"]
            count = data["count"]
            data = data["data"]


        while total < config.autorule_num_alignment_samples and count < config.autorule_total_validation_limit:
            count += 1
            # Randomly sample a problem and 2 kernels
            problem = random.choice(list(processed_kernels.keys()))
            kernels = processed_kernels[problem]["correct"] if "correct" in processed_kernels[problem] else processed_kernels[problem]
            # kernels = processed_kernels[problem]
            while len(kernels) < 2:
                problem = random.choice(list(processed_kernels.keys()))
                kernels = processed_kernels[problem]["correct"] if "correct" in processed_kernels[problem] else processed_kernels[problem]

            kernels = random.sample(kernels, 2)
            kernel1_src = retrieve_kernel_source_from_run_dir(os.path.join(epoch_run_dir, "generation"), config.level, kernels[0]["problem_id"], kernels[0]["sample_id"])
            kernel2_src = retrieve_kernel_source_from_run_dir(os.path.join(epoch_run_dir, "generation"), config.level, kernels[1]["problem_id"], kernels[1]["sample_id"])

            kernel1_is_satisfied = rule_is_satisfied(rule, kernel1_src, llm_client)
            kernel2_is_satisfied = rule_is_satisfied(rule, kernel2_src, llm_client)
            print(f"Kernel 1 is satisfied: {kernel1_is_satisfied}, Kernel 2 is satisfied: {kernel2_is_satisfied}")
            
            if kernel1_is_satisfied and kernel2_is_satisfied:
                both_true += 1
            elif not kernel1_is_satisfied and not kernel2_is_satisfied:
                both_false += 1
            elif kernel1_is_satisfied and not kernel2_is_satisfied:
                # Make sure kernel 1 is faster than kernel 2
                if kernels[0]["runtime"] < kernels[1]["runtime"]:
                    aligned += 1
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": True})
                else:
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": False})
                total += 1
            elif not kernel1_is_satisfied and kernel2_is_satisfied:
                if kernels[0]["runtime"] > kernels[1]["runtime"]:
                    aligned += 1
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": True})
                else:
                    data.append({"kernel1": kernels[0], "kernel2": kernels[1], "kernel1_is_satisfied": kernel1_is_satisfied, "kernel2_is_satisfied": kernel2_is_satisfied, "aligned": False})
                total += 1

            
            if count % 10 == 0:
                alignment_rate = aligned / total if total > 0 else 'divide by zero'
                with open(rule_validation_file, "w") as f:
                    json.dump({
                        "rule": rule,
                        "total": total,
                        "aligned": aligned,
                        "alignment_rate": alignment_rate,
                        "both_false": both_false,
                        "both_true": both_true,
                        "count": count,
                        "data": data
                    }, f, indent=2)
        
        alignment_rate = aligned / total if total > 0 else 'divide by zero'

        with open(rule_validation_file, "w") as f:
            json.dump({
                "rule": rule,
                "total": total,
                "aligned": aligned,
                "alignment_rate": alignment_rate,
                "both_false": both_false,
                "both_true": both_true,
                "count": count,
                "data": data
            }, f, indent=2)

        print(f"Aligned: {aligned}, Total: {total}, Alignment rate: {alignment_rate}, Count: {count}")
        res = {"rule": rule, "total": total, "aligned": aligned, "alignment_rate": alignment_rate, "both_false": both_false, "both_true": both_true, "count": count}
        results.append(res)
    llm_client.end_usage_checkpoint("4_rule_alignment")
    llm_client.save_usage_data()

    with open(os.path.join(autorule_path, "rule_alignment", f"rule_validation_results.json"), "w") as f:
        json.dump({"results": results}, f, indent=2)
    
    filtered_rules = [res["rule"] for res in results if isinstance(res["alignment_rate"], float) and res["alignment_rate"] >= config.autorule_alignment_threshold]
    with open(os.path.join(autorule_path, "rule_alignment", f"filtered_rules.json"), "w") as f:
        json.dump(filtered_rules, f, indent=2)
    with open(os.path.join(autorule_path, f"rules.json"), "w") as f:
        json.dump(filtered_rules, f, indent=2)
 
    return filtered_rules


def retrieve_kernel_source(kernel, level):
    src_file = os.path.join(REPO_TOP_DIR, "runs", kernel["run_name"], f"level_{level}_problem_{kernel['problem_id']}_sample_{kernel['sample_id']}_kernel.py")
    with open(src_file, "r") as f:
        return f.read()


if __name__ == "__main__":
    # filter_wrods = ["rules_filtered_claude_level2", "DeepSeek"]
    # process_generated_kernels_directories(2, "/home/gyeongwk/KernelBench/runs", filter_wrods, "/home/gyeongwk/KernelBench/runs/processed_kernels_epoch_1.json")

    config = parse_evolrule_args()
    # Create inference function with config parameters
    run_dir = os.path.join(REPO_TOP_DIR, "runs", "evolrule_epoch1")
    default_base_api = f"http://{config.vllm_host}:{config.vllm_port}/v1" if config.server_type == "vllm" else None
    llm_client = create_llm_client(os.path.join(run_dir, "llm_usage.json"),
                                   default_model=config.model_name,
                                   default_api_base=default_base_api,
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)
    
    autorule(config, run_dir, llm_client)



# Deprecated
def read_best_k_kernels(level: int, test: bool = False):
    if test:
        with open(os.path.join(KERNEL_BENCH_PATH, f"best_k_kernels_level{level}_small.json"), "r") as f:
            best_k_kernels = json.load(f)
    else:
        with open(os.path.join(KERNEL_BENCH_PATH, f"best_k_kernels_level{level}.json"), "r") as f:
            best_k_kernels = json.load(f)
    return best_k_kernels


# def cross_model_alignment(config):
#     os.makedirs(os.path.join(AUTORULE_PATH, "cross_model_alignment"), exist_ok=True)

#     llm_client_1 = create_llm_client(os.path.join(config.run_dir, "llm_usage.json"),
#                                    default_model=config.model_name_1,
#                                    default_api_base=f"http://{config.vllm_host_1}:{config.vllm_port_1}/v1",
#                                    default_temperature=config.temperature,
#                                    default_max_tokens=config.max_tokens)
#     llm_client_2 = create_llm_client(os.path.join(config.run_dir, "llm_usage.json"),
#                                    default_model=config.model_name_2,
#                                    default_api_base=f"http://{config.vllm_host_2}:{config.vllm_port_2}/v1",
#                                    default_temperature=config.temperature,
#                                    default_max_tokens=config.max_tokens)    

#     rules = json.load(open(config.rule_path, "r"))

#     best_kernels = read_best_k_kernels(config.level, test=False)
#     result = []
#     for rule in rules:
#         print(f"Rule: {rule}")
#         total = 0
#         both_true = 0
#         both_false = 0

#         for _ in range(100):
#             # Sample one kernel
#             problem = random.choice(list(best_kernels.keys()))
#             while len(best_kernels[problem]) < 1:
#                 problem = random.choice(list(best_kernels.keys()))
        
#             kernel = random.choice(best_kernels[problem])
#             kernel_src = retrieve_kernel_source(kernel, config.level)
#             kernel_is_satisfied_1 = rule_is_satisfied(rule, kernel_src, llm_client_1)
#             kernel_is_satisfied_2 = rule_is_satisfied(rule, kernel_src, llm_client_2)

#             if kernel_is_satisfied_1 and kernel_is_satisfied_2:
#                 both_true += 1
#             elif not kernel_is_satisfied_1 and not kernel_is_satisfied_2:
#                 both_false += 1
    
#             total += 1
        
#         aligned = both_true + both_false
        
#         alignment_rate = aligned / total if total > 0 else 'divide by zero'
#         print(f"Alignment rate: {alignment_rate}")

#         result.append({"rule": rule, "alignment_rate": alignment_rate, "total": total, "aligned": aligned, "both_true": both_true, "both_false": both_false})
    
#     with open(os.path.join(AUTORULE_PATH, "cross_model_alignment", f"cross_model_alignment_results_level{config.level}_{config.model_name_1}_{config.model_name_2}.json"), "w") as f:
#         json.dump(result, f, indent=2)

# def cross_model_alignment_main():
#     args = parse_cross_model_alignment_args()
#     cross_model_alignment(args)


