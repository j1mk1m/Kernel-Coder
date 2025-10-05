"""
Dataset processing functions for RL training
"""

################################################################################
# Data set for verl GRPO
################################################################################
TRAIN_PROBLEM_IDS_LEVEL_1 = [1, 4, 6, 8, 11, 12, 13, 15, 21, 22, 25, 27, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 46, 47, 49, 50, 52, 54, 56, 58, 59, 62, 63, 64, 65, 71, 72, 73, 74, 78, 79, 80, 81, 82, 83, 84, 85, 87, 91, 96]
KEVIN_TRAIN_PROBLEM_IDS_LEVEL_1 = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

TRAIN_PROBLEM_IDS_LEVEL_2 = [1, 6, 7, 8, 11, 14, 15, 20, 25, 26, 27, 33, 36, 38, 42, 43, 44, 45, 47, 49, 51, 53, 55, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 79, 81, 84, 85, 88, 89, 90, 96, 98, 99]
KEVIN_TRAIN_PROBLEM_IDS_LEVEL_2 = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

GOOD_PROBLEM_IDS_LEVEL_1 = [1, 7, 9, 11, 12, 17, 19, 20, 21, 28, 29, 30, 44, 45, 55, 88] 
GOOD_PROBLEM_IDS_LEVEL_2 = [4, 12, 16, 25, 31, 35, 40, 53, 54, 57, 64, 69, 70, 71, 76, 82, 90, 93, 98] 

# Set Train and Test sets
TRAIN_PROBLEM_IDS_LEVEL_1 = GOOD_PROBLEM_IDS_LEVEL_1 # set to Kevin's train set
TEST_PROBLEM_IDS_LEVEL_1 = [i for i in range(1, 101) if i not in TRAIN_PROBLEM_IDS_LEVEL_1]
TRAIN_PROBLEM_IDS_LEVEL_2 = GOOD_PROBLEM_IDS_LEVEL_2 # set to Kevin's train set
TEST_PROBLEM_IDS_LEVEL_2 = [i for i in range(1, 101) if i not in TRAIN_PROBLEM_IDS_LEVEL_2]

def check_in_train_dataset(level: int, problem_id: int) -> bool:
    if level == 1:
        return problem_id in TRAIN_PROBLEM_IDS_LEVEL_1
    elif level == 2:
        return problem_id in TRAIN_PROBLEM_IDS_LEVEL_2
    else:
        return False


def get_train_dataset():
    # return [(1, problem) for problem in TRAIN_PROBLEM_IDS_LEVEL_1] # for now use level 1 for training
    return [(1, problem) for problem in TRAIN_PROBLEM_IDS_LEVEL_1] + [(2, problem) for problem in TRAIN_PROBLEM_IDS_LEVEL_2] # for now use level 1 for training


def get_eval_dataset():
    # return [(1, problem) for problem in range(1, 101) if problem not in TRAIN_PROBLEM_IDS_LEVEL_1] # for now use level 1 for evaluation
    return [(1, problem) for problem in range(1, 101) if problem not in TRAIN_PROBLEM_IDS_LEVEL_1] + [(2, problem) for problem in range(1, 101) if problem not in TRAIN_PROBLEM_IDS_LEVEL_2] # for now use level 2 for evaluation
 

def construct_dataset(dataset):
    qa_dataset = []
    for (level, problem) in dataset:
        ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(level, problem, "local")
        question = prompt_base(ref_arch_src)
        answer = ref_arch_src
        qa_dataset.append((question, answer, level, problem))
    
    df = Dataset.from_pandas(pd.DataFrame(qa_dataset, columns=["question", "answer", "level", "problem"]))
    return df


def make_map_fn(split):
    def process_fn(example):
        question = example.pop('question')

        answer = example.pop('answer')
        level = example.pop('level')
        problem = example.pop('problem')
        data = {
            "data_source": "KernelBench",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "reward_model": {
                "style": "custom",
                "ground_truth": answer 
            },
            "extra_info": {
                'split': split,
                'level': level,
                'problem': problem,
                'answer': answer,
                "question": question,
                "interaction_kwargs": {
                    "name": "KernelBench",
                    "query": question,
                    "ground_truth": answer,
                    "split": split,
                    "level": level,
                    "problem": problem,
                }
            }
        }
        return data
    return process_fn


def process_dataset():
    trainset = get_train_dataset()
    evalset = get_eval_dataset()
    train_dataset = construct_dataset(trainset)
    eval_dataset = construct_dataset(evalset)
    train_dataset = train_dataset.map(make_map_fn('train'))
    eval_dataset = eval_dataset.map(make_map_fn('eval'))

    train_dataset.to_parquet(os.path.join(KERNEL_BENCH_PATH, "train_dataset_small.parquet"))
    eval_dataset.to_parquet(os.path.join(KERNEL_BENCH_PATH, "eval_dataset_small.parquet"))


def search_for_best_kernels(k):
    for level in [1, 2]:
        best_k_kernels = {} # problem_id -> best kernels
        for directory in os.listdir(RUNS_DIR):
            if f"level{level}" in directory:
                print(f"Searching for best kernels in {directory}")
                eval_file_path = os.path.join(RUNS_DIR, directory, "eval_results.json")
                if not os.path.exists(eval_file_path):
                    print(f"No eval results found for {directory}")
                    continue
                with open(eval_file_path, "r") as f:
                    eval_results = json.load(f)
                for problem, samples in eval_results[f"{level}"].items():
                    if problem not in best_k_kernels:
                        best_k_kernels[problem] = []
                    for sample_id, eval_result in samples.items():
                        if "metr" in directory and sample_id == "0": continue
                        if eval_result["correctness"]: # initial filter for correct
                            # check if kernel is used 
                            kernel_path = os.path.join(RUNS_DIR, directory, f"level_{level}_problem_{problem}_sample_{sample_id}_kernel.py")
                            kernel_src = read_file(kernel_path)
                            is_kernel_used = is_generated_kernel_used(kernel_src)
                            if not is_kernel_used:
                                continue
                            eval_result["run_name"] = directory
                            best_k_kernels[problem].append(eval_result)
        # sort by runtime
        for problem, kernels in best_k_kernels.items():
            best_k_kernels[problem].sort(key=lambda x: x["runtime"])
            best_k_kernels[problem] = best_k_kernels[problem][:k]

        with open(os.path.join(KERNEL_BENCH_PATH, f"best_k_kernels_level{level}.json"), "w") as f:
            json.dump(best_k_kernels, f, indent=2)
        print(f"Best {k} kernels for level {level} saved to {os.path.join(KERNEL_BENCH_PATH, f'best_k_kernels_level{level}.json')}")


def process_dataset_for_sft(k=1):
    sft_dataset = []
    sft_eval_dataset = []
    for level in [1, 2]:
        TRAIN_SET = TRAIN_PROBLEM_IDS_LEVEL_1 if level == 1 else TRAIN_PROBLEM_IDS_LEVEL_2
        with open(os.path.join(KERNEL_BENCH_PATH, f"best_k_kernels_level{level}.json"), "r") as f:
            best_k_kernels = json.load(f)
        for problem, eval_results in best_k_kernels.items():
            for eval_result in eval_results[:k]:
                run_name = eval_result["run_name"]
                kernel_path = os.path.join(RUNS_DIR, run_name, f"level_{level}_problem_{problem}_sample_{eval_result['sample_id']}_kernel.py")
                kernel_src = read_file(kernel_path)

                response_path = os.path.join(RUNS_DIR, run_name, f"level_{level}_problem_{problem}_sample_{eval_result['sample_id']}_response.txt")
                with open(response_path, "r") as f:
                    response = f.read()
                reasoning = response.split("REASONING TRACE:")[1].split("ANSWER:")[0].strip()

                ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(level, problem, "local")
                question = prompt_base(ref_arch_src)
                answer = "```python\n" + kernel_src + "\n```"
                answer = reasoning + "\n" + answer
                if int(problem) in TRAIN_SET:
                    sft_dataset.append((question, answer, level, problem))
                else:
                    sft_eval_dataset.append((question, answer, level, problem))
    
    print(f"Collected {len(sft_dataset)} train samples and {len(sft_eval_dataset)} eval samples")
    
    df = Dataset.from_pandas(pd.DataFrame(sft_dataset, columns=["question", "answer", "level", "problem"]))
    df.to_parquet(os.path.join(KERNEL_BENCH_PATH, f"sft_dataset_best_{k}_train.parquet"))
    print(f"SFT dataset saved to {os.path.join(KERNEL_BENCH_PATH, f'sft_dataset_best_{k}_train.parquet')}")
    df = Dataset.from_pandas(pd.DataFrame(sft_eval_dataset, columns=["question", "answer", "level", "problem"]))
    df.to_parquet(os.path.join(KERNEL_BENCH_PATH, f"sft_dataset_best_{k}_eval.parquet"))
    print(f"SFT eval dataset saved to {os.path.join(KERNEL_BENCH_PATH, f'sft_dataset_best_{k}_eval.parquet')}")

def get_correct_problems(run_dir):
    eval_file_path = os.path.join(run_dir, "eval_results.json")
    with open(eval_file_path, "r") as f:
        eval_results = json.load(f)
    correct_problems = []
    for level, problems in eval_results.items():
        for problem, samples in problems.items():
            for sample_id, eval_result in samples.items():
                if eval_result["correctness"]:
                    if (level, problem) not in correct_problems:
                        correct_problems.append((level, problem))
    return correct_problems

def process_correct_probems():
    run_dir = "runs/best_of_n_level1_Qwen2.5-Coder-7B-Instruct-SFT"
    correct_problems = get_correct_problems(run_dir)
    print(len(correct_problems))
    run_dir = "runs/best_of_n_level2_Qwen2.5-Coder-7B-Instruct-SFT"
    correct_problems += get_correct_problems(run_dir)
    print(len(correct_problems))

    train_dataset = construct_dataset(correct_problems)
    train_dataset = train_dataset.map(make_map_fn('train'))
    train_dataset.to_parquet(os.path.join(KERNEL_BENCH_PATH, "train_dataset_correct.parquet"))
    print(f"Train dataset saved to {os.path.join(KERNEL_BENCH_PATH, 'train_dataset_correct.parquet')}")

    incorrect_problems = []
    for level in [1, 2]:
        for problem in range(1, 101):
            if (level, problem) not in correct_problems:
                incorrect_problems.append((level, problem))
    incorrect_dataset = construct_dataset(incorrect_problems)
    incorrect_dataset = incorrect_dataset.map(make_map_fn('eval'))
    incorrect_dataset.to_parquet(os.path.join(KERNEL_BENCH_PATH, "eval_dataset_incorrect.parquet"))
    print(f"Incorrect dataset saved to {os.path.join(KERNEL_BENCH_PATH, 'eval_dataset_incorrect.parquet')}")

if __name__ == "__main__":
    process_dataset()