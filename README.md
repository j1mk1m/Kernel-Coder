# Kernel-Coder: Test-time Scaling and Learning during Inference for GPU Kernel Generation

### Environment setup
```
conda create -n kernel python=3.12
conda activate kernel
pip install -r requirements.txt 
```

### Repo Structure
- external: contains external repo dependencies (KernelBench repo)
- KernelCoder: main directory; contains implementation for test-time scaling methods and learning during inference methods
    - benchmarks: contains prompts and evaluation code for benchmarks in a unified format
    - configs.py: argparser definitions
    - test_time_scaling.py: main file for test-time scaling methods
    - memory.py: helper module that implements different "memory" types used for learning during inference methods
    - main.py: main file for learning during inference methods
- scripts: bash scripts for running experiments
- deploy: deployment scripts to SLURM cluster


### Benchmarks
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench?tab=readme-ov-file)
- [FlashInfer-Bench](https://flashinfer-bench.vercel.app/): 

KernelBench arguments
- `--level 0`: if > 0, gets dataset for that level only
- `--hardware`: name of GPU used for reference results
- `--gpu_arch`: name of GPU architecture e.g. Ampere
- `--num_eval_devices 1`: number of GPUs used for evaluation


FlashInfer-Bench arguments
- `--base_traceset_path <path>`: path to the traceset path with task definitions and evaluation workloads
- `--language cuda`: language used for kernels ["cuda", "triton", "python"]
- `--target_gpu`: name of GPU used to run evaluations (used for prompt)


### Test-Time Scaling Methods
Methods
- Base (one shot): one attempt at generating kernel
- Best-of-N: $N$ parallel attempts; best kernel is selected by correctness and efficiency
- Iterative refinement: $N$ sequential attempts; new kernel is generated from execution feedback of previous kernels
- [METR](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/): $N$ sequential attempts similar to iterative refinement; new kernel is generated from execution feedback of a sampled previous kernel (sampled by efficiency)

Test-time Scaling arguments
- `--method base`: choice of ["base", "best-of-N", "iterative refinement", "METR"]
- `--num_parallel 1`: parameter used for best-of-N
- `--num_iterations 1`: parameter used for iterative refinement
- `--num_samples 1`: parameter used for METR


Example script
``` 
python KernelCoder/test_time_scaling.py \
    --run_name test_time_scaling_iterative_refinement_4 \
    --method "iterative refinement" \
    --num_parallel 1 \
    --num_iterations 4 \
    --server_type litellm \
    --model_name gemini/gemini-2.5-flash \
    --max_tokens 8192 \
    --temperature 0.7 \
    --hardware A6000_babel \
    --num_eval_devices 3 \
    --num_cpu_workers 16
```

### Learning during Inference methods
The overall idea of learning during inference is that as we scale the model's experience, the model learns from previous attempts all in natural language space (instead of updating the model weights). We model this behavior as a traidtional ML training loop. We iterate through the dataset, grouping tasks in to batches. The current model (and prompt) is used to generate kernels, which are evaluated. From this trajectory, the model "learns" by extracting useful information and adding this to the prompt. In other words, we optimize over the prompt text space.

#### ReasoningBank
Adapted from [ReasoningBank](https://arxiv.org/pdf/2509.25140v1) paper. Each task is given in a sequential manner. After each task, we extract memory from the experience and add it to a memory bank. For subsequent tasks, we retrieve top-$k$ memory items using embeddings and use it in the prompt to help the model "learn" from previous experience.

Pseudocode:
```
for epoch in range(num_epochs):
	for task in dataset:
		context = memory.retrieve_memory(task)
		solution = generate(task, context)
		eval_result = evaluate(solution)
		memory.extract(task, solution, eval_result)
```

ReasoningBank arguments
- `--memory memory`: set this to memory to use ReasoningBank method
- `--memory_model_name`: name of LLM used for memory extraction process
- `--memory_embedding_model_name`: embedding model name used for memory retrieval process


#### EvolRule
Adapted from [AutoRule](https://arxiv.org/pdf/2506.15651) paper. After rollouts and evaluations are complete, we extract rule-like statements from the trajectories using the process from AutoRule. A key difference from ReasoningBank is that we can extract rules across multiple tasks (while ReasoningBank is from one task), and filter rules by alignment across kernels from multiple tasks. Filtered rules make up the overall ruleset, which gets added to the prompt for the next generation.

```
ruleset = []
for epoch in range(num_epochs):
	for batched_tasks in dataset:
		new_ruleset = []
		for task in batched_tasks:
			solution = generate(task, ruleset)
			eval_result = evaluate(solution)
			rules = extract_rules(solution, eval_result)
			new_ruleset.add(rules)
		
		align_rules(new_ruleset, batch_trajectories)
		ruleset.add(new_ruleset)
```

EvolRule arguments
- `--memory rules`: set this to rules to use EvolRule method
- `--memory_model_name`: name of LLM used for rule extraction process


### Inference
Inference arguments
- `--temperature 0.8`
- `--max_tokens 8192`
- `--model_name gemini/gemini-2.5-flash`: see supported models on [LiteLLM](https://docs.litellm.ai/docs/providers)
- `--server_type litellm`: litellm by default. When using locally hosted model, use "vllm". See below for more details


Running local model via VLLM
1. See [deploy/serve_vllm_babel.sbatch](deploy/serve_vllm_babel.sbatch) script that starts vllm server on babel.
2. Once running, get babel machine name and port number (e.g. `babel-t9-32` and 8082)
3. Specify `--vllm_host_name babel-t9-32` and `--vllm_host_port 8082` arguments and set `--server_type vllm` when running scripts


### SFT Training
TODO


### RL Training
Currently deprecated
