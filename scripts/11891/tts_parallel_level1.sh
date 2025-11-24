python KernelCoder/test_time_scaling.py \
    --run_name test_time_scaling_best_of_n_Qwen2.5-Coder-7B-Instruct_level1 \
    --benchmark KernelBench \
    --level 1 \
    --method "best-of-N" \
    --num_parallel 8 \
    --num_iterations 1 \
    --server_type vllm \
    --vllm_host babel-s9-24 \
    --vllm_port 8084 \
    --model_name hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct \
    --max_tokens 16384 \
    --temperature 0.7 \
    --hardware A6000_babel \
    --num_eval_devices 1 \
    --num_cpu_workers 16

