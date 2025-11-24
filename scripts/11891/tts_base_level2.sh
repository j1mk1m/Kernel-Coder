python KernelCoder/test_time_scaling.py \
    --run_name test_time_scaling_base_Qwen2.5-Coder-7B-Instruct_level2 \
    --benchmark KernelBench \
    --level 2 \
    --method "base" \
    --num_parallel 1 \
    --num_iterations 1 \
    --server_type vllm \
    --vllm_host babel-u9-16 \
    --vllm_port 8082 \
    --model_name hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct \
    --max_tokens 16384 \
    --temperature 0.7 \
    --hardware A6000_babel \
    --num_eval_devices 1 \
    --num_cpu_workers 16

