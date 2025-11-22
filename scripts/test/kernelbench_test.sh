python KernelCoder/test_time_scaling.py \
    --run_name test_time_scaling_test \
    --benchmark "KernelBench" \
    --method "base" \
    --num_parallel 1 \
    --num_iterations 1 \
    --server_type vllm \
    --vllm_host babel-t9-32 \
    --vllm_port 8082 \
    --model_name hosted_vllm/Qwen/QwQ-32B \
    --max_tokens 8192 \
    --temperature 0.7 \
    --hardware A6000_babel \
    --num_eval_devices 1 \
    --num_cpu_workers 16 \
    --test

