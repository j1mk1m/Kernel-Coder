python KernelCoder/test_time_scaling.py \
    --run_name test_time_scaling_base \
    --method "base" \
    --num_parallel 1 \
    --num_iterations 1 \
    --server_type litellm \
    --model_name anthropic/claude-sonnet-4-5-20250929 \
    --max_tokens 8192 \
    --temperature 0.7 \
    --hardware A6000_babel \
    --num_eval_devices 1 \
    --num_cpu_workers 16

