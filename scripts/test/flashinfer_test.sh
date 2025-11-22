python KernelCoder/test_time_scaling.py \
    --run_name flashinfer_test \
    --benchmark "FlashInferBench" \
    --base_traceset_path "/data/user_data/gyeongwk/flashinfer-trace" \
    --language "triton" \
    --target_gpu "A6000" \
    --method "base" \
    --num_parallel 1 \
    --num_iterations 1 \
    --server_type vllm \
    --vllm_host babel-t9-32 \
    --vllm_port 8082 \
    --model_name hosted_vllm/Qwen/QwQ-32B \
    --max_tokens 16384 \
    --temperature 0.7 \
    --test

