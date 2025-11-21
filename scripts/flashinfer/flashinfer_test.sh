python KernelCoder/test_time_scaling.py \
    --run_name flashinfer_reasoning_bank_sequential \
    --benchmark "FlashInferBench" \
    --base_traceset_path "/data/user_data/gyeongwk/flashinfer-trace" \
    --language "triton" \
    --target_gpu "A6000" \
    --method "base" \
    --num_parallel 1 \
    --num_iterations 10 \
    --model_name gemini/gemini-2.5-pro \
    --max_tokens 16384 \
    --temperature 0.7

