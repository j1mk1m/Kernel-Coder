python KernelCoder/main.py \
    --run_name flashinfer_reasoning_bank_sequential_cuda \
    --benchmark "FlashInferBench" \
    --base_traceset_path "/data/user_data/gyeongwk/flashinfer-trace" \
    --language "cuda" \
    --target_gpu "A6000" \
    --memory memory \
    --memory_model_name gemini/gemini-2.5-pro \
    --memory_embedding_model_name gemini/gemini-embedding-001 \
    --num_epochs 1 \
    --batch_size 1 \
    --num_parallel 1 \
    --num_iterations 5 \
    --server_type litellm \
    --model_name gemini/gemini-2.5-pro \
    --max_tokens 32768 \
    --temperature 0.7

