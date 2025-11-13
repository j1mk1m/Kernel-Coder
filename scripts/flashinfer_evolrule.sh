python KernelCoder/main.py \
    --run_name flashinfer_evolrule_sequential \
    --benchmark "FlashInferBench" \
    --base_traceset_path "/data/user_data/gyeongwk/flashinfer-trace" \
    --language "triton" \
    --target_gpu "A6000" \
    --memory rules \
    --memory_model_name gemini/gemini-2.5-flash \
    --memory_embedding_model_name gemini/gemini-embedding-001 \
    --num_epochs 1 \
    --batch_size 50 \
    --num_parallel 1 \
    --num_iterations 5 \
    --server_type litellm \
    --model_name gemini/gemini-2.5-pro \
    --max_tokens 32768 \
    --temperature 0.7

