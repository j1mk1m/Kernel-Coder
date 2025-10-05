python KernelCoder/test_time_scaling.py \
    --run_name test_time_scaling \
    --method base \
    --num_parallel 1 \
    --num_samples 1 \
    --num_iterations 1 \
    --num_best 1 \
    --model_name gemini/gemini-2.0-flash \
    --server_type litellm \
    --num_eval_devices 2 \
    --max_tokens 8192 \
    --temperature 0.7 \
    --hardware 8090_Ti \
    
