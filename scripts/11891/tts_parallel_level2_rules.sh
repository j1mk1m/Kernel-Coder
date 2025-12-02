python KernelCoder/test_time_scaling.py \
    --run_name test_time_scaling_best_of_n_QwQ-32B_level2_rules \
    --benchmark KernelBench \
    --level 2 \
    --method "best-of-N" \
    --num_parallel 4 \
    --num_iterations 1 \
    --server_type vllm \
    --vllm_host babel-t9-32 \
    --vllm_port 8082 \
    --model_name hosted_vllm/Qwen/QwQ-32B \
    --max_tokens 16384 \
    --temperature 0.7 \
    --hardware A6000_babel \
    --num_eval_devices 4 \
    --num_cpu_workers 16 \
    --rules_file runs/evolrule_parallel_8_batch_level2/epoch_0_batch_0/filtered_rules.json

