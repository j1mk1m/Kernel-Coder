python KernelCoder/test_time_scaling.py \
    --run_name test_time_scaling_best_of_n_Qwen2.5-Coder-7B-Instruct-kernel-only-sft-lora_level2 \
    --benchmark KernelBench \
    --level 2 \
    --method "best-of-N" \
    --num_parallel 8 \
    --num_iterations 1 \
    --server_type vllm \
    --vllm_host babel-u9-16 \
    --vllm_port 8082 \
    --model_name hosted_vllm//data/group_data/cx_group/data_centric_llm/zichunyu/11891/kernel-coder_qwen2.5-coder-7b-instruct_kernel-only_sft-lora/global_step_80/hf \
    --max_tokens 16384 \
    --temperature 0.7 \
    --hardware A6000_babel \
    --num_eval_devices 4 \
    --num_cpu_workers 16

