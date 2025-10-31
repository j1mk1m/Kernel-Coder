import argparse
import os

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_inference_args(parser, rl_training=False):
    parser.add_argument("--server_type", type=str, default="vllm")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vllm_host", type=str, default="localhost") # server_type is vllm
    parser.add_argument("--vllm_port", type=int, default=8081) # server_type is vllm
    if not rl_training:    
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--api_query_interval", type=float, default=0.0)


def add_kernelbench_args(parser):
    # Dataset
    parser.add_argument("--dataset_src", type=str, default="local")
    parser.add_argument("--dataset_name", type=str, default="ScalingIntelligence/KernelBench")
    parser.add_argument("--level", type=int, default=1)

    # Evaluation
    parser.add_argument("--eval_mode", type=str, default="local") # should be local
    parser.add_argument("--eval_server_host", type=str, default="localhost") # eval_mode is remote
    parser.add_argument("--eval_server_port", type=int, default=12345) # eval_mode is remote

    parser.add_argument("--hardware", type=str, default="A6000_babel") # GPU hardware type: this should match baseline hardware name
    parser.add_argument("--gpu_arch", type=str, default="Ampere") # GPU architecture: make sure matches hardware type
    parser.add_argument("--num_eval_devices", type=int, default=1) # number of GPUs used for evaluation

    parser.add_argument("--build_cache_with_cpu", type=bool, default=True)
    parser.add_argument("--num_cpu_workers", type=int, default=1)

    parser.add_argument("--num_correct_trials", type=int, default=5)
    parser.add_argument("--num_perf_trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--measure_performance", type=bool, default=True)


def add_evolrule_args(parser):
    parser.add_argument("--autorule_num_samples_per_problem", type=int, default=1)
    parser.add_argument("--autorule_sample_best_and_worst", type=bool, default=True)
    parser.add_argument("--autorule_num_alignment_samples", type=int, default=50)
    parser.add_argument("--autorule_total_validation_limit", type=int, default=200)
    parser.add_argument("--autorule_alignment_threshold", type=float, default=0.70)


def parse_test_time_scaling_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--benchmark", type=str, default="KernelBench")
    add_kernelbench_args(parser)

    # Methods
    parser.add_argument("--method", type=str, default="base")
    parser.add_argument("--prompt", type=str, default="regular")
    parser.add_argument("--num_parallel", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--num_best", type=int, default=1)

    # Inference Server
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    add_inference_args(parser)
    
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    return args


def parse_main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)

    # Benchmark
    parser.add_argument("--benchmark", type=str, default="KernelBench")
    add_kernelbench_args(parser)

    # Memory
    parser.add_argument("--memory", type=str, default="memory")
    parser.add_argument("--memory_model_name", type=str, default="anthropic/claude-haiku-4-5-20251001")
    parser.add_argument("--memory_embedding_model_name", type=str, default="voyage/voyage-3-large")

    # Training
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)

    # Test-time Scaling 
    parser.add_argument("--num_parallel", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=4)
 
    # Inference Server
    parser.add_argument("--model_name", type=str, default="anthropic/claude-haiku-4-5-20251001")
    add_inference_args(parser)
    
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    return args


def parse_eval_server_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_src", type=str, default="local")
    parser.add_argument("--dataset_name", type=str, default="ScalingIntelligence/KernelBench")

    parser.add_argument("--port", type=int, default=12345)

    add_eval_args(parser)

    add_logging_args(parser)

    args = parser.parse_args()
    args.method = "base"
    return args

