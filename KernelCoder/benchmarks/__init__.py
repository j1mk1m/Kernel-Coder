from KernelCoder.benchmarks.KernelBench.benchmark import KernelBenchBenchmark, KernelBenchTraces
from KernelCoder.benchmarks.KernelBench.dataset import KernelBenchDataset
try:
    from KernelCoder.benchmarks.flashinferbench.benchmark import FlashInferBenchBenchmark, FlashInferBenchTraces
    from KernelCoder.benchmarks.flashinferbench.dataset import FlashInferBenchDataset
except ImportError:
    print("FlashInferBench not installed")


def get_benchmark(config, run_dir, llm_client):
    if config.benchmark == "KernelBench":
        return KernelBenchBenchmark("KernelBench", run_dir, llm_client, config), KernelBenchDataset(config, eval=False), KernelBenchDataset(config, eval=True), KernelBenchTraces
    elif config.benchmark == "FlashInferBench":
        return FlashInferBenchBenchmark("FlashInferBench", run_dir, llm_client, config), FlashInferBenchDataset(config, eval=False), FlashInferBenchDataset(config, eval=True), FlashInferBenchTraces
    else:
        raise ValueError(f"Benchmark {config.benchmark} not supported")