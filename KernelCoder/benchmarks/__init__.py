from KernelCoder.benchmarks.KernelBench.benchmark import KernelBenchBenchmark, KernelBenchTraces
from KernelCoder.benchmarks.KernelBench.dataset import KernelBenchDataset


def get_benchmark(config, run_dir, llm_client):
    if config.benchmark == "KernelBench":
        return KernelBenchBenchmark("KernelBench", run_dir, llm_client, config), KernelBenchDataset(config, eval=False), KernelBenchDataset(config, eval=True), KernelBenchTraces
    else:
        raise ValueError(f"Benchmark {config.benchmark} not supported")