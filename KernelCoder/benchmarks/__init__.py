from KernelCoder.benchmarks.KernelBench.benchmark import KernelBenchBenchmark, KernelBenchTraces
from KernelCoder.benchmarks.KernelBench.dataset import KernelBenchDataset
try:
    from KernelCoder.benchmarks.flashinferbench.benchmark import FlashInferBenchBenchmark, FlashInferBenchTraces
    from KernelCoder.benchmarks.flashinferbench.dataset import FlashInferBenchDataset
except Exception as e:
    print("FlashInferBench not found")
    pass


def get_benchmark(config, run_dir, llm_client):
    if config.benchmark == "KernelBench":
        return KernelBenchBenchmark("KernelBench", run_dir, llm_client, config)
    elif config.benchmark == "FlashInferBench":
        return FlashInferBenchBenchmark("FlashInferBench", run_dir, llm_client, config)
    else:
        raise ValueError(f"Benchmark {config.benchmark} not supported")


def get_dataset(config, eval=False, **kwargs):
    if config.benchmark == "KernelBench":
        return KernelBenchDataset(config, eval=False)
        return KernelBenchDataset(config, eval=eval)
    elif config.benchmark == "FlashInferBench":
        return FlashInferBenchDataset(config, eval=eval)
    else:
        raise ValueError(f"Dataset {config.benchmark} not supported")

def get_trace_cls(config):
    if config.benchmark == "KernelBench":
        return KernelBenchTraces
    elif config.benchmark == "FlashInferBench":
        return FlashInferBenchTraces
    else:
        raise ValueError(f"Trace class for {config.benchmark} not supported")