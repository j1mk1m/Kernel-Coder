from KernelCoder.benchmarks.KernelBench.benchmark import KernelBenchBenchmark, KernelBenchTraces
from KernelCoder.benchmarks.KernelBench.dataset import KernelBenchDataset
from KernelCoder.benchmarks.flashinferbench.benchmark import FlashInferBenchBenchmark, FlashInferBenchTraces
from KernelCoder.benchmarks.flashinferbench.dataset import FlashInferBenchDataset


def get_benchmark(config, run_dir, llm_client):
    if config.benchmark == "KernelBench":
        return KernelBenchBenchmark("KernelBench", run_dir, llm_client, config)
    elif config.benchmark == "FlashInferBench":
        return FlashInferBenchBenchmark("FlashInferBench", run_dir, llm_client, config)
    else:
        raise ValueError(f"Benchmark {config.benchmark} not supported")


def get_dataset(config, eval=False, **kwargs):
    if config.benchmark == "KernelBench":
        if kwargs.get("level") is not None:
            return KernelBenchDataset(config, eval=False, level=kwargs["level"])
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