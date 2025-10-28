from KernelBench.benchmark import KernelBenchBenchmark

def get_benchmark(config):
    if config.benchmark == "KernelBench":
        return KernelBenchBenchmark()
    else:
        raise ValueError(f"Benchmark {config.benchmark} not supported")