from torch.utils.data import Dataset

from KernelCoder.benchmarks.flashinferbench.benchmark import FlashInferBenchTask

# Flashinfer Bench dependencies
from flashinfer_bench import TraceSet


class FlashInferBenchDataset(Dataset):
    def __init__(self, config, eval=False):
        base_traceset = TraceSet.from_path(config.base_traceset_path)
        self.dataset = [FlashInferBenchTask(task_id=def_name, definition=definition, task_description=definition.reference) for def_name, definition in base_traceset.definitions.items()]

        if config.test:
            self.dataset = self.dataset[:1]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

