from dataclasses import dataclass

from typing import List

from KernelCoder.benchmarks.benchmark import Task, Solution, EvaluationResult

"""
Components
- Memory: retrieve, consolidate
- Dataset, DataLoader: outputs tasks, need train/eval
- Benchmark: batch_generate, evaluate, extract_memory, Task, Solution, EvaluationResult


# main loop
def main():
    memory = Memory() # initialize

    for i in range(epochs):
        for tasks in dataloader:
            tasks: List[Task] = tasks
            # Roll out batch
            solutions: List[Solution] = batch_generate(tasks, memory)
            evals: List[EvaluationResult] = batch_evaluate(solutions)

            # Extract memory item
            new_memory = batch_extract_memory(solutions, evals) # rule extraction
            memory.consolidate(new_memory)

"""


class KnowledgeBase:
    def __init__(self):
        pass

    def retrieve(self, query):
        pass

    def add(self, entry):
        pass


@dataclass
class MemoryItem:
    title: str
    description: str
    content: str

class Memory(KnowledgeBase):
    def __init__(self):
        self.memory = []

    def retrieve(self, query):
        # for each memory, calculate similarity and output top k most similar (default k=1)
        # gemini-embedding-001 model was used
        pass

    def add(self, entry):
        self.memory.append(entry)

    def extract_memory(self, solutions: List[Solution], evals: List[EvaluationResult]) -> List[MemoryItem]:
        pass

class MaTTSMemory(KnowledgeBase):
    def __init__(self):
        self.memory = []

    def retrieve(self, query):
        pass

    def add(self, entry):
        self.memory.append(entry)


class Rules(KnowledgeBase):
    def __init__(self):
        self.rules = []

    def retrieve(self, query):
        return "\n".join(self.rules)

    def add(self, entry):
        self.rules.append(entry)
    
    def extract(self, solutions: List[Solution], evals: List[EvaluationResult]) -> List[str]:
        pass

def get_memory(config):
    if config.memory == "memory":
        return Memory()
    elif config.memory == "rules":
        return Rules()
    else:
        raise ValueError(f"Invalid memory type: {config.memory}")