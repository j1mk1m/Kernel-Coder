from benchmarks import get_benchmark
from memory import get_memory


def main(config):
    benchmark = get_benchmark(config)
    memory = get_memory(config) 

    for i in range(epochs):
        for tasks in dataloader:
            tasks: List[Task] = tasks
            # Roll out batch
            solutions: List[Solution] = benchmark.generate_solution(tasks)
            evals: List[EvaluationResult] = benchmark.evaluate_solution(solutions)
            trajectory = benchmark.format_trajectory(tasks, solutions, evals)

            # Extract memory item
            new_memory = memory.extract_memory(trajectory) # rule extraction
            memory.add(new_memory)