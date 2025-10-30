from torch.utils.data import Dataset

from external.KernelBench.src.dataset import fetch_ref_arch_from_level_problem_id, level1_representative_subset_problem_ids, level2_representative_subset_problem_ids, level3_representative_subset_problem_ids

from KernelCoder.benchmarks.KernelBench.benchmark import KernelBenchTask


class KernelBenchDataset(Dataset):
    def __init__(self, config, eval=False):
        self.dataset = []
        if config.test:
            self.dataset = [KernelBenchTask(task_id=f"level_1_problem_{problem}", level=1, problem=problem, task_description=fetch_ref_arch_from_level_problem_id(1, problem, config.dataset_src)[0]) for problem in range(1, 3)] 
        elif eval:
            self.dataset.extend([KernelBenchTask(task_id=f"level_1_problem_{problem}", level=1, problem=problem, task_description=fetch_ref_arch_from_level_problem_id(1, problem, config.dataset_src)[0]) for problem in level1_representative_subset_problem_ids])
            self.dataset.extend([KernelBenchTask(task_id=f"level_2_problem_{problem}", level=2, problem=problem, task_description=fetch_ref_arch_from_level_problem_id(2, problem, config.dataset_src)[0]) for problem in level2_representative_subset_problem_ids])
            self.dataset.extend([KernelBenchTask(task_id=f"level_3_problem_{problem}", level=3, problem=problem, task_description=fetch_ref_arch_from_level_problem_id(3, problem, config.dataset_src)[0]) for problem in level3_representative_subset_problem_ids])
        else:
            self.dataset.extend([KernelBenchTask(task_id=f"level_1_problem_{problem}", level=1, problem=problem, task_description=fetch_ref_arch_from_level_problem_id(1, problem, config.dataset_src)[0]) for problem in range(1, 101) if problem not in level1_representative_subset_problem_ids])
            self.dataset.extend([KernelBenchTask(task_id=f"level_2_problem_{problem}", level=2, problem=problem, task_description=fetch_ref_arch_from_level_problem_id(2, problem, config.dataset_src)[0]) for problem in range(1, 101) if problem not in level2_representative_subset_problem_ids])
            self.dataset.extend([KernelBenchTask(task_id=f"level_3_problem_{problem}", level=3, problem=problem, task_description=fetch_ref_arch_from_level_problem_id(3, problem, config.dataset_src)[0]) for problem in range(1, 51) if problem not in level3_representative_subset_problem_ids])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

