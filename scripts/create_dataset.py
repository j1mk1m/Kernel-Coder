"""
Create a prompt/response dataset from a KernelBench run directory.

Given a path to a run directory:
- Read the ``eval_results.json`` file.
- Filter for entries where ``correctness`` is True.
- For each correct sample, build:
  - ``prompt``: generated using KernelBench's ``prompt_base`` from the
    reference architecture.
  - ``response``: loaded from
    ``level_{level}_problem_{problem}_sample_{sample_id}_response.txt``
    inside the run directory.
- Save the resulting dataset as a Parquet file using Hugging Face
  ``datasets``.
"""

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

from datasets import Dataset


def _setup_paths():
    """
    Ensure the repository root (and its `external` directory) are on
    ``sys.path`` so that imports work when this script is run directly.
    """
    this_file = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(this_file))
    external_root = os.path.join(repo_root, "external")
    kernelbench_root = os.path.join(external_root, "KernelBench")

    if repo_root not in sys.path:
        sys.path.append(repo_root)
    if external_root not in sys.path:
        sys.path.append(external_root)
    if kernelbench_root not in sys.path:
        sys.path.append(kernelbench_root)
    
_setup_paths()

from external.KernelBench.src.dataset import (  # type: ignore  # noqa: E402
    fetch_ref_arch_from_level_problem_id,
)
from KernelCoder.benchmarks.KernelBench.prompt import (  # type: ignore  # noqa: E402
    prompt_base,
)


def iter_eval_entries(
    eval_results,
) -> Iterable[Tuple[int, int, int, Dict]]:
    """
    Yield (level, problem_id, sample_id, eval_result) tuples from
    ``eval_results``.

    Supports:
    - Nested dict format produced by KernelBench metrics collator:
      {level: {problem_id: {sample_id: eval_result}}}
    - List of dicts with explicit level/problem/sample_id fields.
    """
    # Nested dict case: {"1": {"1": {"0": eval_result}}}
    if isinstance(eval_results, dict):
        # Heuristic: values are themselves dicts -> nested structure
        for level_key, problems in eval_results.items():
            if not isinstance(problems, dict):
                continue
            for problem_key, samples in problems.items():
                if not isinstance(samples, dict):
                    continue
                for sample_key, eval_result in samples.items():
                    try:
                        level = int(level_key)
                        problem_id = int(problem_key)
                        sample_id = int(sample_key)
                    except ValueError:
                        # Skip malformed keys
                        continue
                    yield level, problem_id, sample_id, eval_result
        return

    # List-of-dicts case: [{"level": ..., "problem_id": ..., "sample_id": ..., ...}, ...]
    if isinstance(eval_results, list):
        for item in eval_results:
            if not isinstance(item, dict):
                continue
            evaluation_id = item.get("evaluation_id") # in format level_{level}_problem_{problem}_solution_{sample}
            level = evaluation_id.split("_")[1]
            problem_id = evaluation_id.split("_")[3]
            sample_id = evaluation_id.split("_")[5]
            yield int(level), int(problem_id), int(sample_id), item
        return

    raise ValueError(
        "Unsupported eval_results.json format: expected nested dict or list of dicts."
    )


def build_records_from_run_dir(run_dir: str) -> List[Dict]:
    """
    Load eval_results.json from ``run_dir`` and construct dataset
    records with fields:
        - prompt
        - response
        - level
        - problem_id
        - sample_id
    Only entries with correctness == True are kept.
    """
    eval_path = os.path.join(run_dir, "eval_results.json")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"eval_results.json not found at {eval_path}")

    with open(eval_path, "r") as f:
        eval_results = json.load(f)

    records: List[Dict] = []

    for level, problem_id, sample_id, eval_result in iter_eval_entries(eval_results):
        if not isinstance(eval_result, dict):
            continue

        if not eval_result.get("correctness", False):
            continue

        # Fetch reference architecture and construct the prompt
        ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(
            level, problem_id, "local"
        )
        prompt = prompt_base(ref_arch_src)

        # Load the corresponding response from disk
        kernel_filename = (
            f"level_{level}_problem_{problem_id}_solution_{sample_id}_kernel.py"
        )
        kernel_path = os.path.join(run_dir, kernel_filename)
        if not os.path.exists(kernel_path):
            # If the kernel file is missing, skip this entry.
            # This can happen if eval_results were collated from partial runs.
            continue
        with open(kernel_path, "r") as kf:
            kernel_src = kf.read()
        response = "```python\n" + kernel_src + "\n```"
        # response_filename = (
        #     f"level_{level}_problem_{problem_id}_solution_{sample_id}_response.txt"
        # )
        # response_path = os.path.join(run_dir, response_filename)
        # if not os.path.exists(response_path):
        #     # If the response file is missing, skip this entry.
        #     # This can happen if eval_results were collated from partial runs.
        #     continue

        records.append(
            {
                "prompt": prompt,
                "response": response,
                "level": level,
                "problem_id": problem_id,
                "sample_id": sample_id,
            }
        )

    return records


def _split_and_save_dataset(
    records: List[Dict], output_path: str, description: str
) -> None:
    """
    Randomly split records into train/eval (80%/20%) and save to separate
    parquet files.

    The given ``output_path`` is treated as a base path:
        - If it ends with '.parquet', the extension is stripped and
          '_train.parquet' / '_eval.parquet' are appended.
        - Otherwise, '_train.parquet' / '_eval.parquet' are appended
          directly.
    """
    if not records:
        raise RuntimeError(f"No records to save for {description}")

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.2, seed=42)

    if output_path.endswith(".parquet"):
        base = output_path[: -len(".parquet")]
    else:
        base = output_path

    train_path = f"{base}_train.parquet"
    eval_path = f"{base}_eval.parquet"

    os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(eval_path) or ".", exist_ok=True)

    print(
        f"{description}: {len(split['train'])} train records, "
        f"{len(split['test'])} eval records"
    )
    split["train"].to_parquet(train_path)
    split["test"].to_parquet(eval_path)
    print(f"Saved train dataset to {train_path}")
    print(f"Saved eval dataset to {eval_path}")


def create_parquet_from_run_dir(run_dir: str, output_path: str) -> None:
    """
    Build a Hugging Face Dataset from a run directory, split into train/eval,
    and save as two Parquet files.
    """
    records = build_records_from_run_dir(run_dir)
    if not records:
        raise RuntimeError(f"No correct evaluations found in {run_dir}")

    print(f"Found {len(records)} correct evaluations in {run_dir}")
    _split_and_save_dataset(records, output_path, f"Run {run_dir}")


def create_parquet_from_run_dirs(run_dirs: List[str], output_path: str) -> None:
    """
    Build a Hugging Face Dataset by aggregating multiple run directories,
    split into train/eval, and save as two Parquet files.

    Args:
        run_dirs: List of run directory paths.
        output_path: Destination Parquet file.
    """
    all_records: List[Dict] = []
    for rd in run_dirs:
        rd_abs = os.path.abspath(rd)
        records = build_records_from_run_dir(rd_abs)
        if not records:
            print(f"[WARN] No correct evaluations found in {rd_abs}, skipping.")
            continue
        # optionally annotate run_dir for downstream analysis
        for rec in records:
            rec.setdefault("run_dir", rd_abs)
        print(f"Found {len(records)} correct evaluations in {rd_abs}")
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(f"No correct evaluations found in any of: {run_dirs}")

    print(
        f"Aggregated total of {len(all_records)} correct evaluations "
        f"from {len(run_dirs)} run directories."
    )

    _split_and_save_dataset(
        all_records, output_path, f"Aggregated runs: {', '.join(run_dirs)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a prompt/response dataset from one or more KernelBench "
            "run directories."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run_dir",
        type=str,
        help=(
            "Path to a single run directory containing eval_results.json and "
            "response files."
        ),
    )
    group.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        help=(
            "Paths to multiple run directories whose correct evaluations "
            "will be aggregated into a single dataset."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Path to the output Parquet file. "
            "For --run_dir, defaults to <run_dir>/kernelbench_prompt_response.parquet. "
            "For --run_dirs, defaults to ./kernelbench_prompt_response_aggregated.parquet."
        ),
    )

    args = parser.parse_args()
    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        if args.output_path is None:
            output_path = os.path.join(
                run_dir, "kernelbench_prompt_response.parquet"
            )
        else:
            output_path = os.path.abspath(args.output_path)

        create_parquet_from_run_dir(run_dir, output_path)
        print(f"Saved dataset with prompt/response pairs to {output_path}")
    else:
        # Multiple run directories
        run_dirs = [os.path.abspath(rd) for rd in args.run_dirs]
        if args.output_path is None:
            output_path = os.path.abspath(
                "kernelbench_prompt_response_aggregated.parquet"
            )
        else:
            output_path = os.path.abspath(args.output_path)

        create_parquet_from_run_dirs(run_dirs, output_path)
        print(
            f"Saved aggregated dataset with prompt/response pairs to {output_path}"
        )


if __name__ == "__main__":
    main()


