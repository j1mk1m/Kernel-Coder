#!/bin/bash
# LoRA SFT training script for Kernel-Coder dataset on Qwen2.5-Coder-7B-Instruct

set -x

REPO_ROOT="$(pwd)"
DATA_DIR="${1:-/tmp/data/kernel-coder/kernel-only}"
OUTPUT_DIR="${2:-/tmp/kernel-coder_qwen2.5-coder-7b-instruct_kernel-only_sft-lora}"

_merge_on_exit() {
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "Training failed (exit code $rc), skipping model merge" >&2
        return
    fi
    latest_dir="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'global_step_*' -printf '%p\n' 2>/dev/null | sort -V | tail -n1)"
    if [ -z "$latest_dir" ]; then
        echo "No global_step_* directory found in $OUTPUT_DIR" >&2
        return
    fi
    echo "Merging latest checkpoint: $latest_dir"
    PYTHONPATH="${REPO_ROOT}/external/verl${PYTHONPATH:+:$PYTHONPATH}" \
        python -m verl.model_merger merge --backend fsdp \
        --local_dir "${latest_dir}/" \
        --target_dir "${latest_dir}/hf"
}
trap _merge_on_exit EXIT

PYTHONPATH="${REPO_ROOT}/external/verl${PYTHONPATH:+:$PYTHONPATH}" \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    hydra.run.dir=$OUTPUT_DIR \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/eval.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=8192 \
    data.truncation='left' \
    optim.lr=1e-4 \
    model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.strategy=fsdp \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=kernel-coder-sft \
    trainer.experiment_name=qwen2.5-coder-7b-instruct-lora \
    trainer.logger='["console","wandb"]' \
    trainer.test_freq=1 \
    trainer.save_freq=10 \
    trainer.total_epochs=5 $@
