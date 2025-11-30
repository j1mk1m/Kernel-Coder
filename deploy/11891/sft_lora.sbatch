#!/bin/bash
#SBATCH --partition preempt
#SBATCH --time 0-23:00 # set time
#SBATCH --job-name=sft_lora

#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=128 # cpus

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log

source ~/anaconda3/etc/profile.d/conda.sh

source ~/.bashrc
conda activate kernel

PROJECT_DIR="/data/group_data/cx_group/data_centric_llm/zichunyu/11891"
DATA_DIR="${PROJECT_DIR}/data/kernel-only"
OUTPUT_DIR="${PROJECT_DIR}/out/kernel-coder_qwen2.5-coder-7b-instruct_kernel-only_sft-lora"

bash scripts/11891/sft_lora.sh "$DATA_DIR" "$OUTPUT_DIR"