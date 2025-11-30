#!/bin/bash
#SBATCH --partition preempt
#SBATCH --time 0-23:00 # set time
#SBATCH --job-name=tts_parallel
#SBATCH --chdir=/home/gyeongwk/Kernel-Coder # working directory

#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=128 # cpus

#SBATCH --output=/home/gyeongwk/logs/%x-%j.out # output log
#SBATCH --error=/home/gyeongwk/logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

source ~/anaconda3/etc/profile.d/conda.sh

source ~/.bashrc
conda activate kernel

DATA_DIR="/tmp/data/kernel-coder/kernel-only"
OUTPUT_DIR="/tmp/kernel-coder_qwen2.5-coder-7b-instruct_kernel-only_sft-lora"

bash scripts/11891/sft_lora.sh "$DATA_DIR" "$OUTPUT_DIR"