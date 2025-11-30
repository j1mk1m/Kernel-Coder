#!/bin/bash
#SBATCH --partition general
#SBATCH --time 1-23:00 # set time
#SBATCH --job-name=vllm
#SBATCH --chdir=/home/gyeongwk/Kernel-Coder # working directory

#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=/home/gyeongwk/logs/%x-%j.out # output log
#SBATCH --error=/home/gyeongwk/logs/%x-%j.err # error log

source ~/anaconda3/etc/profile.d/conda.sh

export HF_HOME=/data/user_data/$USER/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
 
export NCCL_P2P_DISABLE=1
export VLLM_CACHE_DIR=/data/user_data/$USER/.cache/vllm
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

source ~/.bashrc
conda activate test

huggingface-cli login --token "${HUGGINGFACE_TOKEN}"

MODEL="Qwen/QwQ-32B"
PORT=8082

if ss -tulwn | grep -q ":$PORT "; then
    echo "Port $PORT is already in use. Exiting..."
    exit 1
else
    vllm serve $MODEL --host 0.0.0.0 --port $PORT --tensor-parallel-size 8
fi
