#!/bin/bash
#SBATCH --partition general
#SBATCH --time 0-23:00 # set time
#SBATCH --job-name=test
#SBATCH --chdir=/home/gyeongwk/Kernel-Coder # working directory

#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=/home/gyeongwk/logs/%x-%j.out # output log
#SBATCH --error=/home/gyeongwk/logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

source ~/anaconda3/etc/profile.d/conda.sh

export LLM_API_KEY="T5RSn/6GkE0dlna5I/I4P/g/RkmAF0mnH7j5/0fs"

source ~/.bashrc
conda activate kernel

bash scripts/test_time_scaling.sh