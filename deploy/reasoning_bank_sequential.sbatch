#!/bin/bash
#SBATCH --partition general
#SBATCH --time 1-23:59 # set time
#SBATCH --job-name=reasoning_bank_sequential
#SBATCH --chdir=/home/gyeongwk/Kernel-Coder # working directory

#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=/home/gyeongwk/logs/%x-%j.out # output log
#SBATCH --error=/home/gyeongwk/logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

source ~/anaconda3/etc/profile.d/conda.sh

source ~/.bashrc
conda activate kernel

bash scripts/reasoning_bank_sequential.sh