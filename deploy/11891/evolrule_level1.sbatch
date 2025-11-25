#!/bin/bash
#SBATCH --partition preempt
#SBATCH --time 0-23:00 # set time
#SBATCH --exclude=babel-u9-16
#SBATCH --job-name=evolrule
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

bash scripts/11891/evolrule_level1.sh