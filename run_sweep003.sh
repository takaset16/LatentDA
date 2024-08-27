#!/bin/sh

#$ -l rt_G.small=1
#$ -l h_rt=48:00:00
#$ -j y
#$ -cwd

source ~/venv/pytorch_cuda11/bin/activate
source /etc/profile.d/modules.sh

module load gcc/12.2.0
module load python/3.11/3.11.2
module load cuda/11.6/11.6.2
module load cudnn/8.9/8.9.1

wandb agent takase16/AdaLASE_029_CIFAR10/38uv8l8j

deactivate
