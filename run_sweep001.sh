#!/bin/sh

#$ -l rt_G.small=1
#$ -l h_rt=36:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 cuda/11.1/11.1.1 cudnn/8.0/8.0.5
source ~/venv/pytorch_cuda11/bin/activate
git clone https://github.com/pytorch/examples.git

wandb agent takase16/Latent_019/1ecuubtj

deactivate
