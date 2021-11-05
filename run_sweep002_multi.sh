#!/bin/sh -x

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -N anaconda
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5 nccl/2.5/2.5.6-1 openmpi/2.1.6 gcc/7.4.0
source ~/venv/pytorch/bin/activate

wandb agent takase16/LatentDA_011/d47se8wu

deactivate
