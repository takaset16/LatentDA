#!/bin/sh

#$ -l rt_F=16
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 openmpi/4.0.5 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1
source ~/venv/pytorch+horovod_cuda11/bin/activate
git clone -b v0.22.0 https://github.com/horovod/horovod.git

NUM_GPUS_PER_NODE=4
NUM_PROCS=$(expr ${NHOSTS} \* ${NUM_GPUS_PER_NODE})

MPIOPTS="-np ${NUM_PROCS} -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0"


mpirun ${MPIOPTS} python3 main.py --n_model 'ResNet50' --n_data 'ImageNet' --num_epochs 200 --batch_size_training 64 --batch_size_test 64 --flag_als 6 --initial_als_rate 0.1 --flag_sign 0 --iter_interval 1 --flag_adversarial 0 --flag_wandb 1 --flag_lr_schedule 2 --flag_horovod 1 --gpu_multi 0 --loop 0

deactivate
