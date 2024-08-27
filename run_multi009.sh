#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=3:00:00
#$ -j y
#$ -o results/o.$JOB_ID

source ~/venv/pytorch_cuda11/bin/activate
source /etc/profile.d/modules.sh

module load gcc/11.2.0
module load python/3.8/3.8.13
module load openmpi/4.0.5
module load cuda/11.1/11.1.1
module load cudnn/8.0/8.0.5
module load nccl/2.8/2.8.4-1

NGPU=128

export MASTER_ADDR=$(hostname)

mpirun \
  -npernode 4 \
  -np $NGPU \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_CACHE_DISABLE=1 \
  python main.py --n_model 'ResNet50' --n_data 'ImageNet' --num_epochs 90 --batch_size_training 32 --batch_size_test 32 --flag_warmup 0 --flag_layer_rate 0 --flag_als 0 --flag_rmsprop 1 --flag_lr_schedule 2 --n_aug 9 --min_rate_sum 0.2 --flag_random_layer 0 --flag_multi 1 --flag_acc5 1 --flag_wandb 1 --loop 4
