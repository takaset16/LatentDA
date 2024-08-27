#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o results/o.$JOB_ID

export PATH=$HOME/apps/openmpi/bin:$PATH

source ~/venv/pytorch_cuda11/bin/activate
source /etc/profile.d/modules.sh

module load gcc/12.2.0
module load python/3.11/3.11.2
module load cuda/11.6/11.6.2
module load cudnn/8.9/8.9.1
module load nccl/2.12/2.12.12-1

NGPU=4

export MASTER_ADDR=$(hostname)

mpirun \
  -npernode 4 \
  -np $NGPU \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_CACHE_DISABLE=1 \
  python main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 100 --batch_size_training 32 --batch_size_test 32 --flag_layer_rate 1 --flag_adalase 1 --flag_transfer 1 --n_aug 6 --initial_als_rate 0.01 --min_rate_sum 0.1 --flag_multi 1 --flag_wandb 1 --loop 0

mpirun \
  -npernode 4 \
  -np $NGPU \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_CACHE_DISABLE=1 \
  python main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 100 --batch_size_training 32 --batch_size_test 32 --flag_layer_rate 1 --flag_adalase 1 --flag_transfer 1 --n_aug 6 --initial_als_rate 0.01 --min_rate_sum 0.1 --flag_multi 1 --flag_wandb 1 --loop 1

mpirun \
  -npernode 4 \
  -np $NGPU \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_CACHE_DISABLE=1 \
  python main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 100 --batch_size_training 32 --batch_size_test 32 --flag_layer_rate 1 --flag_adalase 1 --flag_transfer 1 --n_aug 6 --initial_als_rate 0.01 --min_rate_sum 0.1 --flag_multi 1 --flag_wandb 1 --loop 2

mpirun \
  -npernode 4 \
  -np $NGPU \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_CACHE_DISABLE=1 \
  python main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 100 --batch_size_training 32 --batch_size_test 32 --flag_layer_rate 1 --flag_adalase 1 --flag_transfer 1 --n_aug 6 --initial_als_rate 0.01 --min_rate_sum 0.1 --flag_multi 1 --flag_wandb 1 --loop 3

mpirun \
  -npernode 4 \
  -np $NGPU \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_CACHE_DISABLE=1 \
  python main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 100 --batch_size_training 32 --batch_size_test 32 --flag_layer_rate 1 --flag_adalase 1 --flag_transfer 1 --n_aug 6 --initial_als_rate 0.01 --min_rate_sum 0.1 --flag_multi 1 --flag_wandb 1 --loop 4

deactivate