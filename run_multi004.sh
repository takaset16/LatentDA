#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=48:00:00
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
  python main.py --n_model 'ResNet50' --n_data 'ImageNet' --num_epochs 100 --batch_size_training 64 --batch_size_test 64 --flag_defaug 1 --flag_layer_rate 1 --flag_adalase 1 --n_aug 5 --flag_rmsprop 0 --initial_als_rate 0.01 --iter_interval 10 --min_rate_sum 0.1 --flag_random_layer 0 --flag_multi 1 --flag_acc5 1 --flag_wandb 1 --loop 0

mpirun \
  -npernode 4 \
  -np $NGPU \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_CACHE_DISABLE=1 \
  python main.py --n_model 'ResNet50' --n_data 'ImageNet' --num_epochs 100 --batch_size_training 64 --batch_size_test 64 --flag_defaug 1 --flag_layer_rate 1 --flag_adalase 1 --n_aug 5 --flag_rmsprop 0 --initial_als_rate 0.01 --iter_interval 10 --min_rate_sum 0.1 --flag_random_layer 0 --flag_multi 1 --flag_acc5 1 --flag_wandb 1 --loop 1

mpirun \
  -npernode 4 \
  -np $NGPU \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_CACHE_DISABLE=1 \
  python main.py --n_model 'ResNet50' --n_data 'ImageNet' --num_epochs 100 --batch_size_training 64 --batch_size_test 64 --flag_defaug 1 --flag_layer_rate 1 --flag_adalase 1 --n_aug 5 --flag_rmsprop 0 --initial_als_rate 0.01 --iter_interval 10 --min_rate_sum 0.1 --flag_random_layer 0 --flag_multi 1 --flag_acc5 1 --flag_wandb 1 --loop 2

mpirun \
  -npernode 4 \
  -np $NGPU \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_CACHE_DISABLE=1 \
  python main.py --n_model 'ResNet50' --n_data 'ImageNet' --num_epochs 100 --batch_size_training 64 --batch_size_test 64 --flag_defaug 1 --flag_layer_rate 1 --flag_adalase 1 --n_aug 5 --flag_rmsprop 0 --initial_als_rate 0.01 --iter_interval 10 --min_rate_sum 0.1 --flag_random_layer 0 --flag_multi 1 --flag_acc5 1 --flag_wandb 1 --loop 3

mpirun \
  -npernode 4 \
  -np $NGPU \
  -x PATH \
  -x LIBRARY_PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_CACHE_DISABLE=1 \
  python main.py --n_model 'ResNet50' --n_data 'ImageNet' --num_epochs 100 --batch_size_training 64 --batch_size_test 64 --flag_defaug 1 --flag_layer_rate 1 --flag_adalase 1 --n_aug 5 --flag_rmsprop 0 --initial_als_rate 0.01 --iter_interval 10 --min_rate_sum 0.1 --flag_random_layer 0 --flag_multi 1 --flag_acc5 1 --flag_wandb 1 --loop 4

deactivate
