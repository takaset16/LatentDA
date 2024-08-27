#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=48:00:00
#$ -j y
#$ -cwd

module load gcc/12.2.0
module load python/3.11/3.11.2
module load cuda/11.6/11.6.2
module load cudnn/8.9/8.9.1
source ~/venv/pytorch_cuda11/bin/activate

python3 main.py --n_model 'MLP' --n_data 'MNIST' --batch_size_test=10000 --flag_adalase=1 --flag_adalase_test=0 --flag_layer_rate=1 --flag_rate_random=0 --flag_rmsprop=0 --initial_als_rate=0.001 --iter_interval=10 --loop=5 --n_aug=7 --num_layer=2

deactivate
