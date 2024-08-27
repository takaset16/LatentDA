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

python3 main.py --n_model 'ViT' --n_data 'CIFAR-10' --num_epochs 100 --n_aug 6 --flag_layer_rate 0 --flag_adalase 0 --flag_transfer 1 --flag_wandb 1 --loop 0

deactivate
