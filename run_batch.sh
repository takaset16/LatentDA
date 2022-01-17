#!/bin/sh

#$ -l rt_F=1
#$ -l h_rt=48:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 cuda/11.1/11.1.1 cudnn/8.0/8.0.5
source ~/venv/pytorch_cuda11/bin/activate
git clone https://github.com/pytorch/examples.git


python3 main.py --n_model 'ResNet50' --n_data 'CIFAR-10' --num_epochs 200 --n_aug 5 --flag_als 5 --als_rate 0.1 --flag_sign 0 --iter_interval 1 --flag_adversarial 1 --flag_wandb 1 --gpu_multi 1 --loop 2

deactivate
