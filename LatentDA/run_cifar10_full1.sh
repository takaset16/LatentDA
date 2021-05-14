#!/bin/sh -x

#$ -l rt_F=1
#$ -l h_rt=36:00:00
#$ -N anaconda
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5 nccl/2.5/2.5.6-1 openmpi/2.1.6 gcc/7.4.0
source ~/venv/pytorch/bin/activate

python main.py --n_model 'ResNet' --n_data 'CIFAR-10' --flag_myaug_training 1 --flag_dropout 1 --layer_aug 0 --layer_drop 1 --flag_random_layer 0 --n_aug 6 --flag_var 1 --gpu_multi 1 --loop 0
python main.py --n_model 'ResNet' --n_data 'CIFAR-10' --flag_myaug_training 1 --flag_dropout 1 --layer_aug 0 --layer_drop 1 --flag_random_layer 0 --n_aug 6 --flag_var 1 --gpu_multi 1 --loop 1
python main.py --n_model 'ResNet' --n_data 'CIFAR-10' --flag_myaug_training 1 --flag_dropout 1 --layer_aug 0 --layer_drop 1 --flag_random_layer 0 --n_aug 6 --flag_var 1 --gpu_multi 1 --loop 2
python main.py --n_model 'ResNet' --n_data 'CIFAR-10' --flag_myaug_training 1 --flag_dropout 1 --layer_aug 0 --layer_drop 1 --flag_random_layer 0 --n_aug 6 --flag_var 1 --gpu_multi 1 --loop 3
python main.py --n_model 'ResNet' --n_data 'CIFAR-10' --flag_myaug_training 1 --flag_dropout 1 --layer_aug 0 --layer_drop 1 --flag_random_layer 0 --n_aug 6 --flag_var 1 --gpu_multi 1 --loop 4

deactivate
