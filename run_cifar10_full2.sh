#!/bin/sh -x

#$ -l rt_G.small=1
#$ -l h_rt=36:00:00
#$ -N anaconda
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5 nccl/2.5/2.5.6-1 openmpi/2.1.6 gcc/7.4.0
source ~/venv/pytorch/bin/activate

for s in 0 1 2 3 4
  do
  python main.py --n_model 'CNN' --n_data 'CIFAR-10' --num_epochs 200 --flag_myaug_training 1 --layer_aug 0 --flag_random_layer 1 --n_aug 1 --flag_als 1 --als_rate 0.001 --epoch_random 5 --gpu_multi 0 --loop $s
  python main.py --n_model 'CNN' --n_data 'CIFAR-10' --num_epochs 200 --flag_myaug_training 1 --layer_aug 0 --flag_random_layer 1 --n_aug 5 --flag_als 1 --als_rate 0.001 --epoch_random 5 --gpu_multi 0 --loop $s
  python main.py --n_model 'CNN' --n_data 'CIFAR-10' --num_epochs 200 --flag_myaug_training 1 --layer_aug 0 --flag_random_layer 1 --n_aug 6 --flag_als 1 --als_rate 0.001 --epoch_random 5 --gpu_multi 0 --loop $s
  python main.py --n_model 'CNN' --n_data 'CIFAR-10' --num_epochs 200 --flag_myaug_training 1 --layer_aug 0 --flag_random_layer 1 --n_aug 1 --flag_als 1 --als_rate 0.001 --epoch_random 10 --gpu_multi 0 --loop $s
  python main.py --n_model 'CNN' --n_data 'CIFAR-10' --num_epochs 200 --flag_myaug_training 1 --layer_aug 0 --flag_random_layer 1 --n_aug 5 --flag_als 1 --als_rate 0.001 --epoch_random 10 --gpu_multi 0 --loop $s
  python main.py --n_model 'CNN' --n_data 'CIFAR-10' --num_epochs 200 --flag_myaug_training 1 --layer_aug 0 --flag_random_layer 1 --n_aug 6 --flag_als 1 --als_rate 0.001 --epoch_random 10 --gpu_multi 0 --loop $s
  done

deactivate

