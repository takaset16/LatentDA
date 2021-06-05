source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5 nccl/2.5/2.5.6-1 openmpi/2.1.6 gcc/7.4.0
source ~/venv/pytorch/bin/activate


python main.py --n_model 'CNN' --n_data 'CIFAR-10' --batch_size_test 10000 --num_epochs 200 --flag_myaug_training 1 --flag_dropout 0 --layer_aug 0 --layer_drop 0 --flag_random_layer 1 --n_aug 3 --flag_var 0 --flag_als 0 --als_rate 0.001 --gpu_multi 1 --loop 0
# python main.py --n_model 'CNN' --n_data 'CIFAR-10' --num_epochs 5 --flag_myaug_training 1 --flag_dropout 0 --layer_aug 0 --layer_drop 0 --flag_random_layer 1 --n_aug 6 --flag_var 0 --flag_als 0 --gpu_multi 0 --loop 0

deactivate
