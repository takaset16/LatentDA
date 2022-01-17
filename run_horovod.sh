module load gcc/9.3.0 python/3.8/3.8.7 openmpi/4.0.5 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1
source ~/venv/pytorch+horovod_cuda11/bin/activate
git clone -b v0.22.0 https://github.com/horovod/horovod.git


# mpirun -np 4 -map-by ppr:4:node -mca pml ob1 python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 200 --n_aug 0 --flag_als 6 --initial_als_rate 0.1 --flag_sign 0 --iter_interval 1 --flag_rate_fix 0 --flag_adversarial 0 --flag_initial_als 0 --flag_wandb 0 --flag_horovod 1 --gpu_multi 0 --loop 0
mpirun -np 4 -map-by ppr:4:node -mca pml ob1 python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 200 --n_aug 0 --flag_als 6 --flag_random_layer 0 --flag_wandb 0 --save_images 0 --flag_wandb 1 --flag_horovod 1 --gpu_multi 0 --loop 0

deactivate
