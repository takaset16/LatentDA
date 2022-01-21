module load gcc/9.3.0 python/3.8/3.8.7 cuda/11.1/11.1.1 cudnn/8.0/8.0.5
source ~/venv/pytorch_cuda11/bin/activate
git clone https://github.com/pytorch/examples.git


# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 200 --n_aug 0 --flag_als 0 --flag_random_layer 0 --flag_wandb 0 --save_images 0 --flag_wandb 0 --gpu_multi 0 --loop 0
# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 200 --n_aug 6 --flag_als 2 --initial_als_rate 0.1 --iter_interval 10 --flag_adversarial 0 --flag_wandb 0 --gpu_multi 0 --loop 0
# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 200 --n_aug 0 --flag_als 3 --initial_als_rate 0.1 --iter_interval 1 --flag_adversarial 0 --flag_wandb 0 --save_file 1 --gpu_multi 0 --loop 0
# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 200 --n_aug 7 --flag_als 4 --initial_als_rate 0.01 --iter_interval 1 --flag_adversarial 0 --flag_wandb 0 --gpu_multi 0 --loop 0
python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 200 --n_aug 0 --flag_als 5 --flag_rate_fix 0 --initial_als_rate 0.1 --iter_interval 100 --flag_adversarial 0 --flag_wandb 0 --gpu_multi 0 --loop 0

deactivate
