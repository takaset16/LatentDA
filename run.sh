module load gcc/12.2.0
module load python/3.11/3.11.2
module load cuda/11.6/11.6.2
module load cudnn/8.9/8.9.1
source ~/venv/pytorch_cuda11/bin/activate

# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 1 --flag_save_model 1 --flag_layer_rate 0 --flag_adalase 0 --flag_transfer 0 --n_aug 7 --flag_wandb 0 --loop 0
# python3 main.py --n_model 'MLP' --n_data 'MNIST' --num_epochs 200 --batch_size_test 10000 --flag_layer_rate 1 --flag_adalase 0 --rate_init 0.3 --n_aug 7 --flag_wandb 0 --loop 0
# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 5 --flag_adalase 0 --rate_init 0.1 0.2 0.4 0.1 0.1 0.1 --flag_layer_rate 1 --n_aug 7 --flag_random_layer 0 --flag_acc5 0 --flag_wandb 0 --loop 0
# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 5 --flag_layer_rate 4 --flag_adalase 0 --layer_aug 0 --flag_transfer 0 --n_aug 5 --flag_random_layer 0 --flag_wandb 0 --loop 0
# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --batch_size_test 256 --flag_layer_rate 1 --num_samples 10 --flag_adalase 1 --flag_transfer 1 --n_aug 6 --initial_als_rate 0.1 --iter_interval 1 --min_rate_sum 0.1 --flag_wandb 0 --loop 4
# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-100' --num_epochs 200 --flag_defaug 1 --flag_layer_rate 1 --flag_adalase 3 --initial_als_rate 0.1 --iter_interval 1 --min_rate_sum 0.1 --n_aug 6 --flag_wandb 0 --loop 3
# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 200 --flag_adalase 0 --flag_random_layer 0 --flag_rmsprop 0 --flag_alstest 0 --n_aug 9 --flag_layer_rate 3 --flag_acc5 0 --flag_wandb 0 --loop 0
# python3 main.py --n_model 'MLP' --n_data 'CIFAR-10' --num_samples 3 --batch_size_test 10000 --flag_adalase 0 --layer_aug 2 --flag_layer_rate 1 --flag_rmsprop 0 --n_aug 7 --num_layer 2 --flag_wandb 0 --loop 0
# python3 main.py --n_model 'MLP' --n_data 'CIFAR-10' --batch_size_test 10000 --flag_layer_rate 1 --flag_adalase 1 --n_aug 6 --initial_als_rate 0.01 --iter_interval 1 --flag_compute_all_losses 1 --flag_wandb 1 --loop 0
# python3 main.py --n_model 'MLP' --n_data 'CIFAR-10' --num_epochs 5 --flag_layer_rate 1 --flag_adalase 1 --n_aug 6 --flag_save_model 1 --initial_als_rate 0.01 --iter_interval 1 --flag_wandb 0 --loop 0

# python3 main.py --n_model 'ResNet50' --n_data 'CIFAR-10' --num_epochs 2 --batch_size_training 64 --batch_size_test 64 --flag_save_model 1 --flag_defaug 1 --flag_layer_rate 0 --flag_adalase 0 --n_aug 6 --flag_random_layer 0 --flag_acc5 1 --flag_wandb 0 --loop 0
# python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 5 --flag_layer_rate 0 --flag_adalase 0 --n_aug 6 --flag_transfer 1 --n_aug_load 6 --initial_als_rate 0.01 --iter_interval 1 --flag_wandb 0 --loop 0

python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 5 --flag_layer_rate 0 --flag_random_layer 1 --flag_adalase 0 --n_aug 6 --flag_transfer 0 --flag_compute_all_losses 1 --initial_als_rate 0.01 --iter_interval 1 --flag_wandb 1 --loop 0

deactivate
