# Latent Data Augmentation


## Paper
[Optimal Layer Selection for Latent Data Augmentation](https://arxiv.org/abs/2408.13426)

## How to Use
Run run.sh (interactive job) or run_sweep001.sh (batch job) on ABCI

Example
```
python3 main.py --n_model 'ResNet18' --n_data 'CIFAR-10' --num_epochs 1 --flag_save_model 1 --flag_layer_rate 0 --flag_adalase 0 --flag_transfer 0 --n_aug 7 --flag_wandb 0 --loop 0
```

## Settings 
--n_model: neural network model   
--n_data: dataset  
--num_epochs: number of training epochs  
--batch_size_training: batch size during training  
--num_samples: number of training samples (default:0)  
--n_aug: kind of data augmentation (0: no data augmentation)  
--flag_random_layer: 1 for randomly selecting DA layers  
--flag_adalase: 1 for AdaLASE  
--initial_als_rate: a hyperparameter for AdaLASE  
--flag_adversarial: 1 for adversarial AdaLASE  
--epoch_random: how many epochs after the start of training to randomly select a DA layer  
--iter_interval: Number of steps to calculate step average in AdaLASE  
--flag_alstest: 1 if test data is used for AdaLASE calculation, 0 if training data is used  
--flag_als_acc: 1 if accuracy is used for AdaLASE calculation, 0 if error is used  
--flag_wandb: 1 for weights and biases  
--save_file: 0 If the resulting file is not output  
--gpu_multi: 1 for multi-GPU  
--loop: seed for repeated trainings with different initial weights  
--mean_visual: Width of averaging when displaying value trends  
--flag_defaug: 1 when using default DA to input  
--flag_sign: 1 when using the sign function on AdaLASE and gradient descent AdaLASE   
  
