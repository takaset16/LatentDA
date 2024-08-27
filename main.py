# coding: utf-8
from mpi4py import MPI
import argparse
import torch
import torch.distributed as dist
import main_nn
import wandb
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', type=int, default=0)
    parser.add_argument('--n_data', default='CIFAR-10')
    parser.add_argument('--hidden_size', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size_training', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=1024)
    parser.add_argument('--n_model', default='CNN')
    parser.add_argument('--opt', type=int, default=1)
    parser.add_argument('--flag_wandb', type=int, default=1)
    parser.add_argument('--n_aug', type=int, default=0)
    parser.add_argument('--flag_multi', type=int, default=0)
    parser.add_argument('--cutout', type=int, default=0)
    parser.add_argument('--flag_randaug', type=int, default=0)
    parser.add_argument('--rand_n', type=int, default=0)
    parser.add_argument('--rand_m', type=int, default=0)
    parser.add_argument('--flag_warmup', type=int, default=0)
    parser.add_argument('--layer_aug', type=int, default=0)
    parser.add_argument('--flag_random_layer', type=int, default=0)
    parser.add_argument('--flag_traintest', type=int, default=0)
    parser.add_argument('--flag_adalase', type=int, default=0)
    parser.add_argument('--num_layer', type=int, default=0)
    parser.add_argument('--initial_als_rate', type=float, default=0.001)
    parser.add_argument('--iter_interval', type=int, default=1)
    parser.add_argument('--flag_defaug', type=int, default=1)
    parser.add_argument('--param_aug', type=float, default=0)
    parser.add_argument('--flag_layer_rate', type=int, default=0)
    parser.add_argument('--flag_rate_random', type=int, default=0)
    parser.add_argument('--rate_init', type=float, default=0)
    parser.add_argument('--rate_init2', type=float, default=0)
    parser.add_argument('--flag_adalase_test', type=int, default=0)
    parser.add_argument('--flag_rmsprop', type=int, default=0)
    parser.add_argument('--flag_acc5', type=int, default=0)
    parser.add_argument('--flag_lr_schedule', type=int, default=1)
    parser.add_argument('--min_rate_sum', type=float, default=0.1)
    parser.add_argument('--flag_transfer', type=int, default=0)
    parser.add_argument('--flag_save_model', type=int, default=0)
    parser.add_argument('--flag_compute_all_losses', type=int, default=0)
    parser.add_argument('--flag_load_my_weights', type=int, default=0)
    parser.add_argument('--n_aug_load', type=int, default=0)
    parser.add_argument('--degree', type=int, default=10)
    parser.add_argument('--requires_grad_transfer', type=int, default=0)
    args = parser.parse_args()

    rank = 0
    n_gpu = 1
    device = None
    if args.flag_multi == 1:
        DEFAULT_MASTER_ADDR = '127.0.0.1'
        master_addr = os.environ.get('MASTER_ADDR', DEFAULT_MASTER_ADDR)

        # [COMM] Initialize process group
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        # print(rank, size)
        n_per_node = torch.cuda.device_count()
        device = rank % n_per_node
        torch.cuda.set_device(device)
        init_method = 'tcp://{}:23456'.format(master_addr)
        dist.init_process_group('nccl', init_method=init_method, world_size=size, rank=rank)

        n_gpu = size

    flag_wandb_init = 0
    if args.flag_wandb == 1:  # Weights and Biases
        if args.flag_multi == 1:
            if rank == 0:
                flag_wandb_init = 1
        else:
            flag_wandb_init = 1

        if flag_wandb_init == 1:
            wandb.init(project="ViT_001", config=args, dir="../../../../../groups/gac50437/wandb/LatentDA")
            args = wandb.config

    main_params = main_nn.MainNN(loop=args.loop,
                                 n_data=args.n_data,
                                 hidden_size=args.hidden_size,
                                 num_samples=args.num_samples,
                                 num_epochs=args.num_epochs,
                                 batch_size_training=args.batch_size_training,
                                 batch_size_test=args.batch_size_test,
                                 n_model=args.n_model,
                                 opt=args.opt,
                                 flag_wandb=args.flag_wandb,
                                 n_aug=args.n_aug,
                                 flag_multi=args.flag_multi,
                                 rank=rank,
                                 n_gpu=n_gpu,
                                 device=device,
                                 layer_aug=args.layer_aug,
                                 flag_random_layer=args.flag_random_layer,
                                 flag_adalase=args.flag_adalase,
                                 num_layer=args.num_layer,
                                 initial_als_rate=args.initial_als_rate,
                                 iter_interval=args.iter_interval,
                                 flag_defaug=args.flag_defaug,
                                 param_aug=args.param_aug,
                                 flag_layer_rate=args.flag_layer_rate,
                                 flag_rate_random=args.flag_rate_random,
                                 rate_init=args.rate_init,
                                 rate_init2=args.rate_init2,
                                 flag_adalase_test=args.flag_adalase_test,
                                 flag_rmsprop=args.flag_rmsprop,
                                 flag_acc5=args.flag_acc5,
                                 flag_warmup=args.flag_warmup,
                                 flag_lr_schedule=args.flag_lr_schedule,
                                 min_rate_sum=args.min_rate_sum,
                                 flag_transfer=args.flag_transfer,
                                 flag_save_model=args.flag_save_model,
                                 flag_compute_all_losses=args.flag_compute_all_losses,
                                 flag_load_my_weights=args.flag_load_my_weights,
                                 n_aug_load=args.n_aug_load,
                                 degree=args.degree,
                                 requires_grad_transfer=args.requires_grad_transfer
                                 )
    main_params.run_main()


if __name__ == '__main__':
    main()
