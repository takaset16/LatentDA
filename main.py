# coding: utf-8
import argparse
import main_nn
import wandb

'''###################################################################################
    n_data: 'MNIST', 'CIFAR-10', 'SVHN', 'STL-10', 'CIFAR-100', 'EMNIST', 
            'COIL-20', 'Fashion-MNIST', 'ImageNet', 'TinyImageNet', 
            'Letter', 'Car', 'Epileptic'
    n_aug: 0(None), 1(flips), 2(crop), 3(transfer), 4(rotation), 5(mixup), 
           6(cutout), 7(random erasing), 8(RICAP), 9(random noise)
    flag_als: 1(ALS), 2(naive-ALS), 3(greedy-ALS)
####################################################################################'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', type=int, default=0)
    parser.add_argument('--n_data', default='CIFAR-10')
    parser.add_argument('--gpu_multi', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size_training', type=int, default=256)
    parser.add_argument('--batch_size_test', type=int, default=1024)
    parser.add_argument('--n_model', default='CNN')
    parser.add_argument('--opt', type=int, default=1)
    parser.add_argument('--save_file', type=int, default=1)
    parser.add_argument('--show_params', type=int, default=0)
    parser.add_argument('--save_images', type=int, default=0)
    parser.add_argument('--flag_wandb', type=int, default=1)
    parser.add_argument('--n_aug', type=int, default=0)
    parser.add_argument('--flag_acc5', type=int, default=0)
    parser.add_argument('--flag_horovod', type=int, default=0)
    parser.add_argument('--cutout', type=int, default=0)
    parser.add_argument('--flag_dropout', type=int, default=0)
    parser.add_argument('--flag_transfer', type=int, default=0)
    parser.add_argument('--flag_randaug', type=int, default=0)
    parser.add_argument('--rand_n', type=int, default=0)
    parser.add_argument('--rand_m', type=int, default=0)
    parser.add_argument('--flag_lars', type=int, default=0)
    parser.add_argument('--lb_smooth', type=float, default=0.0)
    parser.add_argument('--flag_lr_schedule', type=int, default=1)
    parser.add_argument('--flag_warmup', type=int, default=1)
    parser.add_argument('--layer_aug', type=int, default=0)
    parser.add_argument('--layer_drop', type=int, default=0)
    parser.add_argument('--flag_random_layer', type=int, default=0)
    parser.add_argument('--flag_traintest', type=int, default=0)
    parser.add_argument('--flag_als', type=int, default=0)
    parser.add_argument('--als_rate', type=float, default=0.001)
    parser.add_argument('--epoch_random', type=int, default=0)
    parser.add_argument('--iter_interval', type=int, default=1)
    parser.add_argument('--flag_adversarial', type=int, default=0)
    parser.add_argument('--flag_alstest', type=int, default=1)
    parser.add_argument('--flag_als_acc', type=int, default=0)
    args = parser.parse_args()

    if args.flag_wandb == 1:  # Weights and Biases
        wandb.init(project="LatentDA_003", config=args)
        args = wandb.config

    main_params = main_nn.MainNN(loop=args.loop,
                                 n_data=args.n_data,
                                 gpu_multi=args.gpu_multi,
                                 hidden_size=args.hidden_size,
                                 num_samples=args.num_samples,
                                 num_epochs=args.num_epochs,
                                 batch_size_training=args.batch_size_training,
                                 batch_size_test=args.batch_size_test,
                                 n_model=args.n_model,
                                 opt=args.opt,
                                 save_file=args.save_file,
                                 show_params=args.show_params,
                                 save_images=args.save_images,
                                 flag_wandb=args.flag_wandb,
                                 n_aug=args.n_aug,
                                 flag_acc5=args.flag_acc5,
                                 flag_horovod=args.flag_horovod,
                                 cutout=args.cutout,
                                 flag_dropout=args.flag_dropout,
                                 flag_transfer=args.flag_transfer,
                                 flag_randaug=args.flag_randaug,
                                 rand_n=args.rand_n,
                                 rand_m=args.rand_m,
                                 flag_lars=args.flag_lars,
                                 lb_smooth=args.lb_smooth,
                                 flag_lr_schedule=args.flag_lr_schedule,
                                 flag_warmup=args.flag_warmup,
                                 layer_aug=args.layer_aug,
                                 layer_drop=args.layer_drop,
                                 flag_random_layer=args.flag_random_layer,
                                 flag_traintest=args.flag_traintest,
                                 flag_als=args.flag_als,
                                 als_rate=args.als_rate,
                                 epoch_random=args.epoch_random,
                                 iter_interval=args.iter_interval,
                                 flag_adversarial=args.flag_adversarial,
                                 flag_alstest=args.flag_alstest,
                                 flag_als_acc=args.flag_als_acc
                                 )
    main_params.run_main()


if __name__ == '__main__':
    main()
