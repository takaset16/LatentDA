# coding: utf-8
import sys
import timeit
import wandb
import dataset
from models import *
from util import enable_running_stats, disable_running_stats
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torchvision
from warmup_scheduler import GradualWarmupScheduler
from torchinfo import summary
import sys
import time
from torchvision.models.resnet import ResNet18_Weights


class MainNN(object):
    def __init__(self, loop, n_data, hidden_size, num_samples, num_epochs, batch_size_training, batch_size_test,
                 n_model, opt, flag_wandb, flag_multi, rank, n_gpu, device, n_aug, layer_aug, flag_random_layer, flag_adalase,
                 num_layer, initial_als_rate, iter_interval, flag_defaug, param_aug, flag_layer_rate, flag_rate_random,
                 rate_init, rate_init2, flag_adalase_test, flag_rmsprop, flag_acc5, flag_warmup, flag_lr_schedule,
                 min_rate_sum, flag_transfer, flag_save_model, flag_compute_all_losses, flag_load_my_weights, n_aug_load, degree, requires_grad_transfer):
        self.seed = 1001 + loop
        self.train_loader = None
        self.test_loader = None
        self.n_data = n_data  # dataset
        self.input_size = 0  # only MLP
        self.hidden_size = hidden_size  # only MLP
        self.num_classes = 10
        self.num_channel = 0
        self.num_training_data = num_samples
        self.num_test_data = 0
        self.num_epochs = num_epochs
        self.batch_size_training = batch_size_training
        self.batch_size_test = batch_size_test
        self.n_model = n_model
        self.loss_training_batch = None  # minibatch loss
        self.opt = opt  # optimizer
        self.flag_wandb = flag_wandb  # weights and biases
        self.flag_multi = flag_multi
        self.rank = rank
        self.n_gpu = n_gpu
        self.device = device
        self.initial_lr = 0.01
        self.n_aug = n_aug  # data augmentation
        self.flag_defaug = flag_defaug  # Default augmentation
        self.layer_aug = layer_aug
        self.flag_random_layer = flag_random_layer
        self.als_loader = 0
        self.flag_adalase = flag_adalase
        self.num_layer = num_layer
        self.initial_als_rate = initial_als_rate
        self.iter_interval = iter_interval
        self.param_aug = param_aug
        self.iter = 0
        self.flag_layer_rate = flag_layer_rate
        self.flag_rate_random = flag_rate_random
        self.rate_init = rate_init
        self.rate_init2 = rate_init2
        self.flag_adalase_test = flag_adalase_test
        self.flag_rmsprop = flag_rmsprop
        self.flag_acc5 = flag_acc5
        self.flag_warmup = flag_warmup
        self.flag_lr_schedule = flag_lr_schedule
        self.min_rate_sum = min_rate_sum
        self.flag_transfer = flag_transfer
        self.flag_save_model = flag_save_model
        self.flag_compute_all_losses = flag_compute_all_losses
        self.flag_load_my_weights = flag_load_my_weights
        self.n_aug_load = n_aug_load
        self.degree = degree
        self.requires_grad_transfer = requires_grad_transfer

    def run_main(self):
        """Settings for random number"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        """Log, Weights and Biases"""
        flag_rank0 = 0
        if self.flag_multi == 1:
            if self.rank == 0:
                flag_rank0 = 1
        else:
            flag_rank0 = 1

        """Number of layers(positions)"""
        if self.num_layer == 0:
            if self.n_model == 'ResNet18' or self.n_model == 'ResNet50':
                self.num_layer = 6
            elif self.n_model == 'WideResNet28_10':
                self.num_layer = 6
            elif self.n_model == 'MLP':
                self.num_layer = 2

        """dataset"""
        als_dataset = None
        traintest_dataset = dataset.MyDataset_training(n_data=self.n_data, flag_defaug=self.flag_defaug, flag_transfer=self.flag_transfer)

        if self.flag_adalase >= 1 or self.flag_compute_all_losses == 1:
            als_dataset = dataset.MyDataset_als(n_data=self.n_data, flag_defaug=self.flag_defaug, flag_transfer=self.flag_transfer, degree=self.degree)
        self.num_channel, self.num_classes, self.input_size, self.hidden_size, num_samples_all = traintest_dataset.get_info(n_data=self.n_data)

        train_sampler = None
        test_sampler = None
        als_sampler = None

        if self.num_training_data > 0:
            train_dataset, _ = torch.utils.data.random_split(traintest_dataset, [self.num_training_data, num_samples_all - self.num_training_data], generator=torch.Generator().manual_seed(1001))
        else:
            train_dataset = traintest_dataset
        test_dataset = dataset.MyDataset_test(n_data=self.n_data, flag_transfer=self.flag_transfer)

        num_workers = 2
        if self.flag_multi == 1:
            num_workers = 8
            pin = True
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=self.n_gpu, rank=self.rank)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=self.n_gpu, rank=self.rank)

            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size_training, sampler=train_sampler,
                                                            shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=pin, persistent_workers=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, sampler=test_sampler,
                                                           shuffle=(test_sampler is None), num_workers=num_workers, pin_memory=pin, persistent_workers=True)
            if self.flag_adalase >= 1:
                if self.flag_adalase_test == 1:
                    self.als_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, sampler=test_sampler,
                                                                  shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=True)
                else:
                    als_sampler = torch.utils.data.distributed.DistributedSampler(als_dataset, num_replicas=self.n_gpu, rank=self.rank)
                    self.als_loader = torch.utils.data.DataLoader(dataset=als_dataset, batch_size=self.batch_size_test, sampler=als_sampler,
                                                                  shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        else:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size_training, sampler=train_sampler,
                                                            shuffle=True, num_workers=num_workers, pin_memory=False)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, sampler=test_sampler,
                                                           shuffle=False, num_workers=num_workers, pin_memory=False)
            if self.flag_adalase >= 1 or self.flag_compute_all_losses == 1:
                if self.flag_adalase_test == 1:
                    self.als_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, sampler=test_sampler,
                                                                  shuffle=True, num_workers=num_workers, pin_memory=False)
                else:
                    self.als_loader = torch.utils.data.DataLoader(dataset=als_dataset, batch_size=self.batch_size_test, sampler=als_sampler,
                                                                  shuffle=True, num_workers=num_workers, pin_memory=False)

        """neural network model"""
        model = None
        if self.n_model == 'MLP':
            model = mlp.MLPNet(input_size=self.input_size, hidden_size=self.hidden_size, num_classes=self.num_classes)
        elif self.n_model == 'ResNet18':
            if self.n_data == 'ImageNet':
                model = resnet_transfer.resnet18()
            else:
                if self.flag_transfer == 1:  # transfer learning
                    if self.flag_load_my_weights == 1:
                        model = resnet_transfer.resnet18()
                        model.load_state_dict(torch.load("../../../../../groups/gac50437/model_weights/weight_%s_ImageNet_naug%s_seed1001.pth" % (self.n_model, self.n_aug_load)))
                    else:
                        weights = ResNet18_Weights.DEFAULT
                        model = resnet_transfer.resnet18(weights=weights)

                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                    model.num_classes = self.num_classes
                else:
                    model = resnet_cifar.ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=self.num_classes, num_channel=self.num_channel)
        elif self.n_model == 'ResNet50':
            if self.n_data == 'ImageNet':
                model = resnet2.resnet50()
                # print(model.state_dict().keys())
                # model = resnet_transfer.resnet50()
            else:
                if self.flag_transfer == 1:  # transfer learning
                    if self.flag_load_my_weights == 1:
                        # model = resnet_transfer.resnet50()
                        model = resnet2.resnet50()
                        # print(model.state_dict().keys())
                        model.load_state_dict(torch.load("../../../../../groups/gac50437/model_weights/weight_%s_ImageNet_naug%s_seed1001.pth" % (self.n_model, self.n_aug_load)))
                    else:
                        weights = ResNet50_Weights.DEFAULT
                        model = resnet_transfer.resnet18(weights=weights)

                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                    model.num_classes = self.num_classes
                else:
                    model = resnet_cifar.ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=self.num_classes, num_channel=self.num_channel)
        elif self.n_model == 'WideResNet28_10':
            model = wideresnet.WideResNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=self.num_classes, num_channel=self.num_channel)
        elif self.n_model == 'EfficientNet':
            if self.flag_transfer == 1:
                model = EfficientNet.from_pretrained(model_name='efficientnet-b3')
            else:
                model = EfficientNet.from_name(model_name='efficientnet-b3')
        elif self.n_model == 'ViT':
            if self.flag_transfer == 1:  # transfer learning
                model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
                for param in model.parameters():
                    if self.requires_grad_transfer == 0:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                model.heads[0] = nn.Linear(768, self.num_classes)
            else:
                model = vit.ViT(
                    image_size = 256,
                    patch_size = 32,
                    num_classes = 1000,
                    dim = 1024,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048,
                    dropout = 0.1,
                    emb_dropout = 0.1
                )

        """GPU setting"""
        if self.flag_multi == 1:
            device = self.device
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            if device == 'cuda':
                torch.backends.cudnn.benchmark = True
                print('GPU={}'.format(torch.cuda.device_count()))

        """optimizer"""
        if self.n_model == 'ResNet18' or self.n_model == 'ResNet50':
            if self.n_data == "ImageNet":
                lr = 1.0
                weight_decay = 0.0001
            else:
                if self.flag_transfer == 1:  # transfer learning
                    lr = 0.01
                else:
                    lr = 0.1
                weight_decay = 0.0001
        elif self.n_model == 'WideResNet28_10':
            if self.n_data == "ImageNet":
                lr = 1.0
                weight_decay = 0.0001
            elif self.n_data == 'SVHN':
                lr = 0.005
                weight_decay = 0.001
            else:
                if self.flag_transfer == 1:  # transfer learning
                    lr = 0.01
                else:
                    lr = 0.1
                weight_decay = 0.0005
        elif self.n_model == 'MLP':
            lr = 0.02
            weight_decay = 0.0005
        elif self.n_model == 'EfficientNet':
            lr = 0.1
            weight_decay = 0.0005
        else:
            lr = 0.01
            weight_decay = 0.0005

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )

        """Learning rate schedule"""
        scheduler = None
        if self.flag_lr_schedule == 1:  # CosineAnnealingLR
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0.)
        elif self.flag_lr_schedule == 2:  # MultiStepLR
            if self.num_epochs == 90:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80])
            elif self.num_epochs == 180 or self.num_epochs == 200:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160])
            elif self.num_epochs == 270:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [90, 180, 240])
        elif self.flag_lr_schedule == 3:  # StepLR
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        if self.flag_warmup == 1:  # Warmup
            if self.n_model == 'ResNet50' or self.n_model == 'WideResNet28_10' or self.n_model == 'EfficientNet':
                if self.n_data == 'ImageNet':
                    multiplier = 8  # batch size: 2048
                else:
                    multiplier = 2  # batch size: 512
                total_epoch = 3
            elif self.n_model == 'WideResNet28_10':
                multiplier = 2
                if self.n_data == 'SVHN':
                    total_epoch = 3
                else:
                    total_epoch = 5
            else:
                multiplier = 2
                total_epoch = 3

            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=multiplier,
                total_epoch=total_epoch,
                after_scheduler=scheduler
            )

        """Initialization"""
        if self.n_data == "ImageNet":
            self.flag_acc5 = 1

        if self.flag_acc5 == 1:
            results = np.zeros((self.num_epochs, 5))
        else:
            results = np.zeros((self.num_epochs, 4))
        start_time = timeit.default_timer()

        if self.flag_compute_all_losses == 1:
            loss_als_layer = np.zeros(self.num_layer)
            count_layer_max = np.zeros(self.num_layer)

        if self.flag_adalase == 1:
            num_aug = 1
            layer_rate = np.zeros(self.num_layer)
            layer_rate_delta = np.zeros(self.num_layer)
        elif self.flag_adalase == 2:
            num_aug = 2
            layer_rate = np.zeros((self.num_layer, num_aug))
            layer_rate_delta = np.zeros((self.num_layer, num_aug))
        elif self.flag_adalase == 3:
            num_aug = 5
            layer_rate = np.zeros((self.num_layer, num_aug))
            layer_rate_delta = np.zeros((self.num_layer, num_aug))
        else:
            num_aug = 1
            layer_rate = np.zeros(self.num_layer)

        images_als_origin = None
        labels_als_origin = None
        h0_rms = 0
        count_best_layer = 0
        count_worst_layer = 0
        count_all_iter = 0

        """Decide an initial als_rate"""
        if self.flag_adalase == 1:  # AdaLASE
            self.flag_layer_rate = 1
            if self.flag_rate_random == 1:  # Set rate_init to random values
                if self.num_layer == 2:
                    layer_rate[0] = np.random.rand()
                    layer_rate[1] = 1 - layer_rate[0]
                else:
                    for i in range(self.num_layer):
                        layer_rate[i] = np.random.rand()

                    layer_rate = layer_rate / np.sum(layer_rate)
            else:
                sum = 0
                for i in range(self.num_layer):
                    if i < self.num_layer - 1:
                        layer_rate[i] = 1.0 / self.num_layer
                        sum += layer_rate[i]

                layer_rate[self.num_layer - 1] = 1.0 - sum

        elif self.flag_adalase >= 2:  # AdaLASE
            self.flag_layer_rate = 1
            sum = 0
            for i in range(self.num_layer):
                for j in range(num_aug):
                    if i < self.num_layer - 1 or j < num_aug - 1:
                        layer_rate[i][j] = 1.0 / (self.num_layer * num_aug)
                        sum += layer_rate[i][j]

            layer_rate[self.num_layer - 1][num_aug - 1] = 1.0 - sum

        else:
            if self.flag_layer_rate == 1:  # Use layer rate
                if self.n_model == 'MLP':
                    if self.num_layer == 2:
                        if self.rate_init > 1.0:
                            sys.exit()

                        layer_rate[0] = self.rate_init
                        layer_rate[1] = 1.0 - self.rate_init
                        if layer_rate[1] < 0:
                            layer_rate[1] = 0
                    else:
                        if self.rate_init + self.rate_init2 > 1.0:
                            sys.exit()

                        layer_rate[0] = self.rate_init
                        layer_rate[1] = self.rate_init2
                        layer_rate[2] = 1.0 - self.rate_init - self.rate_init2
                        if layer_rate[2] < 0:
                            layer_rate[2] = 0

                elif self.n_model == 'ResNet18':
                    layer_rate_sum = 0
                    for i in range(self.num_layer - 1):
                        layer_rate[i] = self.rate_init[i]
                        layer_rate_sum += self.rate_init[i]
                    layer_rate[self.num_layer - 1] = 1 - layer_rate_sum
            elif self.flag_layer_rate == 2:
                layer_rate = np.arange(self.num_layer)
                layer_rate = layer_rate / np.sum(layer_rate)

                sum = 0
                for i in range(self.num_layer):
                    if i < self.num_layer - 1:
                        sum += layer_rate[i]
                layer_rate[self.num_layer - 1] = 1.0 - sum

            elif self.flag_layer_rate == 3:
                layer_rate = np.arange(self.num_layer)
                layer_rate = np.sort(layer_rate)[::-1]
                layer_rate = layer_rate / np.sum(layer_rate)

                sum = 0
                for i in range(self.num_layer):
                    if i < self.num_layer - 1:
                        sum += layer_rate[i]
                layer_rate[self.num_layer - 1] = 1.0 - sum

            elif self.flag_layer_rate == 4:
                for i in range(self.num_layer):
                    if i < self.num_layer / 2:
                        layer_rate[i] = i + 1
                    else:
                        layer_rate[i] = self.num_layer - i

                layer_rate = layer_rate / np.sum(layer_rate)

                sum = 0
                for i in range(self.num_layer):
                    if i < self.num_layer - 1:
                        sum += layer_rate[i]
                layer_rate[self.num_layer - 1] = 1.0 - sum

            elif self.flag_layer_rate == 5:
                for i in range(self.num_layer):
                    if i < self.num_layer / 2:
                        layer_rate[i] = self.num_layer / 2 - i
                    else:
                        layer_rate[i] = i - self.num_layer / 2 + 1

                layer_rate = layer_rate / np.sum(layer_rate)

                sum = 0
                for i in range(self.num_layer):
                    if i < self.num_layer - 1:
                        sum += layer_rate[i]
                layer_rate[self.num_layer - 1] = 1.0 - sum

        for epoch in range(self.num_epochs):
            """Training"""
            start_epoch_time = timeit.default_timer()
            if self.flag_multi == 1:
                train_sampler.set_epoch(epoch)

            model.train()

            loss_training_all = 0
            loss_test_all = 0
            loss_als_all = 0
            total_steps = len(self.train_loader)
            step = 0

            if self.flag_adalase >= 1 or self.flag_compute_all_losses == 1:
                for i, (images, labels, _) in enumerate(self.als_loader):
                    if i == 0:
                        if np.array(images.data).ndim == 3:
                            images_als_origin = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)
                        else:
                            images_als_origin = images.to(device)
                        labels_als_origin = labels.to(device)

                        break

            "Compute validation loss in all layers"
            layer_loss_als_min = -1
            layer_loss_als_max = -1
            if self.flag_compute_all_losses == 1:
                for p in optimizer.param_groups[0]["params"]:
                    optimizer.state[p]["old_p"] = p.data.clone()

                for j in range(self.num_layer):
                    model.train()
                    for i, (images, labels, _) in enumerate(self.train_loader):
                        if np.array(images.data).ndim == 3:
                            images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)
                        else:
                            images = images.to(device)
                        labels = labels.to(device)

                        if self.n_model == "ViT":
                            images, labels = util.run_n_aug(images, labels, self.n_aug, self.num_classes, 0)
                            outputs = model(images)
                            labels_train_layer = labels
                        else:
                            outputs, labels_train_layer = model(x=images, y=labels, num_layer=self.num_layer, n_aug=self.n_aug, layer_aug=j)

                        if labels_train_layer.ndim == 1:
                            labels_train_layer = torch.eye(self.num_classes, device='cuda')[labels_train_layer].clone()  # To one-hot

                        loss_training = util.dist_loss(outputs, labels_train_layer)
                        loss_training_all += loss_training.item() * outputs.shape[0]  # Sum of losses within this minibatch

                        optimizer.zero_grad()
                        loss_training.backward()  # compute gradients
                        optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        outputs_als_layer, labels_als_layer = model(x=images_als_origin, y=labels_als_origin, num_layer=self.num_layer, n_aug=0, layer_aug=0)

                        if labels_als_layer.ndim == 1:
                            labels_als_layer = torch.eye(self.num_classes, device='cuda')[labels_als_layer].clone()  # To one-hot

                        loss_als_layer[j] = util.dist_loss(outputs_als_layer, labels_als_layer)

                        for p in optimizer.param_groups[0]["params"]:
                            p.data = optimizer.state[p]["old_p"]
                layer_loss_als_min = np.argmin(loss_als_layer)
                layer_loss_als_max = np.argmax(loss_als_layer)
                model.train()

            for i, (images, labels, _) in enumerate(self.train_loader):
                if np.array(images.data).ndim == 3:
                    images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)
                else:
                    images = images.to(device)
                labels = labels.to(device)

                """Compute losses and decide a layer to apply DA"""
                n_aug = self.n_aug
                if self.flag_layer_rate >= 1:
                    n_ex = np.random.choice(a=self.num_layer * num_aug, p=list(layer_rate.flatten()))
                    # print(n_ex)
                    layer_aug = n_ex // num_aug
                    kind_aug = n_ex % num_aug

                    if self.flag_adalase == 1:
                        n_aug = self.n_aug
                    elif self.flag_adalase == 2:
                        if kind_aug == 0:
                            n_aug = 6  # mixup
                        elif kind_aug == 1:
                            n_aug = 7  # cutout
                    elif self.flag_adalase == 3:
                        if kind_aug == 0:
                            if self.n_aug == 6:  # mixup
                                self.param_aug = 0.1
                            elif self.n_aug == 7:  # cutout
                                self.param_aug = 1.0
                        elif kind_aug == 1:
                            if self.n_aug == 6:  # mixup
                                self.param_aug = 0.5
                            elif self.n_aug == 7:  # cutout
                                self.param_aug = 1.5
                        elif kind_aug == 2:
                            if self.n_aug == 6:  # mixup
                                self.param_aug = 1.0
                            elif self.n_aug == 7:  # cutout
                                self.param_aug = 2.0
                        elif kind_aug == 3:
                            if self.n_aug == 6:  # mixup
                                self.param_aug = 2.0
                            elif self.n_aug == 7:  # cutout
                                self.param_aug = 3.0
                        elif kind_aug == 4:
                            if self.n_aug == 6:  # mixup
                                self.param_aug = 5.0
                            elif self.n_aug == 7:  # cutout
                                self.param_aug = 5.0
                    else:
                        n_aug = self.n_aug
                else:
                    if self.flag_random_layer == 1:
                        layer_aug = np.random.randint(self.num_layer)
                    else:
                        layer_aug = self.layer_aug

                if self.flag_compute_all_losses == 1:
                    if layer_aug == layer_loss_als_min:
                        count_best_layer += 1
                    if layer_aug == layer_loss_als_max:
                        count_worst_layer += 1
                    count_layer_max[layer_loss_als_max] += 1
                    count_all_iter += 1

                """Main training"""
                if self.n_model == "ViT":
                    images, labels = util.run_n_aug(images, labels, self.n_aug, self.num_classes, 0)
                    outputs = model(images)
                    labels_train_layer = labels
                else:
                    outputs, labels = model(x=images, y=labels, num_layer=self.num_layer, n_aug=n_aug, layer_aug=layer_aug, param_aug=self.param_aug)

                if labels.ndim == 1:
                    labels = torch.eye(self.num_classes, device='cuda')[labels].clone()  # To one-hot

                loss_training = util.dist_loss(outputs, labels)
                loss_training_all += loss_training.item() * outputs.shape[0]  # Sum of losses within this minibatch

                """Update weights"""
                optimizer.zero_grad()
                loss_training.backward()  # compute gradients
                optimizer.step()

                """Compute layer rate"""
                if self.flag_adalase >= 1:
                    grad_loss_training = None
                    jmax = len(optimizer.param_groups[0]['params'])
                    for j in range(jmax):
                        if optimizer.param_groups[0]['params'][j].grad != None:
                            if grad_loss_training != None:
                                grad_loss_layer = torch.flatten(optimizer.param_groups[0]['params'][j].grad)
                                grad_loss_training = torch.cat((grad_loss_training, grad_loss_layer), 0)
                            else:
                                grad_loss_training = torch.flatten(optimizer.param_groups[0]['params'][j].grad)

                    disable_running_stats(model)

                    # print(grad_loss_training.shape)

                    outputs_als, labels_als = model(x=images_als_origin, y=labels_als_origin, num_layer=self.num_layer, n_aug=0, layer_aug=0)

                    if labels_als.ndim == 1:
                        labels_als = torch.eye(self.num_classes, device='cuda')[labels_als].clone()  # To one-hot

                    loss_training_before = util.dist_loss(outputs_als, labels_als)
                    loss_als_all += loss_training_before.item() * outputs_als.shape[0]

                    optimizer.zero_grad()
                    loss_training_before.backward()  # compute gradients

                    grad_loss_training_aug = None
                    jmax = len(optimizer.param_groups[0]['params'])
                    for j in range(jmax):
                        if optimizer.param_groups[0]['params'][j].grad != None:
                            if grad_loss_training_aug != None:
                                grad_loss_layer = torch.flatten(optimizer.param_groups[0]['params'][j].grad)
                                grad_loss_training_aug = torch.cat((grad_loss_training_aug, grad_loss_layer), 0)
                            else:
                                grad_loss_training_aug = torch.flatten(optimizer.param_groups[0]['params'][j].grad)

                    enable_running_stats(model)

                    grad_loss = torch.dot(torch.t(grad_loss_training_aug), grad_loss_training).cpu()
                    # print(grad_loss)

                    if self.flag_adalase == 1:
                        if self.flag_rmsprop == 1:  # RMSProp
                            h0_rms = 0.9 * h0_rms + 0.1 * (grad_loss ** 2)
                            layer_rate_delta[layer_aug] += self.initial_als_rate * (1 / np.sqrt(h0_rms + 1e-8)) * grad_loss
                        else:  # gradient descent
                            layer_rate_delta[layer_aug] += self.initial_als_rate * grad_loss

                        if (self.iter + 1) % self.iter_interval == 0:
                            layer_rate += layer_rate_delta / self.iter_interval

                            for j in range(self.num_layer):
                                if layer_rate[j] >= 1 - self.min_rate_sum / self.num_layer:
                                    layer_rate[j] = 1 - self.min_rate_sum / self.num_layer
                                elif layer_rate[j] <= self.min_rate_sum / self.num_layer:
                                    layer_rate[j] = self.min_rate_sum / self.num_layer

                            layer_rate = layer_rate / np.sum(layer_rate)
                            layer_rate_delta = np.zeros(self.num_layer)
                        # print(layer_rate[2])

                    elif self.flag_adalase >= 2:
                        if self.flag_rmsprop == 1:  # RMSProp
                            h0_rms = 0.9 * h0_rms + 0.1 * (grad_loss ** 2)
                            layer_rate_delta[layer_aug][kind_aug] += self.initial_als_rate * (1 / np.sqrt(h0_rms + 1e-8)) * grad_loss
                        else:  # gradient descent
                            layer_rate_delta[layer_aug][kind_aug] += self.initial_als_rate * grad_loss

                        if (self.iter + 1) % self.iter_interval == 0:
                            layer_rate += layer_rate_delta / self.iter_interval

                            for j in range(self.num_layer):
                                for k in range(num_aug):
                                    if layer_rate[j][k] >= 1 - self.min_rate_sum / self.num_layer:
                                        layer_rate[j][k] = 1 - self.min_rate_sum / self.num_layer
                                    elif layer_rate[j][k] <= self.min_rate_sum / self.num_layer:
                                        layer_rate[j][k] = self.min_rate_sum / self.num_layer

                            layer_rate = layer_rate / np.sum(layer_rate)
                            layer_rate_delta = np.zeros((self.num_layer, num_aug))

                self.iter += 1
                step += 1
                if scheduler is not None:
                    scheduler.step(epoch + float(step) / total_steps)

            if self.flag_multi == 1:
                tensor = torch.tensor(loss_training_all, device=device)
                dist.reduce(tensor, dst=0)
                loss_training_all = float(tensor)

            loss_training_each = loss_training_all / len(self.train_loader.dataset)
            if self.flag_adalase >= 1:
                loss_als_each = loss_als_all / self.batch_size_test
            else:
                loss_als_each = loss_training_each
            learning_rate = optimizer.param_groups[0]['lr']

            """Test"""
            model.eval()
            with torch.no_grad():
                correct = 0
                if self.flag_acc5 == 1:
                    correct_top5 = 0

                for i, (images, labels, _) in enumerate(self.test_loader):
                    if np.array(images.data).ndim == 3:
                        images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)
                    else:
                        images = images.to(device)
                    labels = labels.to(device)

                    if self.n_model == "ViT":
                        outputs = model(images)
                    else:
                        outputs, _ = model.forward(x=images, y=labels, num_layer=self.num_layer, n_aug=0)

                    logits = F.log_softmax(outputs, dim=1)
                    acc = (logits.max(1)[1] == labels).sum()
                    correct += acc.item()

                    if self.flag_acc5 == 1:
                        _, acc5 = util.correct_top5(outputs.data, labels.long(), topk=(1, 5))
                        correct_top5 += acc5[0].item()

                    if labels.ndim == 1:
                        labels = torch.eye(self.num_classes, device='cuda')[labels].clone()  # To one-hot

                    loss_test = util.dist_loss(outputs, labels)
                    loss_test_all += loss_test.item() * outputs.shape[0]

                if self.flag_multi == 1:
                    tensor = torch.tensor(loss_test_all, device=device)
                    dist.reduce(tensor, dst=0)
                    loss_test_all = float(tensor)

                    tensor = torch.tensor(correct, device=device)
                    dist.reduce(tensor, dst=0)
                    correct = float(tensor)

                    if self.flag_acc5 == 1:
                        tensor = torch.tensor(correct_top5, device=device)
                        dist.reduce(tensor, dst=0)
                        correct_top5 = float(tensor)

                """Compute test results"""
                loss_test_each = loss_test_all / len(self.test_loader.dataset)

                top1_avg = 100.0 * correct / len(self.test_loader.dataset)
                if self.flag_acc5 == 1:
                    top5_avg = 100.0 * correct_top5 / len(self.test_loader.dataset)

                """Run time"""
                end_epoch_time = timeit.default_timer()
                epoch_time = end_epoch_time - start_epoch_time

                """Show results for each epoch"""
                if flag_rank0 == 1:
                    if self.flag_acc5 == 1:
                        print('Epoch [{}/{}], Training Loss: {:.4f}, Test Acc: {:.3f} %, Test Acc5: {:.3f} %, Test Loss: {:.4f}, Epoch Time: {:.2f}s'.
                              format(epoch + 1, self.num_epochs, loss_training_each, top1_avg, top5_avg, loss_test_each, epoch_time))
                    else:
                        print('Epoch [{}/{}], Training Loss: {:.4f}, Test Acc: {:.3f} %, Test Loss: {:.4f}, Epoch Time: {:.2f}s'.
                              format(epoch + 1, self.num_epochs, loss_training_each, top1_avg, loss_test_each, epoch_time))

                    if self.flag_wandb == 1:
                        if self.flag_compute_all_losses == 1:
                            wandb.log({"epoch": epoch,
                                       "loss_training": loss_training_each,
                                       "loss_als": loss_als_each,
                                       "learning_rate": learning_rate,
                                       "test_acc": top1_avg,
                                       "loss_test": loss_test_each,
                                       "loss_als_layer_p0": loss_als_layer[0],
                                       "loss_als_layer_p1": loss_als_layer[1]
                                       })
                        elif self.flag_acc5 == 1:
                            wandb.log({"epoch": epoch,
                                       "loss_training": loss_training_each,
                                       "loss_als": loss_als_each,
                                       "learning_rate": learning_rate,
                                       "test_acc": top1_avg,
                                       "test_acc5": top5_avg,
                                       "loss_test": loss_test_each
                                       })
                        else:
                            wandb.log({"epoch": epoch,
                                       "loss_training": loss_training_each,
                                       "loss_als": loss_als_each,
                                       "learning_rate": learning_rate,
                                       "test_acc": top1_avg,
                                       "loss_test": loss_test_each
                                       })
                        if self.flag_adalase == 1:
                            for j in range(self.num_layer):
                                wandb.log({"epoch_ratio": epoch,
                                           "ratio_p%s" % j: layer_rate[j]
                                           })
                        elif self.flag_adalase >= 2:
                            for j in range(self.num_layer):
                                for k in range(num_aug):
                                    wandb.log({"epoch_ratio": epoch,
                                               "rate_p%s_d%s" % (j, k): layer_rate[j][k]
                                               })

                    results[epoch][0] = loss_training_each
                    results[epoch][1] = top1_avg
                    results[epoch][2] = loss_test_each
                    results[epoch][3] = epoch_time
                    if self.flag_acc5 == 1:
                        results[epoch][4] = top5_avg

        top1_avg_max = np.max(results[:, 1])
        top1_avg_max_index = np.argmax(results[:, 1])
        loss_training_bestacc = results[top1_avg_max_index, 0]
        loss_test_bestacc = results[top1_avg_max_index, 2]
        if self.flag_acc5 == 1:
            top5_avg_max = np.max(results[:, 4])

        end_time = timeit.default_timer()

        """Save results"""
        if flag_rank0 == 1:
            print(' ran for %.4fm' % ((end_time - start_time) / 60.))

            if self.flag_save_model == 1:
                # print(model.module.state_dict().keys())
                torch.save(model.module.state_dict(), "../../../../../groups/gac50437/model_weights/weight_%s_%s_naug%s_seed%s.pth" % (self.n_model, self.n_data, self.n_aug, self.seed))

            if self.flag_wandb == 1:
                wandb.run.summary["best_accuracy"] = top1_avg_max
                wandb.run.summary["loss_training_bestacc"] = loss_training_bestacc
                wandb.run.summary["loss_test_bestacc"] = loss_test_bestacc
                wandb.run.summary["count_bestlayer"] = count_best_layer
                wandb.run.summary["count_worstlayer"] = count_worst_layer
                wandb.run.summary["count_all_iter"] = count_all_iter
                if self.flag_compute_all_losses == 1:
                    wandb.run.summary["count_layer_max_0"] = count_layer_max[0]
                    wandb.run.summary["count_layer_max_1"] = count_layer_max[1]
                if self.flag_acc5 == 1:
                    wandb.run.summary["best_accuracy_top5"] = top5_avg_max

                wandb.finish()
            sys.exit()

        time.sleep(100)
