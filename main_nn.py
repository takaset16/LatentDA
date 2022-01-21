# coding: utf-8
import timeit
import wandb
from warmup_scheduler import GradualWarmupScheduler
import objective
import dataset
from models import *
import matplotlib.pyplot as plt


class MainNN(object):
    def __init__(self, loop, n_data, gpu_multi, hidden_size, num_samples, num_epochs, batch_size_training, batch_size_test, batch_size_als,
                 n_model, opt, save_file, save_images, flag_acc5, flag_horovod, cutout, n_aug, flag_randaug,
                 rand_n, rand_m, flag_lr_schedule, flag_warmup, layer_aug, flag_random_layer, flag_wandb,
                 flag_traintest, flag_als, initial_als_rate, iter_interval, flag_adversarial, flag_alstest, flag_als_acc,
                 temp, flag_defaug, flag_sign, flag_rate_fix):
        """"""
        """基本要素"""
        self.seed = 1001 + loop
        self.train_loader = None
        self.test_loader = None
        self.n_data = n_data  # dataset
        self.gpu_multi = gpu_multi  # Multi-GPU flag
        self.input_size = 0  # only MLP
        self.hidden_size = hidden_size  # only MLP
        self.num_classes = 10
        self.num_channel = 0
        self.size_after_cnn = 0
        self.num_training_data = num_samples
        self.num_test_data = 0
        self.num_epochs = num_epochs
        self.batch_size_training = batch_size_training
        self.batch_size_test = batch_size_test
        self.batch_size_als = batch_size_als
        self.n_model = n_model
        self.loss_training_batch = None  # minibatch loss
        self.opt = opt  # optimizer
        self.save_file = save_file  # save results
        self.save_images = save_images
        self.flag_wandb = flag_wandb  # weights and biases
        self.flag_acc5 = flag_acc5
        self.flag_horovod = flag_horovod
        self.cutout = cutout
        self.flag_traintest = flag_traintest
        self.initial_lr = 0.01

        """訓練改善手法"""
        self.n_aug = n_aug  # data augmentation
        self.flag_randaug = flag_randaug  # RandAugment
        self.rand_n = rand_n  # RandAugment parameter
        self.rand_m = rand_m  # RandAugment parameter
        self.flag_lr_schedule = flag_lr_schedule  # learning rate schedule
        self.flag_warmup = flag_warmup  # Warmup
        self.flag_defaug = flag_defaug  # Default augmentation

        """LatentDA関連"""
        self.layer_aug = layer_aug
        self.flag_random_layer = flag_random_layer
        self.iter = 0
        self.als_loader = 0
        self.flag_als = flag_als
        self.num_layer = 0
        self.initial_als_rate = initial_als_rate
        self.iter_interval = iter_interval
        self.flag_adversarial = flag_adversarial
        self.flag_alstest = flag_alstest
        self.flag_als_accuracy = flag_als_acc
        self.temp = temp
        self.flag_sign = flag_sign
        self.flag_rate_fix = flag_rate_fix
        self.num_aug = 1

    def run_main(self):
        if self.flag_horovod == 1:
            import horovod.torch as hvd

        # print(os.cpu_count())

        """Settings for random number"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.flag_horovod == 1:
            hvd.init()
            torch.cuda.set_device(hvd.local_rank())

        """RandAugment"""
        if self.flag_randaug == 1:
            if self.rand_n == 0 and self.rand_m == 0:
                if self.n_model == 'ResNet':
                    self.rand_n = 2
                    self.rand_m = 9
                elif self.n_model == 'WideResNet':
                    if self.n_data == 'CIFAR-10':
                        self.rand_n = 3
                        self.rand_m = 5
                    elif self.n_data == 'CIFAR-100':
                        self.rand_n = 2
                        self.rand_m = 14
                    elif self.n_data == 'SVHN':
                        self.rand_n = 3
                        self.rand_m = 7
                elif self.n_model == 'ShakeResNet':
                    self.rand_n = 3
                    self.rand_m = 9

        """dataset"""
        als_dataset = None
        traintest_dataset = dataset.MyDataset_training(n_data=self.n_data, num_data=self.num_training_data, seed=self.seed,
                                                       flag_randaug=self.flag_randaug, rand_n=self.rand_n, rand_m=self.rand_m, cutout=self.cutout, flag_defaug=self.flag_defaug)

        if self.flag_als > 0:
            als_dataset = dataset.MyDataset_als(n_data=self.n_data, num_data=self.num_training_data, seed=self.seed,
                                                flag_randaug=self.flag_randaug, rand_n=self.rand_n, rand_m=self.rand_m, cutout=self.cutout, flag_defaug=self.flag_defaug)
        self.num_channel, self.num_classes, self.size_after_cnn, self.input_size, hidden_size = traintest_dataset.get_info(n_data=self.n_data)
        if self.hidden_size == 0:
            self.hidden_size = hidden_size

        train_sampler = None
        test_sampler = None
        als_sampler = None

        if self.flag_traintest == 1:
            n_samples = len(traintest_dataset)
            # train_size = self.num_classes * 100
            train_size = int(n_samples * 0.65)
            test_size = n_samples - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(traintest_dataset, [train_size, test_size])
        else:
            train_dataset = traintest_dataset
            test_dataset = dataset.MyDataset_test(n_data=self.n_data)

        num_workers = 8
        pin = False
        train_shuffle = True
        test_shuffle = False

        if self.flag_horovod == 1:
            # num_workers = 64  # when using 8 nodes
            num_workers = 128  # when using 16 nodes
            self.gpu_multi = 0

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            if self.flag_als > 0:
                als_sampler = torch.utils.data.distributed.DistributedSampler(als_dataset, num_replicas=hvd.size(), rank=hvd.rank())

            train_shuffle = False

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size_training, sampler=train_sampler,
                                                        shuffle=train_shuffle, num_workers=num_workers, pin_memory=pin)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, sampler=test_sampler,
                                                       shuffle=test_shuffle, num_workers=num_workers, pin_memory=pin)
        if self.flag_als > 0:
            self.als_loader = torch.utils.data.DataLoader(dataset=als_dataset, batch_size=self.batch_size_als, sampler=als_sampler,
                                                          shuffle=train_shuffle, num_workers=num_workers, pin_memory=pin)

        """neural network model"""
        model = None
        if self.n_model == 'MLP':
            model = mlp.MLPNet(input_size=self.input_size, hidden_size=self.hidden_size, num_classes=self.num_classes)
        elif self.n_model == 'CNN':
            if self.n_data == 'STL-10':
                model = cnn_stl.ConvNet(num_classes=self.num_classes, num_channel=self.num_channel, size_after_cnn=self.size_after_cnn)
            else:
                model = cnn.ConvNet(num_classes=self.num_classes, num_channel=self.num_channel, size_after_cnn=self.size_after_cnn)
        elif self.n_model == 'ResNet18':
            # model = resnet_cifar.ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=self.num_classes, num_channel=self.num_channel)
            model = resnet.ResNet(depth=18, num_classes=self.num_classes, num_channel=self.num_channel, n_data=self.n_data)
        elif self.n_model == 'ResNet50':
            # model = resnet_cifar.ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=self.num_classes, num_channel=self.num_channel)
            model = resnet.ResNet(depth=50, num_classes=self.num_classes, num_channel=self.num_channel, n_data=self.n_data)
        elif self.n_model == 'WideResNet40_2':
            model = wideresnet.WideResNet(depth=40, widen_factor=2, dropout_rate=0.0, num_classes=self.num_classes, num_channel=self.num_channel,
                                          n_data=self.n_data)
        elif self.n_model == 'WideResNet28_10':
            model = wideresnet.WideResNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=self.num_classes, num_channel=self.num_channel)
        elif self.n_model == 'ShakeResNet':
            # model = shake_resnet.ShakeResNet(depth=26, w_base=32, label=self.num_classes, num_channel=self.num_channel, n_data=self.n_data,
            #                                  n_layer=self.n_layer)  # shakeshake26_2x32d
            # model = shake_resnet.ShakeResNet(depth=26, w_base=64, label=self.num_classes, num_channel=self.num_channel, n_data=self.n_data,
            #                                  n_layer=self.n_layer)  # shakeshake26_2x64d
            model = shake_resnet.ShakeResNet(depth=26, w_base=96, label=self.num_classes, num_channel=self.num_channel, n_data=self.n_data,
                                             n_layer=self.n_layer)  # shakeshake26_2x96d
            # model = shake_resnet.ShakeResNet(depth=26, w_base=112, label=self.num_classes, num_channel=self.num_channel, n_data=self.n_data,
            #                                  n_layer=self.n_layer)  # shakeshake26_2x112d
        elif self.n_model == 'PyramidNet':
            model = PyramidNet(n_data=self.n_data, depth=272, alpha=200, num_classes=self.num_classes, num_channel=self.num_channel,
                               n_layer=self.n_layer, bottleneck=True)
        elif self.n_model == 'EfficientNet':
            model = EfficientNet.from_name('efficientnet-b3', num_classes=self.num_classes, alpha=self.alpha, alpha2=self.alpha2)

        """Show number of paramters"""
        params = 0
        for p in model.parameters():
            if p.requires_grad:
                params += p.numel()
        print(params)

        """GPU setting"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        if device == 'cuda':
            if self.gpu_multi == 1:
                model = torch.nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True
            if self.flag_horovod == 1:
                print('GPU+=1')
            else:
                print('GPU={}'.format(torch.cuda.device_count()))

        """loss function"""
        criterion = objective.SoftCrossEntropy()

        """optimizer"""
        optimizer = 0
        if self.flag_horovod == 1:
            if self.opt == 0:  # Adam
                if self.flag_transfer == 1:  # Transfer learning
                    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001 * hvd.size())
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * hvd.size())
            elif self.opt == 1:  # SGD
                if self.n_model == 'ResNet18' or self.n_model == 'ResNet50' or self.n_model == 'EfficientNet':
                    lr = 0.01
                    weight_decay = 0.0001
                elif self.n_model == 'WideResNet':
                    if self.n_data == 'SVHN':
                        lr = 0.005
                        weight_decay = 0.001
                    else:
                        lr = 0.1
                        weight_decay = 0.0005
                elif self.n_model == 'ShakeResNet':
                    lr = 0.1
                    weight_decay = 0.001
                elif self.n_model == 'PyramidNet':
                    lr = 0.05
                    weight_decay = 0.00005
                elif self.n_model == 'MLP':
                    lr = 0.02
                    weight_decay = 0.0005
                else:
                    lr = 0.01
                    weight_decay = 0.0005

                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=lr,
                    # lr=lr * hvd.size(),
                    momentum=0.9,
                    weight_decay=weight_decay,
                    nesterov=True
                )
                self.initial_lr = lr
                # self.initial_lr = lr * hvd.size()

            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        else:
            if self.opt == 0:  # Adam
                if self.flag_transfer == 1:  # Transfer learning
                    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            elif self.opt == 1:  # SGD
                if self.n_model == 'ResNet18' or self.n_model == 'ResNet50' or self.n_model == 'EfficientNet':
                    lr = 0.1
                    weight_decay = 0.0001
                elif self.n_model == 'WideResNet':
                    if self.n_data == 'SVHN':
                        lr = 0.005
                        weight_decay = 0.001
                    else:
                        lr = 0.1
                        weight_decay = 0.0005
                elif self.n_model == 'ShakeResNet':
                    lr = 0.1
                    weight_decay = 0.001
                elif self.n_model == 'PyramidNet':
                    lr = 0.05
                    weight_decay = 0.00005
                elif self.n_model == 'MLP':
                    lr = 0.02
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
                self.initial_lr = lr

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
            if self.n_model == 'ResNet50' or self.n_model == 'EfficientNet':
                if self.n_data == 'ImageNet':
                    multiplier = 8  # batch size: 2048
                else:
                    multiplier = 2  # batch size: 512
                total_epoch = 3
            elif self.n_model == 'WideResNet':
                multiplier = 2
                if self.n_data == 'SVHN':
                    total_epoch = 3
                else:
                    total_epoch = 5
            elif self.n_model == 'ShakeResNet':
                multiplier = 4
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

        """Number of layers(positions)"""
        if self.n_model == 'CNN':
            if self.n_data == 'STL-10':
                self.num_layer = 7
            else:
                self.num_layer = 5
        elif self.n_model == 'ResNet18' or self.n_model == 'ResNet50':
            if self.n_data == 'CIFAR-10' or self.n_data == 'CIFAR-100' or self.n_data == 'SVHN':
                # self.num_layer = 6
                self.num_layer = 5
            elif self.n_data == 'ImageNet' or self.n_data == 'TinyImageNet':
                self.num_layer = 6
        elif self.n_model == 'WideResNet':
            self.num_layer = 6
        elif self.n_model == 'MLP':
            self.num_layer = 3

        """Initialization"""
        if self.flag_acc5 == 1:
            results = np.zeros((self.num_epochs, 5))
        else:
            results = np.zeros((self.num_epochs, 4))
        start_time = timeit.default_timer()

        if self.flag_als == 3:
            self.num_aug = 10
        elif self.flag_als == 4:
            if self.n_aug == 7:  # cutout
                self.num_aug = 9
        elif self.flag_als == 5:
            self.num_aug = 2
        else:
            self.num_aug = 1

        layer_rate = np.zeros((self.num_layer, self.num_aug))
        layer_rate_delta = np.zeros((self.num_layer, self.num_aug))

        func_sign_iter = np.zeros(self.num_layer)
        count_interval = 0
        count_aug_layer = np.zeros(self.num_layer, dtype=int)
        grad_loss_training_aug = None
        grad_loss = 0
        values_als_before = None
        layer_aug = 0
        kind_aug = 0
        n_aug = self.n_aug
        n_parameter = 0
        images_als_origin = None
        labels_als_origin = None
        step = 0
        h0_rms = 0

        if self.flag_als > 0:
            values_als_before = 0

        """Decide an initial als_rate"""
        if self.flag_rate_fix == 1:
            layer_rate = np.loadtxt('results/rate_final/data_%s_model_%s_aug_%s_als_%s_initial_%s_interval_%s_adv_%s_seed_%s.csv'
                         % (self.n_data, self.n_model, self.n_aug, self.flag_als, self.initial_als_rate, self.iter_interval,
                          self.flag_adversarial, self.seed), delimiter=',')
        else:
            sum = 0

            for i in range(self.num_layer):
                for j in range(self.num_aug):
                    if i < self.num_layer - 1 or j < self.num_aug - 1:
                        layer_rate[i][j] = 1.0 / (self.num_layer * self.num_aug)
                        sum += layer_rate[i][j]

            layer_rate[self.num_layer - 1][self.num_aug - 1] = 1.0 - sum

        for epoch in range(self.num_epochs):
            """Training"""
            start_epoch_time = timeit.default_timer()
            train_loss = None
            als_loss = None
            if self.flag_horovod == 1:
                train_loss = Metric('train_loss', hvd)
                als_loss = Metric('als_loss', hvd)

            loss_training_all = 0
            loss_test_all = 0

            total_steps = len(self.train_loader)
            num_training_data = 0

            if self.flag_als >= 1:
                for i, (images, labels, _) in enumerate(self.als_loader):
                    if i == 0:
                        if np.array(images.data).ndim == 3:
                            images_als_origin = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)
                        else:
                            images_als_origin = images.to(device)
                        labels_als_origin = labels.to(device)

                        break

            for i, (images, labels, _) in enumerate(self.train_loader):
                if np.array(images.data).ndim == 3:
                    images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)
                else:
                    images = images.to(device)
                labels = labels.to(device)

                """Compute losses and decide a layer to apply DA"""
                if self.flag_als > 0:
                    if self.flag_rate_fix == 0:
                        model.eval()
                        if self.flag_als == 1:  # ALS
                            outputs_als, labels_als = model(x=images_als_origin, y=labels_als_origin, n_aug=0, layer_aug=0)

                            if self.flag_als_accuracy == 1:  # if accuracy is used instead of loss for ALS
                                _, predicted = torch.max(outputs_als.data, 1)
                                correct_als = (predicted == labels_als.long()).sum().item()
                                total_als = labels_als.size(0)
                                values_als_before = 100.0 * correct_als / total_als
                            else:
                                if labels_als.ndim == 1:
                                    labels_als = torch.eye(self.num_classes, device='cuda')[labels_als].clone()  # To one-hot

                                values_als_before = criterion.forward(outputs_als, labels_als).item()

                        elif self.flag_als >= 2:  # gradient descent
                            outputs_als, labels_als = model(x=images_als_origin, y=labels_als_origin, n_aug=0, layer_aug=0)

                            if labels_als.ndim == 1:
                                labels_als = torch.eye(self.num_classes, device='cuda')[labels_als].clone()  # To one-hot

                            loss_training_before = criterion.forward(outputs_als, labels_als)

                            optimizer.zero_grad()
                            loss_training_before.backward()  # compute gradients
                            if self.flag_horovod == 1:
                                als_loss.update(loss_training_before.cpu())  # update gradients

                            grad_loss_training_aug = None
                            jmax = len(optimizer.param_groups[0]['params'])
                            for j in range(jmax):
                                if optimizer.param_groups[0]['params'][j].grad != None:
                                    if grad_loss_training_aug != None:
                                        grad_loss_layer = torch.flatten(optimizer.param_groups[0]['params'][j].grad)
                                        grad_loss_training_aug = torch.cat((grad_loss_training_aug, grad_loss_layer), 0)
                                    else:
                                        grad_loss_training_aug = torch.flatten(optimizer.param_groups[0]['params'][j].grad)
                                    # print(grad_loss_training_aug)
                                    # print(torch.numel(grad_loss_training_aug))

                            if self.flag_horovod == 1:
                                optimizer.synchronize()  # discard gradients

                    if self.flag_als == 1:
                        layer_aug = np.random.choice(a=self.num_layer, p=list(layer_rate))
                    elif self.flag_als >= 2:
                        n_ex = np.random.choice(a=self.num_layer * self.num_aug, p=list(layer_rate.flatten()))
                        layer_aug = n_ex // self.num_aug
                        kind_aug = n_ex % self.num_aug
                        if self.flag_als == 3 or self.flag_als == 5:
                            n_aug = kind_aug + 1
                        elif self.flag_als == 4:
                            n_parameter = kind_aug + 1
                else:
                    if self.flag_random_layer == 1:
                        layer_aug = np.random.randint(self.num_layer)
                    else:
                        layer_aug = self.layer_aug

                """Main training"""
                model.train()

                flag_save_images = 0
                if self.save_images == 1:
                    if epoch == 0 and i == 0:
                        flag_save_images = 1
                        # util.save_images(images, 0)  # input images

                outputs, labels = model(x=images, y=labels, n_aug=n_aug, layer_aug=layer_aug, flag_save_images=flag_save_images, n_parameter=n_parameter)

                if labels.ndim == 1:
                    labels = torch.eye(self.num_classes, device='cuda')[labels].clone()  # To one-hot

                loss_training = criterion.forward(outputs, labels)  # default

                loss_training_all += loss_training.item() * outputs.shape[0]  # Sum of losses within this minibatch
                num_training_data += images.shape[0]

                """Update weights"""
                optimizer.zero_grad()
                loss_training.backward()  # compute gradients
                if self.flag_horovod == 1:
                    train_loss.update(loss_training.cpu())  # update gradients

                    with optimizer.skip_synchronize():
                        optimizer.step()  # update weights
                else:
                    optimizer.step()  # update weights

                learning_rate = optimizer.param_groups[0]['lr']
                # als_rate = self.initial_als_rate * (learning_rate / self.initial_lr)

                """Compute layer rate"""
                if self.flag_als > 0 and self.flag_rate_fix == 0:
                    if self.flag_als == 1:
                        model.eval()

                        count_aug_layer[layer_aug] += 1
                        count_aug_layer_visual[layer_aug] += 1

                        outputs_als, labels_als = model(x=images_als_origin, y=labels_als_origin, n_aug=0, layer_aug=0)

                        if self.flag_als_accuracy == 1:
                            _, predicted = torch.max(outputs_als.data, 1)
                            correct_als = (predicted == labels_als_origin.long()).sum().item()
                            total_als = labels_als_origin.size(0)
                            values_als_after = 100.0 * correct_als / total_als
                        else:
                            if labels_als.ndim == 1:
                                labels_als = torch.eye(self.num_classes, device='cuda')[labels_als].clone()  # To one-hot

                            values_als_after = criterion.forward(outputs_als, labels_als).item()

                        delta_loss = values_als_before - values_als_after
                        if i == 0:
                            layer_rate = func_sign_iter / np.sum(func_sign_iter)
                            func_sign_iter = np.zeros(self.num_layer)

                        if self.flag_sign == 1:
                            if delta_loss > 0:
                                func_sign = 1
                            elif delta_loss == 0:
                                func_sign = 0
                            else:
                                func_sign = -1

                            func_sign_iter[layer_aug] += func_sign
                        else:
                            func_sign_iter[layer_aug] += delta_loss

                        if (self.iter + 1) % self.iter_interval == 0:
                            if self.flag_adversarial == 1:
                                for j in range(self.num_layer):
                                    if count_aug_layer[j] > 0:
                                        layer_rate[j] -= als_rate  * func_sign_iter[j] / count_aug_layer[j]
                            else:
                                for j in range(self.num_layer):
                                    if count_aug_layer[j] > 0:
                                        layer_rate[j] += als_rate * func_sign_iter[j] / count_aug_layer[j]

                            func_sign_iter = np.zeros(self.num_layer)

                    elif self.flag_als >= 2:  # gradient descent
                        grad_loss_training = None
                        jmax = len(optimizer.param_groups[0]['params'])
                        for j in range(jmax):
                            if optimizer.param_groups[0]['params'][j].grad != None:
                                if grad_loss_training != None:
                                    grad_loss_layer = torch.flatten(optimizer.param_groups[0]['params'][j].grad)
                                    grad_loss_training = torch.cat((grad_loss_training, grad_loss_layer), 0)
                                else:
                                    grad_loss_training = torch.flatten(optimizer.param_groups[0]['params'][j].grad)

                        # print(torch.numel(grad_loss_training))

                        if self.flag_sign == 1:
                            sign_loss = torch.dot(torch.t(grad_loss_training_aug), grad_loss_training)
                            func_sign = 0

                            if self.flag_sign == 1:
                                if sign_loss > 0:
                                    func_sign = 1
                                elif sign_loss == 0:
                                    func_sign = 0
                                else:
                                    func_sign = -1

                            grad_loss = func_sign
                        else:
                            grad_loss = torch.dot(torch.t(grad_loss_training_aug), grad_loss_training)

                        # gradient descent
                        """
                        if self.flag_adversarial == 1:
                            layer_rate[layer_aug][kind_aug] -= als_rate * grad_loss
                        else:
                            layer_rate[layer_aug][kind_aug] += als_rate * grad_loss
                        """

                        # RMSProp
                        h0_rms = 0.9 * h0_rms + 0.1 * (grad_loss.cpu() ** 2)
                        if self.flag_adversarial == 1:
                            layer_rate_delta[layer_aug][kind_aug] -= self.initial_als_rate * (1 / np.sqrt(h0_rms + 1e-8)) * grad_loss.cpu()
                        else:
                            layer_rate_delta[layer_aug][kind_aug] += self.initial_als_rate * (1 / np.sqrt(h0_rms + 1e-8)) * grad_loss.cpu()

                        if (self.iter + 1) % self.iter_interval == 0:
                            layer_rate += layer_rate_delta / self.iter_interval

                        grad_loss = 0

                    if (self.iter + 1) % self.iter_interval == 0:
                        if self.flag_als >= 2:
                            for j in range(self.num_layer):
                                for k in range(self.num_aug):
                                    if layer_rate[j][k] >= 1 - 0.0001:
                                        layer_rate[j][k] = 1 - 0.0001
                                    elif layer_rate[j][k] <= 0.0001:
                                        layer_rate[j][k] = 0.0001
                        else:
                            for j in range(self.num_layer):
                                if layer_rate[j] >= 1 - 0.0001:
                                    layer_rate[j] = 1 - 0.0001
                                elif layer_rate[j] <= 0.0001:
                                    layer_rate[j] = 0.0001

                        layer_rate = layer_rate / np.sum(layer_rate)

                        if self.flag_als == 1:
                            count_aug_layer = np.zeros(self.num_layer, dtype=int)
                        else:
                            layer_rate_delta = np.zeros((self.num_layer, self.num_aug))

                """Update learning rate"""
                self.iter += 1
                step += 1

                if scheduler is not None:
                    scheduler.step(epoch + float(step) / total_steps)

            loss_training_each = loss_training_all / num_training_data  # Loss for each sample
            step = 0

            """Test"""
            model.eval()
            test_loss = None
            test_accuracy_top1 = None
            test_accuracy_top5 = None
            if self.flag_horovod == 1:
                test_loss = Metric('test_loss', hvd)
                test_accuracy_top1 = Metric('Test_accuracy_top1', hvd)
                test_accuracy_top5 = Metric('Test_accuracy_top5', hvd)

            with torch.no_grad():
                if self.flag_acc5 == 1:
                    top1 = list()
                    top5 = list()
                else:
                    correct = 0
                    total = 0

                num_test_data = 0
                for i, (images, labels, _) in enumerate(self.test_loader):
                    if np.array(images.data).ndim == 3:
                        images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)
                    else:
                        images = images.to(device)
                    labels = labels.to(device)

                    if self.save_images == 1 and epoch == self.num_epochs - 1:
                        util.save_images(images)

                    outputs, _ = model.forward(x=images, y=labels, n_aug=0)

                    if self.flag_acc5 == 1:
                        acc1, acc5 = util.accuracy(outputs.data, labels.long(), topk=(1, 5))
                        top1.append(acc1[0].item())
                        top5.append(acc5[0].item())
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels.long()).sum().item()
                        total += labels.size(0)

                    if labels.ndim == 1:
                        labels = torch.eye(self.num_classes, device='cuda')[labels].clone()  # To one-hot

                    loss_test = criterion.forward(outputs, labels)
                    loss_test_all += loss_test.item() * outputs.shape[0]
                    num_test_data += images.shape[0]

                    if self.flag_horovod == 1:
                        test_loss.update(loss_test.cpu())

            """Compute test results"""
            top5_avg = 0

            if self.flag_acc5 == 1:
                top1_avg = sum(top1) / float(len(top1))
                top5_avg = sum(top5) / float(len(top5))

                if self.flag_horovod == 1:
                    # print(top1_avg)  # before allreduce
                    test_accuracy_top1.update(torch.tensor(top1_avg))
                    test_accuracy_top5.update(torch.tensor(top5_avg))

                    top1_avg = test_accuracy_top1.avg.item()
                    top5_avg = test_accuracy_top5.avg.item()
                    # print(top1_avg)  # after allreduce
            else:
                top1_avg = 100.0 * correct / total

            loss_test_each = loss_test_all / num_test_data  # Loss for each sample

            """Run time"""
            end_epoch_time = timeit.default_timer()
            epoch_time = end_epoch_time - start_epoch_time

            """Show results for each epoch"""
            flag_log = 1
            if self.flag_horovod == 1:
                if hvd.rank() != 0:
                    flag_log = 0

            if flag_log == 1:
                if self.flag_acc5 == 1:
                    print('Epoch [{}/{}], Training Loss: {:.4f}, Top1 Acc: {:.3f} %, Top5 Acc: {:.3f} %, Test Loss: {:.4f}, Epoch Time: {:.2f}s'.
                          format(epoch + 1, self.num_epochs, loss_training_each, top1_avg, top5_avg, loss_test_each, epoch_time))
                else:
                    print('Epoch [{}/{}], Training Loss: {:.4f}, Test Acc: {:.3f} %, Test Loss: {:.4f}, Epoch Time: {:.2f}s'.
                          format(epoch + 1, self.num_epochs, loss_training_each, top1_avg, loss_test_each, epoch_time))

                if self.flag_wandb == 1:
                    for j in range(self.num_layer):
                        for k in range(self.num_aug):
                            wandb.log({"epoch": epoch,
                                       "rate_p%s_d%s" % (j, k): layer_rate[j][k],
                                       "loss_training": loss_training_each,
                                       "learning_rate": learning_rate,
                                       "test_acc": top1_avg,
                                       "loss_test": loss_test_each
                                       })

            if self.flag_acc5 == 1:
                results[epoch][0] = loss_training_each
                results[epoch][1] = top1_avg
                results[epoch][2] = top5_avg
                results[epoch][3] = loss_test_each
                results[epoch][4] = epoch_time
            else:
                results[epoch][0] = loss_training_each
                results[epoch][1] = top1_avg
                results[epoch][2] = loss_test_each
                results[epoch][3] = epoch_time

        end_time = timeit.default_timer()

        flag_log = 1
        if self.flag_horovod == 1:
            if hvd.rank() != 0:
                flag_log = 0

        if flag_log == 1:
            print(' ran for %.4fm' % ((end_time - start_time) / 60.))

            top1_avg_max = np.max(results[:, 1])
            print(top1_avg_max)
            if self.flag_acc5 == 1:
                top5_avg_max = np.max(results[:, 2])
                print(top5_avg_max)

        """Save results"""
        if self.save_file == 1:
            if flag_log == 1:
                if self.flag_horovod == 1:
                    np.savetxt('results/data_%s_model_%s_aug_%s_als_%s_randlayer_%s_layer_aug_%s_interval_%s_adv_%s_seed_%s_acc_%s.csv'
                               % (self.n_data, self.n_model, self.n_aug, self.flag_als, self.flag_random_layer, self.layer_aug, self.iter_interval, self.flag_adversarial,
                                  self.seed, top1_avg_max),
                               results, delimiter=',')

                if self.flag_als >= 1 and self.flag_rate_fix == 0:
                    np.savetxt('results/rate_final/data_%s_model_%s_aug_%s_als_%s_initial_%s_interval_%s_adv_%s_seed_%s.csv'
                               % (self.n_data, self.n_model, self.n_aug, self.flag_als, self.initial_als_rate, self.iter_interval,
                                  self.flag_adversarial, self.seed),
                               layer_rate, delimiter=',')


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name, hvd):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        self.hvd = hvd

    def update(self, val):
        self.sum += self.hvd.allreduce(val, name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n
