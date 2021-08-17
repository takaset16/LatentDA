# coding: utf-8
import timeit
import wandb
from warmup_scheduler import GradualWarmupScheduler
import objective
import dataset
from models import *
import matplotlib.pyplot as plt


class MainNN(object):
    def __init__(self, loop, n_data, gpu_multi, hidden_size, num_samples, num_epochs, batch_size_training, batch_size_test,
                 n_model, opt, save_file, show_params, save_images, flag_acc5, flag_horovod, cutout, n_aug,
                 flag_dropout, flag_transfer, flag_randaug, rand_n, rand_m,
                 flag_lars, lb_smooth, flag_lr_schedule, flag_warmup, layer_aug, layer_drop, flag_random_layer, flag_wandb,
                 flag_traintest, flag_var, batch_size_variance, flag_als, als_rate, epoch_random, iter_interval, flag_adversarial):
        """"""
        """基本要素"""
        self.seed = 1001 + loop
        self.train_loader = None
        self.test_loader = None
        self.n_data = n_data  # データセット
        self.gpu_multi = gpu_multi  # Multi-GPU
        self.input_size = 0  # 入力次元
        self.hidden_size = hidden_size  # MLP
        self.num_classes = 10  # クラス数
        self.num_channel = 0  # チャネル数
        self.size_after_cnn = 0  # cnn層の後のサイズ
        self.num_training_data = num_samples  # 訓練データ数(0で訓練データ全部を使用)
        self.num_test_data = 0  # テストデータ数
        self.num_epochs = num_epochs  # エポック数
        self.batch_size_training = batch_size_training  # 学習時のバッチサイズ
        self.batch_size_test = batch_size_test  # テスト時のバッチサイズ（メモリにのる範囲でできるだけ大きく）
        self.n_model = n_model  # ニューラルネットワークモデル
        self.loss_training_batch = None  # バッチ内誤差
        self.opt = opt  # optimizer
        self.save_file = save_file  # 結果をファイル保存
        self.show_params = show_params  # パラメータ表示
        self.save_images = save_images  # 画像保存
        self.flag_wandb = flag_wandb  # weights and biases
        self.flag_acc5 = flag_acc5
        self.flag_horovod = flag_horovod
        self.cutout = cutout
        self.flag_traintest = flag_traintest
        self.aug_layer_count_all = None

        """訓練改善手法"""
        self.n_aug = n_aug  # data augmentation
        self.flag_transfer = flag_transfer  # Transfer learning
        self.flag_shake = 0  # Shake-shake
        self.flag_randaug = flag_randaug  # RandAugment
        self.rand_n = rand_n  # RandAugment parameter
        self.rand_m = rand_m  # RandAugment parameter
        self.flag_lars = flag_lars  # LARS
        self.lb_smooth = lb_smooth  # SmoothCrossEntropyLoss
        self.flag_lr_schedule = flag_lr_schedule  # 学習率スケジュール
        self.flag_warmup = flag_warmup  # Warmup
        self.flag_dropout = flag_dropout  # Dropout

        """LatentDA関連"""
        self.layer_aug = layer_aug
        self.layer_drop = layer_drop
        self.flag_random_layer = flag_random_layer
        self.iter = 0
        self.flag_var = flag_var
        self.var_train_loader = None
        self.var_test_loader = None
        self.batch_size_variance = batch_size_variance
        self.feature_var_class_train = 0
        self.feature_var_class_test = 0
        self.feature_var_train = 0
        self.feature_var_test = 0
        self.als_loader = 0
        self.flag_als = flag_als
        self.num_layer = 0
        self.als_rate = als_rate
        self.layer_rate_all = None
        self.epoch_random = epoch_random
        self.iter_interval = iter_interval
        self.flag_adversarial = flag_adversarial

    def run_main(self):
        if self.flag_horovod == 1:
            import horovod.torch as hvd

        # print(os.cpu_count())

        """乱数を固定"""
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
        traintest_dataset = dataset.MyDataset_training(n_data=self.n_data, num_data=self.num_training_data, seed=self.seed,
                                                       flag_randaug=self.flag_randaug, rand_n=self.rand_n, rand_m=self.rand_m, cutout=self.cutout)
        self.num_channel, self.num_classes, self.size_after_cnn, self.input_size, self.hidden_size = traintest_dataset.get_info(n_data=self.n_data)

        if self.flag_traintest == 1:
            n_samples = len(traintest_dataset)
            # train_size = self.num_classes * 100
            train_size = int(n_samples * 0.65)
            test_size = n_samples - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(traintest_dataset, [train_size, test_size])

            train_sampler = None
            test_sampler = None
        else:
            train_dataset = traintest_dataset
            test_dataset = dataset.MyDataset_test(n_data=self.n_data)

            train_sampler = train_dataset.sampler
            test_sampler = test_dataset.sampler

        num_workers = 16
        train_shuffle = True
        test_shuffle = False

        if self.flag_horovod == 1:
            num_workers = 4
            self.gpu_multi = 0

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())

        if train_sampler:
            train_shuffle = False
        if test_sampler:
            test_shuffle = False
        test_shuffle = False

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size_training, sampler=train_sampler,
                                                        shuffle=train_shuffle, num_workers=num_workers, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, sampler=test_sampler,
                                                       shuffle=test_shuffle, num_workers=num_workers, pin_memory=True)
        self.var_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size_training, sampler=train_sampler,
                                                            shuffle=train_shuffle, num_workers=num_workers, pin_memory=True)
        self.var_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, sampler=train_sampler,
                                                           shuffle=test_shuffle, num_workers=num_workers, pin_memory=True)
        self.als_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, sampler=test_sampler,
                                                      shuffle=True, num_workers=num_workers, pin_memory=True)
        # self.als_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size_test, sampler=train_sampler,
        #                                               shuffle=True, num_workers=num_workers, pin_memory=True)

        """Transfer learning"""
        if self.flag_transfer == 1:
            pretrained = True
            num_classes = 1000
        else:
            pretrained = False
            num_classes = self.num_classes

        """neural network model"""
        model = None
        if self.n_model == 'MLP':
            model = mlp.MLPNet(input_size=self.input_size, hidden_size=self.hidden_size, num_classes=self.num_classes)
        elif self.n_model == 'CNN':
            model = cnn.ConvNet(num_classes=self.num_classes, num_channel=self.num_channel, size_after_cnn=self.size_after_cnn)
        elif self.n_model == 'ResNet':
            # model = resnet.ResNet(depth=18, num_classes=self.num_classes, num_channel=self.num_channel, n_data=self.n_data,
            #                       n_aug=self.n_aug, bottleneck=True)  # resnet18
            model = resnet.ResNet(depth=50, num_classes=self.num_classes, num_channel=self.num_channel, n_data=self.n_data, bottleneck=True)  # resnet50
            # model = resnet.ResNet(depth=200, num_classes=self.num_classes, num_channel=self.num_channel, n_data=self.n_data, bottleneck=True)  # resnet200
        elif self.n_model == 'WideResNet':
            # model = wideresnet.WideResNet(depth=40, widen_factor=2, dropout_rate=0.0, num_classes=self.num_classes, num_channel=self.num_channel,
            #                               n_data=self.n_data)  # wresnet40_2
            model = wideresnet.WideResNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=self.num_classes, num_channel=self.num_channel)  # wresnet28_10
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

        """Transfer learning"""
        if self.flag_transfer == 1:
            for param in model.parameters():
                param.requires_grad = False

            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)

        """パラメータ表示"""
        if self.show_params == 1:
            params = 0
            for p in model.parameters():
                if p.requires_grad:
                    params += p.numel()
            print(params)

        """GPU利用"""
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
        if self.lb_smooth > 0.0:
            criterion = objective.SmoothCrossEntropyLoss(self.lb_smooth)
        else:
            criterion = objective.SoftCrossEntropy()

        """optimizer"""
        optimizer = 0
        if self.flag_horovod == 1:
            if self.opt == 0:  # Adam
                if self.flag_transfer == 1:  # Transfer learning
                    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001 * hvd.size())  # transfer learning
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * hvd.size())
            elif self.opt == 1:  # SGD
                lr = 0
                weight_decay = 0
                if self.n_model == 'ResNet':  # ResNet
                    lr = 0.1
                    weight_decay = 0.0001
                elif self.n_model == 'WideResNet':  # WideResNet
                    if self.n_data == 'SVHN':
                        lr = 0.005
                        weight_decay = 0.001
                    else:
                        lr = 0.1
                        weight_decay = 0.0005
                elif self.n_model == 'ShakeResNet':  # ShakeResNet
                    lr = 0.1
                    weight_decay = 0.001
                elif self.n_model == 'PyramidNet':  # ShakeResNet
                    lr = 0.05
                    weight_decay = 0.00005

                optimizer = torch.optim.SGD(
                    model.parameters(),
                    # lr=lr * hvd.size(),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=weight_decay,
                    nesterov=True
                )
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        else:
            if self.opt == 0:  # Adam
                if self.flag_transfer == 1:  # Transfer learning
                    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            elif self.opt == 1:  # SGD
                if self.n_model == 'ResNet' or self.n_model == 'EfficientNet':
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
        if self.flag_lars == 1:  # LARS
            from torchlars import LARS
            optimizer = LARS(optimizer)

        """学習率スケジューリング"""
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
            if self.n_model == 'ResNet' or self.n_model == 'EfficientNet':
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
            self.num_layer = 5
        elif self.n_model == 'ResNet':
            if self.n_data == 'CIFAR-10' or self.n_data == 'CIFAR-100':
                self.num_layer = 5
            elif self.n_data == 'ImageNet' or self.n_data == 'TinyImageNet':
                self.num_layer = 6
        elif self.n_model == 'WideResNet':
            self.num_layer = 6
        elif self.n_model == 'MLP':
            self.num_layer = 3

        """ALS"""
        layer_rate = np.zeros(self.num_layer)
        func_sign_random = np.zeros(self.num_layer)
        layer_rate_mean = np.zeros(self.num_layer)
        loss_greedy_als = np.zeros(self.num_layer)
        loss_greedy_als_sum = np.zeros(self.num_layer)

        """Decide an initial als_rate"""
        if self.flag_als >= 1:
            sum = 0

            for i in range(self.num_layer - 1):
                layer_rate[i] = 1.0 / self.num_layer
                sum = sum + layer_rate[i]
            layer_rate[self.num_layer - 1] = 1.0 - sum

            """
            layer_rate[0] = 1 - self.als_rate
            for i in range(self.num_layer - 1):
                layer_rate[i + 1] = self.als_rate / (self.num_layer - 1)
            """
            self.layer_rate_all = np.zeros((self.num_epochs, self.num_layer))

            self.flag_random_layer = 1

        """初期化"""
        if self.flag_acc5 == 1:
            results = np.zeros((self.num_epochs, 5))  # 結果保存用
        else:
            results = np.zeros((self.num_epochs, 4))  # 結果保存用
        start_time = timeit.default_timer()

        aug_layer_count = np.zeros(self.num_layer, dtype=int)
        self.aug_layer_count_all = np.zeros((self.num_epochs, self.num_layer), dtype=int)

        images_als_origin = None
        labels_als_origin = None

        for epoch in range(self.num_epochs):
            """学習"""
            start_epoch_time = timeit.default_timer()
            train_loss = None
            if self.flag_horovod == 1:
                train_loss = Metric('train_loss', hvd)

            loss_training_all = 0
            loss_test_all = 0

            """学習率スケジューリング"""
            if self.flag_lr_schedule == 1:
                if self.num_epochs == 200:
                    if epoch == 100:
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

            total_steps = len(self.train_loader)
            steps = 0
            num_training_data = 0

            for i, (images, labels, _) in enumerate(self.als_loader):
                if i == 0:
                    if np.array(images.data).ndim == 3:
                        images_als_origin = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)  # チャネルの次元を加えて4次元にする
                    else:
                        images_als_origin = images.to(device)
                    labels_als_origin = labels.to(device)

                    break

            for i, (images, labels, _) in enumerate(self.train_loader):
                loss_training_before_all = 0
                loss_training_after_all = 0

                """ALS process before weight update"""
                if self.flag_als == 1:
                    model.eval()

                    outputs_als, labels_als = model(x=images_als_origin, y=labels_als_origin, n_aug=0, layer_aug=0, flag_track=0)  # if using test data
                    """
                    if self.n_aug == 1 or self.n_aug == 5:
                        outputs_als, labels_als = model(x=images_als_origin, y=labels_als_origin, n_aug=6, layer_aug=0, flag_track=0)
                    else:
                        outputs_als, labels_als = model(x=images_als_origin, y=labels_als_origin, n_aug=1, layer_aug=0, flag_track=0)
                    """

                    if labels_als.ndim == 1:
                        labels_als = torch.eye(self.num_classes, device='cuda')[labels_als].clone()  # one-hot表現に変換

                    loss_training_before = criterion.forward(outputs_als, labels_als)
                    loss_training_before_all = loss_training_before.item() * outputs_als.shape[0]  # ミニバッチ内の誤差の合計

                steps += 1
                if np.array(images.data).ndim == 3:
                    images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)  # チャネルの次元を加えて4次元にする
                else:
                    images = images.to(device)
                labels = labels.to(device)

                if self.save_images == 1:
                    util.save_images(images)

                """Decide a layer to apply DA"""
                flag_als = 0
                if self.flag_als >= 1:
                    if epoch < self.epoch_random:
                        flag_als = 0
                    else:
                        flag_als = 1

                layer_aug = 0
                if self.flag_als == 3 and flag_als == 1:  # greedy-ALS
                    model.eval()
                    for j in range(self.num_layer):
                        # outputs, labels_new = model(x=images, y=labels, n_aug=self.n_aug, layer_aug=j, flag_track=0)  # if using training data
                        outputs, labels_new = model(x=images_als_origin, y=labels_als_origin, n_aug=self.n_aug, layer_aug=j, flag_track=0)  # if using test data
                        """
                        if self.n_aug == 1 or self.n_aug == 5:
                            outputs, labels_new = model(x=images_als_origin, y=labels_als_origin, n_aug=6, layer_aug=j, flag_track=0)
                        else:
                            outputs, labels_new = model(x=images_als_origin, y=labels_als_origin, n_aug=1, layer_aug=j, flag_track=0)
                        """
                        if labels_new.ndim == 1:
                            labels_new = torch.eye(self.num_classes, device='cuda')[labels_new].clone()  # one hot表現に変換

                        loss_greedy_als[j] = criterion.forward(outputs, labels_new).item()

                    loss_greedy_als_sum = loss_greedy_als_sum + loss_greedy_als
                    if (self.iter + 1) % self.iter_interval == 0:

                        if self.flag_adversarial == 1:
                            layer_aug = np.argmax(loss_greedy_als_sum)  # reverse
                        else:
                            layer_aug = np.argmin(loss_greedy_als_sum)
                        """
                        if epoch < 50:
                            layer_aug = np.argmax(loss_greedy_als_sum)  # reverse
                        else:
                            layer_aug = np.argmin(loss_greedy_als_sum)
                        """
                        loss_greedy_als_sum = np.zeros(self.num_layer)

                elif (self.flag_als == 1 or self.flag_als == 2) and flag_als == 1:  # ALS or naive-ALS:
                    layer_aug = np.random.choice(a=self.num_layer, p=list(layer_rate))
                else:
                    if self.flag_random_layer == 1:
                        layer_aug = np.random.randint(self.num_layer)
                    else:
                        layer_aug = self.layer_aug
                    """
                    if self.flag_dropout == 1:
                        layer_drop = self.layer_drop
                        # layer_drop = np.random.randint(self.num_layer)
                    """

                """Main training"""
                model.train()
                outputs, labels = model(x=images, y=labels, n_aug=self.n_aug, layer_aug=layer_aug, flag_track=1)

                if labels.ndim == 1:
                    labels = torch.eye(self.num_classes, device='cuda')[labels].clone()  # one hot表現に変換

                loss_training = criterion.forward(outputs, labels)  # default

                loss_training_all += loss_training.item() * outputs.shape[0]  # ミニバッチ内の誤差の合計を足していく
                num_training_data += images.shape[0]
                loss_naive_als = loss_training.item()

                aug_layer_count[layer_aug] = aug_layer_count[layer_aug] + 1

                """Update weights"""
                optimizer.zero_grad()
                loss_training.backward()
                optimizer.step()

                if self.flag_horovod == 1:
                    train_loss.update(loss_training.cpu())

                """ALS process after weight update"""
                if self.flag_als == 1:
                    model.eval()

                    outputs_als, labels_als = model(x=images_als_origin, y=labels_als_origin, n_aug=0, layer_aug=0, flag_track=0)  # if using test data
                    # outputs_als, labels_als = model(x=images_als_origin, y=labels_als_origin, n_aug=self.n_aug, layer_aug=0, flag_track=0)  # use the same DA parameters

                    if labels_als.ndim == 1:
                        labels_als = torch.eye(self.num_classes, device='cuda')[labels_als].clone()  # one hot表現に変換

                    loss_training_after = criterion.forward(outputs_als, labels_als)
                    loss_training_after_all += loss_training_after.item() * outputs_als.shape[0]  # ミニバッチ内の誤差の合計を足していく

                    """Ver1.0"""
                    """
                    if i == 0:
                        print(loss_training_before_all)
                        print(loss_training_after_all)
                    
                    layer_rate[aug_layer] = layer_rate[aug_layer] + self.als_rate * (loss_training_before_all - loss_training_after_all)
                    """

                    """Ver1.1"""
                    delta_loss = loss_training_before_all - loss_training_after_all
                    if epoch < self.epoch_random:
                        if delta_loss > 0:
                            func_sign_random[layer_aug] = func_sign_random[layer_aug] + 1
                    elif self.epoch_random > 0 and epoch == self.epoch_random and i == 0:
                        layer_rate = func_sign_random / np.sum(func_sign_random)

                        for j in range(self.num_layer):
                            if layer_rate[j] >= 1 - self.als_rate:
                                layer_rate[j] = 1 - self.als_rate
                            elif layer_rate[j] <= self.als_rate:
                                layer_rate[j] = self.als_rate

                        layer_rate = layer_rate / np.sum(layer_rate)
                    if epoch >= self.epoch_random:
                        if delta_loss > 0:
                            func_sign = 1
                        elif delta_loss == 0:
                            func_sign = 0
                        else:
                            func_sign = -1

                        layer_rate_mean[layer_aug] = layer_rate_mean[layer_aug] + self.als_rate * func_sign

                        if (self.iter + 1) % self.iter_interval == 0:
                            if self.flag_adversarial == 1:
                                layer_rate = layer_rate - layer_rate_mean / self.iter_interval  # reverse
                            else:
                                layer_rate = layer_rate + layer_rate_mean / self.iter_interval

                elif self.flag_als == 2:  # naive ALS
                    layer_rate_mean[layer_aug] = layer_rate_mean[layer_aug] - self.als_rate * loss_naive_als

                    if (self.iter + 1) % self.iter_interval == 0:
                        if self.flag_adversarial == 1:
                            layer_rate = layer_rate - layer_rate_mean / self.iter_interval  # reverse
                        else:
                            layer_rate = layer_rate + layer_rate_mean / self.iter_interval

                if self.flag_als == 1 or self.flag_als == 2:
                    for j in range(self.num_layer):
                        if layer_rate[j] >= 1 - self.als_rate:
                            layer_rate[j] = 1 - self.als_rate
                        elif layer_rate[j] <= self.als_rate:
                            layer_rate[j] = self.als_rate

                    layer_rate = layer_rate / np.sum(layer_rate)
                    layer_rate_mean = np.zeros(self.num_layer)

                self.iter += 1

            loss_training_each = loss_training_all / num_training_data  # サンプル1つあたりの誤差
            learning_rate = optimizer.param_groups[0]['lr']

            if self.flag_als >= 1:
                self.layer_rate_all[epoch] = layer_rate
            self.aug_layer_count_all[epoch] = aug_layer_count

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
                        images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)  # チャネルの次元を加えて4次元にする
                    else:
                        images = images.to(device)
                    labels = labels.to(device)

                    if self.save_images == 1 and epoch == self.num_epochs - 1:
                        util.save_images(images)

                    outputs, _ = model.forward(x=images, y=labels, n_aug=0, flag_track=1)

                    if self.flag_acc5 == 1:
                        acc1, acc5 = util.accuracy(outputs.data, labels.long(), topk=(1, 5))
                        top1.append(acc1[0].item())
                        top5.append(acc5[0].item())
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels.long()).sum().item()
                        total += labels.size(0)

                    if labels.ndim == 1:
                        labels = torch.eye(self.num_classes, device='cuda')[labels].clone()  # one hot表現に変換

                    loss_test = criterion.forward(outputs, labels)
                    loss_test_all += loss_test.item() * outputs.shape[0]  # ミニバッチ内の誤差の合計を足していく
                    num_test_data += images.shape[0]

                    if self.flag_horovod == 1:
                        test_loss.update(loss_test.cpu())

            """テスト結果算出"""
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
                top1_avg = 100.0 * correct / total  # 正解率を計算

            loss_test_each = loss_test_all / num_test_data  # サンプル1つあたりの誤差

            """学習率スケジューリング"""
            if scheduler is not None:
                scheduler.step(epoch - 1 + float(steps) / total_steps)

            """計算時間"""
            end_epoch_time = timeit.default_timer()
            epoch_time = end_epoch_time - start_epoch_time

            """Show results for each epoch"""
            flag_log = 1
            if self.flag_horovod == 1:
                if hvd.rank() != 0:
                    flag_log = 0

            if flag_log == 1:
                if self.flag_var == 1:
                    if self.flag_acc5 == 1:
                        print('Epoch [{}/{}], Training Loss: {:.4f}, Top1 Acc: {:.3f} %, Top5 Acc: {:.3f} %, Test Loss: {:.4f}, Epoch Time: {:.2f}s'.
                              format(epoch + 1, self.num_epochs, loss_training_each, top1_avg, top5_avg, loss_test_each, epoch_time))
                    else:
                        print('Epoch [{}/{}], Training Loss: {:.4f}, Test Acc: {:.3f} %, Test Loss: {:.4f}, Epoch Time: {:.2f}s'.
                              format(epoch + 1, self.num_epochs, loss_training_each, top1_avg, loss_test_each, epoch_time))
                else:
                    if self.flag_acc5 == 1:
                        print('Epoch [{}/{}], Training Loss: {:.4f}, Top1 Acc: {:.3f} %, Top5 Acc: {:.3f} %, Test Loss: {:.4f}, Epoch Time: {:.2f}s'.
                              format(epoch + 1, self.num_epochs, loss_training_each, top1_avg, top5_avg, loss_test_each, epoch_time))
                    else:
                        print('Epoch [{}/{}], Training Loss: {:.4f}, Test Acc: {:.3f} %, Test Loss: {:.4f}, Epoch Time: {:.2f}s'.
                              format(epoch + 1, self.num_epochs, loss_training_each, top1_avg, loss_test_each, epoch_time))

            if self.flag_wandb == 1:
                wandb.log({"loss_training": loss_training_each}, step=epoch)
                wandb.log({"test_acc": top1_avg}, step=epoch)
                wandb.log({"loss_test": loss_test_each}, step=epoch)
                wandb.log({"epoch_time": epoch_time}, step=epoch)
                # print(learning_rate)
                wandb.log({"learning_rate": learning_rate}, step=epoch)

                if self.flag_als >= 1:
                    for i in range(self.num_layer):
                        wandb.log({'rate_p%s' % i: layer_rate[i]}, step=epoch)
                        wandb.log({'count_p%s' % i: aug_layer_count[i]}, step=epoch)

            if self.save_file == 1:
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

        top1_avg_max = 0
        if flag_log == 1:
            top1_avg_max = np.max(results[:, 1])
            print(top1_avg_max)
            if self.flag_acc5 == 1:
                top5_avg_max = np.max(results[:, 2])
                print(top5_avg_max)

        """ファイル保存"""
        if self.save_file == 1:
            if flag_log == 1:
                if self.flag_randaug == 1:  # RandAugment
                    np.savetxt('results/data_%s_model_%s_num_%s_batch_%s_aug_%s_dropout_%s_layer_aug_%s_layer_drop_%s_randlayer_%s_n_%s_m_%s_seed_%s_acc_%s.csv'
                               % (self.n_data, self.n_model, self.num_training_data, self.batch_size_training, self.n_aug, self.flag_dropout, self.layer_aug, self.layer_drop, self.flag_random_layer, self.rand_n, self.rand_m, self.seed, top1_avg_max),
                               results, delimiter=',')
                else:
                    if self.flag_als >= 1:
                        np.savetxt('results/data_%s_model_%s_num_%s_batch_%s_aug_%s_als_%s_layer_aug_%s_alsrate_%s_epochrand_%s_interval_%s_adv_%s_seed_%s_acc_%s.csv'
                                   % (self.n_data, self.n_model, self.num_training_data, self.batch_size_training, self.n_aug, self.flag_als, self.layer_aug, self.als_rate, self.epoch_random, self.iter_interval, self.flag_adversarial,
                                      self.seed, top1_avg_max),
                                   results, delimiter=',')
                    else:
                        np.savetxt('results/data_%s_model_%s_num_%s_batch_%s_aug_%s_als_%s_layer_aug_%s_randlayer_%s_seed_%s_acc_%s.csv'
                                   % (self.n_data, self.n_model, self.num_training_data, self.batch_size_training, self.n_aug, self.flag_als, self.layer_aug, self.flag_random_layer, self.seed, top1_avg_max),
                                   results, delimiter=',')

                if self.flag_random_layer == 1:
                    if self.flag_als >= 1:
                        np.savetxt('results/random/layer_rate_data_%s_model_%s_num_%s_batch_%s_aug_%s_als_%s_alsrate_%s_epochrand_%s_interval_%s_adv_%s_seed_%s.csv'
                                   % (self.n_data, self.n_model, self.num_training_data, self.batch_size_training, self.n_aug, self.flag_als, self.als_rate, self.epoch_random, self.iter_interval, self.flag_adversarial, self.seed),
                                   self.layer_rate_all, delimiter=',')
                    np.savetxt('results/random/layer_count_data_%s_model_%s_num_%s_batch_%s_aug_%s_als_%s_alsrate_%s_epochrand_%s_interval_%s_adv_%s_seed_%s.csv'
                               % (self.n_data, self.n_model, self.num_training_data, self.batch_size_training, self.n_aug, self.flag_als, self.als_rate, self.epoch_random, self.iter_interval, self.flag_adversarial, self.seed),
                               self.aug_layer_count_all, delimiter=',')
