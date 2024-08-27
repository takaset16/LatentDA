# coding: utf-8
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from PIL import Image
import util

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}


class DataSetXY(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.x)


class MyDataset_training(Dataset):
    def __init__(self, n_data, flag_defaug, flag_transfer):
        self.sampler = None

        """データの前処理"""
        transform_train = None
        if n_data == 'MNIST':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
            ])
        elif n_data == 'CIFAR-10' or n_data == 'CIFAR-100':
            if flag_defaug == 1:
                if flag_transfer == 1:  # transfer learning
                    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                else:
                    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))  # Comment out if you save images
                    ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                ])
        elif n_data == 'SVHN':
            if flag_defaug == 1:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442), std=(0.19803012, 0.20101562, 0.19703614))
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442), std=(0.19803012, 0.20101562, 0.19703614))
                ])
        elif n_data == 'Fashion-MNIST':
            transform_train = transforms.Compose([
                transforms.ToTensor()
            ])
        elif n_data == 'STL-10':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        elif n_data == 'ImageNet':
            if flag_defaug == 1:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                    ),
                    transforms.ToTensor(),
                    Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                    ),
                    transforms.ToTensor(),
                    Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        elif n_data == 'TinyImageNet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                transforms.ToTensor(),
                Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif n_data == 'EMG':
            train_x = np.zeros((624 * 25, 50 * 8))
            train_y = np.zeros(624 * 25)

            for s in range(25):  # 被験者数
                DATA_PATH = "../../../../groups/gac50437/datasets/EMG/features_raw/raw_data_MyoDataset/sub{:02}/training_data.mat"
                LABEL_PATH = "../../../../groups/gac50437/datasets/EMG/features_raw/raw_data_MyoDataset/sub{:02}/training_label.mat"

                training_data = sio.loadmat(DATA_PATH.format(s + 1))["training_data"]
                training_label = sio.loadmat(LABEL_PATH.format(s + 1))["training_label"]
                training_label = training_label - 1

                train_x[s * 624] = training_data.reshape((training_data.shape[0], training_data.shape[1] * training_data.shape[2]))
                train_y[s * 624] = training_label.reshape(training_label.shape[0])

                print(train_x.shape)

                """
                train_x = np.loadtxt("../../datasets/uci/letter/input_data.csv", delimiter=',', dtype=np.float32)
                train_y = np.loadtxt("../../datasets/uci/letter/output_data.csv", delimiter=',', dtype=np.int32)
                self.mydata = DataSetXY(x=torch.from_numpy(train_x).float(), y=train_y)
                _, self.mydata = util.make_training_test_data(self.mydata, int(20000 * 0.35), seed)
                """
            self.mydata = DataSetXY(x=torch.from_numpy(train_x).float(), y=train_y)

        """データ選択"""
        if n_data == 'MNIST':
            self.mydata = torchvision.datasets.MNIST(root='../../datasets/mnist', train=True, transform=transform_train, download=True)
        elif n_data == 'CIFAR-10':  # CIFAR-10
            self.mydata = torchvision.datasets.CIFAR10(root='../../datasets/cifar10', train=True, transform=transform_train, download=True)
        elif n_data == 'SVHN':  # SVHN
            self.mydata = torchvision.datasets.SVHN(root='../../datasets/svhn', split='train', transform=transform_train, download=True)
            # trainset = torchvision.datasets.SVHN(root='../../datasets/svhn', split='train', transform=transform_train, download=True)
            # extraset = torchvision.datasets.SVHN(root='../../datasets/svhn', split='extra', transform=transform_train, download=True)
            # self.mydata = ConcatDataset([trainset, extraset])
        elif n_data == 'CIFAR-100':  # CIFAR-100
            self.mydata = torchvision.datasets.CIFAR100(root='../../datasets/cifar100', train=True, transform=transform_train, download=True)
        elif n_data == 'Fashion-MNIST':  # Fashion-MNIST
            self.mydata = torchvision.datasets.FashionMNIST(root='../../datasets/FashionMNIST', train=True, transform=transform_train, download=True)
        elif n_data == 'ImageNet':  # ImageNet
            self.mydata = torchvision.datasets.ImageFolder(root='../../../../../groups/gac50437/datasets/Imagenet/train', transform=transform_train)

    def __getitem__(self, index):
        x, y = self.mydata[index]

        return x, y, index

    def __len__(self):
        return len(self.mydata)

    def get_info(self, n_data):
        num_channel = 3
        num_classes = 10
        input_size = 0
        hidden_size = 1000

        if n_data == 'MNIST':  # MNIST
            num_channel = 1
            num_classes = 10
            input_size = 28 * 28 * 1
            num_training_data = 60000
            num_test_data = 10000
        elif n_data == 'CIFAR-10':  # CIFAR-10
            num_channel = 3
            num_classes = 10
            input_size = 32 * 32 * 3
            num_training_data = 50000
            num_test_data = 10000
        elif n_data == 'SVHN':  # SVHN
            num_channel = 3
            num_classes = 10
            input_size = 32 * 32 * 3
            num_training_data = 73257
            num_training_data = 73257 + 531131
            num_test_data = 26032
        elif n_data == 'CIFAR-100':  # CIFAR-100
            num_channel = 3
            num_classes = 100
            input_size = 32 * 32 * 3
            num_training_data = 50000
            num_test_data = 10000
        elif n_data == 'Fashion-MNIST':  # Fashion-MNIST
            num_channel = 1
            num_classes = 10
            input_size = 28 * 28 * 1
            num_training_data = 60000
            num_test_data = 10000
        elif n_data == 'ImageNet':  # ImageNet
            num_channel = 3
            num_classes = 1000
            input_size = 224 * 224 * 3  # transform後のサイズ
            num_training_data = 1300000
            num_test_data = 50000

        return num_channel, num_classes, input_size, hidden_size, num_training_data


class MyDataset_test(Dataset):
    def __init__(self, n_data, flag_transfer):
        self.sampler = None

        """データの前処理"""
        transform_test = None
        if n_data == 'MNIST':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
            ])
        elif n_data == 'CIFAR-10' or n_data == 'CIFAR-100':
            if flag_transfer == 1:  # transfer learning
                transform_test = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                ])
        elif n_data == 'SVHN':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442), std=(0.19803012, 0.20101562, 0.19703614))  # Comment out when saving images
            ])
        elif n_data == 'Fashion-MNIST':
            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])
        elif n_data == 'ImageNet':
            transform_test = transforms.Compose([
                transforms.Resize(256, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        """データ選択"""
        if n_data == 'MNIST':  # MNIST
            self.mydata = torchvision.datasets.MNIST(root='../../datasets/mnist', train=False, transform=transform_test, download=True)
        elif n_data == 'CIFAR-10':  # CIFAR-10
            self.mydata = torchvision.datasets.CIFAR10(root='../../datasets/cifar10', train=False, transform=transform_test, download=True)
        elif n_data == 'SVHN':  # SVHN
            self.mydata = torchvision.datasets.SVHN(root='../../datasets/svhn', split='test', transform=transform_test, download=True)
        elif n_data == 'CIFAR-100':  # CIFAR-100
            self.mydata = torchvision.datasets.CIFAR100(root='../../datasets/cifar100', train=False, transform=transform_test, download=True)
        elif n_data == 'Fashion-MNIST':  # Fashion-MNIST
            self.mydata = torchvision.datasets.FashionMNIST(root='../../datasets/FashionMNIST', train=False, transform=transform_test, download=True)
        elif n_data == 'ImageNet':  # ImageNet
            self.mydata = torchvision.datasets.ImageFolder(root='../../../../../groups/gac50437/datasets/Imagenet/val', transform=transform_test)

    def __getitem__(self, index):
        x, y = self.mydata[index]

        return x, y, index

    def __len__(self):
        return len(self.mydata)


class MyDataset_als(Dataset):
    def __init__(self, n_data, flag_defaug, flag_transfer, degree=10):
        self.sampler = None

        """データの前処理"""
        transform_als = None
        if n_data == 'MNIST':
            transform_als = transforms.Compose([
                transforms.RandomRotation(degrees=degree),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
            ])
        elif n_data == 'CIFAR-10' or n_data == 'CIFAR-100':
            if flag_defaug == 1:
                if flag_transfer == 1:  # transfer learning
                    transform_als = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=degree),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                else:
                    transform_als = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=degree),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                    ])
            else:
                transform_als = transforms.Compose([
                    transforms.RandomRotation(degrees=degree),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                ])
        elif n_data == 'SVHN':
            if flag_defaug == 1:
                transform_als = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=degree),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442), std=(0.19803012, 0.20101562, 0.19703614))
                ])
            else:
                transform_als = transforms.Compose([
                    transforms.RandomRotation(degrees=degree),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442), std=(0.19803012, 0.20101562, 0.19703614))
                ])
        elif n_data == 'Fashion-MNIST':
            transform_als = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=degree),
                transforms.ToTensor()
            ])
        elif n_data == 'ImageNet':
            if flag_defaug == 1:
                transform_als = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                    ),
                    transforms.RandomRotation(degrees=degree),
                    transforms.ToTensor(),
                    Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform_als = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                    ),
                    transforms.RandomRotation(degrees=degree),
                    transforms.ToTensor(),
                    Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        """データ選択"""
        if n_data == 'MNIST':
            self.mydata = torchvision.datasets.MNIST(root='../../datasets/mnist', train=True, transform=transform_als, download=True)
        elif n_data == 'CIFAR-10':  # CIFAR-10
            self.mydata = torchvision.datasets.CIFAR10(root='../../datasets/cifar10', train=True, transform=transform_als, download=True)
        elif n_data == 'SVHN':  # SVHN
            self.mydata = torchvision.datasets.SVHN(root='../../datasets/svhn', split='train', transform=transform_als, download=True)
            # trainset = torchvision.datasets.SVHN(root='../../datasets/svhn', split='train', transform=transform_als, download=True)
            # extraset = torchvision.datasets.SVHN(root='../../datasets/svhn', split='extra', transform=transform_als, download=True)
            # self.mydata = ConcatDataset([trainset, extraset])
        elif n_data == 'CIFAR-100':  # CIFAR-100
            self.mydata = torchvision.datasets.CIFAR100(root='../../datasets/cifar100', train=True, transform=transform_als, download=True)
        elif n_data == 'Fashion-MNIST':  # Fashion-MNIST
            self.mydata = torchvision.datasets.FashionMNIST(root='../../datasets/FashionMNIST', train=True, transform=transform_als, download=True)
        elif n_data == 'ImageNet':  # ImageNet
            self.mydata = torchvision.datasets.ImageFolder(root='../../../../../groups/gac50437/datasets/Imagenet/train', transform=transform_als)

    def __getitem__(self, index):
        x, y = self.mydata[index]

        # return x, y
        return x, y, index

    def __len__(self):
        return len(self.mydata)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


