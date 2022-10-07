import os
import time
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import numpy as np
import copy

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.io import read_image

np.random.seed(6)


def get_cls_num_images(dataset, num_classes=None):
    num_classes = num_classes or (max(dataset.targets, default=0) + 1)
    cls_n = [0] * num_classes
    for label in dataset.targets:
        cls_n[label] += 1
    return cls_n


def get_cls_img_id_list(dataset, num_classes):
    ret = [[] for i in range(num_classes)]
    for i, label in enumerate(dataset.targets):
        ret[label].append(i)
    return ret


def create_sub_dataset(dataset, idx_list):
    idx_list.sort()
    ret = copy.deepcopy(dataset)
    if isinstance(dataset, EnImageDataset):
        ret.data = [dataset.data[i] for i in idx_list]
    else:
        ret.data = np.take(dataset.data, idx_list, axis=0)
        ret.targets = np.take(dataset.targets, idx_list, axis=0)
    return ret


def merge_dataset(dst, src):
    if isinstance(dst, EnImageDataset):
        dst.data = dst.data + src.data
    else:
        dst.data = np.concatenate((dst.data, src.data), axis=0)
        dst.targets = np.concatenate((dst.targets, src.targets), axis=0)
    return dst


def up_sample_dataset(dataset, idx_list):
    if isinstance(dataset, EnImageDataset):
        dataset.data += [dataset.data[i] for i in idx_list]
    else:
        dataset.data = np.concatenate([dataset.data, np.take(dataset.data, idx_list, axis=0)])
        dataset.targets = np.concatenate([dataset.targets, np.take(dataset.targets, idx_list, axis=0)])


class EnImageDataset(VisionDataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        with open(os.path.join(self.root, 'train.txt' if train else 'val.txt')) as f:
            self.data = [self.decode_img_label(line) for line in f]
        self.targets = [label for img, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fn, label = self.data[index]
        image = Image.open(os.path.join(self.root, fn))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # print(image.size, image.mode, image.getbands())
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # print(image, label, type(label))
        return image, label

    @staticmethod
    def decode_img_label(line):
        img, _, label = line.rpartition(': ')
        return img, int(label)


def check_split_info(dataset):
    folder = os.path.join('../../data', dataset)
    train_ds = EnImageDataset(folder, True)
    val_ds = EnImageDataset(folder, False)
    with open(os.path.join(folder, 'img_num_per_cls.txt')) as f:
        img_num_per_cls = [int(line) for line in f]

    def print_cls_n(name, cls_n):
        print(f'{name}:')
        print(' ', len(cls_n), 'classes,', sum(cls_n), 'samples')
        print(f'  min: {min(cls_n)}, max: {max(cls_n)}, im: {max(cls_n) / min(cls_n):.0f}')

    print_cls_n('img_num_per_cls', img_num_per_cls)

    train_n = get_cls_num_images(train_ds)
    print_cls_n('train', train_n)
    print_cls_n('val', get_cls_num_images(val_ds))

    assert len(train_n) == len(img_num_per_cls), f'{len(train_n)} == {len(img_num_per_cls)}'
    for a, b in zip(train_n, img_num_per_cls):
        assert a == b
    exit(1)


RGB_statistics = {
    'cifar10': {
        'mean': [x / 255.0 for x in [125.3, 123.0, 113.9]],
        'std': [x / 255.0 for x in [63.0, 62.1, 66.7]]
    },
    'cifar100': {
        'mean': [x / 255.0 for x in [125.3, 123.0, 113.9]],
        'std': [x / 255.0 for x in [63.0, 62.1, 66.7]]
    },
    'inat2017': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'inat2018': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
}


def pad_squeeze(x):
    return F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze()


def create_transform(dataset):
    normalize = transforms.Normalize(RGB_statistics[dataset]['mean'], RGB_statistics[dataset]['std'])
    if dataset in {'cifar10', 'cifar100'}:
        train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(pad_squeeze),
            # transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
            #                                   (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    elif dataset in {'inat2017', 'inat2018'}:
        train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return train, test


# random.seed(2)
def build_dataset(dataset, num_meta):
    transform_train, transform_test = create_transform(dataset)

    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10('../data', train=False, transform=transform_test)
        num_classes = 10
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100('../data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100('../data', train=False, transform=transform_test)
        num_classes = 100
    elif dataset == 'inat2017':
        # check_split_info(dataset)
        train_dataset = EnImageDataset('../data/inat2017', train=True, transform=transform_train)
        test_dataset = EnImageDataset('../data/inat2017', train=False, transform=transform_test)
        num_classes = 5089
    elif dataset == 'inat2018':
        # check_split_info(dataset)
        train_dataset = EnImageDataset('../data/inat2018', train=True, transform=transform_train)
        test_dataset = EnImageDataset('../data/inat2018', train=False, transform=transform_test)
        num_classes = 8142

    # time_log.add('Load Data')

    # if num_meta <= 0:
    #     return None, train_dataset, test_dataset, num_classes

    img_num_list = [num_meta] * num_classes
    # print(dataset, 'img_num_list:', img_num_list)
    data_list_val = get_cls_img_id_list(train_dataset, num_classes)

    # time_log.add('Create Index')

    idx_to_meta = []
    idx_to_train = []
    for cls_idx, img_id_list in enumerate(data_list_val):
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        idx_to_meta.extend(img_id_list[:img_num])
        idx_to_train.extend(img_id_list[img_num:])

    # time_log.add('Split Index')

    train_data = copy.deepcopy(train_dataset)
    train_data_meta = copy.deepcopy(train_dataset)

    if isinstance(train_dataset, EnImageDataset):
        train_data.data = [train_dataset.data[i] for i in idx_to_train]
        train_data.targets = [train_dataset.targets[i] for i in idx_to_train]
        train_data_meta.data = [train_dataset.data[i] for i in idx_to_meta]
        train_data_meta.targets = [train_dataset.targets[i] for i in idx_to_meta]
    else:
        train_data.data = np.delete(train_dataset.data, idx_to_meta, axis=0)
        train_data.targets = np.delete(train_dataset.targets, idx_to_meta, axis=0)
        train_data_meta.data = np.delete(train_dataset.data, idx_to_train, axis=0)
        train_data_meta.targets = np.delete(train_dataset.targets, idx_to_train, axis=0)
    # time_log.add('Extract Data')

    # print('test sample:', test_dataset[0])
    # print('train sample:', train_data[0])
    # exit(1)

    return train_data_meta, train_data, test_dataset, num_classes


def get_img_num_per_cls(dataset, imb_factor=None, num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = (50000 - num_meta) / 10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (50000 - num_meta) / 100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


# This function is used to generate imbalanced test set
'''
def get_img_num_per_cls_test(dataset,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = (10000-num_meta)/10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (10000-num_meta)/100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls
'''
