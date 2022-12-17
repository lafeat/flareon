import torch.utils.data as data
import torch
import torchvision
from torchvision.transforms import transforms
from augment.autoaugment import CIFAR10Policy
import os, glob
from io import BytesIO
import kornia.augmentation as A
import random
import numpy as np
from augment.Cutout import Cutout
from augment.flowaugment import Flow_Augment
from augment.randaugment import Rand_Augment
from PIL import Image


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x

def get_transform(opt, train=True):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if opt.dataset == "cifar10":
        if train:
            if opt.ag == "flowag":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
                transforms_list.append(Flow_Augment(Numbers=2, max_Magnitude=5))
            elif opt.ag == "randag":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
                transforms_list.append(Rand_Augment(Numbers=3, max_Magnitude=5))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    elif opt.dataset == "celeba":
        if opt.ag == "flowag":
            if train:
                transforms_list.append(Flow_Augment(Numbers=2, max_Magnitude=5))
        transforms_list.append(transforms.ToTensor())
    elif opt.dataset == "tinyimagenet":
        if opt.ag == "flowag":
            if train:
                transforms_list.append(Flow_Augment(Numbers=2, max_Magnitude=9))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)

class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_rotation = ProbTransform(A.RandomRotation(30), p=0.5)
        self.random_tra = ProbTransform(A.RandomAffine(degrees=20, translate=(0, 0.2)), p=0.5)
        self.random_shear = ProbTransform(A.RandomAffine(degrees=30, shear = (0, 0.5, 0, 0.5)), p=0.5)
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root,
                                                   split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


class TinyImageNet(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(TinyImageNet).__init__()
        self.class_dict = self._getclass(opt.data_root)
        if train:
            self.filenames = glob.glob(os.path.join(opt.data_root, "tiny-imagenet-200/train/*/*/*.JPEG"))
            self.transform = transforms
        else:
            self.filenames = glob.glob(os.path.join(opt.data_root, "tiny-imagenet-200/val/*/*/*.JPEG"))
            self.transform = transforms

    def _getclass(self, data_root):
        class_dict = {}
        for i, line in enumerate(open(os.path.join(data_root, 'tiny-imagenet-200/wnids.txt'), 'r')):
            class_dict[line.replace('\n', '')] = i

        return class_dict

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if len(image.split()) == 1:
            image = image.convert('RGB')
            bytesIO = BytesIO()
            image.save(bytesIO, format='JPEG')
            image = bytesIO.getvalue()
            with open(img_path, 'wb') as f:
                f.write(image)
            image = Image.open(img_path)

        label = self.class_dict[img_path.split('/')[4]] # From path to debug
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(opt, train=True):
    transform = get_transform(opt, train)
    if opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)
    elif opt.dataset == 'tinyimagenet':
        dataset = TinyImageNet(opt, train, transform)
    else:
        raise Exception("Invalid dataset")
    if train:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs,
                                                 num_workers=opt.num_workers, shuffle=True, drop_last=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs,
                                                 num_workers=opt.num_workers, shuffle=False)
    return dataloader

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x

def get_dataset(opt, train=True):
    if opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train,
                                               transform=ToNumpy(), download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(
            opt,
            split,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]))
    elif opt.dataset == "tinyimagenet":
        dataset = TinyImageNet(opt, train, ToNumpy())
    else:
        raise Exception("Invalid dataset")
    return dataset
