import os
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
import torch
import torch.nn.functional as F
import oxford_flowers_dataset, oxford_pets_dataset
import numpy as np
# from spawrious.torch import get_spawrious_dataset
from torch.utils.data import Dataset

def build_dataset(is_train, args, transform_train=None, data_set_override=None, data_path_override=None):
    if data_path_override is not None:
        data_path = data_path_override
    else:
        data_path = args.data_path

    if data_set_override is not None:
        data_set = data_set_override
    else:
        data_set = args.data_set

    if not transform_train:
        transform_train = is_train
    transform = build_transform(transform_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif data_set == 'IMNET':
        root = os.path.join(data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif data_set == 'IMNET100':
        root = os.path.join(data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100
    elif data_set == 'IMNET100_test':
        root = os.path.join(data_path, 'train' if is_train else 'val')
        dataset = ImageFolderWithPaths(root, transform=transform)
        nb_classes = 100
    elif data_set == 'IMNET_EVAL':
        root = os.path.join(data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100
    # elif data_set == 'RANDOM_LABEL':
    #     root = os.path.join(data_path, 'train' if is_train else 'val')
    #     dataset = ImageFolderWithFixedRandomLabel(root, transform=transform, seed=args.seed)
    #     nb_classes = 100
    # elif data_set == 'GAUSSIAN_IMNET100':
    #     root = os.path.join(data_path, 'train' if is_train else 'val')
    #     dataset = datasets.ImageFolder(root, transform=transform)
    #     nb_classes = 100
    # elif data_set.startswith("spawrious_"):
    #     spawrious = get_spawrious_dataset(dataset_name=data_set.replace("spawrious_", ""), root_dir=data_path)
    #     if is_train:
    #         dataset = spawrious.get_train_dataset()
    #         # dataset = TransformDataset(dataset, transform=transform)
    #     else:
    #         dataset = spawrious.get_test_dataset()
    #         # dataset = TransformDataset(dataset, transform=transform)
    #     nb_classes = 4
    elif data_set == "flowers":
        dataset = oxford_flowers_dataset.Flowers(root=data_path, 
                                     train=is_train,
                                     download=False,
                                     transform=transform)
        nb_classes = 102
    elif data_set == "pets":
        dataset = oxford_pets_dataset.Pets(root=data_path,
                                     train=is_train,
                                     download=True,
                                     transform=transform)
        nb_classes = 37
    elif data_set == "stl10":
        if is_train:
            dataset = datasets.STL10(root=data_path,
                                         split='train',
                                         download=True,
                                         transform=transform)
        else:
            dataset = datasets.STL10(root=data_path,
                                     split='test',
                                     download=True,
                                     transform=transform)
        nb_classes = 10
    elif data_set == "food101":
        if is_train:
            dataset = datasets.Food101(root=data_path,
                                         split='train',
                                         download=True,
                                         transform=transform)
        else:
            dataset = datasets.Food101(root=data_path,
                                     split='test',
                                     download=True,
                                     transform=transform)
        nb_classes = 101
    else:
        raise NotImplementedError()
    args.nb_classes = nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        if args.image_flip:
            transforms_list = list(transform.transforms)
            transforms_list.append(transforms.Lambda(lambda x: x.transpose(1, 2)))
            return transforms.Compose(transforms_list)
        else:
            return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))

    if args.image_flip:
        t.append(transforms.Lambda(lambda x: x.transpose(1, 2)))

    return transforms.Compose(t)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]  # Full path
        name = path.split("/")[-1].split(".")[0]  # Filename only
        mask_path = path.replace("/val/", "/masks/").replace(".JPEG", ".png")

        crop_pct = 224 / 256
        size = int(224 / crop_pct)

        mask_original_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        mask = self.loader(mask_path)
        mask_original = mask_original_transform(mask)

        mask_patch = F.max_pool2d(mask_original.float(), kernel_size=16, stride=16)
        
        return img, label, mask_original, mask_patch, name

class ImageFolderWithFixedRandomLabel(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, seed=42):
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        rng = np.random.default_rng(seed)
        self.random_labels = rng.integers(
            low=len(self.classes),
            high=None,
            size=len(self.samples),
            dtype=np.int64
        )
        print(f"[INFO] Sample random label - first 20 labels: {self.random_labels[:20]}")

    def __getitem__(self, index):
        sample, _ = super().__getitem__(index)
        target = int(self.random_labels[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        x, y = batch[0], batch[1]
        if self.transform is not None:
            if not isinstance(x, Image.Image):
                x = transforms.ToPILImage()(x)
            x = self.transform(x)
        return x, y
