import logging
import os
from pathlib import Path
import json

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from experiments.utils import set_logger

set_logger()


class CIFARData:
    """
    Create train, valid, test iterators for CIFAR-10/100 [1].
    [1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
    """

    def __init__(self, data_dir=None, data_name='cifar10'):
        if data_dir is None:
            data_dir = Path(os.getcwd()) / "dataset"

        self.data_dir = data_dir
        if data_name == 'cifar10':
            split_file_loc = data_dir + "/cifar10_train_validation_indices.json"
        else:  # cifar100
            split_file_loc = data_dir + "/cifar100_train_validation_indices.json"

        self.train_val_idxs = json.load(open(split_file_loc, "r"))
        self.data_name = data_name

    def get_loaders(self, batch_size=256, test_batch_size=512, augment=False, num_workers=4,
                    pin_memory=True):

        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        # define transforms
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        dataset = CIFAR10 if self.data_name == 'cifar10' else CIFAR100

        # load the datafolder
        train_dataset = dataset(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        # valid
        valid_dataset = dataset(
            root=self.data_dir,
            train=True,
            download=True,
            transform=valid_transform  # no aug.
        )
        # test
        test_dataset = dataset(
            root=self.data_dir,
            train=False,
            download=True,
            transform=valid_transform
        )

        # take subset
        train_idxs = self.train_val_idxs['train_indices']
        train_dataset = Subset(train_dataset, indices=train_idxs)
        val_idxs = self.train_val_idxs['validation_indices']
        valid_dataset = Subset(valid_dataset, indices=val_idxs)

        num_train = len(train_dataset)
        num_val = len(valid_dataset)
        num_test = len(test_dataset)

        logging.info(
            f"Train size = {num_train}, validation size = {num_val}, test size = {num_test}"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True
        )

        val_loader = DataLoader(
            valid_dataset,
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False
        )

        return train_loader, val_loader, test_loader