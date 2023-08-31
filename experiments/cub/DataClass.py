import logging
import os
from pathlib import Path
import json
from PIL import Image

from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from experiments.cub.CUB2011 import Cub2011
from experiments.utils import set_logger

set_logger()

class CUBSubSet(Subset):
    """Face Landmarks dataset."""
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target


class CUBData:
    """Source: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

    Create train, valid, test iterators for CIFAR-10 [1].
    [1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4

    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = Path(os.getcwd()) / "dataset"

        self.data_dir = data_dir

    def get_loaders(
            self,
            max_class=200,
            val_samples_per_class=5,
            batch_size=64,
            test_batch_size=128,
            shuffle_train=True,
            num_workers=4,
            pin_memory=True,
            random_state=42
    ):
        # TODO: Check statistics after resize
        # imagenet statistics
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # define transforms
        standard_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset, valid_dataset, test_dataset = \
            self.get_datsets(standard_transforms, max_class)

        # take validation set from train
        indices = np.arange(len(train_dataset))
        targets = np.array(train_dataset.data.target)
        if val_samples_per_class > 0:
            train_indices, val_indices = train_test_split(indices, test_size=val_samples_per_class * max_class,
                                                          stratify=targets,
                                                          random_state=random_state)
            valid_dataset = Subset(valid_dataset, indices=val_indices)

        else:
            train_indices = indices
            val_indices = None

        train_dataset = Subset(train_dataset, indices=train_indices)
        num_train = len(train_dataset)
        num_val = len(val_indices)
        num_test = len(test_dataset)

        logging.info(
            f"\nTrain size = {num_train}, Sum (Err. Check) = {np.sum(train_indices)}"
            f"\nVal size = {num_val}, Sum (Err. Check) = {np.sum(val_indices)}"
            f"\nTest size = {num_test}"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            valid_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader


    def get_datsets(self, transforms, max_class):
        # load the dataset
        train_dataset = Cub2011(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms,
            max_class=max_class
        )
        # valid
        valid_dataset = Cub2011(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms,
            max_class=max_class
        )
        # test
        test_dataset = Cub2011(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transforms,
            max_class=max_class
        )
        return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    cub_data = CUBData()

    standard_train_loader, resized_train_loader, standard_val_loader, \
    resized_val_loader, standard_test_loader, resized_test_loader = cub_data.get_loaders(train_samples_per_class=10)