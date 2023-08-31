import os
from pathlib import Path
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
import logging


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    return tensor * (max_value - min_value) + min_value

def tensor_round(tensor):
    return torch.round(tensor)


class MNISTData:

    def __init__(self, data_path=None):
        if data_path is None:
            data_path = Path(os.getcwd()) / "data"
        self.data_path = data_path

    def get_data_loaders(self, train_batch_size=256, test_batch_size=512, random_seed=42, val_pct=0.1,
                         num_workers=4, pin_memory=True):

        error_msg = "[!] val_pct and partial_pct should be in the range [0, 1]."
        assert ((val_pct >= 0) and (val_pct <= 1)), error_msg

        train_transform = transforms.Compose([
            #transforms.RandomRotation(degrees=(-30, 30)),
            #transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Randomly change the brightness, contrast and saturation of an image. For more transfomations see: https://pytorch.org/docs/stable/torchvision/transforms.html
            transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor
            transforms.Lambda(lambda tensor: min_max_normalization(tensor, 0, 1)),
            transforms.Lambda(lambda tensor: tensor_round(tensor))
            #transforms.Normalize((0.1307,), (0.3081,))  # translate by 0.13 and scale by 0.308
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor
            transforms.Lambda(lambda tensor: min_max_normalization(tensor, 0, 1)),
            transforms.Lambda(lambda tensor: tensor_round(tensor))
            #transforms.Normalize((0.1307,), (0.3081,))  # translate by 0.13 and scale by 0.308
        ])

        # load the datafolder
        train_dataset = MNIST(
            root=self.data_path,
            train=True,
            download=True,
            transform=train_transform
        )

        # valid
        valid_dataset = MNIST(
            root=self.data_path,
            train=True,
            download=True,
            transform=test_transform
        )
        # test
        test_dataset = MNIST(
            root=self.data_path,
            train=False,
            download=True,
            transform=test_transform
        )

        indices = np.arange(len(train_dataset.targets))
        targets = train_dataset.targets.numpy()
        train_indices, val_indices = train_test_split(indices, test_size=val_pct, stratify=targets,
                                                      random_state=random_seed)

        train_dataset = Subset(train_dataset, indices=train_indices)
        val_dataset = Subset(valid_dataset, indices=val_indices)

        num_train = len(train_dataset)
        num_val = len(val_dataset)
        num_test = len(test_dataset)

        logging.info(
            f"Train full size = {num_train}, Sum (Err. Check): {train_indices.sum()}\n"
            f"validation size = {num_val}, Sum (Err. Check): {val_indices.sum()}\n"
            f"Test size = {num_test}"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            drop_last=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            drop_last=False
        )

        return train_loader, val_loader, test_loader