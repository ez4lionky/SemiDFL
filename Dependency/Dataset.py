import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class RepeatedKTransform:
    def __init__(self, transform, k=2):
        self.transform = transform
        self.k = k

    def __call__(self, inp):
        outs = []
        for i in range(self.k):
            outs.append(self.transform(inp))
        return outs


class DatasetLabeled(data.Dataset):
    def __init__(self, samples, labels, transform=None, to_onehot=True, classes=10):
        super(DatasetLabeled, self).__init__()
        samples = (samples / 255.0).float()
        if to_onehot:
            labels = torch.zeros(len(labels), classes).scatter_(1, labels.view(-1, 1).long(), 1)

        self.classes = classes
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        samples = self.samples[index]
        labels = self.labels[index]
        if self.transform is not None:
            samples = self.transform(samples)
        return samples, labels, index

    def __len__(self):
        return len(self.samples)
    
    
class DatasetUnlabeled(data.Dataset):
    def __init__(self, samples, labels, transform=None, classes=10):
        super(DatasetUnlabeled, self).__init__()
        samples = (samples / 255.0).float()
        self.classes = classes
        self.samples = samples
        labels = torch.tensor(labels)
        labels = torch.zeros(len(labels), classes).scatter_(1, labels.view(-1, 1).long(), 1)
        self.labels = labels
        self.transform = RepeatedKTransform(transform)

    def __getitem__(self, index):
        samples = self.samples[index]
        if self.transform is not None:
            samples = self.transform(samples)
        labels = self.labels[index]
        return samples, labels, index

    def __len__(self):
        return len(self.samples)


# Dataset Pseudo Labeled
class DatasetPL(data.Dataset):
    def __init__(self, samples, labels, ori_indexes, transform=None, classes=10):
        super(DatasetPL, self).__init__()
        self.classes = classes
        self.samples = samples
        self.labels = labels
        self.ori_indexes = ori_indexes
        self.transform = transform

    def __getitem__(self, index):
        samples = self.samples[index]
        labels = self.labels[index]
        if self.transform is not None:
            samples = self.transform(samples)
        ori_index = self.ori_indexes[index]
        return samples, labels, ori_index

    def __len__(self):
        return len(self.samples)
