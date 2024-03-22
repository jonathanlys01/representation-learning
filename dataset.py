import torchvision
import torch

import matplotlib.pyplot as plt

import os 
import numpy as np

import torchvision.transforms as T

# MNIST dataset

def get_mnist(batch_size=32, shuffle=True, loader=True):
    dataset =  torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=T.ToTensor(),
        download=True
    )
    if not loader:
        return dataset
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers= 2 if os.cpu_count() > 2 else 1
    )
    return dataloader

cifar_train_transform = T.Compose([
    T.Resize((32, 32)), 
    T.RandomChoice([
        torch.nn.Identity(),
        T.RandomHorizontalFlip(p=1),
        T.RandomVerticalFlip(p=1),
        T.RandomRotation(5),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    ]),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    T.ToTensor(),
])

cifar_eval_transform = T.Compose([
    T.Resize((32, 32)), 
    T.ToTensor(),
])

def get_cifar(batch_size=32, shuffle=True, loader=True, type="train"):
    dataset =  torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        transform=cifar_train_transform if type == "train" else cifar_eval_transform,
        download=True
    )
    if not loader:
        return dataset
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers= 2 if os.cpu_count() > 2 else 1
    )
    return dataloader

classes_cifar = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
classes_mnist = tuple(str(i) for i in range(10))

if __name__ == '__main__':
    dataset = get_cifar(loader=False)
    print(len(dataset))
    idx = np.random.randint(0, len(dataset))
    img, label = dataset[idx]
    img = img.permute(1, 2, 0)
    plt.imshow(img.numpy())
    plt.title(f"Label: {label}, (idx: {idx})")
    plt.show()
