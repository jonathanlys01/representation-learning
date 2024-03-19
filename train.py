import torch
from tqdm import tqdm
from model import CondVariationalAutoEncoder, AutoEncoder, ConvAutoencoder
from dataset import get_mnist, get_cifar
import os
from utils import get_device, KL_divergence

import numpy as np
import matplotlib.pyplot as plt

import argparse


def train_ae(name, n_epochs, type, noise=False, dataset_name="mnist", save=True):
    
    num_channels = 1 if dataset_name == "mnist" else 3
    img_side = 28 if dataset_name == "mnist" else 32
    
    
    if type == "ae":
        model = AutoEncoder(num_channels, img_side, 64)
    elif type == "conv":
        model = ConvAutoencoder(num_channels)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCELoss()
    
    dataset = get_mnist(batch_size=64, shuffle=True) if dataset_name == "mnist" else get_cifar(batch_size=64, shuffle=True)
    
    device = get_device()
    
    model.to(device)
    
    losses = np.zeros(n_epochs)
    
    pbar = tqdm(range(n_epochs))
    
    for epoch in pbar:
        for x, _ in dataset:
            
            x = x.to(device)
            
            if noise:
                x += 0.1 * x.std() * torch.randn_like(x).to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            
            losses[epoch] += loss.item()
            
        pbar.set_description(f"Loss: {losses[epoch]}")
    

    os.makedirs(f"results/{name}", exist_ok=True)
    
    if save:
        torch.save(model.state_dict(), f"results/{name}/model.pth") 
    
    print("Training complete")

    plt.plot(losses)
    plt.title("Loss")
    plt.savefig(f"results/{name}/loss.png")
    plt.close()
    
if __name__ == '__main__':
    
    # python3 train.py -m ae -n 30 -na ae1 -no -d cifar
    # trains an autoencoder for 30 epochs, with noise, and saves the model in results/ae1
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="ae", help="Model to train", 
                        choices=["ae", "vae", "conv"])
    
    parser.add_argument("--n_epochs", "-n", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--name", "-na", type=str, help="Name of the model")
    parser.add_argument("--noise", "-no", action="store_true", help="Add noise to the input") # false by default
    parser.add_argument("--dataset", "-d", type=str, default="cifar", help="Dataset to use", choices=["mnist", "cifar"])
    
    args = parser.parse_args()
    
    if args.model in ["ae", "conv"]:
        train_ae(
            name=args.name,
            n_epochs=args.n_epochs,
            type=args.model,
            noise=args.noise,
            dataset_name=args.dataset
        )

    else:
        print("Not implemented")
        
        
    print("Done")