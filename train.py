import torch
from tqdm import tqdm
from model import CondVariationalAutoEncoder, AutoEncoder, ConvAutoencoder
from dataset import get_mnist
import os
from utils import get_device, KL_divergence

import numpy as np
import matplotlib.pyplot as plt

import argparse


def train_ae(name, n_epochs, save=True, noise=True):
    model = AutoEncoder(28*28, 64) # 28*28 is the size of the MNIST images, 64 is the size of the hidden layer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCELoss()
    
    dataset = get_mnist(batch_size=64, shuffle=True)
    
    device = get_device()
    
    model.to(device)
    
    losses = np.zeros(n_epochs)
    
    pbar = tqdm(range(n_epochs))
    
    for epoch in pbar:
        for x, _ in dataset:
            x = x.view(-1, 28*28).to(device)
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
    
def train_convae(name, n_epochs, save=True, noise=True):
    model = ConvAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCELoss()
    
    dataset = get_mnist(batch_size=64, shuffle=True)
    
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

def train_vae(name, n_epochs, final_lambda_kl=0.5):
    model = CondVariationalAutoEncoder(data_size=28*28, layer_size=196, hidden_size=64, num_classes=10)
    # 28*28 is the size of the MNIST images, 
    # 196 is the size of the intermediate layer
    # 64 is the size of the hidden layer (latent space)
    # 10 is the number of classes in the dataset (0-9)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    
    dataset = get_mnist(batch_size=64, shuffle=True)
    
    device = get_device()
    
    model.to(device)

    
    pbar = tqdm(range(n_epochs))
    
    losses = np.zeros(n_epochs)
    
    for epoch in pbar:
        
        # save the mean/logvar of each class
        mu_tensor = torch.zeros(10, 64).to(device)
        logvar_tensor = torch.zeros(10, 64).to(device)
        n_instances = torch.zeros(10).to(device)
    
        for x, labels in dataset:
            
            x = x.view(-1, 28*28).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output, mu, logvar = model(x, labels)
            loss = criterion(output, x) + (epoch/n_epochs) * final_lambda_kl * KL_divergence(mu, logvar)
            loss.backward()
            optimizer.step()
            
            losses[epoch] += loss.item()
            
            
            for i in range(10):
                selected = labels == i
                mu_tensor[i] += mu[selected].sum(dim=0)
                logvar_tensor[i] += logvar[selected].sum(dim=0)
                n_instances[i] += selected.sum()
        
        
        pbar.set_description(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # [10, 64] / [10, 1] -> [10, 64]
    mean_mu = mu_tensor / n_instances.view(-1, 1) 
    mean_logvar = logvar_tensor / n_instances.view(-1, 1)
    
    to_save = {
        "mu": mean_mu,
        "logvar": mean_logvar
    }
    
    os.makedirs(f"results/{name}", exist_ok=True)
    
    torch.save(to_save, f"results/{name}/params.pth")
    torch.save(model.state_dict(), f"results/{name}/model.pth")
    
    print("Training complete")
    
    plt.plot(losses)
    plt.title("Loss")
    plt.savefig(f"results/{name}/loss.png")
    plt.close()
    
if __name__ == '__main__':
    
    # python train.py -m ae -n 30 -na ae1 -no 
    # trains an autoencoder for 30 epochs, with noise, and saves the model in results/ae1
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="ae", help="Model to train", 
                        choices=["ae", "vae", "conv"])
    
    parser.add_argument("--n_epochs", "-n", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--name", "-na", type=str, help="Name of the model")
    parser.add_argument("--noise", "-no", action="store_true", help="Add noise to the input") # false by default
    
    args = parser.parse_args()
    
    if args.model == "ae":
        train_ae(args.name, args.n_epochs, noise=args.noise)
    elif args.model == "conv":
        train_convae(args.name, args.n_epochs, noise=args.noise)
    elif args.model == "vae":
        train_vae(args.name, args.n_epochs)
    else:
        print("Not implemented")
        
        
    print("Done")