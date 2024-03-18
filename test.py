from dataset import get_mnist
import os
from model import AutoEncoder, ConvAutoencoder, VariationalAutoEncoder
import torch
from utils import get_device
import matplotlib.pyplot as plt

import argparse

def test_auto_encoder(type="ae", name="ae"):
    
    
    if type == "ae":
        model = AutoEncoder(28*28, 64)
    elif type == "conv":
        model = ConvAutoencoder()
        
        
    
    model.load_state_dict(torch.load(f'results/{name}/model.pth'))

    device = get_device()
    
    model.to(device)

    mnist = get_mnist(loader=False) # dataset without loader
    
    with torch.no_grad():
        for i in range(5):  
            
            if type == "ae":
                img = mnist[i][0].view(-1, 28*28).to(device)
                
                
            
            
            elif type == "conv":
                img = mnist[i][0].view(-1, 1, 28, 28).to(device)
                
            if "nois" in name: # noisy or noise in the name
                img += 0.1 * img.std() * torch.randn_like(img).to(device)
            
            output = model(img).cpu().numpy().reshape(28, 28)
            
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img.cpu().numpy().reshape(28, 28), cmap='coolwarm')
            plt.title("Original")
            
            plt.subplot(1, 2, 2)
            plt.imshow(output, cmap='coolwarm')
            plt.title("Reconstructed")
            
            plt.savefig(f"results/{name}/reconstructed_{i}.png")
            
        
def test_variational_auto_encoder(name="vae"):
    
    device = get_device()
    model = VariationalAutoEncoder(28*28, 196, 64).to(device)
    model.load_state_dict(torch.load(f'results/{name}/model.pth'))
    
    
    
    params = torch.load(f'results/{name}/params.pth')
    mu = params["mu"].to(device)
    #logvar = params["logvar"]
    
    with torch.no_grad():
        for i in range(10):
            output = model.decoder(mu[i]).cpu().numpy().reshape(28, 28)
            
            plt.figure()
            plt.imshow(output, cmap='coolwarm')
            plt.title(f"Reconstructed {i}")
            plt.savefig(f"results/{name}/reconstructed_{i}.png")
    
if __name__ == '__main__':
    
    # python3 test.py -t ae -n ae_noisy
    # python3 test.py -t ae -n ae
    
    # python3 test.py -t conv -n conv_noisy
    # python3 test.py -t conv -n conv
    
    # python3 test.py -t vae -n vae
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", "-n", type=str, default="ae", help="Name of the model to test")
    parser.add_argument("--model", "-m", type=str, default="ae", help="Type of model to test", choices=["ae", "conv", "vae"])
    
    args = parser.parse_args()
    
    if args.model in ["ae", "conv"]:
        test_auto_encoder(args.type, args.name)
    elif args.model == "vae":
        test_variational_auto_encoder(args.name)
    