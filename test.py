from dataset import get_mnist
from model import AutoEncoder, ConvAutoencoder, CondVariationalAutoEncoder, VariationalAutoEncoder
import torch
from utils import get_device
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
from tqdm import tqdm
from viz import visualize

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
            plt.close()
            
        
def test_variational_auto_encoder(type="cond",name="vae"):
    
    device = get_device()
    
    if type == "vae":
        model = VariationalAutoEncoder(data_size=28*28, layer_size=196, hidden_size=64)
    elif type == "cond":
        model = CondVariationalAutoEncoder(data_size=28*28, layer_size=196, hidden_size=64, num_classes=10)
    
    model.load_state_dict(torch.load(f'results/{name}/model.pth'))
    
    model.to(device)
    model.eval()
    
    params = torch.load(f'results/{name}/params.pth')
    mu = params["mu"].to(device)
    logvar = params["logvar"]
    
    if type == "cond":
        with torch.inference_mode():
            for i in range(5):  
                number = random.randint(0, 9) # random number between 0 and 9
                
                mu_c = mu[number].view(1, -1).to(device)
                logvar_c = logvar[number].view(1, -1).to(device)
                c = torch.tensor(number).unsqueeze(0).to(device)
                
                z = mu_c + torch.exp(0.5*logvar_c) * torch.randn_like(mu_c)
                
                output = model.generate(z, c).cpu().numpy().reshape(28, 28)
                
                plt.figure()
                plt.imshow(output, cmap='coolwarm')
                plt.title(f"Generated {number}")
                plt.savefig(f"results/{name}/generated_{i}.png")
                plt.close()
    
      
    mnist = get_mnist(loader=False) # dataset without loader
    embeddings = torch.zeros(len(mnist), 64).to(device)
    labels = np.zeros(len(mnist))
    
    with torch.inference_mode():
        for i, (x, label) in tqdm(enumerate(mnist), total=len(mnist)):
            x = x.view(-1, 28*28).to(device)
            mu, _ = model.encode(x)
            embeddings[i] = mu.detach().cpu()
            labels[i] = label
            
    
    embeddings = embeddings.cpu().numpy()
    
    visualize(embeddings, labels, f"results/{name}/embeddings.png", mode="pca")
            
    
            
    
if __name__ == '__main__':
    
    # python3 test.py -m ae -n ae_noisy
    # python3 test.py -m ae -n ae
    
    # python3 test.py -m conv -n conv_noisy
    # python3 test.py -m conv -n conv
    
    # python3 test.py -m vae -n vae
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", "-n", type=str, default="ae", help="Name of the model to test")
    parser.add_argument("--model", "-m", type=str, default="ae", help="Type of model to test", choices=["ae", "conv", "vae"])
    
    args = parser.parse_args()
    
    if args.model in ["ae", "conv"]:
        test_auto_encoder(args.type, args.name)
    elif args.model == "vae":
        test_variational_auto_encoder(args.name)
    