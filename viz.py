import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from dataset import get_mnist, get_cifar
import argparse
import torch
from model import AutoEncoder, ConvAutoencoder, VariationalAutoEncoder
import numpy as np
from tqdm import tqdm

def visualize(embeddings, labels, path, mode="tsne"):
    
    if mode == "pca":
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(embeddings)
    elif mode == "mds":
        assert len(embeddings) <= 5000, "MDS is slow for large number of samples, use PCA or t-SNE"
        mds = MDS(n_components=2, random_state=0, n_jobs=-1)
        embeddings = mds.fit_transform(embeddings)
    elif mode == "tsne":
        assert len(embeddings) <= 10000, "t-SNE is slow for large number of samples, use PCA or MDS"
        tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
        embeddings = tsne.fit_transform(embeddings)
    else:
        raise ValueError(f"Invalid mode {mode}")
    
    plt.figure()
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', alpha=0.3, s=2)
    plt.colorbar()
    plt.title(f"{mode} embeddings")
    plt.savefig(path)
    plt.close()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # python3 viz.py --name "vae_long" --type "vae" --dataset "cifar" -n 1000
    
    # python3 viz.py --name "ae" --type "ae" --dataset "cifar" -n 1000
    
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--n_samples", "-n", type=int, default=1000)
    parser.add_argument("--visualization","-v", type=str, choices=["pca", "mds", "tsne"], default="tsne")
    
    args = parser.parse_args()
    
    if args.dataset == "mnist":
        dataset = get_mnist(loader=False)
    else:
        dataset = get_cifar(type="eval", loader=False)
    
    idx = torch.randperm(len(dataset))[:args.n_samples]
    
    num_channels = 1 if args.dataset == "mnist" else 3
    img_side = 28 if args.dataset == "mnist" else 32
    
    if args.type == "ae":
        model = AutoEncoder(num_channels, img_side, 100)
    elif args.type == "conv":
        model = ConvAutoencoder(num_channels)
    elif args.type == "vae":
        model = VariationalAutoEncoder(num_channels, img_side, 100)
        
    model.load_state_dict(torch.load(f"results/{args.name}/model.pth", map_location="cpu"))
    model.eval()
    
    if args.type == "vae" or args.type == "ae":

        embeddings = torch.zeros(args.n_samples, 100)
    else: # conv
        embeddings = torch.zeros(args.n_samples, 512)
    labels = np.zeros(args.n_samples)
    
    for i, idx in enumerate(tqdm(idx)):
        img, label = dataset[idx]
        img = img.unsqueeze(0)
        if args.type == "ae":
            embeddings[i] = model.encode(img).squeeze()
        elif args.type == "conv":
            x = model.encoder(img)
            x = x.view(x.size(0), -1)
            embeddings[i] = x.squeeze()
        else:
            mu, _ = model.encode(img)
            embeddings[i] = mu.squeeze()
            
        labels[i] = label
    
    embeddings = embeddings.detach().numpy()
    labels = labels.astype(int)
    
    visualize(embeddings, labels, f"results/{args.name}/embeddings.png", mode=args.visualization)
    
    
    
        
    