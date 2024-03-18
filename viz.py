import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA


def visualize(embeddings, labels, path, mode="tsne"):
    
    if mode == "pca":
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(embeddings)
    elif mode == "mds":
        mds = MDS(n_components=2, random_state=0, n_jobs=-1)
        embeddings = mds.fit_transform(embeddings)
    elif mode == "tsne":
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