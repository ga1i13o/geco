from sklearn.decomposition import PCA
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_pca(x, n_components):
    feature_dim = x.shape[-1]
    x = x.reshape(-1,feature_dim)
    pca = PCA(n_components=n_components).fit(x.cpu().numpy())
    components = pca.components_[None, ...]
    components = components.reshape(-1,feature_dim)
    return torch.tensor(components).to(device)


