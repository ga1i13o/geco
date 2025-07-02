from src.models.pca import compute_pca
from sklearn.neighbors import NearestCentroid
import torch
import torch.nn as nn

class Nearest_Centroid_fg_Classifier(nn.Module):
    name = 'nearest_centroid_fg'
    def __init__(self, n_components_pca, x, y_mat):
        '''
        Input:
            n_components_pca: int, number of components for PCA
            x: torch.tensor, shape (B, feature_dim), feature vectors
            y_mat: torch.tensor, shape (B, num_parts), part labels, one-hot encoding, 
            no background part, as we assume to only receive foreground features
        '''
        super().__init__()
        feature_dim = x.shape[-1]
        x = x.reshape(-1,feature_dim)
        x = x/x.norm(dim=-1, keepdim=True)
        self.part_components = compute_pca(x, n_components_pca)
        x_proj = torch.mm(x, self.part_components.t())
        self.classifier = NearestCentroid(metric="manhattan")
        self.num_parts = y_mat.shape[-1]
        y = y_mat.argmax(-1)
        self.parts = torch.unique(y)
        # assert(len(self.parts) == self.num_parts) # check if all parts are present
        x_proj = x_proj/x_proj.norm(dim=-1, keepdim=True)
        self.classifier.fit(x_proj.cpu().numpy(), y.cpu().numpy())
        self.prototypes_proj = torch.tensor(self.classifier.centroids_, device=self.part_components.device, dtype=self.part_components.dtype)    # num_parts, n_components_pca
        self.prototypes = torch.mm(self.prototypes_proj, self.part_components)

    def get_prototypes(self):
        if len(self.parts) == self.num_parts:
            return self.prototypes
        else:
            prototypes = torch.zeros(self.num_parts, self.prototypes.shape[-1], device=self.prototypes.device, dtype=self.prototypes.dtype)
            prototypes[self.parts] = self.prototypes
            return prototypes
        
    def forward(self, x):
        '''
        Input:
            x: torch.tensor, shape (B, feature_dim), feature vectors
        Output:
            y_mat: torch.tensor, shape (B, num_parts), part labels, one-hot encoding
            probably neg values for background, but not trained for that
        '''
        feature_dim = x.shape[-1]
        # get the scalar product of the input with the principal components
        x = x.reshape(-1,feature_dim)
        x_proj = torch.mm(x, self.part_components.t())
        x_proj = x_proj/x_proj.norm(dim=-1, keepdim=True)
        prototypes_proj = self.prototypes_proj/self.prototypes_proj.norm(dim=-1, keepdim=True)
        y_mat = torch.zeros((x.shape[0],self.num_parts)).to(x.device) # B, num_parts
        y_mat_ = torch.mm(x_proj, prototypes_proj.t())
        y_mat[:, self.parts] = y_mat_
        return y_mat
