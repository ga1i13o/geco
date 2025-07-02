import torch

from src.dataset.utils import get_init_feats_and_labels
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from src.models.classifier.supervised.nearest_centroid import Nearest_Centroid_fg_Classifier

def train_classifier(args_seg, dataset_train, model_refine=None):

    if args_seg.model == 'nearest_centroid_fg':
        feats, y_mat = get_init_feats_and_labels(dataset_train, args_seg.num_samples, only_fg=True, model_refine=model_refine)
        model_seg = Nearest_Centroid_fg_Classifier(args_seg.num_pcaparts, feats, y_mat)

    return model_seg

def forward_classifier(dataset, idx, model_seg, model_refine=None):
    data = dataset[idx]

    # get the part segmentation
    imsize = data['imsize']
    if model_seg.name in ['nearest_centroid_fg']:
        ft = data['ft'][None].to(device) # B, C, H, W
        if model_refine is not None:
            ft = model_refine(ft)
        ft_interp = F.interpolate(ft, size=imsize.tolist(), mode='bilinear', align_corners=False)
        prt = model_seg(ft_interp.permute(0,2,3,1).flatten(0,-2)) # B, C, H, W -> H*W, C
        prt = prt.reshape(ft.shape[0], imsize[0], imsize[1], -1).permute(0,3,1,2) # H*W, C -> B, C, H, W

    return prt