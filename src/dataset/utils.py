
import torch
import numpy as np
import torch.nn.functional as F
import copy
from torch.utils.data import ConcatDataset

@torch.no_grad()
def get_feats(dataset, idx, model_refine=None, only_fg=False, feat_interpol=False, prt_interpol=True):
    try:
        ft = dataset[idx]['ft'][None]
    except:
        ft = dataset._get_feat(idx, dataset.featurizer, dataset.featurizer_kwargs)
    if model_refine is not None:
        ft = model_refine(ft)
    feats = ft.permute(0,2,3,1).flatten(0,-2)# B, C, H, W -> H*W, C

    if 'parts_mask' in dataset[idx].keys():
        prt = dataset[idx]['parts_mask'][None]
        ftsize = torch.tensor(ft.shape[-2:])
        prtsize = torch.tensor(prt.shape[-2:])
        if feat_interpol:
            ft_interp = F.interpolate(ft, size=prtsize.tolist(), mode='bilinear', align_corners=False)
            feats = ft_interp.permute(0,2,3,1).flatten(0,-2)# B, C, H, W -> H*W, C
        if prt_interpol:
            prt_interp = F.interpolate(prt, size=ftsize.tolist(), mode='bilinear', align_corners=False)
            parts = prt_interp.permute(0,2,3,1).flatten(0,-2)# B, C, H, W -> H*W, num_parts
        else:
            parts = prt.permute(0,2,3,1).flatten(0,-2)

        if only_fg:
            mask = parts.sum(-1)>0
            feats = feats[mask]
            parts = parts[mask]

    elif 'kps' in dataset[idx].keys() and only_fg:

        # Extract keypoint information
        kp = dataset[idx]['kps']
        imgsize = dataset[idx]['imsize']


        # Filter keypoints based on visibility
        visible_mask = kp[:, 2] > 0.5
        kp_ = kp[visible_mask, :2]

        # Normalize keypoint coordinates to range [-1, 1] for grid sampling
        h, w = ft.shape[2:]
        kp_normalized = torch.stack([
            2 * kp_[:, 1] / (imgsize[1] - 1) - 1,
            2 * kp_[:, 0] / (imgsize[0] - 1) - 1
        ], dim=-1).unsqueeze(0).unsqueeze(0).to(ft.device)  # Shape: (1, num_kps, 2)

        # Use grid_sample to directly sample features at keypoint locations
        # Ensure kp_normalized and ft are on the same device
        feats = F.grid_sample(ft, kp_normalized, mode='bilinear', align_corners=False)
        feats = feats.squeeze(0).squeeze(1).t()  # Shape: (num_vis_kps, C)

        parts = torch.eye(kp.shape[0])[visible_mask] # num_vis_kps, num_parts
    else:
        parts = None

    return feats, parts

@torch.no_grad()
def get_featpairs(dataset, idx, model_refine=None, only_fg=False, feat_interpol=False, prt_interpol=True):
    try:
        ft0 = dataset[idx]['src_ft'][None]
        ft1 = dataset[idx]['trg_ft'][None]
    except:
        raise ValueError('Need to have src_ft and tgt_ft in dataset')
    if model_refine is not None:
        ft0 = model_refine(ft0)
        ft1 = model_refine(ft1)
    feats = [ft0.permute(0,2,3,1).flatten(0,-2), ft1.permute(0,2,3,1).flatten(0,-2)]# B, C, H, W -> H*W, C
    parts = []
    for i,(ft, prefix) in enumerate(zip([ft0, ft1], ['src', 'trg'])):
        if  prefix+'_kps' in dataset[idx].keys():

            kp = dataset[idx][prefix+'_kps']
            imgsize = dataset[idx][prefix+'_imsize']

            # Filter keypoints based on visibility
            visible_mask = kp[:, 2] > 0.5
            kp_ = kp[visible_mask, :2]

            # Normalize keypoint coordinates to range [-1, 1] for grid sampling
            h, w = ft.shape[2:]
            kp_normalized = torch.stack([
                2 * kp_[:, 1] / (imgsize[1] - 1) - 1,
                2 * kp_[:, 0] / (imgsize[0] - 1) - 1
            ], dim=-1).unsqueeze(0).unsqueeze(0).to(ft.device)  # Shape: (1, num_kps, 2)

            # Use grid_sample to directly sample features at keypoint locations
            # Ensure kp_normalized and ft are on the same device
            feats[i] = F.grid_sample(ft, kp_normalized, mode='bilinear', align_corners=False)
            feats[i] = feats[i].squeeze(0).squeeze(1).t()  # Shape: (num_vis_kps, C)

            # free up memory
            torch.cuda.empty_cache()
            parts.append(torch.eye(kp.shape[0])[visible_mask])
        else:
            parts.append(None)

    return feats[0], parts[0], feats[1], parts[1]

@torch.no_grad()
def get_init_feats_and_labels(dataset, N, model_refine=None, feat_interpol=False, prt_interpol=True, only_fg=False):
    # use seeds to make sure the same samples are used for all models
    np.random.seed(0)
    indices = np.random.permutation(np.arange(0, len(dataset)))[:N]
    feats = []
    parts = []
    for idx in indices:
        feats_idx, parts_idx = get_feats(dataset, idx, model_refine=model_refine, only_fg=only_fg, feat_interpol=feat_interpol, prt_interpol=prt_interpol)
        feats.append(feats_idx)
        parts.append(parts_idx)
    feats = torch.cat(feats)
    parts = torch.cat(parts)
    return feats, parts

def to_flattened_idx_torch(x, y, x_width, y_width):
    idx = (y.round() + x.round()*y_width).int()
    x_, y_ = torch.unravel_index(idx, (x_width, y_width))
    # assert that all the indices are the original ones (up to 0.5)
    assert torch.all(torch.abs(x_-x)<0.5)
    assert torch.all(torch.abs(y_-y)<0.5)
    return idx


def get_multi_cat_dataset(dataset_list, featurizer, featurizer_kwargs, cat_list_=None, model_seg_name=None):
    datasets = []
    for dataset in dataset_list:
        cat_list = dataset.all_cats if cat_list_ is None else cat_list_
        for cat in cat_list:
            dataset.padding_kps = True
            dataset.init_kps_cat(cat)
            datasets.append(copy.deepcopy(dataset))
            datasets[-1].featurizer = featurizer
            datasets[-1].featurizer_kwargs = featurizer_kwargs
            datasets[-1].model_seg_name = model_seg_name
            datasets[-1].return_masks = True
    return ConcatDataset(datasets)

# def get_multi_cat_dataset(dataset_train, cat_list, featurizer, featurizer_kwargs):
#     datasets = []
#     for cat in cat_list:
#         dataset_train.padding_kps = True
#         dataset_train.init_kps_cat(cat)
#         datasets.append(copy.deepcopy(dataset_train))
#         datasets[-1].featurizer = featurizer
#         datasets[-1].featurizer_kwargs = featurizer_kwargs
#     return ConcatDataset(datasets)

