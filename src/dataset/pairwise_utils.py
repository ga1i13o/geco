import torch
from src.dataset.random_utils import use_seed
import numpy as np
from src.dataset.utils import to_flattened_idx_torch
import copy
def random_pairs(n_pairs,n_imgs,permute_first=True):
    if permute_first:
        idx0 = torch.randint(n_imgs, (n_pairs,))
    else:
        idx0 = torch.arange(n_imgs).repeat(n_pairs//n_imgs)
        idx0 = idx0[:n_pairs]
    idx1 = torch.randint(n_imgs, (n_pairs,))
    for n in range(n_pairs):
        while idx0[n]==idx1[n]:
            idx1[n] = torch.randint(n_imgs, (1,))
    return idx0, idx1

def random_pairs_2(n_pairs, n_imgs):
    idx0, idx1 =[], []
    while len(idx0) < n_pairs:
        with use_seed(len(idx0) + 123):
            indices = np.random.permutation(np.arange(0, n_imgs))
        middle = n_imgs // 2
        idx0, idx1 = idx0 + indices[:middle].tolist(), idx1 + indices[-middle:].tolist()
    idx0, idx1 = idx0[:n_pairs], idx1[:n_pairs]
    return idx0, idx1

def get_pos_pairs(data_in, b=0):
    src_kps_symm_only = data_in['src_kps_symm_only'][b]
    trg_kps_symm_only = data_in['trg_kps_symm_only'][b]
    src_kps = data_in['src_kps'][b]
    trg_kps = data_in['trg_kps'][b]
    vis_both = torch.bitwise_and(src_kps[:,2] > 0.5, trg_kps[:,2] > 0.5)
    pos_src_kps = src_kps[vis_both]
    pos_trg_kps = trg_kps[vis_both]
    flag_11_src = torch.bitwise_and(src_kps_symm_only[:,2] > 0.5, src_kps[:,2] > 0.5)
    flag_11_trg = torch.bitwise_and(trg_kps_symm_only[:,2] > 0.5, trg_kps[:,2] > 0.5)
    flags_11 = {'pos_src_11': flag_11_src[vis_both], 'pos_trg_11': flag_11_trg[vis_both]}
    return pos_src_kps, pos_trg_kps, flags_11

def get_neg_pairs(data_in, b=0):
    src_kps_symm_only = data_in['src_kps_symm_only'][b]
    trg_kps_symm_only = data_in['trg_kps_symm_only'][b]
    src_kps = data_in['src_kps'][b]
    trg_kps = data_in['trg_kps'][b]
    # match to other points in src
    vis_both = torch.bitwise_and(src_kps_symm_only[:,2] > 0.5, trg_kps[:,2] > 0.5)
    neg_src_kps = src_kps_symm_only.clone()
    neg_trg_kps = trg_kps.clone()
    neg_src_kps[:,2] = vis_both
    neg_trg_kps[:,2] = vis_both
    flag_11_src = torch.bitwise_and(src_kps_symm_only[:,2] > 0.5, src_kps[:,2] > 0.5)
    # match to other points in trg
    vis_both = torch.bitwise_and(src_kps[:,2] > 0.5, trg_kps_symm_only[:,2] > 0.5)
    neg_src_kps[vis_both] = src_kps[vis_both]
    neg_trg_kps[vis_both] = trg_kps_symm_only[vis_both]
    neg_trg_kps[:,2] = torch.bitwise_or(neg_src_kps[:,2] > 0.5, vis_both)
    flag_11_trg = torch.bitwise_and(trg_kps_symm_only[:,2] > 0.5, trg_kps[:,2] > 0.5)
    # only keep the vis_both keypoints
    flags_11 = {'neg_src_11': flag_11_src[neg_src_kps[:,2] > 0.5], 'neg_trg_11': flag_11_trg[neg_trg_kps[:,2] > 0.5]}
    neg_src_kps = neg_src_kps[neg_src_kps[:,2] > 0.5]
    neg_trg_kps = neg_trg_kps[neg_trg_kps[:,2] > 0.5]
    return neg_src_kps, neg_trg_kps, flags_11

def get_bin_pairs(data_in, b=0):
    src_kps_symm_only = data_in['src_kps_symm_only'][b]
    trg_kps_symm_only = data_in['trg_kps_symm_only'][b]
    src_kps = data_in['src_kps'][b]
    trg_kps = data_in['trg_kps'][b]
    occluded = torch.bitwise_xor(src_kps[:,2] == 0, trg_kps[:,2] == 0)
    vis_both = torch.bitwise_or(src_kps[:,2] > 0.5, trg_kps[:,2] > 0.5)
    bin_flag = torch.bitwise_and(occluded, vis_both)
    bin_src_kps = src_kps[bin_flag]
    bin_trg_kps = trg_kps[bin_flag]
    flag = bin_flag[bin_flag]
    flags_11 = {'bin_src_11': flag, 'bin_trg_11': flag}
    return bin_src_kps, bin_trg_kps, flags_11

def get_matches(data_in, b=0):
    pos_src_kps, pos_trg_kps, pos_flag_11 = get_pos_pairs(data_in, b)
    neg_src_kps, neg_trg_kps, neg_flag_11 = get_neg_pairs(data_in, b)
    bin_src_kps, bin_trg_kps, bin_flag_11 = get_bin_pairs(data_in, b)
    # put them in a dictionary
    data_out = {'pos_src_kps': pos_src_kps,
            'pos_trg_kps': pos_trg_kps,
            'neg_src_kps': neg_src_kps,
            'neg_trg_kps': neg_trg_kps,
            'bin_src_kps': bin_src_kps,
            'bin_trg_kps': bin_trg_kps}
    data_out.update(pos_flag_11)
    data_out.update(neg_flag_11)
    data_out.update(bin_flag_11)
    return data_out

def scale_to_feature_dims(data_out, data_in, b=0):
    scale_src = torch.tensor(data_in['src_ft'].shape[-2:])/data_in['src_imsize'][b]# get the scale for the feature image coordinates to the original image coordinates
    scale_trg = torch.tensor(data_in['trg_ft'].shape[-2:])/data_in['trg_imsize'][b]

    data_out_ft = copy.deepcopy(data_out)
    for prefix in ['pos', 'bin', 'neg']:
        src_kps = data_out[prefix+'_src_kps'].clone()
        trg_kps = data_out[prefix+'_trg_kps'].clone()
        # convert to feature image coordinates
        src_kps[:,:2] = (src_kps[:,:2]*scale_src).floor().long()
        trg_kps[:,:2] = (trg_kps[:,:2]*scale_trg).floor().long()
        data_out_ft[prefix+'_src_kps'] = src_kps
        data_out_ft[prefix+'_trg_kps'] = trg_kps
    return data_out_ft

def get_y_mat_gt_assignment(kp0, kp1, ft_size0, ft_size1):
    size = (ft_size0.prod()+1, ft_size1.prod()+1)
    if len(kp0) > 0:
        # enter dummy values for the keypoints that are not vis_both to avoid error in assert of flattened index function
        kp0[kp0[:,2]==0,0] = 0
        kp0[kp0[:,2]==0,1] = 0
        kp1[kp1[:,2]==0,0] = 0
        kp1[kp1[:,2]==0,1] = 0
        kp0_idx = to_flattened_idx_torch(kp0[:,0], kp0[:,1], ft_size0[0].item(), ft_size0[1].item())
        kp1_idx = to_flattened_idx_torch(kp1[:,0], kp1[:,1], ft_size1[0].item(), ft_size1[1].item())

        kp0_idx[kp0[:,2]==0] = ft_size0.prod() # bin idx
        kp1_idx[kp1[:,2]==0] = ft_size1.prod() # bin idx

        indices = torch.stack([kp0_idx, kp1_idx]).unique(dim=1, sorted=False)
        values = torch.ones_like(indices[0]).float()
        matrix  = torch.sparse_coo_tensor(indices, values, size).coalesce()
        
    else:
        # if no keypoints are annotated, return a matrix with only zeros
        matrix = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), [], size).coalesce()
    return matrix