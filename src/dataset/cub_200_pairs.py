from src.dataset.cub_200 import CUBDataset, CUBDatasetBordersCut, CUBDatasetAugmented
from src.dataset.pairwise_utils import random_pairs_2
from src.dataset.random_utils import use_seed
import torch
import numpy as np

def CUBPairDataset(args, **kwargs):
    if args.sup == 'sup_original':
        if args.borders_cut:
            return CUBPairDatasetOrigBC(args, **kwargs)
        else:
            return CUBPairDatasetOrig(args, **kwargs)
    elif args.sup == 'sup_augmented':
        return CUBPairDatasetAugmented(args, **kwargs)
    else:
        raise ValueError(f"Unknown supervision type {args.sup}")
                
class CUBPairDatasetOrig(CUBDataset):
    pck_symm = True

    def __init__(self, args, **kwargs):
        # init the parent
        super().__init__(args, **kwargs)
        idx0, idx1 = random_pairs_2(args.n_pairs, len(self.data))
        self.pairs = list(zip(idx0, idx1))
        self.return_imgs = False
        self.return_masks = False
        assert(args.borders_cut == False)
        
    def __len__(self):
        return len(self.pairs)
    
    def get_imgs(self, idx):
        idx0, idx1 = self.pairs[idx]
        img0 = super().get_img(idx0)
        img1 = super().get_img(idx1)
        return img0, img1
    
    def get_masks(self, idx):
        idx0, idx1 = self.pairs[idx]
        mask0 = super().get_mask(idx0)
        mask1 = super().get_mask(idx1)
        return mask0, mask1
    
    def __getitem__(self, idx):
        idx0, idx1 = self.pairs[idx]
        data0 = super().__getitem__(idx0)
        data1 = super().__getitem__(idx1)
        data = {'src_imsize': data0['imsize'],
                'trg_imsize': data1['imsize'],
                'src_kps': data0['kps'],
                'trg_kps': data1['kps'],
                'trg_kps_symm_only': data1['kps_symm_only'],
                'src_kps_symm_only': data0['kps_symm_only'],
                'src_bndbox': data0['bndbox'],
                'trg_bndbox': data1['bndbox'],
                'cat': self.cat,
                'idx': idx}
                        
        data['numkp'] = data0['kps'].shape[0]
        if self.return_feats:
            data['src_ft'] = data0['ft']
            data['trg_ft'] = data1['ft']
        if self.return_imgs:
            img0,img1 = self.get_imgs(idx)
            data['src_img'] = np.array(img0)
            data['trg_img'] = np.array(img1)
        if self.return_masks:
            mask0,mask1 = self.get_masks(idx)
            data['src_mask'] = mask0
            data['trg_mask'] = mask1
        return data

class CUBPairDatasetOrigBC(CUBDatasetBordersCut):
    pck_symm = True

    def __init__(self, args, **kwargs):
        # init the parent
        super().__init__(args, **kwargs)
        idx0, idx1 = random_pairs_2(args.n_pairs, len(self.data))
        self.pairs = list(zip(idx0, idx1))
        self.return_imgs = False
        self.return_masks = False
        
    def __len__(self):
        return len(self.pairs)
    
    def get_imgs(self, idx):
        idx0, idx1 = self.pairs[idx]
        img0 = super().get_img(idx0)
        img1 = super().get_img(idx1)
        return img0, img1
    
    def __getitem__(self, idx):
        idx0, idx1 = self.pairs[idx]
        data0 = super().__getitem__(idx0)
        data1 = super().__getitem__(idx1)
        data = {'src_imsize': data0['imsize'],
                'trg_imsize': data1['imsize'],
                'src_kps': data0['kps'],
                'trg_kps': data1['kps'],
                'trg_kps_symm_only': data1['kps_symm_only'],
                'src_kps_symm_only': data0['kps_symm_only'],
                'src_bndbox': data0['bndbox'],
                'trg_bndbox': data1['bndbox'],
                'cat': self.cat,
                'idx': idx}
        if self.return_feats:
            data['src_ft'] = data0['ft']
            data['trg_ft'] = data1['ft']
        if self.return_imgs:
            img0,img1 = self.get_imgs(idx)
            data['src_img'] = np.array(img0)
            data['trg_img'] = np.array(img1)
        data['numkp'] = data0['kps'].shape[0]
        if self.return_masks:
            mask0,mask1 = self.get_masks(idx)
            data['src_mask'] = mask0
            data['trg_mask'] = mask1
        return data

class CUBPairDatasetAugmented(CUBDatasetAugmented):
    pck_symm = True

    def __init__(self, args, **kwargs):
        # init the parent
        super().__init__(args, **kwargs)
        idx0, idx1 = random_pairs_2(args.n_pairs, len(self.data))
        self.pairs = list(zip(idx0, idx1))
        self.seed_pairs = 1 # this can be changed for different epochs
        self.return_imgs = False
        self.return_masks = False
        
    def __len__(self):
        return len(self.pairs)
    
    def get_imgs(self, idx):
        idx0, idx1 = self.pairs[idx]
        with use_seed(idx+self.seed_pairs):
            seed_offset0 = torch.randint(1000000, (1,)).item()
            seed_offset1 = torch.randint(1000000, (1,)).item()
        self.seed = self.seed+seed_offset0
        img0 = super().get_img(idx0)
        self.seed = self.seed-seed_offset0
        self.seed = self.seed+seed_offset1
        img1 = super().get_img(idx1)
        self.seed = self.seed-seed_offset1
        return img0, img1
    
    def get_masks(self, idx):
        idx0, idx1 = self.pairs[idx]
        with use_seed(idx+self.seed_pairs):
            seed_offset0 = torch.randint(1000000, (1,)).item()
            seed_offset1 = torch.randint(1000000, (1,)).item()
        self.seed = self.seed+seed_offset0
        mask0 = super().get_mask(idx0)
        self.seed = self.seed-seed_offset0
        self.seed = self.seed+seed_offset1
        mask1 = super().get_mask(idx1)
        self.seed = self.seed-seed_offset1
        return mask0, mask1
    
    def __getitem__(self, idx):
        idx0, idx1 = self.pairs[idx]
        with use_seed(idx+self.seed_pairs):
            seed_offset0 = torch.randint(1000000, (1,)).item()
            seed_offset1 = torch.randint(1000000, (1,)).item()
        self.seed = self.seed+seed_offset0
        data0 = super().__getitem__(idx0)
        self.seed = self.seed-seed_offset0
        self.seed = self.seed+seed_offset1
        data1 = super().__getitem__(idx1)
        self.seed = self.seed-seed_offset1
        data = {'src_imsize': data0['imsize'],
                'trg_imsize': data1['imsize'],
                'src_kps': data0['kps'],
                'trg_kps': data1['kps'],
                'trg_kps_symm_only': data1['kps_symm_only'],
                'src_kps_symm_only': data0['kps_symm_only'],
                'src_bndbox': data0['bndbox'],
                'trg_bndbox': data1['bndbox'],
                'src_hflip': data0['hflip'],
                'trg_hflip': data1['hflip'],
                'cat': self.cat,
                'idx': idx}
                        
        data['numkp'] = len(self.KP_NAMES)

        if self.return_feats:
            data['src_ft'] = data0['ft']
            data['trg_ft'] = data1['ft']
        if self.return_imgs:
            img0,img1 = self.get_imgs(idx)
            data['src_img'] = np.array(img0)
            data['trg_img'] = np.array(img1)
        if self.return_masks:
            mask0,mask1 = self.get_masks(idx)
            data['src_mask'] = mask0
            data['trg_mask'] = mask1
        return data
