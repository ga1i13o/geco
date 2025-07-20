from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from scipy.io import loadmat
import os
from src.dataset.random_utils import use_seed
import torch.nn.functional as F
from typing import Optional, Any
OBJECT_CLASSES = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
                  'chair','cow','table','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
def PFPascalPairs(args, **kwargs):
    if args.sup == 'sup_augmented':
        raise NotImplementedError("Augmented version not implemented, as L_R_Permutation is not available")
    elif args.sup == 'sup_original':
        return PFPascalPairsOrigPadded(args, **kwargs)
class PFPascalPairsOrig(TorchDataset):
    name = 'pfpascalpairs'
    pck_symm = False

    def __init__(self, args, split='test'):
        self.cat = None
        self.root = args.dataset_path
        self.save_path = args.save_path
        self.save_path_masks = args.save_path_masks
        self.split = split
        self.featurizer_name = None
        self.all_cats = OBJECT_CLASSES
        self.return_imgs = False
        self.return_masks = False
        self.return_feats = True
        # Initialize attributes that may be set externally
        self.featurizer: Optional[Any] = None
        self.featurizer_kwargs: Optional[Any] = None
        self.model_seg: Optional[Any] = None
        self.model_seg_name: Optional[str] = None

    def init_files(self):
        assert(self.cat is not None)
        data = pd.read_csv(f'{self.root}/{self.split}_pairs_pf_pascal.csv')
        cls_ids = data.iloc[:,2].values.astype("int") - 1
        cat_id = OBJECT_CLASSES.index(self.cat)
        subset_id = np.where(cls_ids == cat_id)[0]
        subset_pairs = data.iloc[subset_id,:]
        self.subset_pairs = subset_pairs

    def init_kps_cat(self, cat):
        self.cat = cat
        self.init_files()

    def __len__(self):
        assert(self.cat is not None)
        data = pd.read_csv(f'{self.root}/{self.split}_pairs_pf_pascal.csv')
        cls_ids = data.iloc[:,2].values.astype("int") - 1
        cat_id = OBJECT_CLASSES.index(self.cat)
        return len(np.where(cls_ids == cat_id)[0])
    
    def _get_img(self, idx):
        src_img_name = np.array(self.subset_pairs.iloc[:,0])[idx]
        img = Image.open(f'{self.root}/../{src_img_name}')
        return img
    
    def get_imgs(self, idx):
        # might get overwritten by pair dataset
        src_img_name = np.array(self.subset_pairs.iloc[:,0])[idx]
        trg_img_name = np.array(self.subset_pairs.iloc[:,1])[idx]
        
        img0 = Image.open(f'{self.root}/../{src_img_name}')
        img1 = Image.open(f'{self.root}/../{trg_img_name}')
        return img0, img1
    
    def get_feats(self, idx):
        try:
            src_img_name = str(self.subset_pairs.iloc[idx,0]).split('/')[-1].split('.')[0]
            trg_img_name = str(self.subset_pairs.iloc[idx,1]).split('/')[-1].split('.')[0]
            feat0 = self.get_feat(src_img_name)
            feat1 = self.get_feat(trg_img_name)
        except:
            assert(self.featurizer is not None)
            assert(self.featurizer_kwargs is not None)
            img0, img1 = self.get_imgs(idx)
            feat0 = self._get_feat(img0, self.featurizer, self.featurizer_kwargs)
            feat1 = self._get_feat(img1, self.featurizer, self.featurizer_kwargs)
        return feat0, feat1
    
    def get_feat(self, imname):
        assert(self.featurizer_name is not None)
        assert(self.cat is not None)
        path = os.path.join(self.save_path, self.name, self.featurizer_name, self.cat)
        feat = torch.load(os.path.join(path, imname+'.pth')).to(device)
        return feat
    
    def _get_feat(self, img, featurizer, featurizer_kwargs):
        cat = self.cat
        feat = featurizer.forward(img,
                                category=cat,
                                **featurizer_kwargs)
        return feat
    
    def store_feats(self, featurizer, overwrite, featurizer_kwargs):
        self.featurizer_name = featurizer.name
        for cat in self.all_cats:
            self.init_kps_cat(cat)
            path = os.path.join(self.save_path, self.name, self.featurizer_name, cat)
            if not os.path.exists(path):
                os.makedirs(path)
            for idx in range(len(self)):
                # might get overwritten by pair dataset
                src_img_name = np.array(self.subset_pairs.iloc[:,0])[idx].split('/')[-1].split('.')[0]
                trg_img_name = np.array(self.subset_pairs.iloc[:,1])[idx].split('/')[-1].split('.')[0]

                if not overwrite and os.path.exists(os.path.join(path, src_img_name+'.pth')):
                    continue
                img0, img1 = self.get_imgs(idx)
                feat0 = self._get_feat(img0, featurizer, featurizer_kwargs)
                torch.save(feat0.detach().cpu(), os.path.join(path, src_img_name+'.pth'))
                if not overwrite and os.path.exists(os.path.join(path, trg_img_name+'.pth')):
                    continue
                feat1 = self._get_feat(img1, featurizer, featurizer_kwargs)
                torch.save(feat1.detach().cpu(), os.path.join(path, trg_img_name+'.pth'))

    def get_masks(self, idx):
        src_img_name = str(self.subset_pairs.iloc[idx,0]).split('/')[-1].split('.')[0]
        trg_img_name = str(self.subset_pairs.iloc[idx,1]).split('/')[-1].split('.')[0]
        try:
            assert(self.model_seg_name is not None)
            assert(self.cat is not None)
            imname0, imname1 = src_img_name+'.pt', trg_img_name+'.pt'
            mask0 = torch.load(os.path.join(self.save_path_masks, self.name, self.model_seg_name, self.cat, imname0))
            mask1 = torch.load(os.path.join(self.save_path_masks, self.name, self.model_seg_name, self.cat, imname1))
        except:
            assert(self.model_seg is not None)
            img0, img1 = self.get_imgs(idx)
            data_out = self.getitem_wo_feat(idx)
            kps0 = data_out['src_kps']
            kps1 = data_out['trg_kps']
            # Alternative: use the following code to get the masks
            # https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_simple_demo.py
            mask0 = self.model_seg(img0, kps=kps0[kps0[:,2]==1])
            mask1 = self.model_seg(img1, kps=kps1[kps1[:,2]==1])
        # interpolate to 400 x 400
        mask0 = F.interpolate(mask0[None,None].float(), size=(400,400), mode='bilinear', align_corners=False)[0,0]
        mask1 = F.interpolate(mask1[None,None].float(), size=(400,400), mode='bilinear', align_corners=False)[0,0]
        mask0 = np.array(mask0)
        mask1 = np.array(mask1)
        return mask0, mask1
    
    def store_masks(self, overwrite):
        assert(self.model_seg is not None)
        print("saving all %s images' masks..."%self.split)
        path = os.path.join(self.save_path_masks, self.name, self.model_seg.name)
        for cat in tqdm(self.all_cats):
            self.init_kps_cat(cat)
            for idx_ in range(len(self)):

                src_img_name = np.array(self.subset_pairs.iloc[:,0])[idx_].split('/')[-1].split('.')[0]
                trg_img_name = np.array(self.subset_pairs.iloc[:,1])[idx_].split('/')[-1].split('.')[0]
                imname0, imname1 = src_img_name+'.pt', trg_img_name+'.pt'

                if not(os.path.exists(os.path.join(path, cat, imname0))) or not(os.path.exists(os.path.join(path, cat, imname1))) or overwrite:
                    prt0, prt1 = self.get_masks(idx_)
                    prt0, prt1 = torch.tensor(prt0), torch.tensor(prt1)

                if not(os.path.exists(os.path.join(path, cat, imname0)) and not overwrite):
                    torch.cuda.empty_cache()
                    os.makedirs(os.path.join(path, cat), exist_ok=True)
                    torch.save(prt0.cpu(), os.path.join(path, cat, imname0))
                if not(os.path.exists(os.path.join(path, cat, imname1)) and not overwrite):
                    torch.cuda.empty_cache()
                    os.makedirs(os.path.join(path, cat), exist_ok=True)
                    torch.save(prt1.cpu(), os.path.join(path, cat, imname1))
                    
    def process_kps_pascal(self, kps):
        # Step 1: Reshape the array to (20, 2) by adding nan values
        num_pad_rows = 20 - kps.shape[0]
        if num_pad_rows > 0:
            pad_values = np.full((num_pad_rows, 2), np.nan)
            kps = np.vstack((kps, pad_values))
            
        # Step 2: Reshape the array to (20, 3) 
        # Add an extra column: set to 1 if the row does not contain nan, 0 otherwise
        last_col = np.isnan(kps).any(axis=1)
        last_col = np.where(last_col, 0, 1)
        kps = np.column_stack((kps, last_col))

        # Step 3: Replace rows with nan values to all 0's
        mask = np.isnan(kps).any(axis=1)
        kps[mask] = 0

        return torch.tensor(kps).float()

    def getitem_wo_feat(self, idx):
        
        def get_points(point_coords_list, idx):
            X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
            Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
            Xpad = -np.ones(20)
            Xpad[: len(X)] = X
            Ypad = -np.ones(20)
            Ypad[: len(X)] = Y
            Zmask = np.zeros(20)
            Zmask[: len(X)] = 1
            point_coords = np.concatenate(
                (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1,20)), axis=0
            )
            # make arrays float tensor for subsequent processing
            point_coords = torch.Tensor(point_coords.astype(np.float32))
            return point_coords
                
        def read_mat(path, obj_name):
            r"""Reads specified objects from Matlab data file, (.mat)"""
            mat_contents = loadmat(path)
            mat_obj = mat_contents[obj_name]
            return mat_obj
        
        np.random.seed(42)
        # TODO: Find a way to get size w/o loading the image
        src_img_name = np.array(self.subset_pairs.iloc[:,0])[idx]
        trg_img_name = np.array(self.subset_pairs.iloc[:,1])[idx]
        src_fn= f'{self.root}/../{src_img_name}'
        trg_fn= f'{self.root}/../{trg_img_name}'
        src_size=Image.open(src_fn).size
        src_size = torch.tensor((src_size[1], src_size[0])) # width, height -> height, width (PIL uses width, height)
        trg_size=Image.open(trg_fn).size
        trg_size = torch.tensor((trg_size[1], trg_size[0])) # width, height -> height, width (PIL uses width, height)

        if not self.split.startswith('train'):
            point_A_coords = self.subset_pairs.iloc[:,3:5]
            point_B_coords = self.subset_pairs.iloc[:,5:]
            point_coords_src = get_points(point_A_coords, idx).transpose(1,0)
            point_coords_trg = get_points(point_B_coords, idx).transpose(1,0)
        else:
            assert(self.cat is not None)
            src_anns = os.path.join(self.root, 'Annotations', self.cat,
                                    os.path.basename(src_fn))[:-4] + '.mat'
            trg_anns = os.path.join(self.root, 'Annotations', self.cat,
                                    os.path.basename(trg_fn))[:-4] + '.mat'
            point_coords_src = self.process_kps_pascal(read_mat(src_anns, 'kps'))
            point_coords_trg = self.process_kps_pascal(read_mat(trg_anns, 'kps'))

        # change the order of the keypoints to match the x,y format
        point_coords_src[:, :2] = point_coords_src[:, :2].flip(1)
        point_coords_trg[:, :2] = point_coords_trg[:, :2].flip(1)
        # print(src_size)
        source_kps = point_coords_src # N, 3
        target_kps = point_coords_trg # N, 3

        x_min = source_kps[source_kps[:,2]==1][:,1].min().item()
        x_max = source_kps[source_kps[:,2]==1][:,1].max().item()
        y_min = source_kps[target_kps[:,2]==1][:,0].min().item()
        y_max = source_kps[target_kps[:,2]==1][:,0].max().item()
        # src_bndbox = np.array([0,0, src_size[1], src_size[0]])
        src_bndbox = np.array([x_min, y_min, x_max, y_max])
        x_min = target_kps[target_kps[:,2]==1][:,1].min().item()
        x_max = target_kps[target_kps[:,2]==1][:,1].max().item()
        y_min = target_kps[target_kps[:,2]==1][:,0].min().item()
        y_max = target_kps[target_kps[:,2]==1][:,0].max().item()
        trg_bndbox = np.array([0,0, trg_size[1], trg_size[0]])
        # trg_bndbox = np.array([0,0, trg_size[1], trg_size[0]])
        data = {
                'src_imsize': src_size,
                'trg_imsize': trg_size,
                'src_kps': source_kps,
                'trg_kps': target_kps,
                'src_kps_symm_only': torch.zeros_like(source_kps),
                'trg_kps_symm_only': torch.zeros_like(target_kps),
                'src_bndbox': src_bndbox,
                'trg_bndbox': trg_bndbox,
                'src_hflip': False,
                'trg_hflip': False,
                'idx': idx,
                'cat': self.cat}
        
        data['numkp'] = source_kps.shape[0]
        return data

    def __getitem__(self, idx):
        data_out = self.getitem_wo_feat(idx)
        
        if self.return_feats:
            src_ft, trg_ft = self.get_feats(idx)
            data_out['src_ft'] = src_ft[0]
            data_out['trg_ft'] = trg_ft[0]
        if self.return_imgs:
            img0,img1 = self.get_imgs(idx)
            data_out['src_img'] = np.array(img0)
            data_out['trg_img'] = np.array(img1)
        if self.return_masks:
            mask0, mask1 = self.get_masks(idx)
            data_out['src_mask'] = mask0
            data_out['trg_mask'] = mask1
        return data_out
 
class PFPascalPairsOrigPadded(PFPascalPairsOrig):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.n_kps = 100
        self.num_el = args.num_el if 'num_el' in args else None
        self.seed_pairs = 1 # this can be changed for different epochs
    
    def total_len(self):
        assert(self.cat is not None)
        data = pd.read_csv(f'{self.root}/{self.split}_pairs_pf_pascal.csv')
        cls_ids = data.iloc[:,2].values.astype("int") - 1
        cat_id = OBJECT_CLASSES.index(self.cat)
        return len(np.where(cls_ids == cat_id)[0])

    def __len__(self):
        if self.num_el is None:
            return self.total_len()
        return min(self.num_el, self.total_len())
    
    def pad_kps(self, kps):
        kps = kps.clone()
        pad = self.n_kps - kps.shape[0]
        if pad > 0:
            kps = torch.cat([kps, torch.zeros(pad, 3)], dim=0)
        return kps

    def init_kps_cat(self, cat):
        super().init_kps_cat(cat)
        # remove the shuffeled_idx_list
        if hasattr(self, 'shuffeled_idx_list'):
            delattr(self, 'shuffeled_idx_list')
            
    def __getitem__(self, idx):
        totallen = self.total_len()
        subsetlen = self.__len__()
        # create new idx list excluding the idx that have been used
        if not hasattr(self, 'shuffeled_idx_list'):
            with use_seed(self.seed_pairs):
                self.shuffeled_idx_list = np.random.permutation(totallen)
        elif len(self.shuffeled_idx_list) < self.seed_pairs*subsetlen:
            with use_seed(self.seed_pairs):
                self.shuffeled_idx_list = np.append(self.shuffeled_idx_list, np.random.permutation(totallen))
        idx_ = self.shuffeled_idx_list[idx+(self.seed_pairs-1)*subsetlen]

        data_out = super().__getitem__(int(idx_))
        # pad the keypoints
        for k in data_out.keys():
            if 'kps' in k:
                data_out[k] = self.pad_kps(data_out[k])
        return data_out
    
