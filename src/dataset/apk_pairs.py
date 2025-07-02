
from torch.utils.data.dataset import Dataset as TorchDataset
import os
import json
import torch
import PIL.Image as Image
import numpy as np
from glob import glob
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from src.dataset.augmentations import DataAugmentation
from src.dataset.random_utils import use_seed

AP10K_FLIP = [
                [0,1], # eye
                2, # nose
                3, # neck
                4, # root of tail
                [5,8], # shoulder
                [6,9], # elbow # knee
                [12, 15], # knee
                [7,10], # front paw 
                [13,16], # back paw
                [11,14], # hip
                              ]

def AP10KPairs(args, **kwargs):
    if args.sup == 'sup_augmented':
        # return AP10KPairsAugmented(args, **kwargs)
        return AP10KPairsAugmentedPadded(args, **kwargs)
    elif args.sup == 'sup_original':
        return AP10KPairsOrig(args, **kwargs)

class AP10KPairsOrig(TorchDataset):
    name = 'ap10kpairs'
    name_this = 'ap10kpairs'
    # get the permutation from the flip annotation AP10K_FLIP
    KP_LEFT_RIGHT_PERMUTATION = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    KP_SORTED = np.array([0,1,5,8,6,9,12,15,7,10,13,16,11,14,2,3,4]) # the keypoints with geometric orientation at the end
    # seach for each keypoint if it is symmetric
    for item in AP10K_FLIP:
        if isinstance(item, list):
            for i in item:
                KP_LEFT_RIGHT_PERMUTATION[i] = item[(i+1)%len(item)]
    KP_LEFT_RIGHT_PERMUTATION = np.array(KP_LEFT_RIGHT_PERMUTATION)
    KP_WITH_ORIENTATION = (np.linspace(0, 16, 17, dtype=np.int32)-KP_LEFT_RIGHT_PERMUTATION)!=0
    KP_NAMES = ['eye_l', 'eye_r', 'nose', 'neck', 'root of tail', 'shoulder_l', 'elbow_l', 'front paw_l', 'shoulder_r', 'elbow_r', 'front paw_r', 'hip_l', 'knee_l', 'back paw_l', 'hip_r', 'knee_r', 'back paw_r']
    pck_symm=True
    def __init__(self, args, split):
        self.split = split
        self.cat = None
        self.root = args.dataset_path
        self.save_path = args.save_path
        self.save_path_masks = args.save_path_masks
        self.featurizer_name = None
        subfolders = os.listdir(os.path.join(self.root, 'ImageAnnotation'))
        self.all_cats = sorted([item for subfolder in subfolders for item in os.listdir(os.path.join(self.root, 'ImageAnnotation', subfolder))])
        self.all_cats_multipart = self.all_cats
        self.subsample = args.subsample

        self.pairs = {}
        for cat in self.all_cats:
            splt = 'trn' if self.split=="train" else self.split
            np.random.seed(42)
            pairs = sorted(glob(f'{self.root}/PairAnnotation/{splt}/*:{cat}.json'))
            if self.subsample is not None and self.subsample > 0:
                pairs = [pairs[ix] for ix in np.random.choice(len(pairs), self.subsample)]
            self.pairs[cat] = pairs
        # remove the category with no pairs
        self.all_cats = [cat for cat in self.all_cats if len(self.pairs[cat])>0]
        self.return_imgs = False
        self.return_masks = False
        self.return_feats = True

    def __len__(self):
        return len(self.pairs[self.cat])

    def init_kps_cat(self, cat):
        self.cat = cat
        
    def _get_img(self, idx):
        json_path = self.pairs[self.cat][idx]
        with open(json_path) as f:
            data = json.load(f)
        img_path = data["src_json_path"].replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
        return Image.open(img_path).convert('RGB')
    
    def get_imgs(self, idx):
        json_path = self.pairs[self.cat][idx]
        with open(json_path) as f:
            data = json.load(f)
        source_json_path = data["src_json_path"]
        target_json_path = data["trg_json_path"]
        src_img_path = source_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
        trg_img_path = target_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
        return Image.open(src_img_path).convert('RGB'), Image.open(trg_img_path).convert('RGB')

    def get_feats(self, idx):
        try:
            json_path = self.pairs[self.cat][idx]
            with open(json_path) as f:
                data = json.load(f)
            source_imname = data["src_json_path"].split('/')[-1].split('.')[0]
            target_imname = data["trg_json_path"].split('/')[-1].split('.')[0]
            feat0 = self.get_feat(source_imname)
            feat1 = self.get_feat(target_imname)
        except:
            img0, img1 = self.get_imgs(idx)
            feat0 = self._get_feat(img0, self.featurizer, self.featurizer_kwargs)
            feat1 = self._get_feat(img1, self.featurizer, self.featurizer_kwargs)
        return feat0, feat1

    def get_masks(self, idx):
        json_path = self.pairs[self.cat][idx]
        with open(json_path) as f:
            data = json.load(f)
        source_json_path = data["src_json_path"]
        target_json_path = data["trg_json_path"]
        try:
            imname0_, imname1_ = source_json_path.split('/')[-1], target_json_path.split('/')[-1]
            imname0, imname1 = imname0_.split('.')[0]+'.pt', imname1_.split('.')[0]+'.pt'
            mask0 = torch.load(os.path.join(self.save_path_masks, self.name_this, self.model_seg_name, self.cat, imname0))
            mask1 = torch.load(os.path.join(self.save_path_masks, self.name_this, self.model_seg_name, self.cat, imname1))
        except:
            src_img_path = source_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
            trg_img_path = target_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')

            img0, img1 = Image.open(src_img_path).convert('RGB'), Image.open(trg_img_path).convert('RGB')
            data_out = self.getitem_wo_feat(idx)
            kps0 = data_out['src_kps']
            kps1 = data_out['trg_kps']
            mask0 = self.model_seg(img0, data_out['src_bndbox'], kps=kps0[kps0[:,2]==1])
            mask1 = self.model_seg(img1, data_out['trg_bndbox'], kps=kps1[kps1[:,2]==1])
        mask0 = np.array(mask0)
        mask1 = np.array(mask1)
        return mask0, mask1
    
    def store_masks(self, overwrite):
        assert(self.name == self.name_this)
        print("saving all %s images' masks..."%self.split)
        path = os.path.join(self.save_path_masks, self.name_this, self.model_seg.name)
        for cat in tqdm(self.all_cats):
            self.init_kps_cat(cat)
            for idx_ in range(len(self)):

                json_path = self.pairs[self.cat][idx_]
                with open(json_path) as f:
                    data = json.load(f)
                source_json_path = data["src_json_path"]
                target_json_path = data["trg_json_path"]
                imname0_, imname1_ = source_json_path.split('/')[-1], target_json_path.split('/')[-1]
                imname0, imname1 = imname0_.split('.')[0]+'.pt', imname1_.split('.')[0]+'.pt'

                if not(os.path.exists(os.path.join(path, cat, imname0))) or not(os.path.exists(os.path.join(path, cat, imname1))) or overwrite:
                    prt0, prt1 = self.get_masks(idx_)
                    prt0 = torch.tensor(prt0)
                    prt1 = torch.tensor(prt1)

                if not(os.path.exists(os.path.join(path, cat, imname0)) and not overwrite):
                    torch.cuda.empty_cache()
                    os.makedirs(os.path.join(path, cat), exist_ok=True)
                    torch.save(prt0.cpu(), os.path.join(path, cat, imname0))
                if not(os.path.exists(os.path.join(path, cat, imname1)) and not overwrite):
                    torch.cuda.empty_cache()
                    os.makedirs(os.path.join(path, cat), exist_ok=True)
                    torch.save(prt1.cpu(), os.path.join(path, cat, imname1))

    def get_feat(self, imname):
        path = os.path.join(self.save_path, self.name_this, self.featurizer_name, self.cat)
        feat = torch.load(os.path.join(path, imname+'.pth')).to(device)
        return feat
    
    def _get_feat(self, img, featurizer, featurizer_kwargs):
        cat = self.cat
        feat = featurizer.forward(img,
                                category=cat,
                                **featurizer_kwargs)
        return feat
    
    def store_feats(self, featurizer, overwrite, featurizer_kwargs):
        assert(self.name == self.name_this)
        self.featurizer_name = featurizer.name
        for cat in tqdm(self.all_cats):
            self.cat = cat
            path = os.path.join(self.save_path, self.name_this, self.featurizer_name, cat)
            if not os.path.exists(path):
                os.makedirs(path)
            for idx in range(len(self)):
                json_path = self.pairs[self.cat][idx]
                with open(json_path) as f:
                    data = json.load(f)
                source_imname = data["src_json_path"].split('/')[-1].split('.')[0]
                target_imname = data["trg_json_path"].split('/')[-1].split('.')[0]
                if not overwrite and os.path.exists(os.path.join(path, source_imname+'.pth')):
                    continue
                img0, img1 = self.get_imgs(idx)
                feat0 = self._get_feat(img0, featurizer, featurizer_kwargs)
                torch.save(feat0.detach().cpu(), os.path.join(path, source_imname+'.pth'))
                if not overwrite and os.path.exists(os.path.join(path, target_imname+'.pth')):
                    continue
                feat1 = self._get_feat(img1, featurizer, featurizer_kwargs)
                torch.save(feat1.detach().cpu(), os.path.join(path, target_imname+'.pth'))
    
    def get_kp_dict(self, keypoints):
        kp_data = { 'kps': keypoints}
        if self.KP_LEFT_RIGHT_PERMUTATION is not None:
            kps_symm = keypoints.clone()[self.KP_LEFT_RIGHT_PERMUTATION]
            kps_symm_only = kps_symm.clone()
            kps_symm_only[~self.KP_WITH_ORIENTATION, 2] = 0
            # kp_data['kps_symm'] = kps_symm
            kp_data['kps_symm_only'] = kps_symm_only
        return kp_data

    def getitem_wo_feat(self, idx):
        json_path = self.pairs[self.cat][idx]
        with open(json_path) as f:
            data = json.load(f)
        source_json_path = data["src_json_path"]
        target_json_path = data["trg_json_path"]

        with open(source_json_path) as f:
            src_file = json.load(f)
        with open(target_json_path) as f:
            trg_file = json.load(f)
        
        l, t, w, h = src_file["bbox"]
        source_bbox = np.asarray([l, t, l+w, t+h]) # x1, y1, x2, y2
        l, t, w, h = trg_file["bbox"]
        target_bbox = np.asarray([l, t, l+w, t+h])# x1, y1, x2, y2
        
        source_size = torch.tensor([src_file["height"], src_file["width"]])
        target_size = torch.tensor([trg_file["height"], trg_file["width"]])

        src_kps = torch.tensor(src_file["keypoints"]).view(-1, 3).float()
        src_kps[:,-1] /= 2
        # switch x and y
        src_kps = src_kps[:, [1, 0, 2]]
        kp_data_src = self.get_kp_dict(src_kps)
        kp_data_src = {f'src_{k}': v for k, v in kp_data_src.items()}

        trg_kps = torch.tensor(trg_file["keypoints"]).view(-1, 3).float()
        trg_kps[:,-1] /= 2
        # switch x and y
        trg_kps = trg_kps[:, [1, 0, 2]]
        kp_data_trg = self.get_kp_dict(trg_kps)
        kp_data_trg = {f'trg_{k}': v for k, v in kp_data_trg.items()}
        data = {
                'src_imsize': source_size,
                'trg_imsize': target_size,
                'src_bndbox': source_bbox,
                'trg_bndbox': target_bbox,
                'idx': idx,
                'cat': self.cat}
        data.update(kp_data_src)
        data.update(kp_data_trg)
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
        return data_out
class AP10KPairsAugmented(AP10KPairsOrig):
    '''
    Add augmentation to the AP10K dataset
    '''
    def __init__(self, args, split='test'):
        super().__init__(args, split)
        augmentation_args = {
            'crops_scale': (0.5,0.98),
            'crops_size': (400),
            'KP_LEFT_RIGHT_PERMUTATION': self.KP_LEFT_RIGHT_PERMUTATION,
            'flip_aug': args.flip_aug
        }
        self.DataAugmentation = DataAugmentation(**augmentation_args)
        self.seed_pairs = 1
        self.return_imgs=False

    def get_imgs(self, idx):
        img0, img1 = super().get_imgs(idx)
        data = super().getitem_wo_feat(idx)

        # get rid of "src" and "trg" in the keys
        data0 = {k.replace('src_', ''): v for k, v in data.items() if 'src' in k}
        data1 = {k.replace('trg_', ''): v for k, v in data.items() if 'trg' in k}
        with use_seed(idx+self.seed_pairs):
            img0, data0 = self.DataAugmentation(img0, data0)
            img1, data1 = self.DataAugmentation(img1, data1)
        return img0, img1

    def init_kps_cat(self, cat):
        super().init_kps_cat(cat)
        self.DataAugmentation.KP_LEFT_RIGHT_PERMUTATION = self.KP_LEFT_RIGHT_PERMUTATION

    def get_masks(self, idx):
        mask0, mask1 = super().get_masks(idx)
        with use_seed(idx+self.seed_pairs):
            mask0 = self.DataAugmentation.augment_mask(Image.fromarray(mask0))
            mask1 = self.DataAugmentation.augment_mask(Image.fromarray(mask1))
        mask0 = np.array(mask0)
        mask1 = np.array(mask1)
        return mask0, mask1
    
    def __getitem__(self, idx):
        img_src, img_trg = super().get_imgs(idx)
        data = super().getitem_wo_feat(idx)

        # get rid of "src" and "trg" in the keys
        data_src = {k.replace('src_', ''): v for k, v in data.items() if 'src' in k}
        data_trg = {k.replace('trg_', ''): v for k, v in data.items() if 'trg' in k}
        with use_seed(idx+self.seed_pairs):
            img_src, data_src = self.DataAugmentation(img_src, data_src)
            img_trg, data_trg = self.DataAugmentation(img_trg, data_trg)
        data_out = {}
        data_out.update({f'src_{k}': v for k, v in data_src.items()})
        data_out.update({f'trg_{k}': v for k, v in data_trg.items()})
        data_out.update({'idx': idx, 'cat': self.cat})
                
        if self.return_feats:
            # get the features of the augmented image
            ft_src = self._get_feat(img_src, self.featurizer, self.featurizer_kwargs)[0]
            ft_trg = self._get_feat(img_trg, self.featurizer, self.featurizer_kwargs)[0]
            data_out['src_ft'] = ft_src
            data_out['trg_ft'] = ft_trg
        if self.return_imgs:
            img0,img1 = self.get_imgs(idx)
            data_out['src_img'] = np.array(img0)
            data_out['trg_img'] = np.array(img1)
        data_out['numkp'] = len(self.KP_NAMES)
        if self.return_masks:
            mask0,mask1 = self.get_masks(idx)
            data_out['src_mask'] = mask0
            data_out['trg_mask'] = mask1
        return data_out

class AP10KPairsAugmentedPadded(AP10KPairsAugmented):

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.n_kps = 100
        self.num_el = args.num_el if 'num_el' in args else None

    def total_len(self):
        return len(self.pairs[self.cat])
    
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
                # append to numpy array
                self.shuffeled_idx_list = np.append(self.shuffeled_idx_list, np.random.permutation(totallen))
        idx_ = self.shuffeled_idx_list[idx+(self.seed_pairs-1)*subsetlen]

        data_out = super().__getitem__(int(idx_))
        # pad the keypoints
        for k in data_out.keys():
            if 'kps' in k:
                data_out[k] = self.pad_kps(data_out[k])
        return data_out