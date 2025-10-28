
from torch.utils.data.dataset import Dataset as TorchDataset
import os
import json
import torch
import PIL.Image as Image
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from src.dataset.augmentations import DataAugmentation
from src.dataset.random_utils import use_seed
from typing import Optional, Any


class SpairDatasetSingle(TorchDataset):
    KP_LEFT_RIGHT_PERMUTATION = None
    KP_WITH_ORIENTATION = None
    annotation_path_single = 'ImageAnnotation'
    annotation_path = 'PairAnnotation'
    name = 'spair_single'
    name_this = 'spair_single'
    padding_kps = False

    def __init__(self, args, split):
        self.cat = None
        self.root = args.dataset_path
        self.save_path = args.save_path
        self.save_path_masks = args.save_path_masks
        self.featurizer_name = None
        self.all_cats = sorted(os.listdir(os.path.join(self.root, 'JPEGImages')))
        self.rel_path_list_single = {}
        self.all_cats_multipart = self.all_cats
        self.return_feats = True
        # Initialize attributes that may be set externally
        self.featurizer: Optional[Any] = None
        self.featurizer_kwargs: Optional[Any] = None
        self.model_seg: Optional[Any] = None
        self.model_seg_name: Optional[str] = None
        self.return_masks: bool = False
        for cat in self.all_cats:
            # load all image annotations
            self.rel_path_list_single[cat] = os.listdir(os.path.join(self.root, self.annotation_path_single, cat))
        self.split = split
        if self.split in ['train', 'val', 'test']:
            splt = 'trn' if self.split=="train" else self.split
            # load the pair indices for the split and find the indices of the images that are used in the pairs
            with open(os.path.join(self.root, self.annotation_path, splt+".json"), "r") as fp:
                pairs = json.load(fp)
            self.split_indices = {}
            for cat in self.all_cats:
                self.split_indices[cat] = torch.tensor(pairs[cat]).flatten().unique().tolist()
        elif self.split in ['all']:
            self.split_indices = {}
            for cat in self.all_cats:
                self.split_indices[cat] = torch.arange(len(self.rel_path_list_single[cat])).tolist()
        else:
            raise ValueError('split not recognized')

    def __len__(self):
        return len(self.split_indices[self.cat])
        # return len(self.rel_path_list_single[self.cat])
    
    def get_dict_of_img_paths(self):
        print("get images' paths...")
        cat2img = {}
        for cat in self.all_cats:
            cat2img[cat] = []
            for rel_path in self.rel_path_list_single[cat]:
                with open(os.path.join(self.root, self.annotation_path_single, cat, rel_path)) as temp_f:
                    data = json.load(temp_f)
                    temp_f.close()
                imname = data['filename']
                if imname not in cat2img[cat]:
                    cat2img[cat].append(imname)
        return cat2img
    
    def _get_feat(self, idx_, featurizer, featurizer_kwargs):
        cat = self.cat
        img = self._get_img(idx_)
        feat = featurizer.forward(img,
                                category=cat,
                                **featurizer_kwargs)
        return feat
    
    def store_feats(self, featurizer, overwrite, featurizer_kwargs):
        cat2img = self.get_dict_of_img_paths()
        print("saving all %s images' features..."%self.split)
        
        path = os.path.join(self.save_path, self.name_this, featurizer.name)
        os.makedirs(path, exist_ok=True)
        for cat in tqdm(self.all_cats):
            image_list = cat2img[cat]
            for imname in image_list:
                imname_ = imname.split('.')[0]
                if os.path.exists(os.path.join(path, cat, imname_+'.pth')) and not overwrite:
                    continue
                img = Image.open(os.path.join(self.root, 'JPEGImages', cat, imname))
                ft = featurizer.forward(img, category=cat, **featurizer_kwargs).detach().cpu()
                torch.cuda.empty_cache()
                torch.save(ft, os.path.join(path, cat, imname_+'.pth'))

    def get_img(self, idx_):
        # might get overwritten by pair dataset
        return self._get_img(idx_)

    def _get_img(self, idx_):
        assert(self.cat is not None)
        idx = self.split_indices[self.cat][idx_]
        json_path = self.rel_path_list_single[self.cat][idx]
        with open(os.path.join(self.root, self.annotation_path_single, self.cat, json_path)) as temp_f:
            data = json.load(temp_f)
            imname = data['filename']
            img = Image.open(os.path.join(self.root, 'JPEGImages', self.cat, imname)).convert('RGB')
        return img
    
    def get_mask(self, idx_):
        assert(self.cat is not None)
        idx = self.split_indices[self.cat][idx_]
        json_path = self.rel_path_list_single[self.cat][idx]
        with open(os.path.join(self.root, self.annotation_path_single, self.cat, json_path)) as temp_f:
            data = json.load(temp_f)
            imname = data['filename']
        try:
            assert(self.model_seg_name is not None)
            # raise NotImplementedError
            imname_ = imname.split('.')[0]+'.png'
            prt = torch.load(os.path.join(self.save_path_masks, self.name_this, self.model_seg_name, self.cat, imname_))
            # prt = Image.open(os.path.join(self.root, 'Segmentation', self.cat, imname))
        except:
            assert(self.model_seg is not None)
            img = Image.open(os.path.join(self.root, 'JPEGImages', self.cat, imname)).convert('RGB')
            kps = self.get_kp(idx_)
            prt = self.model_seg(img, data['bndbox'], kps=kps[kps[:,2]==1])
        prt = np.array(prt)
        return prt
    
    def store_masks(self, overwrite):
        assert(self.model_seg is not None)
        print("saving all %s images' masks..."%self.split)
        path = os.path.join(self.save_path_masks, self.name_this, self.model_seg.name)
        for cat in tqdm(self.all_cats):
            self.init_kps_cat(cat)
            for idx_ in range(len(self)):
                prt = self.get_mask(idx_)
                prt = torch.tensor(prt)
                # get imname
                assert(self.cat is not None)
                idx = self.split_indices[self.cat][idx_]
                json_path = self.rel_path_list_single[self.cat][idx]
                with open(os.path.join(self.root, self.annotation_path_single, self.cat, json_path)) as temp_f:
                    data = json.load(temp_f)
                    imname = data['filename']
                imname_ = imname.split('.')[0]+'.png'
                if os.path.exists(os.path.join(path, cat, imname_)) and not overwrite:
                    continue
                
                torch.cuda.empty_cache()
                os.makedirs(os.path.join(path, cat), exist_ok=True)
                torch.save(prt, os.path.join(path, cat, imname_))
    
    def init_kps_cat(self, cat):
        self.cat = cat
        self.KP_LEFT_RIGHT_PERMUTATION = self.get_kp_l_r_permutation()
        n_kps = len(self.KP_LEFT_RIGHT_PERMUTATION)
        self.KP_WITH_ORIENTATION =  (np.linspace(0, n_kps-1, n_kps, dtype=np.int32)-self.KP_LEFT_RIGHT_PERMUTATION)!=0
        self.KP_NAMES = self.get_kp_name()
        self.KP_EXCLUDED = self.get_kp_excluded()

    def get_kp_dict(self, keypoints):
        kp_data = { 'kps': keypoints}
        if self.KP_LEFT_RIGHT_PERMUTATION is not None and self.KP_WITH_ORIENTATION is not None:
            kps_symm = keypoints.clone()[self.KP_LEFT_RIGHT_PERMUTATION]
            kps_symm_only = kps_symm.clone()
            kps_symm_only[~self.KP_WITH_ORIENTATION, 2] = 0
            kp_data['kps_symm'] = kps_symm
            kp_data['kps_symm_only'] = kps_symm_only
        return kp_data
    
    def get_kp(self, idx_):
        assert(self.cat is not None)
        idx = self.split_indices[self.cat][idx_]
        json_path = self.rel_path_list_single[self.cat][idx]
        with open(os.path.join(self.root, self.annotation_path_single, self.cat, json_path)) as temp_f:
            data = json.load(temp_f)
            keypoints = data['kps']
        # convert dictionary with int keys to list
        keypoints = [keypoints[str(i)] for i in range(len(keypoints))]
        # append 1 to each keypoint to make it homogenous,
        [keypoints[i].append(1) for i in range(len(keypoints)) if keypoints[i]!=None]
        # if keypoint is None, fill zero
        keypoints = [keypoints[i] if keypoints[i]!=None else [0,0,0] for i in range(len(keypoints))]
        # convert list of list to tensor, fill zero for None
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        keypoints = keypoints[:, [1,0,2]] # WHV to HWV
        return keypoints
    
    def get_feat(self, imname):
        assert(self.featurizer_name is not None)
        assert(self.cat is not None)
        name = 'spair_single'
        path = os.path.join(self.save_path, name, self.featurizer_name)
        imname_  = imname.split('.')[0]
        ft = torch.load(os.path.join(path, self.cat, imname_+'.pth'))[0].to(device)
        return ft
    
    def getitem_wo_feat(self, idx_):
        assert(self.cat is not None)
        idx = self.split_indices[self.cat][idx_]
        # find idx of test image pairs in the category
        json_path = self.rel_path_list_single[self.cat][idx]
        with open(os.path.join(self.root, self.annotation_path_single, self.cat, json_path)) as temp_f:
            data = json.load(temp_f)
        imsize = torch.tensor((data['image_height'], data['image_width']))
        keypoints = self.get_kp(idx_)
        self.KP_NAMES = self.get_kp_name()
        if self.padding_kps:
            num_kp = len(self.KP_NAMES)
        else:
            num_kp = (self.KP_NAMES!= "").sum()
        keypoints = keypoints[:num_kp]
        self.KP_NAMES = self.KP_NAMES[:num_kp].tolist()
        kp_data = self.get_kp_dict(keypoints)
        bndbox = data['bndbox']
        data_out = {'imsize': imsize,  'bndbox':bndbox, 'idx':idx_}
        data_out.update(kp_data)
        return data_out
    
    def __getitem__(self, idx_):
        assert(self.cat is not None)
        idx = self.split_indices[self.cat][idx_]
        # find idx of test image pairs in the category
        json_path = self.rel_path_list_single[self.cat][idx]
        with open(os.path.join(self.root, self.annotation_path_single, self.cat, json_path)) as temp_f:
            data = json.load(temp_f)
        data_out = self.getitem_wo_feat(idx_)
        if self.return_feats:
            try:
                ft = self.get_feat(data['filename'])
            except:
                ft = self._get_feat(idx_, self.featurizer, self.featurizer_kwargs)[0]
            data_out['ft'] = ft
        return data_out
    
    def get_kp_excluded(self):
        excluded = "nostril"
        kp_names = self.get_kp_name()
        kp_excluded = [1 if excluded in name else 0 for name in kp_names]
        return kp_excluded
    
    def get_kp_name(self):
        keypoint_csv = np.loadtxt(self.root+"/spair_keypoint_names.csv", delimiter=",", dtype=str)
        keypoint_csv = keypoint_csv.transpose()
        class_names = [_cls.strip() for _cls in keypoint_csv[1:, 0]]
        cls_idx = class_names.index(self.cat)
        keypoint_csv = keypoint_csv[1:, 1:]
        return keypoint_csv[cls_idx,:]

    def get_kp_l_r_permutation(self):
        keypoint_csv = np.loadtxt(self.root+"/spair_keypoint_l_r_permutation.csv", delimiter=",", dtype=str)
        keypoint_csv = keypoint_csv.transpose()
        class_names = [_cls.strip() for _cls in keypoint_csv[1:, 0]]
        cls_idx = class_names.index(self.cat)
        keypoint_csv = keypoint_csv[1:, 1:]
        string_perm =  keypoint_csv[cls_idx,:]
        # delete None
        if self.padding_kps:
            KP_LEFT_RIGHT_PERMUTATION = [int(x.item()) if x!='' else i for i,x in enumerate(string_perm)]
        else:
            KP_LEFT_RIGHT_PERMUTATION = [int(x.item()) for x in string_perm if x!='']

        return KP_LEFT_RIGHT_PERMUTATION

class SpairDatasetSingleAugmented(SpairDatasetSingle):
    '''
    Add augmentation to the SPair dataset
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
        self.seed = 10

    def _get_img(self, idx):
        img = super()._get_img(idx)
        data = super().getitem_wo_feat(idx)
        with use_seed(idx+self.seed):
            img, data = self.DataAugmentation(img, data)
        return img
    
    def get_mask(self, idx):
        prt = super().get_mask(idx)
        with use_seed(idx+self.seed):
            prt = self.DataAugmentation.augment_mask(Image.fromarray(prt))
        prt = np.array(prt)
        return prt
    
    def _get_feat(self, img, featurizer, featurizer_kwargs):
        cat = self.cat
        feat = featurizer.forward(img,
                                category=cat,
                                **featurizer_kwargs)
        return feat

    def init_kps_cat(self, cat):
        super().init_kps_cat(cat)
        self.DataAugmentation.KP_LEFT_RIGHT_PERMUTATION = self.KP_LEFT_RIGHT_PERMUTATION

    def __getitem__(self, idx):
        img = super()._get_img(idx)
        data = super().getitem_wo_feat(idx)
        # augment the image and the keypoints
        with use_seed(idx+self.seed):
            img, data = self.DataAugmentation(img, data)
        # get the features of the augmented image
        if self.return_feats:
            ft = self._get_feat(img, self.featurizer, self.featurizer_kwargs)[0]
            data['ft'] = ft
        return data
