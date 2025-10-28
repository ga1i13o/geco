
from torch.utils.data.dataset import Dataset as TorchDataset
import os
import json
import torch
import PIL.Image as Image
from src.dataset.spair_single import SpairDatasetSingle,SpairDatasetSingleAugmented
from src.dataset.random_utils import use_seed
import numpy as np

def SpairDataset2(args, **kwargs):
    if args.sup == 'sup_augmented':
        # return SpairDataset2Augmented(args, **kwargs)
        return SpairDataset2AugmentedPadded(args, **kwargs)
    elif args.sup == 'sup_original':
        return SpairDataset2Orig(args, **kwargs)

class SpairDataset2Augmented(SpairDatasetSingleAugmented):
    pck_symm = True
    annotation_path = 'PairAnnotation'

    def __init__(self, args, **kwargs):
        split = kwargs['split']
        # init the parent
        kwargs['split'] = 'all'
        super().__init__(args, **kwargs)
        self.split = split
        self.pairs = {}
        self.return_imgs = False
        self.return_masks = False
        try:
            splt = 'trn' if self.split=="train" else self.split
            with open(os.path.join(self.root, self.annotation_path, splt+".json"), "r") as fp:
                self.pairs = json.load(fp)
        except:
            # self.find_idx_pairs()
            pass
        self.seed_pairs = 1 # this can be changed for different epochs
        
    def __len__(self):
        return len(self.pairs[self.cat])
    
    def get_imgs(self, idx):
        idx0, idx1 = self.pairs[self.cat][idx]
        with use_seed(idx+self.seed_pairs):
            seed_offset0 = torch.randint(1000000, (1,)).item()
            seed_offset1 = torch.randint(1000000, (1,)).item()
        self.seed = self.seed+seed_offset0
        img0 = super()._get_img(idx0)
        self.seed = self.seed-seed_offset0
        self.seed = self.seed+seed_offset1
        img1 = super()._get_img(idx1)
        self.seed = self.seed-seed_offset1
        return img0, img1
    
    def get_masks(self, idx):
        idx0, idx1 = self.pairs[self.cat][idx]
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
        idx0, idx1 = self.pairs[self.cat][idx]
        with use_seed(idx+self.seed_pairs):
            seed_offset0 = torch.randint(1000000, (1,)).item()
            seed_offset1 = torch.randint(1000000, (1,)).item()
        self.seed = self.seed+seed_offset0
        data0 = super().__getitem__(idx0)
        self.seed = self.seed-seed_offset0
        self.seed = self.seed+seed_offset1
        data1 = super().__getitem__(idx1)
        self.seed = self.seed-seed_offset1
        data = {
                'src_imsize': data0['imsize'],
                'trg_imsize': data1['imsize'],
                'src_kps': data0['kps'],
                'trg_kps': data1['kps'],
                'trg_kps_symm_only': data1['kps_symm_only'],
                'src_kps_symm_only': data0['kps_symm_only'],
                'src_bndbox': data0['bndbox'],
                'trg_bndbox': data1['bndbox'],
                'src_hflip': data0['hflip'],
                'trg_hflip': data1['hflip'],
                'idx': idx,
                'cat': self.cat}

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
    

class SpairDataset2Orig(SpairDatasetSingle):
    pck_symm = True
    annotation_path = 'PairAnnotation'
    name = 'spair2'

    def __init__(self, args, **kwargs):
        split = kwargs['split']
        # init the parent
        kwargs['split'] = 'all'
        super().__init__(args, **kwargs)
        self.split = split
        self.pairs = {}
        self.return_imgs = False
        self.return_masks = False
        # try:
        splt = 'trn' if self.split=="train" else self.split
        with open(os.path.join(self.root, self.annotation_path, splt+".json"), "r") as fp:
            self.pairs = json.load(fp)
        # except:
            # self.find_idx_pairs()
            pass

    # def find_idx_pairs(self):
    #     # find indices of test image pairs in the category
    #     splt = 'trn' if self.split=="train" else self.split
    #     rel_path_list_ = os.listdir(os.path.join(self.root, self.annotation_path, splt))
    #     rel_path_list = {}
    #     for cat in self.all_cats:
    #         rel_path_list_cat = []
    #         for rel_path in rel_path_list_:
    #             if cat in rel_path:
    #                 rel_path_list_cat.append(rel_path)
    #         rel_path_list[cat] = rel_path_list_cat

    #     for cat in self.all_cats:
    #         rel_path_list_cat = rel_path_list[cat]
    #         pair_idx_cat = []
    #         # iterate over the img pairs
    #         for rel_path in rel_path_list_cat:
    #             splt = 'trn' if self.split=="train" else self.split
    #             with open(os.path.join(self.root, self.annotation_path, splt, rel_path)) as temp_f:
    #                 data = json.load(temp_f)
    #                 temp_f.close()
    #             src_imname = data['src_imname']
    #             trg_imname = data['trg_imname']
    #             # search for the index of the image in the list
    #             for idx in range(len(self.rel_path_list_single[cat])):
    #                 json_path = self.rel_path_list_single[cat][idx]
    #                 with open(os.path.join(self.root, self.annotation_path_single, cat, json_path)) as temp_f:
    #                     data_img_level = json.load(temp_f)
    #                     imname = data_img_level['filename']
    #                 if src_imname == imname:
    #                     idx0 = idx
    #                 if trg_imname == imname:
    #                     idx1 = idx
    #             pair_idx_cat.append((idx0, idx1))
    #         self.pairs[cat] = pair_idx_cat
    #     splt = 'trn' if self.split=="train" else self.split
    #     with open(os.path.join(self.root, self.annotation_path, splt+".json"), "w") as fp:
    #         json.dump(self.pairs, fp)
        
    def __len__(self):
        return len(self.pairs[self.cat])
    
    def get_imgs(self, idx):
        idx0, idx1 = self.pairs[self.cat][idx]
        img0 = super().get_img(idx0)
        img1 = super().get_img(idx1)
        return img0, img1
    
    def get_masks(self, idx):
        idx0, idx1 = self.pairs[self.cat][idx]
        mask0 = super().get_mask(idx0)
        mask1 = super().get_mask(idx1)
        return mask0, mask1
    
    def __getitem__(self, idx):
        idx0, idx1 = self.pairs[self.cat][idx]
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
                'idx': idx,
                'cat': self.cat}
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
    
class SpairDataset(TorchDataset):
    annotation_path = 'PairAnnotation'
    name = 'spair'
    pck_symm = False
    def __init__(self, args, split = 'test'):
        raise DeprecationWarning("This method is deprecated. Use the SpairDataset2 instead with stored features from SPairSingle.")
        self.cat = None
        self.root = args.dataset_path
        self.save_path = args.save_path
        self.featurizer_name = None
        self.split = split
        splt = 'trn' if self.split=="train" else self.split
        rel_path_list = os.listdir(os.path.join(self.root, self.annotation_path, splt))
        self.all_cats = os.listdir(os.path.join(self.root, 'JPEGImages'))
        self.rel_path_list = {}

        for cat in self.all_cats:
            rel_path_list_cat = []
            for rel_path in rel_path_list:
                if cat in rel_path:
                    rel_path_list_cat.append(rel_path)
            self.rel_path_list[cat] = rel_path_list_cat
        self.return_imgs = False

    def __len__(self):
        raise DeprecationWarning("This method is deprecated. Use the SpairDataset2 instead with stored features from SPairSingle.")
        return len(self.rel_path_list[self.cat])
    
    def get_dict_of_img_paths(self):
        raise DeprecationWarning("This method is deprecated. Use the SpairDataset2 instead with stored features from SPairSingle.")
        print("get images' paths...")
        cat2img = {}
        for cat in self.all_cats:
            cat2img[cat] = []
            rel_path_list_cat = self.rel_path_list[cat]
            for rel_path in rel_path_list_cat:
                splt = 'trn' if self.split=="train" else self.split
                with open(os.path.join(self.root, self.annotation_path, splt, rel_path)) as temp_f:
                    data = json.load(temp_f)
                    temp_f.close()
                src_imname = data['src_imname']
                trg_imname = data['trg_imname']
                if src_imname not in cat2img[cat]:
                    cat2img[cat].append(src_imname)
                if trg_imname not in cat2img[cat]:
                    cat2img[cat].append(trg_imname)
        return cat2img

    def get_imgs(self, idx):
        raise DeprecationWarning("This method is deprecated. Use the SpairDataset2 instead with stored features from SPairSingle.")
        json_path = self.rel_path_list[self.cat][idx]
        splt = 'trn' if self.split=="train" else self.split
        with open(os.path.join(self.root, self.annotation_path, splt, json_path)) as temp_f:
            data = json.load(temp_f)
            src_imname = data['src_imname']
            img0 = Image.open(os.path.join(self.root, 'JPEGImages', self.cat, src_imname)).convert('RGB')
            trg_imname = data['trg_imname']
            img1 = Image.open(os.path.join(self.root, 'JPEGImages', self.cat, trg_imname)).convert('RGB')
        return img0, img1

    def get_one_item(self, idx):
        raise DeprecationWarning("This method is deprecated. Use the SpairDataset2 instead with stored features from SPairSingle.")
        data_pair = self.__getitem__(idx)
        return {'ft': data_pair['src_ft'], 'imsize': data_pair['src_imsize'], 'kps': data_pair['src_kps'], 'bndbox': data_pair['src_bndbox']}
    
    def __getitem__(self, idx):
        raise DeprecationWarning("This method is deprecated. Use the SpairDataset2 instead with stored features from SPairSingle.")
        # find idx of test image pairs in the category
        json_path = self.rel_path_list[self.cat][idx]

        splt = 'trn' if self.split=="train" else self.split
        with open(os.path.join(self.root, self.annotation_path, splt, json_path)) as temp_f:
            data = json.load(temp_f)

        if self.featurizer_name!=None:
            path = os.path.join(self.save_path, self.name, self.featurizer_name)
            feat_dict = torch.load(os.path.join(path, f'{self.cat}.pth'))
            data['src_ft'] = feat_dict[data['src_imname']][0]
            data['trg_ft'] = feat_dict[data['trg_imname']][0]

        # get the spatial resolution of the feature maps to match the original image size, where keypoints are annotated
        data['src_imsize'] = torch.tensor(data['src_imsize'][:2][::-1])
        data['trg_imsize'] = torch.tensor(data['trg_imsize'][:2][::-1]) # W,H(in PIL) to H,W

        # convert list of lists to tensor and swap x,y
        data['src_kps'] = torch.tensor(data['src_kps'])[:,[1,0]] # num_kps, 2
        data['trg_kps'] = torch.tensor(data['trg_kps'])[:,[1,0]] # num_kps, 2
        # add a dimension indicating the visibility
        data['src_kps'] = torch.cat((data['src_kps'], torch.ones(data['src_kps'].shape[0], 1)), dim=1) # num_kps, 3
        data['trg_kps'] = torch.cat((data['trg_kps'], torch.ones(data['trg_kps'].shape[0], 1)), dim=1) # num_kps, 3

        data['idx'] = idx

        if self.return_imgs:
            img0,img1 = self.get_imgs(idx)
            data['src_img'] = np.array(img0)
            data['trg_img'] = np.array(img1)
        return data
    
class SpairDataset2AugmentedPadded(SpairDataset2Augmented):

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
            with use_seed(self.seed_pairs+1):
                self.shuffeled_idx_list = np.append(self.shuffeled_idx_list, np.random.permutation(totallen))
        idx_ = self.shuffeled_idx_list[idx+(self.seed_pairs-1)*subsetlen]

        data_out = super().__getitem__(int(idx_))
        # pad the keypoints
        for k in data_out.keys():
            if 'kps' in k:
                data_out[k] = self.pad_kps(data_out[k])
        return data_out