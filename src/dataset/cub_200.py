from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from scipy.io import loadmat
import os
from src.dataset.augmentations import DataAugmentation
from src.dataset.random_utils import use_seed
import torchvision.transforms.functional as Fvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CUBDataset(TorchDataset):
    name = 'CUB_200_2011'
    name_this = 'CUB_200_2011'
    KP_LEFT_RIGHT_PERMUTATION = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
    KP_NAMES = ['Back', #0
                'Beak', 
                'Belly', #2
                'Breast', 
                'Crown', #4
                'FHead', 
                'LEye', #6
                'LLeg', 
                'LWing', #8
                'Nape', 
                'REye', #10
                'RLeg', 
                'RWing', #12
                'Tail', 
                'Throat']#14
    KP_COLOR_MAPPING = [
        (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255),
        (255, 255, 0), (0, 0, 255), (0, 128, 255), (128, 0, 255), (0, 128, 0),
        (128, 0, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
    ]
    KP_WITH_ORIENTATION = (np.linspace(0, 14, 15, dtype=np.int32)-KP_LEFT_RIGHT_PERMUTATION)!=0

    KP_SORTED = np.array([7, 11, 8, 12, 9, 13, 1, 2, 3, 4, 5, 6, 10, 14, 15])-1 # the keypoints with geometric orientation at the end
    def __init__(self, args, split='test'):
        self.cat = None
        self.all_cats = ['bird']
        self.root = args.dataset_path
        self.root_annotations = args.dataset_path_annotations
        self.save_path = args.save_path
        self.save_path_masks = args.save_path_masks
        self.split = split
        path = os.path.join(self.root_annotations , 'data' , f'{self.split}_cub_cleaned.mat')
        self.data = loadmat(path, struct_as_record=False, squeeze_me=True)['images']
        self.featurizer_name = None
        self.all_cats_multipart = ['bird']
        self.return_feats = True

    def __len__(self):
        return len(self.data)
    
    def get_mask(self, idx):
        # also called by child classes using super()
        path = os.path.join(self.save_path_masks, 'CUB_200_2011', self.model_seg_name)
        try:
            # raise NotImplementedError        
            imname =  self.data[idx].rel_path.replace('.jpg', '.pt').replace('.png', '.pt')
            prt = torch.load(os.path.join(path, imname))
            # path = self.data[idx].rel_path.replace('.jpg', '.png')
            # img_path = os.path.join(self.root, 'CUB_200_2011', 'segmentations', path)
            # prt = Image.open(img_path).convert('L')
        except:
            data = self.data[idx]
            img_path = os.path.join(self.root, 'CUB_200_2011', 'images', data.rel_path)
            img = Image.open(img_path).convert('RGB')
            data_out = self._get_item_wo_feat(idx)
            kps = data_out['kps']
            prt = self.model_seg(img, data_out['bndbox'], kps=kps[kps[:,2]==1])
        prt = np.array(prt)
        return prt
    
    def store_masks(self, overwrite):
        assert(self.name == self.name_this)
        print("saving all %s images' masks..."%self.split)
        self.imsizes = {}
        path = os.path.join(self.save_path_masks, self.name_this, self.model_seg.name)
        for idx in range(len(self.data)):
            imname =  self.data[idx].rel_path.replace('.jpg', '.pt').replace('.png', '.pt')
            if not overwrite and os.path.exists(os.path.join(path, imname)):
                continue
            prt = self.get_mask(idx)
            prt = torch.tensor(prt)
            os.makedirs(os.path.join(path, os.path.dirname(imname)), exist_ok=True)
            torch.save(prt.detach().cpu(), os.path.join(path, imname))
            del prt
            torch.cuda.empty_cache()
    
    def init_kps_cat(self, cat):
        self.cat = cat

    def _get_feat(self, idx, featurizer, featurizer_kwargs):
        # get the feature from the , used by child classes
        cat = self.cat
        img = self._get_img(idx)
        feat = featurizer.forward(img,
                                category=cat,
                                **featurizer_kwargs)
        return feat
    
    def store_feats(self, featurizer, overwrite, featurizer_kwargs):
        assert(self.name == self.name_this)
        # can also be used by child classes
        print("saving all %s images' features..."%self.split)
        self.imsizes = {}
        path = os.path.join(self.save_path, self.name_this, featurizer.name)
        for idx in range(len(self.data)):
            ft_name =  self.data[idx].rel_path.replace('.jpg', '.pth').replace('.png', '.pth')
            if not overwrite and os.path.exists(os.path.join(path, ft_name)):
                continue
            feat = self._get_feat(idx, featurizer, featurizer_kwargs)
            os.makedirs(os.path.join(path, os.path.dirname(ft_name)), exist_ok=True)
            torch.save(feat.detach().cpu(), os.path.join(path, ft_name))
            del feat
            torch.cuda.empty_cache()

    def get_img(self, idx):
        return self._get_img(idx)
    
    def get_bbox(self, idx):
        data = self.data[idx]
        bndbox = np.asarray([data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], np.float32)
        return bndbox, bndbox
    
    def _get_img(self, idx):
        assert(self.name == self.name_this)
        data = self.data[idx]
        img_path = os.path.join(self.root, self.name_this, 'images', data.rel_path)
        img = Image.open(img_path).convert('RGB')
        return img
    
    def get_feat(self, idx):
        # get the feature from saved features, used by child classes
        data = self.data[idx]
        ft_name =  data.rel_path.replace('.jpg', '.pth').replace('.png', '.pth')
        ft = torch.load(os.path.join(self.save_path, self.name, self.featurizer_name, ft_name)).to(device)
        return ft
    
    def get_kp_dict(self, keypoints):
        kps_symm = keypoints.clone()[self.KP_LEFT_RIGHT_PERMUTATION]
        kps_symm_only = kps_symm.clone()
        kps_symm_only[~self.KP_WITH_ORIENTATION, 2] = 0
        kp_data = { 'kps': keypoints, 'kps_symm_only': kps_symm_only}
        return kp_data
    
    def get_kp_orig(self, idx):
        keypoints = self.data[idx].parts.T.copy().astype(np.float32)
        keypoints = torch.from_numpy(keypoints)[:,[1,0,2]] # WHV to HWV
        if torch.any(keypoints[:,0]>self.data[idx].height):
            keypoints[keypoints[:,0]>self.data[idx].height,2] = 0
        if torch.any(keypoints[:,1]>self.data[idx].width):
            keypoints[keypoints[:,1]>self.data[idx].width,2] = 0
        if torch.any(keypoints[:,0]<0):
            keypoints[keypoints[:,0]<0,2] = 0
        if torch.any(keypoints[:,1]<0):
            keypoints[keypoints[:,1]<0,2] = 0
        return keypoints
    
    def get_bbox(self, idx):
        data = self.data[idx]
        bndbox = np.asarray([data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], np.float32)
        adj_bbox = np.asarray([0,0,data.height,data.width], np.float32)
        return bndbox, adj_bbox
    
    def _get_item_wo_feat(self, idx):
        data = self.data[idx]
        # Retrieve the GT bbox, XXX bounding box has values in [0, max(H, W)] included
        bndbox = np.asarray([data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], np.float32)
        imsize = torch.tensor((data.height, data.width))
        data_out = {'imsize': imsize, 'bndbox':bndbox}

        keypoints = self.get_kp_orig(idx)
        kp_data = self.get_kp_dict(keypoints)
        data_out.update(kp_data)
        return data_out

    def __getitem__(self, idx):
        data_out = self._get_item_wo_feat(idx)
        if self.return_feats:
            try:
                ft = self.get_feat(idx)[0]
            except:
                ft = self._get_feat(idx, self.featurizer, self.featurizer_kwargs)[0]
            data_out['ft'] = ft
        return data_out

class CUBDatasetBordersCut(CUBDataset):
    name = 'CUB_200_2011_borders_cut'
    PADDING_BBOX = 0.05
    padding_mode = 'edge'

    def __init__(self, args, split='test'):
        super().__init__(args, split)
        assert(args.borders_cut ==True)

    def get_bbox(self, idx):
        data = self.data[idx]
        bbox = np.asarray([data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], np.float32)
        # Increase bbox size with borders
        bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
        bbox += np.asarray([self.PADDING_BBOX * s for s in [-bw, -bh, bw, bh]], dtype=np.float32)
        # Adjust bbox to a square bbox
        def square_bbox(bbox):
            """Converts a bbox to have a square shape by increasing size along non-max dimension."""
            sq_bbox = [int(round(x)) for x in bbox]  # convert to int
            width, height = sq_bbox[2] - sq_bbox[0], sq_bbox[3] - sq_bbox[1]
            maxdim = max(width, height)
            offset_w, offset_h = [int(round((maxdim - s) / 2.0)) for s in [width, height]]

            sq_bbox[0], sq_bbox[1] = sq_bbox[0] - offset_w, sq_bbox[1] - offset_h
            sq_bbox[2], sq_bbox[3] = sq_bbox[0] + maxdim, sq_bbox[1] + maxdim
            return sq_bbox
        bbox = square_bbox(bbox)# can be bigger than image
        # Shift bbox, if bbox is outside the image scope
        p_left, p_top = max(0, -bbox[0]), max(0, -bbox[1])
        adj_bbox = bbox + np.asarray([p_left, p_top, p_left, p_top], dtype=np.uint16)
        return bbox, adj_bbox

    def _get_img(self, idx):
        data = self.data[idx]
        img_path = os.path.join(self.root, 'CUB_200_2011', 'images', data.rel_path)
        img = Image.open(img_path).convert('RGB')
        bbox, adj_bbox = self.get_bbox(idx)
        # Pad image if bbox is outside the image scope
        p_left, p_top = max(0, -bbox[0]), max(0, -bbox[1])
        p_right, p_bottom = max(0, bbox[2] - img.size[0]), max(0, bbox[3] - img.size[1])
        if sum([p_left, p_top, p_right, p_bottom]) > 0:
            img = Fvision.pad(img, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
        # Crop image, it follows the classical python convention where final values are excluded
        # E.g., img.crop((0, 0, 1, 1)) corresponds to the pixel at [0, 0]
        img = img.crop(adj_bbox)
        return img
    
    def get_mask(self, idx):
        prt = super().get_mask(idx)
        # to pillow image
        prt = Image.fromarray(prt)
        bbox, adj_bbox = self.get_bbox(idx)
        # Pad image if bbox is outside the image scope
        p_left, p_top = max(0, -bbox[0]), max(0, -bbox[1])
        p_right, p_bottom = max(0, bbox[2] - prt.size[0]), max(0, bbox[3] - prt.size[1])
        if sum([p_left, p_top, p_right, p_bottom]) > 0:
            prt = Fvision.pad(prt, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
        # Crop image, it follows the classical python convention where final values are excluded
        # E.g., img.crop((0, 0, 1, 1)) corresponds to the pixel at [0, 0]
        prt = prt.crop(adj_bbox)
        prt = Fvision.to_tensor(prt)[0]
        prt = np.array(prt)
        return prt
    
    def get_kp(self,idx):
        keypoints = super().get_kp_orig(idx)#  HWV 
         # Adjust GT keypoints
        bbox, _ = self.get_bbox(idx)
        visible = keypoints[:, 2].bool()
        keypoints[visible, :2] -= torch.tensor([bbox[1], bbox[0]]).float()
        return keypoints
    
    def getitem_wo_feat(self, idx):
        keypoints = self.get_kp(idx)

        bndbox, adj_bbox = self.get_bbox(idx)
        imsize = (adj_bbox[2]-adj_bbox[0], adj_bbox[3]-adj_bbox[1])

        _, adj_bbox = self.get_bbox(idx)
        imsize = torch.tensor((adj_bbox[2]-adj_bbox[0], adj_bbox[3]-adj_bbox[1]))
        bndbox = [0,0,adj_bbox[2]-adj_bbox[0],adj_bbox[3]-adj_bbox[1]]
        data_out = { 'imsize': imsize, 'bndbox':bndbox, 'idx':idx}
        keypoints = self.get_kp(idx)
        kp_data = self.get_kp_dict(keypoints)
        data_out.update(kp_data)
        return data_out
    
    def __getitem__(self, idx):
        try:
            ft = self.get_feat(idx)[0]
        except:
            ft = self._get_feat(idx, self.featurizer, self.featurizer_kwargs)[0]
        data_out = self.getitem_wo_feat(idx)
        if self.return_feats:
            data_out['ft'] = ft
        return data_out


class CUBDatasetAugmented(CUBDatasetBordersCut):
    '''
    Add augmentation to the CUB dataset
    '''
    def __init__(self, args, split='test'):
        super().__init__(args, split)
        augmentation_args = {
            'crops_scale': (0.5,0.98),
            'crops_size': (400),
            'KP_LEFT_RIGHT_PERMUTATION': self.KP_LEFT_RIGHT_PERMUTATION,
            'flip_aug': args.flip_aug}
        self.DataAugmentation = DataAugmentation(**augmentation_args)
        self.seed = 0

    def _get_img(self, idx):
        img = super()._get_img(idx)
        data = super().getitem_wo_feat(idx)
        with use_seed(idx+self.seed):
            img, data = self.DataAugmentation(img, data)
        return img
    
    def get_mask(self, idx):
        try:
            prt = super().get_mask(idx)
            with use_seed(idx+self.seed):
                prt = self.DataAugmentation.augment_mask(Image.fromarray(prt))
                prt = np.array(prt)
        except:
            pass
        return prt
    
    def _get_feat(self, img, featurizer, featurizer_kwargs):
        cat = self.cat
        feat = featurizer.forward(img,
                                category=cat,
                                **featurizer_kwargs)
        return feat

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
