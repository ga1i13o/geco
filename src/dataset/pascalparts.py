from PIL import Image

import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from scipy.io import loadmat
import os
from tqdm import tqdm
from src.dataset.pascalparts_part2ind import part2ind
from typing import Optional, Any
OBJECT_CLASSES = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
                  'chair','cow','table','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class PascalParts(TorchDataset):
    name = 'pascalparts'
    name_this = 'pascalparts'

    def __init__(self, args, split='test'):
        self.cat = None
        self.root = args.dataset_path
        # join the path

        self.root_annotations = os.path.join(self.root,'Parts/Annotations_Part/')
        self.root_split = os.path.join(self.root,'VOCdevkit/VOC2010/ImageSets/Main/')
        self.root_imgs = os.path.join(self.root,'VOCdevkit/VOC2010/JPEGImages/')
        self.save_path = args.save_path
        self.split = split

        self.featurizer_name = None
        # Initialize attributes that may be set externally
        self.featurizer: Optional[Any] = None
        self.featurizer_kwargs: Optional[Any] = None
        self.model_seg: Optional[Any] = None
        self.model_seg_name: Optional[str] = None
        self.return_masks: bool = False

        self.all_cats = OBJECT_CLASSES
        self.all_cats_multipart = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'person'] # categories with multiple parts that are annotated in many images

        try:
            self.data = torch.load(os.path.join(self.save_path, 'pascalparts.pth'))
        except:
            self.structure_mat()
            self.data = torch.load(os.path.join(self.save_path, 'pascalparts.pth'))

    def init_kps_cat(self, cat):
        self.cat = cat
        cat_idx = OBJECT_CLASSES.index(self.cat)+1 # the index of the category in the pascal parts dataset
        part_dict, KP_LEFT_RIGHT_PERMUTATION, KP_WITH_ORIENTATION = part2ind(cat_idx) # get the part dictionary for the category

        if KP_LEFT_RIGHT_PERMUTATION is not None:
            self.KP_WITH_ORIENTATION = KP_WITH_ORIENTATION
            self.KP_LEFT_RIGHT_PERMUTATION = KP_LEFT_RIGHT_PERMUTATION
        else:
            self.KP_WITH_ORIENTATION = None
            self.KP_LEFT_RIGHT_PERMUTATION = None

        if len(part_dict.values()) == 0:
            part_dict = {'fg': 1}

        self.KP_NAMES = [k for k, v in sorted(part_dict.items(), key=lambda item: item[1])]

    def structure_mat(self):
        data = {}
        mat_filenames = os.listdir(self.root_annotations)
        for cat in OBJECT_CLASSES:
            data[cat] = []

        for idx, annotation_filename in enumerate(mat_filenames):
            mat = loadmat(os.path.join(self.root_annotations, annotation_filename), struct_as_record=False, squeeze_me=True)['anno']
            obj = mat.__dict__['objects']
            # check if obj is array, we only consider images with one object
            if isinstance(obj, np.ndarray):
                continue
            else:
                mat_cat = obj.__dict__['class']
                data[mat_cat].append(mat.__dict__['imname'])
        torch.save(data, os.path.join(self.save_path, 'pascalparts.pth'))

    def __len__(self):
        return len(self.data[self.cat])

    def _get_feat(self, idx, featurizer, featurizer_kwargs):
        cat = self.cat
        img = self.get_img(idx)
        feat = featurizer.forward(img,
                                category=cat,
                                **featurizer_kwargs)
        return feat
    
    def store_feats(self, featurizer, overwrite, featurizer_kwargs):
        assert(self.name == self.name_this)
        print("saving all %s images' features..."%self.split)
        self.imsizes = {}
        path = os.path.join(self.save_path, self.name_this, featurizer.name)
        for cat in tqdm(self.all_cats):
            self.init_kps_cat(cat)
            for idx in range(len(self)):
                ft_name = self.data[self.cat][idx]+'.pth'
                if os.path.exists(os.path.join(path, ft_name)) and not overwrite:
                    continue
                feat = self._get_feat(idx, featurizer, featurizer_kwargs)
                # make directory of parent directory
                os.makedirs(os.path.join(path, os.path.dirname(ft_name)), exist_ok=True)
                torch.save(feat.detach().cpu(), os.path.join(path, ft_name))
                del feat
                torch.cuda.empty_cache()

    def get_img(self, idx):
        imname = self.data[self.cat][idx]+'.jpg'
        img_path = os.path.join(self.root_imgs, imname)
        img = Image.open(img_path).convert('RGB')
        return img
    
    def get_feat(self, idx):
        assert(self.featurizer_name is not None)
        assert(self.cat is not None)
        ft_name =  self.data[self.cat][idx]+'.pth'
        ft = torch.load(os.path.join(self.save_path, self.name_this, self.featurizer_name, ft_name)).to(device)
        return ft
    

    def get_parts(self, idx):
        '''
        Output:
            part_dict: dictionary with part names as keys and part indices as values
            parts_mask: tensor of shape (num_parts, H, W) with one hot encoding of the parts,
                        where num_parts is the number of parts in the category without the background,
                        i.e. part_mask.sum(dim=0) should be 0 for the background

        '''
        name = self.data[self.cat][idx]
        mat = loadmat(os.path.join(self.root_annotations, name+'.mat'), struct_as_record=False, squeeze_me=True)['anno']
        # check if there is only one object in the image
        if isinstance(mat.__dict__['objects'], np.ndarray):
            raise ValueError('Multiple objects in image')
        obj_mask = mat.__dict__['objects'].__dict__['mask']
        assert(self.cat is not None)
        cat_idx = OBJECT_CLASSES.index(self.cat)+1 # the index of the category in the pascal parts dataset
        part_dict, _, _ = part2ind(cat_idx) # get the part dictionary for the category
        if len(part_dict.values()) == 0:
            num_parts = 1
            parts_mask = torch.zeros(num_parts, obj_mask.shape[0], obj_mask.shape[1])
            parts_mask[0, :, :] = torch.tensor(obj_mask)
            part_dict = {'fg': 1}
        else:
            num_parts = max(part_dict.values()) # not really the number of parts, as part indices are not continuous
            parts_mask = torch.zeros(num_parts, obj_mask.shape[0], obj_mask.shape[1])
            # check if there is only one part in the object
            parts = mat.__dict__['objects'].__dict__['parts']
            if not isinstance(parts, np.ndarray):
                parts = [parts]
            for part in parts:
                part_idx = part_dict[part.__dict__['part_name']]-1
                parts_mask[part_idx, :, :] = torch.tensor(part.__dict__['mask'])

        def one_hot_parts(part_mask):
            # just take the occurence with highest value in dimension 1, i.e. the part with the highest index, pascalparts has multiple labels per pixel and we want to take the one with the highest index
            bkg = (part_mask.sum(dim=0, keepdim=True)==0).repeat(part_mask.shape[0],1,1)
            prt_reverse = torch.flip(part_mask, [0])
            label = part_mask.shape[0] - prt_reverse.argmax(dim=0, keepdim=True) - 1 # undo the reverse
            prt = torch.zeros_like(part_mask)
            prt = prt.scatter_(0, label, 1)
            prt[bkg]=0
            return prt

        parts_mask = one_hot_parts(parts_mask)

        num_parts = max(part_dict.values()) # not really the number of parts, as part indices are not continuous
        self.parts = torch.unique( torch.tensor(list(part_dict.values())) -1)
        num_parts_ = len(self.parts)

        if num_parts_ != num_parts:
            parts_mask = parts_mask[self.parts]

        return part_dict, parts_mask

    def __getitem__(self, idx):
        part_dict, parts_mask = self.get_parts(idx)
        try:
            ft = self.get_feat(idx)[0]
        except:
            assert(self.featurizer is not None)
            assert(self.featurizer_kwargs is not None)
            ft = self._get_feat(idx, self.featurizer, self.featurizer_kwargs)[0]
        imsize = torch.tensor((parts_mask.shape[-2], parts_mask.shape[-1]))
        return {'ft': ft, 'imsize': imsize, 'parts_mask': parts_mask, 'part_dict': part_dict, 'idx': idx}
