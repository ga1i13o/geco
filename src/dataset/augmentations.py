

from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import torch


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class DataAugmentation(object):
    def __init__(
        self,
        crops_scale=None,
        crops_size=None,
        KP_LEFT_RIGHT_PERMUTATION=None,
        KP_WITH_ORIENTATION=None,
        color_aug=True,
        flip_aug=True,
    ):
        self.KP_LEFT_RIGHT_PERMUTATION = KP_LEFT_RIGHT_PERMUTATION
        self.KP_WITH_ORIENTATION = KP_WITH_ORIENTATION
        if self.KP_WITH_ORIENTATION is not None:
            self.KP_WITH_ORIENTATION = torch.tensor(self.KP_WITH_ORIENTATION).flatten()
        self.flip_aug = flip_aug
        if crops_scale!=None:
            # random resized crop
            self.crops_size = crops_size
            self.geometric_augmentation_1 = transforms.RandomResizedCrop(
                        crops_size, scale=crops_scale, ratio=(0.9, 1.1),
                    )
        else:
            self.crops_size = None

        # color distorsions / blurring
        if color_aug:
            color_jittering = transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                        p=0.8,
                    )
                ]
            )

            colors = transforms.Compose(
                [
                    GaussianBlur(p=0.1),
                    transforms.RandomSolarize(threshold=128, p=0.2),
                ]
            )

            self.color_augmentation = transforms.Compose([color_jittering, colors])
        else:
            self.color_augmentation = None

    def augment_mask(self, mask):

        # Horizontal flip
        if self.flip_aug:
            hflip = np.random.binomial(1, p=0.5)
        else:
            hflip = False
        # hflip = False
        if hflip:
            if mask is not None:
                mask = F.hflip(mask)

        if self.crops_size!=None:
            i, j, h, w = self.geometric_augmentation_1.get_params(mask, scale=self.geometric_augmentation_1.scale, ratio=self.geometric_augmentation_1.ratio)
            # crop the image
            if mask is not None:
                mask = F.resized_crop(mask, i, j, h, w, (self.crops_size, self.crops_size))
        return mask

    def __call__(self, image, data):
        #{'ft': ft, 'imsize': imsize, 'kps': keypoints, 'kps_symm_only': kps_symm_only, 'bndbox':bndbox, 'idx':idx}

        # Horizontal flip
        if self.flip_aug:
            hflip = np.random.binomial(1, p=0.5)
        else:
            hflip = False
            
        # hflip = False
        if hflip:
            image = F.hflip(image)
            # data['bndbox'] is a list of 4 elements [xmin, ymin, xmax, ymax]
            xmin = data['bndbox'][0]
            xmax = data['bndbox'][2]
            data['bndbox'][0] = data['imsize'][1].item()-xmax
            data['bndbox'][2] = data['imsize'][1].item()-xmin
            data['hflip'] = True
        else:
            data['hflip'] = False

        if hflip:
            if self.KP_WITH_ORIENTATION is not None:
                # Horizontal flip
                data['kps'][:, 1] = data['imsize'][1].item() - data['kps'][:, 1] - 1
                data['kps_symm_only'] = data['kps'].clone()
                
                data['kps_symm_only'][~self.KP_WITH_ORIENTATION, 2] = 0
                data['kps'][self.KP_WITH_ORIENTATION, 2] = 0.5
            elif self.KP_LEFT_RIGHT_PERMUTATION is not None:
                for key in ['kps','kps_symm_only']:
                    data[key][:, 1] = data['imsize'][1].item() - data[key][:, 1] - 1
                    data[key] = data[key][self.KP_LEFT_RIGHT_PERMUTATION]

        # Crop
        if self.crops_size!=None:
            i, j, h, w = self.geometric_augmentation_1.get_params(image, scale=self.geometric_augmentation_1.scale, ratio=self.geometric_augmentation_1.ratio)
            # i,j are the top left corner of the crop (firs element for top, second for left)
            # h,w are the height and width of the crop
            # crop the image
            image = F.resized_crop(image, i, j, h, w, (self.crops_size, self.crops_size))
            data['imsize'] = torch.tensor((self.crops_size, self.crops_size))
            # data['bndbox'] is a list of 4 elements [xmin, ymin, xmax, ymax]
            data['bndbox'] = data['bndbox'] - np.array([j, i, j, i])
            # Scale the bounding box to the new size
            data['bndbox'] = data['bndbox'] * np.array([self.crops_size / w, self.crops_size / h, self.crops_size / w, self.crops_size / h])


        for key in ['kps','kps_symm_only']:
            keypoints = data[key]
            visible = keypoints[:, 2]>0.5
            if self.crops_size!=None:
                # Crop
                keypoints[visible, 0] -= i # left
                keypoints[visible, 1] -= j # top
                keypoints[visible, 0] *= self.crops_size/h
                keypoints[visible, 1] *= self.crops_size/w
                keypoints[visible, 2] *= torch.bitwise_and((keypoints[visible, 0] < data['imsize'][0]) , (keypoints[visible, 1] < data['imsize'][1]) ).float()
                keypoints[visible, 2] *= torch.bitwise_and((keypoints[visible, 0] >= 0 ) , (keypoints[visible, 1] >= 0 )).float()
            data[key] = keypoints


        # Color augmentation
        if self.color_augmentation is not None:
            image = self.color_augmentation(image)

        return image, data
