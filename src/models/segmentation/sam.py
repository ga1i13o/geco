import sys
import torch
import torch.nn as nn
# Ensure the segment_anything package is accessible.
sys.path.append("..")
"""pip install git+https://github.com/facebookresearch/segment-anything.git"""
from segment_anything import sam_model_registry, SamPredictor
device = "cuda"
import numpy as np

def get_name():
    return 'sam'
class SAM(nn.Module):
    name = "sam"
    def __init__(self, args):
        super().__init__()
        """wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"""
        sam_checkpoint = f"{args.path_model_seg}sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
        self.sam = SamPredictor(sam)
        return 

    def forward(self, im, bbox=None, kps=None):
        kps = kps[:, [1,0,2]]
        self.sam.set_image(np.array(im))
        if bbox is not None:
            input_box = np.array(bbox)[None, :]
        else:
            input_box = None
        prt, _, _ = self.sam.predict(box=input_box, multimask_output=False, point_coords=kps[:,:2].cpu().numpy(), point_labels=kps[:,2].cpu().numpy())
        prt = torch.tensor(prt[0])
        return prt