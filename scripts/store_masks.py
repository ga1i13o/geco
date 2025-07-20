import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from src.dataset.cub_200 import CUBDatasetBordersCut, CUBDataset
from src.dataset.spair_single import SpairDatasetSingle
from src.dataset.pascalparts import PascalParts
from src.dataset.apk_pairs import AP10KPairs
from src.dataset.pfpascal_pairs import PFPascalPairs
from src.models.segmentation.sam import SAM

@hydra.main(config_path="../configs")
def main(args: DictConfig):
    # init the dataset
    args.dataset.sup = "sup_original"
    torch.cuda.set_device(0)
    if args.dataset.name == 'spair':
        dataset = SpairDatasetSingle(args.dataset, split=args.dataset.split)
    elif args.dataset.name == 'cub':
        # dataset = CUBDatasetBordersCut(args.dataset, split=args.dataset.split)
        dataset = CUBDataset(args.dataset, split=args.dataset.split)
    elif args.dataset.name == 'pascalparts':
        dataset = PascalParts(args.dataset, split=args.dataset.split)
    elif args.dataset.name == 'ap10k':
        dataset = AP10KPairs(args.dataset, split=args.dataset.split)
    elif args.dataset.name == 'pfpascal':
        dataset = PFPascalPairs(args.dataset, split=args.dataset.split)

    model_seg = SAM(args)
    dataset.model_seg = model_seg
    dataset.return_masks = True
    dataset.model_seg_name = model_seg.name

    overwrite=False
    dataset.store_masks(overwrite)

if __name__ == '__main__':
    main()