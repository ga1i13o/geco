import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from src.dataset.cub_200 import CUBDatasetBordersCut, CUBDataset
from src.dataset.spair_single import SpairDatasetSingle
from src.dataset.pascalparts import PascalParts
from src.dataset.apk_pairs import AP10KPairs
from src.dataset.pfpascal_pairs import PFPascalPairs
from src.models.featurizer.utils import get_featurizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path="../configs")
def main(args: DictConfig):
    # convert hydra to dict
    cfg = OmegaConf.to_container(args)
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

    if args.featurizer.model == 'dift_sd':
        args.featurizer.all_cats = dataset.all_cats
    featurizer = get_featurizer(args.featurizer)
    dataset.featurizer_name = featurizer.name
    dataset.featurizer = featurizer
    dataset.featurizer_kwargs = args.featurizer
    print(dataset.featurizer_name)

    overwrite=False
    dataset.store_feats(featurizer, overwrite, args.featurizer)

if __name__ == '__main__':
    main()