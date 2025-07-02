import torch
from src.logging.log_results import finish_wandb, init_wandb
from src.evaluation.runtime_mem import evaluate
from omegaconf import DictConfig, OmegaConf
import hydra
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from src.models.featurizer_refine import load_checkpoint
from src.dataset.spair_single import SpairDatasetSingle
from src.dataset.cub_200 import CUBDataset
from src.dataset.apk_pairs import AP10KPairs
from src.dataset.pfpascal_pairs import PFPascalPairs
from src.models.featurizer.utils import get_featurizer
import wandb

@hydra.main(config_path="../configs")
def main(args: DictConfig):

    # load the model if feat_refine is part of the config
    if 'feat_refine' in args:
        model_refine = load_checkpoint(args.feat_refine)
    else:
        model_refine = None
        
    cfg = OmegaConf.to_container(args)
    # convert hydra to dict
    if 'feat_refine' in args:
        init_wandb(cfg, f'eval_time_mem {args.feat_refine.init.id}')
    elif 'init' in args.featurizer:
        init_wandb(cfg, f'eval_time_mem {args.featurizer.init.id}')
    else:
        init_wandb(cfg, 'eval_time_mem ')
    if args.dataset.name == "spair":
        dataset_test_time_mem = SpairDatasetSingle(args.dataset, split="test")
    elif args.dataset.name == "cub":
        dataset_test_time_mem = CUBDataset(args.dataset, split="test")
    elif args.dataset.name == 'ap10k':
        dataset_test_time_mem = AP10KPairs(args.dataset, split="test")
    elif args.dataset.name == 'pfpascal':
        dataset_test_time_mem =  PFPascalPairs(args.dataset, split="test")
    
    dataset_test_time_mem.featurizer = get_featurizer(args.featurizer)
    dataset_test_time_mem.featurizer_kwargs = args.featurizer
    results = evaluate(dataset_test_time_mem, args.num_imgs_time_mem, model_refine=model_refine)
    wandb.log(results)
    finish_wandb()

if __name__ == '__main__':
    main()