
from .feat_refine_geosc import AggregationNetwork
from src.dataset.cub_200 import CUBDatasetBordersCut
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import wandb
import os
def get_init_dataset(args):
    dataset_pca = CUBDatasetBordersCut(args.init.dataset, split='train')
    from src.models.featurizer.utils import get_featurizer
    if args.init.featurizer.model == 'dift_sd':
        args.init.featurizer.all_cats = dataset_pca.all_cats
    featurizer = get_featurizer(args.init.featurizer)
    dataset_pca.init_kps_cat(args.init.dataset.cat)
    dataset_pca.featurizer = featurizer
    dataset_pca.featurizer_kwargs = args.init.featurizer
    return dataset_pca

def get_model_refine(args):
    load_pretrained = 'init' in args
    if not load_pretrained:
        if args.model == 'GeoSc':
            model_refine = AggregationNetwork(feature_dims=args.feature_dims, projection_dim=args.projection_dim, device='cuda', feat_map_dropout=args.feat_map_dropout)
        model_refine.id = wandb.run.id if wandb.run!=None else ""
    else:
        model_refine = load_checkpoint(args)
    return model_refine

def load_checkpoint(args):
    model_out_path = Path(args.model_out_path).joinpath(args.init.id)
    name = "last.pth" if 'eval_last' in args.init else "best.pth"
    name = "{}.pth".format(args.init.epoch) if args.model == 'GeoSc' else name
    model_out_path_new = model_out_path.joinpath(name)
    if args.model == 'GeoSc':
        # check if the model exists
        if not model_out_path_new.exists():
            # wget the model
            os.mkdir(model_out_path)
            os.system(f"wget -O {model_out_path_new} {args.init.url}")
        model_refine = AggregationNetwork(feature_dims=args.feature_dims, projection_dim=args.projection_dim, device='cuda')
    try:
        model_refine.load_state_dict(torch.load(model_out_path_new))
    except:
        model_refine = torch.load(model_out_path_new)
    model_refine.id = args.init.id
    return model_refine

def save_checkpoint(model_refine, args, name):
    id = wandb.run.id if wandb.run else "test"
    model_out_path = Path(args.model_out_path).joinpath(id)
    os.makedirs(model_out_path, exist_ok=True)
    model_out_path_new = model_out_path.joinpath(name+".pth")
    # save state dict
    torch.save(model_refine.state_dict(), model_out_path_new)
