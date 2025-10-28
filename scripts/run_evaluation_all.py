import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.logging.log_results import finish_wandb, init_wandb
from src.evaluation.evaluation import Evaluation
from omegaconf import DictConfig, OmegaConf
import hydra
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from src.models.featurizer_refine import load_checkpoint as load_checkpoint_refiner
from src.models.featurizer.utils import get_featurizer
import copy

@hydra.main(config_path="../configs")
def main(args: DictConfig):

    # load the model if feat_refine is part of the config
    if 'feat_refine' in args:
        model_refine = load_checkpoint_refiner(args.feat_refine)
    else:
        model_refine = None

    featurizer = get_featurizer(args.featurizer)

    cfg = OmegaConf.to_container(args)
    # convert hydra to dict
    # if 'feat_refine' in args:
    #     init_wandb(cfg, f'eval {args.feat_refine.init.id}')
    # elif 'init' in args.featurizer:
    #     init_wandb(cfg, f'eval {args.featurizer.init.id}')
    # else:
    #     init_wandb(cfg, 'eval ')

    evaluation_0 = Evaluation(args, featurizer)
    # args_0 = copy.deepcopy(args)
    # args_0.dataset = args.dataset1
    # evaluation_1 = Evaluation(args_0, featurizer)
    # args_2 = copy.deepcopy(args)
    # args_2.dataset = args.dataset2
    # evaluation_2 = Evaluation(args_2, featurizer)
    # args_3 = copy.deepcopy(args)
    # args_3.dataset = args.dataset3
    # evaluation_3 = Evaluation(args_3, featurizer)

    evaluation_0.evaluate_pck(model_refine=model_refine, suffix=evaluation_0.dataset_test_pck.name)
    # evaluation_1.evaluate_pck(model_refine=model_refine, suffix=evaluation_1.dataset_test_pck.name)
    # evaluation_2.evaluate_pck(model_refine=model_refine, suffix=evaluation_2.dataset_test_pck.name)
    # evaluation_3.evaluate_pck(model_refine=model_refine, suffix=evaluation_3.dataset_test_pck.name)

    # evaluation_val_0 = Evaluation(args, featurizer, split='val')
    # args_val_0 = copy.deepcopy(args)
    # args_val_0.dataset = args.dataset1
    # evaluation_val_1 = Evaluation(args_val_0, featurizer, split='val')
    # args_val_2 = copy.deepcopy(args)
    # args_val_2.dataset = args.dataset2
    # evaluation_val_2 = Evaluation(args_val_2, featurizer, split='val')
    # args_val_3 = copy.deepcopy(args)
    # args_val_3.dataset = args.dataset3
    # evaluation_val_3 = Evaluation(args_val_3, featurizer, split='val')

    # n_eval_imgs = None
    # evaluation_val_0.evaluate_pck(model_refine=model_refine, n_pairs_eval_pck=n_eval_imgs,  suffix=evaluation_val_0.dataset_test_pck.name)
    # evaluation_val_1.evaluate_pck(model_refine=model_refine, n_pairs_eval_pck=n_eval_imgs, suffix=evaluation_val_1.dataset_test_pck.name)
    # evaluation_val_2.evaluate_pck(model_refine=model_refine, n_pairs_eval_pck=n_eval_imgs, suffix=evaluation_val_2.dataset_test_pck.name)
    # evaluation_val_3.evaluate_pck(model_refine=model_refine, n_pairs_eval_pck=n_eval_imgs, suffix=evaluation_val_3.dataset_test_pck.name)

    # finish_wandb()

if __name__ == '__main__':
    main()