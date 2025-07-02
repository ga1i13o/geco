import torch
from src.logging.log_results import finish_wandb, init_wandb
from src.evaluation.evaluation import Evaluation
from omegaconf import DictConfig, OmegaConf
import hydra
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from src.models.featurizer_refine import load_checkpoint

@hydra.main(config_path="../configs")
def main(args: DictConfig):

    # load the model if feat_refine is part of the config
    if 'feat_refine' in args:
        model_refine = load_checkpoint(args.feat_refine)
    else:
        model_refine = None
        
    cfg = OmegaConf.to_container(args)

    if 'feat_refine' in args:
        init_wandb(cfg, f'eval_seg {args.feat_refine.init.id}')
    elif 'init' in args.featurizer:
        init_wandb(cfg, f'eval_seg {args.featurizer.init.id}')
    else:
        init_wandb(cfg, 'eval_seg ')

    evaluation = Evaluation(args)
    evaluation.evaluate_seg(model_refine=model_refine)
    finish_wandb()

if __name__ == '__main__':
    main()