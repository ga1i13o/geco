
import torch
from src.models.featurizer.dinov2_lora import DINOv2LoRAFeaturizer
from pathlib import Path
import wandb
import os

def get_featurizer(featurizer_args):
    load_pretrained = 'init' in featurizer_args
    if not load_pretrained:
        if featurizer_args.model == 'dift_sd':
            from src.models.featurizer.dift_sd import SDFeaturizer4Eval
            featurizer = SDFeaturizer4Eval(cat_list=featurizer_args.all_cats)
        elif featurizer_args.model == 'dift_adm':
            from src.models.featurizer.dift_adm import ADMFeaturizer4Eval
            featurizer = ADMFeaturizer4Eval()
        elif featurizer_args.model == 'open_clip':
            from src.models.featurizer.clip import CLIPFeaturizer
            featurizer = CLIPFeaturizer()
        elif featurizer_args.model == 'dino':
            from src.models.featurizer.dino import DINOFeaturizer
            featurizer = DINOFeaturizer(**featurizer_args)
        elif featurizer_args.model == 'dinov2':
            from src.models.featurizer.dinov2 import DINOv2Featurizer
            featurizer = DINOv2Featurizer(**featurizer_args)
        elif featurizer_args.model == 'dinov2lora':
            from src.models.featurizer.dinov2_lora import DINOv2LoRAFeaturizer
            featurizer = DINOv2LoRAFeaturizer(**featurizer_args)
        elif featurizer_args.model == 'sd15ema_dinov2':
            from src.models.featurizer.sd15ema_dinov2 import sd15ema_dinov2_Featurizer
            featurizer = sd15ema_dinov2_Featurizer(**featurizer_args)
        else:
            raise ValueError('featurizer model not supported')
    else:
        featurizer = load_checkpoint(featurizer_args)
    return featurizer

def load_checkpoint(args):
    if args.model == 'dinov2lora':
        model = DINOv2LoRAFeaturizer(**args)
        model_out_path = Path(args.model_out_path).joinpath(args.init.id)
        name = "last_lora_weights.pth" if 'eval_last' in args.init else "best_lora_weights.pth"
        model_out_path_new = model_out_path.joinpath(name)
        model.load_parameters(str(model_out_path_new))
    else:
        raise ValueError
    
    model.id = args.init.id
    model.name = args.init.id
    return model

def load_checkpoint_old(args):
    model_out_path = Path(args.model_out_path).joinpath(args.init.id)
    name = "last.pth" if 'eval_last' in args.init else "best.pth"
    model_out_path_new = model_out_path.joinpath(name)
    if args.model == 'dinov2lora':
        featurizer = DINOv2LoRAFeaturizer(**args)
    try:
        featurizer.load_state_dict(torch.load(model_out_path_new))
    except:
        featurizer = torch.load(model_out_path_new)
    featurizer.id = args.init.id
    featurizer.name = args.init.id
    return featurizer

def save_checkpoint(model, args, name = "lora_weights"):
    id = wandb.run.id if wandb.run else model.id
    model_out_path = Path(args.model_out_path).joinpath(id)
    os.makedirs(model_out_path, exist_ok=True)
    model_out_path_new = model_out_path.joinpath(name+".pth")
    # save state dict
    model.save_parameters(model_out_path_new)

def get_featurizer_name(featurizer_args):
    if featurizer_args.model == 'dift_sd':
        from src.models.featurizer.dift_sd import get_name
        name = get_name(cat_list=featurizer_args.all_cats)
    elif featurizer_args.model == 'dift_adm':
        from src.models.featurizer.dift_adm import get_name
        name = get_name()
    elif featurizer_args.model == 'open_clip':
        from src.models.featurizer.clip import get_name
        name = get_name()
    elif featurizer_args.model == 'dino':
        from src.models.featurizer.dino import get_name
        name = get_name(**featurizer_args)
    elif featurizer_args.model == 'dinov2':
        from src.models.featurizer.dinov2 import get_name
        name = get_name(**featurizer_args)
    elif featurizer_args.model == 'dinov2lora':
        from src.models.featurizer.dinov2_lora import get_name
        name = get_name(**featurizer_args)
    elif featurizer_args.model == 'sd15ema_dinov2':
        from src.models.featurizer.sd15ema_dinov2 import get_name
        name = get_name(**featurizer_args)
    else:
        raise ValueError('featurizer model not supported')
    return name