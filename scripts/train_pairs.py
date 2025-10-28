import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import DictConfig, OmegaConf
import hydra
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import copy
from src.models.featurizer.utils import get_featurizer
from src.models.featurizer.utils import save_checkpoint as save_featurizer
from src.logging.log_results import log_wandb_epoch, log_wandb_cfg, log_wandb_ram_usage, finish_wandb, log_wandb, init_wandb
from src.dataset.cub_200_pairs import CUBPairDataset
from src.dataset.spair import SpairDataset2
from src.dataset.apk_pairs import AP10KPairs
from src.dataset.pfpascal_pairs import PFPascalPairs
from src.evaluation.evaluation import Evaluation
from src.models.featurizer_refine import get_model_refine
from src.models.featurizer_refine import save_checkpoint as save_model_refine
from src.losses import PairwiseLoss
import time
from src.dataset.utils import get_multi_cat_dataset

def set_seed(dataset_train, args, epoch):
    if args.dataset.cat == "all":
        # set seed of all the subdatasets of the concatenated dataset
        for dataset in dataset_train.datasets:
            dataset.seed_pairs = epoch
    else:
        dataset_train.seed_pairs = epoch # set the seed for the dataset to get the same pairs for each run, but different pairs for each epoch

def forward_pass(model_refine, ft0, ft1):
    # get the new features
    if model_refine is not None:
        ft_new = [model_refine(f) for f in [ft0, ft1]]
    else:
        ft_new = [ft0, ft1]
    ft_new_ = [f.permute(0,2,3,1).flatten(1,-2) for f in ft_new] # each of shape (B, H*W, C)
    return ft_new_[0], ft_new_[1]

def train_batch(data, model_refine, loss_fun):
    torch.cuda.empty_cache()
    ft0, ft1 = forward_pass(model_refine, data['src_ft'], data['trg_ft'])
    losses = loss_fun.get_loss(ft0, ft1, data)
    return losses

from tqdm import tqdm
def train_epoch(train_loader, model_refine, optimizer, weights, loss_fun, args, epoch, evaluation_test, evaluation_train, evaluation_val, evaluation_val_gen, evaluation_val_gen2, evaluation_val_gen3, best_pck, featurizer=None):
    # iterate over the dataset
    I = len(train_loader)
    for i, data in enumerate(tqdm(train_loader, ncols=80)):
        start_loss = time.time()
        if model_refine is not None:
            model_refine.train()
        else:
            featurizer.train()
        losses = train_batch(data, model_refine, loss_fun)
        # compute the mean loss
        losses = {k: v.mean()*weights[k] for k,v in losses.items()}
        # backprop
        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        optimizer.step()
        # measure runtime of loss computation
        timediff_loss = time.time()-start_loss
        log_wandb({'loss_time':timediff_loss})
        # log the losses, ram usage and evaluation
        if ((epoch-1)*I+i+1) % round(1*(6/args.batch_size)) == 0:
            log_wandb_epoch(epoch-1+i/I)
        if ((epoch-1)*I+i+1) % round(50*(6/args.batch_size)) == 0:
            losses['epoch'] = epoch-1+i/I
            log_wandb(losses)
            print ('Epoch [{}/{}], Step [{}/{}] Losses saved'.format(epoch, args.epoch,i,I))
        if ((epoch-1)*I+i+1) % round(1000*(6/args.batch_size)) == 0:
            if model_refine is not None:
                model_refine.eval()
            else:
                featurizer.eval()
            start_eval = time.time()
            log_wandb_ram_usage()
            print ('Epoch [{}/{}], Step [{}/{}] Eval start'.format(epoch, args.epoch,i,I))
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for eva in [evaluation_test, evaluation_train, evaluation_val]:
                eva.epoch = epoch-1+i/I
            # for eva in [evaluation_test, evaluation_train, evaluation_val, evaluation_val_gen, evaluation_val_gen2, evaluation_val_gen3]:
            #     eva.epoch = epoch-1+i/I
            print("Epoch [{}/{}], Step [{}/{}] Eval on val set".format(epoch, args.epoch,i,I))
            n_eval_imgs = 10
            evaluation_val.evaluate_pck(model_refine, n_pairs_eval_pck=n_eval_imgs, suffix=evaluation_val.dataset_test_pck.name)
            # evaluation_val_gen.evaluate_pck(model_refine, n_pairs_eval_pck=n_eval_imgs, suffix=evaluation_val_gen.dataset_test_pck.name)
            # evaluation_val_gen2.evaluate_pck(model_refine, n_pairs_eval_pck=n_eval_imgs, suffix=evaluation_val_gen2.dataset_test_pck.name)
            # evaluation_val_gen3.evaluate_pck(model_refine, n_pairs_eval_pck=n_eval_imgs, suffix=evaluation_val_gen3.dataset_test_pck.name)
            log_wandb_ram_usage()
            print ('Epoch [{}/{}], Step [{}/{}] Eval end'.format(epoch, args.epoch,i,I))
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # process checkpoint 
            suffix = evaluation_val.dataset_test_pck.name
            val_pck =evaluation_val.results_pck[f'per point PCK@0.1 _val_{n_eval_imgs}pairs_{suffix}'] 
            # suffix = evaluation_val_gen.dataset_test_pck.name
            # val_pck+=evaluation_val_gen.results_pck[f'per point PCK@0.1 _val_{n_eval_imgs}pairs_{suffix}']
            # suffix = evaluation_val_gen2.dataset_test_pck.name
            # val_pck+=evaluation_val_gen2.results_pck[f'per point PCK@0.1 _val_{n_eval_imgs}pairs_{suffix}']
            # suffix = evaluation_val_gen3.dataset_test_pck.name
            # val_pck+=evaluation_val_gen3.results_pck[f'per point PCK@0.1 _val_{n_eval_imgs}pairs_{suffix}']
            timediff_eval = time.time()-start_eval
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print(f"Epoch [{epoch}/{args.epoch}], Step [{i}/{I}] Eval time: {timediff_eval:.2f} s, PCK@0.1: {val_pck:.2f}")
            log_wandb({'eval_time':timediff_eval})
            if val_pck> best_pck:
                best_pck = val_pck
                if model_refine is not None:
                    save_model_refine(model_refine, args.feat_refine, "best")
                else:
                    save_featurizer(featurizer, args.featurizer, "best_lora_weights")

            if model_refine is not None:
                model_refine.train()
            else:
                featurizer.train()
        if ((epoch-1)*I+i+1) % round(1000*(6/args.batch_size)) == 0:
            if model_refine is not None:
                model_refine.epoch = epoch-1+i/I
                save_model_refine(model_refine, args.feat_refine, "last")
            else:
                featurizer.epoch = epoch-1+i/I
                save_featurizer(featurizer, args.featurizer, "last_lora_weights")
    if model_refine is not None:
        save_model_refine(model_refine, args.feat_refine, "last")
    else:
        save_featurizer(featurizer, args.featurizer, "last_lora_weights")
    return best_pck

@hydra.main(config_path="../configs")
def main(args: DictConfig):
    cfg = OmegaConf.to_container(args)
    init_wandb(cfg, 'train_pairs')
    torch.cuda.set_device(0)
    # init the dataset
    dataset_list = []
    for dataset_name in ['dataset', 'dataset2', 'dataset3', 'dataset4']:
        if dataset_name in args:
            dataset_args = args[dataset_name]
            if dataset_args.name == 'spair':
                dataset_list.append(SpairDataset2(dataset_args, split='train'))
            elif dataset_args.name == 'cub':
                dataset_list.append(CUBPairDataset(dataset_args, split='train'))
            elif dataset_args.name == 'ap10k':
                dataset_list.append(AP10KPairs(dataset_args, split='train'))
            elif dataset_args.name == 'pfpascal':
                dataset_list.append(PFPascalPairs(dataset_args, split='train'))
    # init the featurizer
    log_wandb_ram_usage()
    featurizer = get_featurizer(args.featurizer)
    log_wandb_ram_usage()
    # init the evaluation
    evaluation_test = Evaluation(args, featurizer)
    evaluation_train = Evaluation(args, featurizer, split='train')
    evaluation_val = Evaluation(args, featurizer, split='val')
    log_wandb_ram_usage()
    if args.dataset.cat == "all" or 'dataset2' in args:
        # avoid deep copy of the featurizer
        dataset_train_multi = get_multi_cat_dataset(dataset_list, featurizer, args.featurizer, model_seg_name=args.model_seg_name)
        train_loader = torch.utils.data.DataLoader(dataset_train_multi, batch_size=args.batch_size, shuffle=True)
        for dataset in dataset_train_multi.datasets:
            print(f"Number of pairs in dataset {dataset.name} {dataset.cat}: {len(dataset)}")
    else:
        # We train for each category separately
        dataset_list[-1].featurizer = featurizer
        dataset_list[-1].featurizer_kwargs = args.featurizer
        dataset_list[-1].init_kps_cat(args.dataset.cat)
        dataset_list[-1].model_seg_name = args.model_seg_name 
        dataset_list[-1].return_masks = True
        train_loader = torch.utils.data.DataLoader(dataset_list[-1], batch_size=args.batch_size, shuffle=True)
    log_wandb_ram_usage()
    weights = {'pos':args.losses.pos, 'bin':args.losses.bin, 'neg':args.losses.neg, 'neg_fg_bkg':args.losses.neg_fg_bkg}
    log_wandb_cfg({'weights':weights})
    # init the model
    ep=0
    if 'feat_refine' in args:
        model_refine = get_model_refine(args.feat_refine)
        log_wandb_ram_usage()
        model_refine = model_refine.to(device)
        model_refine = model_refine.train()
        log_wandb_ram_usage()
        # define the optimizer
        # optimizer = torch.optim.Adam(list(model_refine.parameters())+list(matcher.parameters()), lr=args.learning_rate)
        optimizer = torch.optim.AdamW(list(model_refine.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        featurizer = featurizer.to(device)
        featurizer = featurizer.train()
        model_refine = None
        lora_params = [p for p in featurizer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    # define the scheduler
    if args.scheduler is not None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=len(dataset_list[-1])//args.batch_size, epochs=args.epoch, pct_start=0.3)
    else:
        scheduler = None
    # add evaluation for different datasets
    argsgen = copy.deepcopy(args)
    # argsgen.dataset = argsgen.datasetgeneralization
    # evaluation_val_gen = Evaluation(argsgen, featurizer, split='val')
    # argsgen2 = copy.deepcopy(args)
    # argsgen2.dataset = argsgen.datasetgeneralization2
    # evaluation_val_gen2 = Evaluation(argsgen2, featurizer, split='val')
    # argsgen3 = copy.deepcopy(args)
    # argsgen3.dataset = argsgen.datasetgeneralization3
    # evaluation_val_gen3 = Evaluation(argsgen3, featurizer, split='val')
    # define the loss function
    loss_fun = PairwiseLoss(args)
    # start training
    best_pck = 0
    evaluation_val_gen, evaluation_val_gen2, evaluation_val_gen3 = None, None, None
    for epoch in range(ep+1, args.epoch+1):
        # set seed
        set_seed(train_loader.dataset, args, epoch)

        best_pck = train_epoch(train_loader, model_refine, optimizer, weights, loss_fun, args, epoch, evaluation_test, evaluation_train, evaluation_val, evaluation_val_gen, evaluation_val_gen2, evaluation_val_gen3, best_pck, featurizer=featurizer)
        if scheduler is not None:
            scheduler.step()
    # evaluate the best model
    if model_refine is not None:
        OmegaConf.update(args, "feat_refine.init.load_pretrained", True, force_add=True)
        OmegaConf.update(args, "feat_refine.init.id", model_refine.id, force_add=True)
        model_refine = get_model_refine(args.feat_refine)
        evaluation_test.reset_cat()
        evaluation_test.evaluate_pck(model_refine)
    finish_wandb()

if __name__ == '__main__':
    main()
    