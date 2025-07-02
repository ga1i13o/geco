import torch
import numpy as np
from src.evaluation.pck import evaluate as evaluate_pck_cat
from src.evaluation.pck import get_per_point_pck
from src.evaluation.segmentation import evaluate as evaluate_seg_cat
from src.logging.log_results import log_wandb

from src.dataset.cub_200_pairs import CUBPairDataset
from src.dataset.apk_pairs import AP10KPairs
from src.dataset.spair import SpairDataset2
from src.dataset.pfpascal_pairs import PFPascalPairsOrig
from src.dataset.pascalparts import PascalParts
import copy

from src.models.featurizer.utils import get_featurizer_name, get_featurizer
class Evaluation():
    def __init__(self, args, featurizer=None, split='test') -> None:
        self.epoch = None
        self.args = copy.deepcopy(args)
        self.split = split
        torch.cuda.set_device(0)
        self.args.dataset.sup = "sup_original"
        if self.args.dataset.name == 'spair':
            self.dataset_test_pck = SpairDataset2(self.args.dataset, split=split)
            precomputed = False
        elif self.args.dataset.name == 'cub':
            self.args.dataset.borders_cut = False
            self.dataset_test_pck = CUBPairDataset(self.args.dataset, split=split)
            self.args.dataset.borders_cut = True
            precomputed = False
        elif self.args.dataset.name == 'ap10k':
            if hasattr(self.args, 'upsample'):
                self.args.upsample = False
            self.dataset_test_pck = AP10KPairs(self.args.dataset, split=split)
            precomputed = False
        elif self.args.dataset.name == 'pfpascal':
            self.dataset_test_pck = PFPascalPairsOrig(self.args.dataset, split=split)
            precomputed = False
        else:
            self.dataset_test_pck = None
            self.dataset_test_ot_pairs = None
        
        if self.args.dataset.name == 'pascalparts':
            self.dataset_test_segmentation = PascalParts(self.args.dataset, split="test")
            self.dataset_train_segmentation = PascalParts(self.args.dataset, split="train")
            precomputed = False
        else:
            self.dataset_test_segmentation = None
            self.dataset_train_segmentation = None
                
        self.init_featurizer(precomputed, featurizer)
        self.reset_cat()

    def init_featurizer(self, precomputed, featurizer):
        if featurizer is not None:
            if self.args.featurizer.model == 'dift_sd':
                self.args.featurizer.all_cats = self.dataset_test_pck.all_cats
            featurizer_name = featurizer.name
        elif not precomputed:
            featurizer = get_featurizer(self.args.featurizer)
            featurizer_name = featurizer.name
        else:
            featurizer_name = get_featurizer_name(self.args.featurizer)
        # test datasets
        for dataset in [self.dataset_test_pck, self.dataset_test_segmentation, self.dataset_train_segmentation]:
            if dataset is not None:
                if not precomputed:
                    dataset.featurizer_name = featurizer_name
                    dataset.featurizer = featurizer
                    dataset.featurizer_kwargs = self.args.featurizer
                else:
                    dataset.featurizer_name = featurizer_name
                
    def reset_cat(self):
        for dataset in [ self.dataset_test_pck, self.dataset_test_segmentation, self.dataset_train_segmentation]:
            if dataset is not None:
                dataset.all_cats_eval = dataset.all_cats

    def add_suffix(self, result, suffix):
        return {k+suffix:v for k,v in result.items()}
    
    @torch.no_grad()
    def evaluate_pck(self, model_refine=None, n_pairs_eval_pck=None, suffix=''): 
        results = {}
        if self.dataset_test_pck is None:
            return results
        print("evaluate PCK...")

        if n_pairs_eval_pck is  None:
            n_pairs = self.args.n_pairs_eval_pck
        else:
            n_pairs = n_pairs_eval_pck

        # per point PCK (over all categories)
        n_total = {'10': 0, '01': 0, '11': 0, '00': 0, '1x': 0, '1x_hat': 0, '10_hat': 0, '11_hat': 0, '11_overline': 0,'11_underline': 0, '01_overline': 0, '11_tilde': 0}
        n_total_10 = n_total.copy()
        n_total_05 = n_total.copy()
        n_total_15 = n_total.copy()

        for cat in self.dataset_test_pck.all_cats_eval:
            print(f'... for cat: {cat}')
            # evaluate the PCK for each category
            self.dataset_test_pck.init_kps_cat(cat)
            if model_refine is not None:
                model_refine = model_refine.eval()
            cat_results, n_total_05_cat, n_total_10_cat, n_total_15_cat =  evaluate_pck_cat(cat, self.dataset_test_pck, self.args.upsample, self.args.alpha_bbox,  n_pairs=n_pairs, model_refine=model_refine)
            results.update(cat_results)

            # per point PCK (over all categories)
            for key in n_total.keys():
                n_total_05[key] += n_total_05_cat[key]
                n_total_10[key] += n_total_10_cat[key]
                n_total_15[key] += n_total_15_cat[key]
        
        # per point PCK (over all categories)
        for n_total, alph in zip([n_total_10, n_total_05, n_total_15], ['0.1', '0.05', '0.15']):
            results_alph = get_per_point_pck(n_total, '', alph)
            results.update(results_alph)

        if self.split in ['train', 'val']:
            results = self.add_suffix(results, f'_{self.split}')
        if n_pairs_eval_pck is not None:
            results = self.add_suffix(results, f'_{n_pairs}pairs')
        if suffix != '':
            results = self.add_suffix(results, f'_{suffix}')
        if self.epoch is not None:
            results['epoch'] = self.epoch
        log_wandb(results)
        self.results_pck = results

    @torch.no_grad()
    def evaluate_seg(self, model_refine=None, suffix=''):
        results, results_mean = {}, {}
        if self.dataset_test_segmentation is None:
            return results
        print("evaluate Segmentation...")
        modeldict = {'model_refine': model_refine, 'dataset_test': self.dataset_test_segmentation, 'dataset_train': self.dataset_train_segmentation}
        for cat in self.dataset_test_segmentation.all_cats_eval:
            print(f'... for cat: {cat}')
            self.dataset_test_segmentation.init_kps_cat(cat)
            self.dataset_train_segmentation.init_kps_cat(cat)
            if model_refine is not None:
                model_refine = model_refine.eval()
            cat_results = evaluate_seg_cat(self.args, modeldict)
            for k,v in cat_results.items():
                if not k in results_mean.keys():
                    results_mean[k] = []
                results_mean[k].append(v)
            results.update({'segmentation'+cat+'_'+k: v for k, v in cat_results.items()})

        for k,v in results_mean.items():
            results_mean[k] = torch.tensor(v).mean().numpy()
        results.update({'segmentation_mean_'+k: v for k, v in results_mean.items()})
        if self.split in ['train', 'val']:
            results = self.add_suffix(results, f'_{self.split}')
        if suffix != '':
            results = self.add_suffix(results, f'_{suffix}')
        if self.epoch is not None:
            results['epoch'] = self.epoch
        log_wandb(results)
