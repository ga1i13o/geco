from src.evaluation.segmentation_metrics import confusion_matrix, accuracy, mean_precision, mean_recall, mean_iou
from src.models.classifier.utils import train_classifier, forward_classifier
from typing import Dict
import torch.functional as F
import torch
from src.logging.visualization_seg import plot_assignment
from src.evaluation import STOREBASEPATH

def evaluate_img(dataset_test, model_seg, model_refine=None, full_resolution=False, idx=0):
    # get segmentation model
    prt_pred = forward_classifier(dataset_test, idx, model_seg, model_refine=model_refine)

    data = dataset_test[idx]
    prt_gt = data['parts_mask'][None]
    # rescale to fit to gt size
    if full_resolution:
        prt_pred = F.interpolate(prt_pred, size=prt_gt.shape[-2:], mode='bilinear', align_corners=False)
    y_max = len(dataset_test.KP_NAMES)

    indices = prt_gt.sum(1)>1e-20 # only consider foreground pixels
    # generate labels 
    prt_gt = prt_gt.argmax(1)+1
    prt_pred = prt_pred.argmax(1)+1

    prt_pred[~indices] = 0
    prt_gt[~indices] = 0

    conf_matrix = confusion_matrix(prt_pred[indices].detach().cpu()-1, prt_gt[indices].detach().cpu()-1, y_max).detach().cpu()

    cat_metrics: Dict[str, float] = {}
    cat_metrics["acc"] = accuracy(conf_matrix)
    cat_metrics["m_prcn"] = mean_precision(conf_matrix)
    cat_metrics["m_rcll"] = mean_recall(conf_matrix)
    cat_metrics["m_iou"] = mean_iou(conf_matrix)

    return cat_metrics, prt_pred, prt_gt

def evaluate(args, modeldict, full_resolution=False,  path=STOREBASEPATH+'/05_experiments/'):

    y_max = len(modeldict['dataset_train'].KP_NAMES)
    if y_max<2:
        return {}
    modeldict['model_seg']  = train_classifier(args.sup_classifier, modeldict['dataset_train'], model_refine=modeldict['model_refine'])
    pre_list = []
    gt_list = []

    for idx in range(len(modeldict['dataset_test'])):
        _, prt_pred, prt_gt = evaluate_img(modeldict['dataset_test'], modeldict['model_seg'], model_refine=modeldict['model_refine'], full_resolution=full_resolution, idx=idx)
        pre_list.append(prt_pred.flatten())
        gt_list.append(prt_gt.flatten())

    pred = torch.cat(pre_list, 0)
    gt = torch.cat(gt_list, 0)
    conf_matrix_all = confusion_matrix(pred.detach().cpu(), gt.detach().cpu(), y_max+1).detach().cpu()
    assert conf_matrix_all.shape[0] == y_max+1

    def get_results_dict_cat(y_max, conf_matrix, gt, text="", oriented=None):
        results_dict_cat = {}
        # evaluate the confusion matrix for all parts with at least one pixel in the ground truth
        num_gt_pixels = torch.zeros(y_max+1).int()
        for i in range(0, y_max+1):
            num_gt_pixels[i] = torch.sum(gt==i).int()
            if oriented is not None:
                if not oriented[i]:
                    num_gt_pixels[i] = 0
                
        index = num_gt_pixels==0

        conf_matrix = conf_matrix[~index][:,~index]
        results_dict_cat["acc"+text] = accuracy(conf_matrix)
        results_dict_cat["m_prcn"+text] = mean_precision(conf_matrix)
        results_dict_cat["m_rcll"+text] = mean_recall(conf_matrix)
        results_dict_cat["m_iou"+text] = mean_iou(conf_matrix)

        num_gt_pixels = num_gt_pixels[~index]
        conf_matrix_normalized = conf_matrix/num_gt_pixels[None]
        results_dict_cat["acc_precnorm"+text] = accuracy(conf_matrix_normalized)
        results_dict_cat["m_prcn_precnorm"+text] = mean_precision(conf_matrix_normalized)
        results_dict_cat["m_rcll_precnorm"+text] = mean_recall(conf_matrix_normalized)
        results_dict_cat["m_iou_precnorm"+text] = mean_iou(conf_matrix_normalized)
        return results_dict_cat, conf_matrix, conf_matrix_normalized, index
    
    results_dict_cat, conf_matrix, conf_matrix_normalized, index = get_results_dict_cat(y_max, conf_matrix_all, gt)
    visualize = True
    def visualization(conf_matrix, conf_matrix_normalized, index, modeldict, path, text=""):
        if modeldict['model_refine'] is not None:
            store_path = path+'seg/'+modeldict['dataset_test'].name+'/'+modeldict['dataset_test'].cat+'/'+modeldict['model_refine'].id+'/'
        else:
            store_path = path+'seg/'+modeldict['dataset_test'].name+'/'+modeldict['dataset_test'].cat+'/'+modeldict['dataset_test'].featurizer_name+'/'
        # index = assign_dict['conf_matrix_normalized']==torch.nan
        assign_dict = {'conf_matrix': conf_matrix,
                       'conf_matrix_normalized': conf_matrix_normalized}
        kpnames = modeldict['dataset_test'].KP_NAMES.copy()
        kpnames = ['bkg']+kpnames
        kpnames = [kpnames[i] for i in range(len(kpnames)) if i not in torch.where(index)[0]]
        # remove the background
        plot_dict = {k:v[1:,1:] for k,v in assign_dict.items()}
        plot_kpnames = kpnames[1:]
        plot_assignment(plot_dict, plot_kpnames, plot_cmap=True, path=store_path+f'/conf/'+modeldict['dataset_test'].split+"/"+text+"/", limits=[0,  1.0/len(kpnames)])

    if visualize:
        visualization(conf_matrix, conf_matrix_normalized, index, modeldict, path, "all")
    
    oriented = modeldict['dataset_test'].KP_WITH_ORIENTATION
    # add bkg to oriented
    oriented = torch.cat([torch.tensor([True]), torch.tensor(oriented)])
    results_dict_cat_geo, conf_matrix_geo, conf_matrix_normalized_geo, index_geo = get_results_dict_cat(y_max, conf_matrix_all, gt, "_geo", oriented)
    
    if visualize:
        if conf_matrix_geo.shape[0] > 2:
            visualization(conf_matrix_geo, conf_matrix_normalized_geo, index_geo, modeldict, path, "geo")
    
    results_dict_cat.update(results_dict_cat_geo)

    return results_dict_cat
         