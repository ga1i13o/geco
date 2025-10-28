import numpy as np
import torch
from tqdm import tqdm
from src.matcher.argmaxmatcher import ArgmaxMatcher
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import copy
from src.evaluation import STOREBASEPATH
NUM_VIZ = 15
from src.logging.visualization_pck import plot_src, plot_trg

def viz_pck_alpha(dataset, idx, data, src_to_trg_point, heatmap_, store_path, count_viz, key):
    if key == '00':
        return count_viz
    if count_viz[key]<NUM_VIZ: 
        pair_idx = data['idx']
        imgs = dataset.get_imgs(pair_idx)
        if hasattr(dataset, 'KP_EXCLUDED'):
            if dataset.KP_EXCLUDED[idx]:
                return count_viz
        plot_src(imgs[0], data['src_kps'][idx], path=store_path+f'/src_{key}_{count_viz[key]}.png')
        if key == '01':
            trg_point = None
            c = 'g'
            trg_point2 = data['trg_kps_symm_only'][idx]
            c2 = 'r'
        elif key == '11':
            trg_point = data['trg_kps'][idx]
            c = 'g'
            trg_point2 = data['trg_kps_symm_only'][idx]
            c2 = 'r'
        else:
            trg_point = data['trg_kps'][idx]
            c = 'g'
            trg_point2 = None
            c2 = 'r'
        plot_trg(imgs[1], heatmap_, src_to_trg_point=src_to_trg_point, trg_point=trg_point, c=c, trg_point2=trg_point2, c2=c2, path=store_path+f'/trg_{key}_{count_viz[key]}.png', limits=[0,1])
        count_viz[key] = count_viz[key]+1
    return count_viz

def update_counters_pck_alpha(dataset, idx, data, src_to_trg_point, threshold, alpha, match_vis, symm_vis, n_img):
    # match visible, symmcounterpart visible in the target image
    if match_vis and symm_vis:
        key = '11'
        n_img['11'] += 1
        trg_point = data['trg_kps'][idx]
        dist = ((src_to_trg_point[0] - trg_point[0]) ** 2 + (src_to_trg_point[1] - trg_point[1]) ** 2) ** 0.5
        # only symm counterpart is visible in the target image
        trg_point_symm = data['trg_kps_symm_only'][idx]
        dist_symm = ((src_to_trg_point[0] - trg_point_symm[0]) ** 2 + (src_to_trg_point[1] - trg_point_symm[1]) ** 2) ** 0.5
        if (dist / threshold) <= alpha:
            n_img['11_hat'] += 1
            if (dist_symm / threshold) <= alpha:
                n_img['11_tilde'] +=1
        elif (dist_symm / threshold) <= alpha:
            n_img['11_overline'] +=1

    # match visible, symmcounterpart not visible in the target image
    if match_vis and not symm_vis:
        trg_point = data['trg_kps'][idx]
        dist = ((src_to_trg_point[0] - trg_point[0]) ** 2 + (src_to_trg_point[1] - trg_point[1]) ** 2) ** 0.5
        has_orient = dataset.KP_WITH_ORIENTATION[idx] if dataset.pck_symm else False
        if not has_orient:
        # symmcounterpart does not exist
            key = '1x'
            n_img['1x'] += 1
            if (dist / threshold) <= alpha:
                n_img['1x_hat'] += 1
        else:
            # symmcounterpart occluded
            key = '10'
            n_img['10'] += 1
            if (dist / threshold) <= alpha:
                n_img['10_hat'] += 1

    # match not visible, symmcounterpart visible in the target image
    if not match_vis and symm_vis:
        # calculate the distance between the symmetrical point (neg match) in the target image and the predicted point
        # only symm counterpart is visible in the target image
        key = '01'
        n_img['01'] += 1
        trg_point_symm = data['trg_kps_symm_only'][idx]
        dist_symm = ((src_to_trg_point[0] - trg_point_symm[0]) ** 2 + (src_to_trg_point[1] - trg_point_symm[1]) ** 2) ** 0.5
        if (dist_symm / threshold) <= alpha:
            n_img['01_overline'] += 1
            
    # match not visible, symmcounterpart not visible in the target image
    if not match_vis and not symm_vis:
        # no match is visible in the target image
        key = '00'
        n_img['00'] += 1

    return n_img, key

def get_per_point_pck(n_total, cat, alph='0.1'):
    results = {}
    ######### per point pck
    n_total_pck_denom = n_total['10'] + n_total['11'] + n_total['1x']
    n_total_pck_nom = n_total['10_hat'] + n_total['11_hat'] + n_total['1x_hat']
    results['per point PCK@'+alph+' ' + cat] = n_total_pck_nom / n_total_pck_denom * 100
    results['per point PCK@'+alph+' \hat{n}_10/n_10' + cat] = n_total['10_hat'] / n_total['10'] * 100 if n_total['10'] > 0 else 0
    results['per point PCK@'+alph+' \hat{n}_11/n_11' + cat] = n_total['11_hat'] / n_total['11'] * 100 if n_total['11'] > 0 else 0
    results['per point PCK@'+alph+' \hat{n}_1x/n_1x' + cat] = n_total['1x_hat'] / n_total['1x'] * 100 if n_total['1x'] > 0 else 0
    results['per point PCK@'+alph+' \overline{n}_11/n_11 ' + cat] = n_total['11_overline'] / n_total['11'] * 100 if n_total['11'] > 0 else 0
    results['per point PCK@'+alph+' \tilde{n}_11/n_11 ' + cat] = n_total['11_tilde'] / n_total['11'] * 100 if n_total['11'] > 0 else 0
    # results['per point PCK@'+alph+' \overline{n}_01/n_01 ' + cat] = n_total['01_overline'] / n_total['01'] * 100 if n_total['01'] > 0 else 0
    return results

@torch.no_grad()
def evaluate(cat, dataset, upsample, bbox, n_pairs, model_refine=None, path=STOREBASEPATH+'/05_experiments/', visualize=False):
    if model_refine is not None:
        store_path = path+'pck/'+dataset.name+'/'+cat+'/'+model_refine.id+'/'
    else:
        store_path = path+'pck/'+dataset.name+'/'+cat+'/'+dataset.featurizer_name+'/'
    matcher = ArgmaxMatcher()
    results = {}
    count_viz = {'10': 0, '01': 0, '11': 0, '00': 0, '1x': 0}
    # iterate over all categories
    dataset.init_kps_cat(cat)
    # init the counters
    n_total_10 = {'10': 0, '01': 0, '11': 0, '00': 0, '1x': 0, '1x_hat': 0, '10_hat': 0, '11_hat': 0, '11_overline': 0,'11_underline': 0, '01_overline': 0, '11_tilde': 0}
    n_total_05 = copy.deepcopy(n_total_10)
    n_total_15 = copy.deepcopy(n_total_10)
    alpha = 0.1

    # iterate over all test image pairs in the category
    perimg_PCK = []
    for i, data in enumerate(tqdm(dataset)):
        if i == n_pairs:
            break

        # get the data for the pair
        src_ft = data['src_ft'][None].to(device)
        trg_ft = data['trg_ft'][None].to(device)
        if model_refine!=None:
            ft_orig = [src_ft, trg_ft]
            ft_new = [model_refine(f) for f in ft_orig]
            src_ft = ft_new[0]
            trg_ft = ft_new[1]

        # get the spatial resolution of the feature maps to match the original image size, where keypoints are annotated
        src_img_size = data['src_imsize']
        trg_img_size = data['trg_imsize']
        if bbox:
            trg_bndbox = data['trg_bndbox']
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])
        else:
            threshold = max(trg_img_size[0], trg_img_size[1])

        # init the per image counters
        n_img_10 = {'10': 0, '01': 0, '11': 0, '00': 0, '1x': 0, '1x_hat': 0, '10_hat': 0, '11_hat': 0, '11_overline': 0, '01_overline': 0, '11_tilde': 0}
        n_img_05 = copy.deepcopy(n_img_10)
        n_img_15 = copy.deepcopy(n_img_10)

        # iterate over all points in the pair and find the second point in the target image by argmax matching between query feature and all target features
        for idx in range(len(data['src_kps'])):

            # skip the points that are not annotated
            if data['src_kps'][idx][2] == 0:
                continue

            # match the keypoints
            src_point = data['src_kps'][idx]
            ft0, ft1, trg_ft_size = matcher.prepare_one_to_all(src_ft, trg_ft, src_point, src_img_size, trg_img_size, upsample)
            heatmap, prob = matcher(ft0, ft1)
            src_to_trg_point, heatmap_ = matcher.get_one_trg_point(heatmap[0], prob[0], trg_ft_size, trg_img_size)

            match_vis = data['trg_kps'][idx][2] > 0.5
            symm_vis = data['trg_kps_symm_only'][idx][2] > 0.5 if dataset.pck_symm else False

            # update counters
            n_img_10, key_10 = update_counters_pck_alpha(dataset, idx, data, src_to_trg_point, threshold, 0.10, match_vis, symm_vis, n_img_10)
            n_img_05, key_05 = update_counters_pck_alpha(dataset, idx, data, src_to_trg_point, threshold, 0.05, match_vis, symm_vis, n_img_05)
            n_img_15, key_15 = update_counters_pck_alpha(dataset, idx, data, src_to_trg_point, threshold, 0.15, match_vis, symm_vis, n_img_15)
            # visualize the keypoints
            if visualize:
                count_viz = viz_pck_alpha(dataset, idx, data, src_to_trg_point, heatmap_, store_path, count_viz, key_10)

        # update the counters
        for key, value in n_img_10.items():
            n_total_10[key] += value
        for key, value in n_img_05.items():
            n_total_05[key] += value
        for key, value in n_img_15.items():
            n_total_15[key] += value

    n_total_pck_denom = n_total_10['10'] + n_total_10['11'] + n_total_10['1x']

    results_10 = get_per_point_pck(n_total_10, cat, '0.1')
    results.update(results_10)
    results_05 = get_per_point_pck(n_total_05, cat, '0.05')
    results.update(results_05)
    results_15 = get_per_point_pck(n_total_15, cat, '0.15')
    results.update(results_15)

    results['n_10/n ' + cat] = n_total_10['10'] / n_total_pck_denom if n_total_pck_denom > 0 else 0
    results['n_11/n ' + cat] = n_total_10['11'] / n_total_pck_denom if n_total_pck_denom > 0 else 0
    results['n_1x/n ' + cat] = n_total_10['1x'] / n_total_pck_denom if n_total_pck_denom > 0 else 0
    results['n ' + cat] = n_total_pck_denom

    return results, n_total_05, n_total_10, n_total_15