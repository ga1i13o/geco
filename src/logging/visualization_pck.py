
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import copy

def crop_centered(img, kp_dict):
    if img.size[0] > img.size[1]:
        for k, v in kp_dict.items():
            if v is not None:
                kp_dict[k][1] = kp_dict[k][1] - (img.size[0] - img.size[1]) // 2 # H,W -> W,H from PIL to numpy
                if kp_dict[k][1] < 0:
                    kp_dict[k][1] = 0
                elif kp_dict[k][1] >= img.size[1]:
                    kp_dict[k][1] = img.size[1]-1
        img = img.crop(((img.size[0] - img.size[1]) // 2, 0, (img.size[1] + img.size[0]) // 2, img.size[1]))
    else:
        for k, v in kp_dict.items():
            if v is not None:
                kp_dict[k][0] = kp_dict[k][0] - (img.size[1] - img.size[0]) // 2 # H,W -> W,H from PIL to numpy
                if kp_dict[k][0] < 0:
                    kp_dict[k][0] = 0
                elif kp_dict[k][0] >= img.size[0]:
                    kp_dict[k][0] = img.size[0]-1
        img = img.crop((0, (img.size[1] - img.size[0]) // 2, img.size[0], (img.size[1] + img.size[0]) // 2))
    return img, kp_dict

def crop_centered_heatmap(heatmap):
    if heatmap.shape[0] > heatmap.shape[1]:
        heatmap = heatmap[(heatmap.shape[0] - heatmap.shape[1]) // 2:(heatmap.shape[1] + heatmap.shape[0]) // 2, :]
    else:
        heatmap = heatmap[:, (heatmap.shape[1] - heatmap.shape[0]) // 2:(heatmap.shape[1] + heatmap.shape[0]) // 2]
    return heatmap
    
def plot_src(img_, src_point, alpha=0.2, path=None,  scatter_size=70):
    img = img_
    def plotting():
        ax.clear()
        ax.imshow(img)
        ax.axis('off')
        ax.scatter(src_point[1], src_point[0], edgecolor='black', linewidth=1, facecolor='w', s=scatter_size, label="$Q$ query")
        if path is not None:
            ax.legend(fontsize="20", loc ="upper left")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close('all')
    ##############################################
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plotting()
    # ##############################################
    # crop to quadratic
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    kp_dict = {'src': src_point}
    img,kp_dict = crop_centered(img, kp_dict)
    src_point = kp_dict['src']
    if path is not None:
        # add padding to path
        path = path.split('.')
        path = path[0] + '_crop.' + path[1]
    plotting()

def plot_trg(img_, heatmap, path=None, scatter_size=70, alpha_gt=0.2, alpha=0.45, limits=None, src_to_trg_point=None, trg_point=None, c='g', trg_point2=None, c2='r'):
    img = img_.convert('L').copy()
    if limits is None:
        vmax = np.max(heatmap)*255
        vmin = np.min(heatmap)*255
    else:
        vmin = limits[0]*255
        vmax = limits[1]*255

    def plotting_gt():
        ax.clear()
        ax.imshow(img_, alpha=1-alpha_gt, cmap=plt.get_cmap('gray'))
        ax.axis('off')
        if path_plain is not None:
            os.makedirs(os.path.dirname(path_plain), exist_ok=True)
            plt.savefig(path_plain, bbox_inches='tight', pad_inches=0)
            # plt.close('all')
        if trg_point is not None and trg_point[2] > 0:
            ax.scatter(trg_point[1], trg_point[0], edgecolor='black', linewidth=1, facecolor=c, s=scatter_size, label="$Q_{GT}$ visible \u2714 (1)")
            # # plot text at the bottom of the image with a label for the target point
            # x, y = img.size[0] // 2, img.size[1] - 20
            # s = f'$Q$ \n $Q_s$'
            # plt.text(x, y, s, bbox=dict(fill=True, facecolor='w', linewidth=2))
        else:
            # add to legend without scatter
            ax.scatter([], [], edgecolor='black', linewidth=1, facecolor=c, s=scatter_size, label="$Q_{GT}$ visible \u2718 (0)")
        if trg_point2 is not None and trg_point2[2] > 0:
            ax.scatter(trg_point2[1], trg_point2[0], edgecolor='black', linewidth=1, facecolor=c2, s=scatter_size, label="$Q_{Symm}$ visible \u2714 (1)")
        else:
            # add to legend without scatter
            ax.scatter([], [], edgecolor='black', linewidth=1, facecolor=c2, s=scatter_size, label="$Q_{Symm}$ visible \u2718 (0)")
        if path_gt is not None:
            ax.legend(fontsize="20", loc ="upper left")
            os.makedirs(os.path.dirname(path_gt), exist_ok=True)
            plt.savefig(path_gt, bbox_inches='tight', pad_inches=0)
            plt.close('all')

    def plotting_pred():
        ax.clear()
        ax.imshow(img, alpha=1-alpha, cmap=plt.get_cmap('gray'))
        ax.imshow(255 * heatmap, alpha=alpha, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if src_to_trg_point is not None:
            if heatmap[src_to_trg_point[0], src_to_trg_point[1]] > 0.3:
                ax.scatter(src_to_trg_point[1], src_to_trg_point[0], edgecolor='black', linewidth=1, facecolor='y', s=scatter_size)
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close('all')

    ##############################################
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if path is not None:
        path_list = path.split('.')
        path_plain = path_list[0] + '_plain.' + path_list[1]
        path_gt = path_list[0] + '_gt.' + path_list[1]
    plotting_gt()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plotting_pred()

    ##############################################
    # crop to quadratic
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    kp_dict = {'src': src_to_trg_point, 'trg': trg_point, 'trg2': trg_point2}
    img, kp_dict = crop_centered(img, kp_dict)
    img_, _ = crop_centered(img_, copy.deepcopy(kp_dict))
    src_to_trg_point = kp_dict['src']
    trg_point = kp_dict['trg']
    trg_point2 = kp_dict['trg2']
    heatmap = crop_centered_heatmap(heatmap)
    if path is not None:
        path_list = path.split('.')
        path_plain = path_list[0] + '_plain_crop.' + path_list[1]
        path_gt = path_list[0] + '_gt_crop.' + path_list[1]
    plotting_gt()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if path is not None:
        path_list = path.split('.')
        path = path_list[0] + '_crop.' + path_list[1]
    plotting_pred()