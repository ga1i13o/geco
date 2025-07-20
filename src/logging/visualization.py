import gc
import matplotlib.pyplot as plt
import torch
import numpy as np
from src.matcher.argmaxmatcher import ArgmaxMatcher
from torch import nn
from src.dataset.utils import to_flattened_idx_torch
from PIL import Image

class Demo:

    def __init__(self, imgs, ft, img_size):
        self.ft = ft # N+1, C, H, W
        # check if image is pil image
        if imgs[0].__class__.__name__ != 'PngImageFile':
            imgs = [Image.fromarray(img) for img in imgs]
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size

    def plot_imgs_joint(self, fig_size=3):
        # concatenate the images and plot them
        # pad imgs to the same height
        max_h = max([img.size[1] for img in self.imgs])
        imgs = []
        for i in range(len(self.imgs)):
            if self.imgs[i].size[1] < max_h:
                # add zero padding using pillow
                img = Image.new('RGB', (self.imgs[i].size[0], max_h), (0, 0, 0))
                img.paste(self.imgs[i], (0, 0))
            else:
                img = self.imgs[i]
            imgs.append(img)

        img = np.concatenate([np.array(img) for img in imgs], axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(fig_size*len(self.imgs), fig_size))
        plt.tight_layout()
        ax.imshow(img)
        # no axis
        ax.axis('off')
        return fig, ax
    
    def plot_images(self, fig_size=3):
        # plot the source image and the target images with the heatmap corresponding to the source point
        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size*self.num_imgs, fig_size))
        plt.tight_layout()
        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('Source image $S$')
            else:
                axes[i].set_title('Target image $T$')
        return fig, axes
    
    def plot_src(self, axes, src_point, scatter_size):
        # plot src image
        axes[0].clear()
        axes[0].imshow(self.imgs[0])
        axes[0].axis('off')
        axes[0].scatter(src_point[1], src_point[0], c='r', s=scatter_size)
        # scale the point to the feature map size
        scale0 = torch.tensor(self.ft[0].shape[-2:]).float()/torch.tensor(self.img_size[0]).float()
        src_point_ = (src_point[:2]*scale0).clamp(torch.zeros(2), torch.tensor(self.ft[0].shape[-2:])-1)

        idx_src = to_flattened_idx_torch(src_point_[0], src_point_[1], self.ft[0].shape[-2], self.ft[0].shape[-1])
        axes[0].set_title('source image\nidx_src: %d' % idx_src)

    def plot_heatmap(self, axes, i, heatmap, src_to_trg_point, alpha, scatter_size, limits=None):
        # plot the heatmap
        axes[i].clear()
        # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
        # convert to grayscale
        img_i = self.imgs[i].copy()
        img_i = img_i.convert('L')
        axes[i].imshow(img_i, alpha=1-alpha, cmap=plt.get_cmap('gray'))
        if limits is None:
            vmax = np.max(heatmap)*255
            vmin = np.min(heatmap)*255
        else:
            vmin = limits[0]*255
            vmax = limits[1]*255
        axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].axis('off')
        if src_to_trg_point is not None:
            axes[i].scatter(src_to_trg_point[1], src_to_trg_point[0], c='r', s=scatter_size)
        axes[i].set_title('target image\nmax value: %.2f' % (heatmap).max())

    def plot_matched_heatmap(self, axes, src_points, alpha, scatter_size, upsample):
        matcher = ArgmaxMatcher()
        for i in range(1, self.num_imgs):
            # prepare the feature maps for matching
            ft0, ft1, trg_ft_size = matcher.prepare_one_to_all(self.ft[0], self.ft[i], src_points[0], self.img_size[0], self.img_size[i], upsample=upsample)
            # match the feature maps
            heatmap, prob = matcher(ft0, ft1)
            for j in range(1): # only plot the first source point, in case we have multiple source points
                src_to_trg_point, heatmap_ = matcher.get_one_trg_point(heatmap[j], prob[j], trg_ft_size, self.img_size[i])
                self.plot_heatmap(axes, i, heatmap_, src_to_trg_point, alpha, scatter_size, limits=(0.5, 1))
            del heatmap

    def plot_img_pairs_click(self, fig_size=3, alpha=0.45, scatter_size=70, upsample=True):
        fig, axes = self.plot_images(fig_size)
        def onclick(event):
            if event.inaxes == axes[0]:
                with torch.no_grad():
                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
                    src_point = torch.tensor([y,x,1])
                    self.plot_src(axes, src_point, scatter_size)
                    self.plot_matched_heatmap(axes, src_point[None], alpha, scatter_size, upsample)
                    gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def plot_img_pairs(self, src_point, fig_size=3, alpha=0.45, scatter_size=70, upsample=True):
        fig, axes = self.plot_images(fig_size)
        axes[0].clear()
        axes[0].imshow(self.imgs[0])
        axes[0].axis('off')
        axes[0].scatter(int(src_point[1]), int(src_point[0]), c='r', s=scatter_size) # scatter needs flipped x and y
        axes[0].set_title('source image')
        # plot trg heatmap
        self.plot_matched_heatmap(axes, src_point[None], alpha, scatter_size, upsample)
        plt.show()

    def plot_matches(self, src_points, trg_points, fig_size=3, alpha=0.45, scatter_size=40, title='', path=None):
        # fig, axes = self.plot_images(fig_size)
        fig, axes = self.plot_imgs_joint(fig_size)
        #colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
        colours = [
        (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255),
        (255, 255, 0), (0, 0, 255), (0, 128, 255), (128, 0, 255), (0, 128, 0),
        (128, 0, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
        ]

        # Normalize RGB values (0-255 -> 0-1)
        normalized_colours = [(r/255, g/255, b/255) for r, g, b in colours]
        num_colours = len(colours)
        # matplot colours are in the range [0, 1] 
        colours = [(float(r)/255, float(g)/255, float(b)/255) for r, g, b in colours]
        for j in range(len(src_points)):
            src_point = src_points[j]
            if src_point[2]!=0:
                axes.scatter(int(src_point[1]), int(src_point[0]), color=normalized_colours[j%num_colours], s=scatter_size) # scatter needs flipped x and y
        for j in range(len(trg_points)):
            trg_point = trg_points[j]
            if trg_point[2]!=0:
                offset = self.img_size[0][1]
                axes.scatter(int(trg_point[1])+offset, int(trg_point[0]), color=normalized_colours[j%num_colours], s=scatter_size)
        c = 'r' if 'neg' in title else 'g' if 'pos' in title else 'b'
        if 'neg' in title or 'pos' in title:
            # plot red connections for negative matches
            for j in range(len(src_points)):
                src_point = src_points[j]
                trg_point = trg_points[j]
                if src_point[2]!=0 and trg_point[2]!=0:
                    axes.plot([src_point[1], trg_point[1]+offset], [src_point[0], trg_point[0]], color=c, linewidth=2)
        if 'bin' in title:
            # plot lines from the points to the boundaries
            for j in range(len(src_points)):
                src_point = src_points[j]
                if src_point[2]!=0:
                    axes.plot([src_point[1], 0], [src_point[0], src_point[0]], color=c, linewidth=2)
            for j in range(len(trg_points)):
                trg_point = trg_points[j]
                offset1 = self.img_size[1][1]
                if trg_point[2]!=0:
                    axes.plot([trg_point[1]+offset, offset+offset1-1], [trg_point[0], trg_point[0]], color=c, linewidth=2)
        # fig.suptitle(title)
        plt.show()
        if path is not None:
            fig.savefig(path,bbox_inches='tight')
            plt.close('all')


    def helper_fun(self, src_point, heatmap, axes, alpha, scatter_size, limits):
        # scale the point to the feature map size for indexing
        scale0 = torch.tensor(self.ft[0].shape[-2:]).float()/torch.tensor(self.img_size[0]).float()
        src_point_ = (src_point[:2]*scale0).clamp(torch.zeros(2), torch.tensor(self.ft[0].shape[-2:])-1) # ft indexing

        idx_src = to_flattened_idx_torch(src_point_[0], src_point_[1], self.ft[0].shape[-2], self.ft[0].shape[-1])
        heatmap_ = heatmap[idx_src.long().item()]
        heatmap_ = nn.Upsample(size=self.img_size[1].tolist(), mode='bilinear')(torch.tensor(heatmap_).view(1,1,self.ft[1].shape[-2],self.ft[1].shape[-1])).squeeze(0).squeeze(0).cpu().numpy()
        self.plot_heatmap(axes, 1, heatmap_, None, alpha, scatter_size, limits=limits)