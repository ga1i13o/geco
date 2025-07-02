import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision import transforms as T

import torch

def prepare_ViT_images(vit_patch_size, img, img_size):
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        T.ConvertImageDtype(torch.float)
    ])
    img = transform(img)
    w, h = img.shape[-2] - img.shape[-2] % vit_patch_size, img.shape[-1] - img.shape[-1] % vit_patch_size
    img = img[..., :w, :h].float().to(device)
    # check if shape of image has 3 dimensions
    if len(img.shape) == 3:
        img = img[None]
    return img # B, C, W, H

def get_name(log_bin=True, img_size='518', up_ft_index='1', **kwargs):
    name = 'dinov2'
    if log_bin:
        name = 'dinov2_logbin'
    
    model_size = kwargs.get('model_size', "dinov2_vits14")
    if model_size != "dinov2_vits14":
        # remove dinov2_vit from model_size
        model_size = model_size.replace("dinov2_vit", "")
        name += f'_{model_size}'

    return name + f'_%d_upft%d' % (img_size[0], up_ft_index)

class DINOv2Featurizer:
    name = 'dinov2'
    def __init__(self, log_bin=True, img_size='518', up_ft_index='1', **kwargs):
        self.model_size = kwargs.get('model_size', "dinov2_vits14")
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_size).eval().to(device)
        self._use_log_bin = log_bin
        self.name = get_name(log_bin, img_size, up_ft_index, **kwargs)
        self.vit_patch_size = 14

    def get_models(self):
        return [self.model]
    
    def prepare_tokens(self, img_tensor, up_ft_index):
        B, C, W, H = img_tensor.shape
        # out_dict = self.model(pixel_values=img_tensor, return_dict=True, output_hidden_states=True)
        #https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinov2/modeling_dinov2.py#L661
        # out = out_dict.hidden_states[-up_ft_index] # B, w0*h0, D
        out = self.model.get_intermediate_layers(img_tensor, n=up_ft_index)
        out = out[0] # take the output of the the n-th last block
        out = out[:, :, :] # B, w0*h0, D
        D = out.shape[-1]
        out = out.transpose(-2, -1).view(B, D, self.w0, self.h0)
        return out

    @torch.no_grad()
    def forward(self, img, up_ft_index=3, **kwargs):
        # convert pil to tensor
        img_size = kwargs.get('img_size')
        img_tensor = prepare_ViT_images(self.vit_patch_size, img, img_size) # [B, C, W, H]
        self.w0 = img_tensor.shape[-2] // self.vit_patch_size
        self.h0 = img_tensor.shape[-1] // self.vit_patch_size
        out = self.prepare_tokens(img_tensor, up_ft_index) # B, D, w0, h0
        if self._use_log_bin:
            out = self._log_bin(out.permute(0,2,3,1))
            out = out.view(-1,self.w0,self.h0,out.shape[-1]) # B, w0, h0, D'
            out = out.permute(0,3,1,2) # B, D', w0, h0
        return out
    
    def get_w0_h0(self):
        return self.w0, self.h0

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxwxhxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B, w0, h0, d = x.shape
        num_bins = 1 + self.vit_patch_size * hierarchy
        bin_x = x.permute(0, 3, 1, 2) # B, d, w0, h0
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, w0, h0)).to(x.device)
        for y in range(w0):
            for x in range(h0):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < w0 and 0 <= j < h0:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, w0 - 1))
                                temp_j = max(0, min(j, h0 - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, temp_i,
                                                                                                           temp_j]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x