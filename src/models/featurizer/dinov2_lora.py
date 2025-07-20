import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision import transforms as T

import torch
import torch.nn as nn
import math

# LoRA for Dinov2
# This code is based on the original DINOv2 implementation and the LoRA implementation from https://github.com/RobvanGastel/dinov2-finetune/blob/main/dino_finetune/model/dino_v2.py

class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = getattr(qkv, 'in_features', 768)  # Default fallback
        self.w_identity = torch.eye(self.dim)

    def forward(self, x) -> torch.Tensor:
        # Compute the original qkv
        qkv = self.qkv(x)  # Shape: (B, N, 3 * org_C)

        # Compute the new q and v components
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        # Add new q and v components to the original qkv tensor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v

        return qkv
    

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
    name = 'dinov2lora'
    if log_bin:
        name = 'dinov2lora_logbin'
    # Handle both string and list inputs for img_size
    if isinstance(img_size, list):
        img_size_val = img_size[0]
    else:
        img_size_val = img_size
    # Handle up_ft_index conversion
    if hasattr(up_ft_index, '__iter__') and not isinstance(up_ft_index, str):
        up_ft_index_val = up_ft_index[0] if len(up_ft_index) > 0 else 1
    else:
        up_ft_index_val = up_ft_index
    # Convert to int, handling ListConfig objects
    try:
        img_size_int = int(img_size_val)
        up_ft_index_int = int(up_ft_index_val)
    except (ValueError, TypeError):
        # Fallback values if conversion fails
        img_size_int = 518
        up_ft_index_int = 1
    return name + f'_%d_upft%d' % (img_size_int, up_ft_index_int)

class DINOv2LoRAFeaturizer(nn.Module):
    name = 'dinov2lora'
    def __init__(self, log_bin=True, img_size='518', up_ft_index='1', **kwargs):
        super().__init__()
        self.model_size = kwargs.get('model_size', "dinov2_vits14")
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_size)
        self.model = self.model.to(device)  # type: ignore
        for param in self.model.parameters():
            param.requires_grad = False
        self._use_log_bin = log_bin
        self.name = get_name(log_bin, img_size, up_ft_index, **kwargs)
        self.vit_patch_size = 14

        self.lora_layers = list(range(len(self.model.blocks)))
        self.w_a = []
        self.w_b = []
        self.r = kwargs.get('lora_rank', 10)

        for i, block in enumerate(self.model.blocks):
            if i not in self.lora_layers:
                continue
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features

            w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, self.r)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, self.r)

            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])

            block.attn.qkv = LoRA(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False).to(device)
        w_b = nn.Linear(r, dim, bias=False).to(device)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def get_models(self):
        return [self]
    
    def prepare_tokens(self, img_tensor, up_ft_index):
        B, C, W, H = img_tensor.shape
        # out_dict = self.model(pixel_values=img_tensor, return_dict=True, output_hidden_states=True)
        #https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinov2/modeling_dinov2.py#L661
        # out = out_dict.hidden_states[-up_ft_index] # B, w0*h0, D
        out = self.model.get_intermediate_layers(img_tensor, n=up_ft_index)  # type: ignore
        out = out[0] # take the output of the the n-th last block
        out = out[:, :, :] # B, w0*h0, D
        D = out.shape[-1]
        out = out.transpose(-2, -1).view(B, D, self.w0, self.h0)
        return out

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
    
    def save_parameters(self, filename: str) -> None:
        """Save the LoRA weights and decoder weights to a .pt file

        Args:
            filename (str): Filename of the weights
        """
        w_a, w_b = {}, {}
        w_a = {f"w_a_{i:03d}": self.w_a[i].weight.to('cpu') for i in range(len(self.w_a))}
        w_b = {f"w_b_{i:03d}": self.w_b[i].weight.to('cpu') for i in range(len(self.w_a))}

        torch.save({**w_a, **w_b}, filename)

    def load_parameters(self, filename: str) -> None:
        """Load the LoRA and decoder weights from a file

        Args:
            filename (str): File name of the weights
        """
        state_dict = torch.load(filename)

        # Load the LoRA parameters
        for i, w_A_linear in enumerate(self.w_a):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key].to(device)
            w_A_linear.weight = nn.Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_b):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key].to(device)
            w_B_linear.weight = nn.Parameter(saved_tensor)