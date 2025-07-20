import torch
from torch import nn
from torch.nn import functional as F

class ArgmaxMatcher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src_vec, trg_vec):
        dev = src_vec.device
        n,d = src_vec.shape
        m,d = trg_vec.shape
        # match the feature vectors at the source point
        src_vec = F.normalize(src_vec) # 1, C
        trg_vec = F.normalize(trg_vec).transpose(0, 1) #C,  H0W0
        cos_map = torch.mm(src_vec, trg_vec) #1, H0W0
        # get the probability map, which is a one-hot map
        #trg_point = torch.unravel_index(cos_map.argmax(), cos_map.shape)
        prob = torch.zeros(n+1, m+1).to(dev)
        idx_0 = torch.arange(n).to(dev)
        idx_1 = cos_map.argmax(dim = 1)
        prob[idx_0, idx_1] = 1
        return cos_map, prob

    def prepare_one_to_all(self, src_ft, trg_ft, src_point, src_img_size, trg_img_size, upsample):
        # We use the feature maps to match src_point to the all target points

        # upsample the feature maps to match the original image size
        if upsample:
            # print(f"Memory allocated ups: {torch.cuda.memory_allocated()//1024**3} GB")
            src_ft = nn.Upsample(size=src_img_size.tolist(), mode='bilinear')(src_ft)
            # print(f"Memory allocated: {torch.cuda.memory_allocated()//1024**3} GB")
            trg_ft = nn.Upsample(size=trg_img_size.tolist(), mode='bilinear')(trg_ft)
            # print(f"Memory allocated: {torch.cuda.memory_allocated()//1024**3} GB")
        
        # scale the source point to the feature map size, in case we did not upsample the feature maps
        # get the scale factor of the feature maps wrt the original image size
        src_ft_size = src_ft.shape[-2:]
        src_scale = torch.tensor(src_ft_size).float()/src_img_size.float()
        src_point_ = src_point.clone()
        src_point_[:2] = src_point[:2] * src_scale
        src_point_ = src_point_.floor().long()

        # get the feature vector at the source point
        num_channel = src_ft.size(1)
        src_vec = src_ft[0, :, src_point_[0], src_point_[1]].view(1, num_channel) # 1, C
        # get the feature vectors at all target points
        trg_vec = trg_ft.reshape(num_channel, -1).transpose(0, 1) # H0W0, C
        # get the size of the target feature map
        trg_ft_size = trg_ft.shape[-2:]
        return src_vec, trg_vec, trg_ft_size
    
    def get_one_trg_point(self, cos_map, prob, trg_ft_size, trg_img_size):
        # Input: 
        # trg_ft_size: H0, W0 indicating the size of cos_map and prob
        # trg_img_size: H, W indicating the size of the original image
        # in case upsample is True, trg_ft_size=trg_img_size
        # Output:
        # trg_point: the target point in the original image size
        # cos_map: the cosine map in the original image size
        H0,W0 = trg_ft_size[-2], trg_ft_size[-1]
        trg_scale = trg_img_size.float()/torch.tensor(trg_ft_size).float()
        cos_map = cos_map.view(H0,W0) # H0,W0
        bin_prob = prob[-1]
        prob = prob[:-1].view(H0,W0)
        trg_point = torch.tensor(torch.unravel_index(prob.argmax(), prob.shape))
        if prob[trg_point[0], trg_point[1]]<bin_prob:
            trg_point = None
        else:
            # scale the target point to the original image size
            trg_point = (trg_point * trg_scale).floor().long() 
        # scale the cosine map to the original image size
        cos_map = nn.Upsample(size=trg_img_size.tolist(), mode='bilinear')(cos_map.view(1, 1, H0, W0)).squeeze(0).squeeze(0)
        if trg_point is not None:
            return trg_point.detach().cpu().numpy(), cos_map.detach().cpu().numpy()
        else:
            return None, cos_map.detach().cpu().numpy()
