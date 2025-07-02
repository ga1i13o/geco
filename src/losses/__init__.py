from src.dataset.pairwise_utils import get_matches, scale_to_feature_dims, get_y_mat_gt_assignment
from src.matcher.ot_matcher import SoftmaxMatcher
import torch

def pairwise_loss(losses, data, prob, b, prt, masses):
    # evaluate the matches by comparing the ground truth matches with the matches from the model
    data_matches = get_matches(data, b)
    data_matches = scale_to_feature_dims(data_matches, data, b)

    assignment_dict = {}
    ft_orig_b = [data['src_ft'][b], data['trg_ft'][b]]
    ft_size_b = [torch.tensor(f.shape[-2:]) for f in ft_orig_b]

    for prefix in ['pos', 'bin', 'neg']:
        assignment_dict['y_mat_' + prefix] = get_y_mat_gt_assignment(data_matches[prefix+'_src_kps'].clone().detach(), data_matches[prefix+'_trg_kps'].clone().detach(), ft_size_b[0], ft_size_b[1])
        idx = assignment_dict['y_mat_' + prefix].coalesce().indices()
        prob_prefix = prob[idx[0].long(),idx[1].long()]
        # compute the loss
        p = 1/prob.shape[0]
        if prefix in ['pos', 'bin']:
            losses[prefix].append(-torch.log(prob_prefix)*p)
        else:
            losses[prefix].append(-torch.log(1-prob_prefix)*(1-0))

    # compute the loss
    if True:
        prefix = 'neg_fg_bkg'
        idx0 = prt[0]>0
        idx1 = prt[1]==0
        idx0 = torch.cat([idx0, torch.zeros(1).bool().to(idx0.device)]) # add the bin to bool mask
        idx1 = torch.cat([idx1, torch.zeros(1).bool().to(idx1.device)]) # add the bin to bool mask
        prob_prefix = prob[idx0,:][:,idx1]
        losses[prefix].append(-torch.log(1-prob_prefix.flatten())*(1-0))
        idx0 = prt[0]==0
        idx1 = prt[1]>0
        idx0 = torch.cat([idx0, torch.zeros(1).bool().to(idx0.device)])
        idx1 = torch.cat([idx1, torch.zeros(1).bool().to(idx1.device)])
        prob_prefix = prob[idx0,:][:,idx1]
        losses[prefix].append(-torch.log(1-prob_prefix.flatten())*(1-0))

    del data_matches, assignment_dict
    return losses

class PairwiseLoss():
    def __init__(self, args):

        ot_params = {
            'reg':0.1,
            'reg_kl':10,
            'sinkhorn_iterations':10,
            'mass':0.9,
            'bin_score':0.3
        }
        self.args = args
        self.matcher = SoftmaxMatcher(**ot_params)
        self.matcher = self.matcher.eval()        # init dict of losses
        self.prefix = ['pos', 'bin', 'neg', 'neg_fg_bkg']

    def get_loss(self, src_ft_, trg_ft_, data):
        
        losses = {p: [] for p in self.prefix}
        # compute the loss
        B = src_ft_.shape[0]
        for b in range(B):
            prt = [data['src_mask'][b], data['trg_mask'][b]]
            # resize the prt segmentation to feature dimensions
            prt = [torch.nn.functional.interpolate(p[None,None].float(), size=(f.shape[-2], f.shape[-1]), mode='nearest')[0,0].bool() for p,f in zip(prt, [data['src_ft'][b], data['trg_ft'][b]])]
            prt = [p.flatten() for p in prt]
            if prt[0].sum()*prt[1].sum()<1e-10:
                continue
            # get masses
            mass0 = (data['src_kps'][b,:,2].sum()+1)/(data['numkp'][b]+1)*self.matcher.mass
            mass1 = (data['trg_kps'][b,:,2].sum()+1)/(data['numkp'][b]+1)*self.matcher.mass
            prob = self.matcher(src_ft_[b], trg_ft_[b], segmentations=prt, masses=[mass0, mass1])
            # use positive, negative and bin matches to compute the loss
            losses = pairwise_loss(losses, data, prob, b, prt, masses=[mass0, mass1])
        losses = {k: torch.cat(v) for k,v in losses.items()}
        losses = {k: v for k,v in losses.items() if len(v)>0}
        return losses