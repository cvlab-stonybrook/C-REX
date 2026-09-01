import math
import numpy as np

import torch

import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    """

    def __init__(self, cost_class: float = 2, cost_bbox: float = 5, cost_point: float = 5, **kwargs):
        """Creates the matcher
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_point = cost_point
        assert cost_class != 0 or cost_bbox != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets): 
        """ Performs the matching
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].sigmoid().flatten(0, 1)  
        out_point = outputs["pred_points"].flatten(0, 1) 

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]) 
        tgt_point = torch.cat([v["points"] for v in targets]) 

        """ class cost """
        out_prob_exp = out_prob.unsqueeze(1).expand(-1, tgt_ids.shape[0], -1)
        # change nan value of out_prob_exp to 0
        out_prob_exp[torch.isnan(out_prob_exp)] = 0
        tgt_ids_exp = tgt_ids.unsqueeze(0).expand(out_prob.shape[0], -1, -1)
        
        assert (out_prob_exp >= 0).all() and (out_prob_exp <= 1).all(), "Invalid values in out_prob_exp"
        assert (tgt_ids_exp >= 0).all() and (tgt_ids_exp <= 1).all(), "Invalid values in tgt_ids_exp"

        losses = F.binary_cross_entropy(out_prob_exp, tgt_ids_exp, reduction="none")
        cost_class = losses.mean(dim=2)

        """ point cost """
        cost_point = torch.cdist(out_point, tgt_point, p=1) # (bs*nq, sum(n_b))


        # cost
        sizes = [len(v["points"]) for v in targets]
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # i: predicted, j: target



def sigmoid_focal_loss(
    inputs, targets, caption_sizes=None, alpha: float = 0.25, gamma: float = 2, no_reduction=False
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid() 
    ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if no_reduction:
        return loss
    else:

        loss_mean_batch = loss.mean(dim=1) # (bs,256)
        loss_mean_batch_sum = loss_mean_batch.sum(dim=1)/caption_sizes # (bs)
        loss_mean = loss_mean_batch_sum.mean()        

        return loss_mean



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
        1) compute hungarian assignment between ground truth boxes and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction (supervise class and localization)
    """
    def __init__(self, args=None):
        # Create the criterion.

        super().__init__()
        self.matcher = HungarianMatcher(cost_class=5, cost_point=1)
        if args is None:
            self.losses = ['labels', 'points', 'contrast']
            self.weight_dict = {'loss_label': 5, 'loss_point': 1, 'loss_contrast': 0.06}
        else:
            self.losses = args.losses
            self.weight_dict = args.weight_dict

        self.loss_map = {
            x: getattr(self, "loss_" + x.rstrip("s"), None) for x in self.losses
        }


    def loss_label(self, outputs, targets, indices, num_points, caption_sizes, **kwargs): 
        """Classification loss 
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] 

        idx = self._get_src_permutation_idx(indices) # tuple ((b#1,b#1,b#1,b#1..., b#2,b#2,b#2...), (pred_idx, pred_idx, pred_idx...)) # 0, 1 size: sum(nt)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # (nt, 256) re-ordered in pred_idx order but using target labels
        target_classes = torch.full(src_logits.shape, 0.0,
                                    dtype=torch.int64, device=src_logits.device) # (bs, nq, 256)
        target_classes[idx] = target_classes_o.to(torch.int64) # set nt out of nq to target labels, others remain 0

        target_classes = target_classes.type_as(src_logits)
        caption_sizes = torch.cat([c.view(1) for c in caption_sizes], dim=0).to(device) 
        loss_label = sigmoid_focal_loss(src_logits, target_classes, caption_sizes=caption_sizes)


        losses = {'loss_label': loss_label}
        
        return losses

    def loss_point(self, outputs, targets, indices, num_points, **kwargs):
        
        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx] # (nt, 2)
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0) # (nt, 2)

        loss_point = F.l1_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_point.sum() / num_points
        
        return losses

    def loss_contrast(self, outputs, targets, indices, num_points, mask_bi=None, **kwargs):
        assert 'img_embs' in outputs and 'txt_embs' in outputs
        idx = self._get_src_permutation_idx(indices) 
        img_embs = outputs['img_embs'] 
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']

        if img_embs.shape[0] == 1: # only one caption and one image in batch
            return {'loss_contrast': torch.zeros(1).to(device)}
        
        loss_list = []
        for i, img_emb in enumerate(img_embs): 
            bi = mask_bi[i] # index of image in the batch e.g [0,0,0,1,1,1]
            matched_pred_idxes = idx[1][torch.where(idx[0] == bi)[0]] 
            matched_img_tokens = img_emb[matched_pred_idxes] 

            # positive text embedding
            pos_token_mask = token_masks[i].unsqueeze(0)  # (1, longest_cap) for the i-th img/caption
            pos_txt = txt_embs[i].unsqueeze(0)  # (1, longest_cap, 256)
            pos_txt_emb_sum = torch.sum(pos_txt * pos_token_mask.unsqueeze(-1), dim=1) 
            pos_txt_emb = pos_txt_emb_sum / pos_token_mask.sum(dim=1, keepdim=True) # (1,256)

            # negative text embedding
            idxes_for_img = np.where(np.array(mask_bi) == bi)[0]
            neg_idxes = [k for k in idxes_for_img if k != i]

            if len(neg_idxes) != 0:
                neg_token_masks = token_masks[neg_idxes] 
                neg_txts = txt_embs[neg_idxes]  
                neg_txt_embs_sum = torch.sum(neg_txts * neg_token_masks.unsqueeze(-1), dim=1)
                neg_txt_embs = neg_txt_embs_sum / neg_token_masks.sum(dim=1, keepdim=True) # (num_cap-1, 256)

                loss = self.contrastive_loss(matched_img_tokens, pos_txt_emb, neg_txt_embs)

            else: # only one caption for image
                loss = torch.tensor(0.0).to(device)
            
            loss_list.append(loss)
        
        loss_contrast = torch.stack(loss_list).mean()

        losses = {}
        losses['loss_contrast'] = loss_contrast
        
        return losses

    def contrastive_loss(self, img_embs, pos_txt_emb, neg_txt_embs):    
        img_embs = F.normalize(img_embs, p=2, dim=1)
        pos_txt_emb = F.normalize(pos_txt_emb, p=2, dim=1)
        neg_txt_embs = F.normalize(neg_txt_embs, p=2, dim=1)
        
        pos_sim = torch.mm(img_embs, pos_txt_emb.T)
        neg_sims = torch.mm(img_embs, neg_txt_embs.T)

        pos_labels = torch.ones(pos_sim.size()).to(pos_sim.device)
        neg_labels = torch.zeros(neg_sims.size()).to(neg_sims.device)

        pos_loss = F.binary_cross_entropy_with_logits(pos_sim, pos_labels)
        neg_loss = F.binary_cross_entropy_with_logits(neg_sims, neg_labels)

        total_loss = (pos_loss + neg_loss) / 2

        return total_loss
    
    #   Adding supervised contrastive loss here -- KOSTAS - PANOS
    def loss_contrast_txt(self, outputs, targets, indices, num_points, num_points_per_cap, mask_bi=None, temperature=0.07, **kwargs):
        assert 'img_embs' in outputs and 'txt_embs' in outputs

        # torch.autograd.set_detect_anomaly(True)

        idx = self._get_src_permutation_idx(indices)
        img_embs = outputs['img_embs'] 
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']

        d = math.sqrt(img_embs.shape[-1])
        affinity = torch.einsum('bnc,blc->bnl', img_embs, txt_embs) / d
    
        sfm_mask = token_masks.unsqueeze(1).repeat(1, img_embs.shape[1], 1).logical_not()
        affinity.masked_scatter_(sfm_mask, -torch.inf * torch.ones_like(affinity))

        affinity = F.softmax(affinity, dim=-1)

        obj_txt_embs = torch.einsum("bnl,blc->bnc", affinity, txt_embs)

        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1) 
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True) # (b, 256)

        obj_txt_embs_norm = F.normalize(obj_txt_embs, p=2, dim=2)  # Shape: (num_images, embedding_dim)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)  # Shape: (1, embedding_dim)

        token_text_sim = torch.matmul(obj_txt_embs_norm, txt_embs_norm.unsqueeze(2)).squeeze(-1)



        """top_n_values, top_n_indices = torch.topk(token_text_sim, num_points_per_cap, largest=True, dim=1)
        pos = torch.gather(img_embs, 1, top_n_indices.unsqueeze(-1).expand(-1, -1, img_embs.shape[-1]))"""
        
        idx = []
        t_n = []
        for i in range(token_text_sim.shape[0]):
            _, top_n_indices = torch.topk(token_text_sim[i], num_points_per_cap[i].item(), dim=-1)
            t_ = torch.zeros_like(obj_txt_embs[0,:,0], dtype=torch.int)
            t_.scatter_(0, top_n_indices, 1)
            t_n.append(top_n_indices)
            idx.append(t_)

        t_ = torch.stack(idx, 0)

        dot_product_tempered = torch.matmul(obj_txt_embs_norm, obj_txt_embs_norm.permute(0, 2, 1)) / temperature

        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[1])).to(device)
        exp_msk_me = exp_dot_tempered * mask_anchor_out
        d_ = torch.sum(exp_msk_me, dim=1)
        ll = []
        for i in range(token_text_sim.shape[0]):
            denom = torch.gather(d_[i], 0, t_n[i])
            numm = torch.sum(exp_msk_me[i] * t_[i].unsqueeze(0), dim=1)
            numm = torch.gather(numm, 0, t_n[i])

            l = numm / denom
            if num_points_per_cap[i] < 2:
                l = torch.tensor(0.0, device=device)
            else:
                l = -torch.log(l / (num_points_per_cap[i] - 1))
            l = torch.mean(l)
            
            ll.append(l)

        loss = torch.stack(ll).mean()


        losses = {}
        losses['loss_contrast_txt'] = loss
        
        return losses

    def loss_text_contrastive(self, outputs, targets, indices, num_points, num_points_per_cap, mask_bi=None, temperature=0.07, anchor=None, **kwargs):
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']

        B = txt_embs.shape[0]
        if B == 1:
            losses = {}
            losses['loss_text_contrastive'] = torch.tensor(0., device=txt_embs.device)
            return losses
        

        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1)
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True) # (b, 256)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)

        anchor_embs = anchor['txt_embs']
        anchor_masks = anchor['token_masks']
        anchor_sum = torch.sum(anchor_embs * anchor_masks.unsqueeze(-1), dim=1)
        anchor_embs = anchor_sum / anchor_masks.sum(dim=1, keepdim=True) # (b, 256)
        anchor_embs_norm = F.normalize(anchor_embs, p=2, dim=1)


        sim = torch.matmul(txt_embs_norm, txt_embs_norm.T)
        B = txt_embs_norm.shape[0]
        mask = torch.eye(B, device=sim.device)
        
        denom = ((1 - mask) * sim).exp().div(temperature).sum(dim=-1)
        num = torch.matmul(txt_embs_norm, anchor_embs_norm.T).squeeze(-1).exp().div(temperature)

        l = - torch.log(num / denom + 1e-6)
        
        loss = l.mean()
        
        losses = {}
        losses['loss_text_contrastive'] = loss
        
        return losses

    def loss_text_sim(self, outputs, targets, indices, num_points, num_points_per_cap, mask_bi=None, temperature=0.07, **kwargs):
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']
        

        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1)
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True) # (b, 256)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)

        sim = torch.matmul(txt_embs_norm, txt_embs_norm.T)
        B = txt_embs_norm.shape[0]
        mask = torch.eye(B, device=sim.device)
        
        l = ((1 - mask) * sim).mean(dim=-1)
        
        loss = l.mean()
        
        losses = {}
        losses['loss_text_sim'] = loss
        
        return losses



    
    #   Adding supervised contrastive loss here -- KOSTAS - PANOS
    def loss_contrast_img(self, outputs, targets, indices, num_points, num_points_per_cap, mask_bi=None, temperature=0.07, **kwargs):
        assert 'img_embs' in outputs and 'txt_embs' in outputs


        idx = self._get_src_permutation_idx(indices)
        img_embs = outputs['img_embs'] 
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']

        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1) 
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True) # (b, 256)

        img_embs_norm = F.normalize(img_embs, p=2, dim=2)  # Shape: (num_images, embedding_dim)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)  # Shape: (1, embedding_dim)

        token_text_sim = torch.matmul(img_embs_norm, txt_embs_norm.unsqueeze(2)).squeeze(-1)



        """top_n_values, top_n_indices = torch.topk(token_text_sim, num_points_per_cap, largest=True, dim=1)
        pos = torch.gather(img_embs, 1, top_n_indices.unsqueeze(-1).expand(-1, -1, img_embs.shape[-1]))"""
        
        idx = []
        t_n = []
        for i in range(token_text_sim.shape[0]):
            _, top_n_indices = torch.topk(token_text_sim[i], min(900, num_points_per_cap[i].item()), dim=-1)
            t_ = torch.zeros_like(img_embs[0,:,0], dtype=torch.int)
            t_.scatter_(0, top_n_indices, 1)
            t_n.append(top_n_indices)
            idx.append(t_)

        t_ = torch.stack(idx, 0)

        dot_product_tempered = torch.matmul(img_embs_norm, img_embs_norm.permute(0, 2, 1)) / temperature

        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[1])).to(device)
        exp_msk_me = exp_dot_tempered * mask_anchor_out
        d_ = torch.sum(exp_msk_me, dim=1)
        ll = []
        for i in range(token_text_sim.shape[0]):
            denom = torch.gather(d_[i], 0, t_n[i])
            numm = torch.sum(exp_msk_me[i] * t_[i].unsqueeze(0), dim=1)
            numm = torch.gather(numm, 0, t_n[i])

            l = numm / denom
            if num_points_per_cap[i] < 2:
                l = torch.tensor(0.0, device=device)
            else:
                l = -torch.log(l / (num_points_per_cap[i] - 1))
            l = torch.mean(l)
            
            ll.append(l)

        loss = torch.stack(ll).mean()


        losses = {}
        losses['loss_contrast_img'] = loss
        
        return losses
    
    def loss_contrast_img_hm(self, outputs, targets, indices, num_points, num_points_per_cap, mask_bi=None, temperature=0.07, **kwargs):
        assert 'img_embs' in outputs and 'txt_embs' in outputs


        idx = self._get_src_permutation_idx(indices)
        img_embs = outputs['img_embs'] 
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']


        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1) 
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True) # (b, 256)

        img_embs_norm = F.normalize(img_embs, p=2, dim=2)  # Shape: (num_images, embedding_dim)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)  # Shape: (1, embedding_dim)

        token_text_sim = torch.matmul(img_embs_norm, txt_embs_norm.unsqueeze(2)).squeeze(-1)



        """top_n_values, top_n_indices = torch.topk(token_text_sim, num_points_per_cap, largest=True, dim=1)
        pos = torch.gather(img_embs, 1, top_n_indices.unsqueeze(-1).expand(-1, -1, img_embs.shape[-1]))"""
        
        slices = torch.cat([torch.tensor([0], device=num_points_per_cap.device), num_points_per_cap])
        t_n = []
        t__ = []

        for i in range(token_text_sim.shape[0]):
            
            slice = idx[1][slices[i]:(slices[i]+slices[i+1])]

            top_n_indices = slice.to(device)
            t_ = torch.zeros_like(img_embs[0,:,0], dtype=torch.int)
            t_.scatter_(0, top_n_indices, 1)
            t_n.append(top_n_indices)
            t__.append(t_)
        

        

        t_ = torch.stack(t__, 0)


        dot_product_tempered = torch.matmul(img_embs_norm, img_embs_norm.permute(0, 2, 1)) / temperature

        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[1])).to(device)
        exp_msk_me = exp_dot_tempered * mask_anchor_out
        d_ = torch.sum(exp_msk_me, dim=1)
        ll = []
        for i in range(token_text_sim.shape[0]):
            denom = torch.gather(d_[i], 0, t_n[i])
            numm = torch.sum(exp_msk_me[i] * t_[i].unsqueeze(0), dim=1)
            numm = torch.gather(numm, 0, t_n[i])

            l = numm / (denom + 1e-8)
 
            if num_points_per_cap[i] < 2:
                l = torch.tensor(0.0, device=device)
            else:
                l = -torch.log(l / (num_points_per_cap[i] - 1) + 1e-8)
            l = torch.mean(l)
            
            ll.append(l)

        loss = torch.stack(ll).mean()


        losses = {}
        losses['loss_contrast_img_hm'] = loss
        
        return losses

    #   Adding supervised contrastive loss here -- KOSTAS - PANOS
    def loss_contrast_img_sched(self, outputs, targets, indices, num_points, num_points_per_cap, epoch, epoch_stop, mask_bi=None, temperature=0.07, **kwargs):
        assert 'img_embs' in outputs and 'txt_embs' in outputs
        
        scale = max(1 - (epoch / epoch_stop), 0)

        num_points_per_cap = (num_points_per_cap - (scale * 2 * torch.log2(num_points_per_cap)).ceil().to(int)).clamp(min=1, max=None)


        idx = self._get_src_permutation_idx(indices)
        img_embs = outputs['img_embs'] 
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']

        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1) 
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True) # (b, 256)

        img_embs_norm = F.normalize(img_embs, p=2, dim=2)  # Shape: (num_images, embedding_dim)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)  # Shape: (1, embedding_dim)

        token_text_sim = torch.matmul(img_embs_norm, txt_embs_norm.unsqueeze(2)).squeeze(-1)



        """top_n_values, top_n_indices = torch.topk(token_text_sim, num_points_per_cap, largest=True, dim=1)
        pos = torch.gather(img_embs, 1, top_n_indices.unsqueeze(-1).expand(-1, -1, img_embs.shape[-1]))"""
        
        idx = []
        t_n = []
        for i in range(token_text_sim.shape[0]):
            _, top_n_indices = torch.topk(token_text_sim[i], num_points_per_cap[i].item(), dim=-1)
            t_ = torch.zeros_like(img_embs[0,:,0], dtype=torch.int)
            t_.scatter_(0, top_n_indices, 1)
            t_n.append(top_n_indices)
            idx.append(t_)

        t_ = torch.stack(idx, 0)

        dot_product_tempered = torch.matmul(img_embs_norm, img_embs_norm.permute(0, 2, 1)) / temperature

        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[1])).to(device)
        exp_msk_me = exp_dot_tempered * mask_anchor_out
        d_ = torch.sum(exp_msk_me, dim=1)
        ll = []
        for i in range(token_text_sim.shape[0]):
            denom = torch.gather(d_[i], 0, t_n[i])
            numm = torch.sum(exp_msk_me[i] * t_[i].unsqueeze(0), dim=1)
            numm = torch.gather(numm, 0, t_n[i])

            l = numm / denom
            if num_points_per_cap[i] < 2:
                l = torch.tensor(0.0, device=device)
            else:
                l = -torch.log(l / (num_points_per_cap[i] - 1))
            l = torch.mean(l)
            
            ll.append(l)

        loss = torch.stack(ll).mean()


        losses = {}
        losses['loss_contrast_img_sched'] = loss
        
        return losses

    def loss_contrast_img_sched_buffer(self, outputs, targets, indices, num_points, num_points_per_cap, epoch, epoch_stop, mask_bi=None, temperature=0.07, **kwargs):
        assert 'img_embs' in outputs and 'txt_embs' in outputs
        
        scale = max(1 - (epoch / epoch_stop), 0)
        buffer_size = (scale * 2 * torch.log2(num_points_per_cap)).ceil().to(int).clamp(max=num_points_per_cap-2).clamp(min=0)
        # num_pos = (num_points_per_cap - (scale * 2 * torch.log2(num_points_per_cap)).ceil().to(int)).clamp(min=1, max=None)


        idx = self._get_src_permutation_idx(indices)
        img_embs = outputs['img_embs'] 
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']

        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1) 
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True) # (b, 256)

        img_embs_norm = F.normalize(img_embs, p=2, dim=2)  # Shape: (num_images, embedding_dim)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)  # Shape: (1, embedding_dim)

        token_text_sim = torch.matmul(img_embs_norm, txt_embs_norm.unsqueeze(2)).squeeze(-1)



        """top_n_values, top_n_indices = torch.topk(token_text_sim, num_points_per_cap, largest=True, dim=1)
        pos = torch.gather(img_embs, 1, top_n_indices.unsqueeze(-1).expand(-1, -1, img_embs.shape[-1]))"""
        
        idx = []
        t_n = []

        ignored_n = []
        ignored_ = []


        for i in range(token_text_sim.shape[0]):
            _, top_n_indices = torch.topk(token_text_sim[i], num_points_per_cap[i].item() + buffer_size[i].item(), dim=-1, sorted=True)
            t_ = torch.zeros_like(img_embs[0,:,0], dtype=torch.int)
            ign_ = torch.zeros_like(img_embs[0,:,0], dtype=torch.int)
            
            buffer = top_n_indices[num_points_per_cap[i] - buffer_size[i]:]
            top_n_indices = top_n_indices[:num_points_per_cap[i] - buffer_size[i]]


            t_.scatter_(0, top_n_indices, 1)
            t_n.append(top_n_indices)
            idx.append(t_)

            ign_.scatter_(0, buffer, 1)
            ignored_n.append(buffer)
            ignored_.append(ign_)

        t_ = torch.stack(idx, 0)
        ign_ = torch.stack(ignored_, 0)

        dot_product_tempered = torch.matmul(img_embs_norm, img_embs_norm.permute(0, 2, 1)) / temperature

        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[1])).to(device)
        exp_msk_me = exp_dot_tempered * mask_anchor_out
        ####### BUFFER MASK ############
        exp_msk_me = exp_msk_me * (1 - ign_.unsqueeze(1))
        ################################
        d_ = torch.sum(exp_msk_me, dim=1)
        ll = []
        for i in range(token_text_sim.shape[0]):
            denom = torch.gather(d_[i], 0, t_n[i])

            numm = torch.sum(exp_msk_me[i] * t_[i].unsqueeze(0), dim=1)
            numm = torch.gather(numm, 0, t_n[i])

            l = numm / denom
            if num_points_per_cap[i] < 2:
                l = torch.tensor(0.0, device=device)
            else:
                l = -torch.log(l / (num_points_per_cap[i] - 1))
            l = torch.mean(l)
            
            ll.append(l)

        loss = torch.stack(ll).mean()


        losses = {}
        losses['loss_contrast_img_sched_buffer'] = loss
        return losses

 #   Adding supervised contrastive loss here -- KOSTAS - PANOS
    def loss_contrast_img_unmodified(self, outputs, targets, indices, num_points, num_points_per_cap, mask_bi=None, temperature=0.07, **kwargs):
        assert 'img_embs' in outputs and 'txt_embs' in outputs


        idx = self._get_src_permutation_idx(indices)
        img_embs = outputs['img_embs'] 
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']

        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1) 
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True) # (b, 256)

        img_embs_norm = F.normalize(img_embs, p=2, dim=2)  # Shape: (num_images, embedding_dim)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)  # Shape: (1, embedding_dim)

        token_text_sim = torch.matmul(img_embs_norm, txt_embs_norm.unsqueeze(2)).squeeze(-1)



        """top_n_values, top_n_indices = torch.topk(token_text_sim, num_points_per_cap, largest=True, dim=1)
        pos = torch.gather(img_embs, 1, top_n_indices.unsqueeze(-1).expand(-1, -1, img_embs.shape[-1]))"""
        
        idx = []
        t_n = []
        for i in range(token_text_sim.shape[0]):
            _, top_n_indices = torch.topk(token_text_sim[i], num_points_per_cap[i].item(), dim=-1)
            t_ = torch.zeros_like(img_embs[0,:,0], dtype=torch.int)
            t_.scatter_(0, top_n_indices, 1)
            t_n.append(top_n_indices)
            idx.append(t_)

        t_ = torch.stack(idx, 0)

        dot_product_tempered = torch.matmul(img_embs_norm, img_embs_norm.permute(0, 2, 1)) / temperature

        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[1])).to(device)
        exp_msk_me = exp_dot_tempered * mask_anchor_out
        d_ = torch.sum(exp_msk_me, dim=1)
        ll = []

        
        for i in range(token_text_sim.shape[0]):
            denom = torch.gather(d_[i], 0, t_n[i])
            numm = torch.sum(exp_msk_me[i] * t_[i].unsqueeze(0), dim=1)
            numm = torch.gather(numm, 0, t_n[i])

            l = numm / denom
            if num_points_per_cap[i] < 2:
                ll.append(torch.tensor(0.0, device=device))
                continue
            else:
                l = -torch.log(l / (num_points_per_cap[i] - 1))
            
            neg_idx = []
            for j in range(d_[i].shape[-1]):
                if j not in t_n[i]:
                    neg_idx.append(j)
            neg_idx = torch.tensor(neg_idx, device = l.device)
            
            denom2 = torch.gather(d_[i], 0, neg_idx)
            n_ = torch.zeros_like(t_[i], dtype=torch.int)
            n_.scatter_(0, neg_idx, 1)
            numm2 = torch.sum(exp_msk_me[i] * n_.unsqueeze(0), dim=1)
            numm2 = torch.gather(numm2, 0, neg_idx)
            l2 = numm2 / denom2
            negs = l2.shape[0]
            if negs < 2:
                ll.append(torch.tensor(0.0, device=device))
                continue
            else:
                l2 = -torch.log(l2 / (negs - 1))
            
            l = (l.sum() + l2.sum()) / d_.shape[-1]

            ll.append(l)

        loss = torch.stack(ll).mean()


        losses = {}
        losses['loss_contrast_img_unmodified'] = loss
        
        return losses
    
    def loss_contrast_img_static(self, outputs, targets, indices, num_points, num_points_per_cap, mask_bi=None, temperature=0.07, **kwargs):
        assert 'img_embs' in outputs and 'txt_embs' in outputs


        idx = self._get_src_permutation_idx(indices)
        img_embs = outputs['img_embs'] 
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']

        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1) 
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True) # (b, 256)

        img_embs_norm = F.normalize(img_embs, p=2, dim=2)  # Shape: (num_images, embedding_dim)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)  # Shape: (1, embedding_dim)

        token_text_sim = torch.matmul(img_embs_norm, txt_embs_norm.unsqueeze(2)).squeeze(-1)
        


        """top_n_values, top_n_indices = torch.topk(token_text_sim, num_points_per_cap, largest=True, dim=1)
        pos = torch.gather(img_embs, 1, top_n_indices.unsqueeze(-1).expand(-1, -1, img_embs.shape[-1]))"""
        
        idx = []
        t_n = []
        for i in range(token_text_sim.shape[0]):
            _, top_n_indices = torch.topk(token_text_sim[i], min(5, num_points_per_cap[i].item()), dim=-1) # STATIC CHANGE HERE
            t_ = torch.zeros_like(img_embs[0,:,0], dtype=torch.int)
            t_.scatter_(0, top_n_indices, 1)
            t_n.append(top_n_indices)
            idx.append(t_)

        t_ = torch.stack(idx, 0)

        dot_product_tempered = torch.matmul(img_embs_norm, img_embs_norm.permute(0, 2, 1)) / temperature

        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[1])).to(device)
        exp_msk_me = exp_dot_tempered * mask_anchor_out
        d_ = torch.sum(exp_msk_me, dim=1)
        ll = []
        for i in range(token_text_sim.shape[0]):
            denom = torch.gather(d_[i], 0, t_n[i])
            numm = torch.sum(exp_msk_me[i] * t_[i].unsqueeze(0), dim=1)
            numm = torch.gather(numm, 0, t_n[i])

            l = numm / denom
            if num_points_per_cap[i] < 2:
                l = torch.tensor(0.0, device=device)
            else:
                l = -torch.log(l / (min(5, num_points_per_cap[i].item()) - 1))
            l = torch.mean(l)
            
            ll.append(l)

        loss = torch.stack(ll).mean()


        losses = {}
        losses['loss_contrast_img_static'] = loss
        
        return losses
    
    def loss_contrast_img_sqrt(self, outputs, targets, indices, num_points, num_points_per_cap, mask_bi=None, temperature=0.07, **kwargs):
        assert 'img_embs' in outputs and 'txt_embs' in outputs


        idx = self._get_src_permutation_idx(indices)
        img_embs = outputs['img_embs'] 
        txt_embs = outputs['txt_embs'] 
        token_masks = outputs['token_masks']

        txt_emb_sum = torch.sum(txt_embs * token_masks.unsqueeze(-1), dim=1) 
        txt_embs = txt_emb_sum / token_masks.sum(dim=1, keepdim=True) # (b, 256)

        img_embs_norm = F.normalize(img_embs, p=2, dim=2)  # Shape: (num_images, embedding_dim)
        txt_embs_norm = F.normalize(txt_embs, p=2, dim=1)  # Shape: (1, embedding_dim)

        token_text_sim = torch.matmul(img_embs_norm, txt_embs_norm.unsqueeze(2)).squeeze(-1)



        """top_n_values, top_n_indices = torch.topk(token_text_sim, num_points_per_cap, largest=True, dim=1)
        pos = torch.gather(img_embs, 1, top_n_indices.unsqueeze(-1).expand(-1, -1, img_embs.shape[-1]))"""
        
        idx = []
        t_n = []
        for i in range(token_text_sim.shape[0]):
            N = num_points_per_cap[i].item()
            topK = max(N - int((N)**(1/2)), 2)

            _, top_n_indices = torch.topk(token_text_sim[i], min(800, topK), dim=-1) # STATIC CHANGE HERE
            t_ = torch.zeros_like(img_embs[0,:,0], dtype=torch.int)
            t_.scatter_(0, top_n_indices, 1)
            t_n.append(top_n_indices)
            idx.append(t_)

        t_ = torch.stack(idx, 0)

        dot_product_tempered = torch.matmul(img_embs_norm, img_embs_norm.permute(0, 2, 1)) / temperature

        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[1])).to(device)
        exp_msk_me = exp_dot_tempered * mask_anchor_out
        d_ = torch.sum(exp_msk_me, dim=1)
        ll = []
        for i in range(token_text_sim.shape[0]):
            denom = torch.gather(d_[i], 0, t_n[i])
            numm = torch.sum(exp_msk_me[i] * t_[i].unsqueeze(0), dim=1)
            numm = torch.gather(numm, 0, t_n[i])
            N = num_points_per_cap[i].item()
            topK = N + int((N)**(1/2))
            l = numm / denom
            if num_points_per_cap[i] < 2:
                l = torch.tensor(0.0, device=device)
            else:
                l = -torch.log(l / (min(800, topK) - 1))
            l = torch.mean(l)
            
            ll.append(l)

        loss = torch.stack(ll).mean()


        losses = {}
        losses['loss_contrast_img_sqrt'] = loss
        
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        
        assert loss in self.loss_map, f'do you really want to compute {loss} loss?'
        return self.loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, mask_bi=None, anchor=None, epoch=None, epoch_stop=None):

        indices = self.matcher(outputs, targets) # list of tuple [(pred_index, target_index),(pred_index, target_index)]
        
        caption_sizes = [t['caption_size'] for t in targets] 
        bs, _ = outputs['pred_logits'].shape[:2] 

        
        num_points = sum(len(t["labels"]) for t in targets)
        num_points_per_cap = torch.tensor([t['points'].shape[0] for t in targets])
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_points = torch.clamp(num_points / 1, min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, caption_sizes=caption_sizes,
                                        mask_bi=mask_bi, num_points_per_cap=num_points_per_cap, epoch=epoch, epoch_stop=epoch_stop, anchor=anchor))

        # re-arrange indices of outputs based on targets
        for b in range(bs):
            output_indices = indices[b][0] 
            target_indices = indices[b][1] 
            for ti, oi in zip(target_indices, output_indices):
                if ti > len(outputs['pred_points'][b]) - 1: 
                    continue
                temp = outputs['pred_points'][b][ti].clone() 
                outputs['pred_points'][b][ti] = outputs['pred_points'][b][oi]
                outputs['pred_points'][b][oi] = temp
                temp = outputs['pred_logits'][0][ti].clone()
                outputs['pred_logits'][b][ti] = outputs['pred_logits'][b][oi]
                outputs['pred_logits'][b][oi] = temp

        return losses

