# Functions for 1D NMS, modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py
import torch

import nms_1d_cpu


def nms_1d(segs, scores, iou_threshold):
    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())  # 使用 item() 确保存储的是数值而不是 tensor

        if order.numel() == 1:
            break

        xx1 = torch.maximum(segs[i, 0], segs[order[1:], 0])
        xx2 = torch.minimum(segs[i, 1], segs[order[1:], 1])
        inter = torch.clamp(xx2 - xx1, min=0)
        iou = inter / (segs[i, 1] - segs[i, 0] + segs[order[1:], 1] - segs[order[1:], 0] - inter + 1e-6)
        ids = (iou <= iou_threshold).nonzero().squeeze()
        order = order[ids + 1]  # 这里已经确保正确索引，之前的逻辑是正确的

    return torch.tensor(keep, dtype=torch.long)


def softnms_1d(segs, scores, dets, iou_threshold, sigma, min_score, method):
    x1 = segs[:, 0].clone()
    x2 = segs[:, 1].clone()
    scores = scores.clone()

    n_segs = segs.size(0)
    areas = (x2 - x1)
    order = torch.argsort(scores, descending=True)
    inds = torch.arange(n_segs, dtype=torch.long)

    keep = []
    while len(order) > 0:
        i = order[0]
        ix1 = x1[i]
        ix2 = x2[i]
        iscore = scores[i]
        iarea = areas[i]

        keep.append(i)  # Add the index of the segment with the highest score
        dets[len(keep) - 1, 0] = ix1  # Store in dets
        dets[len(keep) - 1, 1] = ix2
        dets[len(keep) - 1, 2] = iscore

        if len(order) == 1:
            break

        xx1 = torch.maximum(ix1, x1[order[1:]])
        xx2 = torch.minimum(ix2, x2[order[1:]])
        inter = torch.clamp(xx2 - xx1, min=0)
        overlap = inter / (iarea + areas[order[1:]] - inter)

        if method == 0:
            weights = (overlap < iou_threshold).float()
        elif method == 1:
            weights = torch.where(overlap >= iou_threshold, 1 - overlap, torch.ones_like(overlap))
        elif method == 2:
            weights = torch.exp(-(overlap ** 2) / sigma)

        scores[order[1:]] *= weights
        next_order = order[1:][scores[order[1:]] >= min_score]
        order = next_order

    return torch.tensor(keep, dtype=torch.long)  # Return indices of the kept segments




class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, min_score, max_num
    ):
        # vanilla nms will not change the score, so we can filter segs first
        is_filtering_by_score = (min_score > 0)
        if is_filtering_by_score:
            valid_mask = scores > min_score
            segs, scores = segs[valid_mask], scores[valid_mask]
            cls_idxs = cls_idxs[valid_mask]
            valid_inds = torch.nonzero(
                valid_mask, as_tuple=False).squeeze(dim=1)

        # nms op; return inds that is sorted by descending order
        inds = nms_1d_cpu.nms(
            segs.contiguous().cpu(),
            scores.contiguous().cpu(),
            iou_threshold=float(iou_threshold))
        # cap by max number
        if max_num > 0:
            inds = inds[:min(max_num, len(inds))]
        # return the sorted segs / scores
        sorted_segs = segs[inds]
        sorted_scores = scores[inds]
        sorted_cls_idxs = cls_idxs[inds]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


class SoftNMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, sigma, min_score, method, max_num
    ):
        # pre allocate memory for sorted results
        dets = segs.new_empty((segs.size(0), 3), device='cpu')
        # softnms op, return dets that stores the sorted segs / scores
        inds = nms_1d_cpu.softnms(
            segs.cpu(),
            scores.cpu(),
            dets.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=int(method))
        
        # inds = softnms_1d(
        #     segs,
        #     scores,
        #     dets,
        #     iou_threshold=float(iou_threshold),
        #     sigma=float(sigma),
        #     min_score=float(min_score),
        #     method=int(method)
        # ).long()

        

        # cap by max number
        if max_num > 0:
            n_segs = min(len(inds), max_num)
        else:
            n_segs = len(inds)
        sorted_segs = dets[:n_segs, :2]
        sorted_scores = dets[:n_segs, 2]
        sorted_cls_idxs = cls_idxs[inds]
        sorted_cls_idxs = sorted_cls_idxs[:n_segs]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    """
        blur localization results by incorporating side segs.
        this is known as bounding box voting in object detection literature.
        slightly boost the performance around iou_threshold
    """

    # *_segs : N_i x 2, all_scores: N,
    # apply offset
    offset_scores = all_scores + score_offset

    # computer overlap between nms and all segs
    # construct the distance matrix of # N_nms x # N_all
    num_nms_segs, num_all_segs = nms_segs.shape[0], all_segs.shape[0]
    ex_nms_segs = nms_segs[:, None].expand(num_nms_segs, num_all_segs, 2)
    ex_all_segs = all_segs[None, :].expand(num_nms_segs, num_all_segs, 2)

    # compute intersection
    left = torch.maximum(ex_nms_segs[:, :, 0], ex_all_segs[:, :, 0])
    right = torch.minimum(ex_nms_segs[:, :, 1], ex_all_segs[:, :, 1])
    inter = (right-left).clamp(min=0)

    # lens of all segments
    nms_seg_lens = ex_nms_segs[:, :, 1] - ex_nms_segs[:, :, 0]
    all_seg_lens = ex_all_segs[:, :, 1] - ex_all_segs[:, :, 0]

    # iou
    iou = inter / (nms_seg_lens + all_seg_lens - inter)

    # get neighbors (# N_nms x # N_all) / weights
    seg_weights = (iou >= iou_threshold).to(all_scores.dtype) * all_scores[None, :] * iou
    seg_weights /= torch.sum(seg_weights, dim=1, keepdim=True)
    refined_segs = seg_weights @ all_segs

    return refined_segs

def batched_nms(
    segs,
    scores,
    cls_idxs,
    iou_threshold,
    min_score,
    max_seg_num,
    use_soft_nms=True,
    multiclass=True,
    sigma=0.5,
    voting_thresh=0.75,
):
    # Based on Detectron2 implementation,
    num_segs = segs.shape[0]
    # corner case, no prediction outputs
    if num_segs == 0:
        return torch.zeros([0, 2]),\
               torch.zeros([0,]),\
               torch.zeros([0,], dtype=cls_idxs.dtype)

    if multiclass:
        # multiclass nms: apply nms on each class independently
        new_segs, new_scores, new_cls_idxs = [], [], []
        for class_id in torch.unique(cls_idxs):
            curr_indices = torch.where(cls_idxs == class_id)[0]
            # soft_nms vs nms
            if use_soft_nms:
                sorted_segs, sorted_scores, sorted_cls_idxs = SoftNMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    sigma,
                    min_score,
                    2,
                    max_seg_num
                )
            else:
                sorted_segs, sorted_scores, sorted_cls_idxs = NMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    min_score,
                    max_seg_num
                )
            # disable seg voting for multiclass nms, no sufficient segs

            # fill in the class index
            new_segs.append(sorted_segs)
            new_scores.append(sorted_scores)
            new_cls_idxs.append(sorted_cls_idxs)

        # cat the results
        new_segs = torch.cat(new_segs)
        new_scores = torch.cat(new_scores)
        new_cls_idxs = torch.cat(new_cls_idxs)

    else:
        # class agnostic
        if use_soft_nms:
            new_segs, new_scores, new_cls_idxs = SoftNMSop.apply(
                segs, scores, cls_idxs, iou_threshold,
                sigma, min_score, 2, max_seg_num
            )
        else:
            new_segs, new_scores, new_cls_idxs = NMSop.apply(
                segs, scores, cls_idxs, iou_threshold,
                min_score, max_seg_num
            )
        # seg voting
        if voting_thresh > 0:
            new_segs = seg_voting(
                new_segs,
                segs,
                scores,
                voting_thresh
            )

    # sort based on scores and return
    # truncate the results based on max_seg_num
    _, idxs = new_scores.sort(descending=True)
    max_seg_num = min(max_seg_num, new_segs.shape[0])
    # needed for multiclass NMS
    new_segs = new_segs[idxs[:max_seg_num]]
    new_scores = new_scores[idxs[:max_seg_num]]
    new_cls_idxs = new_cls_idxs[idxs[:max_seg_num]]
    return new_segs, new_scores, new_cls_idxs
