# coding: utf-8

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vgg16 import vgg
from rpn import RPN
from roihead import RoiHead
import utils

class fastnet(nn.Module):
    def __init__(self):
        super(fastnet, self).__init__()
        
        # prepare
        self.extractor, classifier = vgg()
        self.rpn = RPN()
        self.head = RoiHead(classifier)

        self.anchor_target = utils.anchor_target()
        self.proposal_target = utils.proposal_target()

        self.rpn_sigma = 3.
        self.roi_sigma = 1.

    def forward(self, x, boxes, labels, scale):
        # locs of rpn and roi is (tx,ty,tw,th) from the paper
        img_size = x.shape[2:]
        scale = scale.item()
        # batch size is 1.
        labels = labels[0]
        boxes = boxes[0]

        fm = self.extractor(x)
        rpn_locs, rpn_scores, rois, anchors = self.rpn(fm, img_size, scale)
        # should be done at data transform?
        gt_rpn_loc, gt_rpn_label = self.anchor_target(boxes, anchors, img_size)

        # rpn loss
        gt_rpn_loc   = torch.from_numpy(gt_rpn_loc).cuda()
        gt_rpn_label = torch.from_numpy(gt_rpn_label).cuda()
        rpn_loc_loss = self.loc_loss(rpn_locs[0], gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_scores[0], gt_rpn_label, ignore_index=-1)

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target(rois, boxes, labels)
        roi_locs, roi_scores = self.head(fm, torch.from_numpy(sample_roi))

        # roi loss
        n_roi_locs = len(roi_locs)
        roi_locs = roi_locs.view(n_roi_locs, -1, 4)
        roi_locs = roi_locs[torch.arange(n_roi_locs), gt_roi_label]

        gt_roi_loc = torch.from_numpy(gt_roi_loc).cuda()
        gt_roi_label = torch.from_numpy(gt_roi_label).cuda()
        roi_loc_loss = self.loc_loss(roi_locs, gt_roi_loc, gt_roi_label, self.roi_sigma)
        roi_cls_loss = F.cross_entropy(roi_scores, gt_roi_label)

        losses = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

        return losses

    def loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        in_weight = torch.zeros(gt_loc.shape).cuda()
        in_weight[(gt_label>0).view(-1,1).expand_as(gt_loc)] = 1
        sigma2 = sigma**2
        diff = in_weight * (pred_loc - gt_loc)
        abs_diff = diff.abs()
        flag = (abs_diff < 1./sigma2).float()
        loc_loss = 0.5 * flag * sigma2 * (diff**2) + (1-flag) * (abs_diff - 0.5/sigma2)
        loc_loss = loc_loss.sum()
        loc_loss /= (gt_label>=0).sum()
        return loc_loss

    def predict_net(self, x, img_size, scale):
        fm = self.extractor(x)
        rpn_locs, rpn_scores, rois, anchors = self.rpn(fm, img_size, scale)
        roi_locs, roi_scores = self.head(fm, torch.from_numpy(rois))
        return roi_locs, roi_scores, rois

    def predict(self, img, scale=1., using_gpu=False):
        img_size = img.shape[2:]
        roi_locs, roi_scores, rois = self.predict_net(img, img_size, scale)
        if using_gpu:
            roi_locs, roi_scores = roi_locs.cpu(), roi_scores.cpu()

        n_class = 21
        score_thresh = 0.7
        nms_thresh = 0.3
        mean = torch.Tensor([0., 0., 0., 0.]).repeat(n_class).unsqueeze(0)
        std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(n_class).unsqueeze(0)
        roi_locs = roi_locs * std + mean
        roi_locs = roi_locs.view(-1, n_class, 4)
        rois = torch.from_numpy(rois)
        rois = rois.view(-1, 1, 4).expand_as(roi_locs)

        roi_box = utils.loc2bbox(rois.numpy().reshape(-1, 4), roi_locs.detach().numpy().reshape(-1, 4))
        roi_box = torch.from_numpy(roi_box).view(-1, n_class*4)
        roi_box[:, 0::2] = roi_box[:, 0::2].clamp(0, img_size[1])
        roi_box[:, 1::2] = roi_box[:, 1::2].clamp(0, img_size[0])

        prob = F.softmax(roi_scores, dim=1).detach().numpy()

        bbox = []
        label = []
        score = []
        for i in range(1, n_class):
            roi_box_i = roi_box.view(-1, n_class, 4)[:, i, :]
            roi_box_i = roi_box_i.detach().numpy()
            prob_i = prob[:, i]
            mask = prob_i > score_thresh
            roi_box_i = roi_box_i[mask]
            prob_i = prob_i[mask]

            order = prob_i.argsort()[::-1]
            roi_box_i = roi_box_i[order]
            keep = utils.nms(roi_box_i, nms_thresh)

            bbox.append(roi_box_i[keep])
            label.append((i-1)*torch.ones((len(keep),)))
            score.append(prob_i[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        bbox = bbox * 16

        return bbox, label, score
