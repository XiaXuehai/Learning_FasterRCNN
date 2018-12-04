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

    def forward(self, x, boxes, labels):
        img_size = x.shape[2:]

        fm = self.extractor(x)
        rpn_locs, rpn_scores, rois, anchors = self.rpn(fm, img_size)
        # should be done at data transform?
        gt_rpn_loc, gt_rpn_label = self.anchor_target(boxes[0], anchors, img_size)

        # rpn loss
        gt_rpn_loc = torch.from_numpy(gt_rpn_loc).cuda()
        gt_rpn_label = torch.from_numpy(gt_rpn_label).cuda()
        rpn_loc_loss = self.loc_loss(rpn_locs[0], gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_scores[0], gt_rpn_label, ignore_index=-1)

        sample_roi, gt_roi_loc, gt_roi_score = self.proposal_target(rois, boxes[0], labels[0])
        roi_locs, roi_scores = self.head(fm, torch.from_numpy(sample_roi))

        # roi loss
        n_roi_locs = len(roi_locs)
        roi_locs = roi_locs.view(n_roi_locs, -1, 4)
        roi_locs = roi_locs[torch.arange(n_roi_locs), gt_roi_score]

        gt_roi_loc = torch.from_numpy(gt_roi_loc).cuda()
        gt_roi_score = torch.from_numpy(gt_roi_score).cuda()
        roi_loc_loss = self.loc_loss(roi_locs, gt_roi_loc, gt_roi_score, self.roi_sigma)
        roi_cls_loss = F.cross_entropy(roi_scores, gt_roi_score, ignore_index=-1)


        output = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

        return output

    def loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        in_weight = torch.zeros(gt_loc.shape).cuda()
        in_weight[(gt_label>0).view(-1,1).expand_as(gt_loc)] = 1
        sigma2 = sigma**2
        diff = in_weight * (pred_loc - gt_loc)
        abs_diff = abs(diff)
        flag = (abs_diff < 1./sigma2).float()
        loc_loss = (flag * (sigma2/2.)) * diff**2 + (1-flag) * (abs_diff + 0.5/sigma2)
        loc_loss = loc_loss.sum()
        loc_loss /= (gt_label>=0).sum()
        return loc_loss


from data import data_load
from torchvision import transforms

if __name__ == '__main__':
    transform = [transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
    data = data_load.VOCData('data/train.txt', transform=transform)
    image, label, boxes = data[30]
    x = image.unsqueeze(0)
    print(x.shape)
    
    net = fastnet()
    y = net(x)
    print(y.shape)


    