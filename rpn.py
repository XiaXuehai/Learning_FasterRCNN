# coding:utf-8

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import utils

class RPN(nn.Module):
    def __init__(self, cin=512, cout=512, stride=16):
        super(RPN, self).__init__()

        self.stride = stride

        self.anchor_base = utils.generate_anchor(strides=self.stride)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.locs  = nn.Conv2d(cout, n_anchor*4, 1)
        self.scores = nn.Conv2d(cout, n_anchor*2, 1)
        # TODO: init parameters


    def forward(self, x, img_size):
        n, c, h, w = x.shape
        anchors = utils.get_anchor(self.anchor_base, self.stride, h, w)

        x = F.relu(self.conv1(x))
        rpn_locs   = self.locs(x)
        rpn_scores = self.scores(x)

        rpn_locs   = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        # got foreground scores
        scores = rpn_scores[:, :, 1].detach().cpu().numpy()

        #  transform (tx, ty, tw, th) to (x, y, w, h)
        rois = utils.transform_locs(anchors, rpn_locs.detach().cpu().numpy(), img_size)
        # compromise to 1 batch_size!!!
        rois = rois[0]
        scores = scores[0]

        # TODO: min_size

        pre_nms = 12000
        post_nms = 1000
        nms_thresh = 0.7

        order = scores.argsort()[::-1]
        order = order[:pre_nms]
        # TODO: how to optimize?
        roi_tmp = rois[order,:]
        keep = utils.nms(roi_tmp, nms_thresh, post_nms)
        roi = roi_tmp[keep]

        return rpn_locs, rpn_scores, roi, anchors


if __name__ == '__main__':
    a = RPN()
    x = torch.rand(8, 512, 40, 60)
    a(x, (600, 1000))