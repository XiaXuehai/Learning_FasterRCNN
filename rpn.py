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
        # init parameters
        utils.init_normal(self.conv1, 0., 0.01)
        utils.init_normal(self.locs, 0., 0.01)
        utils.init_normal(self.scores, 0., 0.01)

    def forward(self, x, img_size, scale):
        n, c, h, w = x.shape
        anchors = utils.get_anchor(self.anchor_base, self.stride, h, w)

        x = F.relu(self.conv1(x))
        rpn_locs   = self.locs(x)
        rpn_scores = self.scores(x)

        # rpn_locs is the (tx, ty, tw, th) which means (x_centre, y_centre, w,h)
        rpn_locs   = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        # got foreground scores
        scores = rpn_scores[:, :, 1].detach().cpu().numpy()

        # compromise to 1 batch_size!!!
        rois = rpn_locs[0]
        scores = scores[0]
        # transform (tx, ty, tw, th) to (x, y, w, h) to (x1, y1, x2, y2)
        # and clip by img_size
        rois = utils.transform_locs(anchors, rois.detach().cpu().numpy())
        h, w = img_size
        rois[:, 0] = np.clip(rois[:, 0], 0, w)
        rois[:, 1] = np.clip(rois[:, 1], 0, h)
        rois[:, 2] = np.clip(rois[:, 2], 0, w)
        rois[:, 3] = np.clip(rois[:, 3], 0, h)

        # erase min_size
        min_size = 16 * scale
        ws = rois[:, 2] - rois[:, 0]
        hs = rois[:, 3] - rois[:, 1]
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        rois = rois[keep, :]
        scores = scores[keep]

        pre_nms = 12000
        post_nms = 2000
        nms_thresh = 0.7

        order = scores.argsort()[::-1]
        order = order[:pre_nms]

        roi_tmp = rois[order,:]
        keep = utils.nms(roi_tmp, nms_thresh)
        keep = keep[:post_nms]
        roi = roi_tmp[keep]

        print(len(keep))
        return rpn_locs, rpn_scores, roi, anchors


if __name__ == '__main__':
    a = RPN()
    x = torch.rand(1, 512, 40, 60)
    a(x, (600, 1000), 1.0)