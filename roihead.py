# coding: utf-8
import torch
from torch import nn
from torch.nn import functional as F
import utils

class RoiHead(nn.Module):
    def __init__(self, classifier, n_class=21, roi_size=7, spatial_scale=16.):
        super(RoiHead, self).__init__()
        self.n_class = n_class
        self.roi_size = (roi_size, roi_size)
        self.spatial_scale = spatial_scale

        self.classifier = classifier
        self.locs  = nn.Linear(4096, self.n_class*4)
        self.score = nn.Linear(4096, self.n_class)
        utils.init_normal(self.locs, 0., 0.001)
        utils.init_normal(self.score, 0., 0.01)

    def forward(self, x, rois):
        # reference: https://github.com/SirLPS/roi_pooling/blob/master/speed.py
        # ROI pooling by pytorch
        num_rois, _ = rois.shape
        # rescale to the size of the feature map
        rois.mul_(1/self.spatial_scale)
        rois = rois.long()

        pool = []
        for j in range(num_rois):
            roi = rois[j]
            im = x[:, :, roi[1]:(roi[3]+1), roi[0]:(roi[2]+1)]
            pool.append(F.adaptive_max_pool2d(im, self.roi_size))
        pool = torch.cat(pool, 0)
        pool = pool.view(num_rois, -1)

        # classify and regression
        fc = self.classifier(pool)
        roi_locs = self.locs(fc)
        roi_scores = self.score(fc)

        return roi_locs, roi_scores


from vgg16 import vgg
if __name__ == '__main__':
    _, classifier = vgg()
    net = RoiHead(classifier)
    fm = torch.rand(8, 512, 37, 50)
    roi = torch.rand(8, 300, 4)
    roi_locs, roi_scores = net(fm, roi)
    print(roi_locs.shape)