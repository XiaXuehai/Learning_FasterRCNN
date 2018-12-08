# coding:utf-8

import numpy as np

def generate_anchor(side_lenth=16, strides=16, ratios=[0.5,1,2], scales=[0.5,1,2]):
    py = side_lenth / 2
    px = side_lenth / 2

    anchor_base = np.zeros((len(ratios)*len(scales), 4), dtype=np.float32)

    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = side_lenth * strides * scales[j] * np.sqrt(ratios[i])
            w = side_lenth * strides * scales[j] * np.sqrt(1/ratios[i])

            index = i * len(scales) + j
            anchor_base[index, 0] = px - 0.5 * w
            anchor_base[index, 1] = py - 0.5 * h
            anchor_base[index, 2] = px + 0.5 * w
            anchor_base[index, 3] = py + 0.5 * h

    return anchor_base

def get_anchor(anchor_base, stride, h, w):
    grid_x = np.arange(w) * stride
    grid_y = np.arange(h) * stride
    x, y = np.meshgrid(grid_x, grid_y)
    shift = np.stack((x.ravel(), y.ravel(), x.ravel(), y.ravel()), axis=1)
    # coordinate
    co = np.repeat(shift, len(anchor_base), axis=0)
    # anchors
    an = np.tile(anchor_base, [len(shift), 1])
    anchors = co + an

    anchors = anchors.astype(np.float32)
    return anchors

def transform_locs(anchors, rpn_locs):
    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    x_a = anchors[:, 0] + 0.5 * w_a
    y_a = anchors[:, 1] + 0.5 * h_a

    tx = rpn_locs[:, 0]
    ty = rpn_locs[:, 1]
    tw = rpn_locs[:, 2]
    th = rpn_locs[:, 3]

    dx = tx * w_a + x_a
    dy = ty * h_a + y_a
    dw = np.exp(tw) * w_a
    dh = np.exp(th) * h_a

    dst = np.zeros(rpn_locs.shape, dtype=rpn_locs.dtype)
    dst[:, 0] = dx - 0.5 * dw
    dst[:, 1] = dy - 0.5 * dh
    dst[:, 2] = dx + 0.5 * dw
    dst[:, 3] = dy + 0.5 * dh

    return dst

def nms(rois, nms_thresh):
    x1 = rois[:, 0]
    y1 = rois[:, 1]
    x2 = rois[:, 2]
    y2 = rois[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    n = len(rois)
    order = np.arange(n)
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0., xx2 - xx1 + 1)
        h = np.maximum(0., yy2 - yy1 + 1)
        in_area = w * h
        iou = in_area / (area[i] + area[order[1:]] - in_area)
        # keep the iou less than thresh
        # update the order
        idx = np.where(iou <= nms_thresh)[0]
        order = order[idx+1]

    return keep

def iou(abox, bbox):
    # top-left and bottom-right
    # broadcast
    tl = np.maximum(abox[:, None, :2], bbox[:, :2])
    br = np.minimum(abox[:, None, 2:], bbox[:, 2:])
    wh = br - tl
    wh[wh<0] = 0
    inter = wh[:, :, 0] * wh[:, :, 1]
    a_area = np.prod(abox[:, 2:] - abox[:, :2], axis=1)
    b_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
    # broadcast
    return inter / (a_area[:, None] + b_area - inter)

def bbox2loc(src, dst):
    src_w = src[:, 2] - src[:, 0]
    src_h = src[:, 3] - src[:, 1]
    src_x = src[:, 0] + 0.5 * src_w
    src_y = src[:, 1] + 0.5 * src_h

    dst_w = dst[:, 2] - dst[:, 0]
    dst_h = dst[:, 3] - dst[:, 1]
    dst_x = dst[:, 0] + 0.5 * dst_w
    dst_y = dst[:, 1] + 0.5 * dst_h

    eps = np.finfo(src_h.dtype).eps
    src_h = np.maximum(src_h, eps)
    src_w = np.maximum(src_w, eps)

    tx = (dst_x - src_x) / src_w
    ty = (dst_y - src_y) / src_h
    tw = np.exp(dst_w / src_w)
    th = np.exp(dst_h / src_h)

    loc = np.vstack((tx, ty, tw, th)).transpose()
    return loc

def loc2bbox(src, loc):
    if src.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_w = src[:, 2] - src[:, 0]
    src_h = src[:, 3] - src[:, 1]
    src_x = src[:, 0] + 0.5 * src_w
    src_y = src[:, 1] + 0.5 * src_h

    tx = loc[:, 0]
    ty = loc[:, 1]
    tw = loc[:, 2]
    th = loc[:, 3]

    dx = tx * src_w + src_x
    dy = ty * src_h + src_y
    dw = np.exp(tw) * src_w
    dh = np.exp(th) * src_h

    dst = np.zeros(loc.shape, dtype=loc.dtype)
    dst[:, 0] = dx - 0.5 * dw
    dst[:, 1] = dy - 0.5 * dh
    dst[:, 2] = dx + 0.5 * dw
    dst[:, 3] = dy + 0.5 * dh
    return dst


class anchor_target(object):
    '''get the ground truth for rpn loss'''

    def __init__(self, n_sample=256, iou_pos=0.7, iou_neg = 0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.iou_pos = iou_pos
        self.iou_neg = iou_neg
        self.pos_ratio = pos_ratio

    def __call__(self, boxes, anchors, img_size):
        boxes = boxes.numpy()
        n_anchor = len(anchors)

        h, w = img_size
        index_inside = np.where(
                (anchors[:, 0] >= 0) &
                (anchors[:, 1] >= 0) &
                (anchors[:, 2] <= w) &
                (anchors[:, 3] <= h)
            )[0]
        anchors = anchors[index_inside]
        argmax_ious, label = self.create_label(anchors, boxes)
        loc = bbox2loc(anchors, boxes[argmax_ious])

        gt_rpn_scores = self._unmap(label, n_anchor, index_inside, fill=-1)
        gt_rpn_loc = self._unmap(loc, n_anchor, index_inside, fill=0)

        return gt_rpn_loc, gt_rpn_scores

    def create_label(self, anchor, boxes):
        label = np.empty((len(anchor),), dtype=np.int)
        label.fill(-1)
        ious = iou(anchor, boxes)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(anchor)), argmax_ious]

        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious==gt_max_ious)[0] # more than before

        label[max_ious < self.iou_neg] = 0
        label[max_ious >= self.iou_pos] = 1
        label[gt_argmax_ious] = 1

        n_pos = int(self.n_sample * self.pos_ratio)
        pos_index = np.where(label==1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=len(pos_index)-n_pos, replace=False)
            label[disable_index] = -1

        n_neg = self.n_sample - np.sum(label==1)
        neg_index = np.where(label==0)[0]
        if len(neg_index)>n_neg:
            disable_index = np.random.choice(neg_index, size=len(neg_index) - n_neg, replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _unmap(self, data, count, index, fill=0):

        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=data.dtype)
            ret.fill(fill)
            ret[index] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
            ret.fill(fill)
            ret[index, :] = data
        return ret

class proposal_target(object):
    def __init__(self):
        self.n_sample = 128
        self.pos_ratio = 0.25
        self.iou_pos = 0.5
        self.iou_neg_h = 0.5
        self.iou_neg_l = 0.1

    def __call__(self, rois, boxes, labels,
                 loc_mean=(0., 0., 0., 0.), loc_std=(0.1, 0.1, 0.2, 0.2)):
        n_box, _ = boxes.shape
        boxes = boxes.numpy()
        labels = labels.numpy()

        # to guarantee the ground-truth in samples-rois
        rois = np.concatenate((rois, boxes), axis=0)
        n_pos_roi = int(self.n_sample * self.pos_ratio)
        ious = iou(rois, boxes)
        max_iou = ious.max(axis=1)
        argmax_iou = ious.argmax(axis=1)
        # 0 is background
        iou_label = labels[argmax_iou] + 1

        pos_index = np.where(max_iou>=self.iou_pos)[0]
        n_pos_roi = min(n_pos_roi, len(pos_index))
        if len(pos_index) > n_pos_roi:
            pos_index = np.random.choice(pos_index, size=n_pos_roi, replace=False)

        neg_index = np.where((max_iou<self.iou_neg_h) & (max_iou>=self.iou_neg_l))[0]
        n_neg_roi = self.n_sample - n_pos_roi
        n_neg_roi = min(n_neg_roi, len(neg_index))
        if len(neg_index) > n_neg_roi:
            neg_index = np.random.choice(neg_index, size=n_neg_roi, replace=False)

        keep = np.append(pos_index, neg_index)
        gt_roi_label = iou_label[keep]
        gt_roi_label[n_pos_roi:] = 0
        sample_roi = rois[keep]

        gt_roi_loc = bbox2loc(sample_roi, boxes[argmax_iou[keep]])
        gt_roi_loc = (gt_roi_loc - np.array(loc_mean, dtype=np.float32)) / np.array(loc_std, dtype=np.float32)

        return sample_roi, gt_roi_loc, gt_roi_label

def init_normal(layer, mean, std):
    layer.weight.data.normal_(mean, std)
    layer.bias.data.zero_()

if __name__ == '__main__':
    a = generate_anchor()
    print(a)

    ans = get_anchor(a, 16, 37, 50)
    print(ans.shape)

    rpn_locs = np.random.rand(8, 16650, 4)
    transform_locs(ans, rpn_locs, (600, 800))

    box = np.random.rand(2,4)
    at = anchor_target()
    at(box, ans, (600, 800))