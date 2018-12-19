# coding: utf-8
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.data_load import VOCData
from fr_net import fastnet
from utils import iou

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def calcu_ap(pre_boxes, pre_labels, pre_scores, gt_boxes, gt_labels, iou_thr=0.5):
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for pre_box, pre_label, pre_score, gt_box, gt_label in zip(pre_boxes, pre_labels, pre_scores, gt_boxes, gt_labels):
        gt_difficult = np.zeros(gt_box.shape[0], dtype=bool) # I do't know what's that mean
        # unique the label
        for x in np.unique(np.concatenate((pre_label, gt_label))):
            pre_box_x = pre_box[pre_label==x]
            pre_score_x = pre_score[pre_label==x]

            order = pre_score_x.argsort()[::-1]
            pre_box_x = pre_box_x[order]
            pre_score_x = pre_score_x[order]

            gt_box_x = gt_box[gt_label==x]
            gt_difficult_x = gt_difficult[gt_label==x]

            n_pos[x] += np.logical_not(gt_difficult_x).sum()
            score[x].extend(pre_score_x) # means pre_score_x to list,then score[x]=pre_score_x

            if len(pre_box_x)==0:
                continue
            if len(gt_box_x)==0:
                match[x].extend((0,)*pre_box_x.shape[0]) # all prediction is false negative
                continue

            pre_box_x[:,2:] += 1
            gt_box_x[:, 2:] += 1
            box_iou = iou(pre_box_x, gt_box_x)
            gt_index = box_iou.argmax(axis=1)           # for predict insight
            gt_index[box_iou.max(axis=1)<iou_thr] = -1  # set the small iou index of gt to -1
            del box_iou

            selec = np.zeros(gt_box_x.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx>=0:
                    if not selec[gt_idx]:
                        match[x].append(1)
                    else:
                        match[x].append(0)
                    selec[gt_idx] = True
                else:
                    match[x].append(0)

    ### got precision as recall
    n_class = max(n_pos.keys()) + 1
    prec = [None] * n_class
    rec = [None] * n_class
    for x in n_pos.keys():
        score_x = np.array(score[x])
        match_x = np.array(match[x], dtype=np.int8)

        order = score_x.argsort()[::-1]
        match_x = match_x[order]

        tp = np.cumsum(match_x==1) # accumulation
        fp = np.cumsum(match_x==0)

        prec[x] = tp / (tp + fp)
        if n_pos[x]>0:
            rec[x] = tp / n_pos[x]

    ### calculate the ap
    ap = np.empty(n_class)
    for x in range(n_class):
        if prec[x] is None or rec[x] is None:
            ap[x] = np.nan
            continue
        # I delete the 07_metric
        else:
            mpre = np.concatenate(([0], np.nan_to_num(prec[x]), [0]))
            mrec = np.concatenate(([0], rec[x], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1] #  make mpre be increasing

            i = np.where(mrec[1:]!=mrec[:-1])[0] # position of recall changes

            ap[x] = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1]) # calculate the area

    return ap


def eval(model_name):
    use_gpu = True

    test_dataset = VOCData('data/test_2007.txt',
                            transform=[transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    net = fastnet()
    net.load_state_dict(torch.load(model_name))
    if use_gpu:
        net = net.cuda()

    # evaluate
    pre_boxes, pre_labels, pre_scores, gt_boxes, gt_labels = [],[],[],[],[]
    for i, (image, gt_labelss, gt_boxess, scale) in tqdm(enumerate(test_loader)):
        if use_gpu:
            image = image.cuda()
        gt_box = gt_boxess.squeeze(0).numpy()
        gt_label = gt_labelss.squeeze(0).numpy()
        pre_box, pre_label, pre_score = net.predict(image, scale.item(), use_gpu)

        gt_boxes.append(gt_box)
        gt_labels.append(gt_label)
        pre_boxes.append(pre_box)
        pre_labels.append(pre_label)
        pre_scores.append(pre_score)

    ap = calcu_ap(pre_boxes, pre_labels, pre_scores, gt_boxes, gt_labels)
    print('ap={}, map={:.3f}'.format(ap, np.mean(ap)))


if __name__ == '__main__':
    model_name = 'weight/fastrcnn.weight'
    eval(model_name)