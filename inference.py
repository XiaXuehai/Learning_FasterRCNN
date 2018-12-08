# coding: utf-8

import cv2
import numpy as np
import torch

from fr_net import fastnet
from data.data_load import VOC_label_name


def detect(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (800, 600))
    img_ = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img_ = torch.from_numpy(img_/255.)
    bboxs, labels, scores = net.predict(img_)

    for (box,label) in zip(bboxs, labels):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        print(VOC_label_name[label])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('xx', img)
    cv2.waitKey()

if __name__ == '__main__':
    net = fastnet()
    net.load_state_dict(torch.load('weight/fastrcnn_1.weight'))
    net.eval()

    img_path = 'picture/demo.jpg'
    detect(img_path)


    